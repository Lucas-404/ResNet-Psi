"""
PsiGPT v8 — Reads Paralelas + Cristalização com CV + Morte

Mudanças em relação ao v7:
  1. READ paralelo: todos os tokens do chunk leem o campo de uma vez (matmul)
  2. WRITE+física sequencial mas leve (sem projeção linear no loop)
  3. Crystal: adiciona CV score (estabilidade temporal) + taxa de morte
  4. Campo maior: F=24 (mais representação)

Estrutura por chunk:
  a) Precomputa TODAS as keys e queries do chunk: (B, C, D) → (B*C, D) → linear once
  b) READ paralelo: todos os C tokens leem o campo atual → (B, C, D) em um matmul
  c) Loop leve (só física + crystal, sem linear):
       para cada token t:
         field += key[t] * 0.1
         propaga onda (conv2d ou FFT)
         atualiza crystal (com CV)

Por que funciona:
  - No v7: 256 chamadas Python (N×L), cada uma com 2 linear layers pesadas
  - No v8: 2 linear layers chamadas UMA VEZ por chunk, paralelas
  - Loop restante: só ops element-wise e 1 conv2d → muito mais leve

Trade-off de causality:
  - Dentro do chunk: todos os tokens leem o campo ANTES do chunk começar
  - Entre chunks: campo é atualizado corretamente
  - Equivalente a "chunked attention" — padrão estabelecido (Mamba2, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import gc
import urllib.request

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# PSI FIELD LAYER v8
# ══════════════════════════════════════════════════════════════════════════════

class PsiFieldLayer(nn.Module):
    def __init__(self, embed_dim=128, field_size=24):
        super().__init__()
        self.embed_dim  = embed_dim
        self.field_size = field_size
        FS2 = field_size * field_size

        # Projeções (chamadas UMA VEZ por chunk, não por token)
        self.to_key     = nn.Linear(embed_dim, FS2)
        self.to_query   = nn.Linear(embed_dim, FS2)
        self.from_field = nn.Linear(FS2, embed_dim)
        self.proj_out   = nn.Linear(embed_dim, embed_dim)

        # Física
        self.c2    = nn.Parameter(torch.tensor(0.25))
        self.gamma = nn.Parameter(torch.tensor(0.08))

        # Cristalização com CV
        self.sharpness  = nn.Parameter(torch.tensor(5.0))
        self.amp_thresh = nn.Parameter(torch.tensor(0.05))  # baixo: campo fica em ~0.10
        self.cv_thresh  = nn.Parameter(torch.tensor(0.8))   # CV máximo para cristalizar
        self.ema_decay  = nn.Parameter(torch.tensor(0.90))  # adaptação mais rápida
        self.remit      = nn.Parameter(torch.tensor(0.05))
        self.death_rate = nn.Parameter(torch.tensor(0.001)) # morte suave

        lap = torch.tensor([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap)
        self.field_norm = nn.LayerNorm(FS2)

        self._field   = None
        self._vel     = None
        self._rm      = None   # running mean (para EMA)
        self._rq      = None   # running mean do quadrado (para CV)
        self._crystal = None

    def reset(self, B, device=None):
        dev = device or DEVICE
        FS  = self.field_size
        z   = lambda: torch.zeros(B, FS, FS, device=dev)
        self._field, self._vel, self._rm, self._rq, self._crystal = z(), z(), z(), z(), z()

    def detach_state(self):
        self._field   = self._field.detach()
        self._vel     = self._vel.detach()
        self._rm      = self._rm.detach()
        self._rq      = self._rq.detach()
        self._crystal = self._crystal.detach()

    def process_chunk(self, x_chunk):
        """
        x_chunk: (B, C, D)
        Retorna: (B, C, D)

        PARALLEL read + SEQUENTIAL write/physics.
        """
        B, C, D = x_chunk.shape
        FS  = self.field_size
        FS2 = FS * FS
        dt  = 0.1

        c2    = self.c2.clamp(0.01, 1.0)
        gamma = self.gamma.clamp(0.01, 0.5)
        decay = self.ema_decay.clamp(0.8, 0.99)
        remit = self.remit.clamp(0.0, 0.2)
        death = self.death_rate.clamp(0.0, 0.01)
        sharp = self.sharpness.clamp(1.0, 20.0)
        cv_t  = self.cv_thresh.clamp(0.1, 2.0)

        # ── Passo 1: precomputa TODAS as projeções de uma vez ──
        x_flat = x_chunk.reshape(B * C, D)
        keys   = self.to_key(x_flat).view(B, C, FS, FS)   # (B, C, F, F)
        queries = self.to_query(x_flat).view(B, C, FS2)   # (B, C, F²)

        # ── Passo 2: READ paralelo — todos os tokens leem o campo ATUAL ──
        # Todos os C tokens veem o campo no estado inicial do chunk
        field_flat   = self._field.view(B, 1, FS2)                    # (B, 1, F²)
        interference = queries * field_flat                             # (B, C, F²) broadcast
        out_flat     = self.proj_out(
                           self.from_field(
                               self.field_norm(interference.reshape(B * C, FS2))
                           )
                       )                                               # (B*C, D)
        outputs      = out_flat.view(B, C, D)                         # (B, C, D)

        # ── Passo 3: loop leve — write + física + crystal ──
        # (sem linear layers aqui — keys já foram computadas acima)
        for t in range(C):
            # WRITE: token t perturba o campo
            self._field = self._field + keys[:, t] * 0.1

            # FÍSICA: propagação de onda
            f_pad = F.pad(self._field.unsqueeze(1), (1, 1, 1, 1), mode='circular')
            lap   = F.conv2d(f_pad, self.lap_kernel).squeeze(1)
            acc   = c2 * lap - gamma * self._vel
            self._vel   = self._vel + acc * dt
            self._field = self._field + self._vel * dt

            # CRYSTAL: EMA de amplitude e quadrado (para CV)
            fa       = self._field.abs()
            self._rm = decay * self._rm + (1 - decay) * fa
            self._rq = decay * self._rq + (1 - decay) * fa ** 2

            # CV = std / mean — mede estabilidade temporal
            var = (self._rq - self._rm ** 2).clamp(min=1e-8)
            cv  = var.sqrt() / (self._rm + 1e-8)

            # Score: alta amplitude E baixo CV (região estável)
            amp_score = torch.sigmoid(sharp * (self._rm - self.amp_thresh))
            cv_score  = torch.sigmoid(sharp * (cv_t - cv))
            score     = amp_score * cv_score

            # Dinâmica: cresce em direção ao score, decai constantemente
            self._crystal = self._crystal + 0.1 * (score - self._crystal)
            self._crystal = (self._crystal - death).clamp(min=0.0)

            # Re-emissão: cristais injetam energia de volta
            self._field = self._field + self._crystal * remit

        return outputs

    def get_state_info(self):
        if self._field is None:
            return {}
        return {
            'field_abs_mean':   self._field.abs().mean().item(),
            'crystal_coverage': (self._crystal > 0.5).float().mean().item(),
            'crystal_max':      self._crystal.max().item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO E MODELO
# ══════════════════════════════════════════════════════════════════════════════

class PsiGPTBlock(nn.Module):
    def __init__(self, embed_dim=128, field_size=24, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.psi   = PsiFieldLayer(embed_dim, field_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(0.1),
        )

    def reset(self, B, device=None):
        self.psi.reset(B, device)

    def detach_state(self):
        self.psi.detach_state()

    def forward_chunk(self, x_chunk):
        """x_chunk: (B, C, D) → (B, C, D)"""
        # Psi (READ paralelo + WRITE sequencial)
        psi_out = self.psi.process_chunk(self.norm1(x_chunk))   # (B, C, D)
        x_chunk = x_chunk + psi_out

        # MLP: batched (B*C, D) — uma só chamada
        B, C, D = x_chunk.shape
        mlp_out = self.mlp(self.norm2(x_chunk).reshape(B * C, D)).reshape(B, C, D)
        return x_chunk + mlp_out


class PsiGPT(nn.Module):
    """
    PsiGPT v8:
      - Reads paralelas por chunk (chunk_size × speedup na projeção)
      - Crystal com CV + morte (resolve saturação do v7)
      - Campo 24×24 (mais representação que 16×16)
    """

    def __init__(self, vocab_size, ctx_len=512, embed_dim=128, depth=4,
                 field_size=24, chunk_size=16):
        super().__init__()
        self.ctx_len    = ctx_len
        self.chunk_size = chunk_size
        self.depth      = depth
        self.field_size = field_size

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            PsiGPTBlock(embed_dim, field_size)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        """idx: (B, N) → logits (B, N, vocab)"""
        B, N = idx.shape
        x = self.tok_embed(idx) + self.pos_embed[:, :N]  # (B, N, D)

        for blk in self.blocks:
            blk.reset(B, idx.device)

        outputs = []
        for start in range(0, N, self.chunk_size):
            end     = min(start + self.chunk_size, N)
            x_chunk = x[:, start:end]                  # (B, C, D)

            h = x_chunk
            for blk in self.blocks:
                h = blk.forward_chunk(h)
            outputs.append(h)

            # Truncated BPTT entre chunks
            for blk in self.blocks:
                blk.detach_state()

        x_out  = torch.cat(outputs, dim=1)            # (B, N, D)
        return self.head(self.norm(x_out))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=40):
        B = idx.shape[0]
        for blk in self.blocks:
            blk.reset(B, idx.device)

        # Alimenta o prompt token a token (field acumula contexto)
        for t in range(idx.shape[1]):
            x_t    = self.tok_embed(idx[:, t:t+1]) + self.pos_embed[:, t:t+1]  # (B, 1, D)
            h      = x_t
            for blk in self.blocks:
                h = blk.forward_chunk(h)
        last_h = h[:, -1]  # (B, D)

        # Gera token a token
        for step in range(max_new_tokens):
            logits = self.head(self.norm(last_h)) / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)  # (B, 1)
            idx      = torch.cat([idx, idx_next], dim=1)

            t_pos = min(idx.shape[1] - 1, self.ctx_len - 1)
            x_next = self.tok_embed(idx_next) + self.pos_embed[:, t_pos:t_pos+1]  # (B, 1, D)
            h = x_next
            for blk in self.blocks:
                h = blk.forward_chunk(h)
            last_h = h[:, 0]  # (B, D)

        return idx

    def state_memory_bytes(self, batch_size=1):
        # 5 tensors (field, vel, rm, rq, crystal) × layers × (B, F, F) × 4 bytes
        FS2 = self.field_size ** 2
        return self.depth * 5 * batch_size * FS2 * 4

    def crystal_report(self):
        lines = []
        for i, blk in enumerate(self.blocks):
            info = blk.psi.get_state_info()
            if info:
                lines.append(
                    f"  camada {i}: field_amp={info['field_abs_mean']:.4f}  "
                    f"crystal_cov={info['crystal_coverage']:.1%}  "
                    f"crystal_max={info['crystal_max']:.3f}"
                )
        return '\n'.join(lines) if lines else "  (sem estado)"


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER GPT (referência — igual ao v7)
# ══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv      = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv  = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = F.softmax(
            (q @ k.transpose(-2, -1) / self.head_dim**0.5).masked_fill(mask, float('-inf')),
            dim=-1
        )
        return self.proj_out((attn @ v).transpose(1, 2).reshape(B, N, D))


class TransformerGPTBlock(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = CausalSelfAttention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio), nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=512, embed_dim=128, depth=4, n_heads=4):
        super().__init__()
        self.ctx_len   = ctx_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)
        self.blocks    = nn.Sequential(*[TransformerGPTBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, N = idx.shape
        return self.head(self.norm(self.blocks(self.tok_embed(idx) + self.pos_embed[:, :N])))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx_len:]
            logits   = self(idx_cond)[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def download_shakespeare():
    path = './data/shakespeare.txt'
    os.makedirs('./data', exist_ok=True)
    if not os.path.exists(path):
        print("Baixando Shakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class CharDataset:
    def __init__(self, text, ctx_len=512):
        self.ctx_len    = ctx_len
        chars           = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi       = {c: i for i, c in enumerate(chars)}
        self.itos       = {i: c for c, i in self.stoi.items()}
        self.data       = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def encode(self, s):  return [self.stoi.get(c, 0) for c in s]
    def decode(self, t):  return ''.join(self.itos.get(i, '?') for i in t)

    def get_batch(self, batch_size, device):
        ix = torch.randint(0, len(self.data) - self.ctx_len - 1, (batch_size,))
        x  = torch.stack([self.data[i:i+self.ctx_len]     for i in ix]).to(device)
        y  = torch.stack([self.data[i+1:i+self.ctx_len+1] for i in ix]).to(device)
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# TREINO
# ══════════════════════════════════════════════════════════════════════════════

def treinar(model, dataset, nome, n_steps=3000, batch_size=32, lr=3e-4,
            print_every=100):
    model    = model.to(DEVICE)
    n_params = model.count_params()
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    print(f"\n{'='*65}")
    print(f"{nome}: {n_params:,} params")
    print(f"{'='*65}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    losses, t0 = [], time.time()

    for step in range(n_steps):
        model.train()
        x, y   = dataset.get_batch(batch_size, DEVICE)
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        losses.append(loss.item())

        if (step + 1) % print_every == 0:
            avg = sum(losses[-print_every:]) / print_every
            mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  step {step+1:>5d}/{n_steps}  loss={avg:.4f}  "
                  f"mem={mem:.0f}MB  tempo={time.time()-t0:.0f}s")

        if hasattr(model, 'crystal_report') and (step + 1) % 1000 == 0:
            print("  cristais:")
            print(model.crystal_report())

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    model.eval()
    print(f"\n  --- Gerações ({nome}) ---")
    for prompt in ["ROMEO:\n", "To be, or", "The king is"]:
        ids = torch.tensor([dataset.encode(prompt)], device=DEVICE)
        out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=40)
        print(f"  > {dataset.decode(out[0].tolist())[:280]}\n")

    if hasattr(model, 'crystal_report'):
        print("  Estado final dos cristais:")
        print(model.crystal_report())

    return {
        'loss':   sum(losses[-200:]) / 200,
        'mem_MB': peak_mem,
        'params': n_params,
        'time':   time.time() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    text    = download_shakespeare()
    CTX_LEN = 64
    BATCH   = 32
    STEPS   = 3000
    D       = 128
    DEPTH   = 4
    FS      = 24     # campo maior: 24×24 vs 16×16 do v7
    CHUNK   = 4      # chunk menor: melhor causality, reads ainda paralelos

    dataset = CharDataset(text, ctx_len=CTX_LEN)
    print(f"\nTexto: {len(text):,} chars | Vocab: {dataset.vocab_size} | Ctx: {CTX_LEN}")
    print(f"Campo Ψ: {FS}×{FS} | Chunk: {CHUNK} | Batch: {BATCH}")

    # Referência v7 (resultados conhecidos)
    ref_v7 = {'loss': 1.9736, 'mem_MB': 117, 'params': 1_017_880, 'time': 3352}
    ref_tf = {'loss': 1.3844, 'mem_MB': 1362, 'params': 875_520,  'time': 687}

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    model = PsiGPT(dataset.vocab_size, CTX_LEN, D, DEPTH,
                   field_size=FS, chunk_size=CHUNK)

    state_bytes = model.state_memory_bytes(batch_size=BATCH)
    print(f"Estado Ψ (B={BATCH}): {state_bytes/1024:.1f} KB")
    print(f"Speedup esperado: ~{CHUNK}x nas projeções (reads paralelos)\n")

    res = treinar(model, dataset,
                  f"PsiGPT v8 (campo {FS}×{FS}, chunk={CHUNK}, CV+morte)",
                  STEPS, BATCH)

    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL")
    print(f"{'='*70}")
    print(f"  {'':35s}  {'Transformer':>10s}  {'PsiGPT v7':>10s}  {'PsiGPT v8':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Parâmetros':35s}  {ref_tf['params']:>10,}  {ref_v7['params']:>10,}  {res['params']:>10,}")
    print(f"  {'Loss final':35s}  {ref_tf['loss']:>10.4f}  {ref_v7['loss']:>10.4f}  {res['loss']:>10.4f}")
    print(f"  {'Mem peak (MB)':35s}  {ref_tf['mem_MB']:>10.0f}  {ref_v7['mem_MB']:>10.0f}  {res['mem_MB']:>10.0f}")
    print(f"  {'Tempo (s)':35s}  {ref_tf['time']:>10.0f}  {ref_v7['time']:>10.0f}  {res['time']:>10.0f}")
    if ref_v7['time'] > 0:
        print(f"  {'Speedup vs v7':35s}  {'':>10s}  {'1.0x':>10s}  {ref_v7['time']/res['time']:>9.1f}x")
    print(f"{'='*70}")
