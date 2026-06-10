"""
PsiGPT v11 — Crystal Archive Memory

Ideia central:
  O campo Psi esquece detalhes porque comprime tudo em 400 floats.
  Mas os cristais marcam o que foi importante.

  Solução: após cada chunk, salvar um snapshot do crystal_map no arquivo.
  Na leitura, o campo atual faz atenção sobre os snapshots passados.
  O arquivo tem tamanho fixo (archive_size slots, circular) — memória O(1).

  Antes (v10):
    leitura = campo ativo apenas

  Agora (v11):
    leitura = campo ativo + atenção sobre arquivo de cristais passados

  O arquivo responde à pergunta:
    "Quais padrões cristalizaram antes e são relevantes agora?"
"""

import gc
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()


# ══════════════════════════════════════════════════════════════════════════════
# PSI FIELD LAYER v11 — com arquivo de cristais
# ══════════════════════════════════════════════════════════════════════════════

class PsiFieldLayer(nn.Module):
    def __init__(self, embed_dim=128, field_size=20, archive_size=8):
        super().__init__()
        self.embed_dim   = embed_dim
        self.field_size  = field_size
        self.archive_size = archive_size
        FS2 = field_size * field_size

        # Projeções do campo ativo (mesmo que v10)
        self.to_key     = nn.Linear(embed_dim, FS2)
        self.to_query   = nn.Linear(embed_dim, FS2)
        self.from_field = nn.Linear(FS2, embed_dim)
        self.proj_out   = nn.Linear(embed_dim, embed_dim)

        # Arquivo de cristais
        # O campo atual consulta o arquivo via atenção dot-product
        self.from_archive = nn.Linear(FS2, embed_dim)   # lê valor do arquivo
        self.archive_mix  = nn.Linear(embed_dim * 2, embed_dim)  # mistura campo + arquivo

        # Física
        self.c2    = nn.Parameter(torch.tensor(0.25))
        self.gamma = nn.Parameter(torch.tensor(0.08))

        # Cristalização
        self.sharpness  = nn.Parameter(torch.tensor(5.0))
        self.amp_thresh = nn.Parameter(torch.tensor(1.3))
        self.cv_thresh  = nn.Parameter(torch.tensor(0.8))
        self.ema_decay  = nn.Parameter(torch.tensor(0.90))
        self.remit      = nn.Parameter(torch.tensor(0.01))
        self.death_rate = nn.Parameter(torch.tensor(0.005))
        self.crystal_lam      = nn.Parameter(torch.tensor(0.03))
        self.crystal_read_gain = nn.Parameter(torch.tensor(1.0))

        self.field_norm = nn.LayerNorm(FS2)

        # Estado do campo ativo
        self._field   = None
        self._vel     = None
        self._rm      = None
        self._rq      = None
        self._crystal = None

        # Arquivo de cristais: (B, archive_size, FS2)
        self._archive   = None
        self._archive_ptr = 0
        self._archive_n   = 0   # quantos slots preenchidos

    def reset(self, B, device=None):
        dev = device or DEVICE
        FS  = self.field_size
        FS2 = FS * FS
        z   = lambda: torch.zeros(B, FS, FS, device=dev)
        self._field, self._vel, self._rm, self._rq, self._crystal = z(), z(), z(), z(), z()
        self._archive     = torch.zeros(B, self.archive_size, FS2, device=dev)
        self._archive_ptr = 0
        self._archive_n   = 0

    def detach_state(self):
        self._field   = self._field.detach()
        self._vel     = self._vel.detach()
        self._rm      = self._rm.detach()
        self._rq      = self._rq.detach()
        self._crystal = self._crystal.detach()
        self._archive = self._archive.detach()

    def commit_to_archive(self):
        """
        Salva snapshot do crystal_map atual no arquivo circular.
        Chamado após cada chunk pelo bloco pai.
        """
        B   = self._field.shape[0]
        FS2 = self.field_size ** 2
        self._archive[:, self._archive_ptr] = self._crystal.view(B, FS2).detach()
        self._archive_ptr = (self._archive_ptr + 1) % self.archive_size
        self._archive_n   = min(self._archive_n + 1, self.archive_size)

    def read_archive(self, field_flat):
        """
        Consulta o arquivo com o campo atual como query.
        field_flat: (B, FS2) float32
        Retorna:    (B, embed_dim) float32
        """
        B   = field_flat.shape[0]
        FS2 = self.field_size ** 2

        if self._archive_n == 0:
            return torch.zeros(B, self.embed_dim, device=field_flat.device, dtype=field_flat.dtype)

        # Atenção dot-product: campo atual × snapshots de cristais passados
        k      = self._archive.float()                          # (B, archive_size, FS2)
        scores = (field_flat.unsqueeze(1) * k).sum(dim=-1)     # (B, archive_size)
        scores = scores / (FS2 ** 0.5)

        # Mascara slots não preenchidos via masked_fill (autograd-safe)
        if self._archive_n < self.archive_size:
            mask   = torch.arange(self.archive_size, device=scores.device) >= self._archive_n
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        weights      = F.softmax(scores, dim=-1)                        # (B, archive_size)
        archive_read = (weights.unsqueeze(-1) * k).sum(dim=1)           # (B, FS2)

        return self.from_archive(archive_read)

    def process_chunk(self, x_chunk):
        """
        x_chunk: (B, C, D)
        Retorna: (B, C, D)

        Desabilita AMP completamente — toda a física e os linears do campo
        rodam em float32. Evita NaN causado por overflow em float16.
        O resultado é convertido de volta ao dtype original no final.
        """
        orig_dtype = x_chunk.dtype
        dev = x_chunk.device

        with torch.autocast(device_type=dev.type, enabled=False):
            x_chunk = x_chunk.float()

            self._field   = self._field.float()
            self._vel     = self._vel.float()
            self._rm      = self._rm.float()
            self._rq      = self._rq.float()
            self._crystal = self._crystal.float()

            B, C, D = x_chunk.shape
            FS  = self.field_size
            FS2 = FS * FS
            dt  = 0.1

            c2    = self.c2.float().clamp(0.01, 1.0)
            gamma = self.gamma.float().clamp(0.01, 0.5)
            decay = self.ema_decay.float().clamp(0.8, 0.99)
            remit = self.remit.float().clamp(0.0, 0.2)
            death = self.death_rate.float().clamp(0.0, 0.01)
            sharp = self.sharpness.float().clamp(1.0, 20.0)
            cv_t  = self.cv_thresh.float().clamp(0.1, 2.0)
            lam   = self.crystal_lam.float().clamp(0.0, 0.2)
            rg    = self.crystal_read_gain.float().clamp(0.0, 5.0)

            # Projeções em batch
            x_flat  = x_chunk.reshape(B * C, D)
            keys    = self.to_key(x_flat).view(B, C, FS, FS)
            queries = self.to_query(x_flat).view(B, C, FS2)

            crystal_snapshot = self._crystal.view(B, FS2).detach()

            # Leitura do arquivo ANTES do loop (uma vez por chunk)
            archive_out = self.read_archive(self._field.view(B, FS2))  # (B, embed_dim)
            archive_out = archive_out.unsqueeze(1).expand(B, C, -1)    # (B, C, embed_dim)

            interf_list = []

            for t in range(C):
                # READ crystal-gated
                crystal_gated = self._field.view(B, FS2) * (1.0 + rg * crystal_snapshot)
                interf_list.append(queries[:, t] * crystal_gated)

                # WRITE
                self._field = self._field + keys[:, t] * 0.1

                # FÍSICA
                f   = self._field
                lap = (torch.roll(f,  1, -2) + torch.roll(f, -1, -2) +
                       torch.roll(f,  1, -1) + torch.roll(f, -1, -1) - 4.0 * f)
                acc = c2 * lap - gamma * self._vel

                # CRISTALIZAÇÃO
                fa       = self._field.abs()
                self._rm = decay * self._rm + (1 - decay) * fa
                self._rq = decay * self._rq + (1 - decay) * fa ** 2

                var = (self._rq - self._rm ** 2).clamp(min=1e-8)
                cv  = var.sqrt() / (self._rm + 1e-8)

                amp_score     = torch.sigmoid(sharp * (self._rm - self.amp_thresh))
                cv_score      = torch.sigmoid(sharp * (cv_t - cv))
                self._crystal = self._crystal + 0.1 * (amp_score * cv_score - self._crystal)
                self._crystal = (self._crystal - death).clamp(min=0.0)

                # ACOPLAMENTO CRISTALINO
                f_flat  = self._field.view(B, FS2)
                cv_flat = self._crystal.view(B, FS2) * self._vel.view(B, FS2)
                S_cvf   = (cv_flat * f_flat).sum(dim=1, keepdim=True)
                S_cv    =  cv_flat.sum(dim=1, keepdim=True)
                acc     = acc + (lam * cv_flat * (S_cvf - f_flat * S_cv)).view(B, FS, FS)

                self._vel   = self._vel + acc * dt
                self._field = self._field + self._vel * dt

                scale       = self._field.abs().mean(dim=(-1, -2), keepdim=True).clamp(min=0.1)
                self._field = self._field / scale
                # Bound hard contra picos instáveis (CFL ok mas explicit Euler
                # com writes agressivos pode criar picos. tanh suaviza sem cortar.)
                self._field = torch.tanh(self._field * 0.3) / 0.3
                self._vel   = torch.tanh(self._vel   * 0.1) / 0.1
                self._field = self._field + self._crystal * remit

            # Linears fora do loop — ainda dentro do with (float32)
            interferences = torch.stack(interf_list, dim=1)   # (B, C, FS2)
            BC        = B * C
            interf_n  = self.field_norm(interferences.reshape(BC, FS2))
            field_out = self.from_field(interf_n)

            # Combina campo ativo + arquivo de cristais
            archive_flat = archive_out.reshape(BC, self.embed_dim)
            combined     = self.archive_mix(torch.cat([field_out, archive_flat], dim=-1))
            outputs      = self.proj_out(combined).reshape(B, C, D)

        # Converte de volta ao dtype do modelo (float16 se AMP ativo)
        return outputs.to(orig_dtype)

    def get_state_info(self):
        if self._field is None:
            return {}
        return {
            'field_abs_mean':   self._field.abs().mean().item(),
            'crystal_coverage': (self._crystal > 0.5).float().mean().item(),
            'crystal_max':      self._crystal.max().item(),
            'archive_n':        self._archive_n,
        }


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO E MODELO
# ══════════════════════════════════════════════════════════════════════════════

class PsiGPTBlock(nn.Module):
    def __init__(self, embed_dim=128, field_size=20, archive_size=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.psi   = PsiFieldLayer(embed_dim, field_size, archive_size)
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

    def commit_to_archive(self):
        self.psi.commit_to_archive()

    def forward_chunk(self, x_chunk):
        psi_out = self.psi.process_chunk(self.norm1(x_chunk))
        x_chunk = x_chunk + psi_out
        B, C, D = x_chunk.shape
        mlp_out = self.mlp(self.norm2(x_chunk).reshape(B * C, D)).reshape(B, C, D)
        return x_chunk + mlp_out


class PsiGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=128, embed_dim=128, depth=4,
                 field_size=20, chunk_size=32, archive_size=8):
        super().__init__()
        self.ctx_len    = ctx_len
        self.chunk_size = chunk_size
        self.depth      = depth
        self.field_size = field_size

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            PsiGPTBlock(embed_dim, field_size, archive_size)
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
        B, N = idx.shape
        x = self.tok_embed(idx) + self.pos_embed[:, :N]

        for blk in self.blocks:
            blk.reset(B, idx.device)

        outputs = []
        for start in range(0, N, self.chunk_size):
            end     = min(start + self.chunk_size, N)
            x_chunk = x[:, start:end]
            h       = x_chunk
            for blk in self.blocks:
                h = blk.forward_chunk(h)
            outputs.append(h)
            for blk in self.blocks:
                blk.commit_to_archive()   # snapshot do cristal após cada chunk
                blk.detach_state()

        return self.head(self.norm(torch.cat(outputs, dim=1)))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=40):
        B = idx.shape[0]
        for blk in self.blocks:
            blk.reset(B, idx.device)

        for t in range(idx.shape[1]):
            x_t = self.tok_embed(idx[:, t:t+1]) + self.pos_embed[:, t:t+1]
            h   = x_t
            for blk in self.blocks:
                h = blk.forward_chunk(h)
            for blk in self.blocks:
                blk.commit_to_archive()

        last_h = h[:, -1]

        for _ in range(max_new_tokens):
            logits = self.head(self.norm(last_h)) / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx      = torch.cat([idx, idx_next], dim=1)

            t_pos  = min(idx.shape[1] - 1, self.ctx_len - 1)
            x_next = self.tok_embed(idx_next) + self.pos_embed[:, t_pos:t_pos+1]
            h      = x_next
            for blk in self.blocks:
                h = blk.forward_chunk(h)
            for blk in self.blocks:
                blk.commit_to_archive()
            last_h = h[:, 0]

        return idx

    def state_memory_bytes(self, batch_size=1):
        FS2          = self.field_size ** 2
        field_bytes  = self.depth * 5 * batch_size * FS2 * 4
        archive_size = self.blocks[0].psi.archive_size
        archive_bytes = self.depth * archive_size * batch_size * FS2 * 4
        return field_bytes + archive_bytes

    def crystal_report(self):
        lines = []
        for i, blk in enumerate(self.blocks):
            info = blk.psi.get_state_info()
            if info:
                lines.append(
                    f"  camada {i}: field_amp={info['field_abs_mean']:.4f}  "
                    f"crystal_cov={info['crystal_coverage']:.1%}  "
                    f"crystal_max={info['crystal_max']:.3f}  "
                    f"arquivo={info['archive_n']}/{blk.psi.archive_size} slots"
                )
        return '\n'.join(lines) if lines else "  (sem estado)"


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER REFERÊNCIA
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
        qkv     = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out     = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj_out(out.transpose(1, 2).reshape(B, N, D))


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
        return x + self.mlp(self.norm2(x + self.attn(self.norm1(x))))


class TransformerGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=128, embed_dim=128, depth=4, n_heads=4):
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
            if m.bias is not None:
                nn.init.zeros_(m.bias)
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
    def __init__(self, text, ctx_len=128):
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

def treinar(model, dataset, nome, n_steps=500, batch_size=16, lr=3e-4, print_every=100, use_amp=None):
    # PsiGPT: AMP off — campo já roda em float32, AMP só adiciona instabilidade
    # Transformer: AMP on — tudo é matmul, float16 é estável e rápido
    if use_amp is None:
        use_amp = USE_AMP
    model    = model.to(DEVICE)
    n_params = model.count_params()
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*65}")
    print(f"{nome}: {n_params:,} params")
    print(f"{'='*65}")
    print(f"AMP: {'on' if use_amp else 'off'} | lr: cosine {lr:.0e} → {lr/10:.0e}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    losses, t0 = [], time.time()

    for step in range(n_steps):
        model.train()
        x, y = dataset.get_batch(batch_size, DEVICE)

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if not torch.isfinite(loss):
            print(f"  passo {step+1}: loss nao finita, pulando")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()

        losses.append(loss.item())

        if (step + 1) % print_every == 0:
            avg = sum(losses[-print_every:]) / print_every
            mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  step {step+1:>5d}/{n_steps}  loss={avg:.4f}  mem={mem:.0f}MB  tempo={time.time()-t0:.0f}s")

        if hasattr(model, 'crystal_report') and (step + 1) % 500 == 0:
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
        'loss':   sum(losses[-200:]) / min(200, len(losses)),
        'mem_MB': peak_mem,
        'params': n_params,
        'time':   time.time() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    print(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    CTX_LEN      = 128   # maior que v10 (64) — arquivo tem mais chunks para acumular
    BATCH        = 16
    STEPS        = 500
    D            = 128
    DEPTH        = 4
    FS           = 20
    CHUNK        = 32
    ARCHIVE_SIZE = 8     # últimos 8 chunks = 256 tokens de memória cristalina

    text    = download_shakespeare()
    dataset = CharDataset(text, ctx_len=CTX_LEN)
    print(f"\nTexto: {len(text):,} chars | Vocab: {dataset.vocab_size} | Ctx: {CTX_LEN}")
    print(f"Campo Ψ: {FS}×{FS} | Chunk: {CHUNK} | Archive: {ARCHIVE_SIZE} slots | Batch: {BATCH}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── PsiGPT v11 ──
    psi_model = PsiGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN,
        embed_dim=D,
        depth=DEPTH,
        field_size=FS,
        chunk_size=CHUNK,
        archive_size=ARCHIVE_SIZE,
    )
    state_bytes = psi_model.state_memory_bytes(batch_size=1)
    print(f"Estado Ψ (B=1): {state_bytes/1024:.1f} KB  |  arquivo cobre {ARCHIVE_SIZE*CHUNK} tokens")

    res_psi = treinar(psi_model, dataset,
                      f"PsiGPT v11 (arquivo {ARCHIVE_SIZE} slots, ctx={CTX_LEN})",
                      STEPS, BATCH, lr=3e-4, print_every=1, use_amp=False)  # campo em float32, AMP não ajuda

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Transformer referência ──
    tf_model = TransformerGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN,
        embed_dim=D,
        depth=DEPTH,
        n_heads=4,
    )
    res_tf = treinar(tf_model, dataset,
                     f"Transformer (ctx={CTX_LEN})",
                     STEPS, BATCH, lr=3e-4, use_amp=True)   # transformer é todo matmul, AMP é seguro

    # ── Resultado ──
    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL")
    print(f"{'='*70}")
    print(f"  {'':35s}  {'Transformer':>12s}  {'PsiGPT v11':>12s}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*12}")
    print(f"  {'Parâmetros':35s}  {res_tf['params']:>12,}  {res_psi['params']:>12,}")
    print(f"  {'Loss final':35s}  {res_tf['loss']:>12.4f}  {res_psi['loss']:>12.4f}")
    print(f"  {'Mem peak (MB)':35s}  {res_tf['mem_MB']:>12.0f}  {res_psi['mem_MB']:>12.0f}")
    print(f"  {'Tempo (s)':35s}  {res_tf['time']:>12.0f}  {res_psi['time']:>12.0f}")
    state_kb = psi_model.state_memory_bytes(batch_size=1) // 1024
    print(f"  {'Estado recorrente':35s}  {'N/A (KV$)':>12s}  {state_kb:>10d} KB")
    print(f"  {'Contexto coberto pelo arquivo':35s}  {'ctx total':>12s}  {ARCHIVE_SIZE*CHUNK:>9d} tok")
    print(f"{'='*70}")
