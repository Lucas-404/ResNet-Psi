"""
Wave GPT v2 — Attention Completo por Campo de Ondas

O attention do Transformer tem 3 peças:
  1. Q×K^T  → quem é relevante pra quem (scores)
  2. Softmax → competição (winner-takes-most)
  3. × V    → pega informação dos relevantes

Tradução pro campo de ondas:
  1. Interferência → ondas Q e K propagam no campo, interferem
  2. Cristalização suave → sigmoid + HP contínuo = softmax física
  3. Re-emissão → cristais sobreviventes emitem V de volta

Tudo diferenciável. Sem thresholds duros. Gradiente passa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import os
import urllib.request

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# CRYSTAL ATTENTION — Attention completo por campo de ondas
# ══════════════════════════════════════════════════════════════════════════════

class CrystalAttention(nn.Module):
    """
    Attention por interferência + cristalização + re-emissão.

    1. Cada token emite onda Q (query) e onda K (key) no campo.
       Q e K interferem → campo codifica compatibilidade.

    2. Cristalização suave: envelope do campo passa por sigmoid.
       HP contínuo: regiões que ressoam ganham vida, outras decaem.
       É o softmax — mas por física em vez de exp/sum.

    3. Re-emissão: tokens emitiram V (value) como onda.
       O campo cristalizado modula V — só passa informação das
       regiões que sobreviveram. Tokens irrelevantes foram suprimidos.

    Memória: campo F×F fixo. Não importa quantos tokens.
    """
    def __init__(self, embed_dim=128, field_size=32, n_steps=6, n_crystal_steps=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_size = field_size
        self.n_steps = n_steps
        self.n_crystal_steps = n_crystal_steps

        # Projeções Q, K, V → padrões de onda no campo
        self.to_q = nn.Linear(embed_dim, field_size * field_size)
        self.to_k = nn.Linear(embed_dim, field_size * field_size)
        self.to_v = nn.Linear(embed_dim, field_size * field_size)

        # Parâmetros físicos treináveis
        self.c2 = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.06))
        self.alpha = nn.Parameter(torch.tensor(0.04))

        # Cristalização suave — parâmetros treináveis
        self.crystal_sharpness = nn.Parameter(torch.tensor(2.0))   # mais suave, menos winner-takes-all
        self.crystal_threshold = nn.Parameter(torch.tensor(0.3))   # amplitude mínima
        self.hp_gain = nn.Parameter(torch.tensor(0.1))             # ressonância → HP
        self.hp_decay = nn.Parameter(torch.tensor(0.08))           # decaimento mais forte, mais rotatividade
        self.remit_strength = nn.Parameter(torch.tensor(0.05))     # re-emissão

        # Laplaciano
        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

        # Leitura: campo F×F → embed_dim
        self.from_field = nn.Linear(field_size * field_size, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def propagate(self, field, velocity, source, active):
        """Um step de onda — diferenciável."""
        dt = 0.1
        if active:
            field = field + source * dt

        B, H, W = field.shape
        f_pad = F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular')
        lap = F.conv2d(f_pad, self.lap_kernel).squeeze(1)

        c2 = torch.clamp(self.c2, 0.01, 1.0)
        gamma = torch.clamp(self.gamma, 0.01, 0.5)
        alpha = torch.clamp(self.alpha, 0.0, 0.2)

        acc = c2 * lap - gamma * velocity + alpha * torch.tanh(field) * field
        velocity = velocity + acc * dt
        field = field + velocity * dt
        return field, velocity

    def crystallize(self, field, hp):
        """
        Cristalização suave — diferenciável.

        Em vez de thresholds duros:
          score = sigmoid(sharpness × (|campo| - threshold))
          hp += score × gain
          hp -= decay
          alive = sigmoid(hp × 10)  (suave, não step function)
        """
        sharpness = torch.clamp(self.crystal_sharpness, 1.0, 20.0)
        threshold = torch.clamp(self.crystal_threshold, 0.1, 1.0)
        gain = torch.clamp(self.hp_gain, 0.01, 0.5)
        decay = torch.clamp(self.hp_decay, 0.005, 0.1)

        # Score de amplitude: regiões com oscilação forte
        amp_score = torch.sigmoid(sharpness * (field.abs() - threshold))

        # HP sobe onde ressoa, desce em todo lugar
        hp = hp + amp_score * gain - decay

        # Alive: sigmoid suave (não binário)
        alive = torch.sigmoid(hp * 10.0)

        return hp, alive

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape
        FS = self.field_size

        # ── 1. EMISSÃO: tokens emitem Q, K, V como ondas ──
        q_waves = self.to_q(x).view(B, N, FS, FS)
        k_waves = self.to_k(x).view(B, N, FS, FS)
        v_waves = self.to_v(x).view(B, N, FS, FS)

        # Causalidade: soma cumulativa (token i só vê tokens 0..i)
        q_cumsum = torch.cumsum(q_waves, dim=1)
        k_cumsum = torch.cumsum(k_waves, dim=1)
        v_cumsum = torch.cumsum(v_waves, dim=1)

        # Source de interferência: Q + K somados (interferência)
        # Usar a soma total pra propagar o campo
        q_total = q_waves.sum(dim=1)  # (B, FS, FS)
        k_total = k_waves.sum(dim=1)  # (B, FS, FS)
        source = q_total + k_total     # interferência Q×K no campo

        # ── 2. PROPAGAÇÃO: ondas se espalham e interferem ──
        field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
        velocity = torch.zeros_like(field)

        for s in range(self.n_steps):
            active = s < self.n_steps // 2
            field, velocity = self.propagate(field, velocity, source, active)

        # ── 3. CRISTALIZAÇÃO SUAVE: softmax por física ──
        # HP começa em zero — precisa ressoar pra sobreviver
        hp = torch.zeros_like(field)

        for c in range(self.n_crystal_steps):
            # Propagar mais um pouco (campo vivo)
            field, velocity = self.propagate(field, velocity, source * 0, False)
            # Cristalizar
            hp, alive = self.crystallize(field, hp)

        # alive: (B, FS, FS) — mapa de [0,1] onde cristais sobreviveram

        # ── 4. RE-EMISSÃO: cristais modulam V ──
        # O campo cristalizado atua como filtro sobre V
        # Regiões vivas passam V, regiões mortas bloqueiam
        v_total = v_waves.sum(dim=1)  # (B, FS, FS)

        # Propagar V pelo campo filtrado pelos cristais
        v_filtered = v_total * alive  # só V das regiões cristalizadas passa

        # Re-emitir no campo
        remit = torch.clamp(self.remit_strength, 0.01, 0.2)
        field_out = field * alive + v_filtered * remit

        # ── 5. LEITURA: cada token lê o campo ──
        # Leitura causal: token i lê modulado pela soma cumulativa Q[0..i]
        scale = torch.arange(1, N + 1, device=x.device, dtype=x.dtype).view(1, N, 1, 1)
        causal_q = q_cumsum / scale

        field_expanded = field_out.unsqueeze(1).expand(B, N, FS, FS)
        token_reads = causal_q * field_expanded

        token_reads = token_reads.view(B, N, FS * FS)
        out = self.from_field(token_reads)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-ATTENTION CAUSAL (referência)
# ══════════════════════════════════════════════════════════════════════════════

class SelfAttentionCausal(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCOS
# ══════════════════════════════════════════════════════════════════════════════

class CrystalBlock(nn.Module):
    def __init__(self, embed_dim=128, field_size=32, n_steps=6,
                 n_crystal_steps=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CrystalAttention(embed_dim, field_size, n_steps, n_crystal_steps)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttentionCausal(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS GPT
# ══════════════════════════════════════════════════════════════════════════════

class CrystalGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=256, embed_dim=128, depth=4,
                 field_size=32, n_steps=6, n_crystal_steps=3):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[
            CrystalBlock(embed_dim, field_size, n_steps, n_crystal_steps)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        x = self.tok_embed(idx) + self.pos_embed[:, :N, :]
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


class TransformerGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=256, embed_dim=128, depth=4, n_heads=4):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        x = self.tok_embed(idx) + self.pos_embed[:, :N, :]
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def download_shakespeare():
    path = './data/shakespeare.txt'
    os.makedirs('./data', exist_ok=True)
    if not os.path.exists(path):
        print("Baixando Shakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


class CharDataset:
    def __init__(self, text, ctx_len=256):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.ctx_len = ctx_len

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])

    def get_batch(self, batch_size, device='cpu'):
        ix = torch.randint(0, len(self.data) - self.ctx_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.ctx_len] for i in ix]).to(device)
        y = torch.stack([self.data[i+1:i+self.ctx_len+1] for i in ix]).to(device)
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# TREINO
# ══════════════════════════════════════════════════════════════════════════════

def treinar_modelo(model, dataset, nome, n_steps=3000, batch_size=32, eval_interval=500):
    model = model.to(DEVICE)
    n_params = model.count_params()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f"\n{'='*60}")
    print(f"{nome}: {n_params:,} params")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0_total = time.time()
    losses = []

    for step in range(n_steps):
        model.train()
        x, y = dataset.get_batch(batch_size, DEVICE)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % eval_interval == 0 or step == 0:
            avg_loss = sum(losses[-eval_interval:]) / len(losses[-eval_interval:])
            dt = time.time() - t0_total
            mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  step {step+1:>5d}/{n_steps}  loss={avg_loss:.3f}  "
                  f"mem={mem:.0f}MB  tempo={dt:.0f}s")

    total_time = time.time() - t0_total
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    # Gerar amostras
    model.eval()
    prompts = ["ROMEO:", "To be ", "KING R", "The qu"]
    print(f"\n  --- Geracoes ({nome}) ---")
    for prompt in prompts:
        try:
            prompt_ids = torch.tensor([dataset.encode(prompt)], device=DEVICE)
            generated = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8)
            text_out = dataset.decode(generated[0].tolist())
            # Mostrar só primeiras 2 linhas
            lines = text_out.split('\n')
            preview = '\n'.join(lines[:3])
            print(f"  > {preview}")
            print()
        except Exception as e:
            print(f"  > {prompt}... ERRO: {e}")

    print(f"  Tempo total: {total_time:.0f}s  |  Mem peak: {peak_mem:.0f} MB")

    return {
        'loss': losses[-1],
        'avg_loss': sum(losses[-100:]) / 100,
        'time': total_time,
        'mem': peak_mem,
        'params': n_params,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    text = download_shakespeare()
    print(f"\nShakespeare: {len(text):,} caracteres")

    CTX_LEN = 256
    EMBED_DIM = 128
    DEPTH = 4
    BATCH_SIZE = 32

    dataset = CharDataset(text, ctx_len=CTX_LEN)
    print(f"Vocabulario: {dataset.vocab_size} chars")
    print(f"Contexto: {CTX_LEN} tokens")

    # ── Transformer GPT (3000 steps) ──
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpt_transformer = TransformerGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN, embed_dim=EMBED_DIM, depth=DEPTH, n_heads=4
    )
    res_transformer = treinar_modelo(
        gpt_transformer, dataset, "Transformer GPT",
        n_steps=3000, batch_size=BATCH_SIZE
    )
    del gpt_transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Crystal GPT (6000 steps, campo 32×32) ──
    gpt_crystal = CrystalGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN, embed_dim=EMBED_DIM, depth=DEPTH,
        field_size=32, n_steps=6, n_crystal_steps=3
    )
    res_crystal = treinar_modelo(
        gpt_crystal, dataset, "Crystal GPT (interferencia + cristalizacao + re-emissao)",
        n_steps=6000, batch_size=BATCH_SIZE
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RESUMO
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL — Language Model (Shakespeare)")
    print(f"{'='*70}")
    print(f"  {'':>20s}  {'Transformer':>15s}  {'Crystal GPT':>15s}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}")
    print(f"  {'Parametros':>20s}  {res_transformer['params']:>12,}  {res_crystal['params']:>12,}")
    print(f"  {'Loss final':>20s}  {res_transformer['avg_loss']:>15.3f}  {res_crystal['avg_loss']:>15.3f}")
    print(f"  {'Mem peak (MB)':>20s}  {res_transformer['mem']:>15.0f}  {res_crystal['mem']:>15.0f}")
    print(f"  {'Tempo (s)':>20s}  {res_transformer['time']:>15.0f}  {res_crystal['time']:>15.0f}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}")

    print(f"\n  Attention do Transformer: Q×K^T = O(N²)")
    print(f"  Attention do Crystal: interferencia + cristalizacao = O(F²) fixo")
    print(f"  Campo: 32×32 = 1024 (fixo, independente de N tokens)")
    print(f"{'='*70}")
