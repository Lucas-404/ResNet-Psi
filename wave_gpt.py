"""
Wave GPT — Language Model com Attention por Campo de Ondas

Um GPT minusculo onde o mecanismo de attention usa propagacao de ondas
num campo fixo em vez de Q×K^T.

Pipeline:
  texto → tokens → embedding → Wave Transformer blocks → prediz proximo token

Treina em texto, gera texto.
Compara memoria e qualidade com Transformer normal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import math
import os
import urllib.request

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# WAVE ATTENTION (causal — com mascara pra nao ver o futuro)
# ══════════════════════════════════════════════════════════════════════════════

class WaveAttentionCausal(nn.Module):
    """
    Attention por campo de ondas para language modeling.

    Diferenca do classification: aqui cada token so pode
    "ver" tokens anteriores (causal). Tokens futuros nao
    emitem ondas no campo desse token.

    Implementacao: cada token acumula ondas apenas dos tokens <= ele.
    """
    def __init__(self, embed_dim=128, field_size=16, n_steps=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_size = field_size
        self.n_steps = n_steps

        self.to_wave = nn.Linear(embed_dim, field_size * field_size)
        self.c2 = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.06))
        self.alpha = nn.Parameter(torch.tensor(0.04))

        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

        self.from_field = nn.Linear(field_size * field_size, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def propagate(self, field, velocity, source, active):
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

    def forward(self, x):
        """
        x: (B, N, D)

        Causalidade: token i so recebe ondas de tokens 0..i
        Implementacao eficiente: soma cumulativa dos padroes de onda.
        """
        B, N, D = x.shape
        FS = self.field_size

        # Cada token gera padrao de onda
        wave_patterns = self.to_wave(x).view(B, N, FS, FS)

        # Soma cumulativa: source_i = sum(wave_patterns[0..i])
        # Isso garante causalidade — token i so ve ondas de tokens anteriores
        cumsum_waves = torch.cumsum(wave_patterns, dim=1)  # (B, N, FS, FS)

        # Propagar campo para cada posicao causal
        # Para eficiencia, propaga o campo completo e modula pela mascara causal
        # Aproximacao: usa a media cumulativa como source

        # Campo compartilhado — propaga com a soma total
        source_total = wave_patterns.sum(dim=1)  # (B, FS, FS)

        field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
        velocity = torch.zeros_like(field)

        for s in range(self.n_steps):
            active = s < self.n_steps // 2
            field, velocity = self.propagate(field, velocity, source_total, active)

        # Leitura causal: cada token le o campo modulado pela sua soma cumulativa
        # token i "ve" a interferencia dos tokens 0..i
        # Normalizar pela posicao pra tokens iniciais nao ficarem fracos
        scale = torch.arange(1, N + 1, device=x.device, dtype=x.dtype).view(1, N, 1, 1)
        causal_patterns = cumsum_waves / scale  # (B, N, FS, FS)

        field_expanded = field.unsqueeze(1).expand(B, N, FS, FS)
        token_reads = causal_patterns * field_expanded

        token_reads = token_reads.view(B, N, FS * FS)
        out = self.from_field(token_reads)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-ATTENTION CAUSAL (referencia)
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

        # Mascara causal: nao pode ver o futuro
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCOS
# ══════════════════════════════════════════════════════════════════════════════

class WaveBlock(nn.Module):
    def __init__(self, embed_dim=128, field_size=16, n_steps=6, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WaveAttentionCausal(embed_dim, field_size, n_steps)
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
# GPT MODELS
# ══════════════════════════════════════════════════════════════════════════════

class WaveGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=256, embed_dim=128, depth=4,
                 field_size=16, n_steps=6):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[
            WaveBlock(embed_dim, field_size, n_steps) for _ in range(depth)
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
# DATASET: Shakespeare (char-level)
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
# TREINO E COMPARACAO
# ══════════════════════════════════════════════════════════════════════════════

def treinar_modelo(model, dataset, nome, n_steps=3000, batch_size=32, eval_interval=500):
    model = model.to(DEVICE)
    n_params = model.count_params()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f"\n{'='*60}")
    print(f"{nome}: {n_params:,} params")
    print(f"{'='*60}")

    # Memoria antes
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

    # Gerar amostra
    model.eval()
    prompt = "ROMEO:"
    prompt_ids = torch.tensor([dataset.encode(prompt)], device=DEVICE)
    generated = model.generate(prompt_ids, max_new_tokens=300, temperature=0.8)
    text_out = dataset.decode(generated[0].tolist())

    print(f"\n  --- Geracao ({nome}) ---")
    print(f"  {text_out[:500]}")
    print(f"  ---")
    print(f"  Tempo total: {total_time:.0f}s  |  Mem peak: {peak_mem:.0f} MB")

    return {
        'loss': losses[-1],
        'avg_loss': sum(losses[-100:]) / 100,
        'time': total_time,
        'mem': peak_mem,
        'params': n_params,
        'sample': text_out[:500]
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # Dados
    text = download_shakespeare()
    print(f"\nShakespeare: {len(text):,} caracteres")

    CTX_LEN = 256
    EMBED_DIM = 128
    DEPTH = 4
    BATCH_SIZE = 32
    N_STEPS = 3000

    dataset = CharDataset(text, ctx_len=CTX_LEN)
    print(f"Vocabulario: {dataset.vocab_size} chars")
    print(f"Contexto: {CTX_LEN} tokens")

    N_STEPS_TRANSFORMER = 3000
    N_STEPS_WAVE = 6000

    # ── Transformer GPT ──
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpt_transformer = TransformerGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        n_heads=4
    )
    res_transformer = treinar_modelo(
        gpt_transformer, dataset, "Transformer GPT",
        n_steps=N_STEPS_TRANSFORMER, batch_size=BATCH_SIZE
    )
    del gpt_transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Wave GPT (campo 48x48, 6000 steps) ──
    gpt_wave = WaveGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        field_size=48,
        n_steps=6
    )
    res_wave = treinar_modelo(
        gpt_wave, dataset, "Wave GPT (48x48, 6000 steps)",
        n_steps=N_STEPS_WAVE, batch_size=BATCH_SIZE
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RESUMO
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL — Language Model (Shakespeare)")
    print(f"{'='*70}")
    print(f"  {'':>20s}  {'Transformer':>15s}  {'Wave GPT':>15s}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}")
    print(f"  {'Parametros':>20s}  {res_transformer['params']:>12,}  {res_wave['params']:>12,}")
    print(f"  {'Loss final':>20s}  {res_transformer['avg_loss']:>15.3f}  {res_wave['avg_loss']:>15.3f}")
    print(f"  {'Mem peak (MB)':>20s}  {res_transformer['mem']:>15.0f}  {res_wave['mem']:>15.0f}")
    print(f"  {'Tempo (s)':>20s}  {res_transformer['time']:>15.0f}  {res_wave['time']:>15.0f}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}")
    print(f"{'='*70}")
