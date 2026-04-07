"""
Wave GPT v3 — Campo de Ondas Recorrente

O problema do v2: o campo processa todos os tokens de uma vez.
Perde a ordem. "ROMEO" e "OEMRO" geram o mesmo campo.

A solucao: o campo evolui TOKEN POR TOKEN.

  token 1 → onda no campo → campo propaga → estado_1
  token 2 → onda no campo → campo propaga → estado_2 (lembra token 1)
  token 3 → onda no campo → campo propaga → estado_3 (lembra 1 e 2)

O campo 32x32 eh a memoria. Fixa. Nao cresce com N.
Mas acumula historia porque cada propagacao mistura o token
novo com o historico que ja esta no campo.

Eh uma RNN onde o estado oculto eh um campo de ondas 2D.
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
# WAVE RNN ATTENTION — campo recorrente, token por token
# ══════════════════════════════════════════════════════════════════════════════

class WaveRNNAttention(nn.Module):
    """
    Cada token injeta uma onda no campo.
    O campo propaga (mistura com historico).
    O estado do campo eh lido como contexto pro token.

    Memoria: campo F×F fixo. Sempre.
    Sequencia: natural — cada token ve o campo com toda a historia anterior.
    """
    def __init__(self, embed_dim=128, field_size=24, prop_steps=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_size = field_size
        self.prop_steps = prop_steps  # steps de propagacao por token

        # Token → onda no campo
        self.to_wave = nn.Linear(embed_dim, field_size * field_size)

        # Parametros fisicos treinaveis
        self.c2 = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.03))

        # Gate: quanto do campo antigo manter vs quanto do token novo injetar
        self.gate = nn.Linear(embed_dim, 1)

        # Laplaciano
        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

        # Leitura: campo F×F → embed_dim
        self.from_field = nn.Linear(field_size * field_size, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def propagate_step(self, field, velocity):
        """Um step de propagacao de onda."""
        dt = 0.1
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
        Processa token por token, acumulando no campo.
        """
        B, N, D = x.shape
        FS = self.field_size

        # Todos os tokens → padroes de onda
        wave_patterns = self.to_wave(x).view(B, N, FS, FS)

        # Gate por token: quanto injetar
        gates = torch.sigmoid(self.gate(x)).view(B, N, 1, 1)

        # Campo e velocidade iniciais
        field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
        velocity = torch.zeros_like(field)

        outputs = []

        for t in range(N):
            # Injetar onda do token t no campo (modulada pelo gate)
            injection = wave_patterns[:, t] * gates[:, t]
            field = field + injection * 0.1

            # Propagar — campo mistura token novo com historico
            for _ in range(self.prop_steps):
                field, velocity = self.propagate_step(field, velocity)

            # Ler o campo — contem info de todos os tokens 0..t
            field_flat = field.reshape(B, FS * FS)
            out_t = self.from_field(field_flat)  # (B, D)
            outputs.append(out_t)

        # (B, N, D)
        out = torch.stack(outputs, dim=1)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# VERSAO EFICIENTE: processa em chunks pra nao ser lento demais
# ══════════════════════════════════════════════════════════════════════════════

class WaveRNNChunked(nn.Module):
    """
    Versao que processa em chunks de C tokens.
    Dentro do chunk: todos os tokens de uma vez (paralelo).
    Entre chunks: campo acumula (sequencial).

    Compromisso: nao eh token-por-token puro, mas eh muito mais rapido.
    Chunk de 16 tokens com contexto 256 = 16 steps sequenciais em vez de 256.
    """
    def __init__(self, embed_dim=128, field_size=24, prop_steps=3, chunk_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_size = field_size
        self.prop_steps = prop_steps
        self.chunk_size = chunk_size

        self.to_wave = nn.Sequential(
            nn.Linear(embed_dim, field_size * field_size),
            nn.Dropout(0.2),
        )

        self.c2 = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.03))

        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

        self.from_field = nn.Sequential(
            nn.Linear(field_size * field_size, embed_dim),
            nn.Dropout(0.2),
        )
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def propagate_step(self, field, velocity):
        dt = 0.1
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
        B, N, D = x.shape
        FS = self.field_size
        C = self.chunk_size

        wave_patterns = self.to_wave(x).view(B, N, FS, FS)

        field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
        velocity = torch.zeros_like(field)

        outputs = []

        for start in range(0, N, C):
            end = min(start + C, N)
            chunk = wave_patterns[:, start:end]  # (B, chunk, FS, FS)

            # Soma do chunk inteiro como source
            source = chunk.sum(dim=1)  # (B, FS, FS)
            field = field + source * 0.1

            # Propagar — campo mistura chunk com historico
            for _ in range(self.prop_steps):
                field, velocity = self.propagate_step(field, velocity)

            # Cada token do chunk le o campo
            field_flat = field.reshape(B, 1, FS * FS).expand(B, end - start, FS * FS)
            chunk_out = self.from_field(field_flat)  # (B, chunk, D)
            outputs.append(chunk_out)

        out = torch.cat(outputs, dim=1)  # (B, N, D)
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
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCOS
# ══════════════════════════════════════════════════════════════════════════════

class WaveRNNBlock(nn.Module):
    def __init__(self, embed_dim=128, field_size=24, prop_steps=3,
                 chunk_size=16, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WaveRNNChunked(embed_dim, field_size, prop_steps, chunk_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(0.2),
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

class WaveRNNGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=256, embed_dim=128, depth=4,
                 field_size=24, prop_steps=3, chunk_size=16):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[
            WaveRNNBlock(embed_dim, field_size, prop_steps, chunk_size)
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

def download_text():
    """Baixa Shakespeare + livros do Gutenberg (~6MB total)."""
    os.makedirs('./data', exist_ok=True)

    sources = {
        'shakespeare.txt': "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        'pride_prejudice.txt': "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        'great_expectations.txt': "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
        'tale_two_cities.txt': "https://www.gutenberg.org/cache/epub/98/pg98.txt",
        'oliver_twist.txt': "https://www.gutenberg.org/cache/epub/730/pg730.txt",
        'moby_dick.txt': "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    }

    all_text = []
    for fname, url in sources.items():
        path = f'./data/{fname}'
        if not os.path.exists(path):
            print(f"Baixando {fname}...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  ERRO baixando {fname}: {e}")
                continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            all_text.append(f.read())

    text = '\n\n'.join(all_text)
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

    model.eval()
    prompts = ["It was ", "The man", "She had"]
    print(f"\n  --- Geracoes ({nome}) ---")
    for prompt in prompts:
        try:
            prompt_ids = torch.tensor([dataset.encode(prompt)], device=DEVICE)
            generated = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8)
            text_out = dataset.decode(generated[0].tolist())
            lines = text_out.split('\n')
            preview = '\n'.join(lines[:4])
            print(f"  > {preview}")
            print()
        except Exception as e:
            print(f"  > {prompt}... ERRO: {e}")

    print(f"  Tempo total: {total_time:.0f}s  |  Mem peak: {peak_mem:.0f} MB")

    return {
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

    text = download_text()
    print(f"\nTexto total: {len(text):,} caracteres ({len(text)/(1024*1024):.1f} MB)")

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

    gpt_t = TransformerGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN, embed_dim=EMBED_DIM, depth=DEPTH, n_heads=4
    )
    res_t = treinar_modelo(gpt_t, dataset, "Transformer GPT", n_steps=3000, batch_size=BATCH_SIZE)
    del gpt_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Wave RNN GPT (6000 steps, campo 24×24, chunks de 16) ──
    gpt_w = WaveRNNGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN, embed_dim=EMBED_DIM, depth=DEPTH,
        field_size=24, prop_steps=3, chunk_size=16
    )
    res_w = treinar_modelo(gpt_w, dataset, "Wave RNN GPT (campo recorrente 24x24)",
                           n_steps=3000, batch_size=BATCH_SIZE)

    # ══════════════════════════════════════════════════════════════════════════
    # RESUMO
    # ═════════════════════════════════════════════════════════════════════════=

    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL — Shakespeare char-level LM")
    print(f"{'='*70}")
    print(f"  {'':>20s}  {'Transformer':>15s}  {'Wave RNN':>15s}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}")
    print(f"  {'Parametros':>20s}  {res_t['params']:>12,}  {res_w['params']:>12,}")
    print(f"  {'Loss final':>20s}  {res_t['avg_loss']:>15.3f}  {res_w['avg_loss']:>15.3f}")
    print(f"  {'Mem peak (MB)':>20s}  {res_t['mem']:>15.0f}  {res_w['mem']:>15.0f}")
    print(f"  {'Tempo (s)':>20s}  {res_t['time']:>15.0f}  {res_w['time']:>15.0f}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}")
    print(f"\n  Transformer: Q×K^T = O(N²) por camada")
    print(f"  Wave RNN: campo 24×24 = 576 fixo, recorrente por chunks de 16")
    print(f"  Com N=256: parecido. Com N=10000+: Transformer explode, Wave nao.")
    print(f"{'='*70}")
