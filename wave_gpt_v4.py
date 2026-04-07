"""
Wave GPT v4 — Tokens como Padroes 2D ("imagens de texto")

O problema do v3: cada token eh 1 caractere. Projecao linear pra campo.
Sem estrutura geometrica. Campo nao consegue gerar assinaturas distintas.
Resultado: colapsa em repeticao ("IIIII", "eeeee").

A solucao: CHUNKS de caracteres viram padroes 2D.

Em vez de:
  char 'a' → embedding 128d → linear → campo 24×24 (projecao sem estrutura)

Fazemos:
  chunk "the " (16 chars) → cada char vira embedding → reshape 4×4×embed
  → convolucao 2D → padrao 2D rico → injeta no campo como "imagem"

Cada chunk de texto gera uma ASSINATURA GEOMETRICA unica no campo.
"the " gera um padrao diferente de "and " porque a disposicao espacial
dos embeddings eh diferente.

Isso eh exatamente como a ResNet-Psi funciona com imagens:
  imagem 28×28 → projecao → campo → interferencia → assinatura

Agora:
  chunk de texto → "imagem 2D" → campo → interferencia → assinatura
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import os
import urllib.request
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK ENCODER — transforma chunk de chars em padrao 2D
# ══════════════════════════════════════════════════════════════════════════════

class ChunkTo2D(nn.Module):
    """
    Pega um chunk de C caracteres e gera um padrao 2D (field_size × field_size).

    Processo:
      1. Cada char → embedding (C, embed_dim)
      2. Reshape pra grid 2D: (C, embed_dim) → (1, sqrt(C) * sqrt(embed), ...)
      3. Conv2D pra gerar padrao no tamanho do campo

    O ponto: caracteres VIZINHOS no texto ficam VIZINHOS no grid 2D.
    "the " → t,h,e,' ' ficam em posicoes adjacentes.
    Isso cria estrutura geometrica que o campo de ondas pode explorar.
    """
    def __init__(self, vocab_size, char_embed_dim=32, chunk_size=16, field_size=24):
        super().__init__()
        self.chunk_size = chunk_size
        self.char_embed_dim = char_embed_dim
        self.field_size = field_size

        self.char_embed = nn.Embedding(vocab_size, char_embed_dim)

        # chunk_size chars × char_embed_dim → reshape pra 2D
        # ex: 16 chars × 32 embed = 512 valores → ~22×23 ou usamos conv
        # Melhor: tratar como imagem 1-canal de (chunk_size, char_embed_dim)
        # = (16, 32) → conv2d → (field_size, field_size)

        # Conv que transforma (1, chunk_size, char_embed_dim) → (1, field_size, field_size)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((field_size, field_size)),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, chunk_ids):
        """
        chunk_ids: (B, C) — indices de caracteres
        retorna: (B, field_size, field_size) — padrao 2D
        """
        B, C = chunk_ids.shape
        # (B, C) → (B, C, char_embed_dim)
        emb = self.char_embed(chunk_ids)
        # (B, C, D) → (B, 1, C, D) — tratar como imagem 1-canal
        emb = emb.unsqueeze(1)
        # Conv → (B, 1, field_size, field_size)
        pattern = self.conv(emb)
        return pattern.squeeze(1)  # (B, field_size, field_size)


# ══════════════════════════════════════════════════════════════════════════════
# WAVE FIELD — propagacao de ondas (igual ResNet-Psi)
# ══════════════════════════════════════════════════════════════════════════════

class WaveField(nn.Module):
    def __init__(self, field_size=24, prop_steps=4):
        super().__init__()
        self.field_size = field_size
        self.prop_steps = prop_steps

        self.c2 = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.08))
        self.alpha = nn.Parameter(torch.tensor(0.04))

        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

    def propagate(self, field, velocity):
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

    def forward(self, field, velocity, source):
        """Injeta source no campo e propaga."""
        field = field + source * 0.1
        for _ in range(self.prop_steps):
            field, velocity = self.propagate(field, velocity)
        return field, velocity


# ══════════════════════════════════════════════════════════════════════════════
# WAVE CHUNK ATTENTION — chunks viram imagens → campo → leitura
# ══════════════════════════════════════════════════════════════════════════════

class WaveChunkAttention(nn.Module):
    """
    Attention por campo de ondas com tokens como padroes 2D.

    Pipeline:
      1. Sequencia de N chars → dividir em chunks de C
      2. Cada chunk → padrao 2D via ChunkTo2D (como uma "imagem")
      3. Padrao 2D → injetado no campo de ondas (acumulativo)
      4. Campo propaga (mistura chunk com historico)
      5. Campo eh lido → saida do chunk

    Causalidade: chunk t so ve campo com chunks 0..t (recorrente).
    Memoria: campo F×F fixo. Sempre. Independente de N.
    """
    def __init__(self, vocab_size, embed_dim=128, field_size=24,
                 prop_steps=4, chunk_size=16, char_embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_size = field_size
        self.chunk_size = chunk_size

        # Chunk → padrao 2D
        self.chunk_to_2d = ChunkTo2D(vocab_size, char_embed_dim, chunk_size, field_size)

        # Campo de ondas
        self.wave_field = WaveField(field_size, prop_steps)

        # Leitura: campo → embed_dim por posicao no chunk
        self.from_field = nn.Sequential(
            nn.Linear(field_size * field_size, embed_dim),
            nn.Dropout(0.2),
        )

        # Posicao dentro do chunk (pra diferenciar tokens no mesmo chunk)
        self.pos_in_chunk = nn.Parameter(torch.randn(1, chunk_size, embed_dim) * 0.02)

        self.proj_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, token_ids, token_embeds):
        """
        token_ids: (B, N) — IDs originais dos chars
        token_embeds: (B, N, D) — embeddings ja com pos_embed
        retorna: (B, N, D)
        """
        B, N, D = token_embeds.shape
        FS = self.field_size
        C = self.chunk_size

        # Pad pra multiplo de chunk_size
        pad_len = (C - N % C) % C
        if pad_len > 0:
            token_ids = F.pad(token_ids, (0, pad_len), value=0)
            token_embeds = F.pad(token_embeds, (0, 0, 0, pad_len))
        N_padded = token_ids.shape[1]
        n_chunks = N_padded // C

        # Reshape em chunks: (B, n_chunks, C)
        chunk_ids = token_ids.view(B, n_chunks, C)

        # Campo inicial
        field = torch.zeros(B, FS, FS, device=token_ids.device, dtype=token_embeds.dtype)
        velocity = torch.zeros_like(field)

        outputs = []

        for i in range(n_chunks):
            # Chunk → padrao 2D: (B, C) → (B, FS, FS)
            pattern_2d = self.chunk_to_2d(chunk_ids[:, i])

            # Injetar no campo e propagar
            field, velocity = self.wave_field(field, velocity, pattern_2d)

            # Ler campo → embedding
            field_flat = field.reshape(B, FS * FS)
            field_read = self.from_field(field_flat)  # (B, D)

            # Expandir pra cada posicao no chunk + posicao local
            chunk_len = min(C, N - i * C) if i == n_chunks - 1 and pad_len > 0 else C
            field_expanded = field_read.unsqueeze(1).expand(B, C, D)  # (B, C, D)

            # Somar posicao dentro do chunk pra diferenciar tokens
            chunk_out = field_expanded + self.pos_in_chunk[:, :C, :]

            outputs.append(chunk_out)

        # (B, N_padded, D) → cortar pro tamanho original
        out = torch.cat(outputs, dim=1)[:, :N, :]
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCOS
# ══════════════════════════════════════════════════════════════════════════════

class WaveChunkBlock(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, field_size=24,
                 prop_steps=4, chunk_size=16, char_embed_dim=32, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WaveChunkAttention(
            vocab_size, embed_dim, field_size, prop_steps, chunk_size, char_embed_dim
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(0.2),
        )

    def forward(self, token_ids, x):
        x = x + self.attn(token_ids, self.norm1(x))
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
# MODELOS GPT
# ══════════════════════════════════════════════════════════════════════════════

class WaveChunkGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=256, embed_dim=128, depth=4,
                 field_size=24, prop_steps=4, chunk_size=16, char_embed_dim=32):
        super().__init__()
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            WaveChunkBlock(vocab_size, embed_dim, field_size, prop_steps,
                           chunk_size, char_embed_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        x = self.tok_embed(idx) + self.pos_embed[:, :N, :]
        for block in self.blocks:
            x = block(idx, x)
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
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, tokens):
        return ''.join([self.itos.get(t, '?') for t in tokens])

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    print(f"Chunk size: 16 chars → padrao 2D → campo 24×24")

    # ── Transformer GPT (referencia) ──
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

    # ── Wave Chunk GPT (chunks como imagens 2D) ──
    gpt_w = WaveChunkGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN, embed_dim=EMBED_DIM, depth=DEPTH,
        field_size=24, prop_steps=4, chunk_size=16, char_embed_dim=32
    )
    res_w = treinar_modelo(gpt_w, dataset, "Wave Chunk GPT (chunks→2D→campo 24x24)",
                           n_steps=3000, batch_size=BATCH_SIZE)

    # ══════════════════════════════════════════════════════════════════════════
    # RESUMO
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL — v4: chunks de texto como padroes 2D")
    print(f"{'='*70}")
    print(f"  {'':>25s}  {'Transformer':>15s}  {'Wave Chunk':>15s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*15}")
    print(f"  {'Parametros':>25s}  {res_t['params']:>12,}  {res_w['params']:>12,}")
    print(f"  {'Loss final':>25s}  {res_t['avg_loss']:>15.3f}  {res_w['avg_loss']:>15.3f}")
    print(f"  {'Mem peak (MB)':>25s}  {res_t['mem']:>15.0f}  {res_w['mem']:>15.0f}")
    print(f"  {'Tempo (s)':>25s}  {res_t['time']:>15.0f}  {res_w['time']:>15.0f}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*15}")
    print(f"\n  Diferenca do v3:")
    print(f"    v3: char individual → linear → campo (sem estrutura)")
    print(f"    v4: chunk 16 chars → conv2d → padrao 2D → campo (com estrutura geometrica)")
    print(f"  Memoria: campo 24×24 = 576 fixo. Sempre O(1).")
    print(f"{'='*70}")
