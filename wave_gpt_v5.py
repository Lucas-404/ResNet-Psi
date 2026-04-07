"""
Wave GPT v5 — Assinaturas Hierarquicas (versao vetorizada)

Hierarquia de assinaturas no campo de ondas:
  Char:     cada letra → onda simples
  Palavra:  chars da palavra somam → assinatura unica (scatter_add, paralelo)
  Contexto: assinaturas de palavras acumulam no campo (sequencial por palavra)

Diferenca do v5 anterior: VETORIZADO.
Em vez de iterar char-por-char, batch-por-batch:
  - Chars → ondas: tudo em paralelo (matmul)
  - Chars → palavras: scatter_add (paralelo)
  - Palavras → campo: sequencial mas sao ~50 palavras, nao 256 chars
  - Leitura: expandir campo pra todos os chars da palavra (paralelo)

~50 steps sequenciais em vez de 256. Muito mais rapido.
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
# WAVE FIELD
# ══════════════════════════════════════════════════════════════════════════════

class WaveField(nn.Module):
    def __init__(self, field_size=24, prop_steps=3):
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
        field = field + source * 0.1
        for _ in range(self.prop_steps):
            field, velocity = self.propagate(field, velocity)
        return field, velocity


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL WAVE ATTENTION — VETORIZADO
# ══════════════════════════════════════════════════════════════════════════════

class HierWaveAttention(nn.Module):
    """
    Assinaturas hierarquicas vetorizadas.

    1. Todos os chars → ondas (paralelo, matmul)
    2. Agrupar chars por palavra via word_ids (scatter_add, paralelo)
       Resultado: assinatura de cada palavra = soma das ondas dos seus chars
    3. Processar palavras sequencialmente no campo (campo acumula contexto)
       Sao ~50 palavras por sequencia de 256 chars (nao 256 steps)
    4. Cada char recebe: embedding original + assinatura da sua palavra + campo de contexto
    """
    def __init__(self, embed_dim=128, field_size=24, prop_steps=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_size = field_size
        FS2 = field_size * field_size

        # Char → onda
        self.char_to_wave = nn.Linear(embed_dim, FS2)

        # Campo
        self.wave = WaveField(field_size, prop_steps)

        # Leitura
        self.word_read = nn.Linear(FS2, embed_dim)
        self.ctx_read = nn.Linear(FS2, embed_dim)

        # Combinar: char + palavra + contexto → saida
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, word_ids, n_words_per_batch):
        """
        x: (B, N, D)
        word_ids: (B, N) — ID da palavra a que cada char pertence (0,0,0,1,1,2,2,2,...)
        n_words_per_batch: (B,) — quantas palavras em cada item do batch
        """
        B, N, D = x.shape
        FS = self.field_size
        FS2 = FS * FS

        # 1. Todos os chars → ondas (paralelo)
        char_waves = self.char_to_wave(x)  # (B, N, FS2)

        # 2. Agrupar por palavra: scatter_add
        # Para cada batch, somar as ondas dos chars que pertencem a mesma palavra
        max_words = n_words_per_batch.max().item()

        # word_ids expandido pra FS2 dims
        word_ids_exp = word_ids.unsqueeze(-1).expand(B, N, FS2)  # (B, N, FS2)

        # Scatter: somar chars da mesma palavra
        word_waves = torch.zeros(B, max_words, FS2, device=x.device, dtype=x.dtype)
        word_waves.scatter_add_(1, word_ids_exp, char_waves)
        # word_waves: (B, max_words, FS2) — assinatura de cada palavra

        # 3. Processar palavras sequencialmente no campo
        ctx_field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
        ctx_velocity = torch.zeros_like(ctx_field)

        # Guardar campo de contexto DEPOIS de cada palavra
        # pra que chars da palavra W vejam contexto das palavras 0..W-1
        ctx_fields_per_word = []

        # Tambem guardar assinatura de cada palavra apos propagar no campo local
        word_sigs = []

        for w in range(max_words):
            # Assinatura da palavra w: propagar num campo temporario
            word_source = word_waves[:, w].view(B, FS, FS)

            word_field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
            word_vel = torch.zeros_like(word_field)
            word_field, word_vel = self.wave(word_field, word_vel, word_source)

            # Salvar assinatura da palavra
            word_sigs.append(word_field.reshape(B, FS2))

            # Salvar contexto ANTES de adicionar esta palavra
            # (chars da palavra W veem contexto de 0..W-1, causal)
            ctx_fields_per_word.append(ctx_field.reshape(B, FS2).clone())

            # Acumular no campo de contexto
            ctx_field, ctx_velocity = self.wave(ctx_field, ctx_velocity, word_field)

        # word_sigs: list de (B, FS2), len = max_words
        word_sigs = torch.stack(word_sigs, dim=1)       # (B, max_words, FS2)
        ctx_per_word = torch.stack(ctx_fields_per_word, dim=1)  # (B, max_words, FS2)

        # 4. Cada char recebe a assinatura da sua palavra + contexto
        # Usar word_ids pra indexar
        # word_info[b, t] = word_sigs[b, word_ids[b,t]]
        word_ids_read = word_ids.unsqueeze(-1).expand(B, N, FS2)
        word_info_raw = torch.gather(word_sigs, 1, word_ids_read)   # (B, N, FS2)
        ctx_info_raw = torch.gather(ctx_per_word, 1, word_ids_read)  # (B, N, FS2)

        word_info = self.word_read(word_info_raw)   # (B, N, D)
        ctx_info = self.ctx_read(ctx_info_raw)      # (B, N, D)

        # 5. Combinar: char original + palavra + contexto
        combined = torch.cat([x, word_info, ctx_info], dim=-1)  # (B, N, 3D)
        return self.combine(combined)  # (B, N, D)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCOS + MODELOS
# ══════════════════════════════════════════════════════════════════════════════

class HierWaveBlock(nn.Module):
    def __init__(self, embed_dim=128, field_size=24, prop_steps=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = HierWaveAttention(embed_dim, field_size, prop_steps)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(0.2),
        )

    def forward(self, x, word_ids, n_words):
        x = x + self.attn(self.norm1(x), word_ids, n_words)
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


class HierWaveGPT(nn.Module):
    def __init__(self, vocab_size, ctx_len=256, embed_dim=128, depth=4,
                 field_size=24, prop_steps=3):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            HierWaveBlock(embed_dim, field_size, prop_steps)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self._sep_ids = set()

    def set_vocab(self, itos):
        seps = set(' \t\n\r.,;:!?-()[]{}"\'/\\@#$%&*+=<>~`|^')
        self._sep_ids = set()
        for i, c in itos.items():
            if c in seps:
                self._sep_ids.add(i)

    def _compute_word_ids(self, idx):
        """Computa word_ids e n_words a partir dos token IDs."""
        B, N = idx.shape
        word_ids = torch.zeros(B, N, dtype=torch.long, device=idx.device)
        n_words = torch.zeros(B, dtype=torch.long, device=idx.device)

        for b in range(B):
            wid = 0
            for t in range(N):
                if t > 0 and idx[b, t-1].item() in self._sep_ids:
                    wid += 1
                word_ids[b, t] = wid
            n_words[b] = wid + 1

        return word_ids, n_words

    def forward(self, idx, word_ids=None, n_words=None):
        B, N = idx.shape
        if word_ids is None:
            word_ids, n_words = self._compute_word_ids(idx)

        x = self.tok_embed(idx) + self.pos_embed[:, :N, :]
        for block in self.blocks:
            x = block(x, word_ids, n_words)
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

    def set_vocab(self, itos):
        pass

    def forward(self, idx, word_ids=None, n_words=None):
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
    return '\n\n'.join(all_text)


class CharDataset:
    def __init__(self, text, ctx_len=256):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.ctx_len = ctx_len

        # Pre-computar word_ids pro texto inteiro
        print("  Pre-computando word IDs...")
        seps = set(' \t\n\r.,;:!?-()[]{}"\'/\\@#$%&*+=<>~`|^')
        self.word_starts = torch.zeros(len(text), dtype=torch.bool)
        self.word_starts[0] = True
        for i in range(1, len(text)):
            if text[i-1] in seps:
                self.word_starts[i] = True

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, tokens):
        return ''.join([self.itos.get(t, '?') for t in tokens])

    def get_batch(self, batch_size, device='cpu'):
        ix = torch.randint(0, len(self.data) - self.ctx_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.ctx_len] for i in ix]).to(device)
        y = torch.stack([self.data[i+1:i+self.ctx_len+1] for i in ix]).to(device)

        # Computar word_ids pra cada item do batch
        word_ids = torch.zeros(batch_size, self.ctx_len, dtype=torch.long)
        n_words = torch.zeros(batch_size, dtype=torch.long)
        for b in range(batch_size):
            wid = 0
            starts = self.word_starts[ix[b]:ix[b]+self.ctx_len]
            for t in range(self.ctx_len):
                if t > 0 and starts[t]:
                    wid += 1
                word_ids[b, t] = wid
            n_words[b] = wid + 1

        return x, y, word_ids.to(device), n_words.to(device)


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
        x, y, word_ids, n_words = dataset.get_batch(batch_size, DEVICE)

        logits = model(x, word_ids, n_words)
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
    print(f"Hierarquia: char → palavra (scatter_add) → contexto (campo acumulativo)")

    # ── Transformer GPT ──
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpt_t = TransformerGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN, embed_dim=EMBED_DIM, depth=DEPTH, n_heads=4
    )
    gpt_t.set_vocab(dataset.itos)
    res_t = treinar_modelo(gpt_t, dataset, "Transformer GPT", n_steps=3000, batch_size=BATCH_SIZE)
    del gpt_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Hierarchical Wave GPT ──
    gpt_w = HierWaveGPT(
        vocab_size=dataset.vocab_size,
        ctx_len=CTX_LEN, embed_dim=EMBED_DIM, depth=DEPTH,
        field_size=24, prop_steps=3
    )
    gpt_w.set_vocab(dataset.itos)
    res_w = treinar_modelo(gpt_w, dataset, "Hier Wave GPT (char→word→context)",
                           n_steps=3000, batch_size=BATCH_SIZE)

    # ══════════════════════════════════════════════════════════════════════════
    # RESUMO
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL — v5: assinaturas hierarquicas (vetorizado)")
    print(f"{'='*70}")
    print(f"  {'':>25s}  {'Transformer':>15s}  {'Hier Wave':>15s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*15}")
    print(f"  {'Parametros':>25s}  {res_t['params']:>12,}  {res_w['params']:>12,}")
    print(f"  {'Loss final':>25s}  {res_t['avg_loss']:>15.3f}  {res_w['avg_loss']:>15.3f}")
    print(f"  {'Mem peak (MB)':>25s}  {res_t['mem']:>15.0f}  {res_w['mem']:>15.0f}")
    print(f"  {'Tempo (s)':>25s}  {res_t['time']:>15.0f}  {res_w['time']:>15.0f}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*15}")
    print(f"\n  Hierarquia:")
    print(f"    1. Char → onda (paralelo, matmul)")
    print(f"    2. Palavra → scatter_add das ondas dos chars (paralelo)")
    print(f"    3. Contexto → campo acumula palavras (~50 steps sequenciais)")
    print(f"    4. Cada char le: char + assinatura palavra + campo contexto")
    print(f"  Memoria: campo 24×24 fixo. Sempre O(1).")
    print(f"{'='*70}")
