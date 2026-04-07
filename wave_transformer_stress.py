"""
Stress Test de Memória: Transformer vs Wave Transformer

Testa com sequências cada vez maiores pra ver quando o Transformer
explode de memória e o Wave sobrevive.

O ponto: Q×K^T escala O(N²). Campo de ondas é fixo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# WAVE ATTENTION (campo fixo)
# ══════════════════════════════════════════════════════════════════════════════

class WaveAttention(nn.Module):
    def __init__(self, embed_dim=64, field_size=16, n_steps=8):
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

        wave_patterns = self.to_wave(x).view(B, N, FS, FS)
        source = wave_patterns.sum(dim=1)

        field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
        velocity = torch.zeros_like(field)

        for s in range(self.n_steps):
            active = s < self.n_steps // 2
            field, velocity = self.propagate(field, velocity, source, active)

        field_expanded = field.unsqueeze(1).expand_as(wave_patterns)
        token_reads = wave_patterns * field_expanded
        token_reads = token_reads.view(B, N, FS * FS)
        out = self.from_field(token_reads)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-ATTENTION (QKV original)
# ══════════════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4):
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
        attn = (q @ k.transpose(-2, -1)) / scale  # <-- N×N aqui!
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# MEDIR MEMÓRIA
# ══════════════════════════════════════════════════════════════════════════════

def limpar_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def memoria_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0

def testar_memoria(attn_module, nome, n_tokens, embed_dim=64, batch_size=1):
    """Testa forward + backward e mede memória peak."""
    limpar_gpu()
    attn_module = attn_module.to(DEVICE)

    x = torch.randn(batch_size, n_tokens, embed_dim, device=DEVICE, requires_grad=True)

    limpar_gpu()
    torch.cuda.reset_peak_memory_stats()

    try:
        t0 = time.time()
        # Forward
        out = attn_module(x)
        loss = out.sum()
        # Backward
        loss.backward()
        dt = time.time() - t0

        mem = memoria_mb()
        print(f"  {nome:<30s}  N={n_tokens:>6d}  mem={mem:>8.1f} MB  tempo={dt:>6.2f}s  OK")
        return mem, dt, True
    except RuntimeError as e:
        if "out of memory" in str(e):
            limpar_gpu()
            print(f"  {nome:<30s}  N={n_tokens:>6d}  >>> OOM (sem memoria!) <<<")
            return None, None, False
        else:
            raise e
    finally:
        limpar_gpu()


# ══════════════════════════════════════════════════════════════════════════════
# STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print()

    EMBED_DIM = 64
    BATCH_SIZE = 4

    # Sequências cada vez maiores
    token_counts = [64, 256, 512, 1024, 2048, 4096, 8192, 10000, 16384, 32768]

    print(f"{'='*80}")
    print(f"STRESS TEST: Transformer vs Wave Transformer")
    print(f"  embed_dim={EMBED_DIM}, batch_size={BATCH_SIZE}")
    print(f"  Transformer: Q×K^T = O(N²) memoria")
    print(f"  Wave: campo 16×16 = O(1) memoria (fixo)")
    print(f"{'='*80}")

    resultados_transformer = {}
    resultados_wave = {}

    for n_tokens in token_counts:
        print(f"\n--- {n_tokens} tokens ---")

        # Transformer
        attn_qkv = SelfAttention(EMBED_DIM, n_heads=4)
        mem_t, dt_t, ok_t = testar_memoria(attn_qkv, "Transformer (QKV)", n_tokens, EMBED_DIM, BATCH_SIZE)
        if ok_t:
            resultados_transformer[n_tokens] = (mem_t, dt_t)
        del attn_qkv
        limpar_gpu()

        # Wave
        attn_wave = WaveAttention(EMBED_DIM, field_size=16, n_steps=8)
        mem_w, dt_w, ok_w = testar_memoria(attn_wave, "Wave (campo 16x16)", n_tokens, EMBED_DIM, BATCH_SIZE)
        if ok_w:
            resultados_wave[n_tokens] = (mem_w, dt_w)
        del attn_wave
        limpar_gpu()

        # Se os dois falharam, para
        if not ok_t and not ok_w:
            print("  Ambos OOM. Parando.")
            break

    # ══════════════════════════════════════════════════════════════════════════
    # RESUMO
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"RESUMO — Consumo de Memoria (forward + backward)")
    print(f"{'='*80}")
    print(f"  {'Tokens':>8s}  {'Transformer':>14s}  {'Wave':>14s}  {'Razao T/W':>10s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}")

    for n in token_counts:
        t_str = f"{resultados_transformer[n][0]:.1f} MB" if n in resultados_transformer else "OOM"
        w_str = f"{resultados_wave[n][0]:.1f} MB" if n in resultados_wave else "OOM"

        if n in resultados_transformer and n in resultados_wave:
            razao = resultados_transformer[n][0] / resultados_wave[n][0]
            r_str = f"{razao:.1f}x"
        else:
            r_str = "---"

        print(f"  {n:>8d}  {t_str:>14s}  {w_str:>14s}  {r_str:>10s}")

    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}")
    print(f"{'='*80}")

    # Tempos
    print(f"\n  {'Tokens':>8s}  {'T tempo':>10s}  {'W tempo':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")
    for n in token_counts:
        t_str = f"{resultados_transformer[n][1]:.2f}s" if n in resultados_transformer else "OOM"
        w_str = f"{resultados_wave[n][1]:.2f}s" if n in resultados_wave else "OOM"
        print(f"  {n:>8d}  {t_str:>10s}  {w_str:>10s}")
    print(f"{'='*80}")
