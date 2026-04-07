"""
Experimento 2: Lei de escala do PsiField.

Pergunta: capacidade = k × N²?

Metodologia:
  - Varia o tamanho do campo: 24, 32, 48, 64, 96, 128
  - Para cada tamanho: injeta padrões sequencialmente até saturar
  - Mede ponto de saturação (n_padrões) e ocupação no momento da saturação
  - Plota capacidade vs N² — se for linha reta: lei de escala confirmada
  - Extrai constante k = capacidade / N²

Se k for constante entre tamanhos: a capacidade é determinada pelas
constantes físicas (γ, β, σ), não pela geometria.
Se k variar com N: existe efeito de escala geométrico.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RN_psi_mnist import (
    psi_step,
    STIM_ON, STIM_TOTAL, DEVICE,
    CRYSTAL_W, CRYSTAL_K, CRYSTAL_A_MIN, CRYSTAL_CV_MAX,
    CRYSTAL_SEP, CRYSTAL_MAX, CRYSTAL_PATTERN, CRYSTAL_REMIT,
    PSI_C2, PSI_GAMMA, PSI_ALPHA, PSI_BETA, PSI_DT,
)

# ── Campo e cristais redimensionáveis ─────────────────────────────────────────

class ScalableCrystalMemory:
    """CrystalMemory parametrizado pelo tamanho do campo."""

    def __init__(self, B, field_size, dtype=torch.float32):
        self.B          = B
        self.FS         = field_size
        self.dtype      = dtype

        self.crystal_map = torch.zeros(B, field_size, field_size, device=DEVICE, dtype=dtype)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, field_size, field_size, device=DEVICE, dtype=dtype)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, field_size, field_size, device=DEVICE, dtype=dtype)

        ks = 2 * CRYSTAL_SEP + 1
        self._dilate = torch.ones(1, 1, ks, ks, device=DEVICE, dtype=dtype)

    def update_envelope(self, field):
        self.window_max = torch.max(self.window_max, field.abs())
        self.window_step += 1
        if self.window_step >= CRYSTAL_W:
            self.env_buffer[:, self.window_idx] = self.window_max
            self.window_max  = torch.zeros_like(self.window_max)
            self.window_step = 0
            self.window_idx  = (self.window_idx + 1) % CRYSTAL_K

    def try_crystallize(self, field):
        env  = self.env_buffer
        mean = env.mean(dim=1)
        std  = env.std(dim=1)
        cv   = std / (mean + 1e-8)

        candidates = ((mean > CRYSTAL_A_MIN) & (cv < CRYSTAL_CV_MAX) & (mean < 8.0)).float()

        occupied = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0, 1),
                  (CRYSTAL_SEP,) * 4, mode='circular'),
            self._dilate
        ).squeeze(1).clamp(0, 1)

        new_sites = candidates * (1.0 - occupied)
        scored    = new_sites * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + scored, 0, 10.)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        field = field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field)
        return torch.clamp(field, -10., 10.)


def psi_step_scaled(field, velocity, sources, active, field_size):
    """psi_step parametrizado pelo tamanho do campo."""
    from RN_psi_mnist import _LAP_KERNEL, _DT, _GAMMA, _ALPHA, _BETA, _C2

    lap_kernel = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                               device=DEVICE).view(1,1,3,3).to(field.dtype)

    if active:
        field = field + sources * (_DT * 0.1)

    inp    = field.unsqueeze(1)
    padded = F.pad(inp, (1,1,1,1), mode='circular')
    lap    = F.conv2d(padded, lap_kernel).squeeze(1)

    nonlinear = _ALPHA * torch.tanh(field) * field
    dissip    = _BETA  * field * field**2
    acc       = _C2 * lap - _GAMMA * velocity + nonlinear - dissip

    velocity  = velocity + acc * _DT
    field     = field    + velocity * _DT
    field     = torch.clamp(field,    -10., 10.)
    velocity  = torch.clamp(velocity,  -5.,  5.)
    return field, velocity


def make_perturbation(text, field_size):
    """Perturbação gaussiana adaptada ao tamanho do campo."""
    field   = torch.zeros(field_size, field_size, device=DEVICE)
    n_cells = 8
    sigma   = 0.06

    coords  = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg  = torch.meshgrid(coords, coords, indexing='ij')

    for i, c in enumerate(text):
        v   = ord(c) / 127.0
        cx  = ((ord(c) % n_cells) + 0.5) / n_cells
        cy  = (((ord(c) // n_cells) % n_cells) + 0.5) / n_cells
        amp = 1.5 + v * 2.5
        g   = amp * torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))
        field = field + g * float(np.cos(i * 0.3))
    return field.unsqueeze(0)


def run_saturation(field_size, sequence, sat_threshold=5):
    """
    Injeta sequência num campo de tamanho field_size.
    Retorna histórico de cristais e ponto de saturação.
    sat_threshold: delta mínimo de cristais para considerar ainda crescendo.
    """
    field    = torch.zeros(1, field_size, field_size, device=DEVICE)
    velocity = torch.zeros_like(field)
    memory   = ScalableCrystalMemory(1, field_size)

    history = []
    prev_n  = 0

    for i, word in enumerate(sequence):
        pert = make_perturbation(word, field_size)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                active = s < STIM_ON
                field, velocity = psi_step_scaled(field, velocity, pert, active, field_size)
                memory.update_envelope(field)
                if memory.window_idx > 0:
                    memory.try_crystallize(field)
                field = memory.remit(field)

        nc  = int((memory.crystal_map > 0.01).sum().item())
        occ = nc / (field_size * field_size) * 100
        history.append({'n': i+1, 'word': word, 'n_crystals': nc, 'ocupacao': occ})
        prev_n = nc

    # Ponto de saturação: última entrada onde delta > threshold
    sat_point = len(sequence)
    for i in range(1, len(history)):
        delta = history[i]['n_crystals'] - history[i-1]['n_crystals']
        if delta < sat_threshold and i > 2:
            sat_point = i
            break

    return history, sat_point


# ── Sequência de teste (50 palavras para garantir saturação) ──────────────────

sequence = [
    "amor", "casa", "vida", "sol", "lua", "mar", "rio", "flor", "vento", "fogo",
    "agua", "terra", "ar", "luz", "sombra", "tempo", "espaco", "mente", "corpo", "alma",
    "paz", "guerra", "bem", "mal", "verdade", "erro", "caminho", "porta", "janela", "chave",
    "pedra", "metal", "vidro", "madeira", "ferro", "ouro", "prata", "cobre", "bronze", "aco",
    "verde", "azul", "vermelho", "amarelo", "branco", "preto", "roxo", "laranja", "rosa", "cinza",
]

# Tamanhos de campo a testar
field_sizes = [24, 32, 48, 64, 96, 128]

print("="*60)
print("LEI DE ESCALA DO PSIFIELD")
print(f"Sequência: {len(sequence)} palavras")
print(f"Tamanhos: {field_sizes}")
print("="*60)

results = []

for fs in field_sizes:
    print(f"\nCampo {fs}×{fs} = {fs*fs} posições...")
    history, sat_pt = run_saturation(fs, sequence)

    sat_nc  = history[sat_pt-1]['n_crystals'] if sat_pt <= len(history) else history[-1]['n_crystals']
    sat_occ = history[sat_pt-1]['ocupacao']   if sat_pt <= len(history) else history[-1]['ocupacao']
    max_nc  = max(h['n_crystals'] for h in history)
    max_occ = max(h['ocupacao']   for h in history)

    k = sat_pt / (fs * fs)   # capacidade por posição

    results.append({
        'field_size': fs,
        'n2': fs * fs,
        'sat_point': sat_pt,
        'sat_crystals': sat_nc,
        'sat_ocupacao': sat_occ,
        'max_crystals': max_nc,
        'max_ocupacao': max_occ,
        'k': k,
        'history': history,
    })

    print(f"  Saturação: entrada #{sat_pt} | {sat_nc} cristais | {sat_occ:.1f}% ocupado")
    print(f"  Máximo: {max_nc} cristais | {max_occ:.1f}% ocupado")
    print(f"  k = {k:.6f} padrões/posição")

# ── Análise da lei de escala ──────────────────────────────────────────────────

print(f"\n{'='*60}")
print("RESULTADO: Lei de Escala")
print(f"{'='*60}")
print(f"\n{'Campo':>8}  {'N²':>8}  {'Capacidade':>11}  {'Ocup%':>7}  {'k':>10}  {'k×N²':>8}")
print("-"*60)

n2s = [r['n2'] for r in results]
caps = [r['sat_point'] for r in results]
ks  = [r['k'] for r in results]

for r in results:
    print(f"  {r['field_size']:>3}×{r['field_size']:<3}  {r['n2']:>8}  "
          f"{r['sat_point']:>11}  {r['sat_ocupacao']:>6.1f}%  "
          f"{r['k']:>10.6f}  {r['k']*r['n2']:>8.1f}")

# Fit linear: capacidade = k × N²
k_mean = np.mean(ks)
k_std  = np.std(ks)
print(f"\n  k médio: {k_mean:.6f} ± {k_std:.6f}")
print(f"  k constante? {'SIM' if k_std/k_mean < 0.2 else 'NÃO'} (CV={k_std/k_mean:.2f})")

# R² do fit linear
n2_arr  = np.array(n2s)
cap_arr = np.array(caps)
pred    = k_mean * n2_arr
ss_res  = np.sum((cap_arr - pred)**2)
ss_tot  = np.sum((cap_arr - cap_arr.mean())**2)
r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0
print(f"  R² do fit capacidade = k×N²: {r2:.4f}")

if r2 > 0.9:
    print(f"  -> Lei de escala LINEAR confirmada: capacidade ≈ {k_mean:.4f} × N²")
else:
    print(f"  -> Lei de escala NÃO é linear — outro regime")

# ── Visualizações ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Lei de Escala do PsiField — Capacidade vs Área do Campo', fontsize=12)

# 1. Capacidade vs N²
axes[0].scatter(n2s, caps, s=80, color='#e6194b', zorder=5)
n2_range = np.linspace(min(n2s)*0.9, max(n2s)*1.1, 100)
axes[0].plot(n2_range, k_mean * n2_range, '--', color='gray',
             label=f'k×N² (k={k_mean:.4f}, R²={r2:.3f})')
for r in results:
    axes[0].annotate(f"{r['field_size']}×{r['field_size']}",
                     (r['n2'], r['sat_point']),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[0].set_xlabel('N² (área do campo)')
axes[0].set_ylabel('Capacidade (padrões até saturação)')
axes[0].set_title('Capacidade vs Área')
axes[0].legend(); axes[0].grid(alpha=0.3)

# 2. k (constante) por tamanho
axes[1].bar([f"{r['field_size']}×{r['field_size']}" for r in results], ks,
             color='#3cb44b', alpha=0.8)
axes[1].axhline(k_mean, color='red', linestyle='--', label=f'k médio={k_mean:.4f}')
axes[1].set_xlabel('Tamanho do campo')
axes[1].set_ylabel('k = capacidade / N²')
axes[1].set_title('Constante k por Tamanho')
axes[1].legend(); axes[1].grid(alpha=0.3)

# 3. Curvas de crescimento por tamanho
colors = ['#e6194b', '#f58231', '#3cb44b', '#4363d8', '#911eb4', '#42d4f4']
for r, color in zip(results, colors):
    ns   = [h['n'] for h in r['history']]
    ncs  = [h['n_crystals'] for h in r['history']]
    norm = [n / r['n2'] * 100 for n in ncs]   # normalizado por N²
    axes[2].plot(ns, norm, 'o-', color=color, linewidth=1.5, markersize=3,
                 label=f"{r['field_size']}×{r['field_size']}")

axes[2].set_xlabel('N entradas injetadas')
axes[2].set_ylabel('Ocupação (% do campo)')
axes[2].set_title('Curvas de Saturação Normalizadas')
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_scaling_law.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_scaling_law.png")
print("Pronto.")
