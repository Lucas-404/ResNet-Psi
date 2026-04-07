"""
Experimento: Densidade de informação do PsiField em bits.

Pergunta: quantos bits de informação estão armazenados em cada cristal?

Metodologia:
  1. Gera pares de entradas com diferença controlada (0% a 100%)
  2. Para cada par mede se os cristais são distinguíveis (fingerprint distance)
  3. Acha o limiar mínimo de diferença que produz cristais distinguíveis
  4. Calcula bits por cristal = log2(estados distinguíveis)
  5. Multiplica pelo número de cristais = bits totais do campo

  bits por cristal = log2(1 / limiar_normalizado)
  bits totais      = n_cristais × bits_por_cristal

Repete para campos de tamanhos diferentes → curva de densidade por área.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

import torch.nn.functional as F

from RN_psi_mnist import (
    FIELD_SIZE, STIM_ON, STIM_TOTAL,
    DEVICE, CRYSTAL_PATTERN, CRYSTAL_SEP,
    CRYSTAL_K, CRYSTAL_W, CRYSTAL_A_MIN, CRYSTAL_CV_MAX, CRYSTAL_REMIT,
    _DT, _GAMMA, _ALPHA, _BETA, _C2,
)

PATCH = 2 * CRYSTAL_PATTERN + 1   # 11


# ── Campo e cristais escaláveis (sem import externo) ─────────────────────────

class ScalableCrystalMemory:
    def __init__(self, B, field_size, dtype=torch.float32):
        self.B  = B
        self.FS = field_size
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
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate
        ).squeeze(1).clamp(0,1)
        new_sites = candidates * (1.0 - occupied)
        self.crystal_map = torch.clamp(self.crystal_map + new_sites * field.abs(), 0, 10.)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        field = field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field)
        return torch.clamp(field, -10., 10.)


def psi_step_scaled(field, velocity, sources, active):
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

# ── Encoder e campo ───────────────────────────────────────────────────────────

def make_perturbation(text, field_size=FIELD_SIZE):
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
        g   = amp * torch.exp(-((xg-cx)**2 + (yg-cy)**2) / (2*sigma**2))
        field = field + g * float(np.cos(i * 0.3))
    return field.unsqueeze(0)


def run_field(text, field_size=FIELD_SIZE):
    pert     = make_perturbation(text, field_size)
    field    = pert.clone()
    velocity = torch.zeros_like(field)
    memory   = ScalableCrystalMemory(1, field_size)

    with torch.no_grad():
        for s in range(STIM_TOTAL):
            active = s < STIM_ON
            field, velocity = psi_step_scaled(field, velocity, pert, active)
            memory.update_envelope(field)
            if memory.window_idx > 0:
                memory.try_crystallize(field)
            field = memory.remit(field)

    return memory.crystal_map.squeeze(0).cpu().numpy()


def extract_crystal_fingerprints(cmap, thr=0.01):
    """Retorna lista de (posição, fingerprint) para cada cristal."""
    visited = np.zeros_like(cmap, dtype=bool)
    crystals = []
    ys, xs = np.where(cmap > thr)
    if len(ys) == 0:
        return crystals
    order = np.argsort(-cmap[ys, xs])
    ys, xs = ys[order], xs[order]

    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        y0 = max(0, y-CRYSTAL_SEP); y1 = min(cmap.shape[0], y+CRYSTAL_SEP+1)
        x0 = max(0, x-CRYSTAL_SEP); x1 = min(cmap.shape[1], x+CRYSTAL_SEP+1)
        visited[y0:y1, x0:x1] = True

        py0 = max(0, y-CRYSTAL_PATTERN); py1 = min(cmap.shape[0], y+CRYSTAL_PATTERN+1)
        px0 = max(0, x-CRYSTAL_PATTERN); px1 = min(cmap.shape[1], x+CRYSTAL_PATTERN+1)
        patch = cmap[py0:py1, px0:px1]

        pad_h = PATCH - patch.shape[0]
        pad_w = PATCH - patch.shape[1]
        patch = np.pad(patch, ((0,pad_h),(0,pad_w)))
        norm  = np.linalg.norm(patch)
        if norm < 1e-8:
            continue
        fp = patch.flatten() / norm
        crystals.append({'pos': (x,y), 'fp': fp, 'energia': float(cmap[y,x])})

    return crystals


def crystal_distance(crys1, crys2):
    """
    Distância média entre cristais em posições comuns.
    Retorna: distância ∈ [0,1] onde 0=idênticos, 1=completamente diferentes.
    """
    matched_dists = []
    for c1 in crys1:
        for c2 in crys2:
            d = np.sqrt((c1['pos'][0]-c2['pos'][0])**2 + (c1['pos'][1]-c2['pos'][1])**2)
            if d < CRYSTAL_SEP:
                sim = float(np.dot(c1['fp'], c2['fp']))
                sim = max(-1.0, min(1.0, sim))
                matched_dists.append(1.0 - sim)   # distância = 1 - similaridade
    if not matched_dists:
        return 1.0   # sem cristais comuns = máxima diferença
    return float(np.mean(matched_dists))


# ── Gerador de entradas com diferença controlada ──────────────────────────────

BASE_CHARS = "abcdefghijklmnopqrstuvwxyz"

def interpolate_texts(text_a, text_b, alpha):
    """
    Interpola entre dois textos: alpha=0 → text_a, alpha=1 → text_b.
    Substitui floor(alpha * len) caracteres de text_a por text_b.
    """
    n     = max(len(text_a), len(text_b))
    ta    = (text_a + text_a * n)[:n]
    tb    = (text_b + text_b * n)[:n]
    n_sub = int(round(alpha * n))
    chars = list(ta)
    for i in range(n_sub):
        chars[i] = tb[i]
    return ''.join(chars)


def generate_input_pairs(n_pairs=30, word_len=4, seed=42):
    """
    Gera pares de textos com diferença controlada de 0% a 100%.
    Retorna lista de (alpha, text_a, text_b) ordenada por alpha.
    """
    rng   = np.random.RandomState(seed)
    pairs = []
    alphas = np.linspace(0.0, 1.0, 11)   # 0%, 10%, 20%, ..., 100%

    for alpha in alphas:
        for _ in range(n_pairs // len(alphas)):
            ta = ''.join(rng.choice(list(BASE_CHARS), word_len))
            tb = ''.join(rng.choice(list(BASE_CHARS), word_len))
            ti = interpolate_texts(ta, tb, alpha)
            pairs.append((alpha, ta, ti))

    return pairs


# ── Experimento principal: limiar de distinguibilidade ───────────────────────

def measure_distinguishability(field_size=FIELD_SIZE, n_pairs=40, word_len=4):
    """
    Para um dado tamanho de campo:
    - Gera pares com diferença controlada alpha ∈ [0,1]
    - Mede distância de cristal para cada par
    - Acha o limiar alpha onde cristais passam a ser distinguíveis
    """
    pairs = generate_input_pairs(n_pairs, word_len)

    results = []
    for alpha, ta, tb in pairs:
        cmap_a = run_field(ta, field_size)
        cmap_b = run_field(tb, field_size)
        crys_a = extract_crystal_fingerprints(cmap_a)
        crys_b = extract_crystal_fingerprints(cmap_b)

        if not crys_a or not crys_b:
            continue

        dist = crystal_distance(crys_a, crys_b)
        results.append({'alpha': alpha, 'dist': dist, 'ta': ta, 'tb': tb})

    # Agrupa por alpha e calcula média
    alpha_vals = sorted(set(r['alpha'] for r in results))
    summary = []
    for a in alpha_vals:
        dists = [r['dist'] for r in results if r['alpha'] == a]
        summary.append({
            'alpha': a,
            'dist_mean': float(np.mean(dists)),
            'dist_std':  float(np.std(dists)),
            'n': len(dists),
        })

    # Limiar: menor alpha onde dist_mean > 0.05 (cristais distinguíveis)
    DIST_THRESHOLD = 0.05
    threshold_alpha = 1.0
    for s in summary:
        if s['dist_mean'] > DIST_THRESHOLD:
            threshold_alpha = s['alpha']
            break

    # Bits por cristal = log2(1 / threshold_alpha)
    if threshold_alpha > 0:
        bits_per_crystal = np.log2(1.0 / threshold_alpha)
    else:
        bits_per_crystal = np.log2(n_pairs)   # limite superior

    # Cristais típicos no campo
    sample_cmap = run_field("amor", field_size)
    n_crys = len(extract_crystal_fingerprints(sample_cmap))

    total_bits = bits_per_crystal * n_crys

    return {
        'field_size': field_size,
        'n2': field_size * field_size,
        'threshold_alpha': threshold_alpha,
        'bits_per_crystal': bits_per_crystal,
        'n_crystals': n_crys,
        'total_bits': total_bits,
        'bits_per_position': total_bits / (field_size * field_size),
        'summary': summary,
    }


# ── Roda para múltiplos tamanhos ──────────────────────────────────────────────

field_sizes = [24, 48, 96, 128]

print("="*60)
print("DENSIDADE DE INFORMAÇÃO DO PSIFIELD (bits)")
print("="*60)

all_results = []

for fs in field_sizes:
    print(f"\nCampo {fs}×{fs}...")
    r = measure_distinguishability(fs, n_pairs=44, word_len=4)
    all_results.append(r)

    print(f"  Limiar de distinguibilidade: {r['threshold_alpha']*100:.0f}% diferença")
    print(f"  Bits por cristal           : {r['bits_per_crystal']:.2f} bits")
    print(f"  Cristais típicos           : {r['n_crystals']}")
    print(f"  Bits totais no campo       : {r['total_bits']:.1f} bits")
    print(f"  Bits por posição (N²)      : {r['bits_per_position']:.4f} bits/px")

    # Curva de distinguibilidade
    print(f"\n  alpha  dist_media  distinguível?")
    for s in r['summary']:
        flag = "SIM" if s['dist_mean'] > 0.05 else "   "
        print(f"  {s['alpha']:.1f}    {s['dist_mean']:.4f}±{s['dist_std']:.4f}   {flag}")

# ── Resumo comparativo ────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("RESUMO: Densidade de Informação por Tamanho de Campo")
print(f"{'='*60}")
print(f"\n{'Campo':>8}  {'Limiar':>8}  {'Bits/cristal':>13}  {'Cristais':>9}  {'Bits total':>11}  {'Bits/px':>9}")
print("-"*65)
for r in all_results:
    print(f"  {r['field_size']:>3}×{r['field_size']:<3}  "
          f"{r['threshold_alpha']*100:>7.0f}%  "
          f"{r['bits_per_crystal']:>13.2f}  "
          f"{r['n_crystals']:>9}  "
          f"{r['total_bits']:>11.1f}  "
          f"{r['bits_per_position']:>9.4f}")

# ── Visualizações ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Densidade de Informação do PsiField', fontsize=13)

colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']

# 1. Curvas de distinguibilidade por tamanho
for r, color in zip(all_results, colors):
    alphas = [s['alpha'] for s in r['summary']]
    dists  = [s['dist_mean'] for s in r['summary']]
    stds   = [s['dist_std'] for s in r['summary']]
    axes[0].plot(alphas, dists, 'o-', color=color,
                 label=f"{r['field_size']}×{r['field_size']}", linewidth=2)
    axes[0].fill_between(alphas,
                          [d-s for d,s in zip(dists,stds)],
                          [d+s for d,s in zip(dists,stds)],
                          alpha=0.15, color=color)

axes[0].axhline(0.05, color='gray', linestyle='--', label='Limiar distinguível')
axes[0].set_xlabel('Diferença entre entradas (α)')
axes[0].set_ylabel('Distância de cristal')
axes[0].set_title('Curva de Distinguibilidade')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# 2. Bits por cristal por tamanho
fss   = [r['field_size'] for r in all_results]
bpcs  = [r['bits_per_crystal'] for r in all_results]
axes[1].bar([f"{fs}×{fs}" for fs in fss], bpcs, color=colors[:len(fss)], alpha=0.8)
for i, (fs, bpc) in enumerate(zip(fss, bpcs)):
    axes[1].text(i, bpc + 0.05, f"{bpc:.2f}", ha='center', va='bottom', fontsize=9)
axes[1].set_xlabel('Tamanho do campo')
axes[1].set_ylabel('Bits por cristal')
axes[1].set_title('Bits por Cristal vs Tamanho')
axes[1].grid(alpha=0.3)

# 3. Bits totais vs N²
n2s       = [r['n2'] for r in all_results]
tot_bits  = [r['total_bits'] for r in all_results]
bpp       = [r['bits_per_position'] for r in all_results]

ax3 = axes[2]
ax3b = ax3.twinx()
ax3.bar([f"{r['field_size']}×{r['field_size']}" for r in all_results],
        tot_bits, color=colors[:len(all_results)], alpha=0.7, label='Bits totais')
ax3b.plot([f"{r['field_size']}×{r['field_size']}" for r in all_results],
          bpp, 'ko-', linewidth=2, label='Bits/posição')
ax3.set_xlabel('Tamanho do campo')
ax3.set_ylabel('Bits totais no campo')
ax3b.set_ylabel('Bits por posição (N²)')
ax3.set_title('Capacidade Total e Densidade')
ax3.legend(loc='upper left', fontsize=8)
ax3b.legend(loc='upper right', fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_bits_density.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_bits_density.png")
print("Pronto.")
