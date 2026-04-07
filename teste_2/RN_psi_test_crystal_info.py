"""
Experimento 1: O que cada cristal individual armazena?

Pergunta: um cristal é só posição + energia, ou tem estrutura interna?

Metodologia:
  - Injeta dois padrões com cristais na mesma posição (anagramas)
  - Compara energia, gradiente local, e padrão 11x11 de cada cristal
  - Injeta padrões progressivamente diferentes e mede quando o cristal muda
  - Verifica se cristais em posições iguais têm conteúdo diferente

Métricas por cristal:
  - posição (x, y)
  - energia total (soma do patch 11x11)
  - gradiente médio (variação espacial interna)
  - entropia local (distribuição de energia no patch)
  - fingerprint: vetor 121D do patch normalizado
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

from RN_psi_mnist import (
    psi_step, CrystalMemory,
    FIELD_SIZE, STIM_ON, STIM_TOTAL,
    DEVICE, CRYSTAL_PATTERN, CRYSTAL_SEP,
)

PATCH = 2 * CRYSTAL_PATTERN + 1   # 11

# ── Encoder físico ────────────────────────────────────────────────────────────

def char_to_gaussian(c):
    v       = ord(c) / 127.0
    n_cells = 8
    cx      = ((ord(c) % n_cells) + 0.5) / n_cells
    cy      = (((ord(c) // n_cells) % n_cells) + 0.5) / n_cells
    amp     = 1.5 + v * 2.5
    sigma   = 0.06
    coords  = torch.linspace(0., 1., FIELD_SIZE, device=DEVICE)
    xg, yg  = torch.meshgrid(coords, coords, indexing='ij')
    return amp * torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))

def text_to_perturbation(text):
    field = torch.zeros(FIELD_SIZE, FIELD_SIZE, device=DEVICE)
    for i, c in enumerate(text):
        field = field + char_to_gaussian(c) * float(np.cos(i * 0.3))
    return field.unsqueeze(0)

def run_field(text):
    pert     = text_to_perturbation(text)
    field    = pert.clone()
    velocity = torch.zeros_like(field)
    memory   = CrystalMemory(1)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            active = s < STIM_ON
            field, velocity = psi_step(field, velocity, pert, active)
            memory.update_envelope(field)
            if memory.window_idx > 0:
                memory.try_crystallize(field)
            field = memory.remit(field, None)
    return memory.crystal_map.squeeze(0).cpu().numpy(), field.squeeze(0).cpu().numpy()


# ── Extração de cristais individuais ─────────────────────────────────────────

def extract_crystals(cmap, field_snap, thr=0.01):
    """
    Encontra cristais no crystal_map e extrai propriedades de cada um.
    Retorna lista de dicts com posição, energia, patch, fingerprint.
    """
    mask     = cmap > thr
    labeled  = []
    visited  = np.zeros_like(mask, dtype=bool)

    ys, xs = np.where(mask)
    # Ordena por energia decrescente
    energies = cmap[ys, xs]
    order    = np.argsort(-energies)
    ys, xs   = ys[order], xs[order]

    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        # Marca região ao redor como visitada (exclusão espacial)
        y0 = max(0, y - CRYSTAL_SEP)
        y1 = min(FIELD_SIZE, y + CRYSTAL_SEP + 1)
        x0 = max(0, x - CRYSTAL_SEP)
        x1 = min(FIELD_SIZE, x + CRYSTAL_SEP + 1)
        visited[y0:y1, x0:x1] = True

        # Extrai patch 11×11 do crystal_map
        py0 = max(0, y - CRYSTAL_PATTERN)
        py1 = min(FIELD_SIZE, y + CRYSTAL_PATTERN + 1)
        px0 = max(0, x - CRYSTAL_PATTERN)
        px1 = min(FIELD_SIZE, x + CRYSTAL_PATTERN + 1)
        patch = cmap[py0:py1, px0:px1]

        # Patch do campo instantâneo (estado da onda no momento da cristalização)
        fpatch = field_snap[py0:py1, px0:px1]

        # Métricas internas do cristal
        eng    = float(patch.sum())
        if eng < 1e-8:
            continue

        # Gradiente interno (variação espacial)
        grad_x = float(np.abs(np.diff(patch, axis=1)).mean()) if patch.shape[1] > 1 else 0
        grad_y = float(np.abs(np.diff(patch, axis=0)).mean()) if patch.shape[0] > 1 else 0
        grad   = (grad_x + grad_y) / 2

        # Entropia local (distribuição de energia)
        flat = patch.flatten()
        flat = flat / (flat.sum() + 1e-8)
        flat = np.clip(flat, 1e-10, 1)
        ent  = float(scipy_entropy(flat))

        # Fingerprint: patch normalizado como vetor
        pad_h = PATCH - patch.shape[0]
        pad_w = PATCH - patch.shape[1]
        patch_padded = np.pad(patch, ((0, pad_h), (0, pad_w)))
        fp = patch_padded.flatten()
        fp = fp / (np.linalg.norm(fp) + 1e-8)

        labeled.append({
            'x': int(x), 'y': int(y),
            'energia': eng,
            'gradiente': grad,
            'entropia': ent,
            'fingerprint': fp,
            'patch': patch_padded,
        })

    return labeled


def fingerprint_similarity(fp1, fp2):
    return float(np.dot(fp1, fp2))   # cosseno (já normalizados)


# ── Experimento A: cristais de anagramas na mesma posição ────────────────────

print("="*60)
print("EXPERIMENTO A: Cristais de anagramas têm conteúdo diferente?")
print("="*60)

pairs = [
    ("amor", "armo"),   # 99.2% IoU
    ("amor", "mora"),   # 97.8% IoU
    ("amor", "roma"),   # 96.7% IoU
]

for w1, w2 in pairs:
    cmap1, f1 = run_field(w1)
    cmap2, f2 = run_field(w2)

    crys1 = extract_crystals(cmap1, f1)
    crys2 = extract_crystals(cmap2, f2)

    # Encontra cristais em posições próximas (mesma região)
    matched = []
    for c1 in crys1:
        for c2 in crys2:
            dist = np.sqrt((c1['x']-c2['x'])**2 + (c1['y']-c2['y'])**2)
            if dist < CRYSTAL_SEP:
                sim = fingerprint_similarity(c1['fingerprint'], c2['fingerprint'])
                matched.append({
                    'pos': (c1['x'], c1['y']),
                    'eng1': c1['energia'], 'eng2': c2['energia'],
                    'grad1': c1['gradiente'], 'grad2': c2['gradiente'],
                    'ent1': c1['entropia'], 'ent2': c2['entropia'],
                    'fp_sim': sim, 'dist': dist,
                })

    if not matched:
        print(f"\n  {w1} vs {w2}: nenhum cristal em posição comum")
        continue

    fp_sims  = [m['fp_sim']  for m in matched]
    eng_diff = [abs(m['eng1']-m['eng2'])/(m['eng1']+1e-8) for m in matched]

    print(f"\n  {w1} vs {w2}: {len(matched)} cristais em posição comum")
    print(f"    Similaridade de fingerprint: {np.mean(fp_sims):.4f} ± {np.std(fp_sims):.4f}")
    print(f"    Diferença de energia: {np.mean(eng_diff)*100:.2f}% ± {np.std(eng_diff)*100:.2f}%")
    print(f"    (1.0 = idênticos, 0.0 = completamente diferentes)")

    # Detalhes dos 3 primeiros
    for m in matched[:3]:
        print(f"    Pos ({m['pos'][0]:2d},{m['pos'][1]:2d}): "
              f"eng={m['eng1']:.2f}→{m['eng2']:.2f}  "
              f"fp_sim={m['fp_sim']:.4f}")


# ── Experimento B: gradiente de diferença ────────────────────────────────────

print(f"\n{'='*60}")
print("EXPERIMENTO B: Quando cristais começam a diferir?")
print("Palavras progressivamente mais diferentes de 'amor'")
print("="*60)

# Sequência de palavras com distância crescente de "amor"
# distância = número de caracteres diferentes
targets = [
    ("amor", "amor"),   # idêntico
    ("amor", "amop"),   # 1 char diferente (r→p)
    ("amor", "amaz"),   # 2 chars diferentes
    ("amor", "abcd"),   # 3 chars diferentes
    ("amor", "xyzt"),   # 4 chars completamente diferentes
]

cmap_ref, f_ref = run_field("amor")
crys_ref = extract_crystals(cmap_ref, f_ref)

print(f"\n  Referência 'amor': {len(crys_ref)} cristais")
print(f"\n  {'Par':20s}  {'Cristais comuns':>15}  {'FP sim médio':>13}  {'Eng diff%':>10}")
print("  " + "-"*65)

for w1, w2 in targets:
    cmap2, f2 = run_field(w2)
    crys2 = extract_crystals(cmap2, f2)

    matched = []
    for c1 in crys_ref:
        for c2 in crys2:
            dist = np.sqrt((c1['x']-c2['x'])**2 + (c1['y']-c2['y'])**2)
            if dist < CRYSTAL_SEP:
                sim = fingerprint_similarity(c1['fingerprint'], c2['fingerprint'])
                matched.append({
                    'fp_sim': sim,
                    'eng_diff': abs(c1['energia']-c2['energia'])/(c1['energia']+1e-8)
                })

    if not matched:
        print(f"  {'amor vs '+w2:20s}  {'0':>15}  {'—':>13}  {'—':>10}")
        continue

    fp_mean  = np.mean([m['fp_sim'] for m in matched])
    eng_mean = np.mean([m['eng_diff'] for m in matched]) * 100
    n_ref    = len(crys_ref)
    print(f"  {'amor vs '+w2:20s}  {len(matched):>6}/{n_ref:<8}  {fp_mean:>13.4f}  {eng_mean:>9.2f}%")


# ── Experimento C: entropia interna dos cristais ──────────────────────────────

print(f"\n{'='*60}")
print("EXPERIMENTO C: Estrutura interna — cristais têm informação?")
print("Comparando entropia e gradiente interno de cristais reais vs ruído")
print("="*60)

words_test = ["amor", "casa", "vida", "sol", "lua", "mar"]
all_entropies = []
all_gradients = []

for word in words_test:
    cmap, f = run_field(word)
    crys = extract_crystals(cmap, f)
    ents = [c['entropia'] for c in crys]
    grads = [c['gradiente'] for c in crys]
    all_entropies.extend(ents)
    all_gradients.extend(grads)
    print(f"  {word:8s}: {len(crys):3d} cristais  "
          f"entropia={np.mean(ents):.3f}±{np.std(ents):.3f}  "
          f"gradiente={np.mean(grads):.4f}±{np.std(grads):.4f}")

# Ruído aleatório como baseline
noise_map = np.random.rand(FIELD_SIZE, FIELD_SIZE) * 0.5
noise_field = np.random.randn(FIELD_SIZE, FIELD_SIZE) * 0.1
crys_noise = extract_crystals(noise_map, noise_field)
if crys_noise:
    ents_n  = [c['entropia'] for c in crys_noise]
    grads_n = [c['gradiente'] for c in crys_noise]
    print(f"  {'ruído':8s}: {len(crys_noise):3d} cristais  "
          f"entropia={np.mean(ents_n):.3f}±{np.std(ents_n):.3f}  "
          f"gradiente={np.mean(grads_n):.4f}±{np.std(grads_n):.4f}")

print(f"\n  Cristais reais vs ruído:")
print(f"    Entropia real:  {np.mean(all_entropies):.4f}")
print(f"    Gradiente real: {np.mean(all_gradients):.6f}")
if crys_noise:
    print(f"    Entropia ruído: {np.mean(ents_n):.4f}")
    print(f"    Gradiente ruído:{np.mean(grads_n):.6f}")


# ── Visualização ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Estrutura Interna dos Cristais — O que cada cristal armazena?', fontsize=12)

# Patches dos 4 primeiros cristais de "amor"
cmap_amor, f_amor = run_field("amor")
crys_amor = extract_crystals(cmap_amor, f_amor)

for i, (ax, c) in enumerate(zip(axes[0], crys_amor[:4])):
    ax.imshow(c['patch'], cmap='inferno', interpolation='nearest')
    ax.set_title(f"Cristal #{i+1} de 'amor'\npos=({c['x']},{c['y']})\n"
                 f"eng={c['energia']:.2f} ent={c['entropia']:.3f}", fontsize=8)
    ax.axis('off')

# Patches dos 4 primeiros cristais de "armo" nas mesmas posições
cmap_armo, f_armo = run_field("armo")
crys_armo = extract_crystals(cmap_armo, f_armo)

# Encontra os cristais de armo que correspondem aos de amor
matched_armo = []
for c1 in crys_amor[:4]:
    best = None
    best_dist = 999
    for c2 in crys_armo:
        dist = np.sqrt((c1['x']-c2['x'])**2 + (c1['y']-c2['y'])**2)
        if dist < CRYSTAL_SEP and dist < best_dist:
            best_dist = dist
            best = c2
    matched_armo.append(best)

for i, (ax, c_amor, c_armo) in enumerate(zip(axes[1], crys_amor[:4], matched_armo)):
    if c_armo is not None:
        fp_sim = fingerprint_similarity(c_amor['fingerprint'], c_armo['fingerprint'])
        ax.imshow(c_armo['patch'], cmap='inferno', interpolation='nearest')
        ax.set_title(f"Cristal #{i+1} de 'armo'\npos=({c_armo['x']},{c_armo['y']})\n"
                     f"fp_sim={fp_sim:.4f}", fontsize=8)
    else:
        ax.text(0.5, 0.5, 'sem correspondência', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Cristal #{i+1} — sem par", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('viz_crystal_internal.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_crystal_internal.png")
print("Pronto.")
