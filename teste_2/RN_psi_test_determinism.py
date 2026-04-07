"""
Teste de determinismo e discriminação do PsiField.

Injeta ondas diretamente com parâmetros controlados — sem dataset.

Pergunta 1: mesmos parâmetros → mesmo crystal_map? (determinismo)
Pergunta 2: parâmetros diferentes → crystal_maps diferentes? (discriminação)
Pergunta 3: quantos cristais emergem e onde?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from RN_psi_mnist import (
    run_psi_field, CrystalMemory,
    emit_waves, psi_step,
    FIELD_SIZE, STIM_ON, STIM_TOTAL, PSI_DT,
    N_WAVES, DEVICE
)

# ── Define dois padrões de onda manualmente ────────────────────────────────
# Padrão A: ondas centradas na metade esquerda, frequência baixa
# Padrão B: ondas centradas na metade direita, frequência alta
# Padrão A': idêntico ao A — para testar determinismo

def make_wave_params(freq_base, pos_x_base, pos_y_base, B=1):
    """Cria parâmetros de onda controlados."""
    wp = torch.zeros(B, N_WAVES, 6, device=DEVICE)
    for w in range(N_WAVES):
        wp[:, w, 0] = 3.0                                      # amp
        wp[:, w, 1] = freq_base + w * 0.3                     # freq
        wp[:, w, 2] = w * 0.5                                  # phase
        wp[:, w, 3] = 0.001                                    # decay
        wp[:, w, 4] = pos_x_base + (w % 4) * 0.1             # pos_x
        wp[:, w, 5] = pos_y_base + (w // 4) * 0.1            # pos_y
    return wp

wp_A  = make_wave_params(freq_base=2.0, pos_x_base=0.2, pos_y_base=0.2)  # padrão A
wp_Ap = make_wave_params(freq_base=2.0, pos_x_base=0.2, pos_y_base=0.2)  # padrão A' (idêntico)
wp_B  = make_wave_params(freq_base=5.0, pos_x_base=0.6, pos_y_base=0.6)  # padrão B (diferente)

print("Rodando campo para A, A' e B...")
cmap_A  = run_psi_field(wp_A ).view(FIELD_SIZE, FIELD_SIZE).cpu().numpy()
cmap_Ap = run_psi_field(wp_Ap).view(FIELD_SIZE, FIELD_SIZE).cpu().numpy()
cmap_B  = run_psi_field(wp_B ).view(FIELD_SIZE, FIELD_SIZE).cpu().numpy()

# ── Métricas ───────────────────────────────────────────────────────────────
def similarity(a, b):
    a_flat = a.flatten(); b_flat = b.flatten()
    if a_flat.std() < 1e-8 or b_flat.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])

def analyze(cmap, label):
    mask      = cmap > 0.01
    n         = int(mask.sum())
    energia   = float(cmap[mask].sum()) if n > 0 else 0.0
    densidade = n / (FIELD_SIZE * FIELD_SIZE) * 100
    if n > 0:
        xs, ys   = np.where(mask)
        cx, cy   = xs.mean(), ys.mean()
        spread_x, spread_y = xs.std(), ys.std()
    else:
        cx = cy = spread_x = spread_y = 0.0

    print(f"\n  [{label}]")
    print(f"    Cristais   : {n}")
    print(f"    Densidade  : {densidade:.2f}% do campo")
    print(f"    Energia    : {energia:.4f} total | {energia/n:.4f} por cristal" if n > 0 else "    Energia: 0")
    print(f"    Centro     : ({cx:.1f}, {cy:.1f})")
    print(f"    Dispersão  : x={spread_x:.1f} y={spread_y:.1f}")
    return mask, n

print(f"\n{'='*55}")
print("ANÁLISE DE CRISTAIS:")
mA,  nA  = analyze(cmap_A,  "Padrão A  (freq=2.0, pos=esquerda)")
mAp, nAp = analyze(cmap_Ap, "Padrão A' (idêntico ao A)")
mB,  nB  = analyze(cmap_B,  "Padrão B  (freq=5.0, pos=direita)")

def overlap(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter / union) if union > 0 else 0.0

print(f"\n{'='*55}")
print("SIMILARIDADE:")
print(f"  A  vs A' [idênticos]  : {similarity(cmap_A, cmap_Ap):.6f}  (esperado: ~1.0)")
print(f"  A  vs B  [diferentes] : {similarity(cmap_A, cmap_B):.6f}   (esperado: ~0.0)")
print(f"\nSOBREPOSIÇÃO DE CRISTAIS:")
print(f"  A  vs A' [idênticos]  : {overlap(mA, mAp)*100:.1f}%  (esperado: ~100%)")
print(f"  A  vs B  [diferentes] : {overlap(mA, mB)*100:.1f}%   (esperado: ~0%)")
print(f"{'='*55}")

# ── Visualização ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Teste de Determinismo e Discriminação do PsiField', fontsize=13)

cmaps_all  = [cmap_A, cmap_Ap, cmap_B]
titles_all = [
    f"Padrão A\nfreq=2.0 pos=esquerda\n{nA} cristais",
    f"Padrão A' (idêntico)\nfreq=2.0 pos=esquerda\n{nAp} cristais",
    f"Padrão B\nfreq=5.0 pos=direita\n{nB} cristais",
]

# Linha 1: crystal_maps
vmax = max(c.max() for c in cmaps_all) + 1e-6
for i, (cmap, title) in enumerate(zip(cmaps_all, titles_all)):
    axes[0][i].imshow(cmap, cmap='inferno', vmin=0, vmax=vmax, interpolation='nearest')
    axes[0][i].set_title(title, fontsize=9)
    axes[0][i].axis('off')

# Linha 2: sobreposição
# A vs A'
ov_AA = np.zeros((FIELD_SIZE, FIELD_SIZE, 3))
ov_AA[mA & ~mAp, 0] = 0.9
ov_AA[~mA & mAp, 2] = 0.9
ov_AA[mA & mAp,  1] = 0.9
axes[1][0].imshow(ov_AA, interpolation='nearest')
axes[1][0].set_title(f'A vs A\' — sobreposição\nverde=ambos | sobrep={overlap(mA,mAp)*100:.1f}%', fontsize=9)
axes[1][0].axis('off')

# A vs B
ov_AB = np.zeros((FIELD_SIZE, FIELD_SIZE, 3))
ov_AB[mA & ~mB, 0] = 0.9
ov_AB[~mA & mB, 2] = 0.9
ov_AB[mA & mB,  1] = 0.9
axes[1][1].imshow(ov_AB, interpolation='nearest')
axes[1][1].set_title(f'A vs B — sobreposição\nverde=ambos | sobrep={overlap(mA,mB)*100:.1f}%', fontsize=9)
axes[1][1].axis('off')

# Densidade de energia comparada
axes[1][2].bar(['A', "A'", 'B'],
               [cmap_A.sum(), cmap_Ap.sum(), cmap_B.sum()],
               color=['#e6194b', '#3cb44b', '#4363d8'])
axes[1][2].set_title('Energia total por padrão', fontsize=9)
axes[1][2].set_ylabel('Energia (soma do crystal_map)')
axes[1][2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_determinism_test.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n-> viz_determinism_test.png")
