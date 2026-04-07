"""
Auditoria 29: Attention Física

Testa duas ideias pra refinar o campo SEM treino, SEM matrizes:

1. c² variável — velocidade de onda depende da intensidade da entrada.
   Onde o pixel é forte → onda propaga mais rápido → mais interferência.
   Onde o pixel é fraco → onda devagar → menos energia.
   Isso é REFRAÇÃO. Luz muda de velocidade no vidro. Mesma física.

2. Duas passadas (self-attention física) —
   Passada 1: propaga normal, gera mapa de energia.
   Passada 2: usa mapa de energia pra mudar c² e γ.
   O campo "presta atenção" em si mesmo. Regiões que vibraram
   forte na primeira vez ganham mais peso na segunda.

Sem matrizes. Sem treino. Só física adaptativa.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'C:/ResNet-Psi')
from resnet_psi import (
    DEVICE, FIELD_SIZE, PSI_DT, PSI_GAMMA, PSI_ALPHA, PSI_BETA, PSI_C2,
    STIM_ON, STIM_TOTAL, CRYSTAL_W, CRYSTAL_K, CRYSTAL_A_MIN, CRYSTAL_CV_MAX,
    CRYSTAL_SEP, CRYSTAL_REMIT,
    CrystalCompetitivo, build_gaussians, ResNetPsi
)

print(f"Dispositivo: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# EQUAÇÃO DE ONDA COM c² VARIÁVEL
# ══════════════════════════════════════════════════════════════════════════════

def psi_step_adaptive(field, velocity, sources, active, c2_map, gamma_map):
    """
    Equação de onda com c² e γ variáveis por região.

    c2_map: (B, FS, FS) — velocidade de onda local
    gamma_map: (B, FS, FS) — amortecimento local
    """
    dt = PSI_DT
    alpha = PSI_ALPHA
    beta = PSI_BETA

    if active:
        field = field + sources * (dt * 0.1)

    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                          device=DEVICE).view(1, 1, 3, 3).to(field.dtype)
    lap = F.conv2d(F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular'),
                   lap_k).squeeze(1)

    # c² variável: cada região tem sua velocidade
    acc = c2_map * lap - gamma_map * velocity + alpha * torch.tanh(field) * field - beta * field * field**2
    velocity = torch.clamp(velocity + acc * dt, -5., 5.)
    field    = torch.clamp(field + velocity * dt, -10., 10.)
    return field, velocity


# ══════════════════════════════════════════════════════════════════════════════
# MODO 1: c² VARIÁVEL (baseado na entrada)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cmaps_c2_variavel(X, PG, field_size=FIELD_SIZE, bs=64, c2_boost=0.5):
    """
    c² varia com a perturbação: onde o sinal é forte, onda propaga mais rápido.

    c2(x,y) = PSI_C2 + c2_boost * |perturbação(x,y)| / max(|perturbação|)
    """
    N = len(X)
    n_pixels = PG.shape[0]
    out = []

    for i in range(0, N, bs):
        batch = X[i:i+bs]
        B = len(batch)
        pert = (batch.view(B, n_pixels) @ PG.to(batch.dtype)).view(B, field_size, field_size)

        # c² variável: normaliza perturbação pra [0, 1], escala com boost
        pert_norm = pert.abs() / (pert.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
        c2_map = PSI_C2 + c2_boost * pert_norm
        gamma_map = torch.full_like(pert, PSI_GAMMA)

        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(B, field_size)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step_adaptive(f, v, pert, s < STIM_ON, c2_map, gamma_map)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)

        out.append(mem.crystal_map)

    return torch.cat(out, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# MODO 2: DUAS PASSADAS (self-attention física)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cmaps_duas_passadas(X, PG, field_size=FIELD_SIZE, bs=64,
                                 c2_boost=0.3, gamma_reduce=0.5):
    """
    Passada 1: propaga normal, gera mapa de energia.
    Passada 2: usa mapa de energia pra modular c² e γ.

    Onde vibrou forte na passada 1:
      - c² aumenta (propaga mais rápido)
      - γ diminui (amortece menos)
    Onde não vibrou:
      - c² fica baixo
      - γ fica alto (mata rápido)
    """
    N = len(X)
    n_pixels = PG.shape[0]
    out = []

    for i in range(0, N, bs):
        batch = X[i:i+bs]
        B = len(batch)
        pert = (batch.view(B, n_pixels) @ PG.to(batch.dtype)).view(B, field_size, field_size)

        # ── PASSADA 1: propaga normal, registra energia ──
        f1, v1 = pert.clone(), torch.zeros_like(pert)
        energy_map = torch.zeros_like(pert)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f1, v1 = psi_step_adaptive(
                    f1, v1, pert, s < STIM_ON,
                    torch.full_like(pert, PSI_C2),
                    torch.full_like(pert, PSI_GAMMA))
                energy_map = energy_map + f1.abs()

        # Normaliza mapa de energia pra [0, 1]
        energy_norm = energy_map / (energy_map.amax(dim=(1, 2), keepdim=True) + 1e-8)

        # ── PASSADA 2: física adaptada pelo mapa de energia ──
        c2_map = PSI_C2 + c2_boost * energy_norm
        gamma_map = PSI_GAMMA * (1.0 - gamma_reduce * energy_norm)
        gamma_map = torch.clamp(gamma_map, 0.01, 0.2)

        f2, v2 = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(B, field_size)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f2, v2 = psi_step_adaptive(f2, v2, pert, s < STIM_ON, c2_map, gamma_map)
                mem.update_envelope(f2)
                if mem.window_idx > 0:
                    mem.try_crystallize(f2)
                f2 = mem.remit(f2)

        out.append(mem.crystal_map)

    return torch.cat(out, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# MODO 3: c² VARIÁVEL + DUAS PASSADAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cmaps_combo(X, PG, field_size=FIELD_SIZE, bs=64,
                         c2_boost_1=0.3, c2_boost_2=0.5, gamma_reduce=0.5):
    """
    Passada 1: c² variável pela entrada (refração).
    Passada 2: c² variável pela energia da passada 1 (self-attention).
    """
    N = len(X)
    n_pixels = PG.shape[0]
    out = []

    for i in range(0, N, bs):
        batch = X[i:i+bs]
        B = len(batch)
        pert = (batch.view(B, n_pixels) @ PG.to(batch.dtype)).view(B, field_size, field_size)

        # Refração pela entrada
        pert_norm = pert.abs() / (pert.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
        c2_map_1 = PSI_C2 + c2_boost_1 * pert_norm

        # ── PASSADA 1: com refração ──
        f1, v1 = pert.clone(), torch.zeros_like(pert)
        energy_map = torch.zeros_like(pert)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f1, v1 = psi_step_adaptive(
                    f1, v1, pert, s < STIM_ON,
                    c2_map_1, torch.full_like(pert, PSI_GAMMA))
                energy_map = energy_map + f1.abs()

        energy_norm = energy_map / (energy_map.amax(dim=(1, 2), keepdim=True) + 1e-8)

        # ── PASSADA 2: refração + attention ──
        c2_map_2 = PSI_C2 + c2_boost_2 * energy_norm
        gamma_map = PSI_GAMMA * (1.0 - gamma_reduce * energy_norm)
        gamma_map = torch.clamp(gamma_map, 0.01, 0.2)

        f2, v2 = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(B, field_size)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f2, v2 = psi_step_adaptive(f2, v2, pert, s < STIM_ON, c2_map_2, gamma_map)
                mem.update_envelope(f2)
                if mem.window_idx > 0:
                    mem.try_crystallize(f2)
                f2 = mem.remit(f2)

        out.append(mem.crystal_map)

    return torch.cat(out, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# TESTAR
# ══════════════════════════════════════════════════════════════════════════════

from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

N_TRAIN = 500
N_TEST  = 1000

# Pegar 50 por classe (balanceado)
por_classe = {c: [] for c in range(10)}
for i in range(len(train_ds)):
    img, lab = train_ds[i]
    if len(por_classe[lab]) < 50:
        por_classe[lab].append(img.squeeze(0))
    if all(len(v) >= 50 for v in por_classe.values()):
        break

train_imgs = torch.stack([img for c in range(10) for img in por_classe[c]]).to(DEVICE)
train_labs = np.array([c for c in range(10) for _ in por_classe[c]])

test_imgs = torch.stack([test_ds[i][0].squeeze(0) for i in range(N_TEST)]).to(DEVICE)
test_labs = np.array([test_ds[i][1] for i in range(N_TEST)])

PG = build_gaussians((28, 28))


def testar_modo(nome, cmaps_train, cmaps_test):
    """Calcula acurácia por protótipos."""
    from resnet_psi import build_prototypes, classify_euclidean
    protos = build_prototypes(cmaps_train, train_labs, 10)
    preds = classify_euclidean(cmaps_test, protos)
    acc = (preds == test_labs).mean() * 100
    return acc


# ── Baseline: original ──
print("\n" + "="*60)
print("BASELINE: ResNet-Ψ original (c² fixo, 1 passada)")
print("="*60)
t0 = time.time()
from resnet_psi import compute_crystal_maps
cmaps_train_orig = compute_crystal_maps(train_imgs, PG, bs=32)
cmaps_test_orig  = compute_crystal_maps(test_imgs, PG, bs=64)
acc_orig = testar_modo("Original", cmaps_train_orig, cmaps_test_orig)
print(f"  Original: {acc_orig:.1f}%  ({time.time()-t0:.0f}s)")

# ── Modo 1: c² variável ──
print("\n" + "="*60)
print("MODO 1: c² variável (refração pela entrada)")
print("="*60)
t0 = time.time()
cmaps_train_c2 = compute_cmaps_c2_variavel(train_imgs, PG, bs=32)
cmaps_test_c2  = compute_cmaps_c2_variavel(test_imgs, PG, bs=64)
acc_c2 = testar_modo("c² variável", cmaps_train_c2, cmaps_test_c2)
print(f"  c² variável: {acc_c2:.1f}%  ({time.time()-t0:.0f}s)")

# ── Modo 2: duas passadas ──
print("\n" + "="*60)
print("MODO 2: Duas passadas (self-attention física)")
print("="*60)
t0 = time.time()
cmaps_train_2p = compute_cmaps_duas_passadas(train_imgs, PG, bs=32)
cmaps_test_2p  = compute_cmaps_duas_passadas(test_imgs, PG, bs=64)
acc_2p = testar_modo("Duas passadas", cmaps_train_2p, cmaps_test_2p)
print(f"  Duas passadas: {acc_2p:.1f}%  ({time.time()-t0:.0f}s)")

# ── Modo 3: combo ──
print("\n" + "="*60)
print("MODO 3: c² variável + duas passadas (combo)")
print("="*60)
t0 = time.time()
cmaps_train_cb = compute_cmaps_combo(train_imgs, PG, bs=32)
cmaps_test_cb  = compute_cmaps_combo(test_imgs, PG, bs=64)
acc_cb = testar_modo("Combo", cmaps_train_cb, cmaps_test_cb)
print(f"  Combo: {acc_cb:.1f}%  ({time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# RESUMO
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESUMO — Attention Física (Zero Treino, MNIST)")
print(f"{'='*60}")
print(f"  Original (c² fixo):          {acc_orig:.1f}%")
print(f"  c² variável (refração):      {acc_c2:.1f}%")
print(f"  Duas passadas (attention):    {acc_2p:.1f}%")
print(f"  Combo (refração + attention): {acc_cb:.1f}%")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════════════════════
# VIZ: crystal maps comparados
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle(f'Auditoria 29 — Attention Física\n'
             f'Original={acc_orig:.1f}% | c²var={acc_c2:.1f}% | '
             f'2passadas={acc_2p:.1f}% | Combo={acc_cb:.1f}%',
             fontsize=12, fontweight='bold')

titulos = ['Original', 'c² variável', 'Duas passadas', 'Combo']
cmaps_list = [cmaps_test_orig, cmaps_test_c2, cmaps_test_2p, cmaps_test_cb]

for row, (titulo, cmaps) in enumerate(zip(titulos, cmaps_list)):
    for col in range(5):
        cm = cmaps[col].cpu().numpy()
        cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
        axes[row][col].imshow(cm, cmap='hot')
        if col == 0:
            axes[row][col].set_ylabel(titulo, fontsize=10, fontweight='bold')
        axes[row][col].set_title(f'dig={test_labs[col]}', fontsize=9)
        axes[row][col].axis('off')

plt.tight_layout()
plt.savefig('viz_audit_29_attention.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_29_attention.png")
print("Pronto.")
