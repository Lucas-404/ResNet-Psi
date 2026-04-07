"""
Auditoria 13: EMNIST Letters — Protótipos Cristalinos

26 classes (letras a-z), mesmo mecanismo do MNIST/Fashion.
Testa generalização pra letras manuscritas e escalabilidade de classes.

Zero decoder. Zero treino. Só física + média.
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

from torchvision import datasets, transforms

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ── Constantes ──────────────────────────────────────────────────────────────
FIELD_SIZE     = 48
PSI_DT         = 0.05
PSI_GAMMA      = 0.06
PSI_ALPHA      = 0.04
PSI_BETA       = 0.005
PSI_C2         = 0.3
STIM_ON        = 40
STIM_TOTAL     = 80

CRYSTAL_W      = 20
CRYSTAL_K      = 3
CRYSTAL_A_MIN  = 0.3
CRYSTAL_CV_MAX = 0.15
CRYSTAL_SEP    = 5
CRYSTAL_REMIT  = 0.05

_DT    = torch.tensor(PSI_DT,    device=DEVICE)
_GAMMA = torch.tensor(PSI_GAMMA, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_BETA  = torch.tensor(PSI_BETA,  device=DEVICE)
_C2    = torch.tensor(PSI_C2,    device=DEVICE)

N_CLASSES = 26
# EMNIST letters: labels 1-26 (a=1, b=2, ..., z=26)
CLASS_NAMES = [chr(ord('a') + i) for i in range(26)]


# ── Cristalização Competitiva ───────────────────────────────────────────────

class CrystalCompetitivo:
    def __init__(self, B, FS=FIELD_SIZE, sharpness=5.0, decay=0.02, ressonance_boost=0.1):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
        self.sharpness = sharpness
        self.decay = decay
        self.ressonance_boost = ressonance_boost
        ks = 2 * CRYSTAL_SEP + 1
        self._dilate = torch.ones(1, 1, ks, ks, device=DEVICE)

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
        cv   = env.std(dim=1) / (mean + 1e-8)
        amp_score = torch.sigmoid(self.sharpness * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(self.sharpness * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(self.sharpness * (8.0 - mean))
        cand = amp_score * cv_score * sat_score
        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0,1)
        new_crystals = cand * (1.0 - occ) * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + new_crystals, 0, 10.)
        self.crystal_hp = torch.where(
            new_crystals > 0.01,
            torch.clamp(self.crystal_hp + 1.0, 0, 5.0),
            self.crystal_hp)
        ressonance = field.abs() * (self.crystal_map > 0.01).float()
        self.crystal_hp = self.crystal_hp + ressonance * self.ressonance_boost
        self.crystal_hp = self.crystal_hp - self.decay
        alive = (self.crystal_hp > 0).float()
        self.crystal_map = self.crystal_map * alive
        self.crystal_hp  = torch.clamp(self.crystal_hp * alive, 0, 5.0)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


# ── Física ──────────────────────────────────────────────────────────────────

def psi_step(field, velocity, sources, active):
    lap_k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                          device=DEVICE).view(1,1,3,3).to(field.dtype)
    if active:
        field = field + sources * (_DT * 0.1)
    lap = F.conv2d(F.pad(field.unsqueeze(1),(1,1,1,1),mode='circular'), lap_k).squeeze(1)
    acc = _C2*lap - _GAMMA*velocity + _ALPHA*torch.tanh(field)*field - _BETA*field*field**2
    velocity = torch.clamp(velocity + acc*_DT, -5., 5.)
    field    = torch.clamp(field + velocity*_DT, -10., 10.)
    return field, velocity


def build_gaussians(field_size=FIELD_SIZE, sigma=0.04):
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            gs.append(torch.exp(-((xg-cx)**2+(yg-cy)**2)/(2*sigma**2)))
    return torch.stack(gs).view(784, -1)


def compute_crystal_maps_batch(X, PG, bs=64):
    N, out = len(X), []
    for i in range(0, N, bs):
        B = X[i:i+bs]
        pert = (B.view(len(B), 784) @ PG.to(B.dtype)).view(len(B), FIELD_SIZE, FIELD_SIZE)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(len(B), FIELD_SIZE)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)
        out.append(mem.crystal_map)
    return torch.cat(out, dim=0)


# ── EMNIST Letters ───────────────────────────────────────────────────────────

print("\nCarregando EMNIST Letters...")
tf = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1722,),(0.3309,))])

# EMNIST letters: split='letters', labels de 1 a 26
train_ds = datasets.EMNIST('./data', split='letters', train=True,  download=True, transform=tf)
test_ds  = datasets.EMNIST('./data', split='letters', train=False, download=True, transform=tf)

print(f"  Treino: {len(train_ds)} amostras | Teste: {len(test_ds)} amostras")

# Separar por classe (labels 1-26 → índice 0-25)
train_by_class = {i: [] for i in range(N_CLASSES)}
for img, label in train_ds:
    cls = label - 1  # converter 1-26 para 0-25
    if cls < N_CLASSES:
        train_by_class[cls].append(img.squeeze(0))

all_test_imgs = []
all_test_labels = []
for img, label in test_ds:
    cls = label - 1
    if cls < N_CLASSES:
        all_test_imgs.append(img.squeeze(0))
        all_test_labels.append(cls)
all_test_labels = np.array(all_test_labels)

print(f"  Amostras por classe (treino): {[len(train_by_class[i]) for i in range(5)]}... (primeiras 5)")

PG = build_gaussians()

# ── Experimento ──────────────────────────────────────────────────────────────

N_PROTO_LIST = [10, 100, 500, 2000]
N_TEST = 520  # 20 por classe × 26 classes

print(f"\n{'='*70}")
print("AUDITORIA 13: EMNIST Letters — Protótipos Cristalinos")
print(f"26 classes | Teste: {N_TEST} imagens | Zero treino")
print(f"Referência aleatório: {100/N_CLASSES:.1f}%")
print(f"{'='*70}")

# Selecionar subset de teste balanceado
test_subset_idx = []
counts = [0] * N_CLASSES
n_per_class = N_TEST // N_CLASSES
for i, label in enumerate(all_test_labels):
    if counts[label] < n_per_class:
        test_subset_idx.append(i)
        counts[label] += 1
    if all(c >= n_per_class for c in counts):
        break

print(f"\nPré-computando crystal_maps de teste ({len(test_subset_idx)} imgs)...")
test_imgs_tensor = torch.stack([all_test_imgs[i] for i in test_subset_idx]).to(DEVICE)
test_labels_sub = all_test_labels[test_subset_idx]
t1 = time.time()
test_cmaps = compute_crystal_maps_batch(test_imgs_tensor, PG)
print(f"  Pronto: {time.time()-t1:.0f}s")

all_results = []
t0 = time.time()

for n_proto in N_PROTO_LIST:
    print(f"\n── {n_proto} exemplos por protótipo ──")

    print(f"  Computando crystal_maps de treino ({N_CLASSES} classes × {n_proto} imgs)...")
    t1 = time.time()
    prototypes = {}
    n_crystals = []
    for cls in range(N_CLASSES):
        imgs_cls = train_by_class[cls][:n_proto]
        if len(imgs_cls) == 0:
            print(f"    AVISO: classe {CLASS_NAMES[cls]} sem amostras!")
            prototypes[cls] = torch.zeros(FIELD_SIZE, FIELD_SIZE, device=DEVICE)
            n_crystals.append(0)
            continue
        imgs_tensor = torch.stack(imgs_cls).to(DEVICE)
        cmaps = compute_crystal_maps_batch(imgs_tensor, PG)
        prototypes[cls] = cmaps.mean(dim=0)
        n_crystals.append((prototypes[cls] > 0.01).float().sum().item())
    print(f"  Protótipos prontos: {time.time()-t1:.0f}s")

    avg_crystals = np.mean(n_crystals)
    print(f"  Cristais por protótipo: {avg_crystals:.0f} (média)")

    # Distância euclidiana vetorizada
    proto_matrix = torch.stack([prototypes[cls] for cls in range(N_CLASSES)]).view(N_CLASSES, -1)
    test_flat = test_cmaps.view(len(test_subset_idx), -1)

    # cdist: (N_test, N_classes)
    dists = torch.cdist(test_flat.float(), proto_matrix.float())
    preds = dists.argmin(dim=1).cpu().numpy()
    acc = (preds == test_labels_sub).mean() * 100

    # Acurácia por classe (top-5 melhores e piores)
    per_class_acc = {}
    for cls in range(N_CLASSES):
        mask = test_labels_sub == cls
        if mask.sum() > 0:
            per_class_acc[cls] = (preds[mask] == test_labels_sub[mask]).mean() * 100

    sorted_cls = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)

    all_results.append({
        'n_proto': n_proto,
        'acc': acc,
        'avg_crystals': avg_crystals,
        'per_class_acc': per_class_acc,
    })

    print(f"  → Acurácia geral: {acc:.1f}%  (aleatório: {100/N_CLASSES:.1f}%)")
    print(f"  → Top-5 classes: {[(CLASS_NAMES[c], f'{a:.0f}%') for c, a in sorted_cls[:5]]}")
    print(f"  → Bot-5 classes: {[(CLASS_NAMES[c], f'{a:.0f}%') for c, a in sorted_cls[-5:]]}")


# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESULTADO FINAL: EMNIST Letters (26 classes)")
print(f"{'='*70}")
print(f"\n{'N exemplos':>12} {'Acurácia':>10} {'Cristais/proto':>15}")
print("-"*42)
for r in all_results:
    print(f"  {r['n_proto']:>10}   {r['acc']:>7.1f}%   {r['avg_crystals']:>12.0f}")

print(f"\nReferência aleatório: {100/N_CLASSES:.1f}%")
print(f"Referência MNIST (10 classes): 77.4%")
print(f"Referência Fashion (10 classes): 67.0%")
print(f"\nTempo total: {time.time()-t0:.0f}s")


# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Auditoria 13 — EMNIST Letters: Protótipos Cristalinos (zero treino)', fontsize=13)

ns  = [r['n_proto'] for r in all_results]
acc = [r['acc'] for r in all_results]

ax = axes[0]
ax.plot(ns, acc, 's-', color='#4363d8', linewidth=2, markersize=8, label='EMNIST Letters (26 cls)')
ax.axhline(y=100/N_CLASSES, color='gray',   linestyle='--', alpha=0.6, label=f'Aleatório ({100/N_CLASSES:.1f}%)')
ax.axhline(y=67.0,          color='green',  linestyle='--', alpha=0.6, label='Fashion-MNIST (67.0%)')
ax.axhline(y=77.4,          color='orange', linestyle='--', alpha=0.6, label='MNIST (77.4%)')
ax.set_xlabel('Exemplos por protótipo')
ax.set_ylabel('Acurácia (%)')
ax.set_title('Acurácia vs N exemplos')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Acurácia por classe (último ponto)
ax = axes[1]
if all_results:
    last = all_results[-1]
    classes = list(range(N_CLASSES))
    accs_per_cls = [last['per_class_acc'].get(c, 0) for c in classes]
    bars = ax.bar(CLASS_NAMES, accs_per_cls, color='#4363d8', alpha=0.7)
    ax.axhline(y=last['acc'], color='red', linestyle='--', alpha=0.7, label=f'Média ({last["acc"]:.1f}%)')
    ax.axhline(y=100/N_CLASSES, color='gray', linestyle='--', alpha=0.5, label=f'Aleatório')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Acurácia (%)')
    ax.set_title(f'Acurácia por letra ({all_results[-1]["n_proto"]} exemplos/proto)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig('viz_audit_13_emnist.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_13_emnist.png")


# ── Visualização dos protótipos no campo (último n_proto) ─────────────────────

print("\nGerando visualização dos protótipos no campo...")

# Recomputar protótipos com 500 exemplos para a visualização
N_VIZ = 500
prototypes_viz = {}
for cls in range(N_CLASSES):
    imgs_cls = train_by_class[cls][:N_VIZ]
    imgs_tensor = torch.stack(imgs_cls).to(DEVICE)
    cmaps = compute_crystal_maps_batch(imgs_tensor, PG)
    prototypes_viz[cls] = cmaps.mean(dim=0)

global_proto_viz = torch.stack([prototypes_viz[cls] for cls in range(N_CLASSES)]).mean(dim=0)

# Plot 1: Protótipos brutos (crystal_map médio por letra)
fig, axes = plt.subplots(4, 7, figsize=(20, 12))
fig.suptitle(f'Protótipos Cristalinos — EMNIST Letters ({N_VIZ} exemplos/classe)\nO que o campo "vê" de cada letra', fontsize=14)
for i, cls in enumerate(range(N_CLASSES)):
    ax = axes[i // 7][i % 7]
    proto = prototypes_viz[cls].cpu().numpy()
    ax.imshow(proto, cmap='hot', aspect='auto')
    acc_cls = all_results[-1]['per_class_acc'].get(cls, 0)
    ax.set_title(f'{CLASS_NAMES[cls].upper()}  ({acc_cls:.0f}%)', fontsize=10, fontweight='bold')
    ax.axis('off')
# Apagar subplots vazios (26 letras, 28 células)
for i in range(N_CLASSES, 28):
    axes[i // 7][i % 7].axis('off')
plt.tight_layout()
plt.savefig('viz_audit_13_prototipos.png', dpi=130, bbox_inches='tight')
plt.close()
print("-> viz_audit_13_prototipos.png")

# Plot 2: Protótipos diferenciais (letra - média global) — o que distingue cada letra
fig, axes = plt.subplots(4, 7, figsize=(20, 12))
fig.suptitle(f'Protótipos Diferenciais — EMNIST Letters\nO que cada letra tem de ÚNICO no campo (letra - média global)', fontsize=14)
for i, cls in enumerate(range(N_CLASSES)):
    ax = axes[i // 7][i % 7]
    diff = (prototypes_viz[cls] - global_proto_viz).cpu().numpy()
    vmax = np.abs(diff).max()
    ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    acc_cls = all_results[-1]['per_class_acc'].get(cls, 0)
    ax.set_title(f'{CLASS_NAMES[cls].upper()}  ({acc_cls:.0f}%)', fontsize=10, fontweight='bold')
    ax.axis('off')
for i in range(N_CLASSES, 28):
    axes[i // 7][i % 7].axis('off')
plt.tight_layout()
plt.savefig('viz_audit_13_diferenciais.png', dpi=130, bbox_inches='tight')
plt.close()
print("-> viz_audit_13_diferenciais.png")

print("Pronto.")
