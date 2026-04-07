"""
Auditoria 11: Limites do ResNet-Ψ

Parte A: MNIST com campo 96×96 (protótipos — sem decoder)
Parte B: CIFAR-10 (imagens coloridas 32×32 RGB)

Ambos sem treino nenhum. Só física + protótipos.
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

# ── Constantes físicas ──────────────────────────────────────────────────────
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


# ── Cristalização Competitiva ───────────────────────────────────────────────

class CrystalCompetitivo:
    def __init__(self, B, FS, sharpness=5.0, decay=0.02, ressonance_boost=0.1):
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


def build_gaussians(field_size, n_pixels=28, sigma=0.04):
    """Constrói mapa de gaussianas para n_pixels × n_pixels → field_size × field_size."""
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for pi in range(n_pixels):
        for pj in range(n_pixels):
            cx = 0.1 + 0.8 * pi / (n_pixels - 1)
            cy = 0.1 + 0.8 * pj / (n_pixels - 1)
            gs.append(torch.exp(-((xg-cx)**2+(yg-cy)**2)/(2*sigma**2)))
    return torch.stack(gs).view(n_pixels*n_pixels, -1)


def compute_crystal_maps_batch(X, PG, field_size, n_pixels=28, bs=64):
    """Computa crystal_maps em batch."""
    N, out = len(X), []
    npx = n_pixels * n_pixels
    for i in range(0, N, bs):
        B = X[i:i+bs]
        pert = (B.view(len(B), npx) @ PG.to(B.dtype)).view(len(B), field_size, field_size)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(len(B), field_size)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)
        out.append(mem.crystal_map)
    return torch.cat(out, dim=0)


def run_prototype_experiment(ds_name, train_by_class, test_imgs, test_labels,
                              PG, field_size, n_pixels, n_proto, n_test):
    """Roda experimento de protótipos completo."""

    # Subset balanceado de teste
    test_idx = []
    counts = [0] * 10
    for i, label in enumerate(test_labels):
        if counts[label] < n_test // 10:
            test_idx.append(i)
            counts[label] += 1
        if all(c >= n_test // 10 for c in counts):
            break

    # Crystal maps teste
    print(f"  Crystal_maps teste ({len(test_idx)})...", end=' ', flush=True)
    t1 = time.time()
    test_tensor = torch.stack([test_imgs[i] for i in test_idx]).to(DEVICE)
    test_labs = test_labels[test_idx]
    test_cmaps = compute_crystal_maps_batch(test_tensor, PG, field_size, n_pixels)
    print(f"{time.time()-t1:.0f}s")

    # Crystal maps treino + protótipos
    print(f"  Crystal_maps treino ({n_proto}×10)...", end=' ', flush=True)
    t1 = time.time()
    prototypes = {}
    for cls in range(10):
        imgs = torch.stack(train_by_class[cls][:n_proto]).to(DEVICE)
        cmaps = compute_crystal_maps_batch(imgs, PG, field_size, n_pixels)
        prototypes[cls] = cmaps.mean(dim=0)
    print(f"{time.time()-t1:.0f}s")

    # Classificação euclidiana
    correct = 0
    for i in range(len(test_idx)):
        cmap = test_cmaps[i].flatten()
        best_cls, best_dist = -1, float('inf')
        for cls in range(10):
            dist = ((cmap - prototypes[cls].flatten())**2).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_cls = cls
        if best_cls == test_labs[i]:
            correct += 1
    acc = correct / len(test_idx) * 100

    # Stats
    n_crys_avg = np.mean([(prototypes[cls] > 0.01).float().sum().item() for cls in range(10)])

    return acc, n_crys_avg, prototypes


# ══════════════════════════════════════════════════════════════════════════════
# PARTE A: MNIST com campo 96×96
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PARTE A: MNIST — Campo 48×48 vs 96×96")
print(f"{'='*70}")

tf_mnist = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
train_mnist = datasets.MNIST('./data', train=True, download=True, transform=tf_mnist)
test_mnist  = datasets.MNIST('./data', train=False, download=True, transform=tf_mnist)

train_by_class_mnist = {i: [] for i in range(10)}
for img, label in train_mnist:
    train_by_class_mnist[label].append(img.squeeze(0))

test_imgs_mnist = []
test_labels_mnist = []
for img, label in test_mnist:
    test_imgs_mnist.append(img.squeeze(0))
    test_labels_mnist.append(label)
test_labels_mnist = np.array(test_labels_mnist)

N_PROTO = 1000
N_TEST  = 1000

results_a = {}

for fs in [48, 96]:
    print(f"\n── Campo {fs}×{fs} ──")
    PG = build_gaussians(fs, n_pixels=28)
    bs = 64 if fs <= 64 else 32
    acc, n_crys, protos = run_prototype_experiment(
        'MNIST', train_by_class_mnist, test_imgs_mnist, test_labels_mnist,
        PG, fs, 28, N_PROTO, N_TEST)
    results_a[fs] = {'acc': acc, 'n_crys': n_crys}
    print(f"  → Acurácia: {acc:.1f}% | Cristais médios: {n_crys:.0f}")
    del PG
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ══════════════════════════════════════════════════════════════════════════════
# PARTE B: CIFAR-10
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PARTE B: CIFAR-10 (RGB 32×32)")
print(f"{'='*70}")

CIFAR_CLASSES = ['Avião', 'Auto', 'Pássaro', 'Gato', 'Cervo',
                 'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

# CIFAR-10: converte RGB → grayscale para o campo 2D
tf_cifar = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4809,),(0.2255,)),
])

train_cifar = datasets.CIFAR10('./data', train=True, download=True, transform=tf_cifar)
test_cifar  = datasets.CIFAR10('./data', train=False, download=True, transform=tf_cifar)

train_by_class_cifar = {i: [] for i in range(10)}
for img, label in train_cifar:
    train_by_class_cifar[label].append(img.squeeze(0))

test_imgs_cifar = []
test_labels_cifar = []
for img, label in test_cifar:
    test_imgs_cifar.append(img.squeeze(0))
    test_labels_cifar.append(label)
test_labels_cifar = np.array(test_labels_cifar)

# CIFAR é 32×32, campo 64×64
FIELD_CIFAR = 64
PG_cifar = build_gaussians(FIELD_CIFAR, n_pixels=32)

results_b = {}

for n_proto in [100, 500, 1000, 5000]:
    print(f"\n── {n_proto} exemplos por protótipo ──")
    acc, n_crys, protos_cifar = run_prototype_experiment(
        'CIFAR-10', train_by_class_cifar, test_imgs_cifar, test_labels_cifar,
        PG_cifar, FIELD_CIFAR, 32, n_proto, N_TEST)
    results_b[n_proto] = {'acc': acc, 'n_crys': n_crys}
    print(f"  → Acurácia: {acc:.1f}% | Cristais médios: {n_crys:.0f}")

    for cls in range(10):
        n_c = (protos_cifar[cls] > 0.01).float().sum().item()
        print(f"    {CIFAR_CLASSES[cls]:>10}: {n_c:.0f} cristais")


# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESULTADO FINAL")
print(f"{'='*70}")

print(f"\nParte A — MNIST campo maior:")
for fs, r in results_a.items():
    print(f"  Campo {fs}×{fs}: {r['acc']:.1f}% ({r['n_crys']:.0f} cristais)")

print(f"\nParte B — CIFAR-10 (grayscale → campo 64×64):")
for np_, r in results_b.items():
    print(f"  {np_} exemplos: {r['acc']:.1f}% ({r['n_crys']:.0f} cristais)")

print(f"\nReferências:")
print(f"  MNIST 48×48 protótipos: 77.4%")
print(f"  Fashion 48×48 protótipos: 67.0%")
print(f"  Aleatório: 10%")


# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Auditoria 11 — Campo Maior + CIFAR-10', fontsize=13)

# Plot A: MNIST campo comparação
ax = axes[0]
fields = list(results_a.keys())
accs = [results_a[f]['acc'] for f in fields]
ax.bar([f'{f}×{f}' for f in fields], accs, color=['#4363d8', '#3cb44b'], alpha=0.8)
ax.axhline(y=77.4, color='orange', linestyle='--', alpha=0.5, label='Ref 48² (77.4%)')
ax.set_ylabel('Acurácia (%)')
ax.set_title('MNIST: Campo 48² vs 96²')
ax.legend()
ax.grid(alpha=0.3, axis='y')
for i, (f, a) in enumerate(zip(fields, accs)):
    ax.text(i, a + 0.5, f'{a:.1f}%', ha='center', fontsize=10)

# Plot B: CIFAR curva
ax = axes[1]
ns = list(results_b.keys())
accs_cifar = [results_b[n]['acc'] for n in ns]
ax.plot(ns, accs_cifar, 'o-', color='#e6194b', linewidth=2, markersize=8)
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Aleatório (10%)')
ax.set_xlabel('Exemplos por protótipo')
ax.set_ylabel('Acurácia (%)')
ax.set_title('CIFAR-10: Acurácia vs N exemplos')
ax.legend()
ax.grid(alpha=0.3)

# Plot C: Protótipos CIFAR
ax = axes[2]
ax.set_title('Protótipos CIFAR-10')
ax.axis('off')
for i in range(10):
    row, col = i // 5, i % 5
    sub = fig.add_axes([0.68 + col*0.06, 0.55 - row*0.4, 0.055, 0.35])
    proto = protos_cifar[i].cpu().numpy()
    vmax = np.abs(proto).max()
    sub.imshow(proto, cmap='hot', aspect='auto')
    sub.set_title(CIFAR_CLASSES[i], fontsize=5)
    sub.axis('off')

plt.savefig('viz_audit_11_campo96_cifar.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_11_campo96_cifar.png")
print("Pronto.")
