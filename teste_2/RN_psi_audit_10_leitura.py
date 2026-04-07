"""
Auditoria 10: Métodos de Leitura dos Protótipos

Mesmo crystal_maps, diferentes formas de comparar.
Testa em MNIST e Fashion-MNIST com 5000 exemplos por protótipo.

Métodos:
  1. Distância euclidiana (baseline atual — 77% MNIST, 67% Fashion)
  2. Correlação coseno
  3. Distância ponderada por variância inter-classe
  4. Distância ponderada por discriminância (Fisher)
  5. Mahalanobis simplificado (variância por pixel)

Zero decoder. Zero treino. Só estatística dos protótipos.
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


# ── Métodos de classificação ────────────────────────────────────────────────

def classify_euclidean(test_cmaps, test_labels, prototypes):
    """Distância euclidiana ao protótipo mais próximo."""
    correct = 0
    for i in range(len(test_labels)):
        cmap = test_cmaps[i].flatten()
        best_cls, best_dist = -1, float('inf')
        for cls in range(10):
            dist = ((cmap - prototypes[cls].flatten())**2).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_cls = cls
        if best_cls == test_labels[i]:
            correct += 1
    return correct / len(test_labels) * 100


def classify_cosine(test_cmaps, test_labels, prototypes):
    """Correlação coseno."""
    correct = 0
    for i in range(len(test_labels)):
        cmap = test_cmaps[i].flatten()
        cmap_n = cmap / (cmap.norm() + 1e-8)
        best_cls, best_sim = -1, -999
        for cls in range(10):
            p = prototypes[cls].flatten()
            p_n = p / (p.norm() + 1e-8)
            sim = (cmap_n * p_n).sum().item()
            if sim > best_sim:
                best_sim = sim
                best_cls = cls
        if best_cls == test_labels[i]:
            correct += 1
    return correct / len(test_labels) * 100


def classify_weighted_variance(test_cmaps, test_labels, prototypes, class_cmaps):
    """
    Distância ponderada pela variância inter-classe.
    Pixels onde os protótipos mais divergem pesam mais.
    """
    # Computa variância entre protótipos (por pixel)
    proto_stack = torch.stack([prototypes[cls].flatten() for cls in range(10)])  # (10, D)
    inter_var = proto_stack.var(dim=0)  # (D,) — variância entre classes
    # Normaliza pesos
    weights = inter_var / (inter_var.sum() + 1e-8)

    correct = 0
    for i in range(len(test_labels)):
        cmap = test_cmaps[i].flatten()
        best_cls, best_dist = -1, float('inf')
        for cls in range(10):
            diff = cmap - prototypes[cls].flatten()
            dist = (weights * diff**2).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_cls = cls
        if best_cls == test_labels[i]:
            correct += 1
    return correct / len(test_labels) * 100


def classify_fisher(test_cmaps, test_labels, prototypes, class_cmaps):
    """
    Discriminante de Fisher por pixel.
    Peso = variância inter-classe / variância intra-classe.
    Pixels discriminativos pesam mais.
    """
    # Variância inter-classe
    proto_stack = torch.stack([prototypes[cls].flatten() for cls in range(10)])
    inter_var = proto_stack.var(dim=0)

    # Variância intra-classe (média das variâncias dentro de cada classe)
    intra_vars = []
    for cls in range(10):
        cmaps_cls = class_cmaps[cls].view(len(class_cmaps[cls]), -1)
        intra_vars.append(cmaps_cls.var(dim=0))
    intra_var = torch.stack(intra_vars).mean(dim=0)

    # Fisher = inter / intra
    fisher = inter_var / (intra_var + 1e-8)
    weights = fisher / (fisher.sum() + 1e-8)

    correct = 0
    for i in range(len(test_labels)):
        cmap = test_cmaps[i].flatten()
        best_cls, best_dist = -1, float('inf')
        for cls in range(10):
            diff = cmap - prototypes[cls].flatten()
            dist = (weights * diff**2).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_cls = cls
        if best_cls == test_labels[i]:
            correct += 1
    return correct / len(test_labels) * 100


def classify_mahalanobis_diag(test_cmaps, test_labels, prototypes, class_cmaps):
    """
    Mahalanobis diagonal: distância normalizada pela variância por pixel de cada classe.
    """
    # Variância por classe por pixel
    class_vars = {}
    for cls in range(10):
        cmaps_cls = class_cmaps[cls].view(len(class_cmaps[cls]), -1)
        class_vars[cls] = cmaps_cls.var(dim=0) + 1e-6  # evita divisão por zero

    correct = 0
    for i in range(len(test_labels)):
        cmap = test_cmaps[i].flatten()
        best_cls, best_dist = -1, float('inf')
        for cls in range(10):
            diff = cmap - prototypes[cls].flatten()
            dist = (diff**2 / class_vars[cls]).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_cls = cls
        if best_cls == test_labels[i]:
            correct += 1
    return correct / len(test_labels) * 100


# ── Experimento ──────────────────────────────────────────────────────────────

N_PROTO = 1000
N_TEST  = 1000

PG = build_gaussians()

DATASETS = [
    ('MNIST', datasets.MNIST, (0.1307,), (0.3081,)),
    ('Fashion-MNIST', datasets.FashionMNIST, (0.2860,), (0.3530,)),
]

print(f"\n{'='*75}")
print("AUDITORIA 10: Métodos de Leitura dos Protótipos")
print(f"Protótipos: {N_PROTO} exemplos | Teste: {N_TEST} imagens")
print(f"{'='*75}")

all_dataset_results = {}

for ds_name, ds_class, mean, std in DATASETS:
    print(f"\n{'━'*40}")
    print(f"  {ds_name}")
    print(f"{'━'*40}")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = ds_class('./data', train=True, download=True, transform=tf)
    test_ds  = ds_class('./data', train=False, download=True, transform=tf)

    train_by_class = {i: [] for i in range(10)}
    for img, label in train_ds:
        train_by_class[label].append(img.squeeze(0))

    all_test_imgs, all_test_labels = [], []
    for img, label in test_ds:
        all_test_imgs.append(img.squeeze(0))
        all_test_labels.append(label)
    all_test_labels = np.array(all_test_labels)

    # Subset balanceado
    test_idx = []
    counts = [0] * 10
    for i, label in enumerate(all_test_labels):
        if counts[label] < N_TEST // 10:
            test_idx.append(i)
            counts[label] += 1
        if all(c >= N_TEST // 10 for c in counts):
            break

    # Crystal maps de teste
    print(f"  Computando crystal_maps teste ({N_TEST})...")
    t1 = time.time()
    test_tensor = torch.stack([all_test_imgs[i] for i in test_idx]).to(DEVICE)
    test_labels = all_test_labels[test_idx]
    test_cmaps = compute_crystal_maps_batch(test_tensor, PG)
    print(f"  Pronto: {time.time()-t1:.0f}s")

    # Crystal maps de treino + protótipos
    print(f"  Computando crystal_maps treino ({N_PROTO}×10)...")
    t1 = time.time()
    class_cmaps = {}
    for cls in range(10):
        imgs = torch.stack(train_by_class[cls][:N_PROTO]).to(DEVICE)
        class_cmaps[cls] = compute_crystal_maps_batch(imgs, PG)
    print(f"  Pronto: {time.time()-t1:.0f}s")

    prototypes = {cls: class_cmaps[cls].mean(dim=0) for cls in range(10)}

    # Testa cada método
    methods = [
        ('Euclidiana',          lambda: classify_euclidean(test_cmaps, test_labels, prototypes)),
        ('Coseno',              lambda: classify_cosine(test_cmaps, test_labels, prototypes)),
        ('Pond. variância',     lambda: classify_weighted_variance(test_cmaps, test_labels, prototypes, class_cmaps)),
        ('Fisher',              lambda: classify_fisher(test_cmaps, test_labels, prototypes, class_cmaps)),
        ('Mahalanobis diag.',   lambda: classify_mahalanobis_diag(test_cmaps, test_labels, prototypes, class_cmaps)),
    ]

    results = {}
    for mname, mfunc in methods:
        t1 = time.time()
        acc = mfunc()
        elapsed = time.time() - t1
        results[mname] = acc
        print(f"    {mname:<22} {acc:.1f}%  ({elapsed:.1f}s)")

    all_dataset_results[ds_name] = results


# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("RESULTADO FINAL")
print(f"{'='*75}")
print(f"\n{'Método':<22} {'MNIST':>8} {'Fashion':>10}")
print("-"*42)

for mname in ['Euclidiana', 'Coseno', 'Pond. variância', 'Fisher', 'Mahalanobis diag.']:
    m_acc = all_dataset_results['MNIST'].get(mname, 0)
    f_acc = all_dataset_results['Fashion-MNIST'].get(mname, 0)
    print(f"  {mname:<22} {m_acc:>6.1f}%  {f_acc:>8.1f}%")

best_mnist = max(all_dataset_results['MNIST'].items(), key=lambda x: x[1])
best_fashion = max(all_dataset_results['Fashion-MNIST'].items(), key=lambda x: x[1])
print(f"\n  Melhor MNIST:   {best_mnist[0]} ({best_mnist[1]:.1f}%)")
print(f"  Melhor Fashion: {best_fashion[0]} ({best_fashion[1]:.1f}%)")
print(f"\n  Ref. anterior MNIST: 77.4% (euclidiana)")
print(f"  Ref. anterior Fashion: 67.0% (correlação)")


# ── Plot ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle('Auditoria 10 — Métodos de Leitura dos Protótipos', fontsize=13)

methods_names = ['Euclidiana', 'Coseno', 'Pond. variância', 'Fisher', 'Mahalanobis diag.']
mnist_accs = [all_dataset_results['MNIST'][m] for m in methods_names]
fashion_accs = [all_dataset_results['Fashion-MNIST'][m] for m in methods_names]

x = np.arange(len(methods_names))
w = 0.35
ax.bar(x - w/2, mnist_accs, w, label='MNIST', color='#4363d8', alpha=0.8)
ax.bar(x + w/2, fashion_accs, w, label='Fashion-MNIST', color='#e6194b', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(methods_names, fontsize=8, rotation=15)
ax.set_ylabel('Acurácia (%)')
ax.axhline(y=77.4, color='#4363d8', linestyle='--', alpha=0.3, label='MNIST ref (77.4%)')
ax.axhline(y=67.0, color='#e6194b', linestyle='--', alpha=0.3, label='Fashion ref (67.0%)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

for i, (m, f) in enumerate(zip(mnist_accs, fashion_accs)):
    ax.text(i - w/2, m + 0.5, f'{m:.1f}', ha='center', fontsize=7)
    ax.text(i + w/2, f + 0.5, f'{f:.1f}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('viz_audit_10_leitura.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_10_leitura.png")
print("Pronto.")
