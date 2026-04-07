"""
Auditoria 16: Protótipos Múltiplos por Classe (Subtipos)

Em vez de 1 protótipo por classe (média global),
agrupa crystal_maps por similaridade dentro de cada classe
e cria N subtipos por classe.

Testa: 1, 3, 5, 10 subtipos por classe.
Classificação: distância ao subtipo mais próximo de qualquer classe.

Zero treino — KMeans é só geometria, sem backprop.
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
from sklearn.cluster import KMeans

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

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


# ── Carregar MNIST ────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

train_by_class = {i: [] for i in range(10)}
for img, label in train_ds:
    train_by_class[label].append(img.squeeze(0))

test_imgs, test_labels = [], []
for img, label in test_ds:
    test_imgs.append(img.squeeze(0))
    test_labels.append(label)
test_labels = np.array(test_labels)

PG = build_gaussians()

N_PROTO  = 1000  # exemplos por classe para computar subtipos
N_TEST   = 500   # exemplos de teste

# ── Pré-computar crystal_maps de treino ───────────────────────────────────────

print(f"\nComputando crystal_maps de treino ({N_PROTO} por classe)...")
t0 = time.time()
train_cmaps = {}
for cls in range(10):
    imgs = torch.stack(train_by_class[cls][:N_PROTO]).to(DEVICE)
    train_cmaps[cls] = compute_crystal_maps_batch(imgs, PG)
    print(f"  Classe {cls}: {len(train_cmaps[cls])} crystal_maps")
print(f"  Pronto: {time.time()-t0:.0f}s")

# ── Pré-computar crystal_maps de teste ───────────────────────────────────────

print(f"\nComputando crystal_maps de teste ({N_TEST})...")
test_subset_idx = []
counts = [0]*10
for i, label in enumerate(test_labels):
    if counts[label] < N_TEST//10:
        test_subset_idx.append(i)
        counts[label] += 1
    if all(c >= N_TEST//10 for c in counts):
        break

test_tensor = torch.stack([test_imgs[i] for i in test_subset_idx]).to(DEVICE)
test_labels_sub = test_labels[test_subset_idx]
t1 = time.time()
test_cmaps = compute_crystal_maps_batch(test_tensor, PG)
test_flat  = test_cmaps.view(len(test_subset_idx), -1).float()
print(f"  Pronto: {time.time()-t1:.0f}s")

# ── Experimento: N subtipos por classe ────────────────────────────────────────

N_SUBTIPOS_LIST = [1, 3, 5, 10]
results = []

print(f"\n{'='*60}")
print("AUDITORIA 16: Protótipos Múltiplos por Classe")
print(f"{'='*60}")

for n_sub in N_SUBTIPOS_LIST:
    print(f"\n── {n_sub} subtipo(s) por classe ──")
    t1 = time.time()

    all_prototypes = []   # lista de (protótipo, classe)
    all_proto_labels = []

    for cls in range(10):
        cmaps_flat = train_cmaps[cls].view(N_PROTO, -1).cpu().numpy()

        if n_sub == 1:
            # Média global — baseline
            proto = train_cmaps[cls].mean(dim=0)
            all_prototypes.append(proto.view(1, -1).float().to(DEVICE))
            all_proto_labels.extend([cls])
        else:
            # KMeans para agrupar variações dentro da classe
            km = KMeans(n_clusters=n_sub, random_state=42, n_init=5)
            km.fit(cmaps_flat)
            for k in range(n_sub):
                mask = km.labels_ == k
                if mask.sum() == 0:
                    continue
                proto = train_cmaps[cls][mask].mean(dim=0)
                all_prototypes.append(proto.view(1, -1).float().to(DEVICE))
                all_proto_labels.append(cls)

    # Matriz de protótipos: (N_total_protos, FS*FS)
    proto_matrix = torch.cat(all_prototypes, dim=0)
    proto_labels = np.array(all_proto_labels)

    # Distância de cada teste ao protótipo mais próximo
    dists = torch.cdist(test_flat, proto_matrix)
    nearest_proto = dists.argmin(dim=1).cpu().numpy()
    preds = proto_labels[nearest_proto]
    acc = (preds == test_labels_sub).mean() * 100

    print(f"  Total de protótipos: {len(proto_labels)} ({n_sub} × 10 classes)")
    print(f"  Acurácia: {acc:.1f}%  (tempo: {time.time()-t1:.0f}s)")

    results.append({'n_sub': n_sub, 'acc': acc, 'n_proto': len(proto_labels)})

# ── Resumo ────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("RESULTADO FINAL")
print(f"{'='*60}")
print(f"\n{'Subtipos':>10} {'Protótipos':>12} {'Acurácia':>10}")
print("-"*36)
for r in results:
    print(f"  {r['n_sub']:>8}   {r['n_proto']:>10}   {r['acc']:>7.1f}%")
print(f"\nReferência MNIST protótipos (1 subtipo, 5000 ex): 77.4%")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Auditoria 16 — Protótipos Múltiplos por Classe', fontsize=13)

ax = axes[0]
ns  = [r['n_sub'] for r in results]
acc = [r['acc']   for r in results]
ax.plot(ns, acc, 'o-', color='#4363d8', linewidth=2, markersize=10)
ax.axhline(y=77.4, color='orange', linestyle='--', alpha=0.7, label='Referência (77.4%)')
ax.set_xlabel('Subtipos por classe')
ax.set_ylabel('Acurácia (%)')
ax.set_title('Acurácia vs N subtipos')
ax.set_xticks(ns)
ax.legend()
ax.grid(alpha=0.3)

# Visualizar subtipos do dígito 3
ax = axes[1]
ax.axis('off')
n_sub_viz = min(5, N_SUBTIPOS_LIST[-1])
cmaps_3 = train_cmaps[3].view(N_PROTO, -1).cpu().numpy()
km_viz = KMeans(n_clusters=n_sub_viz, random_state=42, n_init=5)
km_viz.fit(cmaps_3)

fig2, axes2 = plt.subplots(1, n_sub_viz, figsize=(4*n_sub_viz, 4))
fig2.suptitle(f'Subtipos do dígito 3 ({n_sub_viz} clusters)\nCada subtipo = variação real da escrita', fontsize=12)
for k in range(n_sub_viz):
    mask = km_viz.labels_ == k
    proto_k = train_cmaps[3][mask].mean(dim=0).cpu().numpy()
    axes2[k].imshow(proto_k, cmap='hot', aspect='auto')
    axes2[k].set_title(f'Subtipo {k+1}\n({mask.sum()} exemplos)', fontsize=9)
    axes2[k].axis('off')
plt.tight_layout()
plt.savefig('viz_audit_16_subtipos_3.png', dpi=130, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(ns, acc, 'o-', color='#4363d8', linewidth=2, markersize=10)
plt.axhline(y=77.4, color='orange', linestyle='--', alpha=0.7, label='Referência (77.4%)')
plt.xlabel('Subtipos por classe')
plt.ylabel('Acurácia (%)')
plt.title('Auditoria 16 — Acurácia vs N subtipos por classe')
plt.xticks(ns)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('viz_audit_16_acuracia.png', dpi=130, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_16_acuracia.png")
print(f"-> viz_audit_16_subtipos_3.png")
print("Pronto.")
