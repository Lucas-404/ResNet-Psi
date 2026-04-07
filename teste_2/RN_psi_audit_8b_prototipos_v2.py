"""
Auditoria 8b: Protótipos v2

Problema do v1: campo satura (~1860 cristais), todos protótipos ficam iguais.

Solução: ao invés de acumular no mesmo campo,
  1. Gera crystal_map individual para cada exemplo
  2. Faz a MÉDIA dos crystal_maps da mesma classe → protótipo
  3. Subtrai o protótipo médio global (o que é comum a todas classes)
  4. Classifica por correlação com protótipo diferencial

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
    """Computa crystal_maps individuais em batch."""
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
        out.append(mem.crystal_map)  # (B, FS, FS)
    return torch.cat(out, dim=0)


# ── MNIST ────────────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

# Organiza por classe
train_by_class = {i: [] for i in range(10)}
for img, label in train_ds:
    train_by_class[label].append(img.squeeze(0))

test_by_class = {i: [] for i in range(10)}
all_test_imgs = []
all_test_labels = []
for img, label in test_ds:
    all_test_imgs.append(img.squeeze(0))
    all_test_labels.append(label)
all_test_labels = np.array(all_test_labels)

PG = build_gaussians()

# ── Experimento ──────────────────────────────────────────────────────────────

N_PROTO_LIST = [500, 1000, 2000, 5000]
N_TEST = 500

print(f"\n{'='*70}")
print("AUDITORIA 8b: Protótipos v2 — Média + Subtração")
print(f"Teste: {N_TEST} imagens | Método: correlação diferencial")
print(f"{'='*70}")

# Subset de teste balanceado
test_subset_idx = []
counts = [0] * 10
for i, label in enumerate(all_test_labels):
    if counts[label] < N_TEST // 10:
        test_subset_idx.append(i)
        counts[label] += 1
    if all(c >= N_TEST // 10 for c in counts):
        break

# Pré-computa crystal_maps de teste
print(f"\nPré-computando crystal_maps de teste ({N_TEST} imgs)...")
test_imgs_tensor = torch.stack([all_test_imgs[i] for i in test_subset_idx]).to(DEVICE)
test_labels_sub = all_test_labels[test_subset_idx]
t1 = time.time()
test_cmaps = compute_crystal_maps_batch(test_imgs_tensor, PG)
print(f"  Pronto: {time.time()-t1:.0f}s")

all_results = []
t0 = time.time()

for n_proto in N_PROTO_LIST:
    print(f"\n── {n_proto} exemplos por protótipo ──")

    # Computa crystal_maps individuais para cada classe
    print(f"  Computando crystal_maps de treino...")
    t1 = time.time()
    class_cmaps = {}
    for cls in range(10):
        imgs = torch.stack(train_by_class[cls][:n_proto]).to(DEVICE)
        cmaps = compute_crystal_maps_batch(imgs, PG)
        class_cmaps[cls] = cmaps
    print(f"  Crystal_maps prontos: {time.time()-t1:.0f}s")

    # Protótipo = MÉDIA dos crystal_maps da classe
    prototypes = {}
    for cls in range(10):
        prototypes[cls] = class_cmaps[cls].mean(dim=0)  # (FS, FS)

    # Protótipo médio global
    global_proto = torch.stack([prototypes[cls] for cls in range(10)]).mean(dim=0)

    # Protótipos diferenciais = protótipo - média global
    diff_protos = {}
    for cls in range(10):
        diff_protos[cls] = prototypes[cls] - global_proto

    # Stats
    for cls in range(10):
        n_crys = (prototypes[cls] > 0.01).float().sum().item()
        n_diff = (diff_protos[cls].abs() > 0.01).float().sum().item()
        print(f"    Classe {cls}: {n_crys:.0f} cristais, {n_diff:.0f} diferenciais")

    # ── Classificação ────────────────────────────────────────────────────

    # Método 1: Correlação direta (sem subtração)
    correct_direct = 0
    for i in range(len(test_subset_idx)):
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
        if best_cls == test_labels_sub[i]:
            correct_direct += 1
    acc_direct = correct_direct / len(test_subset_idx) * 100

    # Método 2: Correlação diferencial (com subtração da média)
    correct_diff = 0
    for i in range(len(test_subset_idx)):
        cmap = test_cmaps[i] - global_proto  # subtrai média global
        cmap_f = cmap.flatten()
        cmap_n = cmap_f / (cmap_f.norm() + 1e-8)
        best_cls, best_sim = -1, -999
        for cls in range(10):
            p = diff_protos[cls].flatten()
            p_n = p / (p.norm() + 1e-8)
            sim = (cmap_n * p_n).sum().item()
            if sim > best_sim:
                best_sim = sim
                best_cls = cls
        if best_cls == test_labels_sub[i]:
            correct_diff += 1
    acc_diff = correct_diff / len(test_subset_idx) * 100

    # Método 3: Distância euclidiana ao protótipo
    correct_dist = 0
    for i in range(len(test_subset_idx)):
        cmap = test_cmaps[i].flatten()
        best_cls, best_dist = -1, float('inf')
        for cls in range(10):
            p = prototypes[cls].flatten()
            dist = ((cmap - p)**2).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_cls = cls
        if best_cls == test_labels_sub[i]:
            correct_dist += 1
    acc_dist = correct_dist / len(test_subset_idx) * 100

    all_results.append({
        'n_proto': n_proto,
        'acc_direct': acc_direct,
        'acc_diff': acc_diff,
        'acc_dist': acc_dist,
    })

    print(f"  → Correlação direta: {acc_direct:.1f}%")
    print(f"  → Correlação diferencial: {acc_diff:.1f}%")
    print(f"  → Distância euclidiana: {acc_dist:.1f}%")


# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESULTADO FINAL")
print(f"{'='*70}")
print(f"\n{'N exemplos':>12} {'Corr direta':>13} {'Corr diff':>11} {'Dist euclid':>13}")
print("-"*55)
for r in all_results:
    print(f"  {r['n_proto']:>10}   {r['acc_direct']:>8.1f}%   {r['acc_diff']:>7.1f}%   {r['acc_dist']:>9.1f}%")

print(f"\nReferência: aleatório = 10.0%")
print(f"Referência: crystal competitivo + linear decoder = 88.1%")
print(f"\nTempo total: {time.time()-t0:.0f}s")


# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Auditoria 8b — Protótipos v2: Média + Subtração', fontsize=13)

ns = [r['n_proto'] for r in all_results]

ax = axes[0]
ax.plot(ns, [r['acc_direct'] for r in all_results], 'o-', color='#e6194b',
        label='Correlação direta', linewidth=2, markersize=8)
ax.plot(ns, [r['acc_diff'] for r in all_results], 's-', color='#3cb44b',
        label='Correlação diferencial', linewidth=2, markersize=8)
ax.plot(ns, [r['acc_dist'] for r in all_results], '^-', color='#4363d8',
        label='Distância euclidiana', linewidth=2, markersize=8)
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Aleatório (10%)')
ax.axhline(y=88.1, color='orange', linestyle='--', alpha=0.5, label='Compet.+Linear (88.1%)')
ax.set_xlabel('Exemplos por protótipo')
ax.set_ylabel('Acurácia (%)')
ax.set_title('Acurácia vs N exemplos')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Protótipos diferenciais do último experimento
ax = axes[1]
ax.set_title('Protótipos diferenciais (último N)')
ax.axis('off')
for i in range(10):
    row, col = i // 5, i % 5
    sub = fig.add_axes([0.55 + col*0.085, 0.55 - row*0.4, 0.08, 0.35])
    proto_diff = diff_protos[i].cpu().numpy()
    vmax = np.abs(proto_diff).max()
    sub.imshow(proto_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    sub.set_title(f'{i}', fontsize=8)
    sub.axis('off')

plt.savefig('viz_audit_8b_prototipos_v2.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_8b_prototipos_v2.png")
print("Pronto.")
