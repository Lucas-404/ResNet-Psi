"""
Auditoria 7: Cristalização v2 — Suave + Competição

Compara 3 variantes de cristalização:
  1. ORIGINAL: thresholds duros (amplitude > 0.3, CV < 0.15)
  2. SUAVE: sigmoid ao invés de step function
  3. SUAVE + COMPETIÇÃO: cristais decaem se não ressoam com a onda

Mede: MI, memória, densidade, acurácia com decoder linear e MLP.
2000 amostras para MI, full MNIST para acurácia.
"""

import torch
import torch.nn as nn
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
from sklearn.metrics import mutual_info_score

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ── Constantes físicas ────────────────────────────────────────────────────
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

BATCH_SIZE = 512
LR         = 1e-3
MAX_EPOCHS = 60
PATIENCE   = 10
N_SEEDS    = 5


# ── Variante 1: Cristalização ORIGINAL ──────────────────────────────────────

class CrystalOriginal:
    """Cristalização original com thresholds duros."""
    def __init__(self, B, FS=FIELD_SIZE):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
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
        # Thresholds DUROS
        cand = ((mean > CRYSTAL_A_MIN) & (cv < CRYSTAL_CV_MAX) & (mean < 8.0)).float()
        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0,1)
        self.crystal_map = torch.clamp(
            self.crystal_map + cand*(1.0-occ)*field.abs(), 0, 10.)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


# ── Variante 2: Cristalização SUAVE ─────────────────────────────────────────

class CrystalSuave:
    """Cristalização com sigmoid ao invés de thresholds duros."""
    def __init__(self, B, FS=FIELD_SIZE, sharpness=10.0):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
        self.sharpness = sharpness
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
        # Thresholds SUAVES via sigmoid
        amp_score = torch.sigmoid(self.sharpness * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(self.sharpness * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(self.sharpness * (8.0 - mean))
        cand = amp_score * cv_score * sat_score

        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0,1)
        self.crystal_map = torch.clamp(
            self.crystal_map + cand*(1.0-occ)*field.abs(), 0, 10.)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


# ── Variante 3: Cristalização SUAVE + COMPETIÇÃO ────────────────────────────

class CrystalCompetitivo:
    """
    Cristalização suave + competição:
    - Cristais ganham "vida" quando ressoam com a onda
    - Cristais perdem "vida" quando não ressoam
    - Cristais com vida <= 0 morrem (são removidos)
    """
    def __init__(self, B, FS=FIELD_SIZE, sharpness=10.0, decay=0.02, ressonance_boost=0.1):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(B, FS, FS, device=DEVICE)  # "vida" dos cristais
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

        # Thresholds SUAVES
        amp_score = torch.sigmoid(self.sharpness * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(self.sharpness * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(self.sharpness * (8.0 - mean))
        cand = amp_score * cv_score * sat_score

        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0,1)

        # Novos cristais
        new_crystals = cand * (1.0 - occ) * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + new_crystals, 0, 10.)

        # Atualiza HP: novos cristais nascem com HP = 1
        self.crystal_hp = torch.where(
            new_crystals > 0.01,
            torch.clamp(self.crystal_hp + 1.0, 0, 5.0),
            self.crystal_hp
        )

        # COMPETIÇÃO: ressonância = onda forte onde há cristal
        ressonance = field.abs() * (self.crystal_map > 0.01).float()
        # Cristais que ressoam ganham vida
        self.crystal_hp = self.crystal_hp + ressonance * self.ressonance_boost
        # Todos decaem um pouco
        self.crystal_hp = self.crystal_hp - self.decay
        # Cristais com HP <= 0 morrem
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


def precompute_fields(X, PG, crystal_class, bs=64, **kwargs):
    """Pré-computa crystal maps para qualquer variante de cristalização."""
    N, out = len(X), []
    t0 = time.time()
    for i in range(0, N, bs):
        B = X[i:i+bs]
        pert = (B.view(len(B), 784) @ PG.to(B.dtype)).view(len(B), FIELD_SIZE, FIELD_SIZE)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = crystal_class(len(B), FIELD_SIZE, **kwargs)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)

        out.append(mem.crystal_map.view(len(B), -1).half())

        if (i//bs) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    {min(i+bs,N)}/{N} ({min(i+bs,N)/N*100:.0f}%)  {elapsed:.0f}s", end='\r')

    print(f"    {N}/{N} (100%)  {time.time()-t0:.1f}s      ")
    return torch.cat(out, dim=0)


# ── MI ───────────────────────────────────────────────────────────────────────

def compute_mi(representations, labels, n_clusters_list=[8, 16, 32, 64, 128]):
    X = representations
    best_mi = 0.0
    for k in n_clusters_list:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=200)
        clusters = km.fit_predict(X)
        mi = mutual_info_score(labels, clusters)
        mi_bits = mi / np.log(2)
        if mi_bits > best_mi:
            best_mi = mi_bits
    return best_mi


def count_nonzero(reps, threshold=0.01):
    nonzero = np.abs(reps) > threshold
    n_values = nonzero.sum(axis=1).mean()
    mem_kb = n_values * 4 / 1024
    return n_values, mem_kb


# ── Treino decoder ───────────────────────────────────────────────────────────

def train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte, input_dim, nonlinear=False, seed=0):
    torch.manual_seed(seed)
    if nonlinear:
        dec = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 10),
        ).to(DEVICE)
    else:
        dec = nn.Linear(input_dim, 10).to(DEVICE)

    n_params = sum(p.numel() for p in dec.parameters())
    opt  = torch.optim.AdamW(dec.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()

    best_val, best_sd, pat = 0.0, None, 0
    for ep in range(1, MAX_EPOCHS+1):
        dec.train()
        perm = torch.randperm(len(CMtr), device=DEVICE)
        for i in range(0, len(CMtr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            crit(dec(CMtr[idx].float()), Ytr[idx]).backward()
            opt.step()
        sch.step()
        dec.eval()
        with torch.no_grad():
            c = 0
            for i in range(0, len(CMva), 1024):
                c += (dec(CMva[i:i+1024].float()).argmax(1) == Yva[i:i+1024]).sum().item()
            va = c / len(CMva) * 100
        if va > best_val:
            best_val = va
            best_sd = {k:v.clone() for k,v in dec.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    dec.load_state_dict(best_sd)
    dec.eval()
    with torch.no_grad():
        c = 0
        for i in range(0, len(CMte), 1024):
            c += (dec(CMte[i:i+1024].float()).argmax(1) == Yte[i:i+1024]).sum().item()
        te_acc = c / len(CMte) * 100

    return te_acc, n_params


# ── MNIST ────────────────────────────────────────────────────────────────────

def load_mnist():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,),(0.3081,))])
    tr = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    te = datasets.MNIST('./data', train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0].squeeze(0) for i in range(len(tr))])
    Ytr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    Xte = torch.stack([te[i][0].squeeze(0) for i in range(len(te))])
    Yte = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)
    torch.manual_seed(42)
    perm = torch.randperm(len(Xtr))
    Xva, Yva = Xtr[perm[-10000:]], Ytr[perm[-10000:]]
    Xtr, Ytr = Xtr[perm[:-10000]], Ytr[perm[:-10000]]
    return (Xtr.to(DEVICE), Ytr.to(DEVICE),
            Xva.to(DEVICE), Yva.to(DEVICE),
            Xte.to(DEVICE), Yte.to(DEVICE))


# ── Experimento ──────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
Xtr, Ytr, Xva, Yva, Xte, Yte = load_mnist()
print(f"Treino: {len(Xtr)} | Val: {len(Xva)} | Teste: {len(Xte)}")

PG = build_gaussians()

# Variantes a testar
variants = [
    ("Original (duros)",  CrystalOriginal,   {}),
    ("Suave (sigmoid)",   CrystalSuave,      {'sharpness': 5.0}),
    ("Competitivo",       CrystalCompetitivo, {'sharpness': 5.0, 'decay': 0.02, 'ressonance_boost': 0.1}),
]

input_dim = FIELD_SIZE * FIELD_SIZE

print(f"\n{'='*80}")
print("AUDITORIA 7: Cristalização v2 — Suave + Competição")
print(f"{'='*80}")

all_results = {}
t0 = time.time()

for vname, vclass, vkwargs in variants:
    print(f"\n── {vname} ──")

    # Pré-computa
    print("  Pré-computando treino...")
    CMtr = precompute_fields(Xtr, PG, vclass, **vkwargs)
    print("  Pré-computando validação...")
    CMva = precompute_fields(Xva, PG, vclass, **vkwargs)
    print("  Pré-computando teste...")
    CMte = precompute_fields(Xte, PG, vclass, **vkwargs)

    # Stats
    n_crys = (CMtr > 0.01).float().sum(dim=1).mean().item()
    reps_np = CMte[:2000].float().cpu().numpy()
    labels_np = Yte[:2000].cpu().numpy()
    n_vals, mem_kb = count_nonzero(reps_np)

    # MI
    mi = compute_mi(reps_np, labels_np)
    density = mi / mem_kb if mem_kb > 0 else 0

    print(f"  Cristais médios: {n_crys:.1f}")
    print(f"  MI: {mi:.2f} bits | Mem: {mem_kb:.1f} KB | Densidade: {density:.3f} bits/KB")

    # Acurácia com decoder
    accs_linear = []
    accs_mlp = []
    for seed in range(N_SEEDS):
        acc_l, np_l = train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte,
                                     input_dim, nonlinear=False, seed=seed)
        acc_m, np_m = train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte,
                                     input_dim, nonlinear=True, seed=seed)
        accs_linear.append(acc_l)
        accs_mlp.append(acc_m)
        print(f"    seed {seed}: linear={acc_l:.2f}%  MLP={acc_m:.2f}%")

    arr_l = np.array(accs_linear)
    arr_m = np.array(accs_mlp)

    all_results[vname] = {
        'n_crys': n_crys, 'mi': mi, 'mem_kb': mem_kb, 'density': density,
        'linear_mean': arr_l.mean(), 'linear_std': arr_l.std(),
        'mlp_mean': arr_m.mean(), 'mlp_std': arr_m.std(),
        'n_values': n_vals,
    }

    print(f"  Linear: {arr_l.mean():.2f}% ± {arr_l.std():.2f}%")
    print(f"  MLP:    {arr_m.mean():.2f}% ± {arr_m.std():.2f}%")

    # Libera memória
    del CMtr, CMva, CMte
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*95}")
print("RESULTADO FINAL: Comparação de Cristalizações")
print(f"{'='*95}")
print(f"\n{'Variante':<22} {'Cristais':>9} {'MI bits':>8} {'Mem KB':>8} "
      f"{'bits/KB':>8} {'Linear':>10} {'MLP':>10}")
print("-"*85)

for vname in [v[0] for v in variants]:
    r = all_results[vname]
    print(f"  {vname:<22} {r['n_crys']:>7.0f}   {r['mi']:>6.2f}   {r['mem_kb']:>6.1f}   "
          f"{r['density']:>6.3f}   {r['linear_mean']:>5.2f}±{r['linear_std']:.2f}   "
          f"{r['mlp_mean']:>5.2f}±{r['mlp_std']:.2f}")

# ── Melhor variante ──────────────────────────────────────────────────────────

best_density = max(all_results.items(), key=lambda x: x[1]['density'])
best_linear  = max(all_results.items(), key=lambda x: x[1]['linear_mean'])
best_mlp     = max(all_results.items(), key=lambda x: x[1]['mlp_mean'])
best_mi      = max(all_results.items(), key=lambda x: x[1]['mi'])

print(f"\n  Melhor densidade: {best_density[0]} ({best_density[1]['density']:.3f} bits/KB)")
print(f"  Melhor MI:        {best_mi[0]} ({best_mi[1]['mi']:.2f} bits)")
print(f"  Melhor linear:    {best_linear[0]} ({best_linear[1]['linear_mean']:.2f}%)")
print(f"  Melhor MLP:       {best_mlp[0]} ({best_mlp[1]['mlp_mean']:.2f}%)")

print(f"\nTempo total: {time.time()-t0:.0f}s")


# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('Auditoria 7 — Cristalização v2: Original vs Suave vs Competição', fontsize=13)

names = [v[0] for v in variants]
colors = ['#e6194b', '#f58231', '#3cb44b']

# 1. Acurácia
ax = axes[0]
linear_means = [all_results[n]['linear_mean'] for n in names]
mlp_means    = [all_results[n]['mlp_mean'] for n in names]
x = np.arange(len(names))
w = 0.35
ax.barh(x - w/2, linear_means, w, color=[c for c in colors], alpha=0.6, label='Linear')
ax.barh(x + w/2, mlp_means, w, color=[c for c in colors], alpha=1.0, label='MLP')
ax.set_yticks(x)
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Teste (%)')
ax.set_title('Acurácia')
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis='x')

# 2. MI
ax = axes[1]
mis = [all_results[n]['mi'] for n in names]
ax.barh(range(len(names)), mis, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('MI (bits)')
ax.set_title('Informação Mútua')
ax.grid(alpha=0.3, axis='x')

# 3. Densidade
ax = axes[2]
dens = [all_results[n]['density'] for n in names]
ax.barh(range(len(names)), dens, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('bits/KB')
ax.set_title('Densidade Informacional')
ax.grid(alpha=0.3, axis='x')

# 4. Cristais vs MI
ax = axes[3]
for i, n in enumerate(names):
    r = all_results[n]
    ax.scatter(r['n_crys'], r['mi'], color=colors[i], s=100,
              label=n, zorder=5, edgecolors='black', linewidth=0.5)
ax.set_xlabel('N° Cristais')
ax.set_ylabel('MI (bits)')
ax.set_title('Cristais vs MI')
ax.legend(fontsize=6)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_audit_7_cristalizacao_v2.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_7_cristalizacao_v2.png")
print("Pronto.")
