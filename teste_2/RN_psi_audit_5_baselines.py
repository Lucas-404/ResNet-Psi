"""
Auditoria 5: Baselines MNIST — comparação justa

Mesmo split, mesma métrica, 10 seeds.
Modelos com número de parâmetros comparável ao ResNet-Ψ.

1. Linear 784→10                     (7.850 params)
2. ResNet-Ψ 48×48 + decoder linear   (23.050 params)  ← re-usa crystal maps
3. MLP 784→128→10                    (101.770 params)
4. MLP 784→256→10                    (203.530 params)
5. CNN pequena (params ~ ResNet-Ψ)   (~25k params)

Todas com 10 seeds, early stopping, mesmo scheduler.
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

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ── Hiperparâmetros comuns ──────────────────────────────────────────────────
BATCH_SIZE = 512
LR         = 1e-3
MAX_EPOCHS = 60
PATIENCE   = 10
N_SEEDS    = 10

# ── Constantes do campo (para ResNet-Ψ) ────────────────────────────────────
PSI_DT     = 0.05
PSI_GAMMA  = 0.06
PSI_ALPHA  = 0.04
PSI_BETA   = 0.005
PSI_C2     = 0.3
STIM_ON    = 40
STIM_TOTAL = 80
FIELD_SIZE = 48

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


# ── Física do campo ────────────────────────────────────────────────────────

class CrystalMem:
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


def precompute_crystal_maps(X, PG, bs=64):
    N, out = len(X), []
    for i in range(0, N, bs):
        B = X[i:i+bs]
        pert = (B.view(len(B), 784) @ PG.to(B.dtype)).view(len(B), FIELD_SIZE, FIELD_SIZE)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalMem(len(B))
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)
        out.append(mem.crystal_map.view(len(B), -1).half())
    return torch.cat(out, dim=0)


# ── MNIST ───────────────────────────────────────────────────────────────────

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


# ── Treino genérico ────────────────────────────────────────────────────────

def train_model(model, Xtr, Ytr, Xva, Yva, Xte, Yte, seed=0, flat_input=True):
    """Treina modelo com early stopping. Retorna (val_acc, test_acc)."""
    torch.manual_seed(seed)
    # Re-init pesos
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.to(DEVICE)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()

    best_val, best_sd, pat = 0.0, None, 0

    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            x = Xtr[idx]
            if flat_input:
                x = x.view(len(idx), -1).float()
            opt.zero_grad(set_to_none=True)
            crit(model(x), Ytr[idx]).backward()
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, len(Xva), 1024):
                x = Xva[i:i+1024]
                if flat_input:
                    x = x.view(len(x), -1).float()
                correct += (model(x).argmax(1) == Yva[i:i+1024]).sum().item()
            va = correct / len(Xva) * 100

        if va > best_val:
            best_val = va
            best_sd  = {k:v.clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(Xte), 1024):
            x = Xte[i:i+1024]
            if flat_input:
                x = x.view(len(x), -1).float()
            correct += (model(x).argmax(1) == Yte[i:i+1024]).sum().item()
        te_acc = correct / len(Xte) * 100

    return best_val, te_acc


# ── Modelos ─────────────────────────────────────────────────────────────────

def make_linear():
    return nn.Linear(784, 10)

def make_mlp_128():
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

def make_mlp_256():
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def make_small_cnn():
    """CNN com ~25k params para comparação justa com ResNet-Ψ."""
    return nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),
        nn.Conv2d(1, 8, 3, padding=1),    # (8, 28, 28)
        nn.ReLU(),
        nn.MaxPool2d(2),                    # (8, 14, 14)
        nn.Conv2d(8, 16, 3, padding=1),    # (16, 14, 14)
        nn.ReLU(),
        nn.MaxPool2d(2),                    # (16, 7, 7)
        nn.Flatten(),                       # (784)
        nn.Linear(16*7*7, 10),             # (10)
    )


# ── Experimento ─────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
Xtr, Ytr, Xva, Yva, Xte, Yte = load_mnist()

# Pré-computa crystal maps para ResNet-Ψ (1 vez — determinístico)
print("Pré-computando crystal maps (48×48)...")
PG = build_gaussians()
t0 = time.time()
CMtr = precompute_crystal_maps(Xtr, PG)
CMva = precompute_crystal_maps(Xva, PG)
CMte = precompute_crystal_maps(Xte, PG)
t_pre = time.time() - t0
print(f"  Pré-computação: {t_pre:.1f}s")
del PG

print(f"\n{'='*75}")
print(f"AUDITORIA 5: Baselines MNIST — {N_SEEDS} seeds cada")
print(f"{'='*75}")

# Definição dos modelos
models_def = [
    ("Linear 784→10",          make_linear,   True,  False),
    ("ResNet-Ψ 48² + Linear",  None,          True,  True),  # usa crystal maps
    ("MLP 784→128→10",         make_mlp_128,  True,  False),
    ("MLP 784→256→10",         make_mlp_256,  True,  False),
    ("CNN pequena (~25k)",     make_small_cnn, True,  False),
]

all_results = {}

for name, make_fn, flat, use_cmap in models_def:
    accs = []
    for seed in range(N_SEEDS):
        if use_cmap:
            # ResNet-Ψ: treina decoder linear sobre crystal maps
            dec = nn.Linear(FIELD_SIZE*FIELD_SIZE, 10)
            torch.manual_seed(seed)
            nn.init.xavier_normal_(dec.weight)
            nn.init.zeros_(dec.bias)

            # Treina sobre crystal maps
            dec.to(DEVICE)
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

            n_params = sum(p.numel() for p in dec.parameters())
        else:
            model = make_fn()
            n_params = sum(p.numel() for p in model.parameters())
            _, te_acc = train_model(model, Xtr, Ytr, Xva, Yva, Xte, Yte,
                                     seed=seed, flat_input=flat)

        accs.append(te_acc)
        print(f"  {name:>25}  seed {seed}: {te_acc:.2f}%")

    arr = np.array(accs)
    all_results[name] = {
        'mean': arr.mean(), 'std': arr.std(), 'n_params': n_params, 'accs': arr
    }
    print(f"  {'→':>25}  {arr.mean():.2f}% ± {arr.std():.2f}%  ({n_params} params)\n")


# ── Resumo ──────────────────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("RESULTADO FINAL: Comparação justa")
print(f"{'='*75}")
print(f"\n{'Modelo':>25}  {'Params':>8}  {'Teste':>14}  {'Eficiência':>12}")
print("-"*65)

for name, r in all_results.items():
    eff = r['mean'] / (r['n_params'] / 1000)  # % por k-params
    print(f"  {name:>25}  {r['n_params']:>8}  "
          f"{r['mean']:>6.2f}±{r['std']:.2f}%  "
          f"{eff:>8.2f} %/kp")

# Significância estatística: ResNet-Ψ vs Linear
if "Linear 784→10" in all_results and "ResNet-Ψ 48² + Linear" in all_results:
    from scipy import stats as sp_stats
    a = all_results["Linear 784→10"]['accs']
    b = all_results["ResNet-Ψ 48² + Linear"]['accs']
    t_stat, p_val = sp_stats.ttest_ind(a, b)
    print(f"\n  Linear vs ResNet-Ψ: t={t_stat:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  → Diferença SIGNIFICATIVA (p < 0.05)")
    else:
        print(f"  → Diferença NÃO significativa (p = {p_val:.4f})")

# ── Plot ────────────────────────────────────────────────────────────────────

names   = list(all_results.keys())
means   = [all_results[n]['mean'] for n in names]
stds    = [all_results[n]['std']  for n in names]
params  = [all_results[n]['n_params'] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Auditoria 5 — Baselines MNIST ({N_SEEDS} seeds)', fontsize=13)

# 1. Acurácia
colors = ['#a9a9a9', '#e6194b', '#3cb44b', '#4363d8', '#f58231']
axes[0].barh(names, means, xerr=stds, color=colors, capsize=4, alpha=0.8)
axes[0].set_xlabel('Teste (%)')
axes[0].set_title('Acurácia no Teste')
axes[0].grid(alpha=0.3, axis='x')
for i, (m, s) in enumerate(zip(means, stds)):
    axes[0].text(m + s + 0.3, i, f"{m:.1f}%", va='center', fontsize=9)

# 2. Eficiência (acc / k-params)
effs = [m / (p/1000) for m, p in zip(means, params)]
axes[1].barh(names, effs, color=colors, alpha=0.8)
axes[1].set_xlabel('Acurácia / kilo-parâmetros')
axes[1].set_title('Eficiência de Parâmetros')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('viz_audit_5_baselines.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_5_baselines.png")
print("Pronto.")
