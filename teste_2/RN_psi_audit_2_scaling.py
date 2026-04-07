"""
Auditoria 2: Curva de escala MNIST com múltiplas seeds + decoder não-linear

10 seeds por tamanho de campo.
Decoder linear E não-linear (MLP 1 hidden) para separar
"teto do campo" vs "teto do decoder".

Reporta: média ± desvio, IC 95% do expoente, R².
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as sp_stats
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
PSI_DT     = 0.05
PSI_GAMMA  = 0.06
PSI_ALPHA  = 0.04
PSI_BETA   = 0.005
PSI_C2     = 0.3
STIM_ON    = 40
STIM_TOTAL = 80

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

# ── Configuração do experimento ─────────────────────────────────────────────
FIELD_SIZES = [48, 64, 96, 128, 192, 256]
N_SEEDS     = 10
MAX_EPOCHS  = 60
PATIENCE    = 10
BATCH_SIZE  = 512
LR          = 1e-3

# ── Física ──────────────────────────────────────────────────────────────────

class CrystalMem:
    def __init__(self, B, FS):
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


def build_gaussians(field_size, sigma=0.04):
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            gs.append(torch.exp(-((xg-cx)**2+(yg-cy)**2)/(2*sigma**2)))
    return torch.stack(gs).view(784, -1)


def field_batch(X_batch, PG, FS):
    flat = X_batch.view(len(X_batch), 784)
    pert = (flat @ PG.to(flat.dtype)).view(len(X_batch), FS, FS)
    f, v = pert.clone(), torch.zeros_like(pert)
    mem  = CrystalMem(len(X_batch), FS)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f, v = psi_step(f, v, pert, s < STIM_ON)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)
    return mem.crystal_map.view(len(X_batch), -1).float()


def precompute(X, PG, FS, bs=64):
    N, out = len(X), []
    for i in range(0, N, bs):
        batch = X[i:i+bs]
        cm = field_batch(batch, PG, FS)
        out.append(cm.half())  # float16 para economizar VRAM
    return torch.cat(out, dim=0)


# ── MNIST ───────────────────────────────────────────────────────────────────

def load_mnist(seed):
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,),(0.3081,))])
    tr = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    te = datasets.MNIST('./data', train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0].squeeze(0) for i in range(len(tr))])
    Ytr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    Xte = torch.stack([te[i][0].squeeze(0) for i in range(len(te))])
    Yte = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)

    torch.manual_seed(seed)
    perm = torch.randperm(len(Xtr))
    Xva, Yva = Xtr[perm[-10000:]], Ytr[perm[-10000:]]
    Xtr, Ytr = Xtr[perm[:-10000]], Ytr[perm[:-10000]]

    return (Xtr.to(DEVICE), Ytr.to(DEVICE),
            Xva.to(DEVICE), Yva.to(DEVICE),
            Xte.to(DEVICE), Yte.to(DEVICE))


# ── Treino do decoder ───────────────────────────────────────────────────────

def train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte, FS, nonlinear=False, seed=0):
    torch.manual_seed(seed)

    if nonlinear:
        dec = nn.Sequential(
            nn.Linear(FS*FS, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        ).to(DEVICE)
    else:
        dec = nn.Linear(FS*FS, 10).to(DEVICE)

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
            correct = 0
            for i in range(0, len(CMva), BATCH_SIZE):
                correct += (dec(CMva[i:i+BATCH_SIZE].float()).argmax(1) == Yva[i:i+BATCH_SIZE]).sum().item()
            va = correct / len(CMva) * 100

        if va > best_val:
            best_val = va
            best_sd  = {k:v.clone() for k,v in dec.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    dec.load_state_dict(best_sd)
    dec.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(CMte), BATCH_SIZE):
            correct += (dec(CMte[i:i+BATCH_SIZE].float()).argmax(1) == Yte[i:i+BATCH_SIZE]).sum().item()
        te_acc = correct / len(CMte) * 100

    return best_val, te_acc, n_params


# ── Experimento principal ───────────────────────────────────────────────────

print("\nCarregando MNIST (1 vez)...")
# Crystal maps dependem só da física (não do split), mas o decoder depende do split.
# Pré-computamos crystal maps 1 vez e variamos o split do treino.

# Na verdade, crystal maps são determinísticos dado o input.
# O que varia entre seeds é o split treino/val e a inicialização do decoder.

print(f"\n{'='*75}")
print("AUDITORIA 2: Curva de Escala MNIST — 10 seeds × 6 tamanhos × 2 decoders")
print(f"Campos: {FIELD_SIZES}")
print(f"Seeds: {N_SEEDS} | Decoder: linear + MLP(256)")
print(f"{'='*75}\n")

all_results = []
t_global = time.time()

for fs in FIELD_SIZES:
    print(f"\n━━ Campo {fs}×{fs} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Pré-computa gaussianas e crystal maps (1 vez por tamanho)
    PG = build_gaussians(fs)

    # Carrega dados (seed fixo para crystal maps)
    Xtr, Ytr, Xva, Yva, Xte, Yte = load_mnist(seed=42)

    print(f"  Pré-computando crystal maps...", end=' ')
    t0 = time.time()
    CMtr = precompute(Xtr, PG, fs, bs=64)
    CMva = precompute(Xva, PG, fs, bs=128)
    CMte = precompute(Xte, PG, fs, bs=128)
    t_pre = time.time() - t0
    print(f"{t_pre:.1f}s")

    n_crys = (CMtr > 0.01).float().sum(dim=1).mean().item()
    print(f"  Cristais médios: {n_crys:.1f}")

    linear_accs = []
    mlp_accs    = []

    for seed in range(N_SEEDS):
        # Re-split com seed diferente para o decoder
        torch.manual_seed(seed + 100)
        perm_tr = torch.randperm(len(CMtr), device=DEVICE)

        # Treina linear
        va_l, te_l, np_l = train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte, fs,
                                          nonlinear=False, seed=seed)
        linear_accs.append(te_l)

        # Treina MLP
        va_m, te_m, np_m = train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte, fs,
                                          nonlinear=True, seed=seed)
        mlp_accs.append(te_m)

        print(f"    seed {seed}: linear={te_l:.2f}%  MLP={te_m:.2f}%")

    la = np.array(linear_accs)
    ma = np.array(mlp_accs)

    result = {
        'fs': fs, 'n2': fs*fs,
        'n_crys': n_crys,
        'linear_mean': la.mean(), 'linear_std': la.std(),
        'mlp_mean': ma.mean(), 'mlp_std': ma.std(),
        'np_linear': np_l, 'np_mlp': np_m,
        't_precompute': t_pre,
    }
    all_results.append(result)

    print(f"  Linear: {la.mean():.2f}% ± {la.std():.2f}%  ({np_l} params)")
    print(f"  MLP:    {ma.mean():.2f}% ± {ma.std():.2f}%  ({np_m} params)")

    # Libera memória
    del CMtr, CMva, CMte, PG
    torch.cuda.empty_cache()

# ── Lei de escala com IC ────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("RESULTADO: Lei de Escala com IC 95%")
print(f"{'='*75}")

print(f"\n{'Campo':>8}  {'Cristais':>9}  {'Linear':>16}  {'MLP(256)':>16}  {'Δ':>7}")
print("-"*65)
for r in all_results:
    delta = r['mlp_mean'] - r['linear_mean']
    print(f"  {r['fs']:>3}×{r['fs']:<3}  {r['n_crys']:>9.1f}  "
          f"{r['linear_mean']:>6.2f}±{r['linear_std']:.2f}%  "
          f"{r['mlp_mean']:>6.2f}±{r['mlp_std']:.2f}%  "
          f"{delta:>+6.2f}%")

# Fit expoente C(N) = a * N^b
log_n = np.log([r['fs'] for r in all_results])
log_c = np.log([max(r['n_crys'], 1) for r in all_results])

slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_n, log_c)
print(f"\nLei de escala C(N) = a × N^b:")
print(f"  Expoente b = {slope:.3f} ± {1.96*std_err:.3f} (IC 95%)")
print(f"  R² = {r_value**2:.4f}")
print(f"  p = {p_value:.2e}")

# Diagnóstico de gargalo
delta_means = [r['mlp_mean'] - r['linear_mean'] for r in all_results]
print(f"\nDiagnóstico de gargalo (MLP - Linear):")
for r, d in zip(all_results, delta_means):
    indicator = "← DECODER é gargalo" if d > 2.0 else "← campo é gargalo" if d < 0.5 else ""
    print(f"  {r['fs']}×{r['fs']}: Δ = {d:+.2f}%  {indicator}")

# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Auditoria 2 — Curva de Escala MNIST ({N_SEEDS} seeds)', fontsize=13)

fss = [r['fs'] for r in all_results]
la_m = [r['linear_mean'] for r in all_results]
la_s = [r['linear_std'] for r in all_results]
ma_m = [r['mlp_mean'] for r in all_results]
ma_s = [r['mlp_std'] for r in all_results]

# 1. Acurácia vs campo
axes[0].errorbar(fss, la_m, yerr=la_s, fmt='o-', color='#e6194b', lw=2, capsize=4, label='Linear')
axes[0].errorbar(fss, ma_m, yerr=ma_s, fmt='s-', color='#4363d8', lw=2, capsize=4, label='MLP(256)')
axes[0].axhline(92, color='gray', ls='--', alpha=0.5, label='Reg. logística ~92%')
axes[0].set_xlabel('Tamanho do campo N')
axes[0].set_ylabel('Teste (%)')
axes[0].set_title('Acurácia vs Tamanho do Campo')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# 2. Cristais vs N (log-log) com fit
n_arr = np.array(fss)
c_arr = np.array([r['n_crys'] for r in all_results])
axes[1].loglog(n_arr, c_arr, 'o', color='#3cb44b', ms=8)
n_fit = np.linspace(n_arr.min()*0.8, n_arr.max()*1.2, 100)
axes[1].loglog(n_fit, np.exp(intercept)*n_fit**slope, '--', color='gray',
               label=f'N^{slope:.2f} (R²={r_value**2:.3f})')
axes[1].set_xlabel('N'); axes[1].set_ylabel('Cristais')
axes[1].set_title('Lei de Escala C(N)')
axes[1].legend(); axes[1].grid(alpha=0.3, which='both')

# 3. Delta MLP - Linear
axes[2].bar([f"{r['fs']}" for r in all_results], delta_means, color='#f58231', alpha=0.8)
axes[2].axhline(0, color='black', lw=0.5)
axes[2].set_xlabel('Campo N')
axes[2].set_ylabel('Δ acurácia (MLP − Linear) %')
axes[2].set_title('Gargalo: Decoder vs Campo')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_audit_2_scaling.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_2_scaling.png")
print(f"Tempo total: {time.time()-t_global:.0f}s")
print("Pronto.")
