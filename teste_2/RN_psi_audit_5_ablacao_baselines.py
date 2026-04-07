"""
Auditoria 5: Ablação de cristais + Baselines MNIST

A pergunta central: a cristalização adiciona capacidade ao campo?

Configurações testadas (10 seeds cada):
  1. ResNet-Ψ COM cristais + decoder linear     (contribuição dos cristais)
  2. ResNet-Ψ SEM cristais + decoder linear      (campo puro, sem cristalização)
  3. ResNet-Ψ COM cristais + decoder MLP(256)    (teto do campo com cristais)
  4. ResNet-Ψ SEM cristais + decoder MLP(256)    (teto do campo sem cristais)
  5. Linear 784→10                                (baseline mínimo)
  6. MLP 784→256→10                               (baseline comparável)
  7. CNN pequena (~25k params)                    (baseline com estrutura espacial)

Se (1) > (2): cristais são essenciais — contribuição central do paper.
Se (3) > (4): cristais ajudam mesmo com decoder forte.
Se (1) > (5): campo+cristais > regressão logística.
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

# ── Hiperparâmetros ─────────────────────────────────────────────────────────
BATCH_SIZE = 512
LR         = 1e-3
MAX_EPOCHS = 60
PATIENCE   = 10
N_SEEDS    = 10
FIELD_SIZE = 48

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


# ── Física ──────────────────────────────────────────────────────────────────

class CrystalMem:
    """Crystal memory com flag para desativar cristalização."""
    def __init__(self, B, FS=FIELD_SIZE, enable_crystals=True):
        self.enable = enable_crystals
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
        ks = 2 * CRYSTAL_SEP + 1
        self._dilate = torch.ones(1, 1, ks, ks, device=DEVICE)

    def update_envelope(self, field):
        if not self.enable:
            return
        self.window_max = torch.max(self.window_max, field.abs())
        self.window_step += 1
        if self.window_step >= CRYSTAL_W:
            self.env_buffer[:, self.window_idx] = self.window_max
            self.window_max  = torch.zeros_like(self.window_max)
            self.window_step = 0
            self.window_idx  = (self.window_idx + 1) % CRYSTAL_K

    def try_crystallize(self, field):
        if not self.enable:
            return
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
        if not self.enable:
            return field
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


def precompute_fields(X, PG, enable_crystals=True, use_field_state=False, bs=64):
    """
    Pré-computa representações do campo.

    enable_crystals=True:  retorna crystal_map (representação esparsa)
    enable_crystals=False, use_field_state=False: retorna crystal_map (vazio, só zeros)
    enable_crystals=False, use_field_state=True:  retorna campo final (estado da onda)
    """
    N, out = len(X), []
    t0 = time.time()
    for i in range(0, N, bs):
        B = X[i:i+bs]
        pert = (B.view(len(B), 784) @ PG.to(B.dtype)).view(len(B), FIELD_SIZE, FIELD_SIZE)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalMem(len(B), FIELD_SIZE, enable_crystals=enable_crystals)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)

        if use_field_state and not enable_crystals:
            # Sem cristais: usa o estado final do campo como representação
            out.append(f.view(len(B), -1).half())
        else:
            out.append(mem.crystal_map.view(len(B), -1).half())

        if (i//bs) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    {min(i+bs,N)}/{N} ({min(i+bs,N)/N*100:.0f}%)  {elapsed:.0f}s", end='\r')

    print(f"    {N}/{N} (100%)  {time.time()-t0:.1f}s      ")
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


# ── Treino genérico ─────────────────────────────────────────────────────────

def train_decoder_on_maps(CMtr, Ytr, CMva, Yva, CMte, Yte, input_dim, nonlinear=False, seed=0):
    """Treina decoder sobre representações pré-computadas."""
    torch.manual_seed(seed)

    if nonlinear:
        dec = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
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


def train_standard_model(model, Xtr, Ytr, Xva, Yva, Xte, Yte, seed=0):
    """Treina modelo convencional sobre pixels."""
    torch.manual_seed(seed)
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
            x = Xtr[idx].view(len(idx), -1).float()
            opt.zero_grad(set_to_none=True)
            crit(model(x), Ytr[idx]).backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            c = 0
            for i in range(0, len(Xva), 1024):
                x = Xva[i:i+1024].view(-1, 784).float()
                c += (model(x).argmax(1) == Yva[i:i+1024]).sum().item()
            va = c / len(Xva) * 100
        if va > best_val:
            best_val = va
            best_sd = {k:v.clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        c = 0
        for i in range(0, len(Xte), 1024):
            x = Xte[i:i+1024].view(-1, 784).float()
            c += (model(x).argmax(1) == Yte[i:i+1024]).sum().item()
        te_acc = c / len(Xte) * 100

    return te_acc, sum(p.numel() for p in model.parameters())


# ── Experimento ─────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
Xtr, Ytr, Xva, Yva, Xte, Yte = load_mnist()
print(f"Treino: {len(Xtr)} | Val: {len(Xva)} | Teste: {len(Xte)}")

PG = build_gaussians()

# ── Pré-computa os 3 tipos de representação do campo ────────────────────────

print("\n[1/3] Campo COM cristais...")
CM_crys_tr = precompute_fields(Xtr, PG, enable_crystals=True)
CM_crys_va = precompute_fields(Xva, PG, enable_crystals=True)
CM_crys_te = precompute_fields(Xte, PG, enable_crystals=True)
n_crys = (CM_crys_tr > 0.01).float().sum(dim=1).mean().item()
print(f"  Cristais médios: {n_crys:.1f}")

print("\n[2/3] Campo SEM cristais (estado final da onda)...")
CM_wave_tr = precompute_fields(Xtr, PG, enable_crystals=False, use_field_state=True)
CM_wave_va = precompute_fields(Xva, PG, enable_crystals=False, use_field_state=True)
CM_wave_te = precompute_fields(Xte, PG, enable_crystals=False, use_field_state=True)
n_nonzero = (CM_wave_tr.abs() > 0.01).float().sum(dim=1).mean().item()
print(f"  Posições ativas (campo): {n_nonzero:.1f}")

print("\n[3/3] Baselines prontos (treinam sobre pixels)")

del PG  # libera memória

# ── Treina tudo ─────────────────────────────────────────────────────────────

input_dim = FIELD_SIZE * FIELD_SIZE

print(f"\n{'='*75}")
print(f"AUDITORIA 5: Ablação de Cristais + Baselines — {N_SEEDS} seeds")
print(f"{'='*75}")

configs = [
    # (nome, tipo)
    # tipo: 'crystal_linear', 'crystal_mlp', 'wave_linear', 'wave_mlp', 'standard'
    ("Ψ+cristais+linear",  'crystal_linear'),
    ("Ψ+cristais+MLP",     'crystal_mlp'),
    ("Ψ-cristais+linear",  'wave_linear'),
    ("Ψ-cristais+MLP",     'wave_mlp'),
    ("Linear 784→10",      'linear'),
    ("MLP 784→256→10",     'mlp256'),
    ("CNN pequena",         'cnn'),
]

all_results = {}

for name, typ in configs:
    accs = []
    n_params = 0

    for seed in range(N_SEEDS):
        if typ == 'crystal_linear':
            acc, n_params = train_decoder_on_maps(
                CM_crys_tr, Ytr, CM_crys_va, Yva, CM_crys_te, Yte,
                input_dim, nonlinear=False, seed=seed)
        elif typ == 'crystal_mlp':
            acc, n_params = train_decoder_on_maps(
                CM_crys_tr, Ytr, CM_crys_va, Yva, CM_crys_te, Yte,
                input_dim, nonlinear=True, seed=seed)
        elif typ == 'wave_linear':
            acc, n_params = train_decoder_on_maps(
                CM_wave_tr, Ytr, CM_wave_va, Yva, CM_wave_te, Yte,
                input_dim, nonlinear=False, seed=seed)
        elif typ == 'wave_mlp':
            acc, n_params = train_decoder_on_maps(
                CM_wave_tr, Ytr, CM_wave_va, Yva, CM_wave_te, Yte,
                input_dim, nonlinear=True, seed=seed)
        elif typ == 'linear':
            model = nn.Linear(784, 10)
            acc, n_params = train_standard_model(
                model, Xtr, Ytr, Xva, Yva, Xte, Yte, seed=seed)
        elif typ == 'mlp256':
            model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
            acc, n_params = train_standard_model(
                model, Xtr, Ytr, Xva, Yva, Xte, Yte, seed=seed)
        elif typ == 'cnn':
            model = nn.Sequential(
                nn.Unflatten(1, (1, 28, 28)),
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16*7*7, 10),
            )
            acc, n_params = train_standard_model(
                model, Xtr, Ytr, Xva, Yva, Xte, Yte, seed=seed)

        accs.append(acc)
        print(f"  {name:>22}  seed {seed}: {acc:.2f}%")

    arr = np.array(accs)
    all_results[name] = {
        'mean': arr.mean(), 'std': arr.std(),
        'n_params': n_params, 'accs': arr, 'type': typ,
    }
    print(f"  {'→':>22}  {arr.mean():.2f}% ± {arr.std():.2f}%  ({n_params} params)\n")


# ── Análise de ablação ──────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("RESULTADO 1: Ablação de Cristais")
print(f"{'='*75}")

pairs = [
    ("Ψ+cristais+linear", "Ψ-cristais+linear", "Linear"),
    ("Ψ+cristais+MLP",    "Ψ-cristais+MLP",    "MLP"),
]

for with_name, without_name, dec_type in pairs:
    w  = all_results[with_name]
    wo = all_results[without_name]
    delta = w['mean'] - wo['mean']

    # t-test pareado
    from scipy import stats as sp_stats
    t_stat, p_val = sp_stats.ttest_rel(w['accs'], wo['accs'])

    print(f"\n  Decoder {dec_type}:")
    print(f"    COM cristais: {w['mean']:.2f}% ± {w['std']:.2f}%")
    print(f"    SEM cristais: {wo['mean']:.2f}% ± {wo['std']:.2f}%")
    print(f"    Δ = {delta:+.2f}%  (t={t_stat:.2f}, p={p_val:.4f})")
    if p_val < 0.05:
        if delta > 0:
            print(f"    → CRISTAIS AJUDAM significativamente (p < 0.05)")
        else:
            print(f"    → CRISTAIS PIORAM significativamente (p < 0.05)")
    else:
        print(f"    → Diferença NÃO significativa (p = {p_val:.4f})")


# ── Tabela comparativa completa ─────────────────────────────────────────────

print(f"\n{'='*75}")
print("RESULTADO 2: Comparação Completa")
print(f"{'='*75}")
print(f"\n{'Modelo':>22}  {'Params':>8}  {'Teste':>14}  {'%/kp':>8}")
print("-"*58)

for name in [c[0] for c in configs]:
    r = all_results[name]
    eff = r['mean'] / (r['n_params'] / 1000) if r['n_params'] > 0 else 0
    print(f"  {name:>22}  {r['n_params']:>8}  "
          f"{r['mean']:>6.2f}±{r['std']:.2f}%  "
          f"{eff:>7.2f}")


# ── Significância: ResNet-Ψ vs baselines ────────────────────────────────────

print(f"\n{'='*75}")
print("RESULTADO 3: Testes de significância")
print(f"{'='*75}")

comparisons = [
    ("Ψ+cristais+linear", "Linear 784→10"),
    ("Ψ+cristais+MLP",    "MLP 784→256→10"),
    ("Ψ+cristais+MLP",    "CNN pequena"),
]

for a_name, b_name in comparisons:
    a = all_results[a_name]
    b = all_results[b_name]
    t_stat, p_val = sp_stats.ttest_ind(a['accs'], b['accs'])
    delta = a['mean'] - b['mean']
    winner = a_name if delta > 0 else b_name
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"\n  {a_name} vs {b_name}:")
    print(f"    {a['mean']:.2f}% vs {b['mean']:.2f}%  Δ={delta:+.2f}%  p={p_val:.4f} {sig}")
    print(f"    Vencedor: {winner}")


# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Auditoria 5 — Ablação + Baselines MNIST ({N_SEEDS} seeds)', fontsize=13)

names  = [c[0] for c in configs]
means  = [all_results[n]['mean'] for n in names]
stds   = [all_results[n]['std']  for n in names]
params = [all_results[n]['n_params'] for n in names]

# Cores: cristais=vermelho, sem cristais=laranja, baselines=azul/verde
colors = ['#e6194b', '#e6194b', '#f58231', '#f58231', '#a9a9a9', '#4363d8', '#3cb44b']
hatches = ['', '///', '', '///', '', '', '']

# 1. Acurácia com barras de erro
bars = axes[0].barh(range(len(names)), means, xerr=stds, color=colors,
                     capsize=4, alpha=0.8, edgecolor='black', linewidth=0.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names, fontsize=8)
axes[0].set_xlabel('Teste (%)')
axes[0].set_title('Acurácia no Teste')
axes[0].grid(alpha=0.3, axis='x')
for i, (m, s) in enumerate(zip(means, stds)):
    axes[0].text(m + s + 0.3, i, f"{m:.1f}%", va='center', fontsize=8)

# 2. Ablação direta
abl_names = ['Linear', 'MLP(256)']
abl_with  = [all_results["Ψ+cristais+linear"]['mean'], all_results["Ψ+cristais+MLP"]['mean']]
abl_without = [all_results["Ψ-cristais+linear"]['mean'], all_results["Ψ-cristais+MLP"]['mean']]
abl_w_std = [all_results["Ψ+cristais+linear"]['std'], all_results["Ψ+cristais+MLP"]['std']]
abl_wo_std = [all_results["Ψ-cristais+linear"]['std'], all_results["Ψ-cristais+MLP"]['std']]

x = np.arange(len(abl_names))
w = 0.35
axes[1].bar(x - w/2, abl_with, w, yerr=abl_w_std, label='COM cristais',
            color='#e6194b', capsize=5, alpha=0.8)
axes[1].bar(x + w/2, abl_without, w, yerr=abl_wo_std, label='SEM cristais',
            color='#f58231', capsize=5, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'Decoder {n}' for n in abl_names])
axes[1].set_ylabel('Teste (%)')
axes[1].set_title('Ablação: COM vs SEM cristais')
axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
# Anotar deltas
for i in range(len(abl_names)):
    delta = abl_with[i] - abl_without[i]
    y_pos = max(abl_with[i], abl_without[i]) + 1
    axes[1].text(i, y_pos, f'Δ={delta:+.1f}%', ha='center', fontsize=10, fontweight='bold')

# 3. Eficiência (acc / k-params)
effs = [m / (p/1000) if p > 0 else 0 for m, p in zip(means, params)]
axes[2].barh(range(len(names)), effs, color=colors, alpha=0.8,
              edgecolor='black', linewidth=0.5)
axes[2].set_yticks(range(len(names)))
axes[2].set_yticklabels(names, fontsize=8)
axes[2].set_xlabel('Acurácia / kilo-parâmetros')
axes[2].set_title('Eficiência de Parâmetros')
axes[2].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('viz_audit_5_ablacao_baselines.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_5_ablacao_baselines.png")
print(f"\nTempo total: — veja timestamps acima")
print("Pronto.")
