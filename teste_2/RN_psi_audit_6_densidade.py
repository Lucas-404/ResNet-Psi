"""
Auditoria 6: Densidade Informacional — ResNet-Ψ vs Redes Convencionais

Pergunta: A ResNet-Ψ armazena mais informação por unidade de memória?

Mede MI(entrada; representação) em bits para:
  - ResNet-Ψ crystal_map (48, 64, 96, 128)
  - ResNet-Ψ campo bruto  (48, 64, 96, 128)
  - MLP hidden layers     (32, 64, 128, 256, 512)
  - CNN feature maps      (4, 8, 16, 32 canais)

Plota: MI (bits) vs Memória (KB) para cada sistema.

Se ResNet-Ψ tem curva acima das redes → densidade superior.
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
    def __init__(self, B, FS, enable_crystals=True):
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


def compute_psi_representations(X, field_size, enable_crystals=True, bs=64):
    """Computa representações do campo para um subset de imagens."""
    PG = build_gaussians(field_size)
    FS = field_size
    reps = []

    for i in range(0, len(X), bs):
        B = X[i:i+bs]
        pert = (B.view(len(B), 784) @ PG.to(B.dtype)).view(len(B), FS, FS)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalMem(len(B), FS, enable_crystals=enable_crystals)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)

        if enable_crystals:
            reps.append(mem.crystal_map.view(len(B), -1).cpu().numpy())
        else:
            reps.append(f.view(len(B), -1).cpu().numpy())

    del PG
    return np.concatenate(reps, axis=0)


# ── MI via clustering ────────────────────────────────────────────────────────

def compute_mi(representations, labels, n_clusters_list=[8, 16, 32, 64, 128]):
    """
    MI(entrada; representação) em bits via KMeans.
    Testa vários k e pega o máximo (saturação).
    """
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


def count_nonzero_memory(representations, threshold=0.01):
    """Conta memória efetiva: valores não-zero em float32 (4 bytes cada)."""
    nonzero = np.abs(representations) > threshold
    n_values = nonzero.sum(axis=1).mean()  # média por amostra
    memory_kb = n_values * 4 / 1024  # float32 = 4 bytes
    return n_values, memory_kb


# ── Representações de redes convencionais ────────────────────────────────────

def compute_mlp_representations(X, hidden_size, seed=42):
    """
    MLP com pesos aleatórios (sem treino) → hidden activations.
    Isso é o equivalente justo: representação sem treino do encoder.
    """
    torch.manual_seed(seed)
    W = torch.randn(784, hidden_size, device=DEVICE) * (2.0 / 784)**0.5
    b = torch.zeros(hidden_size, device=DEVICE)

    reps = []
    for i in range(0, len(X), 512):
        batch = X[i:i+512].view(-1, 784).float()
        h = torch.relu(batch @ W + b)
        reps.append(h.cpu().numpy())

    return np.concatenate(reps, axis=0)


def compute_cnn_representations(X, n_channels, seed=42):
    """
    CNN com pesos aleatórios (sem treino) → feature maps achatados.
    2 camadas conv + pooling, sem treino.
    """
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(1, n_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(n_channels, n_channels*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
    ).to(DEVICE)
    model.eval()

    reps = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = X[i:i+512].view(-1, 1, 28, 28).float()
            h = model(batch)
            reps.append(h.cpu().numpy())

    return np.concatenate(reps, axis=0)


def compute_trained_mlp_representations(X, Y, hidden_size, seed=42):
    """
    MLP TREINADO → hidden activations.
    Treina com backprop e extrai ativações da hidden layer.
    """
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # Treina por 10 épocas
    model.train()
    for ep in range(10):
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), 256):
            idx = perm[i:i+256]
            x = X[idx].view(-1, 784).float()
            opt.zero_grad(set_to_none=True)
            crit(model(x), Y[idx]).backward()
            opt.step()

    # Extrai representações da hidden layer
    model.eval()
    reps = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = X[i:i+512].view(-1, 784).float()
            h = torch.relu(model[0](x))  # ativação da primeira camada
            reps.append(h.cpu().numpy())

    return np.concatenate(reps, axis=0)


def compute_trained_cnn_representations(X, Y, n_channels, seed=42):
    """
    CNN TREINADA → feature maps.
    """
    torch.manual_seed(seed)
    conv_layers = nn.Sequential(
        nn.Conv2d(1, n_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(n_channels, n_channels*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
    )
    feat_dim = n_channels * 2 * 7 * 7
    model = nn.Sequential(
        conv_layers,
        nn.Linear(feat_dim, 10),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(10):
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), 256):
            idx = perm[i:i+256]
            x = X[idx].view(-1, 1, 28, 28).float()
            opt.zero_grad(set_to_none=True)
            crit(model(x), Y[idx]).backward()
            opt.step()

    # Extrai representações das conv layers
    model.eval()
    reps = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = X[i:i+512].view(-1, 1, 28, 28).float()
            h = conv_layers(x)
            reps.append(h.cpu().numpy())

    return np.concatenate(reps, axis=0)


# ── MNIST ────────────────────────────────────────────────────────────────────

def load_mnist_subset(n_samples=2000):
    """Subset balanceado para MI (não precisa de 60k, MI satura rápido)."""
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,),(0.3081,))])
    ds = datasets.MNIST('./data', train=True, download=True, transform=tf)

    # Pega n_samples balanceado por classe
    per_class = n_samples // 10
    selected_x, selected_y = [], []
    counts = [0] * 10

    for i in range(len(ds)):
        x, y = ds[i]
        if counts[y] < per_class:
            selected_x.append(x.squeeze(0))
            selected_y.append(y)
            counts[y] += 1
        if all(c >= per_class for c in counts):
            break

    X = torch.stack(selected_x).to(DEVICE)
    Y = torch.tensor(selected_y, dtype=torch.long, device=DEVICE)
    return X, Y


# ── Experimento principal ────────────────────────────────────────────────────

N_SAMPLES = 2000  # subset para MI (suficiente, mais rápido)
N_SEEDS   = 5     # seeds para MI (KMeans é estocástico)

print(f"\nCarregando MNIST ({N_SAMPLES} amostras balanceadas)...")
X, Y = load_mnist_subset(N_SAMPLES)
labels = Y.cpu().numpy()
print(f"  Carregado: {len(X)} amostras, {len(np.unique(labels))} classes")

print(f"\n{'='*75}")
print("AUDITORIA 6: Densidade Informacional")
print(f"MI(entrada; representação) vs Memória por amostra")
print(f"{'='*75}")

results = []
t0 = time.time()

# ── 1. ResNet-Ψ COM cristais (vários tamanhos de campo) ────────────────────

PSI_FIELDS = [48, 64, 96, 128]

print(f"\n── ResNet-Ψ COM cristais ──")
for fs in PSI_FIELDS:
    print(f"  Campo {fs}×{fs}...", end=' ', flush=True)
    t1 = time.time()
    reps = compute_psi_representations(X, fs, enable_crystals=True, bs=64)
    n_vals, mem_kb = count_nonzero_memory(reps)

    mis = []
    for seed in range(N_SEEDS):
        mi = compute_mi(reps, labels, n_clusters_list=[8, 16, 32, 64, 128])
        mis.append(mi)
    mi_mean = np.mean(mis)
    mi_std  = np.std(mis)

    density = mi_mean / mem_kb if mem_kb > 0 else 0
    elapsed = time.time() - t1

    results.append({
        'name': f'Ψ+cristais {fs}²',
        'system': 'psi_crystal',
        'mi_mean': mi_mean, 'mi_std': mi_std,
        'mem_kb': mem_kb, 'n_values': n_vals,
        'density': density, 'dim': fs*fs,
    })
    print(f"MI={mi_mean:.2f}±{mi_std:.2f} bits  "
          f"mem={mem_kb:.1f} KB  "
          f"dens={density:.3f} bits/KB  "
          f"({elapsed:.0f}s)")


# ── 2. ResNet-Ψ SEM cristais (campo bruto) ─────────────────────────────────

print(f"\n── ResNet-Ψ SEM cristais (onda pura) ──")
for fs in PSI_FIELDS:
    print(f"  Campo {fs}×{fs}...", end=' ', flush=True)
    t1 = time.time()
    reps = compute_psi_representations(X, fs, enable_crystals=False, bs=64)
    n_vals, mem_kb = count_nonzero_memory(reps)

    mis = []
    for seed in range(N_SEEDS):
        mi = compute_mi(reps, labels, n_clusters_list=[8, 16, 32, 64, 128])
        mis.append(mi)
    mi_mean = np.mean(mis)
    mi_std  = np.std(mis)

    density = mi_mean / mem_kb if mem_kb > 0 else 0

    results.append({
        'name': f'Ψ-cristais {fs}²',
        'system': 'psi_wave',
        'mi_mean': mi_mean, 'mi_std': mi_std,
        'mem_kb': mem_kb, 'n_values': n_vals,
        'density': density, 'dim': fs*fs,
    })
    elapsed = time.time() - t1
    print(f"MI={mi_mean:.2f}±{mi_std:.2f} bits  "
          f"mem={mem_kb:.1f} KB  "
          f"dens={density:.3f} bits/KB  "
          f"({elapsed:.0f}s)")


# ── 3. MLP sem treino (pesos aleatórios) ───────────────────────────────────

MLP_SIZES = [32, 64, 128, 256, 512, 1024]

print(f"\n── MLP sem treino (pesos aleatórios) ──")
for hs in MLP_SIZES:
    print(f"  Hidden {hs}...", end=' ', flush=True)
    t1 = time.time()
    reps = compute_mlp_representations(X, hs)
    n_vals, mem_kb = count_nonzero_memory(reps)

    # Memória do modelo = pesos (784*hs + hs) * 4 bytes
    model_mem_kb = (784 * hs + hs) * 4 / 1024

    mis = []
    for seed in range(N_SEEDS):
        mi = compute_mi(reps, labels, n_clusters_list=[8, 16, 32, 64, 128])
        mis.append(mi)
    mi_mean = np.mean(mis)
    mi_std  = np.std(mis)

    # Densidade: MI / memória da representação
    density = mi_mean / mem_kb if mem_kb > 0 else 0

    results.append({
        'name': f'MLP-rand {hs}',
        'system': 'mlp_random',
        'mi_mean': mi_mean, 'mi_std': mi_std,
        'mem_kb': mem_kb, 'n_values': n_vals,
        'density': density, 'dim': hs,
        'model_mem_kb': model_mem_kb,
    })
    elapsed = time.time() - t1
    print(f"MI={mi_mean:.2f}±{mi_std:.2f} bits  "
          f"mem={mem_kb:.1f} KB  "
          f"dens={density:.3f} bits/KB  "
          f"({elapsed:.0f}s)")


# ── 4. MLP TREINADO ────────────────────────────────────────────────────────

print(f"\n── MLP TREINADO (backprop 10 épocas) ──")
for hs in MLP_SIZES:
    print(f"  Hidden {hs}...", end=' ', flush=True)
    t1 = time.time()
    reps = compute_trained_mlp_representations(X, Y, hs)
    n_vals, mem_kb = count_nonzero_memory(reps)

    model_mem_kb = (784 * hs + hs) * 4 / 1024

    mis = []
    for seed in range(N_SEEDS):
        mi = compute_mi(reps, labels, n_clusters_list=[8, 16, 32, 64, 128])
        mis.append(mi)
    mi_mean = np.mean(mis)
    mi_std  = np.std(mis)

    density = mi_mean / mem_kb if mem_kb > 0 else 0

    results.append({
        'name': f'MLP-train {hs}',
        'system': 'mlp_trained',
        'mi_mean': mi_mean, 'mi_std': mi_std,
        'mem_kb': mem_kb, 'n_values': n_vals,
        'density': density, 'dim': hs,
        'model_mem_kb': model_mem_kb,
    })
    elapsed = time.time() - t1
    print(f"MI={mi_mean:.2f}±{mi_std:.2f} bits  "
          f"mem={mem_kb:.1f} KB  "
          f"dens={density:.3f} bits/KB  "
          f"({elapsed:.0f}s)")


# ── 5. CNN sem treino ──────────────────────────────────────────────────────

CNN_CHANNELS = [4, 8, 16, 32]

print(f"\n── CNN sem treino (pesos aleatórios) ──")
for nc in CNN_CHANNELS:
    print(f"  Canais {nc}...", end=' ', flush=True)
    t1 = time.time()
    reps = compute_cnn_representations(X, nc)
    n_vals, mem_kb = count_nonzero_memory(reps)

    mis = []
    for seed in range(N_SEEDS):
        mi = compute_mi(reps, labels, n_clusters_list=[8, 16, 32, 64, 128])
        mis.append(mi)
    mi_mean = np.mean(mis)
    mi_std  = np.std(mis)

    density = mi_mean / mem_kb if mem_kb > 0 else 0

    results.append({
        'name': f'CNN-rand {nc}ch',
        'system': 'cnn_random',
        'mi_mean': mi_mean, 'mi_std': mi_std,
        'mem_kb': mem_kb, 'n_values': n_vals,
        'density': density, 'dim': nc*2*7*7,
    })
    elapsed = time.time() - t1
    print(f"MI={mi_mean:.2f}±{mi_std:.2f} bits  "
          f"mem={mem_kb:.1f} KB  "
          f"dens={density:.3f} bits/KB  "
          f"({elapsed:.0f}s)")


# ── 6. CNN TREINADA ────────────────────────────────────────────────────────

print(f"\n── CNN TREINADA (backprop 10 épocas) ──")
for nc in CNN_CHANNELS:
    print(f"  Canais {nc}...", end=' ', flush=True)
    t1 = time.time()
    reps = compute_trained_cnn_representations(X, Y, nc)
    n_vals, mem_kb = count_nonzero_memory(reps)

    mis = []
    for seed in range(N_SEEDS):
        mi = compute_mi(reps, labels, n_clusters_list=[8, 16, 32, 64, 128])
        mis.append(mi)
    mi_mean = np.mean(mis)
    mi_std  = np.std(mis)

    density = mi_mean / mem_kb if mem_kb > 0 else 0

    results.append({
        'name': f'CNN-train {nc}ch',
        'system': 'cnn_trained',
        'mi_mean': mi_mean, 'mi_std': mi_std,
        'mem_kb': mem_kb, 'n_values': n_vals,
        'density': density, 'dim': nc*2*7*7,
    })
    elapsed = time.time() - t1
    print(f"MI={mi_mean:.2f}±{mi_std:.2f} bits  "
          f"mem={mem_kb:.1f} KB  "
          f"dens={density:.3f} bits/KB  "
          f"({elapsed:.0f}s)")


# ── 7. Pixels crus (baseline) ─────────────────────────────────────────────

print(f"\n── Pixels crus (784 valores) ──")
reps_pixels = X.view(len(X), -1).cpu().numpy()
n_vals_px, mem_kb_px = count_nonzero_memory(reps_pixels)
mis_px = []
for seed in range(N_SEEDS):
    mi = compute_mi(reps_pixels, labels, n_clusters_list=[8, 16, 32, 64, 128])
    mis_px.append(mi)
mi_px_mean = np.mean(mis_px)
mi_px_std  = np.std(mis_px)
density_px = mi_px_mean / mem_kb_px if mem_kb_px > 0 else 0

results.append({
    'name': 'Pixels crus 784',
    'system': 'pixels',
    'mi_mean': mi_px_mean, 'mi_std': mi_px_std,
    'mem_kb': mem_kb_px, 'n_values': n_vals_px,
    'density': density_px, 'dim': 784,
})
print(f"  MI={mi_px_mean:.2f}±{mi_px_std:.2f} bits  "
      f"mem={mem_kb_px:.1f} KB  "
      f"dens={density_px:.3f} bits/KB")


# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*85}")
print("RESULTADO: Densidade Informacional Comparada")
print(f"{'='*85}")
print(f"\n{'Sistema':<22} {'MI (bits)':>12} {'Mem (KB)':>10} {'Valores':>10} {'bits/KB':>10}")
print("-"*70)

for r in sorted(results, key=lambda x: -x['density']):
    print(f"  {r['name']:<22} {r['mi_mean']:>8.2f}±{r['mi_std']:.2f} "
          f"{r['mem_kb']:>10.1f} {r['n_values']:>10.0f} {r['density']:>10.3f}")

print(f"\nTempo total: {time.time()-t0:.0f}s")


# ── Plot principal: MI vs Memória ────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Auditoria 6 — Densidade Informacional: MI vs Memória', fontsize=14)

# Cores e marcadores por sistema
style = {
    'psi_crystal':  {'color': '#e6194b', 'marker': 's', 'label': 'Ψ+cristais'},
    'psi_wave':     {'color': '#f58231', 'marker': 'D', 'label': 'Ψ onda pura'},
    'mlp_random':   {'color': '#a9a9a9', 'marker': 'o', 'label': 'MLP random'},
    'mlp_trained':  {'color': '#4363d8', 'marker': '^', 'label': 'MLP treinado'},
    'cnn_random':   {'color': '#bfef45', 'marker': 'v', 'label': 'CNN random'},
    'cnn_trained':  {'color': '#3cb44b', 'marker': 'P', 'label': 'CNN treinada'},
    'pixels':       {'color': '#000000', 'marker': 'X', 'label': 'Pixels crus'},
}

# Plot 1: MI vs Memória (log-log)
ax = axes[0]
for sys_name, sty in style.items():
    pts = [r for r in results if r['system'] == sys_name]
    if not pts:
        continue
    mems = [r['mem_kb'] for r in pts]
    mis  = [r['mi_mean'] for r in pts]
    errs = [r['mi_std'] for r in pts]
    ax.errorbar(mems, mis, yerr=errs, fmt=sty['marker']+'-',
                color=sty['color'], label=sty['label'],
                markersize=8, capsize=3, linewidth=2)
ax.set_xlabel('Memória por amostra (KB)')
ax.set_ylabel('MI (bits)')
ax.set_title('MI vs Memória')
ax.set_xscale('log')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Plot 2: Densidade (bits/KB) por sistema
ax = axes[1]
# Agrupa por sistema, pega média de densidade
sys_densities = {}
for r in results:
    s = r['system']
    if s not in sys_densities:
        sys_densities[s] = []
    sys_densities[s].append(r['density'])

sys_names_sorted = sorted(sys_densities.keys(),
                           key=lambda s: np.mean(sys_densities[s]), reverse=True)
colors_bar = [style[s]['color'] for s in sys_names_sorted]
labels_bar = [style[s]['label'] for s in sys_names_sorted]
means_bar  = [np.mean(sys_densities[s]) for s in sys_names_sorted]
stds_bar   = [np.std(sys_densities[s]) for s in sys_names_sorted]

bars = ax.barh(range(len(means_bar)), means_bar, xerr=stds_bar,
               color=colors_bar, capsize=4, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(labels_bar)))
ax.set_yticklabels(labels_bar, fontsize=9)
ax.set_xlabel('Densidade (bits/KB)')
ax.set_title('Densidade Média por Sistema')
ax.grid(alpha=0.3, axis='x')

# Plot 3: MI vs Dimensão da representação
ax = axes[2]
for sys_name, sty in style.items():
    pts = [r for r in results if r['system'] == sys_name]
    if not pts:
        continue
    dims = [r['dim'] for r in pts]
    mis  = [r['mi_mean'] for r in pts]
    errs = [r['mi_std'] for r in pts]
    ax.errorbar(dims, mis, yerr=errs, fmt=sty['marker']+'-',
                color=sty['color'], label=sty['label'],
                markersize=8, capsize=3, linewidth=2)
ax.set_xlabel('Dimensão da representação')
ax.set_ylabel('MI (bits)')
ax.set_title('MI vs Dimensão')
ax.set_xscale('log')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_audit_6_densidade.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_6_densidade.png")
print("Pronto.")
