"""
Auditoria 1: β_c via Informação Mútua (não tautológica)

Substitui o cálculo circular β_c = log₂(1/α) por mutual information
entre entrada e fingerprint do cristal.

Grid de (γ, β) para verificar se β_c é constante ou depende da física.
20 seeds por configuração.

Resultado: tabela (γ, β) → β_c ± σ
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
import time
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")

# ── Constantes base (defaults de RN_psi_mnist.py) ──────────────────────────
FIELD_SIZE     = 48
PSI_C2         = 0.3
PSI_ALPHA      = 0.04
PSI_DT         = 0.05
STIM_ON        = 40
STIM_TOTAL     = 80

CRYSTAL_W      = 20
CRYSTAL_K      = 3
CRYSTAL_A_MIN  = 0.3
CRYSTAL_CV_MAX = 0.15
CRYSTAL_SEP    = 5
CRYSTAL_REMIT  = 0.05
CRYSTAL_PATTERN = 5
PATCH          = 2 * CRYSTAL_PATTERN + 1  # 11

_DT    = torch.tensor(PSI_DT, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_C2    = torch.tensor(PSI_C2, device=DEVICE)

# ── Física parametrizada por (gamma, beta) ──────────────────────────────────

def psi_step(field, velocity, sources, active, gamma, beta):
    lap_k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                          device=DEVICE).view(1,1,3,3).to(field.dtype)
    if active:
        field = field + sources * (_DT * 0.1)
    lap = F.conv2d(F.pad(field.unsqueeze(1),(1,1,1,1),mode='circular'), lap_k).squeeze(1)
    acc = _C2*lap - gamma*velocity + _ALPHA*torch.tanh(field)*field - beta*field*field**2
    velocity = torch.clamp(velocity + acc*_DT, -5., 5.)
    field    = torch.clamp(field + velocity*_DT, -10., 10.)
    return field, velocity


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


# ── Encoder de texto (mesmo do RN_psi_test_bits.py) ────────────────────────

BASE_CHARS = "abcdefghijklmnopqrstuvwxyz"

def make_perturbation(text, field_size=FIELD_SIZE):
    field   = torch.zeros(field_size, field_size, device=DEVICE)
    n_cells = 8
    sigma   = 0.06
    coords  = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg  = torch.meshgrid(coords, coords, indexing='ij')
    for i, c in enumerate(text):
        v   = ord(c) / 127.0
        cx  = ((ord(c) % n_cells) + 0.5) / n_cells
        cy  = (((ord(c) // n_cells) % n_cells) + 0.5) / n_cells
        amp = 1.5 + v * 2.5
        g   = amp * torch.exp(-((xg-cx)**2 + (yg-cy)**2) / (2*sigma**2))
        field = field + g * float(np.cos(i * 0.3))
    return field.unsqueeze(0)


def run_field(text, gamma, beta, field_size=FIELD_SIZE):
    pert     = make_perturbation(text, field_size)
    field    = pert.clone()
    velocity = torch.zeros_like(field)
    memory   = CrystalMem(1, field_size)
    _g = torch.tensor(gamma, device=DEVICE)
    _b = torch.tensor(beta, device=DEVICE)

    with torch.no_grad():
        for s in range(STIM_TOTAL):
            active = s < STIM_ON
            field, velocity = psi_step(field, velocity, pert, active, _g, _b)
            memory.update_envelope(field)
            if memory.window_idx > 0:
                memory.try_crystallize(field)
            field = memory.remit(field)

    return memory.crystal_map.squeeze(0).cpu().numpy()


def extract_fingerprints(cmap, thr=0.01):
    visited = np.zeros_like(cmap, dtype=bool)
    crystals = []
    ys, xs = np.where(cmap > thr)
    if len(ys) == 0:
        return crystals
    order = np.argsort(-cmap[ys, xs])
    ys, xs = ys[order], xs[order]

    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        y0 = max(0, y-CRYSTAL_SEP); y1 = min(cmap.shape[0], y+CRYSTAL_SEP+1)
        x0 = max(0, x-CRYSTAL_SEP); x1 = min(cmap.shape[1], x+CRYSTAL_SEP+1)
        visited[y0:y1, x0:x1] = True

        py0 = max(0, y-CRYSTAL_PATTERN); py1 = min(cmap.shape[0], y+CRYSTAL_PATTERN+1)
        px0 = max(0, x-CRYSTAL_PATTERN); px1 = min(cmap.shape[1], x+CRYSTAL_PATTERN+1)
        patch = cmap[py0:py1, px0:px1]

        pad_h = PATCH - patch.shape[0]
        pad_w = PATCH - patch.shape[1]
        patch = np.pad(patch, ((0,pad_h),(0,pad_w)))
        norm  = np.linalg.norm(patch)
        if norm < 1e-8:
            continue
        fp = patch.flatten() / norm
        crystals.append(fp)

    return crystals


# ── Mutual Information via clustering ───────────────────────────────────────

def compute_mi_bits(texts, gamma, beta, n_clusters_list=[4, 8, 16, 32]):
    """
    Calcula I(entrada; representação cristalina) em bits.

    1. Para cada texto, computa crystal_map e extrai fingerprint médio
    2. Clusteriza fingerprints (k-means)
    3. Calcula MI entre label da entrada e cluster assignment
    4. Retorna MI / n_cristais_médio = bits por cristal

    Repete para vários k e pega o máximo (saturação).
    """
    # Computa representações
    all_fps = []
    all_n_crys = []
    valid_indices = []

    for i, text in enumerate(texts):
        cmap = run_field(text, gamma, beta)
        fps  = extract_fingerprints(cmap)
        if len(fps) == 0:
            continue
        # Fingerprint médio como representação da entrada
        mean_fp = np.mean(fps, axis=0)
        all_fps.append(mean_fp)
        all_n_crys.append(len(fps))
        valid_indices.append(i)

    if len(all_fps) < 10:
        return None, 0, 0

    X = np.array(all_fps)
    n_crys_mean = np.mean(all_n_crys)

    # Labels = índice da entrada (cada texto é uma "classe")
    labels = np.array(valid_indices)

    # MI para diferentes k
    best_mi = 0
    for k in n_clusters_list:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, n_init=5, random_state=42, max_iter=100)
        clusters = km.fit_predict(X)
        mi = mutual_info_score(labels, clusters)
        mi_bits = mi / np.log(2)  # nats → bits
        if mi_bits > best_mi:
            best_mi = mi_bits

    beta_c = best_mi / max(n_crys_mean, 1)
    return beta_c, best_mi, n_crys_mean


# ── Experimento principal ───────────────────────────────────────────────────

# Grid de hiperparâmetros
GAMMAS = [0.03, 0.04, 0.06, 0.08, 0.10]
BETAS  = [0.002, 0.005, 0.008, 0.01]
N_SEEDS = 10   # seeds por configuração
N_INPUTS = 60  # entradas distintas por seed

print("="*70)
print("AUDITORIA 1: β_c via Informação Mútua")
print(f"Grid: {len(GAMMAS)} γ × {len(BETAS)} β = {len(GAMMAS)*len(BETAS)} configurações")
print(f"Seeds: {N_SEEDS} | Entradas por seed: {N_INPUTS}")
print("="*70)

results = []
t0 = time.time()

for gamma in GAMMAS:
    for beta in BETAS:
        seed_results = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            texts = [''.join(rng.choice(list(BASE_CHARS), 4)) for _ in range(N_INPUTS)]

            bc, mi_total, n_crys = compute_mi_bits(texts, gamma, beta)
            if bc is not None:
                seed_results.append({
                    'beta_c': bc, 'mi_total': mi_total, 'n_crys': n_crys
                })

        if not seed_results:
            print(f"  γ={gamma:.3f} β={beta:.4f}: SEM CRISTAIS")
            continue

        bcs = [r['beta_c'] for r in seed_results]
        mis = [r['mi_total'] for r in seed_results]
        ncs = [r['n_crys'] for r in seed_results]

        bc_mean = np.mean(bcs)
        bc_std  = np.std(bcs)
        mi_mean = np.mean(mis)
        nc_mean = np.mean(ncs)

        results.append({
            'gamma': gamma, 'beta': beta,
            'beta_c_mean': bc_mean, 'beta_c_std': bc_std,
            'mi_mean': mi_mean, 'n_crys_mean': nc_mean,
            'n_seeds': len(seed_results),
        })

        elapsed = time.time() - t0
        print(f"  γ={gamma:.3f} β={beta:.4f}: β_c = {bc_mean:.4f} ± {bc_std:.4f}  "
              f"MI={mi_mean:.2f} bits  cristais={nc_mean:.1f}  [{elapsed:.0f}s]")

# ── Resumo ──────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESUMO: β_c por configuração física")
print(f"{'='*70}")
print(f"\n{'γ':>6}  {'β':>7}  {'β_c':>8}  {'± σ':>7}  {'MI (bits)':>10}  {'Cristais':>9}  {'CV':>6}")
print("-"*60)

all_bcs = []
for r in results:
    cv = r['beta_c_std'] / (r['beta_c_mean'] + 1e-8)
    all_bcs.append(r['beta_c_mean'])
    print(f"  {r['gamma']:.3f}  {r['beta']:.4f}  {r['beta_c_mean']:>8.4f}  {r['beta_c_std']:>7.4f}  "
          f"{r['mi_mean']:>10.3f}  {r['n_crys_mean']:>9.1f}  {cv:>6.2f}")

if all_bcs:
    global_mean = np.mean(all_bcs)
    global_std  = np.std(all_bcs)
    global_cv   = global_std / (global_mean + 1e-8)
    print(f"\nβ_c global: {global_mean:.4f} ± {global_std:.4f} (CV = {global_cv:.2f})")
    if global_cv < 0.15:
        print("→ β_c É APROXIMADAMENTE CONSTANTE entre configurações")
    elif global_cv < 0.30:
        print("→ β_c VARIA MODERADAMENTE com (γ, β)")
    else:
        print("→ β_c DEPENDE FORTEMENTE de (γ, β) — NÃO é constante universal")

print(f"\nTempo total: {time.time()-t0:.0f}s")
print("Pronto.")
