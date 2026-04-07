"""
Auditoria 21b: Varredura de threshold no Recurrence Plot

Testa thresholds: 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50
Para cada um: mostra RPs das 3 classes + ratio inter/intra dos crystal maps.
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wfdb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, B, FS=FIELD_SIZE):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(B, FS, FS, device=DEVICE)
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
        amp_score = torch.sigmoid(5.0 * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(5.0 * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(5.0 * (8.0 - mean))
        cand = amp_score * cv_score * sat_score
        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0, 1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0, 1)
        new_crystals = cand * (1.0 - occ) * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + new_crystals, 0, 10.)
        self.crystal_hp = torch.where(
            new_crystals > 0.01,
            torch.clamp(self.crystal_hp + 1.0, 0, 5.0),
            self.crystal_hp)
        ressonance = field.abs() * (self.crystal_map > 0.01).float()
        self.crystal_hp = self.crystal_hp + ressonance * 0.1
        self.crystal_hp = self.crystal_hp - 0.02
        alive = (self.crystal_hp > 0).float()
        self.crystal_map = self.crystal_map * alive
        self.crystal_hp  = torch.clamp(self.crystal_hp * alive, 0, 5.0)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


def psi_step(field, velocity, sources, active):
    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                          device=DEVICE).view(1, 1, 3, 3).to(field.dtype)
    if active:
        field = field + sources * (_DT * 0.1)
    lap = F.conv2d(F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular'), lap_k).squeeze(1)
    acc = _C2 * lap - _GAMMA * velocity + _ALPHA * torch.tanh(field) * field - _BETA * field * field**2
    velocity = torch.clamp(velocity + acc * _DT, -5., 5.)
    field    = torch.clamp(field + velocity * _DT, -10., 10.)
    return field, velocity


def build_gaussians(input_size=28, field_size=FIELD_SIZE, sigma=0.04):
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for pi in range(input_size):
        for pj in range(input_size):
            cx = 0.1 + 0.8 * pi / (input_size - 1)
            cy = 0.1 + 0.8 * pj / (input_size - 1)
            gs.append(torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2)))
    return torch.stack(gs).view(input_size * input_size, -1)


def recurrence_plot(signal, size=28, threshold=None):
    sig = np.array(signal, dtype=np.float32)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
    idx = np.linspace(0, len(sig) - 1, size).astype(int)
    sig = sig[idx]
    diff = np.abs(sig[:, None] - sig[None, :])
    if threshold is None:
        # Continuo: similaridade direta, sem binarizacao
        return (1.0 - diff).astype(np.float32)
    return (diff < threshold).astype(np.float32)


def run_field(img_28, PG):
    img_t = torch.tensor(img_28, dtype=torch.float32, device=DEVICE)
    pert = (img_t.view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)
    f, v = pert.clone(), torch.zeros_like(pert)
    mem = CrystalCompetitivo(1)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f, v = psi_step(f, v, pert, s < STIM_ON)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)
    return mem.crystal_map.squeeze(0)


def ratio_intra_inter(cmaps):
    classes = list(cmaps.keys())
    intra, inter = [], []
    for cls in classes:
        a = torch.stack(cmaps[cls]).view(len(cmaps[cls]), -1).float()
        d = torch.cdist(a, a).cpu().numpy()
        idx = np.triu_indices(len(d), k=1)
        intra.extend(d[idx].tolist())
    for i, cls_a in enumerate(classes):
        for cls_b in classes[i+1:]:
            a = torch.stack(cmaps[cls_a]).view(len(cmaps[cls_a]), -1).float()
            b = torch.stack(cmaps[cls_b]).view(len(cmaps[cls_b]), -1).float()
            inter.extend(torch.cdist(a, b).cpu().numpy().flatten().tolist())
    intra = np.array(intra)
    inter = np.array(inter)
    return inter.mean() / (intra.mean() + 1e-8), intra.mean(), inter.mean()


def norm(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = x - x.min()
    return x / (x.max() + 1e-8)


# -- Baixar segmentos (uma vez) -----------------------------------------------

print("Baixando MIT-BIH...")
RECORDS = ['100', '101', '105', '106', '108']
SEG_LEN = 360
N_POR_CLASSE = 20

segmentos = {'N': [], 'V': [], 'A': []}
for rec_id in RECORDS:
    try:
        record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
        ann    = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
        signal = record.p_signal[:, 0]
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol == 'N' and len(segmentos['N']) < N_POR_CLASSE:
                cls = 'N'
            elif symbol == 'V' and len(segmentos['V']) < N_POR_CLASSE:
                cls = 'V'
            elif symbol in ('A', 'a') and len(segmentos['A']) < N_POR_CLASSE:
                cls = 'A'
            else:
                continue
            start, end = sample - SEG_LEN // 2, sample + SEG_LEN // 2
            if start < 0 or end > len(signal):
                continue
            segmentos[cls].append(signal[start:end])
        if all(len(v) >= N_POR_CLASSE for v in segmentos.values()):
            break
    except Exception as e:
        print(f"  {rec_id}: erro ({e})")

for cls in segmentos:
    segmentos[cls] = segmentos[cls][:N_POR_CLASSE]
print(f"Segmentos: { {k: len(v) for k, v in segmentos.items()} }")

PG = build_gaussians(input_size=28)

# -- Varredura de thresholds --------------------------------------------------

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, None]  # None = continuo
CLASSES = list(segmentos.keys())
N_VIZ = 3  # exemplos por classe na visualizacao

resultados = []

# Plot principal: RPs por threshold
fig, axes = plt.subplots(len(THRESHOLDS), len(CLASSES) * N_VIZ + 1,
                          figsize=(4 * (len(CLASSES) * N_VIZ + 1), 3 * len(THRESHOLDS)))
fig.suptitle('Auditoria 21b — Varredura de Threshold no Recurrence Plot', fontsize=13)

nomes = {'N': 'Normal', 'V': 'Ventricular', 'A': 'Atrial'}

for row, thr in enumerate(THRESHOLDS):
    label = f"thr={thr}" if thr is not None else "continuo"
    print(f"\n{label}...")

    # Computa RPs e crystal maps
    cmaps = {cls: [] for cls in CLASSES}
    rps_ex = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        for seg in segmentos[cls]:
            rp = recurrence_plot(seg, size=28, threshold=thr)
            cmaps[cls].append(run_field(rp, PG))
            if len(rps_ex[cls]) < N_VIZ:
                rps_ex[cls].append(rp)

    ratio, intra, inter = ratio_intra_inter(cmaps)
    resultados.append((thr, ratio, intra, inter))
    print(f"  Ratio={ratio:.2f}x  intra={intra:.3f}  inter={inter:.3f}")

    # Linha do plot: RPs das 3 classes + ratio
    col = 0
    for cls in CLASSES:
        for i in range(N_VIZ):
            axes[row][col].imshow(rps_ex[cls][i], cmap='binary', vmin=0, vmax=1)
            axes[row][col].axis('off')
            if row == 0:
                axes[row][col].set_title(f'{nomes[cls]} {i+1}', fontsize=8)
            col += 1

    # Ultima coluna: ratio
    axes[row][col].axis('off')
    cor = 'green' if ratio > 2 else ('orange' if ratio > 1.5 else 'red')
    label = f"thr={thr}" if thr is not None else "continuo"
    axes[row][col].text(0.5, 0.5, f'{label}\nRatio\n{ratio:.2f}x',
                        ha='center', va='center', fontsize=14,
                        fontweight='bold', color=cor,
                        transform=axes[row][col].transAxes)
    if row == 0:
        axes[row][col].set_title('Ratio', fontsize=8)

plt.tight_layout()
plt.savefig('viz_audit_21b_thresholds.png', dpi=110, bbox_inches='tight')
plt.close()

# -- Curva de ratio vs threshold ----------------------------------------------

fig2, ax = plt.subplots(figsize=(8, 5))
labels = [str(r[0]) if r[0] is not None else 'cont.' for r in resultados]
ratios = [r[1] for r in resultados]
intras = [r[2] for r in resultados]
inters = [r[3] for r in resultados]

x = range(len(labels))
ax.plot(x, ratios, 'o-', color='blue', linewidth=2, markersize=8, label='Ratio inter/intra')
ax2 = ax.twinx()
ax2.plot(x, intras, 's--', color='green', linewidth=1.5, markersize=6, label='Intra (media)')
ax2.plot(x, inters, '^--', color='red',   linewidth=1.5, markersize=6, label='Inter (media)')

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_xlabel('Threshold')
ax.set_ylabel('Ratio inter/intra', color='blue')
ax2.set_ylabel('Distancia media')
ax.set_title('Separacao de classes ECG vs Threshold')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

best_idx = int(np.argmax(ratios))
ax.axvline(x=best_idx, color='orange', linestyle='--', alpha=0.7, label=f'Melhor: {labels[best_idx]}')
ax.text(best_idx + 0.1, ratios[best_idx] * 0.95, f'{ratios[best_idx]:.2f}x', color='orange', fontsize=11)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_audit_21b_curva.png', dpi=110, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print(f"RESUMO")
print(f"{'='*50}")
for thr, ratio, intra, inter in resultados:
    label = f"thr={thr:.2f}" if thr is not None else "continuo"
    print(f"  {label:12s}  ratio={ratio:.2f}x  intra={intra:.3f}  inter={inter:.3f}")

best = max(resultados, key=lambda x: x[1])
best_label = f"thr={best[0]}" if best[0] is not None else "continuo"
print(f"\n  Melhor: {best_label} — Ratio {best[1]:.2f}x")
print(f"\n-> viz_audit_21b_thresholds.png")
print(f"-> viz_audit_21b_curva.png")
print("Pronto.")
