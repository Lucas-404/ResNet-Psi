"""
Auditoria 21: ResNet-Psi em sinais ECG — Classificacao real

Pergunta: o campo consegue classificar batimentos cardiacos sem treino?

Pipeline:
1. Baixa MIT-BIH Arrhythmia (wfdb)
2. Converte segmentos ECG em Recurrence Plot continuo (sem threshold)
3. Roda o campo em cada segmento
4. Classifica via leave-one-out: protótipo = media dos outros da mesma classe

Metrica: acuracia leave-one-out vs chance aleatoria.
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
print(f"Dispositivo: {DEVICE}")

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


def recurrence_plot(signal, size=28):
    """Recurrence Plot continuo: similaridade entre todos os pares de pontos."""
    sig = np.array(signal, dtype=np.float32)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
    idx = np.linspace(0, len(sig) - 1, size).astype(int)
    sig = sig[idx]
    diff = np.abs(sig[:, None] - sig[None, :])
    return (1.0 - diff).astype(np.float32)  # continuo: 1=identico, 0=maximo diferente


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


def norm(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = x - x.min()
    return x / (x.max() + 1e-8)


# -- Baixar MIT-BIH -----------------------------------------------------------

print("\nBaixando MIT-BIH Arrhythmia...")

RECORDS      = ['100', '101', '105', '106', '108']
SEG_LEN      = 360
N_POR_CLASSE = 30

segmentos = {'N': [], 'V': [], 'A': []}

for rec_id in RECORDS:
    try:
        record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
        ann    = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
        signal = record.p_signal[:, 0]
        for sample, symbol in zip(ann.sample, ann.symbol):
            if   symbol == 'N'          and len(segmentos['N']) < N_POR_CLASSE: cls = 'N'
            elif symbol == 'V'          and len(segmentos['V']) < N_POR_CLASSE: cls = 'V'
            elif symbol in ('A', 'a')   and len(segmentos['A']) < N_POR_CLASSE: cls = 'A'
            else: continue
            start, end = sample - SEG_LEN // 2, sample + SEG_LEN // 2
            if start < 0 or end > len(signal): continue
            segmentos[cls].append(signal[start:end])
        print(f"  {rec_id}: { {k: len(v) for k, v in segmentos.items()} }")
        if all(len(v) >= N_POR_CLASSE for v in segmentos.values()):
            break
    except Exception as e:
        print(f"  {rec_id}: erro ({e})")

for cls in segmentos:
    segmentos[cls] = segmentos[cls][:N_POR_CLASSE]

print(f"Total: { {k: len(v) for k, v in segmentos.items()} }")

# -- Crystal maps -------------------------------------------------------------

print("\nComputando crystal maps...")
PG = build_gaussians(input_size=28)

cmaps = {cls: [] for cls in segmentos}
for cls, segs in segmentos.items():
    for seg in segs:
        rp   = recurrence_plot(seg, size=28)
        cmap = run_field(rp, PG)
        cmaps[cls].append(cmap)
    print(f"  {cls}: {len(cmaps[cls])} prontos")

# -- Classificacao leave-one-out ----------------------------------------------

print("\nClassificando (leave-one-out)...")

classes  = list(cmaps.keys())
acertos  = 0
total    = 0
erros_por_classe = {cls: 0 for cls in classes}

for cls_true in classes:
    n = len(cmaps[cls_true])
    for i in range(n):
        # Protótipo de cada classe excluindo o exemplo atual
        dists = {}
        for cls in classes:
            outros = [cmaps[cls][j] for j in range(len(cmaps[cls]))
                      if not (cls == cls_true and j == i)]
            proto = torch.stack(outros).mean(dim=0).view(-1).float()
            dists[cls] = (cmaps[cls_true][i].view(-1).float() - proto).norm().item()
        pred = min(dists, key=dists.get)
        if pred == cls_true:
            acertos += 1
        else:
            erros_por_classe[cls_true] += 1
        total += 1

acc    = acertos / total * 100
chance = 100 / len(classes)

print(f"\n{'='*50}")
print(f"RESULTADO — Classificacao Leave-One-Out")
print(f"{'='*50}")
print(f"  Acuracia : {acc:.1f}%")
print(f"  Chance   : {chance:.1f}%")
print(f"  Ganho    : {acc - chance:+.1f}%")
print(f"  Acertos  : {acertos}/{total}")
print(f"\n  Erros por classe:")
for cls in classes:
    n = len(cmaps[cls])
    e = erros_por_classe[cls]
    print(f"    {cls}: {e}/{n} erros ({(n-e)/n*100:.0f}% acerto)")

# -- Visualizacao -------------------------------------------------------------

nomes = {'N': 'Normal', 'V': 'Ventricular', 'A': 'Atrial'}
N_VIZ = min(5, min(len(v) for v in cmaps.values()))

fig, axes = plt.subplots(len(classes) * 2, N_VIZ, figsize=(4 * N_VIZ, 4 * len(classes)))
fig.suptitle(f'Auditoria 21 — ECG Leave-One-Out\n'
             f'Acuracia={acc:.1f}% | Chance={chance:.1f}% | Ganho={acc-chance:+.1f}%',
             fontsize=12, fontweight='bold')

for row, cls in enumerate(classes):
    axes[row*2][0].set_ylabel(f'{nomes.get(cls,cls)}\nSinal', fontsize=9, fontweight='bold')
    axes[row*2+1][0].set_ylabel(f'{nomes.get(cls,cls)}\nCrystal', fontsize=9, fontweight='bold')
    for col in range(N_VIZ):
        axes[row*2][col].plot(segmentos[cls][col], linewidth=0.8)
        axes[row*2][col].axis('off')
        axes[row*2+1][col].imshow(norm(cmaps[cls][col]), cmap='hot')
        axes[row*2+1][col].axis('off')

plt.tight_layout()
plt.savefig('viz_audit_21_ecg.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_21_ecg.png")
print("Pronto.")
