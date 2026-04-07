"""
Auditoria 23: ECG temporal — sinal entra ponto a ponto no campo

Cada step do campo recebe o proximo ponto do sinal ECG como perturbacao.
A dinamica temporal do sinal vira dinamica ondulatoria no campo.

Diferente do audit 22 onde todos os pontos entravam de uma vez (sem ordem temporal).
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
    def __init__(self, FS=FIELD_SIZE):
        self.crystal_map = torch.zeros(1, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(1, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(1, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(1, FS, FS, device=DEVICE)
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


def psi_step_temporal(field, velocity, valor, source_mask):
    """
    Step da equacao de onda com injecao temporal.
    valor: escalar — valor do sinal nesse instante
    source_mask: (1, FS, FS) — onde no campo o sinal e injetado
    """
    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                          device=DEVICE).view(1, 1, 3, 3).to(field.dtype)
    # Injeta o valor do sinal como perturbacao na posicao central
    field = field + source_mask * valor * (_DT * 0.5)
    lap = F.conv2d(F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular'), lap_k).squeeze(1)
    acc = _C2 * lap - _GAMMA * velocity + _ALPHA * torch.tanh(field) * field - _BETA * field * field**2
    velocity = torch.clamp(velocity + acc * _DT, -5., 5.)
    field    = torch.clamp(field + velocity * _DT, -10., 10.)
    return field, velocity


def build_source_mask(field_size=FIELD_SIZE, sigma=0.08):
    """Gaussiana central — onde o sinal e injetado no campo."""
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    cx, cy = 0.5, 0.5
    mask = torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))
    return mask.unsqueeze(0)  # (1, FS, FS)


def run_field_temporal(signal):
    """
    Processa sinal ECG ponto a ponto.
    signal: array 1D normalizado
    Retorna crystal_map (FS, FS)
    """
    source_mask = build_source_mask()
    f = torch.zeros(1, FIELD_SIZE, FIELD_SIZE, device=DEVICE)
    v = torch.zeros(1, FIELD_SIZE, FIELD_SIZE, device=DEVICE)
    mem = CrystalCompetitivo()

    with torch.no_grad():
        for t, valor in enumerate(signal):
            f, v = psi_step_temporal(f, v, float(valor), source_mask)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)

    return mem.crystal_map.squeeze(0)


def normalizar(seg):
    s = np.array(seg, dtype=np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return s - 0.5  # centra em zero


def norm_viz(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = x - x.min()
    return x / (x.max() + 1e-8)


# -- Baixar MIT-BIH -----------------------------------------------------------

print("\nBaixando MIT-BIH...")

TODOS_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119',
    '121','122','123','124',
    '200','201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221','222','223',
    '230','231','232','233','234','228'
]
RECORDS_TESTE  = ['230','231','232','233','234','228']
RECORDS_TREINO = [r for r in TODOS_RECORDS if r not in RECORDS_TESTE]

SEG_LEN      = 360
N_POR_CLASSE = 50

def coletar(records, n):
    segs = {'N': [], 'V': [], 'A': []}
    for rec_id in records:
        try:
            record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
            ann    = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]
            for sample, symbol in zip(ann.sample, ann.symbol):
                if   symbol == 'N'       and len(segs['N']) < n: cls = 'N'
                elif symbol == 'V'       and len(segs['V']) < n: cls = 'V'
                elif symbol in ('A','a') and len(segs['A']) < n: cls = 'A'
                else: continue
                start, end = sample - SEG_LEN//2, sample + SEG_LEN//2
                if start < 0 or end > len(signal): continue
                segs[cls].append(normalizar(signal[start:end]))
            if all(len(v) >= n for v in segs.values()):
                break
        except: pass
    for cls in segs: segs[cls] = segs[cls][:n]
    return segs

print("Treino...")
seg_treino = coletar(RECORDS_TREINO, N_POR_CLASSE)
print(f"  { {k: len(v) for k, v in seg_treino.items()} }")

print("Teste...")
seg_teste = coletar(RECORDS_TESTE, 30)
print(f"  { {k: len(v) for k, v in seg_teste.items()} }")

# -- Crystal maps -------------------------------------------------------------

classes = ['N', 'V', 'A']

print("\nComputando crystal maps de treino (ponto a ponto)...")
cmaps_treino = {cls: [] for cls in classes}
for cls in classes:
    for i, seg in enumerate(seg_treino[cls]):
        cmaps_treino[cls].append(run_field_temporal(seg))
    print(f"  {cls}: {len(cmaps_treino[cls])} prontos")

print("\nComputando crystal maps de teste...")
cmaps_teste = {cls: [] for cls in classes}
for cls in classes:
    for seg in seg_teste[cls]:
        cmaps_teste[cls].append(run_field_temporal(seg))
    print(f"  {cls}: {len(cmaps_teste[cls])} prontos")

# -- Prototipos + classificacao -----------------------------------------------

prototipos = {cls: torch.stack(cmaps_treino[cls]).mean(dim=0).view(-1).float()
              for cls in classes}

print("\nClassificando...")
acertos = 0
total   = 0
erros   = {cls: 0 for cls in classes}

for cls_true in classes:
    for cmap in cmaps_teste[cls_true]:
        dists = {cls: (cmap.view(-1).float() - prototipos[cls]).norm().item()
                 for cls in classes}
        pred = min(dists, key=dists.get)
        if pred == cls_true: acertos += 1
        else: erros[cls_true] += 1
        total += 1

acc    = acertos / total * 100
chance = 100 / len(classes)

print(f"\n{'='*50}")
print(f"RESULTADO — ECG temporal ponto a ponto")
print(f"{'='*50}")
print(f"  Acuracia : {acc:.1f}%")
print(f"  Chance   : {chance:.1f}%")
print(f"  Ganho    : {acc - chance:+.1f}%")
print(f"  Acertos  : {acertos}/{total}")
print(f"\n  Por classe:")
for cls in classes:
    n = len(cmaps_teste[cls])
    e = erros[cls]
    print(f"    {cls}: {n-e}/{n} ({(n-e)/n*100:.0f}%)")

# -- Visualizacao -------------------------------------------------------------

nomes = {'N': 'Normal', 'V': 'Ventricular', 'A': 'Atrial'}
N_VIZ = min(5, min(len(v) for v in cmaps_teste.values()))

fig, axes = plt.subplots(len(classes) * 2, N_VIZ, figsize=(4*N_VIZ, 4*len(classes)))
fig.suptitle(f'Auditoria 23 — ECG temporal ponto a ponto\n'
             f'Acuracia={acc:.1f}% | Chance={chance:.1f}% | Ganho={acc-chance:+.1f}%',
             fontsize=12, fontweight='bold')

for row, cls in enumerate(classes):
    axes[row*2][0].set_ylabel(f'{nomes[cls]}\nSinal', fontsize=9, fontweight='bold')
    axes[row*2+1][0].set_ylabel(f'{nomes[cls]}\nCrystal', fontsize=9, fontweight='bold')
    for col in range(N_VIZ):
        axes[row*2][col].plot(seg_teste[cls][col], linewidth=0.8)
        axes[row*2][col].axis('off')
        axes[row*2+1][col].imshow(norm_viz(cmaps_teste[cls][col]), cmap='hot')
        axes[row*2+1][col].axis('off')

plt.tight_layout()
plt.savefig('viz_audit_23_ecg_temporal.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_23_ecg_temporal.png")
print("Pronto.")
