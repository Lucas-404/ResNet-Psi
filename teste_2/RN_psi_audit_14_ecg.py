"""
Auditoria 14: ECG — Detecção de Arritmia via Protótipos Cristalinos

MIT-BIH Arrhythmia Dataset via wfdb.
Sinal 1D de ECG → projeção 2D temporal → campo de ondas → cristalização.

Cada ponto do sinal vira uma gaussiana na posição horizontal do campo.
Amplitude do sinal = amplitude da gaussiana.
O campo captura a estrutura temporal do batimento.

Zero decoder. Zero treino. Só física + média por classe.
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

import wfdb
from wfdb import processing

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

# MIT-BIH classes principais
# N=Normal, V=PVC, A=APB, R=RBBB, L=LBBB
BEAT_LABELS = {
    'N': 0,  # Normal
    'V': 1,  # Ventricular premature contraction
    'A': 2,  # Atrial premature beat
    'R': 3,  # Right bundle branch block
    'L': 4,  # Left bundle branch block
}
CLASS_NAMES = ['Normal', 'PVC', 'APB', 'RBBB', 'LBBB']
N_CLASSES = len(BEAT_LABELS)

# Segmento de batimento: 180 amostras centradas no pico R (~0.5s a 360Hz)
BEAT_LEN = 180


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


def build_ecg_projection(beat_len=BEAT_LEN, field_size=FIELD_SIZE, sigma=0.04):
    """
    Projeção trajetória: gaussianas apenas no eixo temporal (X).
    A posição Y é determinada pela amplitude do sinal em runtime.
    O batimento desenha uma curva no campo 2D.
    """
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, _ = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for i in range(beat_len):
        cx = 0.1 + 0.8 * i / (beat_len - 1)
        gs.append(torch.exp(-((xg - cx)**2) / (2*sigma**2)))
    return torch.stack(gs)  # (beat_len, FS, FS)


def beats_to_field(beats, PG_base, field_size=FIELD_SIZE, sigma=0.04):
    """
    beats: (B, beat_len)
    Para cada ponto i, posição Y = amplitude normalizada → [0.1, 0.9]
    O sinal desenha uma trajetória 2D no campo.
    """
    B, L = beats.shape
    coords = torch.linspace(0., 1., field_size, device=beats.device)
    _, yg = torch.meshgrid(coords, coords, indexing='ij')

    amp_min = beats.min(dim=1, keepdim=True)[0]
    amp_max = beats.max(dim=1, keepdim=True)[0]
    beats_norm = 0.1 + 0.8 * (beats - amp_min) / (amp_max - amp_min + 1e-8)

    pert = torch.zeros(B, field_size, field_size, device=beats.device)
    for i in range(L):
        cy = beats_norm[:, i].view(B, 1, 1)
        gauss_y = torch.exp(-((yg.unsqueeze(0) - cy)**2) / (2*sigma**2))
        gauss_x = PG_base[i].unsqueeze(0)
        pert += gauss_x * gauss_y
    return pert


def compute_crystal_maps_batch(X, PG, bs=64):
    """X: (B, beat_len) — sinais ECG"""
    N, out = len(X), []
    for i in range(0, N, bs):
        B = X[i:i+bs]
        pert = beats_to_field(B.float(), PG)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(len(B), FIELD_SIZE)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)
        out.append(mem.crystal_map)
    return torch.cat(out, dim=0)


# ── Carregar MIT-BIH ─────────────────────────────────────────────────────────

print("\nCarregando MIT-BIH Arrhythmia Dataset...")

# Registros do MIT-BIH (48 registros, usamos os principais)
RECORDS = ['100', '101', '102', '103', '104', '105', '106', '107',
           '108', '109', '111', '112', '113', '114', '115', '116',
           '117', '118', '119', '121', '122', '123', '124', '200',
           '201', '202', '203', '205', '207', '208', '209', '210',
           '212', '213', '214', '215', '217', '219', '220', '221',
           '222', '223', '228', '230', '231', '232', '233', '234']

beats_by_class = {i: [] for i in range(N_CLASSES)}
total_loaded = 0

for idx, rec in enumerate(RECORDS):
    try:
        record = wfdb.rdrecord(rec, pn_dir='mitdb')
        annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')

        signal = record.p_signal[:, 0]  # canal MLII
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        symbols = annotation.symbol
        samples = annotation.sample

        rec_count = 0
        for sym, samp in zip(symbols, samples):
            if sym not in BEAT_LABELS:
                continue
            cls = BEAT_LABELS[sym]

            start = samp - BEAT_LEN // 2
            end   = samp + BEAT_LEN // 2
            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end].astype(np.float32)
            beats_by_class[cls].append(beat)
            total_loaded += 1
            rec_count += 1

        print(f"  [{idx+1:2d}/{len(RECORDS)}] {rec}... {rec_count} batimentos")
    except Exception as e:
        print(f"  [{idx+1:2d}/{len(RECORDS)}] {rec}... erro: {e}")
        continue

print(f"  Total de batimentos carregados: {total_loaded}")
for cls in range(N_CLASSES):
    print(f"  {CLASS_NAMES[cls]:>10}: {len(beats_by_class[cls])} batimentos")

# Verificar classes com amostras suficientes
valid_classes = [cls for cls in range(N_CLASSES) if len(beats_by_class[cls]) >= 20]
print(f"\n  Classes válidas (≥20 amostras): {[CLASS_NAMES[c] for c in valid_classes]}")

PG = build_ecg_projection()

# ── Experimento ──────────────────────────────────────────────────────────────

N_PROTO_LIST = [10, 50, 200]
N_TEST_PER_CLASS = 20

print(f"\n{'='*70}")
print("AUDITORIA 14: ECG MIT-BIH — Protótipos Cristalinos")
print(f"Classes: {[CLASS_NAMES[c] for c in valid_classes]}")
print(f"Teste: {N_TEST_PER_CLASS} batimentos por classe | Zero treino")
print(f"Referência aleatório: {100/len(valid_classes):.1f}%")
print(f"{'='*70}")

# Separar treino/teste
train_by_class = {}
test_by_class  = {}
for cls in valid_classes:
    beats = beats_by_class[cls]
    np.random.seed(42)
    np.random.shuffle(beats)
    test_by_class[cls]  = beats[:N_TEST_PER_CLASS]
    train_by_class[cls] = beats[N_TEST_PER_CLASS:]

# Pré-computar crystal_maps de teste
print(f"\nPré-computando crystal_maps de teste...")
test_tensors = []
test_labels  = []
for cls in valid_classes:
    for beat in test_by_class[cls]:
        test_tensors.append(torch.tensor(beat))
        test_labels.append(cls)
test_tensor = torch.stack(test_tensors).to(DEVICE)
test_labels = np.array(test_labels)
t1 = time.time()
test_cmaps = compute_crystal_maps_batch(test_tensor, PG)
print(f"  Pronto: {time.time()-t1:.0f}s  ({len(test_tensor)} batimentos)")

all_results = []
t0 = time.time()

for n_proto in N_PROTO_LIST:
    # Verificar se todas as classes têm exemplos suficientes
    classes_ok = [cls for cls in valid_classes if len(train_by_class[cls]) >= n_proto]
    if len(classes_ok) < 2:
        print(f"\n── {n_proto} exemplos: classes insuficientes, pulando")
        continue

    print(f"\n── {n_proto} exemplos por protótipo ({len(classes_ok)} classes) ──")
    t1 = time.time()

    prototypes = {}
    n_crystals = []
    for cls in classes_ok:
        beats = train_by_class[cls][:n_proto]
        tensor = torch.tensor(np.array(beats)).to(DEVICE)
        cmaps = compute_crystal_maps_batch(tensor, PG)
        prototypes[cls] = cmaps.mean(dim=0)
        n_crystals.append((prototypes[cls] > 0.01).float().sum().item())
    print(f"  Protótipos prontos: {time.time()-t1:.0f}s | Cristais/proto: {np.mean(n_crystals):.0f}")

    # Classificar apenas exemplos de teste das classes com protótipo
    mask = np.isin(test_labels, classes_ok)
    test_flat = test_cmaps[mask].view(mask.sum(), -1).float()
    labels_sub = test_labels[mask]

    proto_matrix = torch.stack([prototypes[cls] for cls in classes_ok]).view(len(classes_ok), -1).float()
    dists = torch.cdist(test_flat, proto_matrix)
    pred_idx = dists.argmin(dim=1).cpu().numpy()
    preds = np.array([classes_ok[i] for i in pred_idx])
    acc = (preds == labels_sub).mean() * 100

    per_class_acc = {}
    for cls in classes_ok:
        mask_cls = labels_sub == cls
        if mask_cls.sum() > 0:
            per_class_acc[cls] = (preds[mask_cls] == labels_sub[mask_cls]).mean() * 100

    all_results.append({
        'n_proto': n_proto,
        'acc': acc,
        'classes': classes_ok,
        'per_class_acc': per_class_acc,
        'avg_crystals': np.mean(n_crystals),
    })

    print(f"  → Acurácia geral: {acc:.1f}%  (aleatório: {100/len(classes_ok):.1f}%)")
    for cls in classes_ok:
        print(f"    {CLASS_NAMES[cls]:>10}: {per_class_acc.get(cls, 0):.0f}%")


# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESULTADO FINAL: ECG MIT-BIH")
print(f"{'='*70}")
for r in all_results:
    n_cls = len(r['classes'])
    print(f"  {r['n_proto']:>4} exemplos | {n_cls} classes | Acurácia: {r['acc']:.1f}%  (aleatório: {100/n_cls:.1f}%)")
print(f"\nTempo total: {time.time()-t0:.0f}s")


# ── Plot ─────────────────────────────────────────────────────────────────────

if not all_results:
    print("Sem resultados para plotar.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Auditoria 14 — ECG MIT-BIH: Protótipos Cristalinos (zero treino)', fontsize=13)

    # Curva de acurácia
    ax = axes[0]
    ns  = [r['n_proto'] for r in all_results]
    acc = [r['acc'] for r in all_results]
    ax.plot(ns, acc, 'o-', color='#e6194b', linewidth=2, markersize=8)
    ax.axhline(y=100/len(valid_classes), color='gray', linestyle='--', alpha=0.6,
               label=f'Aleatório ({100/len(valid_classes):.1f}%)')
    ax.set_xlabel('Exemplos por protótipo')
    ax.set_ylabel('Acurácia (%)')
    ax.set_title('ECG: Acurácia vs N exemplos')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Acurácia por classe (último ponto)
    ax = axes[1]
    last = all_results[-1]
    cls_names = [CLASS_NAMES[c] for c in last['classes']]
    cls_accs  = [last['per_class_acc'].get(c, 0) for c in last['classes']]
    ax.bar(cls_names, cls_accs, color='#e6194b', alpha=0.7)
    ax.axhline(y=last['acc'], color='red', linestyle='--', alpha=0.7,
               label=f'Média ({last["acc"]:.1f}%)')
    ax.axhline(y=100/len(last['classes']), color='gray', linestyle='--', alpha=0.5,
               label='Aleatório')
    ax.set_ylabel('Acurácia (%)')
    ax.set_title(f'Acurácia por tipo de batimento ({last["n_proto"]} exemplos/proto)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('viz_audit_14_ecg.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n-> viz_audit_14_ecg.png")

    # Visualização dos protótipos no campo
    last_proto_result = all_results[-1]
    n_cls_viz = len(last_proto_result['classes'])
    fig, axes = plt.subplots(2, n_cls_viz, figsize=(4*n_cls_viz, 8))
    fig.suptitle('Protótipos Cristalinos — ECG MIT-BIH\nO que o campo "vê" de cada tipo de batimento', fontsize=13)

    # Recomputar protótipos finais para visualização
    prototypes_viz = {}
    for cls in last_proto_result['classes']:
        beats = train_by_class[cls][:last_proto_result['n_proto']]
        tensor = torch.tensor(np.array(beats)).to(DEVICE)
        cmaps = compute_crystal_maps_batch(tensor, PG)
        prototypes_viz[cls] = cmaps.mean(dim=0)

    global_proto = torch.stack([prototypes_viz[c] for c in last_proto_result['classes']]).mean(dim=0)

    for i, cls in enumerate(last_proto_result['classes']):
        # Protótipo bruto
        ax = axes[0][i]
        ax.imshow(prototypes_viz[cls].cpu().numpy(), cmap='hot', aspect='auto')
        ax.set_title(f"{CLASS_NAMES[cls]}\n({last_proto_result['per_class_acc'].get(cls,0):.0f}%)", fontsize=10)
        ax.axis('off')

        # Protótipo diferencial
        ax = axes[1][i]
        diff = (prototypes_viz[cls] - global_proto).cpu().numpy()
        vmax = np.abs(diff).max()
        ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title('Diferencial', fontsize=8)
        ax.axis('off')

    axes[0][0].set_ylabel('Protótipo', fontsize=9)
    axes[1][0].set_ylabel('Diferencial', fontsize=9)
    plt.tight_layout()
    plt.savefig('viz_audit_14_ecg_prototipos.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"-> viz_audit_14_ecg_prototipos.png")

print("Pronto.")
