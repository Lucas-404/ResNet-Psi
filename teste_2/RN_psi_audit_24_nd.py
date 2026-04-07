"""
Auditoria 24: ResNet-Psi N-Dimensional

Testa a arquitetura ND com:
1. MNIST (2D) — confirma que funciona igual à versão original
2. ECG (1D) — campo 1D nativo, sem projeção forçada

A mesma física, sem porteiro.
"""

import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'C:/ResNet-Psi')
from resnet_psi_nd import ResNetPsiND, DEVICE

print(f"Dispositivo: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# TESTE 1: MNIST (2D)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TESTE 1: MNIST — campo 2D (entrada 28x28 → campo 48x48)")
print("="*60)

from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

# Pega subconjunto pra ser rápido
N_TRAIN = 500
N_TEST  = 200

train_imgs = torch.stack([train_ds[i][0].squeeze(0) for i in range(N_TRAIN)]).to(DEVICE)
train_labels = np.array([train_ds[i][1] for i in range(N_TRAIN)])
test_imgs  = torch.stack([test_ds[i][0].squeeze(0) for i in range(N_TEST)]).to(DEVICE)
test_labels = np.array([test_ds[i][1] for i in range(N_TEST)])

rn2d = ResNetPsiND(input_shape=(28, 28), field_shape=(48, 48), n_classes=10)
rn2d.fit(train_imgs, train_labels, bs=32)
acc_mnist = rn2d.score(test_imgs, test_labels, bs=32)
print(f"\n  MNIST 2D: {acc_mnist:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TESTE 2: ECG (1D)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TESTE 2: ECG — campo 1D (entrada 360 → campo 512)")
print("="*60)

import wfdb

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

SEG_LEN = 360

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
                s = signal[start:end].astype(np.float32)
                s = (s - s.min()) / (s.max() - s.min() + 1e-8)
                segs[cls].append(s)
            if all(len(v) >= n for v in segs.values()):
                break
        except: pass
    for cls in segs: segs[cls] = segs[cls][:n]
    return segs

print("\nBaixando MIT-BIH...")
seg_treino = coletar(RECORDS_TREINO, 50)
seg_teste  = coletar(RECORDS_TESTE, 30)
print(f"  Treino: { {k: len(v) for k, v in seg_treino.items()} }")
print(f"  Teste : { {k: len(v) for k, v in seg_teste.items()} }")

# Montar tensores
classes = ['N', 'V', 'A']
cls_to_int = {'N': 0, 'V': 1, 'A': 2}

X_train = torch.tensor(np.stack([s for cls in classes for s in seg_treino[cls]]),
                        dtype=torch.float32, device=DEVICE)
y_train = np.array([cls_to_int[cls] for cls in classes for _ in seg_treino[cls]])

X_test = torch.tensor(np.stack([s for cls in classes for s in seg_teste[cls]]),
                       dtype=torch.float32, device=DEVICE)
y_test = np.array([cls_to_int[cls] for cls in classes for _ in seg_teste[cls]])

rn1d = ResNetPsiND(input_shape=(360,), field_shape=(512,), n_classes=3)
rn1d.fit(X_train, y_train, bs=10)
acc_ecg = rn1d.score(X_test, y_test, bs=10)
print(f"\n  ECG 1D: {acc_ecg:.1f}%")
print(f"  Chance: 33.3%")

# ══════════════════════════════════════════════════════════════════════════════
# RESUMO
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESUMO — ResNet-Ψ N-Dimensional")
print(f"{'='*60}")
print(f"  MNIST  2D (28x28 → 48x48) : {acc_mnist:.1f}%")
print(f"  ECG    1D (360   → 512)   : {acc_ecg:.1f}%  (chance=33.3%)")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════════════════════
# VIZ: crystal maps 1D e 2D lado a lado
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 5, figsize=(20, 6))
fig.suptitle(f'Auditoria 24 — ResNet-Ψ N-Dimensional\n'
             f'MNIST 2D={acc_mnist:.1f}% | ECG 1D={acc_ecg:.1f}%',
             fontsize=12, fontweight='bold')

# MNIST: 5 exemplos
cmaps_mnist = rn2d.extract(test_imgs[:5], bs=5)
for i in range(5):
    cm = cmaps_mnist[i].cpu().numpy()
    cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
    axes[0][i].imshow(cm, cmap='hot')
    axes[0][i].set_title(f'MNIST dig={test_labels[i]}', fontsize=9)
    axes[0][i].axis('off')

# ECG: 5 exemplos (primeiro de cada classe + 2 extras)
ecg_viz = X_test[:5]
cmaps_ecg = rn1d.extract(ecg_viz, bs=5)
labels_viz = y_test[:5]
nomes = {0: 'Normal', 1: 'Ventr.', 2: 'Atrial'}
for i in range(5):
    cm = cmaps_ecg[i].cpu().numpy()
    cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
    axes[1][i].plot(cm, linewidth=0.8)
    axes[1][i].set_title(f'ECG {nomes.get(labels_viz[i], "?")}', fontsize=9)
    axes[1][i].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('viz_audit_24_nd.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_24_nd.png")
print("Pronto.")
