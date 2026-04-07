"""
Auditoria 22: ECG direto no campo — entrada 1D nativa

Sem Recurrence Plot. O sinal ECG entra diretamente como perturbacao 1D
no campo 2D. Cada ponto temporal vira uma gaussiana numa linha horizontal.

Classifica via leave-one-out com prototipos.
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
import sys
sys.path.insert(0, 'C:/ResNet-Psi')
from resnet_psi import build_gaussians, compute_crystal_maps, DEVICE, FIELD_SIZE

print(f"Dispositivo: {DEVICE}")

# -- Baixar MIT-BIH -----------------------------------------------------------

print("\nBaixando MIT-BIH...")

# Todos os registros MIT-BIH disponiveis
TODOS_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119',
    '121','122','123','124',
    '200','201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221','222','223','228',
    '230','231','232','233','234'
]
# Ultimos 6 reservados para teste
RECORDS_TESTE  = ['230','231','232','233','234','228']
RECORDS_TREINO = [r for r in TODOS_RECORDS if r not in RECORDS_TESTE]

SEG_LEN        = 360
N_POR_CLASSE   = 200  # pega tudo que tiver

def coletar_segmentos(records, n_por_classe):
    segs = {'N': [], 'V': [], 'A': []}
    for rec_id in records:
        try:
            record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
            ann    = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]
            for sample, symbol in zip(ann.sample, ann.symbol):
                if   symbol == 'N'       and len(segs['N']) < n_por_classe: cls = 'N'
                elif symbol == 'V'       and len(segs['V']) < n_por_classe: cls = 'V'
                elif symbol in ('A','a') and len(segs['A']) < n_por_classe: cls = 'A'
                else: continue
                start, end = sample - SEG_LEN // 2, sample + SEG_LEN // 2
                if start < 0 or end > len(signal): continue
                segs[cls].append(signal[start:end])
            print(f"  {rec_id}: { {k: len(v) for k, v in segs.items()} }")
            if all(len(v) >= n_por_classe for v in segs.values()):
                break
        except Exception as e:
            print(f"  {rec_id}: erro ({e})")
    for cls in segs:
        segs[cls] = segs[cls][:n_por_classe]
    return segs

print("Pacientes de TREINO:")
seg_treino = coletar_segmentos(RECORDS_TREINO, N_POR_CLASSE)
print("Pacientes de TESTE:")
seg_teste  = coletar_segmentos(RECORDS_TESTE,  N_POR_CLASSE)

print(f"\nTreino: { {k: len(v) for k, v in seg_treino.items()} }")
print(f"Teste : { {k: len(v) for k, v in seg_teste.items()} }")

# -- Normalizar sinais --------------------------------------------------------

def normalizar(seg):
    s = np.array(seg, dtype=np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return s

# -- Projecao 1D --------------------------------------------------------------

print(f"\nConstruindo projecao 1D ({SEG_LEN} pontos)...")
PG = build_gaussians((SEG_LEN,), field_size=FIELD_SIZE, sigma=0.04)
print(f"  PG shape: {PG.shape}  ({SEG_LEN} gaussianas no campo {FIELD_SIZE}x{FIELD_SIZE})")

# -- Crystal maps -------------------------------------------------------------

print("\nComputando crystal maps...")

classes = list(seg_treino.keys())

def extrair_cmaps(segmentos):
    cmaps = {cls: [] for cls in classes}
    for cls in classes:
        segs_norm = [normalizar(s) for s in segmentos[cls]]
        if len(segs_norm) == 0:
            continue
        X = torch.tensor(np.stack(segs_norm), dtype=torch.float32, device=DEVICE)
        cmaps_cls = compute_crystal_maps(X, PG, field_size=FIELD_SIZE, bs=10)
        cmaps[cls] = list(cmaps_cls)
    return cmaps

print("Crystal maps de treino...")
cmaps_treino = extrair_cmaps(seg_treino)
for cls in classes:
    print(f"  {cls}: {len(cmaps_treino[cls])} prontos")

print("Crystal maps de teste...")
cmaps_teste = extrair_cmaps(seg_teste)
for cls in classes:
    print(f"  {cls}: {len(cmaps_teste[cls])} prontos")

# -- Prototipos = media dos crystal maps de treino ----------------------------

prototipos = {}
for cls in classes:
    prototipos[cls] = torch.stack(cmaps_treino[cls]).mean(dim=0).view(-1).float()

# -- Classificacao: teste em pacientes novos ----------------------------------

print("\nClassificando pacientes novos...")

acertos = 0
total   = 0
erros_por_classe = {cls: 0 for cls in classes}

for cls_true in classes:
    for cmap in cmaps_teste[cls_true]:
        dists = {cls: (cmap.view(-1).float() - prototipos[cls]).norm().item()
                 for cls in classes}
        pred = min(dists, key=dists.get)
        if pred == cls_true:
            acertos += 1
        else:
            erros_por_classe[cls_true] += 1
        total += 1

acc    = acertos / total * 100
chance = 100 / len(classes)

print(f"\n{'='*50}")
print(f"RESULTADO — ECG 1D direto no campo")
print(f"Treino: {RECORDS_TREINO} | Teste: {RECORDS_TESTE}")
print(f"{'='*50}")
print(f"  Acuracia : {acc:.1f}%")
print(f"  Chance   : {chance:.1f}%")
print(f"  Ganho    : {acc - chance:+.1f}%")
print(f"  Acertos  : {acertos}/{total}")
print(f"\n  Por classe:")
for cls in classes:
    n = len(cmaps_teste[cls])
    e = erros_por_classe[cls]
    print(f"    {cls}: {n-e}/{n} acertos ({(n-e)/n*100:.0f}%)")

# -- Visualizacao -------------------------------------------------------------

nomes = {'N': 'Normal', 'V': 'Ventricular', 'A': 'Atrial'}
N_VIZ = min(5, min(len(v) for v in cmaps_teste.values()))

fig, axes = plt.subplots(len(classes) * 2, N_VIZ, figsize=(4 * N_VIZ, 4 * len(classes)))
fig.suptitle(f'Auditoria 22 — ECG 1D direto no campo\n'
             f'Acuracia={acc:.1f}% | Chance={chance:.1f}% | Ganho={acc-chance:+.1f}%\n'
             f'Treino: {RECORDS_TREINO} | Teste: {RECORDS_TESTE}',
             fontsize=11, fontweight='bold')

for row, cls in enumerate(classes):
    axes[row*2][0].set_ylabel(f'{nomes.get(cls,cls)}\nSinal', fontsize=9, fontweight='bold')
    axes[row*2+1][0].set_ylabel(f'{nomes.get(cls,cls)}\nCrystal', fontsize=9, fontweight='bold')
    for col in range(N_VIZ):
        seg = normalizar(seg_teste[cls][col])
        axes[row*2][col].plot(seg, linewidth=0.8)
        axes[row*2][col].axis('off')
        cmap = cmaps_teste[cls][col].cpu().numpy()
        cmap = (cmap - cmap.min()) / (cmap.max() - cmap.min() + 1e-8)
        axes[row*2+1][col].imshow(cmap, cmap='hot')

plt.tight_layout()
plt.savefig('viz_audit_22_ecg_1d.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_22_ecg_1d.png")
print("Pronto.")
