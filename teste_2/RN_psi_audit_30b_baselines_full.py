"""
Auditoria 30b: Baselines com dataset completo

Mesmos baselines do Audit 30, mas com as MESMAS condições
do resultado de 77.4%:
- Treino: 5000 por classe (50000 total)
- Teste: 10000

Se os baselines também derem ~77%, a física não faz diferença.
Se derem menos, a física está fazendo algo.
"""

import torch
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'C:/ResNet-Psi')
from resnet_psi import ResNetPsi, build_gaussians, DEVICE

print(f"Dispositivo: {DEVICE}")

from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

# Dataset completo
print("Carregando dataset completo...")
X_train = np.stack([train_ds[i][0].squeeze(0).numpy().flatten() for i in range(len(train_ds))])
y_train = np.array([train_ds[i][1] for i in range(len(train_ds))])

X_test = np.stack([test_ds[i][0].squeeze(0).numpy().flatten() for i in range(len(test_ds))])
y_test = np.array([test_ds[i][1] for i in range(len(test_ds))])

print(f"  Treino: {len(X_train)}  Teste: {len(X_test)}")

resultados = {}


def prototipos_classificar(X_train, y_train, X_test, y_test, n_classes=10):
    protos = {}
    for c in range(n_classes):
        protos[c] = X_train[y_train == c].mean(axis=0)
    proto_stack = np.stack([protos[c] for c in range(n_classes)])

    # Processar em batches pra não estourar memória
    preds = []
    bs = 1000
    for i in range(0, len(X_test), bs):
        batch = X_test[i:i+bs]
        dists = np.linalg.norm(batch[:, None, :] - proto_stack[None, :, :], axis=2)
        preds.append(dists.argmin(axis=1))
    preds = np.concatenate(preds)
    return (preds == y_test).mean() * 100


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 1: Pixels brutos
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 1: Pixels brutos")
print("="*60)
t0 = time.time()
acc_pixel = prototipos_classificar(X_train, y_train, X_test, y_test)
print(f"  MNIST: {acc_pixel:.1f}%  ({time.time()-t0:.0f}s)")
resultados['Pixels brutos'] = acc_pixel


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 2: Projeção aleatória
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 2: Projeção aleatória (784 → 2304)")
print("="*60)

accs = []
for seed in range(5):
    np.random.seed(seed)
    W = np.random.randn(784, 2304).astype(np.float32) / np.sqrt(784)
    proj_train = X_train @ W
    proj_test = X_test @ W
    acc = prototipos_classificar(proj_train, y_train, proj_test, y_test)
    accs.append(acc)
    print(f"  Seed {seed}: {acc:.1f}%")

acc_rand = np.mean(accs)
print(f"  Média: {acc_rand:.1f}% ± {np.std(accs):.1f}%")
resultados['Projeção aleatória'] = acc_rand


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 3: PCA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 3: PCA (784 → 50)")
print("="*60)
from sklearn.decomposition import PCA

t0 = time.time()
pca = PCA(n_components=50)
pca_train = pca.fit_transform(X_train)
pca_test = pca.transform(X_test)
acc_pca = prototipos_classificar(pca_train, y_train, pca_test, y_test)
print(f"  MNIST: {acc_pca:.1f}%  ({time.time()-t0:.0f}s)")
resultados['PCA (50 comp.)'] = acc_pca


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 4: Random Kitchen Sinks
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 4: Random Kitchen Sinks")
print("="*60)

accs_rks = []
for seed in range(5):
    np.random.seed(seed + 100)
    W_rks = np.random.randn(784, 2304).astype(np.float32) * 0.1
    b_rks = np.random.uniform(0, 2 * np.pi, 2304).astype(np.float32)
    rks_train = np.cos(X_train @ W_rks + b_rks)
    rks_test = np.cos(X_test @ W_rks + b_rks)
    acc = prototipos_classificar(rks_train, y_train, rks_test, y_test)
    accs_rks.append(acc)
    print(f"  Seed {seed}: {acc:.1f}%")

acc_rks = np.mean(accs_rks)
print(f"  Média: {acc_rks:.1f}% ± {np.std(accs_rks):.1f}%")
resultados['Random Kitchen Sinks'] = acc_rks


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 5: Gaussiana sem onda
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 5: Gaussiana sem onda")
print("="*60)
t0 = time.time()
PG = build_gaussians((28, 28)).cpu().numpy()
proj_train = X_train @ PG
proj_test = X_test @ PG
acc_gauss = prototipos_classificar(proj_train, y_train, proj_test, y_test)
print(f"  MNIST: {acc_gauss:.1f}%  ({time.time()-t0:.0f}s)")
resultados['Gaussiana sem onda'] = acc_gauss


# ══════════════════════════════════════════════════════════════════════════════
# ResNet-Psi (dataset completo)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ResNet-Ψ (dataset completo)")
print("="*60)

train_imgs = torch.tensor(X_train.reshape(-1, 28, 28), dtype=torch.float32, device=DEVICE)
test_imgs = torch.tensor(X_test.reshape(-1, 28, 28), dtype=torch.float32, device=DEVICE)

t0 = time.time()
rn = ResNetPsi()
rn.fit(train_imgs, y_train, bs=64)
acc_rn = rn.score(test_imgs, y_test, bs=64)
print(f"  MNIST: {acc_rn:.1f}%  ({time.time()-t0:.0f}s)")
resultados['ResNet-Ψ'] = acc_rn


# ══════════════════════════════════════════════════════════════════════════════
# RESUMO
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESUMO — Baselines (Dataset Completo, 60k treino, 10k teste)")
print(f"{'='*60}")
print(f"  {'Método':<25s}  {'MNIST':>7s}")
print(f"  {'-'*25}  {'-'*7}")

ordem = ['Pixels brutos', 'Projeção aleatória', 'PCA (50 comp.)',
         'Random Kitchen Sinks', 'Gaussiana sem onda', 'ResNet-Ψ']

for nome in ordem:
    r = resultados[nome]
    print(f"  {nome:<25s}  {r:>6.1f}%")

print(f"  {'-'*25}  {'-'*7}")
print(f"  {'Chance':<25s}  {'10.0%':>7s}")
print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfico
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle('Auditoria 30b — ResNet-Ψ vs Baselines (Dataset Completo)',
             fontsize=12, fontweight='bold')

nomes = ordem
accs = [resultados[n] for n in nomes]
cores = ['#9E9E9E', '#9E9E9E', '#9E9E9E', '#9E9E9E', '#FF9800', '#2196F3']

bars = ax.barh(range(len(nomes)), accs, color=cores, alpha=0.85)
ax.set_yticks(range(len(nomes)))
ax.set_yticklabels(nomes, fontsize=11)
ax.set_xlabel('Acurácia (%)', fontsize=11)
ax.axvline(x=10, color='gray', linestyle='--', alpha=0.3)
ax.invert_yaxis()
ax.set_xlim(0, 85)

for i, acc in enumerate(accs):
    ax.text(acc + 0.5, i, f'{acc:.1f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('viz_audit_30b_baselines_full.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_30b_baselines_full.png")
print("Pronto.")
