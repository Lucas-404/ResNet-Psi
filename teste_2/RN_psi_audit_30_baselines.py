"""
Auditoria 30: Baselines Formais

Compara a ResNet-Psi com métodos que NÃO usam física:

1. Projeção Aleatória — matriz aleatória do mesmo tamanho (784 → 2304)
   Se der resultado parecido, a física não está fazendo nada.

2. PCA — reduz dimensão e classifica por protótipos
   Baseline clássico de representação sem treino supervisionado.

3. Pixels brutos — protótipos direto nos pixels (sem projeção nenhuma)
   O mínimo absoluto.

4. Random Kitchen Sinks (Rahimi & Recht, 2007) — features aleatórias
   Aproximação de kernel via projeção não-linear aleatória.

Todos usam protótipos + distância euclidiana, mesma condição que a ResNet-Psi.
Se a ResNet-Psi não ganhar desses, o campo não está fazendo nada especial.
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
from resnet_psi import ResNetPsi, DEVICE

print(f"Dispositivo: {DEVICE}")

from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

fash_train = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
fash_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)


def carregar(ds, n_per_class=50, n_classes=10):
    por_classe = {c: [] for c in range(n_classes)}
    for i in range(len(ds)):
        img, lab = ds[i]
        if len(por_classe[lab]) < n_per_class:
            por_classe[lab].append(img.squeeze(0).numpy().flatten())
        if all(len(v) >= n_per_class for v in por_classe.values()):
            break
    X = np.vstack([np.stack(por_classe[c]) for c in range(n_classes)])
    y = np.array([c for c in range(n_classes) for _ in por_classe[c]])
    return X, y


def carregar_teste(ds, n=1000):
    X = np.stack([ds[i][0].squeeze(0).numpy().flatten() for i in range(n)])
    y = np.array([ds[i][1] for i in range(n)])
    return X, y


def prototipos_classificar(X_train, y_train, X_test, y_test, n_classes=10):
    """Protótipos + distância euclidiana (mesmo método da ResNet-Psi)."""
    protos = {}
    for c in range(n_classes):
        protos[c] = X_train[y_train == c].mean(axis=0)

    proto_stack = np.stack([protos[c] for c in range(n_classes)])
    dists = np.linalg.norm(X_test[:, None, :] - proto_stack[None, :, :], axis=2)
    preds = dists.argmin(axis=1)
    acc = (preds == y_test).mean() * 100
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# Carregar dados
# ══════════════════════════════════════════════════════════════════════════════

X_train_m, y_train_m = carregar(train_ds, 50)
X_test_m, y_test_m = carregar_teste(test_ds, 1000)

X_train_f, y_train_f = carregar(fash_train, 50)
X_test_f, y_test_f = carregar_teste(fash_test, 1000)

resultados = {}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 1: Pixels brutos
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 1: Pixels brutos (protótipos direto)")
print("="*60)

acc_pixel_m = prototipos_classificar(X_train_m, y_train_m, X_test_m, y_test_m)
acc_pixel_f = prototipos_classificar(X_train_f, y_train_f, X_test_f, y_test_f)
print(f"  MNIST:   {acc_pixel_m:.1f}%")
print(f"  Fashion: {acc_pixel_f:.1f}%")
resultados['Pixels brutos'] = {'mnist': acc_pixel_m, 'fashion': acc_pixel_f}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 2: Projeção aleatória (mesma dimensão que o campo 48x48 = 2304)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 2: Projeção aleatória (784 → 2304)")
print("="*60)

np.random.seed(42)
DIM_OUT = 48 * 48  # mesma dimensão do crystal map

# Testar 5 seeds pra ser justo
accs_rand_m = []
accs_rand_f = []
for seed in range(5):
    np.random.seed(seed)
    W = np.random.randn(784, DIM_OUT).astype(np.float32) / np.sqrt(784)

    proj_train_m = X_train_m @ W
    proj_test_m = X_test_m @ W
    acc = prototipos_classificar(proj_train_m, y_train_m, proj_test_m, y_test_m)
    accs_rand_m.append(acc)

    proj_train_f = X_train_f @ W
    proj_test_f = X_test_f @ W
    acc = prototipos_classificar(proj_train_f, y_train_f, proj_test_f, y_test_f)
    accs_rand_f.append(acc)

acc_rand_m = np.mean(accs_rand_m)
acc_rand_f = np.mean(accs_rand_f)
print(f"  MNIST:   {acc_rand_m:.1f}% ± {np.std(accs_rand_m):.1f}%  (5 seeds)")
print(f"  Fashion: {acc_rand_f:.1f}% ± {np.std(accs_rand_f):.1f}%  (5 seeds)")
resultados['Projeção aleatória'] = {'mnist': acc_rand_m, 'fashion': acc_rand_f}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 3: PCA (mesma dimensão)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 3: PCA (784 → 50 componentes)")
print("="*60)

from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca_train_m = pca.fit_transform(X_train_m)
pca_test_m = pca.transform(X_test_m)
acc_pca_m = prototipos_classificar(pca_train_m, y_train_m, pca_test_m, y_test_m)

pca_f = PCA(n_components=50)
pca_train_f = pca_f.fit_transform(X_train_f)
pca_test_f = pca_f.transform(X_test_f)
acc_pca_f = prototipos_classificar(pca_train_f, y_train_f, pca_test_f, y_test_f)

print(f"  MNIST:   {acc_pca_m:.1f}%")
print(f"  Fashion: {acc_pca_f:.1f}%")
resultados['PCA (50 comp.)'] = {'mnist': acc_pca_m, 'fashion': acc_pca_f}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 4: Random Kitchen Sinks (features não-lineares aleatórias)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 4: Random Kitchen Sinks (784 → 2304, RBF)")
print("="*60)

accs_rks_m = []
accs_rks_f = []
for seed in range(5):
    np.random.seed(seed + 100)
    W_rks = np.random.randn(784, DIM_OUT).astype(np.float32) * 0.1
    b_rks = np.random.uniform(0, 2 * np.pi, DIM_OUT).astype(np.float32)

    # Features: cos(X @ W + b) — aproxima kernel RBF
    rks_train_m = np.cos(X_train_m @ W_rks + b_rks)
    rks_test_m = np.cos(X_test_m @ W_rks + b_rks)
    acc = prototipos_classificar(rks_train_m, y_train_m, rks_test_m, y_test_m)
    accs_rks_m.append(acc)

    rks_train_f = np.cos(X_train_f @ W_rks + b_rks)
    rks_test_f = np.cos(X_test_f @ W_rks + b_rks)
    acc = prototipos_classificar(rks_train_f, y_train_f, rks_test_f, y_test_f)
    accs_rks_f.append(acc)

acc_rks_m = np.mean(accs_rks_m)
acc_rks_f = np.mean(accs_rks_f)
print(f"  MNIST:   {acc_rks_m:.1f}% ± {np.std(accs_rks_m):.1f}%  (5 seeds)")
print(f"  Fashion: {acc_rks_f:.1f}% ± {np.std(accs_rks_f):.1f}%  (5 seeds)")
resultados['Random Kitchen Sinks'] = {'mnist': acc_rks_m, 'fashion': acc_rks_f}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 5: Projeção gaussiana SEM física (só a projeção, sem onda)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 5: Projeção gaussiana SEM onda (só gaussianas)")
print("="*60)

from resnet_psi import build_gaussians

PG = build_gaussians((28, 28)).cpu().numpy()

proj_train_m = X_train_m @ PG
proj_test_m = X_test_m @ PG
acc_gauss_m = prototipos_classificar(proj_train_m, y_train_m, proj_test_m, y_test_m)

proj_train_f = X_train_f @ PG
proj_test_f = X_test_f @ PG
acc_gauss_f = prototipos_classificar(proj_train_f, y_train_f, proj_test_f, y_test_f)

print(f"  MNIST:   {acc_gauss_m:.1f}%")
print(f"  Fashion: {acc_gauss_f:.1f}%")
resultados['Gaussiana sem onda'] = {'mnist': acc_gauss_m, 'fashion': acc_gauss_f}


# ══════════════════════════════════════════════════════════════════════════════
# ResNet-Psi (referência)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ResNet-Ψ (referência)")
print("="*60)

train_imgs_m = torch.tensor(X_train_m.reshape(-1, 28, 28), dtype=torch.float32, device=DEVICE)
test_imgs_m = torch.tensor(X_test_m.reshape(-1, 28, 28), dtype=torch.float32, device=DEVICE)

rn = ResNetPsi()
rn.fit(train_imgs_m, y_train_m, bs=32)
acc_rn_m = rn.score(test_imgs_m, y_test_m, bs=64)

train_imgs_f = torch.tensor(X_train_f.reshape(-1, 28, 28), dtype=torch.float32, device=DEVICE)
test_imgs_f = torch.tensor(X_test_f.reshape(-1, 28, 28), dtype=torch.float32, device=DEVICE)

rn_f = ResNetPsi()
rn_f.fit(train_imgs_f, y_train_f, bs=32)
acc_rn_f = rn_f.score(test_imgs_f, y_test_f, bs=64)

print(f"  MNIST:   {acc_rn_m:.1f}%")
print(f"  Fashion: {acc_rn_f:.1f}%")
resultados['ResNet-Ψ'] = {'mnist': acc_rn_m, 'fashion': acc_rn_f}


# ══════════════════════════════════════════════════════════════════════════════
# RESUMO
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESUMO — Baselines Formais (Protótipos + Euclidiana)")
print(f"{'='*60}")
print(f"  {'Método':<25s}  {'MNIST':>7s}  {'Fashion':>7s}")
print(f"  {'-'*25}  {'-'*7}  {'-'*7}")

ordem = ['Pixels brutos', 'Projeção aleatória', 'PCA (50 comp.)',
         'Random Kitchen Sinks', 'Gaussiana sem onda', 'ResNet-Ψ']

for nome in ordem:
    r = resultados[nome]
    print(f"  {nome:<25s}  {r['mnist']:>6.1f}%  {r['fashion']:>6.1f}%")

print(f"  {'-'*25}  {'-'*7}  {'-'*7}")
print(f"  {'Chance':<25s}  {'10.0%':>7s}  {'10.0%':>7s}")
print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfico
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Auditoria 30 — ResNet-Ψ vs Baselines (Protótipos + Euclidiana, Zero Treino)',
             fontsize=12, fontweight='bold')

nomes = ordem
cores = ['#9E9E9E', '#9E9E9E', '#9E9E9E', '#9E9E9E', '#FF9800', '#2196F3']

for ax_idx, (ds_nome, ds_key) in enumerate([('MNIST', 'mnist'), ('Fashion-MNIST', 'fashion')]):
    ax = axes[ax_idx]
    accs = [resultados[n][ds_key] for n in nomes]
    bars = ax.barh(range(len(nomes)), accs, color=cores, alpha=0.85)
    ax.set_yticks(range(len(nomes)))
    ax.set_yticklabels(nomes, fontsize=10)
    ax.set_xlabel('Acurácia (%)')
    ax.set_title(ds_nome, fontsize=12)
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 85)

    for i, acc in enumerate(accs):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('viz_audit_30_baselines.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_30_baselines.png")
print("Pronto.")
