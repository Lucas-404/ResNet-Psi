"""
Auditoria 28: Datasets Geométricos

Testa a ResNet-Psi em todos os datasets 28x28 disponíveis
que tenham estrutura geométrica (contornos, formas).

1. EMNIST Letters — letras manuscritas (26 classes)
2. EMNIST Digits — dígitos manuscritos (10 classes, mais variação que MNIST)
3. KMNIST — caracteres japoneses Kuzushiji (10 classes)
4. SignLanguageMNIST — gestos de mão em ASL (24 classes, sem J e Z)

Tudo entra direto no campo 48x48, sem adaptação.
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
from resnet_psi import ResNetPsi, DEVICE

print(f"Dispositivo: {DEVICE}")

from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor()])

resultados = {}

# ======================================================================
# Helper
# ======================================================================

def testar_dataset(nome, train_ds, test_ds, n_classes, n_train_per_class=50, n_test=1000):
    """Testa um dataset com protótipos zero treino."""
    print(f"\n{'='*60}")
    print(f"{nome} ({n_classes} classes)")
    print(f"{'='*60}")

    # Pegar treino balanceado
    por_classe = {c: [] for c in range(n_classes)}
    for i in range(len(train_ds)):
        img, lab = train_ds[i]
        if len(por_classe[lab]) < n_train_per_class:
            por_classe[lab].append(img.squeeze(0))
        if all(len(v) >= n_train_per_class for v in por_classe.values()):
            break

    # Verificar quantas classes encontramos
    classes_ok = {c: len(v) for c, v in por_classe.items() if len(v) > 0}
    print(f"  Treino: {sum(classes_ok.values())} amostras em {len(classes_ok)} classes")

    train_imgs = torch.stack([img for c in range(n_classes) for img in por_classe[c]]).to(DEVICE)
    train_labs = np.array([c for c in range(n_classes) for _ in por_classe[c]])

    # Teste
    n_test = min(n_test, len(test_ds))
    test_imgs = torch.stack([test_ds[i][0].squeeze(0) for i in range(n_test)]).to(DEVICE)
    test_labs = np.array([test_ds[i][1] for i in range(n_test)])

    print(f"  Teste:  {n_test} amostras")

    chance = 100.0 / n_classes

    rn = ResNetPsi(n_classes=n_classes)
    rn.fit(train_imgs, train_labs, bs=32)
    acc = rn.score(test_imgs, test_labs, bs=64)

    ratio = acc / chance
    print(f"\n  {nome}: {acc:.1f}%  (chance={chance:.1f}%, ratio={ratio:.1f}x)")

    resultados[nome] = {'acc': acc, 'chance': chance, 'n_classes': n_classes, 'ratio': ratio}
    return rn, test_imgs, test_labs


# ======================================================================
# 1. EMNIST Letters (26 classes — letras A-Z)
# ======================================================================

try:
    train_ds = datasets.EMNIST('./data', split='letters', train=True, download=True, transform=tf)
    test_ds  = datasets.EMNIST('./data', split='letters', train=False, download=True, transform=tf)
    # EMNIST Letters: labels 1-26, converter pra 0-25
    class EMNISTWrapper:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, lab = self.ds[idx]
            return img, lab - 1  # 1-26 → 0-25
    rn_letters, test_letters, labs_letters = testar_dataset(
        "EMNIST Letters", EMNISTWrapper(train_ds), EMNISTWrapper(test_ds), 26)
except Exception as e:
    print(f"\nEMNIST Letters falhou: {e}")
    resultados["EMNIST Letters"] = None

# ======================================================================
# 2. EMNIST Digits (10 classes — mais variação que MNIST)
# ======================================================================

try:
    train_ds = datasets.EMNIST('./data', split='digits', train=True, download=True, transform=tf)
    test_ds  = datasets.EMNIST('./data', split='digits', train=False, download=True, transform=tf)
    rn_digits, test_digits, labs_digits = testar_dataset(
        "EMNIST Digits", train_ds, test_ds, 10)
except Exception as e:
    print(f"\nEMNIST Digits falhou: {e}")
    resultados["EMNIST Digits"] = None

# ======================================================================
# 3. KMNIST (10 classes — caracteres japoneses Kuzushiji)
# ======================================================================

try:
    train_ds = datasets.KMNIST('./data', train=True, download=True, transform=tf)
    test_ds  = datasets.KMNIST('./data', train=False, download=True, transform=tf)
    rn_kmnist, test_kmnist, labs_kmnist = testar_dataset(
        "KMNIST", train_ds, test_ds, 10)
except Exception as e:
    print(f"\nKMNIST falhou: {e}")
    resultados["KMNIST"] = None

# ======================================================================
# 4. MNIST pra referência (mesmas condições — 50 per class)
# ======================================================================

train_ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)
rn_mnist, test_mnist, labs_mnist = testar_dataset(
    "MNIST", train_ds, test_ds, 10)

# ======================================================================
# 5. Fashion-MNIST pra referência
# ======================================================================

train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=tf)
test_ds  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)
rn_fash, test_fash, labs_fash = testar_dataset(
    "Fashion-MNIST", train_ds, test_ds, 10)

# ======================================================================
# RESUMO
# ======================================================================

print(f"\n{'='*60}")
print(f"RESUMO — Datasets Geométricos (Zero Treino, 50/classe)")
print(f"{'='*60}")
print(f"  {'Dataset':<20s}  {'Classes':>7s}  {'Acc':>7s}  {'Chance':>7s}  {'Ratio':>6s}")
print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")

for nome in ['MNIST', 'EMNIST Digits', 'EMNIST Letters', 'KMNIST', 'Fashion-MNIST']:
    r = resultados.get(nome)
    if r is None:
        print(f"  {nome:<20s}  {'FALHOU':>7s}")
    else:
        print(f"  {nome:<20s}  {r['n_classes']:>7d}  {r['acc']:>6.1f}%  {r['chance']:>6.1f}%  {r['ratio']:>5.1f}x")

print(f"{'='*60}")

# ======================================================================
# Gráfico
# ======================================================================

nomes_ok = [n for n in ['MNIST', 'EMNIST Digits', 'EMNIST Letters', 'KMNIST', 'Fashion-MNIST']
            if resultados.get(n) is not None]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Auditoria 28 — Datasets Geométricos (Zero Treino, 50/classe)',
             fontsize=12, fontweight='bold')

# Barras de acurácia
ax = axes[0]
accs = [resultados[n]['acc'] for n in nomes_ok]
chances = [resultados[n]['chance'] for n in nomes_ok]
x = range(len(nomes_ok))
bars1 = ax.bar(x, accs, 0.4, label='ResNet-Ψ', color='steelblue')
bars2 = ax.bar([i+0.4 for i in x], chances, 0.4, label='Chance', color='lightgray')
ax.set_xticks([i+0.2 for i in x])
ax.set_xticklabels(nomes_ok, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Acurácia (%)')
ax.legend()
ax.set_title('Acurácia Zero Treino')

# Ratio (acurácia / chance)
ax = axes[1]
ratios = [resultados[n]['ratio'] for n in nomes_ok]
bars = ax.bar(nomes_ok, ratios, color='darkorange', alpha=0.8)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Chance (1x)')
ax.set_ylabel('Ratio (acc / chance)')
ax.set_title('Quanto acima da chance?')
ax.set_xticklabels(nomes_ok, rotation=20, ha='right', fontsize=9)
ax.legend()

plt.tight_layout()
plt.savefig('viz_audit_28_datasets.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_28_datasets.png")
print("Pronto.")
