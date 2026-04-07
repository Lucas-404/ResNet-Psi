"""
Auditoria 25: Few-Shot Classification

Quanto exemplos a ResNet-Psi precisa por classe pra classificar?
Testa 1-shot, 2-shot, 5-shot, 10-shot, 50-shot no MNIST e Fashion-MNIST.

Se 1-shot funcionar, e resultado forte:
nenhum sistema faz classificacao com 1 exemplo e zero treino.
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

# ======================================================================
# Carregar dados
# ======================================================================

from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST('./data', train=True,  download=True, transform=tf)
mnist_test  = datasets.MNIST('./data', train=False, download=True, transform=tf)

fash_train = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
fash_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)

# Teste: 1000 exemplos (rapido mas estatisticamente ok)
N_TEST = 1000

def carregar_teste(ds, n):
    imgs = torch.stack([ds[i][0].squeeze(0) for i in range(n)]).to(DEVICE)
    labs = np.array([ds[i][1] for i in range(n)])
    return imgs, labs

test_imgs_m, test_labs_m = carregar_teste(mnist_test, N_TEST)
test_imgs_f, test_labs_f = carregar_teste(fash_test, N_TEST)

# ======================================================================
# Few-shot: pegar K exemplos por classe
# ======================================================================

def pegar_fewshot(ds, k_per_class, n_classes=10, seed=42):
    """Pega exatamente k exemplos por classe, com seed fixa."""
    rng = np.random.RandomState(seed)

    # Organizar por classe
    por_classe = {c: [] for c in range(n_classes)}
    for i in range(len(ds)):
        _, label = ds[i]
        if len(por_classe[label]) < k_per_class:
            por_classe[label].append(i)
        if all(len(v) >= k_per_class for v in por_classe.values()):
            break

    # Montar tensores
    indices = []
    for c in range(n_classes):
        chosen = por_classe[c][:k_per_class]
        indices.extend(chosen)

    imgs = torch.stack([ds[idx][0].squeeze(0) for idx in indices]).to(DEVICE)
    labs = np.array([ds[idx][1] for idx in indices])
    return imgs, labs

# ======================================================================
# Rodar few-shot
# ======================================================================

SHOTS = [1, 2, 5, 10, 50]

results_mnist = {}
results_fash  = {}

print("\n" + "="*60)
print("FEW-SHOT: MNIST")
print("="*60)

for k in SHOTS:
    train_imgs, train_labs = pegar_fewshot(mnist_train, k)
    rn = ResNetPsi()
    rn.fit(train_imgs, train_labs, bs=32)
    acc = rn.score(test_imgs_m, test_labs_m, bs=64)
    results_mnist[k] = acc
    print(f"  {k:2d}-shot: {acc:.1f}%  ({k*10} exemplos total)")

print("\n" + "="*60)
print("FEW-SHOT: Fashion-MNIST")
print("="*60)

for k in SHOTS:
    train_imgs, train_labs = pegar_fewshot(fash_train, k)
    rn = ResNetPsi()
    rn.fit(train_imgs, train_labs, bs=32)
    acc = rn.score(test_imgs_f, test_labs_f, bs=64)
    results_fash[k] = acc
    print(f"  {k:2d}-shot: {acc:.1f}%  ({k*10} exemplos total)")

# ======================================================================
# Resumo
# ======================================================================

print(f"\n{'='*60}")
print(f"RESUMO — Few-Shot Zero Treino")
print(f"{'='*60}")
print(f"  {'Shots':>5s}  {'MNIST':>7s}  {'Fashion':>7s}  {'Chance':>7s}")
print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")
for k in SHOTS:
    print(f"  {k:5d}  {results_mnist[k]:6.1f}%  {results_fash[k]:6.1f}%  {'10.0%':>7s}")
print(f"{'='*60}")

# ======================================================================
# Grafico
# ======================================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(SHOTS, [results_mnist[k] for k in SHOTS], 'o-', label='MNIST', linewidth=2, markersize=8)
ax.plot(SHOTS, [results_fash[k] for k in SHOTS], 's-', label='Fashion-MNIST', linewidth=2, markersize=8)
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Chance (10%)')
ax.set_xlabel('Exemplos por classe (K-shot)')
ax.set_ylabel('Acuracia (%)')
ax.set_title('ResNet-Psi: Few-Shot Classification (Zero Treino)')
ax.legend()
ax.set_xscale('log')
ax.set_xticks(SHOTS)
ax.set_xticklabels([str(k) for k in SHOTS])
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 85)

plt.tight_layout()
plt.savefig('viz_audit_25_fewshot.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_25_fewshot.png")
print("Pronto.")
