"""
Auditoria 26: Detecção de Anomalia

Mostra só exemplos "normais" (um dígito) pro campo.
Testa se anomalias (outros dígitos) geram crystal maps diferentes
sem nunca ter visto uma anomalia.

Teste 1: Treina só com "1". Testa com "1" (normal) e outros (anomalia).
Teste 2: Treina só com "sneaker". Testa com "sneaker" vs outros.

Métrica: distância ao protótipo normal.
Se anomalias ficam mais longe, a ResNet-Psi detecta sem saber o que é.
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
from resnet_psi import ResNetPsi, compute_crystal_maps, DEVICE

print(f"Dispositivo: {DEVICE}")

from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST('./data', train=True,  download=True, transform=tf)
mnist_test  = datasets.MNIST('./data', train=False, download=True, transform=tf)

fash_train = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
fash_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)


def pegar_por_classe(ds, classe, n):
    """Pega n exemplos de uma classe específica."""
    imgs = []
    for i in range(len(ds)):
        _, label = ds[i]
        if label == classe:
            imgs.append(ds[i][0].squeeze(0))
            if len(imgs) >= n:
                break
    return torch.stack(imgs).to(DEVICE)


def pegar_exceto_classe(ds, classe, n_per_class, n_classes=10):
    """Pega n exemplos de cada classe EXCETO a classe normal."""
    imgs = []
    labs = []
    contagem = {c: 0 for c in range(n_classes) if c != classe}
    for i in range(len(ds)):
        _, label = ds[i]
        if label != classe and contagem.get(label, n_per_class) < n_per_class:
            imgs.append(ds[i][0].squeeze(0))
            labs.append(label)
            contagem[label] += 1
            if all(v >= n_per_class for v in contagem.values()):
                break
    return torch.stack(imgs).to(DEVICE), np.array(labs)


def teste_anomalia(ds_train, ds_test, classe_normal, nome_classe, nome_dataset, n_train=100, n_test_normal=100, n_test_anomalia=20):
    """
    Treina protótipo com uma classe.
    Mede distância de normais e anomalias ao protótipo.
    """
    print(f"\n  Classe normal: {classe_normal} ({nome_classe})")

    # Protótipo: média dos crystal maps da classe normal
    train_imgs = pegar_por_classe(ds_train, classe_normal, n_train)
    rn = ResNetPsi()
    cmaps_train = rn.extract(train_imgs, bs=32)
    prototipo = cmaps_train.mean(dim=0).flatten()

    # Normais no teste
    test_normal = pegar_por_classe(ds_test, classe_normal, n_test_normal)
    cmaps_normal = rn.extract(test_normal, bs=32)

    # Anomalias no teste
    test_anomalia, labs_anomalia = pegar_exceto_classe(ds_test, classe_normal, n_test_anomalia)
    cmaps_anomalia = rn.extract(test_anomalia, bs=32)

    # Distâncias ao protótipo
    dist_normal = torch.norm(cmaps_normal.view(len(cmaps_normal), -1) - prototipo.unsqueeze(0), dim=1).cpu().numpy()
    dist_anomalia = torch.norm(cmaps_anomalia.view(len(cmaps_anomalia), -1) - prototipo.unsqueeze(0), dim=1).cpu().numpy()

    # Estatísticas
    mean_n = dist_normal.mean()
    mean_a = dist_anomalia.mean()
    std_n  = dist_normal.std()
    ratio  = mean_a / (mean_n + 1e-8)

    print(f"    Dist normal:   {mean_n:.2f} ± {std_n:.2f}")
    print(f"    Dist anomalia: {mean_a:.2f} ± {dist_anomalia.std():.2f}")
    print(f"    Ratio (anomalia/normal): {ratio:.2f}x")

    # Classificação por threshold (normal < threshold, anomalia >= threshold)
    # Usar média + 2*std como threshold
    threshold = mean_n + 2 * std_n
    tp = (dist_anomalia >= threshold).sum()  # anomalias detectadas
    fp = (dist_normal >= threshold).sum()    # normais falso-alarme
    fn = (dist_anomalia < threshold).sum()   # anomalias perdidas
    tn = (dist_normal < threshold).sum()     # normais corretos

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"    Threshold (mean+2*std): {threshold:.2f}")
    print(f"    Precision: {precision:.1%}  Recall: {recall:.1%}  F1: {f1:.1%}")

    # Distância por classe de anomalia
    dist_por_classe = {}
    for c in sorted(set(labs_anomalia)):
        mask = labs_anomalia == c
        d = dist_anomalia[mask].mean()
        dist_por_classe[c] = d

    return {
        'dist_normal': dist_normal,
        'dist_anomalia': dist_anomalia,
        'labs_anomalia': labs_anomalia,
        'dist_por_classe': dist_por_classe,
        'ratio': ratio,
        'f1': f1,
        'threshold': threshold,
        'prototipo': prototipo,
    }


# ======================================================================
# TESTE 1: MNIST — dígito "1" como normal
# ======================================================================

print("\n" + "="*60)
print("TESTE 1: MNIST — '1' é normal, resto é anomalia")
print("="*60)

res_mnist_1 = teste_anomalia(mnist_train, mnist_test, classe_normal=1,
                              nome_classe="dígito 1", nome_dataset="MNIST")

# ======================================================================
# TESTE 2: MNIST — dígito "0" como normal
# ======================================================================

print("\n" + "="*60)
print("TESTE 2: MNIST — '0' é normal, resto é anomalia")
print("="*60)

res_mnist_0 = teste_anomalia(mnist_train, mnist_test, classe_normal=0,
                              nome_classe="dígito 0", nome_dataset="MNIST")

# ======================================================================
# TESTE 3: Fashion — "Sneaker" (7) como normal
# ======================================================================

print("\n" + "="*60)
print("TESTE 3: Fashion — 'Sneaker' é normal, resto é anomalia")
print("="*60)

fash_nomes = {0:'T-shirt', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat',
              5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

res_fash = teste_anomalia(fash_train, fash_test, classe_normal=7,
                           nome_classe="Sneaker", nome_dataset="Fashion")

# ======================================================================
# TESTE 4: MNIST — todas as classes como normal (uma por vez)
# ======================================================================

print("\n" + "="*60)
print("TESTE 4: MNIST — cada classe como normal")
print("="*60)

ratios_all = []
f1s_all = []
for c in range(10):
    res = teste_anomalia(mnist_train, mnist_test, classe_normal=c,
                          nome_classe=f"dígito {c}", nome_dataset="MNIST",
                          n_train=50, n_test_normal=50, n_test_anomalia=10)
    ratios_all.append(res['ratio'])
    f1s_all.append(res['f1'])

print(f"\n  Média ratio: {np.mean(ratios_all):.2f}x")
print(f"  Média F1:    {np.mean(f1s_all):.1%}")

# ======================================================================
# RESUMO
# ======================================================================

print(f"\n{'='*60}")
print(f"RESUMO — Detecção de Anomalia (Zero Treino)")
print(f"{'='*60}")
print(f"  MNIST '1':      ratio={res_mnist_1['ratio']:.2f}x  F1={res_mnist_1['f1']:.1%}")
print(f"  MNIST '0':      ratio={res_mnist_0['ratio']:.2f}x  F1={res_mnist_0['f1']:.1%}")
print(f"  Fashion Sneaker: ratio={res_fash['ratio']:.2f}x  F1={res_fash['f1']:.1%}")
print(f"  MNIST todas:     ratio={np.mean(ratios_all):.2f}x  F1={np.mean(f1s_all):.1%}")
print(f"{'='*60}")

# ======================================================================
# Visualização
# ======================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Auditoria 26 — Detecção de Anomalia (Zero Treino)', fontsize=13, fontweight='bold')

# Plot 1: Histograma MNIST "1"
ax = axes[0][0]
ax.hist(res_mnist_1['dist_normal'], bins=30, alpha=0.6, label='Normal (1)', color='green')
ax.hist(res_mnist_1['dist_anomalia'], bins=30, alpha=0.6, label='Anomalia', color='red')
ax.axvline(res_mnist_1['threshold'], color='black', linestyle='--', label=f'Threshold={res_mnist_1["threshold"]:.1f}')
ax.set_title(f'MNIST: "1" como normal (ratio={res_mnist_1["ratio"]:.2f}x)')
ax.legend(fontsize=8)
ax.set_xlabel('Distância ao protótipo')

# Plot 2: Histograma MNIST "0"
ax = axes[0][1]
ax.hist(res_mnist_0['dist_normal'], bins=30, alpha=0.6, label='Normal (0)', color='green')
ax.hist(res_mnist_0['dist_anomalia'], bins=30, alpha=0.6, label='Anomalia', color='red')
ax.axvline(res_mnist_0['threshold'], color='black', linestyle='--', label=f'Threshold={res_mnist_0["threshold"]:.1f}')
ax.set_title(f'MNIST: "0" como normal (ratio={res_mnist_0["ratio"]:.2f}x)')
ax.legend(fontsize=8)
ax.set_xlabel('Distância ao protótipo')

# Plot 3: Histograma Fashion "Sneaker"
ax = axes[1][0]
ax.hist(res_fash['dist_normal'], bins=30, alpha=0.6, label='Normal (Sneaker)', color='green')
ax.hist(res_fash['dist_anomalia'], bins=30, alpha=0.6, label='Anomalia', color='red')
ax.axvline(res_fash['threshold'], color='black', linestyle='--', label=f'Threshold={res_fash["threshold"]:.1f}')
ax.set_title(f'Fashion: "Sneaker" como normal (ratio={res_fash["ratio"]:.2f}x)')
ax.legend(fontsize=8)
ax.set_xlabel('Distância ao protótipo')

# Plot 4: Distância por classe (MNIST "1" como normal)
ax = axes[1][1]
classes = sorted(res_mnist_1['dist_por_classe'].keys())
dists = [res_mnist_1['dist_por_classe'][c] for c in classes]
colors = ['green' if c == 1 else 'red' for c in classes]
bars = ax.bar([str(c) for c in classes], dists, color=colors, alpha=0.7)
ax.axhline(res_mnist_1['dist_normal'].mean(), color='green', linestyle='--', label=f'Normal "1" média')
ax.set_title('Distância por dígito (normal="1")')
ax.set_xlabel('Dígito')
ax.set_ylabel('Distância ao protótipo "1"')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('viz_audit_26_anomalia.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_26_anomalia.png")
print("Pronto.")
