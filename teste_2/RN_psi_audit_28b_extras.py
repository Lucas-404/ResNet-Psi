"""
Auditoria 28b: Datasets Extras

1. KMNIST — baixar via URL alternativa (huggingface/zenodo)
2. USPS — dígitos manuscritos 16x16 (redimensionar pra 28x28)
3. SVHN — números de casa (Street View House Numbers), grayscale

Mais pontos no mapa de domínio.
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

resultados = {}

def testar(nome, train_imgs, train_labs, test_imgs, test_labs, n_classes):
    print(f"\n{'='*60}")
    print(f"{nome} ({n_classes} classes)")
    print(f"{'='*60}")
    print(f"  Treino: {len(train_imgs)}  Teste: {len(test_imgs)}")

    chance = 100.0 / n_classes
    rn = ResNetPsi(n_classes=n_classes)
    rn.fit(train_imgs, train_labs, bs=32)
    acc = rn.score(test_imgs, test_labs, bs=64)
    ratio = acc / chance
    print(f"\n  {nome}: {acc:.1f}%  (chance={chance:.1f}%, ratio={ratio:.1f}x)")
    resultados[nome] = {'acc': acc, 'chance': chance, 'n_classes': n_classes, 'ratio': ratio}
    return rn

# ======================================================================
# 1. KMNIST — tentar baixar
# ======================================================================

print("\nTentando KMNIST...")
try:
    # Timeout mais longo
    import socket
    socket.setdefaulttimeout(30)

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.KMNIST('./data', train=True, download=True, transform=tf)
    test_ds  = datasets.KMNIST('./data', train=False, download=True, transform=tf)

    # Pegar 50 por classe
    por_classe = {c: [] for c in range(10)}
    for i in range(len(train_ds)):
        img, lab = train_ds[i]
        if len(por_classe[lab]) < 50:
            por_classe[lab].append(img.squeeze(0))
        if all(len(v) >= 50 for v in por_classe.values()):
            break

    train_imgs = torch.stack([img for c in range(10) for img in por_classe[c]]).to(DEVICE)
    train_labs = np.array([c for c in range(10) for _ in por_classe[c]])

    test_imgs = torch.stack([test_ds[i][0].squeeze(0) for i in range(1000)]).to(DEVICE)
    test_labs = np.array([test_ds[i][1] for i in range(1000)])

    testar("KMNIST", train_imgs, train_labs, test_imgs, test_labs, 10)
except Exception as e:
    print(f"  KMNIST falhou: {e}")
    print("  Pulando...")
    resultados["KMNIST"] = None

# ======================================================================
# 2. USPS — dígitos 16x16 redimensionados pra 28x28
# ======================================================================

print("\nTentando USPS...")
try:
    tf_usps = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    train_ds = datasets.USPS('./data', train=True, download=True, transform=tf_usps)
    test_ds  = datasets.USPS('./data', train=False, download=True, transform=tf_usps)

    por_classe = {c: [] for c in range(10)}
    for i in range(len(train_ds)):
        img, lab = train_ds[i]
        if len(por_classe[lab]) < 50:
            por_classe[lab].append(img.squeeze(0))
        if all(len(v) >= 50 for v in por_classe.values()):
            break

    train_imgs = torch.stack([img for c in range(10) for img in por_classe[c]]).to(DEVICE)
    train_labs = np.array([c for c in range(10) for _ in por_classe[c]])

    n_test = min(1000, len(test_ds))
    test_imgs = torch.stack([test_ds[i][0].squeeze(0) for i in range(n_test)]).to(DEVICE)
    test_labs = np.array([test_ds[i][1] for i in range(n_test)])

    testar("USPS", train_imgs, train_labs, test_imgs, test_labs, 10)
except Exception as e:
    print(f"  USPS falhou: {e}")
    resultados["USPS"] = None

# ======================================================================
# 3. SVHN — números de casa (grayscale)
# ======================================================================

print("\nTentando SVHN...")
try:
    tf_svhn = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    train_ds = datasets.SVHN('./data', split='train', download=True, transform=tf_svhn)
    test_ds  = datasets.SVHN('./data', split='test', download=True, transform=tf_svhn)

    por_classe = {c: [] for c in range(10)}
    for i in range(len(train_ds)):
        img, lab = train_ds[i]
        if len(por_classe[lab]) < 50:
            por_classe[lab].append(img.squeeze(0))
        if all(len(v) >= 50 for v in por_classe.values()):
            break

    train_imgs = torch.stack([img for c in range(10) for img in por_classe[c]]).to(DEVICE)
    train_labs = np.array([c for c in range(10) for _ in por_classe[c]])

    n_test = min(1000, len(test_ds))
    test_imgs = torch.stack([test_ds[i][0].squeeze(0) for i in range(n_test)]).to(DEVICE)
    test_labs = np.array([test_ds[i][1] for i in range(n_test)])

    testar("SVHN", train_imgs, train_labs, test_imgs, test_labs, 10)
except Exception as e:
    print(f"  SVHN falhou: {e}")
    resultados["SVHN"] = None

# ======================================================================
# 4. MedMNIST extras — se disponível
# ======================================================================

print("\nTentando MedMNIST extras...")
try:
    from medmnist import OrganMNIST_Axial, BloodMNIST

    # BloodMNIST — células sanguíneas (8 classes)
    train_ds = BloodMNIST(split='train', download=True, size=28)
    test_ds  = BloodMNIST(split='test', download=True, size=28)

    imgs_raw = train_ds.imgs
    labs_raw = train_ds.labels.flatten()
    n_classes = len(set(labs_raw))

    por_classe = {c: [] for c in range(n_classes)}
    for i, lab in enumerate(labs_raw):
        lab = int(lab)
        if len(por_classe[lab]) < 50:
            por_classe[lab].append(imgs_raw[i])
        if all(len(v) >= 50 for v in por_classe.values()):
            break

    train_np = np.stack([img for c in range(n_classes) for img in por_classe[c]])
    train_labs = np.array([c for c in range(n_classes) for _ in por_classe[c]])

    # Grayscale se RGB
    if len(train_np.shape) == 4:
        train_np = train_np.mean(axis=-1)

    train_imgs = torch.tensor(train_np, dtype=torch.float32, device=DEVICE) / 255.0

    test_np = test_ds.imgs
    test_labs_raw = test_ds.labels.flatten()
    if len(test_np.shape) == 4:
        test_np = test_np.mean(axis=-1)

    n_test = min(1000, len(test_np))
    test_imgs = torch.tensor(test_np[:n_test], dtype=torch.float32, device=DEVICE) / 255.0
    test_labs = np.array(test_labs_raw[:n_test], dtype=int)

    testar("BloodMNIST", train_imgs, train_labs, test_imgs, test_labs, n_classes)
except Exception as e:
    print(f"  BloodMNIST falhou: {e}")
    resultados["BloodMNIST"] = None

# ======================================================================
# RESUMO
# ======================================================================

print(f"\n{'='*60}")
print(f"RESUMO — Datasets Extras (Zero Treino, 50/classe)")
print(f"{'='*60}")
print(f"  {'Dataset':<20s}  {'Classes':>7s}  {'Acc':>7s}  {'Chance':>7s}  {'Ratio':>6s}")
print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")

for nome in ['KMNIST', 'USPS', 'SVHN', 'BloodMNIST']:
    r = resultados.get(nome)
    if r is None:
        print(f"  {nome:<20s}  {'FALHOU':>7s}")
    else:
        print(f"  {nome:<20s}  {r['n_classes']:>7d}  {r['acc']:>6.1f}%  {r['chance']:>6.1f}%  {r['ratio']:>5.1f}x")

print(f"{'='*60}")

# ======================================================================
# Gráfico
# ======================================================================

nomes_ok = [n for n in ['KMNIST', 'USPS', 'SVHN', 'BloodMNIST'] if resultados.get(n) is not None]

if nomes_ok:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Auditoria 28b — Datasets Extras (Zero Treino)', fontsize=12, fontweight='bold')

    accs = [resultados[n]['acc'] for n in nomes_ok]
    chances = [resultados[n]['chance'] for n in nomes_ok]
    x = range(len(nomes_ok))
    ax.bar(x, accs, 0.4, label='ResNet-Ψ', color='steelblue')
    ax.bar([i+0.4 for i in x], chances, 0.4, label='Chance', color='lightgray')
    ax.set_xticks([i+0.2 for i in x])
    ax.set_xticklabels(nomes_ok, fontsize=10)
    ax.set_ylabel('Acurácia (%)')
    ax.legend()

    for i, n in enumerate(nomes_ok):
        r = resultados[n]
        ax.text(i, r['acc'] + 1, f"{r['ratio']:.1f}x", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('viz_audit_28b_extras.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n-> viz_audit_28b_extras.png")

print("Pronto.")
