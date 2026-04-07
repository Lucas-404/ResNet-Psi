"""
Auditoria 28c: BloodMNIST + OrganMNIST

Datasets MedMNIST com estrutura geométrica:
1. BloodMNIST — células sanguíneas (8 classes)
2. OrganAMNIST — órgãos abdominais axial CT (11 classes)
3. PathMNIST — histopatologia colorectal (9 classes)
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

import medmnist

resultados = {}

def preparar_medmnist(ds, max_per_class=50):
    imgs_raw = ds.imgs
    labs_raw = ds.labels.flatten()
    n_classes = len(set(labs_raw))

    if max_per_class is not None:
        indices = []
        contagem = {}
        for i, lab in enumerate(labs_raw):
            lab = int(lab)
            contagem.setdefault(lab, 0)
            if contagem[lab] < max_per_class:
                indices.append(i)
                contagem[lab] += 1
        imgs_raw = imgs_raw[indices]
        labs_raw = labs_raw[indices]

    # Grayscale se RGB
    if len(imgs_raw.shape) == 4:
        imgs_raw = imgs_raw.mean(axis=-1)

    imgs = torch.tensor(imgs_raw, dtype=torch.float32, device=DEVICE) / 255.0
    labs = np.array(labs_raw, dtype=int)
    return imgs, labs, n_classes


def testar_medmnist(nome, classe_ds):
    print(f"\n{'='*60}")
    print(f"{nome}")
    print(f"{'='*60}")

    try:
        train_ds = classe_ds(split='train', download=True, size=28)
        test_ds  = classe_ds(split='test',  download=True, size=28)

        train_imgs, train_labs, n_classes = preparar_medmnist(train_ds, max_per_class=50)
        test_imgs, test_labs, _ = preparar_medmnist(test_ds, max_per_class=None)

        # Limitar teste
        if len(test_imgs) > 1000:
            test_imgs = test_imgs[:1000]
            test_labs = test_labs[:1000]

        print(f"  Treino: {len(train_imgs)} ({n_classes} classes)")
        print(f"  Teste:  {len(test_imgs)}")

        chance = 100.0 / n_classes
        rn = ResNetPsi(n_classes=n_classes)
        rn.fit(train_imgs, train_labs, bs=32)
        acc = rn.score(test_imgs, test_labs, bs=64)
        ratio = acc / chance

        print(f"\n  {nome}: {acc:.1f}%  (chance={chance:.1f}%, ratio={ratio:.1f}x)")
        resultados[nome] = {'acc': acc, 'chance': chance, 'n_classes': n_classes, 'ratio': ratio}
    except Exception as e:
        print(f"  Falhou: {e}")
        resultados[nome] = None


# ======================================================================
# Rodar
# ======================================================================

testar_medmnist("BloodMNIST", medmnist.BloodMNIST)
testar_medmnist("OrganAMNIST", medmnist.OrganAMNIST)
testar_medmnist("PathMNIST", medmnist.PathMNIST)
testar_medmnist("OCTMNIST", medmnist.OCTMNIST)
testar_medmnist("RetinaMNIST", medmnist.RetinaMNIST)

# ======================================================================
# RESUMO
# ======================================================================

print(f"\n{'='*60}")
print(f"RESUMO — MedMNIST Extras (Zero Treino, 50/classe)")
print(f"{'='*60}")
print(f"  {'Dataset':<20s}  {'Classes':>7s}  {'Acc':>7s}  {'Chance':>7s}  {'Ratio':>6s}")
print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")

for nome in ['BloodMNIST', 'OrganAMNIST', 'PathMNIST', 'OCTMNIST', 'RetinaMNIST']:
    r = resultados.get(nome)
    if r is None:
        print(f"  {nome:<20s}  {'FALHOU':>7s}")
    else:
        print(f"  {nome:<20s}  {r['n_classes']:>7d}  {r['acc']:>6.1f}%  {r['chance']:>6.1f}%  {r['ratio']:>5.1f}x")

print(f"{'='*60}")
print("Pronto.")
