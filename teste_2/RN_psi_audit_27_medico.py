"""
Auditoria 27: Imagens Médicas

Testa a ResNet-Psi em raio-X de tórax (pneumonia vs normal).
Dataset: PneumoniaMNIST do MedMNIST (28x28, grayscale, 2 classes).

Mesmo tamanho do MNIST, entrada direta sem adaptação.
Se funcionar, prova que o mecanismo generaliza pra domínio médico.
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
# Baixar MedMNIST
# ======================================================================

try:
    import medmnist
    from medmnist import PneumoniaMNIST, DermaMNIST, BreastMNIST
except ImportError:
    print("Instalando medmnist...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'medmnist', '-q'])
    import medmnist
    from medmnist import PneumoniaMNIST, DermaMNIST, BreastMNIST

print(f"MedMNIST versão: {medmnist.__version__}")

# ======================================================================
# TESTE 1: PneumoniaMNIST (raio-X tórax: normal vs pneumonia)
# ======================================================================

print("\n" + "="*60)
print("TESTE 1: PneumoniaMNIST (raio-X tórax)")
print("  Normal (0) vs Pneumonia (1)")
print("="*60)

train_ds = PneumoniaMNIST(split='train', download=True, size=28)
test_ds  = PneumoniaMNIST(split='test',  download=True, size=28)

def preparar_medmnist(ds, max_per_class=None):
    """Converte dataset MedMNIST pra tensores."""
    imgs_raw = ds.imgs        # (N, 28, 28) uint8
    labs_raw = ds.labels.flatten()  # (N,)

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

    imgs = torch.tensor(imgs_raw, dtype=torch.float32, device=DEVICE) / 255.0
    labs = np.array(labs_raw, dtype=int)

    print(f"  Amostras: {len(imgs)}")
    for c in sorted(set(labs)):
        print(f"    Classe {c}: {(labs == c).sum()}")

    return imgs, labs

print("\nTreino:")
train_imgs_p, train_labs_p = preparar_medmnist(train_ds, max_per_class=500)
print("Teste:")
test_imgs_p, test_labs_p = preparar_medmnist(test_ds)

rn_pneumonia = ResNetPsi(n_classes=2)
rn_pneumonia.fit(train_imgs_p, train_labs_p, bs=32)
acc_pneumonia = rn_pneumonia.score(test_imgs_p, test_labs_p, bs=32)
print(f"\n  PneumoniaMNIST: {acc_pneumonia:.1f}%  (chance=50%)")

# ======================================================================
# TESTE 2: DermaMNIST (lesões de pele, 7 classes)
# ======================================================================

print("\n" + "="*60)
print("TESTE 2: DermaMNIST (lesões de pele)")
print("  7 classes de lesões dermatológicas")
print("="*60)

train_ds_d = DermaMNIST(split='train', download=True, size=28)
test_ds_d  = DermaMNIST(split='test',  download=True, size=28)

print("\nTreino:")
train_imgs_d, train_labs_d = preparar_medmnist(train_ds_d, max_per_class=200)
print("Teste:")
test_imgs_d, test_labs_d = preparar_medmnist(test_ds_d)

# DermaMNIST é RGB (3 canais) — converter pra grayscale
if len(train_imgs_d.shape) == 4 and train_imgs_d.shape[-1] == 3:
    # (N, 28, 28, 3) → (N, 28, 28)
    train_imgs_d = train_imgs_d.mean(dim=-1)
    test_imgs_d  = test_imgs_d.mean(dim=-1)
elif len(train_imgs_d.shape) == 4 and train_imgs_d.shape[1] == 3:
    # (N, 3, 28, 28) → (N, 28, 28)
    train_imgs_d = train_imgs_d.mean(dim=1)
    test_imgs_d  = test_imgs_d.mean(dim=1)

rn_derma = ResNetPsi(n_classes=7)
rn_derma.fit(train_imgs_d, train_labs_d, bs=32)
acc_derma = rn_derma.score(test_imgs_d, test_labs_d, bs=32)
print(f"\n  DermaMNIST: {acc_derma:.1f}%  (chance=14.3%)")

# ======================================================================
# TESTE 3: BreastMNIST (ultrassom mama: normal/benigno/maligno)
# ======================================================================

print("\n" + "="*60)
print("TESTE 3: BreastMNIST (ultrassom de mama)")
print("  Normal (0) vs Benigno (1) vs Maligno (2)")
print("="*60)

train_ds_b = BreastMNIST(split='train', download=True, size=28)
test_ds_b  = BreastMNIST(split='test',  download=True, size=28)

print("\nTreino:")
train_imgs_b, train_labs_b = preparar_medmnist(train_ds_b)
print("Teste:")
test_imgs_b, test_labs_b = preparar_medmnist(test_ds_b)

# BreastMNIST pode ser RGB
if len(train_imgs_b.shape) == 4 and train_imgs_b.shape[-1] == 3:
    train_imgs_b = train_imgs_b.mean(dim=-1)
    test_imgs_b  = test_imgs_b.mean(dim=-1)
elif len(train_imgs_b.shape) == 4 and train_imgs_b.shape[1] == 3:
    train_imgs_b = train_imgs_b.mean(dim=1)
    test_imgs_b  = test_imgs_b.mean(dim=1)

n_classes_b = len(set(train_labs_b))
rn_breast = ResNetPsi(n_classes=n_classes_b)
rn_breast.fit(train_imgs_b, train_labs_b, bs=32)
acc_breast = rn_breast.score(test_imgs_b, test_labs_b, bs=32)
chance_b = 100.0 / n_classes_b
print(f"\n  BreastMNIST: {acc_breast:.1f}%  (chance={chance_b:.1f}%)")

# ======================================================================
# RESUMO
# ======================================================================

print(f"\n{'='*60}")
print(f"RESUMO — Imagens Médicas (Zero Treino)")
print(f"{'='*60}")
print(f"  PneumoniaMNIST (2 cls): {acc_pneumonia:.1f}%  (chance=50.0%)")
print(f"  DermaMNIST     (7 cls): {acc_derma:.1f}%  (chance=14.3%)")
print(f"  BreastMNIST    ({n_classes_b} cls): {acc_breast:.1f}%  (chance={chance_b:.1f}%)")
print(f"{'='*60}")

# ======================================================================
# Visualização: exemplos + crystal maps
# ======================================================================

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
fig.suptitle(f'Auditoria 27 — Imagens Médicas (Zero Treino)\n'
             f'Pneumonia={acc_pneumonia:.1f}% | Derma={acc_derma:.1f}% | Breast={acc_breast:.1f}%',
             fontsize=12, fontweight='bold')

# Linha 1: PneumoniaMNIST (3 exemplos + 3 crystal maps)
cmaps_p = rn_pneumonia.extract(test_imgs_p[:3], bs=3)
nomes_p = {0: 'Normal', 1: 'Pneumo'}
for i in range(3):
    axes[0][i].imshow(test_imgs_p[i].cpu().numpy(), cmap='gray')
    axes[0][i].set_title(f'Pneumo: {nomes_p.get(test_labs_p[i], "?")}', fontsize=9)
    axes[0][i].axis('off')

    cm = cmaps_p[i].cpu().numpy()
    cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
    axes[0][i+3].imshow(cm, cmap='hot')
    axes[0][i+3].set_title(f'Crystal map', fontsize=9)
    axes[0][i+3].axis('off')

# Linha 2: DermaMNIST
cmaps_d = rn_derma.extract(test_imgs_d[:3], bs=3)
for i in range(3):
    axes[1][i].imshow(test_imgs_d[i].cpu().numpy(), cmap='gray')
    axes[1][i].set_title(f'Derma cls={test_labs_d[i]}', fontsize=9)
    axes[1][i].axis('off')

    cm = cmaps_d[i].cpu().numpy()
    cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
    axes[1][i+3].imshow(cm, cmap='hot')
    axes[1][i+3].set_title(f'Crystal map', fontsize=9)
    axes[1][i+3].axis('off')

# Linha 3: BreastMNIST
cmaps_b = rn_breast.extract(test_imgs_b[:3], bs=3)
nomes_b = {0: 'Normal', 1: 'Benigno', 2: 'Maligno'}
for i in range(3):
    axes[2][i].imshow(test_imgs_b[i].cpu().numpy(), cmap='gray')
    axes[2][i].set_title(f'Breast: {nomes_b.get(test_labs_b[i], "?")}', fontsize=9)
    axes[2][i].axis('off')

    cm = cmaps_b[i].cpu().numpy()
    cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
    axes[2][i+3].imshow(cm, cmap='hot')
    axes[2][i+3].set_title(f'Crystal map', fontsize=9)
    axes[2][i+3].axis('off')

plt.tight_layout()
plt.savefig('viz_audit_27_medico.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_27_medico.png")
print("Pronto.")
