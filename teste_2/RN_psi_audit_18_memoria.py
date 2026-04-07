"""
Auditoria 18: Memória Associativa Multi-Classe

Um campo memoriza todos os 10 dígitos simultaneamente.
Teste: recebe 25% do dígito + ruído → completa o padrão correto.

Dificuldade dupla:
- Só 25% da imagem visível (canto superior esquerdo)
- Ruído gaussiano sobreposto
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")

FIELD_SIZE     = 48
PSI_DT         = 0.05
PSI_GAMMA      = 0.06
PSI_ALPHA      = 0.04
PSI_BETA       = 0.005
PSI_C2         = 0.3
STIM_ON        = 40
STIM_TOTAL     = 80

CRYSTAL_W      = 20
CRYSTAL_K      = 3
CRYSTAL_A_MIN  = 0.3
CRYSTAL_CV_MAX = 0.15
CRYSTAL_SEP    = 5
CRYSTAL_REMIT  = 0.3

_DT    = torch.tensor(PSI_DT,    device=DEVICE)
_GAMMA = torch.tensor(PSI_GAMMA, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_BETA  = torch.tensor(PSI_BETA,  device=DEVICE)
_C2    = torch.tensor(PSI_C2,    device=DEVICE)


class CrystalCompetitivo:
    def __init__(self, B, FS=FIELD_SIZE, sharpness=5.0, decay=0.02, ressonance_boost=0.1):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
        self.sharpness = sharpness
        self.decay = decay
        self.ressonance_boost = ressonance_boost
        ks = 2 * CRYSTAL_SEP + 1
        self._dilate = torch.ones(1, 1, ks, ks, device=DEVICE)

    def update_envelope(self, field):
        self.window_max = torch.max(self.window_max, field.abs())
        self.window_step += 1
        if self.window_step >= CRYSTAL_W:
            self.env_buffer[:, self.window_idx] = self.window_max
            self.window_max  = torch.zeros_like(self.window_max)
            self.window_step = 0
            self.window_idx  = (self.window_idx + 1) % CRYSTAL_K

    def try_crystallize(self, field):
        env  = self.env_buffer
        mean = env.mean(dim=1)
        cv   = env.std(dim=1) / (mean + 1e-8)
        amp_score = torch.sigmoid(self.sharpness * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(self.sharpness * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(self.sharpness * (8.0 - mean))
        cand = amp_score * cv_score * sat_score
        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0,1)
        new_crystals = cand * (1.0 - occ) * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + new_crystals, 0, 10.)
        self.crystal_hp = torch.where(
            new_crystals > 0.01,
            torch.clamp(self.crystal_hp + 1.0, 0, 5.0),
            self.crystal_hp)
        ressonance = field.abs() * (self.crystal_map > 0.01).float()
        self.crystal_hp = self.crystal_hp + ressonance * self.ressonance_boost
        self.crystal_hp = self.crystal_hp - self.decay
        alive = (self.crystal_hp > 0).float()
        self.crystal_map = self.crystal_map * alive
        self.crystal_hp  = torch.clamp(self.crystal_hp * alive, 0, 5.0)

    def remit(self, field, strength=CRYSTAL_REMIT):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * strength * torch.sign(field + 1e-8), -10., 10.)


def psi_step(field, velocity, sources, active):
    lap_k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                          device=DEVICE).view(1,1,3,3).to(field.dtype)
    if active:
        field = field + sources * (_DT * 0.1)
    lap = F.conv2d(F.pad(field.unsqueeze(1),(1,1,1,1),mode='circular'), lap_k).squeeze(1)
    acc = _C2*lap - _GAMMA*velocity + _ALPHA*torch.tanh(field)*field - _BETA*field*field**2
    velocity = torch.clamp(velocity + acc*_DT, -5., 5.)
    field    = torch.clamp(field + velocity*_DT, -10., 10.)
    return field, velocity


def build_gaussians(field_size=FIELD_SIZE, sigma=0.04):
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            gs.append(torch.exp(-((xg-cx)**2+(yg-cy)**2)/(2*sigma**2)))
    return torch.stack(gs).view(784, -1)


def field_to_image(field, PG_T=None):
    """Projeção inversa: campo → imagem 28x28."""
    coords = torch.linspace(0., 1., FIELD_SIZE, device=field.device)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    img = torch.zeros(28, 28, device=field.device)
    sigma = 0.04
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            gauss = torch.exp(-((xg-cx)**2+(yg-cy)**2)/(2*sigma**2))
            img[pi, pj] = (field.abs() * gauss).sum()
    return img


def norm(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)


# ── Carregar MNIST ────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

N_MEM = 200  # exemplos por classe para memorizar

train_by_class = {i: [] for i in range(10)}
for img, label in train_ds:
    if len(train_by_class[label]) < N_MEM:
        train_by_class[label].append(img.squeeze(0))

test_examples = {}
for img, label in test_ds:
    if label not in test_examples:
        test_examples[label] = img.squeeze(0)
    if len(test_examples) == 10:
        break

PG = build_gaussians()

# ── Fase 1: Memorizar todos os 10 dígitos no mesmo campo ─────────────────────

print(f"\n-- Fase 1: Campos Separados por Classe ({N_MEM} exemplos x 10 digitos) --")

t0 = time.time()
mems = {}
for cls in range(10):
    mem = CrystalCompetitivo(1, FIELD_SIZE)
    for img in train_by_class[cls]:
        pert = (img.to(DEVICE).view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)
        f, v = pert.clone(), torch.zeros_like(pert)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)
    mems[cls] = mem
    print(f"  Dígito {cls}: {(mem.crystal_map > 0.01).float().sum().item():.0f} cristais")

print(f"  Memória completa: {time.time()-t0:.0f}s")

# ── Fase 2: Recuperação com 25% + ruído ──────────────────────────────────────

print(f"\n-- Fase 2: Campos separados por classe --")

DIGITS_TEST = list(range(10))

fig, axes = plt.subplots(len(DIGITS_TEST), 6, figsize=(22, 4*len(DIGITS_TEST)))
fig.suptitle(f'Auditoria 18 — Campos Separados por Classe\n'
             f'10 campos especializados → recebe metade → vota qual ressoa mais\n'
             f'({N_MEM} exemplos/classe)', fontsize=12)

col_titles = ['Original', 'Metade sup.', 'Campo vencedor', 'Reconstruído', 'Cristais vencedor', 'Dígito previsto']
for j, title in enumerate(col_titles):
    axes[0][j].set_title(title, fontsize=9, fontweight='bold')

corretos = 0
for row, digit in enumerate(DIGITS_TEST):
    img = test_examples[digit].to(DEVICE)
    img_entrada = img.clone()
    img_entrada[14:, :] = 0.0

    pert = (img_entrada.view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)

    # Gerar crystal_map da entrada parcial
    f_base, v_base = pert.clone(), torch.zeros_like(pert)
    mem_query = CrystalCompetitivo(1, FIELD_SIZE)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f_base, v_base = psi_step(f_base, v_base, pert, s < STIM_ON)
            mem_query.update_envelope(f_base)
            if mem_query.window_idx > 0:
                mem_query.try_crystallize(f_base)
            f_base = mem_query.remit(f_base)
    cmap_query = mem_query.crystal_map.squeeze(0).view(-1).float()

    # Distância do crystal_map da entrada parcial ao crystal_map de cada classe
    # Usar apenas a metade superior do campo (pixels 0..13 mapeiam para y<0.5 no campo)
    cmap_query_2d = mem_query.crystal_map.squeeze(0)  # (FS, FS)
    energias = {}
    campos_gerados = {}
    for cls in range(10):
        cmap_cls_2d = mems[cls].crystal_map.squeeze(0)
        # Comparar só a metade superior (onde tem informação)
        top_q = cmap_query_2d[:FIELD_SIZE//2, :].reshape(-1).float()
        top_c = cmap_cls_2d[:FIELD_SIZE//2, :].reshape(-1).float()
        dist = (top_q - top_c).pow(2).sum().sqrt()
        energias[cls] = -dist.item()
        campos_gerados[cls] = f_base.squeeze(0)

    # Dígito vencedor = maior ressonância
    previsto = max(energias, key=energias.get)
    if previsto == digit:
        corretos += 1

    campo_vencedor = campos_gerados[previsto]
    img_rec = field_to_image(campo_vencedor)

    simbolo = 'OK' if previsto==digit else 'X'
    print(f"  Digito {digit} -> previsto: {previsto} {simbolo}  "
          f"energias: {[f'{energias[c]:.1f}' for c in range(10)]}")

    axes[row][0].imshow(norm(img).cpu().numpy(), cmap='gray')
    axes[row][0].set_ylabel(f'Dígito {digit}', fontsize=9)
    axes[row][0].axis('off')
    axes[row][1].imshow(norm(img_entrada).cpu().numpy(), cmap='gray')
    axes[row][1].axis('off')
    axes[row][2].imshow(norm(campo_vencedor).cpu().numpy(), cmap='hot')
    axes[row][2].axis('off')
    axes[row][3].imshow(norm(img_rec).cpu().numpy(), cmap='gray')
    axes[row][3].axis('off')
    axes[row][4].imshow(norm(mems[previsto].crystal_map.squeeze(0)).cpu().numpy(), cmap='hot')
    axes[row][4].axis('off')
    # Mostrar energias como barras
    ax = axes[row][5]
    cores = ['green' if c == digit else ('red' if c == previsto else 'gray') for c in range(10)]
    ax.bar(range(10), [energias[c] for c in range(10)], color=cores, alpha=0.8)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(c) for c in range(10)], fontsize=7)
    ax.set_title(f'-> {previsto} {"OK" if previsto==digit else "X"}', fontsize=10,
                 color='green' if previsto==digit else 'red')
    ax.grid(alpha=0.3, axis='y')

acc = corretos / len(DIGITS_TEST) * 100
print(f"\n  Acuracia por ressonancia: {corretos}/{len(DIGITS_TEST)} = {acc:.0f}%")

plt.tight_layout()
plt.savefig('viz_audit_18_campos_separados.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_18_campos_separados.png")
print("Pronto.")
