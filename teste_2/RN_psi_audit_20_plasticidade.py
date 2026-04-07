"""
Auditoria 20: Plasticidade do Campo

Pergunta: o campo consegue representar padroes novos que nunca viu?

Pipeline:
1. Soma "1" + "7" numa unica imagem (padrao nunca visto no treino)
2. Roda o campo normalmente
3. Observa o crystal_map resultante — ele representa "17" como forma propria?
4. Compara visualmente com crystal_maps de "1" puro e "7" puro

Se o campo cristaliza "17" como padrao distinto (nao apenas "1" ou "7"),
isso e plasticidade: representacao espontanea de novos padroes via fisica pura.
"""

import torch
import torch.nn.functional as F
import numpy as np
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
CRYSTAL_REMIT  = 0.05

_DT    = torch.tensor(PSI_DT,    device=DEVICE)
_GAMMA = torch.tensor(PSI_GAMMA, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_BETA  = torch.tensor(PSI_BETA,  device=DEVICE)
_C2    = torch.tensor(PSI_C2,    device=DEVICE)


class CrystalCompetitivo:
    def __init__(self, B, FS=FIELD_SIZE):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
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
        amp_score = torch.sigmoid(5.0 * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(5.0 * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(5.0 * (8.0 - mean))
        cand = amp_score * cv_score * sat_score
        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0, 1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0, 1)
        new_crystals = cand * (1.0 - occ) * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + new_crystals, 0, 10.)
        self.crystal_hp = torch.where(
            new_crystals > 0.01,
            torch.clamp(self.crystal_hp + 1.0, 0, 5.0),
            self.crystal_hp)
        ressonance = field.abs() * (self.crystal_map > 0.01).float()
        self.crystal_hp = self.crystal_hp + ressonance * 0.1
        self.crystal_hp = self.crystal_hp - 0.02
        alive = (self.crystal_hp > 0).float()
        self.crystal_map = self.crystal_map * alive
        self.crystal_hp  = torch.clamp(self.crystal_hp * alive, 0, 5.0)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


def psi_step(field, velocity, sources, active):
    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                          device=DEVICE).view(1, 1, 3, 3).to(field.dtype)
    if active:
        field = field + sources * (_DT * 0.1)
    lap = F.conv2d(F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular'), lap_k).squeeze(1)
    acc = _C2 * lap - _GAMMA * velocity + _ALPHA * torch.tanh(field) * field - _BETA * field * field**2
    velocity = torch.clamp(velocity + acc * _DT, -5., 5.)
    field    = torch.clamp(field + velocity * _DT, -10., 10.)
    return field, velocity


def build_gaussians(field_size=FIELD_SIZE, sigma=0.04):
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            gs.append(torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2)))
    return torch.stack(gs).view(784, -1)


def run_field(img):
    """Roda campo para uma imagem (28,28), retorna (crystal_map, campo_final, campo_sem_cristais)."""
    pert = (img.view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)

    # Com cristais
    f, v = pert.clone(), torch.zeros_like(pert)
    mem = CrystalCompetitivo(1)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f, v = psi_step(f, v, pert, s < STIM_ON)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)

    # Sem cristais — campo bruto
    f2, v2 = pert.clone(), torch.zeros_like(pert)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f2, v2 = psi_step(f2, v2, pert, s < STIM_ON)

    return mem.crystal_map.squeeze(0), f.squeeze(0), f2.squeeze(0)


def field_to_image(field, field_size=FIELD_SIZE, sigma=0.04):
    """Converte campo (FS,FS) para imagem 28x28 via broadcasting."""
    coords = torch.linspace(0., 1., field_size, device=field.device)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    px = torch.linspace(0.1, 0.9, 28, device=field.device)
    py = torch.linspace(0.1, 0.9, 28, device=field.device)
    cx, cy = torch.meshgrid(px, py, indexing='ij')
    dx = xg.unsqueeze(0).unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1)
    dy = yg.unsqueeze(0).unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1)
    gauss = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return (field.abs().unsqueeze(0).unsqueeze(0) * gauss).sum(dim=(-2, -1))


def norm(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)


# -- Carregar MNIST -----------------------------------------------------------

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

PG = build_gaussians()

# Separar por classe
by_class = {i: [] for i in range(10)}
for img, label in test_ds:
    by_class[label].append(img.squeeze(0))

# -- Pares para testar plasticidade ------------------------------------------
# Pares geometricamente parecidos
PARES = [(1, 7), (3, 8), (4, 9), (0, 8)]
N_EXEMPLOS = 5  # exemplos por par

print(f"\nRodando campo para {len(PARES)} pares x {N_EXEMPLOS} exemplos...")

fig, axes = plt.subplots(len(PARES) * N_EXEMPLOS, 8,
                          figsize=(24, 4 * len(PARES) * N_EXEMPLOS))
fig.suptitle('Auditoria 20 — Plasticidade: com cristais vs sem cristais',
             fontsize=13, fontweight='bold')

col_titles = ['Digito A', 'Digito B', 'A+B entrada',
              'Bruto A', 'Bruto B', 'Bruto A+B',
              'Crystal A+B', 'Diferenca\n(crystal - bruto)']
for j, t in enumerate(col_titles):
    axes[0][j].set_title(t, fontsize=9, fontweight='bold')

row = 0
for cls_a, cls_b in PARES:
    for i in range(N_EXEMPLOS):
        img_a = by_class[cls_a][i].to(DEVICE)
        img_b = by_class[cls_b][i].to(DEVICE)

        img_ab = (img_a + img_b)
        img_ab = img_ab / img_ab.max().clamp(min=1e-8)

        cmap_a,  field_a,  bruto_a  = run_field(img_a)
        cmap_b,  field_b,  bruto_b  = run_field(img_b)
        cmap_ab, field_ab, bruto_ab = run_field(img_ab)

        diferenca = cmap_ab.abs() - bruto_ab.abs()

        ax = axes[row]
        ax[0].imshow(norm(img_a).cpu().numpy(),    cmap='gray'); ax[0].axis('off')
        ax[1].imshow(norm(img_b).cpu().numpy(),    cmap='gray'); ax[1].axis('off')
        ax[2].imshow(norm(img_ab).cpu().numpy(),   cmap='gray'); ax[2].axis('off')
        ax[3].imshow(norm(bruto_a).cpu().numpy(),  cmap='hot');  ax[3].axis('off')
        ax[4].imshow(norm(bruto_b).cpu().numpy(),  cmap='hot');  ax[4].axis('off')
        ax[5].imshow(norm(bruto_ab).cpu().numpy(), cmap='hot');  ax[5].axis('off')
        ax[6].imshow(norm(cmap_ab).cpu().numpy(),  cmap='hot');  ax[6].axis('off')
        ax[7].imshow(diferenca.cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1); ax[7].axis('off')

        if i == 0:
            ax[0].set_ylabel(f'{cls_a}+{cls_b}', fontsize=11, fontweight='bold')

        row += 1
        print(f"  Par {cls_a}+{cls_b}, exemplo {i+1} OK")

plt.tight_layout()
plt.savefig('viz_audit_20_plasticidade.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n-> viz_audit_20_plasticidade.png")
print("Pronto.")
