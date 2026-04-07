"""
Auditoria 17: Geração por Ressonância — Completar Padrão Parcial

Fase 1 — Memorização: campo processa imagem completa, cristais armazenam padrão.
Fase 2 — Geração: campo recebe metade da imagem, cristais ressoam e completam.

O campo gera a parte que faltava — sem treino, só física.
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
CRYSTAL_REMIT  = 0.3   # mais forte para geração

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


def memorizar(img_tensor, PG):
    """Fase 1: campo processa imagem completa. Retorna crystal_map memorizado."""
    pert = (img_tensor.view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)
    f, v = pert.clone(), torch.zeros_like(pert)
    mem = CrystalCompetitivo(1, FIELD_SIZE)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f, v = psi_step(f, v, pert, s < STIM_ON)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)
    return mem  # retorna o objeto com crystal_map e crystal_hp


def gerar(img_parcial, mem, PG, steps_geracao=120, remit_strength=0.3):
    """
    Fase 2: campo recebe imagem parcial, cristais memorizados ressoam e completam.
    Retorna o estado final do campo como 'geração'.
    """
    pert = (img_parcial.view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)
    f, v = pert.clone(), torch.zeros_like(pert)

    with torch.no_grad():
        for s in range(steps_geracao):
            # Estímulo parcial ativo por metade do tempo
            f, v = psi_step(f, v, pert, s < steps_geracao // 2)
            # Cristais memorizados re-emitem continuamente
            f = mem.remit(f, strength=remit_strength)

    return f.squeeze(0)  # (FS, FS) — estado final do campo


def field_to_image(field, field_size=FIELD_SIZE):
    """Converte campo de volta para imagem 28x28 via projeção inversa simples."""
    # Usar energia do campo em regiões correspondentes a cada pixel
    coords = torch.linspace(0., 1., field_size, device=field.device)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')

    img = torch.zeros(28, 28, device=field.device)
    sigma = 0.04
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            gauss = torch.exp(-((xg-cx)**2+(yg-cy)**2)/(2*sigma**2))
            # Energia do campo nessa região
            img[pi, pj] = (field.abs() * gauss).sum()
    return img


# ── Carregar MNIST ────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor()])
test_ds = datasets.MNIST('./data', train=False, download=True, transform=tf)

PG = build_gaussians()

# Selecionar exemplos de cada dígito
examples = {}
for img, label in test_ds:
    if label not in examples:
        examples[label] = img.squeeze(0)
    if len(examples) == 10:
        break

# ── Experimento ───────────────────────────────────────────────────────────────

print("\n── Teste de Geração por Ressonância ──")
print("Memoriza imagem completa → apresenta metade → campo completa")

DIGITS_TO_TEST = [0, 3, 7, 9]
N_MEMORIZAR = 50  # exemplos por classe para construir memória coletiva

# ── Carregar treino para memória coletiva ─────────────────────────────────────
print("\nCarregando MNIST treino para memória coletiva...")
train_ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
train_by_class = {i: [] for i in range(10)}
for img, label in train_ds:
    if len(train_by_class[label]) < N_MEMORIZAR:
        train_by_class[label].append(img.squeeze(0))

def memorizar_coletivo(imgs_list, PG):
    """Memoriza vários exemplos — cristais acumulam o padrão coletivo da classe."""
    mem_coletivo = CrystalCompetitivo(1, FIELD_SIZE)
    for img in imgs_list:
        pert = (img.to(DEVICE).view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)
        f, v = pert.clone(), torch.zeros_like(pert)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem_coletivo.update_envelope(f)
                if mem_coletivo.window_idx > 0:
                    mem_coletivo.try_crystallize(f)
                f = mem_coletivo.remit(f)
    return mem_coletivo

def norm(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)

# ── Experimento 1: memória individual (igual antes) ───────────────────────────
print("\n── Experimento 1: Memória Individual (1 exemplo) ──")
fig1, axes1 = plt.subplots(len(DIGITS_TO_TEST), 5, figsize=(18, 4*len(DIGITS_TO_TEST)))
fig1.suptitle('Memória Individual — Memoriza 1 exemplo → completa', fontsize=13)
col_titles = ['Original', 'Metade sup.', 'Campo gerado', 'Reconstruído', 'Crystal map']
for j, title in enumerate(col_titles):
    axes1[0][j].set_title(title, fontsize=10, fontweight='bold')

for row, digit in enumerate(DIGITS_TO_TEST):
    img = examples[digit].to(DEVICE)
    img_top = img.clone(); img_top[14:, :] = 0.0
    mem = memorizar(img, PG)
    campo_gerado = gerar(img_top, mem, PG)
    img_rec = field_to_image(campo_gerado)
    axes1[row][0].imshow(norm(img).cpu().numpy(), cmap='gray')
    axes1[row][0].set_ylabel(f'Dígito {digit}', fontsize=10)
    axes1[row][0].axis('off')
    axes1[row][1].imshow(norm(img_top).cpu().numpy(), cmap='gray')
    axes1[row][1].axis('off')
    axes1[row][2].imshow(norm(campo_gerado).cpu().numpy(), cmap='hot')
    axes1[row][2].axis('off')
    axes1[row][3].imshow(norm(img_rec).cpu().numpy(), cmap='gray')
    axes1[row][3].axis('off')
    axes1[row][4].imshow(norm(mem.crystal_map.squeeze(0)).cpu().numpy(), cmap='hot')
    axes1[row][4].axis('off')
    print(f"  Dígito {digit}: {(mem.crystal_map > 0.01).float().sum().item():.0f} cristais")

plt.tight_layout()
plt.savefig('viz_audit_17a_individual.png', dpi=130, bbox_inches='tight')
plt.close()
print("-> viz_audit_17a_individual.png")

# ── Experimento 2: memória coletiva (N exemplos) ──────────────────────────────
print(f"\n── Experimento 2: Memória Coletiva ({N_MEMORIZAR} exemplos por classe) ──")
fig2, axes2 = plt.subplots(len(DIGITS_TO_TEST), 5, figsize=(18, 4*len(DIGITS_TO_TEST)))
fig2.suptitle(f'Memória Coletiva — Memoriza {N_MEMORIZAR} exemplos → completa padrão novo', fontsize=13)
for j, title in enumerate(col_titles):
    axes2[0][j].set_title(title, fontsize=10, fontweight='bold')

for row, digit in enumerate(DIGITS_TO_TEST):
    print(f"  Dígito {digit}: memorizando {N_MEMORIZAR} exemplos...")
    t1 = time.time()
    mem_col = memorizar_coletivo(train_by_class[digit], PG)
    n_cris = (mem_col.crystal_map > 0.01).float().sum().item()
    print(f"    {n_cris:.0f} cristais coletivos ({time.time()-t1:.1f}s)")

    # Testar com exemplo DIFERENTE dos memorizado
    img_teste = examples[digit].to(DEVICE)
    img_top = img_teste.clone(); img_top[14:, :] = 0.0
    campo_gerado = gerar(img_top, mem_col, PG)
    img_rec = field_to_image(campo_gerado)

    axes2[row][0].imshow(norm(img_teste).cpu().numpy(), cmap='gray')
    axes2[row][0].set_ylabel(f'Dígito {digit}', fontsize=10)
    axes2[row][0].axis('off')
    axes2[row][1].imshow(norm(img_top).cpu().numpy(), cmap='gray')
    axes2[row][1].axis('off')
    axes2[row][2].imshow(norm(campo_gerado).cpu().numpy(), cmap='hot')
    axes2[row][2].axis('off')
    axes2[row][3].imshow(norm(img_rec).cpu().numpy(), cmap='gray')
    axes2[row][3].axis('off')
    axes2[row][4].imshow(norm(mem_col.crystal_map.squeeze(0)).cpu().numpy(), cmap='hot')
    axes2[row][4].axis('off')

plt.tight_layout()
plt.savefig('viz_audit_17b_coletiva.png', dpi=130, bbox_inches='tight')
plt.close()
print("-> viz_audit_17b_coletiva.png")
print("Pronto.")
