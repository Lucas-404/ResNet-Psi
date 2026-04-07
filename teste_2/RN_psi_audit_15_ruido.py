"""
Auditoria 15: Mapeamento de Cristais Fantasma

Cristais que aparecem sem estímulo = ruído estrutural do campo.
Esses cristais nascem das frequências naturais do campo, não da entrada.
Devem ser removidos para melhorar a discriminabilidade.

Pipeline:
1. Rodar campo com entrada zero → crystal_map fantasma
2. Rodar campo com entradas reais → crystal_maps normais
3. Subtrair fantasma → crystal_maps limpos
4. Comparar acurácia antes/depois da limpeza
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
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


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


def run_field(pert):
    """Roda o campo com perturbação dada. Retorna crystal_map."""
    B = pert.shape[0]
    f, v = pert.clone(), torch.zeros_like(pert)
    mem = CrystalCompetitivo(B, FIELD_SIZE)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f, v = psi_step(f, v, pert, s < STIM_ON)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)
    return mem.crystal_map


def compute_crystal_maps_batch(X, PG, bs=64):
    N, out = len(X), []
    for i in range(0, N, bs):
        B = X[i:i+bs]
        pert = (B.view(len(B), 784) @ PG.to(B.dtype)).view(len(B), FIELD_SIZE, FIELD_SIZE)
        out.append(run_field(pert))
    return torch.cat(out, dim=0)


# ── Carregar MNIST ────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

train_by_class = {i: [] for i in range(10)}
for img, label in train_ds:
    train_by_class[label].append(img.squeeze(0))

test_imgs, test_labels = [], []
for img, label in test_ds:
    test_imgs.append(img.squeeze(0))
    test_labels.append(label)
test_labels = np.array(test_labels)

PG = build_gaussians()

# ── 1. Mapa de cristais fantasma (entrada zero) ───────────────────────────────

print("\n── Passo 1: Cristais fantasma (entrada zero) ──")
N_ZERO = 50
zero_input = torch.zeros(N_ZERO, FIELD_SIZE, FIELD_SIZE, device=DEVICE)
ghost_cmaps = run_field(zero_input)
ghost_mask = ghost_cmaps.mean(dim=0)  # (FS, FS) — média dos cristais fantasma

n_ghost = (ghost_mask > 0.01).float().sum().item()
print(f"  Cristais fantasma: {n_ghost:.0f} posições ativas (de {FIELD_SIZE**2})")
print(f"  Cobertura: {100*n_ghost/FIELD_SIZE**2:.1f}% do campo")

# ── 2. Crystal maps reais (MNIST) ─────────────────────────────────────────────

print("\n── Passo 2: Crystal maps reais (MNIST, 500 por classe) ──")
N_PROTO = 500
N_TEST  = 500

# Teste
test_subset_idx = []
counts = [0]*10
for i, label in enumerate(test_labels):
    if counts[label] < N_TEST//10:
        test_subset_idx.append(i)
        counts[label] += 1
    if all(c >= N_TEST//10 for c in counts):
        break

test_tensor = torch.stack([test_imgs[i] for i in test_subset_idx]).to(DEVICE)
test_labels_sub = test_labels[test_subset_idx]
print(f"  Computando {len(test_subset_idx)} crystal_maps de teste...")
t1 = time.time()
test_cmaps = compute_crystal_maps_batch(test_tensor, PG)
print(f"  Pronto: {time.time()-t1:.0f}s")

# Protótipos
print(f"  Computando protótipos ({N_PROTO} por classe)...")
t1 = time.time()
prototypes = {}
for cls in range(10):
    imgs = torch.stack(train_by_class[cls][:N_PROTO]).to(DEVICE)
    cmaps = compute_crystal_maps_batch(imgs, PG)
    prototypes[cls] = cmaps.mean(dim=0)
print(f"  Pronto: {time.time()-t1:.0f}s")

# ── 3. Classificação SEM filtro ───────────────────────────────────────────────

print("\n── Passo 3: Classificação SEM filtro de ruído ──")
proto_matrix = torch.stack([prototypes[cls] for cls in range(10)]).view(10, -1).float()
test_flat = test_cmaps.view(len(test_subset_idx), -1).float()
dists = torch.cdist(test_flat, proto_matrix)
preds = dists.argmin(dim=1).cpu().numpy()
acc_sem_filtro = (preds == test_labels_sub).mean() * 100
print(f"  Acurácia SEM filtro: {acc_sem_filtro:.1f}%")

# ── 4. Filtro: remover cristais comuns a todas as classes ─────────────────────

print("\n── Passo 4: Filtro de variância entre classes ──")

# Empilhar protótipos: (10, FS, FS)
proto_stack = torch.stack([prototypes[cls] for cls in range(10)])

# Variância entre classes por posição
inter_class_var = proto_stack.var(dim=0)  # (FS, FS)

# Posições com baixa variância entre classes = aparecem igual em todas = ruído
threshold = inter_class_var.mean() * 0.5  # abaixo de 50% da média = ruído
noise_mask = (inter_class_var < threshold).float()  # 1 onde tem ruído
clean_mask = 1.0 - noise_mask                        # 1 onde NÃO tem ruído

n_removidos = noise_mask.sum().item()
n_mantidos  = clean_mask.sum().item()
print(f"  Variância inter-classe média: {inter_class_var.mean().item():.4f}")
print(f"  Threshold: {threshold.item():.4f}")
print(f"  Posições removidas (baixa variância): {n_removidos:.0f} ({100*n_removidos/FIELD_SIZE**2:.1f}%)")
print(f"  Posições mantidas (alta variância):   {n_mantidos:.0f} ({100*n_mantidos/FIELD_SIZE**2:.1f}%)")

# Aplicar máscara
test_cmaps_clean = test_cmaps * clean_mask.unsqueeze(0)
prototypes_clean = {cls: prototypes[cls] * clean_mask for cls in range(10)}

# ── 5. Classificação COM filtro ───────────────────────────────────────────────

print("\n── Passo 5: Classificação COM filtro de ruído ──")
proto_matrix_clean = torch.stack([prototypes_clean[cls] for cls in range(10)]).view(10, -1).float()
test_flat_clean = test_cmaps_clean.view(len(test_subset_idx), -1).float()
dists_clean = torch.cdist(test_flat_clean, proto_matrix_clean)
preds_clean = dists_clean.argmin(dim=1).cpu().numpy()
acc_com_filtro = (preds_clean == test_labels_sub).mean() * 100
print(f"  Acurácia COM filtro: {acc_com_filtro:.1f}%")

# ── Resumo ────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("RESULTADO: Filtro de Cristais Fantasma")
print(f"{'='*60}")
print(f"  Sem filtro:  {acc_sem_filtro:.1f}%")
print(f"  Com filtro:  {acc_com_filtro:.1f}%")
print(f"  Diferença:   {acc_com_filtro - acc_sem_filtro:+.1f}%")
print(f"  Cristais removidos: {n_removidos:.0f} ({100*n_removidos/FIELD_SIZE**2:.1f}% do campo)")

# ── Visualização ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(f'Auditoria 15 — Filtro de Cristais Fantasma\n'
             f'Sem filtro: {acc_sem_filtro:.1f}% | Com filtro: {acc_com_filtro:.1f}%', fontsize=13)

# Mapa de variância entre classes
ax = axes[0][0]
ax.imshow(inter_class_var.cpu().numpy(), cmap='hot')
ax.set_title(f'Variância Entre Classes\n(alto = discriminativo)')
ax.axis('off')

# Máscara de ruído
ax = axes[0][1]
ax.imshow(noise_mask.cpu().numpy(), cmap='Reds')
ax.set_title(f'Máscara de Ruído\n({n_removidos:.0f} removidos)')
ax.axis('off')

# Máscara limpa
ax = axes[0][2]
ax.imshow(clean_mask.cpu().numpy(), cmap='Greens')
ax.set_title(f'Máscara Limpa\n({n_mantidos:.0f} mantidos)')
ax.axis('off')

# Comparação acurácias
ax = axes[0][3]
bars = ax.bar(['Sem filtro', 'Com filtro'], [acc_sem_filtro, acc_com_filtro],
              color=['#e6194b', '#3cb44b'], alpha=0.8)
ax.set_ylabel('Acurácia (%)')
ax.set_title('Impacto do Filtro')
ax.set_ylim(0, 100)
for bar, val in zip(bars, [acc_sem_filtro, acc_com_filtro]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center', fontsize=11)
ax.grid(alpha=0.3, axis='y')

# Protótipos: dígito 0 sem e com filtro
for i, cls in enumerate([0, 3, 7, 9]):
    ax = axes[1][i]
    proto_orig  = prototypes[cls].cpu().numpy()
    proto_clean = prototypes_clean[cls].cpu().numpy()
    # Lado a lado: original (esquerda) | limpo (direita)
    combined = np.concatenate([proto_orig, proto_clean], axis=1)
    ax.imshow(combined, cmap='hot', aspect='auto')
    ax.set_title(f'Dígito {cls}\nOriginal | Limpo', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('viz_audit_15_ruido.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_15_ruido.png")
print("Pronto.")
