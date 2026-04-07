"""
Auditoria 8: Cristais como Protótipos de Classe

Alimenta N exemplos da mesma classe no mesmo campo.
Os cristais acumulados formam um "protótipo" da classe.
Classificação por ressonância: entrada nova vs 10 protótipos.

Zero decoder. Zero treino. Só física.
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

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ── Constantes ──────────────────────────────────────────────────────────────
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


# ── Cristalização Competitiva ───────────────────────────────────────────────

class CrystalCompetitivo:
    def __init__(self, FS=FIELD_SIZE, sharpness=5.0, decay=0.02, ressonance_boost=0.1):
        self.crystal_map = torch.zeros(1, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(1, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(1, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(1, FS, FS, device=DEVICE)
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


# ── Física ──────────────────────────────────────────────────────────────────

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


def image_to_perturbation(img, PG):
    """Converte uma imagem MNIST em perturbação no campo."""
    return (img.view(1, 784) @ PG.to(img.dtype)).view(1, FIELD_SIZE, FIELD_SIZE)


# ── Construir protótipos ────────────────────────────────────────────────────

def build_prototype(images, PG):
    """
    Alimenta N imagens da mesma classe no MESMO campo.
    Cristais se acumulam. O crystal_map final é o protótipo.
    """
    mem = CrystalCompetitivo(FIELD_SIZE)

    for img in images:
        pert = image_to_perturbation(img, PG)
        field = pert.clone()
        velocity = torch.zeros_like(field)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                field, velocity = psi_step(field, velocity, pert, s < STIM_ON)
                mem.update_envelope(field)
                if mem.window_idx > 0:
                    mem.try_crystallize(field)
                field = mem.remit(field)

        # Campo reseta entre exemplos, mas cristais persistem
        # (o campo da próxima imagem começa limpo, cristais ficam)

    return mem.crystal_map.squeeze(0)  # (FS, FS)


# ── Classificação por ressonância ───────────────────────────────────────────

def classify_by_resonance(img, prototypes, PG):
    """
    Joga a imagem no campo COM cada protótipo.
    Mede a energia de interação (ressonância) entre onda e protótipo.
    Classe = protótipo com maior ressonância.
    """
    pert = image_to_perturbation(img, PG)
    scores = []

    for proto in prototypes:
        field = pert.clone()
        velocity = torch.zeros_like(field)

        # Injeta protótipo como crystal_map fixo
        proto_map = proto.unsqueeze(0)  # (1, FS, FS)

        with torch.no_grad():
            resonance_total = 0.0
            for s in range(STIM_TOTAL):
                field, velocity = psi_step(field, velocity, pert, s < STIM_ON)
                # Ressonância = correlação entre campo e protótipo
                resonance = (field.abs() * proto_map).sum().item()
                resonance_total += resonance

                # Re-emissão do protótipo
                if proto_map.abs().max() > 1e-6:
                    field = torch.clamp(
                        field + proto_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)

        scores.append(resonance_total)

    return np.argmax(scores)


def classify_by_correlation(img, prototypes, PG):
    """
    Método mais simples: computa crystal_map da imagem,
    mede correlação coseno com cada protótipo.
    Sem re-simular a física.
    """
    mem = CrystalCompetitivo(FIELD_SIZE)
    pert = image_to_perturbation(img, PG)
    field = pert.clone()
    velocity = torch.zeros_like(field)

    with torch.no_grad():
        for s in range(STIM_TOTAL):
            field, velocity = psi_step(field, velocity, pert, s < STIM_ON)
            mem.update_envelope(field)
            if mem.window_idx > 0:
                mem.try_crystallize(field)
            field = mem.remit(field)

    cmap = mem.crystal_map.squeeze(0).flatten()
    cmap_norm = cmap / (cmap.norm() + 1e-8)

    scores = []
    for proto in prototypes:
        p = proto.flatten()
        p_norm = p / (p.norm() + 1e-8)
        cos_sim = (cmap_norm * p_norm).sum().item()
        scores.append(cos_sim)

    return np.argmax(scores)


# ── MNIST ────────────────────────────────────────────────────────────────────

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

# Organiza por classe
train_by_class = {i: [] for i in range(10)}
for img, label in train_ds:
    train_by_class[label].append(img.squeeze(0).to(DEVICE))

test_images = []
test_labels = []
for img, label in test_ds:
    test_images.append(img.squeeze(0).to(DEVICE))
    test_labels.append(label)
test_labels = np.array(test_labels)

PG = build_gaussians()

# ── Experimento ──────────────────────────────────────────────────────────────

N_PROTO_LIST = [5, 10, 20, 50]  # exemplos por protótipo
N_TEST = 500  # imagens de teste (50 por classe)

print(f"\n{'='*70}")
print("AUDITORIA 8: Cristais como Protótipos de Classe")
print(f"Teste: {N_TEST} imagens | Métodos: ressonância + correlação")
print(f"{'='*70}")

# Subset de teste balanceado
test_subset_idx = []
counts = [0] * 10
for i, label in enumerate(test_labels):
    if counts[label] < N_TEST // 10:
        test_subset_idx.append(i)
        counts[label] += 1
    if all(c >= N_TEST // 10 for c in counts):
        break

all_results = []
t0 = time.time()

for n_proto in N_PROTO_LIST:
    print(f"\n── {n_proto} exemplos por protótipo ──")

    # Constrói protótipos
    print(f"  Construindo 10 protótipos ({n_proto} imgs cada)...")
    t1 = time.time()
    prototypes = []
    for cls in range(10):
        imgs = train_by_class[cls][:n_proto]
        proto = build_prototype(imgs, PG)
        n_crys = (proto > 0.01).float().sum().item()
        prototypes.append(proto)
        print(f"    Classe {cls}: {n_crys:.0f} cristais")
    proto_time = time.time() - t1
    print(f"  Protótipos prontos: {proto_time:.0f}s")

    # Classifica por correlação (rápido)
    print(f"  Classificando por correlação ({N_TEST} imgs)...")
    t1 = time.time()
    correct_corr = 0
    for idx in test_subset_idx:
        pred = classify_by_correlation(test_images[idx], prototypes, PG)
        if pred == test_labels[idx]:
            correct_corr += 1
    acc_corr = correct_corr / len(test_subset_idx) * 100
    corr_time = time.time() - t1
    print(f"  Correlação: {acc_corr:.1f}% ({corr_time:.0f}s)")

    # Classifica por ressonância (lento — só 100 imagens)
    n_res_test = min(100, len(test_subset_idx))
    print(f"  Classificando por ressonância ({n_res_test} imgs)...")
    t1 = time.time()
    correct_res = 0
    for i, idx in enumerate(test_subset_idx[:n_res_test]):
        pred = classify_by_resonance(test_images[idx], prototypes, PG)
        if pred == test_labels[idx]:
            correct_res += 1
        if (i+1) % 20 == 0:
            print(f"    {i+1}/{n_res_test}...", end='\r')
    acc_res = correct_res / n_res_test * 100
    res_time = time.time() - t1
    print(f"  Ressonância: {acc_res:.1f}% ({res_time:.0f}s)       ")

    all_results.append({
        'n_proto': n_proto,
        'acc_corr': acc_corr,
        'acc_res': acc_res,
        'proto_time': proto_time,
    })

    print(f"  → Correlação: {acc_corr:.1f}% | Ressonância: {acc_res:.1f}%")


# ── Resumo ───────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESULTADO FINAL: Classificação por Protótipos Cristalinos")
print(f"{'='*70}")
print(f"\n{'N exemplos':>12} {'Correlação':>12} {'Ressonância':>13}")
print("-"*40)
for r in all_results:
    print(f"  {r['n_proto']:>10}   {r['acc_corr']:>8.1f}%   {r['acc_res']:>9.1f}%")

# Referências
print(f"\nReferência: classificação aleatória = 10.0%")
print(f"Referência: crystal_map + linear decoder = 80.8%")
print(f"Referência: crystal_map competitivo + linear = 88.1%")
print(f"\nTempo total: {time.time()-t0:.0f}s")


# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Auditoria 8 — Cristais como Protótipos de Classe', fontsize=13)

ns = [r['n_proto'] for r in all_results]
acc_corrs = [r['acc_corr'] for r in all_results]
acc_ress  = [r['acc_res'] for r in all_results]

# Acurácia vs N exemplos
ax = axes[0]
ax.plot(ns, acc_corrs, 'o-', color='#e6194b', label='Correlação coseno', linewidth=2, markersize=8)
ax.plot(ns, acc_ress,  's-', color='#3cb44b', label='Ressonância física', linewidth=2, markersize=8)
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Aleatório (10%)')
ax.axhline(y=80.8, color='blue', linestyle='--', alpha=0.5, label='Crystal+Linear decoder (80.8%)')
ax.axhline(y=88.1, color='orange', linestyle='--', alpha=0.5, label='Competitivo+Linear (88.1%)')
ax.set_xlabel('Exemplos por protótipo')
ax.set_ylabel('Acurácia (%)')
ax.set_title('Acurácia vs N exemplos por classe')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Protótipos visualizados
ax = axes[1]
ax.set_title('Protótipos cristalinos (50 exemplos)')
ax.axis('off')
# Mini-grid 2×5 dos protótipos do último experimento
if len(all_results) > 0:
    last_protos = prototypes  # do último loop
    for i in range(10):
        row, col = i // 5, i % 5
        sub = fig.add_axes([0.55 + col*0.085, 0.55 - row*0.4, 0.08, 0.35])
        sub.imshow(last_protos[i].cpu().numpy(), cmap='hot', aspect='auto')
        sub.set_title(f'{i}', fontsize=8)
        sub.axis('off')

plt.savefig('viz_audit_8_prototipos.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_8_prototipos.png")
print("Pronto.")
