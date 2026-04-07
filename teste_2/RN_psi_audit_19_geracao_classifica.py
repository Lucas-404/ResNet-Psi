"""
Auditoria 19: Geracao + Classificacao em Sequencia

Pipeline:
1. Prototipos: memoriza N exemplos por classe (Audit 8)
2. Geracao: campo recebe metade da imagem, cristais completam (Audit 17)
3. Classificacao: crystal_map do campo gerado vs prototipos (Audit 8)

Pergunta: completar o padrao antes de classificar melhora a acuracia
comparado com classificar direto da metade?
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


def run_field(imgs_batch, PG):
    """Roda campo em batch, retorna crystal_maps."""
    B = imgs_batch.shape[0]
    pert = (imgs_batch.view(B, 784) @ PG).view(B, FIELD_SIZE, FIELD_SIZE)
    f, v = pert.clone(), torch.zeros_like(pert)
    mem = CrystalCompetitivo(B, FIELD_SIZE)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f, v = psi_step(f, v, pert, s < STIM_ON)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)
    return mem.crystal_map  # (B, FS, FS)


def gerar_campo(img_parcial, mem_classe, steps=120, remit_strength=0.3):
    """
    Fase de geracao: campo recebe entrada parcial,
    cristais da classe re-emitem e completam o padrao.
    Retorna o campo final e um novo crystal_map extraido dele.
    """
    pert = (img_parcial.view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)
    f, v = pert.clone(), torch.zeros_like(pert)
    with torch.no_grad():
        for s in range(steps):
            f, v = psi_step(f, v, pert, s < steps // 2)
            f = mem_classe.remit(f, strength=remit_strength)
    return f  # (1, FS, FS)


def field_to_image(field, field_size=FIELD_SIZE, sigma=0.04):
    """Converte campo (FS,FS) para imagem 28x28 via broadcasting — sem matriz."""
    coords = torch.linspace(0., 1., field_size, device=field.device)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')  # (FS, FS)

    px = torch.linspace(0.1, 0.9, 28, device=field.device)
    py = torch.linspace(0.1, 0.9, 28, device=field.device)
    cx, cy = torch.meshgrid(px, py, indexing='ij')  # (28, 28)

    # (28, 28, FS, FS) via broadcasting
    dx = xg.unsqueeze(0).unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1)
    dy = yg.unsqueeze(0).unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1)
    gauss = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))  # (28, 28, FS, FS)

    return (field.abs().unsqueeze(0).unsqueeze(0) * gauss).sum(dim=(-2, -1))  # (28, 28)


def norm(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)


# -- Carregar MNIST -----------------------------------------------------------

print("\nCarregando MNIST...")
tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

PG = build_gaussians()

N_PROTO = 500   # exemplos por classe para prototipos
N_TEST  = 200   # exemplos de teste por classe

# Separar treino por classe
train_by_class = {i: [] for i in range(10)}
for img, label in train_ds:
    train_by_class[label].append(img.squeeze(0))

# Separar teste por classe
test_by_class = {i: [] for i in range(10)}
for img, label in test_ds:
    if len(test_by_class[label]) < N_TEST:
        test_by_class[label].append(img.squeeze(0))
    if all(len(v) >= N_TEST for v in test_by_class.values()):
        break

# -- Fase 1: Construir prototipos por classe ----------------------------------

print(f"\n-- Fase 1: Prototipos ({N_PROTO} exemplos/classe) --")
t0 = time.time()
prototipos = {}
mems_classe = {}
for cls in range(10):
    imgs = torch.stack(train_by_class[cls][:N_PROTO]).to(DEVICE)
    # Prototipos: media dos crystal_maps (como Audit 8)
    cmaps = run_field(imgs, PG)
    prototipos[cls] = cmaps.mean(dim=0)  # (FS, FS)

    # Memoria coletiva da classe (para geracao)
    mem = CrystalCompetitivo(1, FIELD_SIZE)
    for img in train_by_class[cls][:N_PROTO]:
        pert = (img.to(DEVICE).view(1, 784) @ PG).view(1, FIELD_SIZE, FIELD_SIZE)
        ff, vv = pert.clone(), torch.zeros_like(pert)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                ff, vv = psi_step(ff, vv, pert, s < STIM_ON)
                mem.update_envelope(ff)
                if mem.window_idx > 0:
                    mem.try_crystallize(ff)
                ff = mem.remit(ff)
    mems_classe[cls] = mem
    print(f"  Classe {cls}: prototipo pronto, {(mem.crystal_map > 0.01).float().sum().item():.0f} cristais na memoria")

print(f"  Tempo: {time.time()-t0:.0f}s")

proto_matrix = torch.stack([prototipos[cls] for cls in range(10)]).view(10, -1).float()

# -- Fase 2: Classificar com imagem completa vs metade vs geracao -------------

print(f"\n-- Fase 2: Comparando 3 abordagens com {N_TEST} exemplos/classe --")

resultados = {'completa': [], 'metade': [], 'gerada': []}
labels_all = []

t1 = time.time()
for cls in range(10):
    for img in test_by_class[cls]:
        img = img.to(DEVICE)
        img_metade = img.clone()
        img_metade[14:, :] = 0.0  # zera metade inferior

        labels_all.append(cls)

        # Abordagem 1: imagem completa → crystal_map → distancia ao prototipo
        cmap_completa = run_field(img.unsqueeze(0), PG).squeeze(0).view(-1).float()

        # Abordagem 2: metade da imagem → crystal_map → distancia ao prototipo
        cmap_metade = run_field(img_metade.unsqueeze(0), PG).squeeze(0).view(-1).float()

        # Abordagem 3: metade → gerar com media de todas as memorias → field_to_image → run_field → classificar
        # Usar media dos campos gerados por cada classe (sem saber a classe de antemao)
        # Alternativa mais simples: gerar sem memoria especifica, so com o campo livre
        # Gerar com cada memoria, converter para imagem, extrair crystal_map, classificar
        campos_img = []
        for c in range(10):
            campo_gerado = gerar_campo(img_metade, mems_classe[c]).squeeze(0)  # (FS, FS)
            img_gerada = field_to_image(campo_gerado)  # (28, 28)
            campos_img.append(img_gerada)
        # Media das imagens geradas pelas 10 memorias
        img_gerada_media = torch.stack(campos_img).mean(dim=0)  # (28, 28)
        cmap_gerado = run_field(img_gerada_media.unsqueeze(0), PG).squeeze(0).view(-1).float()
        dists_gerado = torch.cdist(cmap_gerado.unsqueeze(0), proto_matrix).squeeze(0)
        pred_gerada = dists_gerado.argmin().item()
        resultados['gerada'].append(pred_gerada)

        # Classificar completa e metade
        dists_c = torch.cdist(cmap_completa.unsqueeze(0), proto_matrix).squeeze(0)
        resultados['completa'].append(dists_c.argmin().item())

        dists_m = torch.cdist(cmap_metade.unsqueeze(0), proto_matrix).squeeze(0)
        resultados['metade'].append(dists_m.argmin().item())

    print(f"  Classe {cls} OK ({time.time()-t1:.0f}s acumulado)")

labels_np = np.array(labels_all)
acc_completa = (np.array(resultados['completa']) == labels_np).mean() * 100
acc_metade   = (np.array(resultados['metade'])   == labels_np).mean() * 100
acc_gerada   = (np.array(resultados['gerada'])   == labels_np).mean() * 100

print(f"\n{'='*50}")
print(f"RESULTADO FINAL ({N_TEST} exemplos/classe = {N_TEST*10} total)")
print(f"{'='*50}")
print(f"  Imagem completa  : {acc_completa:.1f}%")
print(f"  Metade da imagem : {acc_metade:.1f}%")
print(f"  Geracao + classif: {acc_gerada:.1f}%")
print(f"  Ganho geracao vs metade: {acc_gerada - acc_metade:+.1f}%")

# -- Plot ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Auditoria 19 - Geracao + Classificacao em Sequencia', fontsize=13)

ax = axes[0]
nomes = ['Completa', 'Metade', 'Geracao+Class']
accs  = [acc_completa, acc_metade, acc_gerada]
cores = ['#3cb44b', '#e6194b', '#4363d8']
bars = ax.bar(nomes, accs, color=cores, alpha=0.85)
ax.set_ylabel('Acuracia (%)')
ax.set_title('Comparacao das 3 abordagens')
ax.set_ylim(0, 100)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center', fontsize=11)
ax.axhline(y=77.4, color='orange', linestyle='--', alpha=0.6, label='Ref 77.4%')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Mostrar exemplos de geracao para alguns digitos
ax = axes[1]
ax.axis('off')
ax.set_title('Ver viz_audit_19_exemplos.png para exemplos visuais')

plt.tight_layout()
plt.savefig('viz_audit_19_resultado.png', dpi=130, bbox_inches='tight')
plt.close()

# -- Exemplos visuais ---------------------------------------------------------

DIGITOS_VIZ = [0, 1, 3, 7]
fig2, axes2 = plt.subplots(len(DIGITOS_VIZ), 5, figsize=(18, 4*len(DIGITOS_VIZ)))
fig2.suptitle(f'Auditoria 19 - Exemplos de Geracao\n'
              f'Completa={acc_completa:.1f}% | Metade={acc_metade:.1f}% | Gerada={acc_gerada:.1f}%',
              fontsize=12)

col_titles = ['Original', 'Metade sup.', 'Campo gerado\n(mem. classe correta)', 'Cristais mem.\n(classe correta)', 'Energias por classe']
for j, title in enumerate(col_titles):
    axes2[0][j].set_title(title, fontsize=9, fontweight='bold')

for row, digit in enumerate(DIGITOS_VIZ):
    img = test_by_class[digit][0].to(DEVICE)
    img_metade = img.clone(); img_metade[14:, :] = 0.0

    campo_gerado = gerar_campo(img_metade, mems_classe[digit]).squeeze(0)
    img_gerada_viz = field_to_image(campo_gerado)
    cmap_gerado_viz = run_field(img_gerada_viz.unsqueeze(0), PG).squeeze(0).view(-1).float()

    # Energias: distancia do crystal_map gerado ao prototipo de cada classe
    dists_viz = torch.cdist(cmap_gerado_viz.unsqueeze(0), proto_matrix).squeeze(0).cpu().numpy()
    pred_viz = dists_viz.argmin()

    axes2[row][0].imshow(norm(img).cpu().numpy(), cmap='gray')
    axes2[row][0].set_ylabel(f'Digito {digit}', fontsize=9)
    axes2[row][0].axis('off')

    axes2[row][1].imshow(norm(img_metade).cpu().numpy(), cmap='gray')
    axes2[row][1].axis('off')

    axes2[row][2].imshow(norm(campo_gerado).cpu().numpy(), cmap='hot')
    axes2[row][2].axis('off')

    axes2[row][3].imshow(norm(mems_classe[digit].crystal_map.squeeze(0)).cpu().numpy(), cmap='hot')
    axes2[row][3].axis('off')

    ax5 = axes2[row][4]
    cores_bar = ['green' if c == digit else ('red' if c == pred_viz else 'gray') for c in range(10)]
    ax5.bar(range(10), -dists_viz, color=cores_bar, alpha=0.8)
    ax5.set_xticks(range(10))
    ax5.set_xticklabels([str(c) for c in range(10)], fontsize=7)
    simbolo = 'OK' if pred_viz == digit else 'X'
    ax5.set_title(f'-> {pred_viz} {simbolo}', fontsize=10,
                  color='green' if pred_viz == digit else 'red')
    ax5.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('viz_audit_19_exemplos.png', dpi=130, bbox_inches='tight')
plt.close()

print(f"\n-> viz_audit_19_resultado.png")
print(f"-> viz_audit_19_exemplos.png")
print("Pronto.")
