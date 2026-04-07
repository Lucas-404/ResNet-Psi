"""
ResNet-Ψ N-Dimensional

O campo tem a mesma dimensão do dado.
Dado 1D → campo 1D. Dado 2D → campo 2D. Dado 3D → campo 3D.
Sem porteiro, sem projeção, sem achatar.

A equação de onda é a mesma em qualquer dimensão:
    acc = c²∇²Ψ − γ·v + α·tanh(Ψ)·Ψ − β·Ψ³

Os cristais se formam ao redor do dado, na dimensão natural dele.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# Constantes físicas (mesmas de sempre)
PSI_DT         = 0.05
PSI_GAMMA      = 0.06
PSI_ALPHA      = 0.04
PSI_BETA       = 0.005
PSI_C2         = 0.3
STIM_ON        = 40
STIM_TOTAL     = 80

# Cristalização
CRYSTAL_W      = 20
CRYSTAL_K      = 3
CRYSTAL_A_MIN  = 0.3
CRYSTAL_CV_MAX = 0.15
CRYSTAL_SEP    = 5
CRYSTAL_REMIT  = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# LAPLACIANO N-DIMENSIONAL
# ══════════════════════════════════════════════════════════════════════════════

def build_laplacian_kernel(ndim):
    """
    Constroi kernel do Laplaciano para qualquer dimensão.
    1D: [1, -2, 1]
    2D: [[0,1,0],[1,-4,1],[0,1,0]]
    3D: kernel 3x3x3 com centro=-6, faces=1
    ND: generalização — centro = -2*ndim, vizinhos = 1
    """
    shape = [3] * ndim
    kernel = torch.zeros(shape, device=DEVICE)

    # Centro = -2 * ndim
    center = tuple([1] * ndim)
    kernel[center] = -2.0 * ndim

    # Cada vizinho direto (face) = 1
    for d in range(ndim):
        for offset in [-1, 1]:
            idx = list(center)
            idx[d] += offset
            kernel[tuple(idx)] = 1.0

    # Reshape para conv: (1, 1, *shape)
    return kernel.view(1, 1, *shape)


def laplacian(field, lap_kernel, ndim):
    """
    Aplica Laplaciano N-dimensional com fronteira circular.
    field: (B, *spatial_dims)
    """
    # Adiciona dimensão de canal: (B, 1, *spatial_dims)
    x = field.unsqueeze(1)

    # Padding circular: 1 em cada lado, cada dimensão
    pad = (1, 1) * ndim  # F.pad espera (dim_n, dim_n, ..., dim_1, dim_1) de trás pra frente
    x = F.pad(x, pad, mode='circular')

    # Convolução N-dimensional
    if ndim == 1:
        out = F.conv1d(x, lap_kernel)
    elif ndim == 2:
        out = F.conv2d(x, lap_kernel)
    elif ndim == 3:
        out = F.conv3d(x, lap_kernel)
    else:
        raise ValueError(f"Laplaciano suporta até 3D, recebeu {ndim}D")

    return out.squeeze(1)  # (B, *spatial_dims)


# ══════════════════════════════════════════════════════════════════════════════
# EQUAÇÃO DE ONDA N-DIMENSIONAL
# ══════════════════════════════════════════════════════════════════════════════

def psi_step_nd(field, velocity, sources, active, lap_kernel, ndim):
    """
    Um step da equação de onda em qualquer dimensão.
        acc = c²∇²Ψ − γ·v + α·tanh(Ψ)·Ψ − β·Ψ³
    """
    dt    = PSI_DT
    gamma = PSI_GAMMA
    alpha = PSI_ALPHA
    beta  = PSI_BETA
    c2    = PSI_C2

    if active:
        field = field + sources * (dt * 0.1)

    lap = laplacian(field, lap_kernel, ndim)
    acc = c2 * lap - gamma * velocity + alpha * torch.tanh(field) * field - beta * field * field**2
    velocity = torch.clamp(velocity + acc * dt, -5., 5.)
    field    = torch.clamp(field + velocity * dt, -10., 10.)
    return field, velocity


# ══════════════════════════════════════════════════════════════════════════════
# CRISTALIZAÇÃO COMPETITIVA N-DIMENSIONAL
# ══════════════════════════════════════════════════════════════════════════════

class CrystalCompetitivoND:
    """
    Cristalização que funciona em qualquer dimensão.
    A lógica é a mesma: envelope → sigmoid → competição → HP → vida/morte.
    Apenas a exclusão espacial se adapta à dimensão.
    """

    def __init__(self, B, spatial_shape, sharpness=5.0, decay=0.02, ressonance_boost=0.1):
        self.ndim = len(spatial_shape)
        self.spatial_shape = spatial_shape

        full_shape = (B, *spatial_shape)
        self.crystal_map = torch.zeros(full_shape, device=DEVICE)
        self.crystal_hp  = torch.zeros(full_shape, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, *spatial_shape, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(full_shape, device=DEVICE)
        self.sharpness = sharpness
        self.decay = decay
        self.ressonance_boost = ressonance_boost

        # Kernel de dilatação para exclusão espacial
        ks = 2 * CRYSTAL_SEP + 1
        dilate_shape = [ks] * self.ndim
        self._dilate = torch.ones(1, 1, *dilate_shape, device=DEVICE)

    def update_envelope(self, field):
        self.window_max = torch.max(self.window_max, field.abs())
        self.window_step += 1
        if self.window_step >= CRYSTAL_W:
            self.env_buffer[:, self.window_idx] = self.window_max
            self.window_max = torch.zeros_like(self.window_max)
            self.window_step = 0
            self.window_idx = (self.window_idx + 1) % CRYSTAL_K

    def try_crystallize(self, field):
        env  = self.env_buffer
        mean = env.mean(dim=1)
        cv   = env.std(dim=1) / (mean + 1e-8)

        amp_score = torch.sigmoid(self.sharpness * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(self.sharpness * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(self.sharpness * (8.0 - mean))
        cand = amp_score * cv_score * sat_score

        # Exclusão espacial N-dimensional
        cm = self.crystal_map.unsqueeze(1).clamp(0, 1)
        pad = (CRYSTAL_SEP,) * (2 * self.ndim)
        cm_padded = F.pad(cm, pad, mode='circular')

        if self.ndim == 1:
            occ = F.conv1d(cm_padded, self._dilate).squeeze(1).clamp(0, 1)
        elif self.ndim == 2:
            occ = F.conv2d(cm_padded, self._dilate).squeeze(1).clamp(0, 1)
        elif self.ndim == 3:
            occ = F.conv3d(cm_padded, self._dilate).squeeze(1).clamp(0, 1)

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


# ══════════════════════════════════════════════════════════════════════════════
# PROJEÇÃO: ENTRADA → CAMPO (mesma dimensão)
# ══════════════════════════════════════════════════════════════════════════════

def build_gaussians_nd(input_shape, field_shape, sigma=0.04):
    """
    Projeção N-dimensional: cada elemento da entrada vira uma gaussiana
    no campo de mesma dimensionalidade.

    input_shape: tupla — forma da entrada (ex: (360,), (28,28), (28,28,10))
    field_shape: tupla — forma do campo (ex: (48,), (48,48), (48,48,48))

    Retorna: (n_elementos, prod(field_shape))
    """
    ndim = len(input_shape)
    assert len(field_shape) == ndim, f"Entrada {ndim}D mas campo {len(field_shape)}D"

    # Coordenadas do campo em cada dimensão
    coords = [torch.linspace(0., 1., fs, device=DEVICE) for fs in field_shape]
    grids = torch.meshgrid(*coords, indexing='ij')  # cada (fs0, fs1, ...)

    # Para cada elemento da entrada, cria uma gaussiana centrada na posição correspondente
    gs = []
    ranges = [range(s) for s in input_shape]

    # Iterar sobre todos os indices da entrada
    import itertools
    for idx in itertools.product(*ranges):
        # Centro da gaussiana: posição normalizada no campo
        centers = []
        for d in range(ndim):
            c = 0.1 + 0.8 * idx[d] / max(input_shape[d] - 1, 1)
            centers.append(c)

        # Distância ao quadrado em todas as dimensões
        dist_sq = torch.zeros(field_shape, device=DEVICE)
        for d in range(ndim):
            dist_sq = dist_sq + (grids[d] - centers[d])**2

        gauss = torch.exp(-dist_sq / (2 * sigma**2))
        gs.append(gauss.view(-1))

    n_elementos = 1
    for d in input_shape:
        n_elementos *= d
    return torch.stack(gs)  # (n_elementos, prod(field_shape))


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE: ENTRADA → CRYSTAL MAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_crystal_maps_nd(X, PG, field_shape, bs=64):
    """
    Pipeline completo N-dimensional.

    X: (N, *input_shape) — batch de entradas
    PG: (n_elementos, prod(field_shape)) — projeção
    field_shape: tupla — forma do campo

    Retorna: (N, *field_shape) — crystal maps
    """
    N = len(X)
    ndim = len(field_shape)
    n_elements = PG.shape[0]
    field_vol = PG.shape[1]
    lap_kernel = build_laplacian_kernel(ndim)
    out = []

    for i in range(0, N, bs):
        batch = X[i:i+bs]
        B = len(batch)

        # Projetar entrada no campo
        pert = (batch.view(B, n_elements) @ PG.to(batch.dtype)).view(B, *field_shape)
        f = pert.clone()
        v = torch.zeros_like(pert)
        mem = CrystalCompetitivoND(B, field_shape)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step_nd(f, v, pert, s < STIM_ON, lap_kernel, ndim)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)

        out.append(mem.crystal_map)

    return torch.cat(out, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class ResNetPsiND:
    """
    ResNet-Ψ N-Dimensional.

    O campo tem a mesma dimensão do dado.

    Uso:
        # Imagem 28x28 → campo 48x48
        rn = ResNetPsiND(input_shape=(28,28), field_shape=(48,48))

        # ECG 360 pontos → campo 1D de 512
        rn = ResNetPsiND(input_shape=(360,), field_shape=(512,))

        # Qualquer coisa
        rn.fit(X_train, y_train)
        preds = rn.predict(X_test)
    """

    def __init__(self, input_shape, field_shape=None, n_classes=10, sigma=0.04):
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.input_shape = input_shape
        self.ndim = len(input_shape)

        # Campo padrão: ~48 em cada dimensão
        if field_shape is None:
            field_shape = tuple([48] * self.ndim)
        self.field_shape = field_shape

        self.n_classes = n_classes
        print(f"ResNet-Ψ {self.ndim}D: entrada {input_shape} → campo {field_shape}")
        self.PG = build_gaussians_nd(input_shape, field_shape, sigma)
        self.prototypes = None

    def extract(self, X, bs=64):
        X = X.to(DEVICE)
        return compute_crystal_maps_nd(X, self.PG, self.field_shape, bs)

    def fit(self, X_train, y_train, bs=64):
        print(f"Extraindo crystal_maps de {len(X_train)} amostras...")
        t0 = time.time()
        cmaps = self.extract(X_train, bs)
        print(f"  Pronto: {time.time()-t0:.0f}s")

        if isinstance(y_train, torch.Tensor):
            labels = y_train.cpu().numpy()
        else:
            labels = np.array(y_train)

        self.prototypes = {}
        for cls in range(self.n_classes):
            mask = labels == cls
            if mask.sum() > 0:
                self.prototypes[cls] = cmaps[mask].mean(dim=0)
        return self

    def predict(self, X_test, bs=64):
        cmaps = self.extract(X_test, bs)
        N = len(cmaps)
        n_classes = len(self.prototypes)

        proto_stack = torch.stack([self.prototypes[c].flatten() for c in range(n_classes)])
        cmaps_flat = cmaps.view(N, -1)
        dists = torch.cdist(cmaps_flat.unsqueeze(0).float(), proto_stack.unsqueeze(0).float()).squeeze(0)
        return dists.argmin(dim=1).cpu().numpy()

    def score(self, X_test, y_test, bs=64):
        preds = self.predict(X_test, bs)
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.cpu().numpy()
        else:
            y_test = np.array(y_test)
        return (preds == y_test).mean() * 100
