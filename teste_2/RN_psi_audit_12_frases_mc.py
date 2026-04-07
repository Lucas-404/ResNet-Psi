"""
Auditoria 12: Completação de Frases — Multiple Choice

A ResNet-Ψ classifica entre poucas opções (5) ao invés de gerar texto.
Cada palavra é injetada sequencialmente no campo como perturbação.
Os cristais acumulados representam a frase inteira.
Classificação por distância euclidiana ao protótipo (zero treino).

Hipótese: se palavras similares criam padrões de onda similares,
frases com significado parecido terão crystal_maps parecidos.
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
STEPS_PER_WORD = 30     # steps de estímulo por palavra
STEPS_SILENCE  = 10     # steps de silêncio entre palavras

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

# ── Embedding dimension ────────────────────────────────────────────────────
EMBED_DIM = 16   # cada palavra vira vetor de 16 dims


# ── Cristalização Competitiva ───────────────────────────────────────────────

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
    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                          device=DEVICE).view(1, 1, 3, 3).to(field.dtype)
    if active:
        field = field + sources * (_DT * 0.1)
    lap = F.conv2d(F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular'), lap_k).squeeze(1)
    acc = _C2 * lap - _GAMMA * velocity + _ALPHA * torch.tanh(field) * field - _BETA * field * field**2
    velocity = torch.clamp(velocity + acc * _DT, -5., 5.)
    field    = torch.clamp(field + velocity * _DT, -10., 10.)
    return field, velocity


# ── Embedding: palavra → perturbação no campo ──────────────────────────────
#
# Embeddings manuais com estrutura semântica.
# 16 dimensões, cada uma representando um eixo de significado:
#   0: ser_vivo       (1 = vivo, -1 = objeto)
#   1: aquático       (1 = água, -1 = terra/ar)
#   2: aéreo          (1 = voa, -1 = não voa)
#   3: terrestre      (1 = terra, -1 = não)
#   4: veículo        (1 = transporte, -1 = não)
#   5: móvel/objeto   (1 = móvel/utensílio, -1 = não)
#   6: tem_pernas     (1 = sim, -1 = não)
#   7: tem_motor      (1 = sim, -1 = não)
#   8: doméstico      (1 = casa, -1 = selvagem)
#   9: tamanho        (1 = grande, -1 = pequeno)
#  10: movimento      (1 = se move, -1 = parado)
#  11: natural        (1 = natural, -1 = artificial)
#  12: som            (1 = faz som, -1 = silencioso)
#  13: comida_rel     (1 = relacionado a comida, -1 = não)
#  14: madeira_metal  (1 = madeira, -1 = metal)
#  15: ruído          variação aleatória por palavra

# Palavras com semântica codificada
#                          0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
WORD_EMBEDDINGS = {
    # ── Respostas (classes) ──
    "pássaro":  np.array([ 1.0, -0.5,  1.0, -0.5, -1.0, -1.0,  0.5, -1.0, -0.5, -0.5,  1.0,  1.0,  1.0, -0.5, -1.0,  0.1], dtype=np.float32),
    "peixe":    np.array([ 1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5, -0.3,  1.0,  1.0, -0.5, 0.8, -1.0, -0.2], dtype=np.float32),
    "gato":     np.array([ 1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -0.3,  0.5,  1.0,  1.0, -0.5, -1.0,  0.3], dtype=np.float32),
    "carro":    np.array([-1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  -1.0, 1.0, -0.5,  0.8,  1.0, -1.0,  1.0, -1.0, -1.0, -0.1], dtype=np.float32),
    "mesa":     np.array([-1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  0.3, -1.0, -0.5, -1.0,  1.0,  1.0,  0.2], dtype=np.float32),

    # ── Palavras de contexto: animais/voo ──
    "animal":   np.array([ 1.0,  0.0,  0.0,  0.5, -1.0, -1.0,  0.5, -1.0,  0.0,  0.0,  0.5,  1.0,  0.5, -0.3, -1.0,  0.0], dtype=np.float32),
    "bicho":    np.array([ 1.0,  0.0,  0.0,  0.5, -1.0, -1.0,  0.5, -1.0,  0.0, -0.2,  0.5,  1.0,  0.5, -0.3, -1.0,  0.15], dtype=np.float32),
    "ave":      np.array([ 1.0, -0.3,  0.9, -0.3, -1.0, -1.0,  0.5, -1.0, -0.5, -0.5,  1.0,  1.0,  1.0, -0.3, -1.0, -0.1], dtype=np.float32),
    "voa":      np.array([ 0.3, -0.5,  1.0, -1.0, -0.5, -1.0, -0.5, -0.5, -0.5,  0.0,  1.0,  0.5,  0.3, -1.0, -1.0,  0.05], dtype=np.float32),
    "asas":     np.array([ 0.5, -0.3,  1.0, -0.8, -1.0, -1.0, -1.0, -1.0, -0.5, -0.3,  0.8,  1.0,  0.0, -1.0, -1.0, -0.15], dtype=np.float32),
    "céu":      np.array([ 0.0, -0.5,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  0.0,  1.0, -0.5, -1.0, -1.0,  0.25], dtype=np.float32),
    "penas":    np.array([ 0.5, -0.3,  0.8, -0.3, -1.0, -1.0, -1.0, -1.0, -0.5, -0.5,  0.0,  1.0, -0.5, -1.0, -1.0, -0.3], dtype=np.float32),
    "ovo":      np.array([ 0.5,  0.0,  0.3,  0.0, -1.0, -1.0, -1.0, -1.0,  0.5, -0.8, -1.0,  1.0, -1.0,  1.0, -1.0,  0.1], dtype=np.float32),
    "bico":     np.array([ 0.5, -0.2,  0.7, -0.2, -1.0, -1.0, -1.0, -1.0, -0.5, -0.5,  0.0,  1.0,  0.3, 0.5, -1.0, -0.05], dtype=np.float32),
    "ninho":    np.array([ 0.3, -0.3,  0.8, -0.5, -1.0, -1.0, -1.0, -1.0, -0.5, -0.7, -1.0,  1.0, -0.5, -0.5, 1.0,  0.2], dtype=np.float32),
    "canta":    np.array([ 0.5, -0.3,  0.6,  0.0, -1.0, -1.0, -1.0, -1.0, -0.3, -0.5,  0.3,  1.0,  1.0, -0.5, -1.0, -0.25], dtype=np.float32),

    # ── Palavras de contexto: água ──
    "nada":     np.array([ 0.3,  1.0, -1.0, -0.8, -1.0, -1.0, -0.5, -1.0, -0.5,  0.0,  1.0,  0.5, -0.5, -0.5, -1.0,  0.1], dtype=np.float32),
    "água":     np.array([ 0.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5,  0.5,  0.5,  1.0, -0.3, -0.3, -1.0, -0.1], dtype=np.float32),
    "barbatanas":np.array([0.5,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.3,  1.0,  1.0, -0.5,  0.3, -1.0, 0.15], dtype=np.float32),
    "rio":      np.array([ 0.0,  1.0, -1.0, -0.5, -1.0, -1.0, -1.0, -1.0, -0.8,  0.5,  0.5,  1.0, -0.3, 0.3, -1.0, -0.2], dtype=np.float32),
    "guelras":  np.array([ 0.5,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.3,  0.0,  1.0, -0.5,  0.0, -1.0, 0.05], dtype=np.float32),
    "mar":      np.array([ 0.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  0.5,  1.0, -0.3,  0.5, -1.0, -0.15], dtype=np.float32),
    "escamas":  np.array([ 0.5,  0.8, -0.8, -0.5, -1.0, -1.0, -1.0, -1.0, -0.5, -0.3,  0.0,  1.0, -0.5,  0.3, -1.0,  0.1], dtype=np.float32),
    "oceano":   np.array([ 0.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  0.5,  1.0, -0.3,  0.5, -1.0, -0.3], dtype=np.float32),

    # ── Palavras de contexto: gato ──
    "mia":      np.array([ 0.5, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -0.5,  0.3,  1.0,  1.0, -0.5, -1.0,  0.2], dtype=np.float32),
    "estimação":np.array([ 0.5, -0.5, -0.5,  0.5, -1.0, -1.0,  0.5, -1.0,  1.0, -0.3,  0.3,  1.0,  0.3, -0.3, -1.0, -0.1], dtype=np.float32),
    "bigodes":  np.array([ 0.5, -0.5, -1.0,  0.8, -1.0, -1.0,  1.0, -1.0,  0.8, -0.5,  0.0,  1.0, -0.3, -0.5, -1.0,  0.15], dtype=np.float32),
    "rabo":     np.array([ 0.5, -0.3, -0.5,  0.5, -1.0, -1.0,  1.0, -1.0,  0.5, -0.3,  0.5,  1.0, -0.3, -0.5, -1.0, -0.05], dtype=np.float32),
    "ratos":    np.array([ 1.0, -0.5, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  -0.5, -0.8,  0.8,  1.0,  0.3,  0.0, -1.0,  0.1], dtype=np.float32),
    "caça":     np.array([ 0.3, -0.5, -0.5,  0.5, -1.0, -1.0,  0.5, -1.0, -0.5,  0.0,  1.0,  0.5,  0.3, 0.3, -1.0, -0.2], dtype=np.float32),
    "ronrona":  np.array([ 0.5, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -0.5, -0.3,  1.0,  0.8, -0.5, -1.0, 0.25], dtype=np.float32),
    "sofá":     np.array([-0.8, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  0.3, -1.0, -0.3, -1.0, -0.3,  0.8, -0.15], dtype=np.float32),
    "arranha":  np.array([ 0.5, -0.8, -1.0,  0.8, -1.0, -1.0,  1.0, -1.0,  0.5, -0.3,  0.8,  1.0,  0.5, -0.5, -1.0,  0.1], dtype=np.float32),
    "independente": np.array([0.3, -0.3, -0.5, 0.5, -1.0, -1.0, 1.0, -1.0, 0.3, -0.3, 0.5, 1.0, -0.3, -0.5, -1.0, -0.05], dtype=np.float32),
    "preguiçoso": np.array([0.5, -0.8, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -0.3, -0.5, 1.0, -0.3, -0.5, -1.0, 0.2], dtype=np.float32),

    # ── Palavras de contexto: carro ──
    "estrada":  np.array([-0.5, -1.0, -1.0,  1.0,  0.8, -0.5, -1.0,  0.5, -0.5,  0.8,  0.8, -0.5, -0.3, -1.0, -0.5, -0.1], dtype=np.float32),
    "rodas":    np.array([-1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -0.5,  0.3,  1.0, -1.0,  0.3, -1.0, -1.0,  0.15], dtype=np.float32),
    "gasolina": np.array([-1.0, -0.5, -1.0,  0.5,  1.0, -1.0, -1.0,  1.0, -0.5,  0.0,  0.5, -1.0, -0.3, -1.0, -1.0, -0.25], dtype=np.float32),
    "transporte":np.array([-0.5,-0.5, -0.5, 0.5,  1.0, -0.5, -0.5,  0.8, -0.3,  0.5,  1.0, -0.8,  0.3, -0.5, -1.0, 0.05], dtype=np.float32),
    "motor":    np.array([-1.0, -1.0, -1.0,  0.5,  1.0, -0.5, -1.0,  1.0, -0.5,  0.5,  0.8, -1.0,  1.0, -1.0, -1.0, -0.15], dtype=np.float32),
    "veículo":  np.array([-1.0, -0.8, -0.5,  0.8,  1.0, -0.5, -1.0,  1.0, -0.3,  0.5,  1.0, -1.0,  0.3, -0.5, -1.0, 0.1], dtype=np.float32),
    "passeio":  np.array([-0.3, -0.5, -0.5,  0.5,  0.8, -0.5, -0.5,  0.5,  0.3,  0.3,  1.0, -0.5,  0.0, -0.5, -1.0, -0.05], dtype=np.float32),
    "asfalto":  np.array([-0.8, -1.0, -1.0,  1.0,  0.8, -0.5, -1.0,  0.3, -0.5,  0.8,  0.0, -1.0, -0.5, -1.0, -1.0, 0.2], dtype=np.float32),
    "buzina":   np.array([-1.0, -1.0, -1.0,  0.5,  1.0, -1.0, -1.0,  1.0, -0.3,  0.3,  0.3, -1.0,  1.0, -1.0, -1.0, -0.1], dtype=np.float32),
    "trânsito": np.array([-0.5, -1.0, -1.0,  1.0,  1.0, -0.5, -1.0,  0.8, -0.5,  0.5,  0.5, -1.0,  0.8, -1.0, -1.0, 0.15], dtype=np.float32),
    "estaciona":np.array([-0.8, -1.0, -1.0,  1.0,  1.0, -0.5, -1.0,  0.5, -0.3,  0.5, -0.5, -1.0, -0.3, -1.0, -1.0, -0.2], dtype=np.float32),
    "garagem":  np.array([-0.8, -1.0, -1.0,  1.0,  0.8,  0.5, -1.0,  0.3,  1.0,  0.5, -0.8, -0.5, -0.5, -1.0,  0.3, 0.1], dtype=np.float32),

    # ── Palavras de contexto: mesa ──
    "cozinha":  np.array([-0.8, -0.5, -1.0,  1.0, -1.0,  1.0,  0.5, -1.0,  1.0,  0.5, -0.5, -0.3, -0.3,  1.0,  0.5, -0.1], dtype=np.float32),
    "pernas":   np.array([ 0.0, -0.5, -1.0,  0.8, -1.0,  0.5,  1.0, -1.0,  0.5,  0.0,  0.3,  0.0, -0.5, -0.3,  0.5, 0.15], dtype=np.float32),
    "prato":    np.array([-0.8, -0.3, -1.0,  0.5, -1.0,  1.0, -1.0, -1.0,  1.0, -0.5, -0.8, -0.5, -0.5,  1.0,  0.3, -0.2], dtype=np.float32),
    "cima":     np.array([ 0.0, -0.5,  0.3,  0.0, -0.5,  0.3, -0.5, -0.5,  0.0,  0.3,  0.0,  0.0, -0.5, -0.3,  0.0,  0.05], dtype=np.float32),
    "móvel":    np.array([-1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  0.5, -1.0,  1.0,  0.3, -1.0, -0.3, -1.0,  0.3,  0.8, -0.15], dtype=np.float32),
    "jantar":   np.array([-0.3, -0.5, -1.0,  0.5, -1.0,  0.5, -0.3, -1.0,  1.0,  0.3, -0.3, -0.3, -0.3,  1.0,  0.5,  0.1], dtype=np.float32),
    "madeira":  np.array([-1.0, -0.5, -1.0,  0.8, -1.0,  0.8,  0.3, -1.0,  0.5,  0.3, -1.0,  1.0, -0.5,  0.0,  1.0, -0.25], dtype=np.float32),
    "serve":    np.array([-0.3, -0.3, -0.5,  0.3, -0.5,  0.5, -0.3, -0.5,  0.5,  0.0,  0.0, -0.3, -0.5,  0.8,  0.0,  0.05], dtype=np.float32),
    "estuda":   np.array([-0.3, -0.5, -1.0,  0.5, -1.0,  0.5, -0.3, -1.0,  1.0, -0.3, -0.5, -0.3, -0.3,  0.0,  0.5, -0.1], dtype=np.float32),
    "apoiar":   np.array([-0.5, -0.5, -1.0,  0.5, -1.0,  0.5, -0.3, -1.0,  0.5,  0.0, -0.5, -0.3, -0.5,  0.0,  0.3,  0.15], dtype=np.float32),
    "braços":   np.array([ 0.3, -0.5, -1.0,  0.8, -1.0, -0.5,  0.5, -1.0,  0.3,  0.0,  0.5,  1.0, -0.3, -0.3, -0.5, -0.05], dtype=np.float32),

    # ── Palavras comuns / conectivos (semântica neutra) ──
    "que":      np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.1], dtype=np.float32),
    "com":      np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.1], dtype=np.float32),
    "de":       np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.05], dtype=np.float32),
    "e":        np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.05], dtype=np.float32),
    "no":       np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.15], dtype=np.float32),
    "na":       np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.15], dtype=np.float32),
    "em":       np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.08], dtype=np.float32),
    "por":      np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.08], dtype=np.float32),
    "tem":      np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.12], dtype=np.float32),
    "vive":     np.array([ 0.5,  0.0,  0.0,  0.0, -0.5, -0.5,  0.3, -0.5,  0.0,  0.0,  0.5,  0.8,  0.0, -0.3, -0.5, -0.1], dtype=np.float32),
    "mora":     np.array([ 0.3,  0.0,  0.0,  0.3, -0.5, -0.3,  0.3, -0.5,  0.3,  0.0,  0.0,  0.5, -0.3, -0.3, -0.3,  0.1], dtype=np.float32),
    "respira":  np.array([ 0.8,  0.3, -0.3,  0.0, -1.0, -1.0,  0.0, -1.0, -0.3,  0.0,  0.3,  1.0,  0.3, -0.3, -1.0, -0.05], dtype=np.float32),
    "bota":     np.array([ 0.3,  0.0,  0.0,  0.0, -0.5, -0.5,  0.0, -0.5,  0.0, -0.3,  0.3,  0.5, -0.5,  0.0, -0.5,  0.15], dtype=np.float32),
    "faz":      np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.3,  0.0,  0.0,  0.0,  0.0, -0.12], dtype=np.float32),
    "anda":     np.array([ 0.0, -0.3, -0.5,  0.8,  0.5, -0.5,  0.5,  0.3, -0.3,  0.3,  1.0, -0.3,  0.0, -0.5, -0.5,  0.08], dtype=np.float32),
    "precisa":  np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.1], dtype=np.float32),
    "fica":     np.array([-0.3, -0.3, -0.5,  0.5, -0.5,  0.5, -0.3, -0.5,  0.5,  0.0, -0.8, -0.3, -0.5,  0.3,  0.3, -0.08], dtype=np.float32),
    "coloca":   np.array([-0.3, -0.3, -0.5,  0.3, -0.5,  0.3, -0.3, -0.5,  0.3,  0.0,  0.3, -0.3, -0.5,  0.3,  0.0,  0.12], dtype=np.float32),
    "feita":    np.array([-0.5, -0.5, -0.5,  0.5, -0.5,  0.5, -0.3, -0.5,  0.3,  0.0, -0.5, -0.3, -0.5,  0.0,  0.5, -0.15], dtype=np.float32),
    "onde":     np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.18], dtype=np.float32),
    "se":       np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.18], dtype=np.float32),
    "os":       np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.06], dtype=np.float32),
    "pra":      np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.06], dtype=np.float32),
    "quatro":   np.array([ 0.0,  0.0, -0.5,  0.5,  0.3,  0.3,  0.5,  0.0,  0.3,  0.3,  0.0, -0.3, -0.5,  0.0,  0.0,  0.1], dtype=np.float32),
    "árvore":   np.array([-0.3, -0.3,  0.3,  0.5, -1.0, -0.5, -1.0, -1.0, -0.5,  0.8, -1.0,  1.0, -0.3, -0.3,  1.0, -0.1], dtype=np.float32),
    "cobra":    np.array([ 1.0, -0.3, -1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  0.5,  1.0,  0.3, -0.5, -1.0,  0.2], dtype=np.float32),
    "pedra":    np.array([-1.0, -0.3, -1.0,  1.0, -1.0, -0.5, -1.0, -1.0, -0.5,  0.3, -1.0,  1.0, -1.0, -1.0,  0.5, -0.2], dtype=np.float32),
    "casa":     np.array([-0.8, -0.5, -1.0,  1.0, -1.0,  1.0, -0.5, -1.0,  1.0,  0.8, -1.0, -0.3, -0.3,  0.3,  0.5,  0.1], dtype=np.float32),
    "livro":    np.array([-1.0, -1.0, -1.0,  0.5, -1.0,  0.5, -1.0, -1.0,  1.0, -0.3, -1.0, -0.3, -1.0, -0.5,  0.5, -0.15], dtype=np.float32),
    "comer":    np.array([ 0.3, -0.3, -0.5,  0.3, -0.5,  0.3,  0.0, -0.5,  0.5,  0.0,  0.3,  0.5, -0.3,  1.0,  0.0,  0.08], dtype=np.float32),
}


def word_to_embedding(word, dim=EMBED_DIM):
    """Retorna embedding semântico da palavra. Fallback: hash."""
    if word in WORD_EMBEDDINGS:
        return WORD_EMBEDDINGS[word]
    # Fallback: hash determinístico para palavras desconhecidas
    import hashlib
    h = hashlib.sha256(word.encode('utf-8')).digest()
    vals = np.array([b for b in h[:dim]], dtype=np.float32)
    vals = (vals / 128.0) - 1.0
    # Escala menor pra não dominar
    return vals * 0.3


def build_word_projection(dim=EMBED_DIM, field_size=FIELD_SIZE):
    """Matriz de projeção: vetor de palavra → campo 2D."""
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for d in range(dim):
        cx = 0.1 + 0.8 * (d % 4) / 3.0
        cy = 0.1 + 0.8 * (d // 4) / max((dim // 4) - 1, 1)
        sigma = 0.08
        gs.append(torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2)))
    return torch.stack(gs).view(dim, -1)   # (dim, field_size²)


# Vocabulário e projeção
WORD_PROJ = build_word_projection()

# Cache de embeddings
_embed_cache = {}

def word_to_field(word):
    """Palavra → perturbação 2D no campo."""
    if word not in _embed_cache:
        vec = torch.tensor(word_to_embedding(word), device=DEVICE)
        field = (vec @ WORD_PROJ).view(FIELD_SIZE, FIELD_SIZE)
        _embed_cache[word] = field
    return _embed_cache[word]


# ── Pipeline: frase → crystal_map ──────────────────────────────────────────

def sentence_to_crystal_map(words):
    """
    Processa uma frase palavra por palavra no campo.
    Cada palavra é injetada como perturbação, cristais se formam,
    próxima palavra interage com cristais anteriores.
    """
    field    = torch.zeros(1, FIELD_SIZE, FIELD_SIZE, device=DEVICE)
    velocity = torch.zeros_like(field)
    mem = CrystalCompetitivo(1, FIELD_SIZE)

    total_steps = 0
    for word in words:
        source = word_to_field(word).unsqueeze(0)  # (1, FS, FS)

        # Steps com estímulo da palavra
        for s in range(STEPS_PER_WORD):
            field, velocity = psi_step(field, velocity, source, active=True)
            mem.update_envelope(field)
            if mem.window_idx > 0:
                mem.try_crystallize(field)
            field = mem.remit(field)
            total_steps += 1

        # Steps de silêncio (campo evolui sozinho)
        for s in range(STEPS_SILENCE):
            field, velocity = psi_step(field, velocity, source, active=False)
            mem.update_envelope(field)
            if mem.window_idx > 0:
                mem.try_crystallize(field)
            field = mem.remit(field)
            total_steps += 1

    return mem.crystal_map.squeeze(0)   # (FS, FS)


def batch_sentence_crystal_maps(sentences):
    """Processa lista de frases (lista de lista de palavras)."""
    out = []
    for words in sentences:
        cmap = sentence_to_crystal_map(words)
        out.append(cmap)
    return torch.stack(out)   # (N, FS, FS)


# ── Dataset: frases com multiple choice ────────────────────────────────────

# Categorias semânticas claras
DATASET = [
    # Formato: (frase_contexto, resposta_correta, [distratores])
    # ── Animais que voam ──
    (["animal", "que", "voa"],           "pássaro",  ["peixe", "cobra", "pedra", "mesa"]),
    (["bicho", "com", "asas"],           "pássaro",  ["gato", "peixe", "carro", "casa"]),
    (["vive", "no", "céu"],              "pássaro",  ["peixe", "cobra", "mesa", "livro"]),
    (["tem", "penas", "e", "voa"],       "pássaro",  ["gato", "peixe", "pedra", "carro"]),
    (["bota", "ovo", "e", "voa"],        "pássaro",  ["cobra", "peixe", "mesa", "carro"]),

    # ── Animais que nadam ──
    (["animal", "que", "nada"],          "peixe",    ["pássaro", "gato", "mesa", "pedra"]),
    (["vive", "na", "água"],             "peixe",    ["pássaro", "cobra", "carro", "livro"]),
    (["bicho", "com", "barbatanas"],     "peixe",    ["gato", "pássaro", "mesa", "pedra"]),
    (["mora", "no", "rio"],              "peixe",    ["pássaro", "gato", "carro", "casa"]),
    (["respira", "por", "guelras"],      "peixe",    ["gato", "cobra", "pedra", "mesa"]),

    # ── Animais terrestres ──
    (["animal", "que", "mia"],           "gato",     ["peixe", "pássaro", "mesa", "carro"]),
    (["bicho", "de", "estimação"],       "gato",     ["peixe", "cobra", "pedra", "livro"]),
    (["tem", "bigodes", "e", "rabo"],    "gato",     ["peixe", "pássaro", "mesa", "carro"]),
    (["caça", "ratos"],                  "gato",     ["peixe", "cobra", "pedra", "livro"]),
    (["ronrona", "no", "sofá"],          "gato",     ["peixe", "pássaro", "mesa", "carro"]),

    # ── Veículos ──
    (["anda", "na", "estrada"],          "carro",    ["pássaro", "peixe", "gato", "livro"]),
    (["tem", "quatro", "rodas"],         "carro",    ["peixe", "gato", "mesa", "pedra"]),
    (["precisa", "de", "gasolina"],      "carro",    ["pássaro", "gato", "pedra", "livro"]),
    (["transporte", "com", "motor"],     "carro",    ["peixe", "pássaro", "gato", "mesa"]),
    (["veículo", "de", "passeio"],       "carro",    ["peixe", "cobra", "pedra", "livro"]),

    # ── Móveis ──
    (["fica", "na", "cozinha"],          "mesa",     ["pássaro", "peixe", "cobra", "carro"]),
    (["tem", "quatro", "pernas"],        "mesa",     ["pássaro", "peixe", "cobra", "carro"]),
    (["coloca", "prato", "em", "cima"],  "mesa",     ["pássaro", "gato", "carro", "livro"]),
    (["móvel", "de", "jantar"],          "mesa",     ["pássaro", "peixe", "cobra", "carro"]),
    (["feita", "de", "madeira"],         "mesa",     ["pássaro", "peixe", "cobra", "carro"]),
]

# Gera variações para ter mais dados
EXTRA_DATASET = [
    # ── Mais animais que voam (variações) ──
    (["ave", "que", "canta"],            "pássaro",  ["peixe", "gato", "carro", "mesa"]),
    (["tem", "bico", "e", "asas"],       "pássaro",  ["peixe", "cobra", "pedra", "carro"]),
    (["faz", "ninho", "na", "árvore"],   "pássaro",  ["peixe", "gato", "mesa", "carro"]),

    # ── Mais animais que nadam ──
    (["nada", "no", "mar"],              "peixe",    ["pássaro", "gato", "carro", "mesa"]),
    (["tem", "escamas"],                 "peixe",    ["gato", "pássaro", "mesa", "carro"]),
    (["vive", "no", "oceano"],           "peixe",    ["gato", "cobra", "pedra", "mesa"]),

    # ── Mais gatos ──
    (["bicho", "que", "arranha"],        "gato",     ["peixe", "pássaro", "carro", "mesa"]),
    (["animal", "independente"],         "gato",     ["peixe", "pássaro", "carro", "mesa"]),
    (["bicho", "preguiçoso"],            "gato",     ["peixe", "cobra", "carro", "mesa"]),

    # ── Mais carros ──
    (["anda", "no", "asfalto"],          "carro",    ["pássaro", "peixe", "gato", "mesa"]),
    (["buzina", "no", "trânsito"],       "carro",    ["pássaro", "peixe", "gato", "mesa"]),
    (["estaciona", "na", "garagem"],     "carro",    ["peixe", "cobra", "pedra", "mesa"]),

    # ── Mais mesas ──
    (["serve", "pra", "comer"],          "mesa",     ["pássaro", "peixe", "cobra", "carro"]),
    (["onde", "se", "estuda"],           "mesa",     ["pássaro", "peixe", "gato", "carro"]),
    (["apoiar", "os", "braços"],         "mesa",     ["pássaro", "peixe", "cobra", "carro"]),
]

ALL_DATA = DATASET + EXTRA_DATASET


# ── Experimento ──────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("AUDITORIA 12: Completação de Frases — Multiple Choice")
print(f"Dataset: {len(ALL_DATA)} frases | 5 classes | Zero treino")
print(f"{'='*65}")

# 5 classes
CLASSES = ["pássaro", "peixe", "gato", "carro", "mesa"]
cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

print(f"\nClasses: {CLASSES}")
print(f"Steps por palavra: {STEPS_PER_WORD} estímulo + {STEPS_SILENCE} silêncio")

# ── Teste 1: Cada resposta correta gera um crystal_map diferente? ─────────

print(f"\n── Teste 1: Assinaturas das classes ──")
print("Gerando crystal_maps das palavras-classe...")

class_cmaps = {}
for cls in CLASSES:
    cmap = sentence_to_crystal_map([cls])
    class_cmaps[cls] = cmap
    n_crys = (cmap > 0.01).float().sum().item()
    print(f"  '{cls}': {n_crys:.0f} cristais")

# Similaridade entre classes
print("\nSimilaridade coseno entre crystal_maps das classes:")
print(f"{'':>10}", end='')
for c in CLASSES:
    print(f"  {c:>8}", end='')
print()
for ci in CLASSES:
    print(f"{ci:>10}", end='')
    vi = class_cmaps[ci].flatten()
    vi_n = vi / (vi.norm() + 1e-8)
    for cj in CLASSES:
        vj = class_cmaps[cj].flatten()
        vj_n = vj / (vj.norm() + 1e-8)
        sim = (vi_n * vj_n).sum().item()
        print(f"  {sim:>8.3f}", end='')
    print()


# ── Teste 2: Frases da mesma classe geram crystal_maps similares? ─────────

print(f"\n── Teste 2: Consistência intra-classe ──")
print("Gerando crystal_maps de todas as frases...")

t0 = time.time()
all_cmaps = []
all_labels = []
all_sentences = []

for ctx, answer, distractors in ALL_DATA:
    cmap = sentence_to_crystal_map(ctx)
    all_cmaps.append(cmap)
    all_labels.append(cls_to_idx[answer])
    all_sentences.append(' '.join(ctx))

all_cmaps = torch.stack(all_cmaps)
all_labels = np.array(all_labels)
print(f"  {len(all_cmaps)} frases processadas em {time.time()-t0:.0f}s")

# Similaridade média intra vs inter classe
print("\nSimilaridade média intra-classe vs inter-classe:")
for cls_idx, cls_name in enumerate(CLASSES):
    mask = all_labels == cls_idx
    n = mask.sum()
    if n < 2:
        continue
    cls_cmaps = all_cmaps[mask]
    # Intra: média de similaridade entre pares da mesma classe
    intra_sims = []
    for i in range(n):
        for j in range(i+1, n):
            vi = cls_cmaps[i].flatten()
            vj = cls_cmaps[j].flatten()
            vi_n = vi / (vi.norm() + 1e-8)
            vj_n = vj / (vj.norm() + 1e-8)
            intra_sims.append((vi_n * vj_n).sum().item())
    # Inter: média com todas as outras classes
    inter_sims = []
    other_cmaps = all_cmaps[~mask]
    for i in range(min(n, 5)):
        for j in range(min(len(other_cmaps), 10)):
            vi = cls_cmaps[i].flatten()
            vj = other_cmaps[j].flatten()
            vi_n = vi / (vi.norm() + 1e-8)
            vj_n = vj / (vj.norm() + 1e-8)
            inter_sims.append((vi_n * vj_n).sum().item())

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    print(f"  {cls_name:>10}: intra={intra_mean:.3f}  inter={inter_mean:.3f}  gap={intra_mean-inter_mean:+.3f}")


# ── Teste 3: Classificação por protótipos ─────────────────────────────────

print(f"\n── Teste 3: Classificação por protótipos ──")

# Leave-one-out: para cada frase, o protótipo é feito com as OUTRAS
correct = 0
results_detail = []

for i in range(len(all_cmaps)):
    test_cmap = all_cmaps[i]
    test_label = all_labels[i]

    # Protótipos sem a amostra i
    prototypes = {}
    for cls_idx in range(len(CLASSES)):
        mask = (all_labels == cls_idx)
        mask[i] = False
        if mask.sum() == 0:
            prototypes[cls_idx] = torch.zeros(FIELD_SIZE, FIELD_SIZE, device=DEVICE)
        else:
            prototypes[cls_idx] = all_cmaps[mask].mean(dim=0)

    # Distância euclidiana
    best_cls, best_dist = -1, float('inf')
    for cls_idx in range(len(CLASSES)):
        dist = ((test_cmap - prototypes[cls_idx])**2).sum().item()
        if dist < best_dist:
            best_dist = dist
            best_cls = cls_idx

    is_correct = (best_cls == test_label)
    if is_correct:
        correct += 1
    results_detail.append({
        'sentence': all_sentences[i],
        'true': CLASSES[test_label],
        'pred': CLASSES[best_cls],
        'correct': is_correct,
    })

acc = correct / len(all_cmaps) * 100


# ── Teste 4: Multiple choice (como o usuário usaria) ─────────────────────

print(f"\n── Teste 4: Multiple Choice ──")

correct_mc = 0
for ctx, answer, distractors in ALL_DATA:
    # Crystal map da frase-contexto
    ctx_cmap = sentence_to_crystal_map(ctx)

    # Crystal map de cada opção (contexto + opção)
    options = [answer] + distractors
    best_opt, best_dist = None, float('inf')
    for opt in options:
        opt_cmap = sentence_to_crystal_map(ctx + [opt])
        # Compara com protótipo da classe da opção
        # Mais simples: qual opção gera crystal_map mais distinto?
        # Ou: qual frase completa (ctx + opt) é mais "estável"?
        energy = opt_cmap.abs().sum().item()
        # A opção correta deveria gerar mais cristais (mais ressonância)
        # porque as palavras do contexto "pedem" aquela resposta
        if energy < best_dist:  # ou > dependendo da hipótese
            best_dist = energy
            best_opt = opt

    if best_opt == answer:
        correct_mc += 1

# Tenta também com max energy
correct_mc_max = 0
for ctx, answer, distractors in ALL_DATA:
    ctx_cmap = sentence_to_crystal_map(ctx)
    options = [answer] + distractors
    best_opt, best_energy = None, -1
    for opt in options:
        opt_cmap = sentence_to_crystal_map(ctx + [opt])
        energy = opt_cmap.abs().sum().item()
        if energy > best_energy:
            best_energy = energy
            best_opt = opt
    if best_opt == answer:
        correct_mc_max += 1

acc_mc_min = correct_mc / len(ALL_DATA) * 100
acc_mc_max = correct_mc_max / len(ALL_DATA) * 100


# ── Resumo ────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("RESULTADO FINAL")
print(f"{'='*65}")

print(f"\n  Protótipos (leave-one-out):  {acc:.1f}%  ({correct}/{len(all_cmaps)})")
print(f"  Multiple choice (min energy): {acc_mc_min:.1f}%  ({correct_mc}/{len(ALL_DATA)})")
print(f"  Multiple choice (max energy): {acc_mc_max:.1f}%  ({correct_mc_max}/{len(ALL_DATA)})")
print(f"  Referência: aleatório = 20.0% (5 classes)")

# Detalhes por classe
print(f"\n  Por classe (protótipos):")
for cls_idx, cls_name in enumerate(CLASSES):
    mask = all_labels == cls_idx
    n_total = mask.sum()
    n_correct = sum(1 for r in results_detail if r['true'] == cls_name and r['correct'])
    print(f"    {cls_name:>10}: {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)")

# Erros
print(f"\n  Erros:")
for r in results_detail:
    if not r['correct']:
        print(f"    '{r['sentence']}' → pred={r['pred']} (real={r['true']})")

print(f"\n  Tempo total: {time.time()-t0:.0f}s")


# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Auditoria 12 — Crystal Maps por Classe (Média)', fontsize=13)

# Protótipos médios por classe
for cls_idx, cls_name in enumerate(CLASSES):
    ax = axes[0, cls_idx]
    mask = all_labels == cls_idx
    proto = all_cmaps[mask].mean(dim=0).cpu().numpy()
    vmax = np.abs(proto).max() if np.abs(proto).max() > 0 else 1
    ax.imshow(proto, cmap='hot', vmin=0, vmax=vmax, aspect='equal')
    n_crys = (all_cmaps[mask].mean(dim=0) > 0.01).float().sum().item()
    ax.set_title(f'{cls_name}\n({n_crys:.0f} cristais)', fontsize=9)
    ax.axis('off')

# Exemplos individuais (1 por classe)
for cls_idx, cls_name in enumerate(CLASSES):
    ax = axes[1, cls_idx]
    mask = all_labels == cls_idx
    idx = np.where(mask)[0][0]
    cmap = all_cmaps[idx].cpu().numpy()
    vmax = np.abs(cmap).max() if np.abs(cmap).max() > 0 else 1
    ax.imshow(cmap, cmap='hot', vmin=0, vmax=vmax, aspect='equal')
    ax.set_title(f"'{all_sentences[idx]}'", fontsize=7)
    ax.axis('off')

axes[0, 0].set_ylabel('Protótipo\n(média)', fontsize=9)
axes[1, 0].set_ylabel('Exemplo\nindividual', fontsize=9)

plt.tight_layout()
plt.savefig('viz_audit_12_frases_mc.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_audit_12_frases_mc.png")
print("Pronto.")
