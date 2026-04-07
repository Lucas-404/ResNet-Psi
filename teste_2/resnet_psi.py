"""
ResNet-Ψ — Nível 3 — XOR Temporal v2 + Ablações + Paridade 3 e 4 Bits — CUDA BATCHED

TESTES:
  Original:  XOR Temporal 2-bit com/sem campo (referência)
  Teste 1:   Ablação de Cristais — desativa cristalização
  Teste 2:   Estresse de Silêncio — 300 steps de silêncio
  Teste 3:   Paridade 3 Bits — 8 combinações, XOR(S1,S2,S3)
  Teste 4:   Paridade 4 Bits — 16 combinações, XOR(S1,S2,S3,S4)

Janelas (padrão):
  t=0-60:    S1 emite (60 steps, 4 janelas de envelope de 15)
  t=61-100:  Silêncio (40 steps)
  t=101-130: S2 emite (30 steps)
  t=131-200: Ressonância (70 steps)
  t=200:     Leitura

Paridade 4-bit (espaçada):
  t=0-40:    S1 | t=80-120: S2 | t=160-200: S3 | t=240-280: S4 | t=350: Leitura
  40 steps de silêncio entre cada bit para cristalização
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import cma
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List

import warnings
warnings.filterwarnings('ignore')

# Dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" usando dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f" CUDA version: {torch.version.cuda}")
else:
    print(" CUDA não disponível — rodando em CPU")

# ============================================
# CONFIGURAÇÃO TEMPORAL
# ============================================

@dataclass
class TimeConfig:
    s1_start:      int = 0
    s1_end:        int = 60
    silence_start: int = 61
    silence_end:   int = 100
    s2_start:      int = 101
    s2_end:        int = 130
    s3_start:      Optional[int] = None
    s3_end:        Optional[int] = None
    s4_start:      Optional[int] = None
    s4_end:        Optional[int] = None
    read:          int = 200

    @property
    def is_3bit(self):
        return self.s3_start is not None and self.s4_start is None

    @property
    def is_4bit(self):
        return self.s4_start is not None

    @property
    def monitor_steps(self) -> List[int]:
        steps = [self.s1_end, self.silence_end, self.s2_end]
        if self.s3_end is not None:
            steps.append(self.s3_end)
        if self.s4_end is not None:
            steps.append(self.s4_end)
        steps.append(self.read)
        return steps

DEFAULT_TIME_CONFIG = TimeConfig()

SILENCE_STRESS_CONFIG = TimeConfig(
    s1_start=0,
    s1_end=60,
    silence_start=61,
    silence_end=360,   # 300 steps de silêncio
    s2_start=361,
    s2_end=390,
    read=460,
)

PARITY_3BIT_TIME = TimeConfig(
    s1_start=0,
    s1_end=60,
    silence_start=61,
    silence_end=100,
    s2_start=101,
    s2_end=130,
    s3_start=131,   # S3 imediatamente após S2
    s3_end=160,
    read=230,       # leitura após interação de S3
)

# Paridade 4-bit: 40 steps de silêncio entre cada bit para cristalização
PARITY_4BIT_TIME = TimeConfig(
    s1_start=0,    s1_end=40,
    silence_start=41, silence_end=79,
    s2_start=80,   s2_end=120,
    s3_start=160,  s3_end=200,
    s4_start=240,  s4_end=280,
    read=350,
)

# ============================================
# DATASETS
# ============================================

TEMPORAL_XOR_DATA = torch.tensor([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
], dtype=torch.float32, device=DEVICE)

TEMPORAL_XOR = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]

PARITY_3BIT_DATA = torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
], dtype=torch.float32, device=DEVICE)

PARITY_3BIT = [(int(r[0]),int(r[1]),int(r[2]),int(r[3])) for r in PARITY_3BIT_DATA]

# Paridade 4-bit: XOR(S1,S2,S3,S4) — 16 combinações
PARITY_4BIT_DATA = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
], dtype=torch.float32, device=DEVICE)

PARITY_4BIT = [(int(r[0]),int(r[1]),int(r[2]),int(r[3]),int(r[4])) for r in PARITY_4BIT_DATA]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, float(x)))))

def bce_loss(pred, target):
    eps = 1e-7
    p = max(eps, min(1 - eps, pred))
    return -(target * math.log(p) + (1 - target) * math.log(1 - p))

def bce_loss_batched(preds, targets):
    """BCE loss em batch na GPU."""
    eps = 1e-7
    p = torch.clamp(preds, eps, 1 - eps)
    return -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p)).sum()


# ============================================
# PSI-FIELD (Batched + CUDA)
# ============================================

class PsiField:
    def __init__(self, batch_size=4, size=48, damping=0.03,
                 dissipation=0.005, disable_crystals=False):
        self.batch_size = batch_size
        self.size = size
        self.damping = damping
        self.dissipation = dissipation
        self.disable_crystals = disable_crystals

        self.field = torch.zeros(batch_size, size, size, device=DEVICE)
        self.velocity = torch.zeros(batch_size, size, size, device=DEVICE)

        coords = torch.linspace(0, 1, size, device=DEVICE)
        self.x_grid, self.y_grid = torch.meshgrid(coords, coords, indexing='ij')

        # Ondas: (B, W, 6) = [amp, freq, phase, decay, pos_x, pos_y]
        self.wave_params = torch.zeros(batch_size, 2, 6, device=DEVICE)
        self.wave_mask = torch.zeros(batch_size, 2, 1, 1, device=DEVICE)

        self.step_count = 0

        # Cristais
        self.max_crystals = 0 if disable_crystals else 100
        self.crystal_radius = 5
        self.cpat_size = 2 * self.crystal_radius + 1  # 11
        self.crystal_positions = torch.zeros(batch_size, self.max_crystals, 2, dtype=torch.long, device=DEVICE)
        self.crystal_patterns = torch.zeros(batch_size, self.max_crystals, self.cpat_size, self.cpat_size, device=DEVICE)
        self.crystal_active = torch.zeros(batch_size, self.max_crystals, device=DEVICE)
        self.crystal_count = [0] * batch_size

        # Offsets pré-computados para gather/scatter
        ps = self.cpat_size
        self._off_x = torch.arange(ps, device=DEVICE).unsqueeze(1).expand(ps, ps).reshape(-1)
        self._off_y = torch.arange(ps, device=DEVICE).unsqueeze(0).expand(ps, ps).reshape(-1)
        decay_coords = torch.arange(ps, dtype=torch.float32, device=DEVICE) - self.crystal_radius
        dx, dy = torch.meshgrid(decay_coords, decay_coords, indexing='ij')
        self.crystal_reemit_decay = 1.0 / (1.0 + torch.sqrt(dx**2 + dy**2))

        # Envelope tracking
        self.envelope_window = 15
        self.envelope_num_windows = 4
        self.current_window_max = torch.zeros(batch_size, size, size, device=DEVICE)
        self.envelope_history = torch.zeros(
            batch_size, self.envelope_num_windows, size, size, device=DEVICE
        )
        self.window_step_count = 0
        self.completed_windows = 0

        # Laplaciano — kernel uma vez
        self.laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)

    def emit_all_waves(self, t):
        """Emissão vetorizada — todas as ondas de todos os batches de uma vez."""
        amp   = self.wave_params[:, :, 0]
        freq  = self.wave_params[:, :, 1]
        phase = self.wave_params[:, :, 2]
        decay = self.wave_params[:, :, 3]
        pos_x = self.wave_params[:, :, 4]
        pos_y = self.wave_params[:, :, 5]

        xg = self.x_grid.unsqueeze(0).unsqueeze(0)
        yg = self.y_grid.unsqueeze(0).unsqueeze(0)

        px = pos_x.unsqueeze(-1).unsqueeze(-1)
        py = pos_y.unsqueeze(-1).unsqueeze(-1)

        dx = xg - px
        dy = yg - py
        distance = torch.sqrt(dx**2 + dy**2 + 1e-8)

        f = freq.unsqueeze(-1).unsqueeze(-1)
        p = phase.unsqueeze(-1).unsqueeze(-1)
        d = decay.unsqueeze(-1).unsqueeze(-1)
        a = amp.unsqueeze(-1).unsqueeze(-1)

        oscillation = torch.sin(f * t + p - f * distance)
        temporal_decay = torch.exp(-d * t)
        spatial_decay = 1.0 / (1.0 + distance)

        per_wave = a * oscillation * temporal_decay * spatial_decay
        per_wave = per_wave * self.wave_mask

        return per_wave.sum(dim=1)  # (B, S, S)

    def step(self):
        dt = 0.05
        t = self.step_count * dt

        # Emissão batched
        emission = self.emit_all_waves(t)
        self.field = self.field + emission * dt * 0.1

        # Laplaciano via Conv2D
        input_tensor = self.field.unsqueeze(1)
        padded = F.pad(input_tensor, (1, 1, 1, 1), mode='circular')
        laplacian = F.conv2d(padded, self.laplacian_kernel).squeeze(1)

        # Não-linearidade e Física
        nonlinear = 0.04 * torch.tanh(self.field) * self.field

        # Escudo de Cristal: onde há cristal, dissipação cai 70%
        crystal_mask = torch.zeros_like(self.field)
        if not self.disable_crystals:
            for b, cc in enumerate(self.crystal_count):
                if cc > 0:
                    px = self.crystal_positions[b, :cc, 0]
                    py = self.crystal_positions[b, :cc, 1]
                    crystal_mask[b, px, py] = 1.0

        effective_dissipation = self.dissipation * (1.0 - 0.7 * crystal_mask)

        acceleration = (
            0.3 * laplacian
            - self.damping * self.velocity
            + nonlinear
            - effective_dissipation * self.field * (self.field ** 2)
        )

        # Re-emissão vetorizada de cristais
        if not self.disable_crystals and self.crystal_active.any():
            r = self.crystal_radius
            pw = self.size + 2 * r  # 58

            padded_c = F.pad(self.field.unsqueeze(1), (r, r, r, r), mode='constant', value=0).squeeze(1)
            padded_flat = padded_c.reshape(self.batch_size, -1)

            cx = self.crystal_positions[:, :, 0]
            cy = self.crystal_positions[:, :, 1]
            lin_idx = (
                (cx.unsqueeze(-1) + self._off_x.unsqueeze(0).unsqueeze(0)) * pw +
                (cy.unsqueeze(-1) + self._off_y.unsqueeze(0).unsqueeze(0))
            )

            regions = torch.gather(
                padded_flat.unsqueeze(1).expand(-1, self.max_crystals, -1), 2, lin_idx
            )

            pat_flat = self.crystal_patterns.reshape(self.batch_size, self.max_crystals, -1)
            p_norm = pat_flat / (torch.norm(pat_flat, dim=-1, keepdim=True) + 1e-8)
            r_norm = regions / (torch.norm(regions, dim=-1, keepdim=True) + 1e-8)
            corr = (p_norm * r_norm).sum(dim=-1)

            mask = self.crystal_active * (corr.abs() > 0.1).float()
            scale = corr * 0.35 * mask

            contrib = (
                self.crystal_patterns
                * self.crystal_reemit_decay.unsqueeze(0).unsqueeze(0)
                * scale.unsqueeze(-1).unsqueeze(-1)
            )
            contrib_flat = contrib.reshape(self.batch_size, -1)
            idx_flat = lin_idx.reshape(self.batch_size, -1)

            padded_accum = torch.zeros_like(padded_flat)
            padded_accum.scatter_add_(1, idx_flat, contrib_flat)
            accum = padded_accum.reshape(self.batch_size, pw, pw)
            self.field = self.field + accum[:, r:r+self.size, r:r+self.size]

        self.velocity = self.velocity + acceleration * dt
        self.field = self.field + self.velocity * dt

        self.field = torch.clamp(self.field, -10.0, 10.0)
        self.velocity = torch.clamp(self.velocity, -5.0, 5.0)

        # Envelope tracking
        self.current_window_max = torch.max(self.current_window_max, torch.abs(self.field))
        self.window_step_count += 1

        if self.window_step_count >= self.envelope_window:
            idx = self.completed_windows % self.envelope_num_windows
            self.envelope_history[:, idx] = self.current_window_max.clone()
            self.completed_windows += 1
            self.current_window_max = torch.zeros_like(self.current_window_max)
            self.window_step_count = 0

            if not self.disable_crystals and self.completed_windows >= self.envelope_num_windows:
                self.try_crystallize()

        self.step_count += 1

    def try_crystallize(self):
        """Cristalização — roda raramente (~3x por avaliação)."""
        env_mean = torch.mean(self.envelope_history, dim=1)
        env_std = torch.std(self.envelope_history, dim=1)
        cv = env_std / (env_mean + 1e-8)

        strong = env_mean > 0.15
        stable = cv < 0.18
        not_saturated = env_mean < 9.0

        candidates_mask = strong & stable & not_saturated

        r = self.crystal_radius
        ps = self.cpat_size

        for b in range(self.batch_size):
            if not candidates_mask[b].any():
                continue

            positions = torch.nonzero(candidates_mask[b])
            positions_cpu = positions.cpu().numpy()

            cc = self.crystal_count[b]
            if cc > 0:
                existing = self.crystal_positions[b, :cc].cpu().numpy().astype(np.float64)
            else:
                existing = np.empty((0, 2), dtype=np.float64)

            formed = 0
            for i in range(len(positions_cpu)):
                if formed >= 10 or cc >= self.max_crystals:
                    break

                px, py = int(positions_cpu[i, 0]), int(positions_cpu[i, 1])

                if cc > 0:
                    dists = np.sqrt(((existing - np.array([px, py], dtype=np.float64))**2).sum(axis=1))
                    if dists.min() < r:
                        continue

                x_s, x_e = max(0, px - r), min(self.size, px + r + 1)
                y_s, y_e = max(0, py - r), min(self.size, py + r + 1)
                h, w = x_e - x_s, y_e - y_s
                if h < 3 or w < 3:
                    continue

                pattern_full = torch.zeros(ps, ps, device=DEVICE)
                ox = x_s - (px - r)
                oy = y_s - (py - r)
                pattern_full[ox:ox+h, oy:oy+w] = self.field[b, x_s:x_e, y_s:y_e]

                self.crystal_patterns[b, cc] = pattern_full
                self.crystal_positions[b, cc, 0] = px
                self.crystal_positions[b, cc, 1] = py
                self.crystal_active[b, cc] = 1.0

                cc += 1
                formed += 1

                if existing.size > 0:
                    existing = np.vstack([existing, [px, py]])
                else:
                    existing = np.array([[px, py]], dtype=np.float64)

            self.crystal_count[b] = cc

    def read_at(self, positions):
        """Leitura vetorizada — retorna (B, P)."""
        pos_t = torch.tensor(positions, dtype=torch.float32)
        ix = (pos_t[:, 0] * (self.size - 1)).long()
        iy = (pos_t[:, 1] * (self.size - 1)).long()
        ix = torch.clamp(ix, 0, self.size - 1)
        iy = torch.clamp(iy, 0, self.size - 1)
        return self.field[:, ix, iy]  # (B, P)


# ============================================
# EMITTERS & DECODER
# ============================================

class TemporalEmitter:
    """Emitter para até 4 estímulos (S1, S2, S3, S4).

    Shape (4, 2, 2, 2): [n_stimuli=4, n_bits=2, n_waves=2, n_wave_params=2]
    Posições: S1→x=0.2, S2→x=0.4, S3→x=0.6, S4→x=0.8
    """
    def __init__(self):
        self.params = np.zeros((4, 2, 2, 2))

    def get_waves_for_batch(self, stimulus_idx, bit_values):
        B = len(bit_values)
        vals = bit_values.long().cpu().numpy()

        # Posições espaciais bem separadas para 4 estímulos
        positions = [
            (0.2, 0.4, 0.6),   # S1 → quarto esquerdo
            (0.4, 0.4, 0.6),   # S2 → centro-esquerdo
            (0.6, 0.4, 0.6),   # S3 → centro-direito
            (0.8, 0.4, 0.6),   # S4 → quarto direito
        ]
        pos_x, pos_y_0, pos_y_1 = positions[stimulus_idx]

        raw = self.params[stimulus_idx]
        selected = raw[vals]  # (B, 2, 2)

        # Amplitude mínima alta: todo bit injeta onda real (mín 2.0)
        amp = np.abs(selected[:, :, 0]) * 1.5 + 2.0

        # Frequência separada por banda:
        #   bit 0 → banda baixa  [2.0, ~6.0]
        #   bit 1 → banda alta   [7.0, ~11.0]
        bit_offsets = np.where(vals[:, np.newaxis] == 0, 2.0, 7.0)  # (B, W)
        freq = np.abs(selected[:, :, 1]) * 2.0 + bit_offsets

        wave_params = torch.zeros(B, 2, 6, device=DEVICE)
        wave_params[:, :, 0] = torch.tensor(amp, dtype=torch.float32, device=DEVICE)
        wave_params[:, :, 1] = torch.tensor(freq, dtype=torch.float32, device=DEVICE)
        wave_params[:, :, 2] = 0.0
        wave_params[:, :, 3] = 0.001
        wave_params[:, :, 4] = pos_x
        wave_params[:, 0, 5] = pos_y_0
        wave_params[:, 1, 5] = pos_y_1

        wave_mask = torch.ones(B, 2, 1, 1, device=DEVICE)
        return wave_params, wave_mask

    def num_params(self):
        return self.params.size  # 32

    def set_flat_params(self, flat):
        self.params = flat.reshape(4, 2, 2, 2)


class TemporalDecoder:
    def __init__(self):
        self.positions = [
            (0.2, 0.4), (0.2, 0.6),
            (0.4, 0.4), (0.4, 0.6),
            (0.5, 0.4), (0.5, 0.6),
            (0.7, 0.4), (0.7, 0.6),
        ]
        self.weights = torch.zeros(8, device=DEVICE)

    def predict(self, psi_field):
        values = psi_field.read_at(self.positions)  # (B, 8)
        logits = torch.sum(values * self.weights, dim=1)
        return torch.sigmoid(logits)  # (B,)

    def num_params(self):
        return 8

    def set_flat_params(self, flat):
        self.weights = torch.tensor(flat, dtype=torch.float32, device=DEVICE)


# ============================================
# EXECUÇÃO — 2-BIT XOR
# ============================================

def run_batched_xor(emitter, decoder, damping, dissipation,
                    use_field=True, disable_crystals=False,
                    time_cfg: TimeConfig = None):
    if time_cfg is None:
        time_cfg = DEFAULT_TIME_CONFIG

    psi = PsiField(batch_size=4, damping=damping, dissipation=dissipation,
                   disable_crystals=disable_crystals)

    s1_bits = TEMPORAL_XOR_DATA[:, 0]
    s2_bits = TEMPORAL_XOR_DATA[:, 1]

    s1_params, s1_mask = emitter.get_waves_for_batch(0, s1_bits)
    s2_params, s2_mask = emitter.get_waves_for_batch(1, s2_bits)
    silence_mask = torch.zeros(4, 2, 1, 1, device=DEVICE)

    crystal_logs = [{} for _ in range(4)]

    for step in range(time_cfg.read):
        if use_field:
            if time_cfg.s1_start <= step <= time_cfg.s1_end:
                psi.wave_params = s1_params
                psi.wave_mask = s1_mask
            elif time_cfg.s2_start <= step <= time_cfg.s2_end:
                psi.wave_params = s2_params
                psi.wave_mask = s2_mask
            else:
                psi.wave_mask = silence_mask
        else:
            if time_cfg.s2_start <= step <= time_cfg.s2_end:
                psi.wave_params = s2_params
                psi.wave_mask = s2_mask
            else:
                psi.wave_mask = silence_mask

        psi.step()

        if step in time_cfg.monitor_steps:
            for b in range(4):
                crystal_logs[b][step] = psi.crystal_count[b]

    preds = decoder.predict(psi)
    energies = torch.sum(psi.field**2, dim=(1, 2))
    return preds, crystal_logs, energies


def evaluate_temporal(params, n_emitter, n_decoder,
                      use_field=True, verbose=False,
                      disable_crystals=False,
                      time_cfg: TimeConfig = None):
    emitter = TemporalEmitter()
    decoder = TemporalDecoder()

    emitter.set_flat_params(params[:n_emitter].copy())
    decoder.set_flat_params(params[n_emitter:n_emitter + n_decoder].copy())

    raw_damping = float(params[n_emitter + n_decoder])
    raw_dissipation = float(params[n_emitter + n_decoder + 1])

    damping = 0.005 + (0.15 - 0.005) * sigmoid(raw_damping)
    dissipation = 0.001 + (0.02 - 0.001) * sigmoid(raw_dissipation)

    preds, crystal_logs, energies = run_batched_xor(
        emitter, decoder, damping, dissipation,
        use_field=use_field, disable_crystals=disable_crystals,
        time_cfg=time_cfg,
    )

    targets = TEMPORAL_XOR_DATA[:, 2]
    total_loss = bce_loss_batched(preds, targets)

    preds_np = preds.detach().cpu().numpy()
    energies_np = energies.detach().cpu().numpy()

    return (total_loss.item(), preds_np.tolist(), damping, dissipation,
            crystal_logs, energies_np.tolist())


def train_temporal(use_field=True, label="", max_gen=200,
                   disable_crystals=False,
                   time_cfg: TimeConfig = None):
    if time_cfg is None:
        time_cfg = DEFAULT_TIME_CONFIG

    n_emitter = TemporalEmitter().num_params()
    n_decoder = TemporalDecoder().num_params()
    n_physics = 2
    n_total = n_emitter + n_decoder + n_physics

    crystal_tag = " [SEM CRISTAIS]" if disable_crystals else ""
    print(f"  Params: {n_emitter}(E) + {n_decoder}(D) + {n_physics}(Física) = {n_total}{crystal_tag}")
    if use_field:
        print(f"  Campo: 48×48 | Batch: 4 XOR simultâneos")
        print(f"  S1[{time_cfg.s1_start}-{time_cfg.s1_end}] → "
              f"Silêncio[{time_cfg.silence_start}-{time_cfg.silence_end}] → "
              f"S2[{time_cfg.s2_start}-{time_cfg.s2_end}] → "
              f"Leitura[{time_cfg.read}]")
    else:
        print(f"  Sem campo (S1 não injetado)")
    print()

    x0 = np.random.randn(n_total) * 0.3
    es = cma.CMAEvolutionStrategy(
        x0, 0.5,
        {'popsize': 20, 'seed': 42, 'maxiter': max_gen, 'verbose': -9}
    )

    start = time.time()
    best_loss = float('inf')
    best_preds = None
    best_params = None
    best_damping = 0
    best_dissipation = 0
    history_loss = []
    history_preds = []

    gen = 0
    while not es.stop():
        gen += 1
        candidates = es.ask()
        fitnesses = []
        gen_best_loss = float('inf')
        gen_best_preds = None
        gen_best_damping = 0
        gen_best_dissipation = 0
        gen_best_crystals = []

        for c in candidates:
            (loss, preds, damp, diss,
             crystal_logs, energies) = evaluate_temporal(
                c, n_emitter, n_decoder,
                use_field=use_field,
                disable_crystals=disable_crystals,
                time_cfg=time_cfg,
            )
            fitnesses.append(loss)
            if loss < gen_best_loss:
                gen_best_loss = loss
                gen_best_preds = preds
                gen_best_damping = damp
                gen_best_dissipation = diss
                gen_best_crystals = crystal_logs

        es.tell(candidates, fitnesses)
        history_loss.append(gen_best_loss)
        history_preds.append(gen_best_preds)

        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_preds = gen_best_preds
            best_params = candidates[np.argmin(fitnesses)].copy()
            best_damping = gen_best_damping
            best_dissipation = gen_best_dissipation

        if gen % 20 == 0 or gen == 1:
            correct = sum(
                1 for p, (_, _, t) in zip(gen_best_preds, TEMPORAL_XOR)
                if (p > 0.5) == (t > 0.5)
            )
            pstr = ", ".join([f"{p:.2f}" for p in gen_best_preds])
            elapsed = time.time() - start

            if gen_best_crystals and not disable_crystals:
                nc_str = " | ".join(
                    f"t{s}:{gen_best_crystals[0].get(s, 0)}"
                    for s in time_cfg.monitor_steps
                )
            else:
                nc_str = "desativados" if disable_crystals else "n/a"

            print(
                f"  Gen {gen:>4} | Loss: {gen_best_loss:.4f} | "
                f"[{pstr}] | {correct}/4 | "
                f"γ={gen_best_damping:.4f} β={gen_best_dissipation:.4f} | "
                f"Cristais({nc_str}) | "
                f"{elapsed:.0f}s"
            )

        if best_loss < 0.5:
            correct = sum(
                1 for p, (_, _, t) in zip(best_preds, TEMPORAL_XOR)
                if (p > 0.5) == (t > 0.5)
            )
            if correct == 4:
                print(f"\n  → Resolvido na geração {gen}!")
                break

    elapsed = time.time() - start
    correct = sum(
        1 for p, (_, _, t) in zip(best_preds, TEMPORAL_XOR)
        if (p > 0.5) == (t > 0.5)
    )

    print(f"\n  RESULTADO {label}:")
    print(f"  Loss: {best_loss:.4f} | {correct}/4 | {elapsed:.1f}s")
    print(f"  Damping aprendido: γ = {best_damping:.4f}")
    print(f"  Dissipação aprendida: β = {best_dissipation:.4f}")

    if best_params is not None:
        print(f"\n  Detalhes por input:")
        (_, final_preds, _, _, final_crystals, final_energies) = evaluate_temporal(
            best_params, n_emitter, n_decoder,
            use_field=use_field, verbose=True,
            disable_crystals=disable_crystals,
            time_cfg=time_cfg,
        )
        for (s1, s2, t), p, cl, en in zip(
                TEMPORAL_XOR, final_preds, final_crystals, final_energies):
            b = 1 if p > 0.5 else 0
            s = "✓" if b == int(t) else "✗"
            crystal_detail = ", ".join(
                f"t{step}:{cl.get(step, 0)}" for step in time_cfg.monitor_steps
            )
            print(
                f"    S1={s1} S2={s2} → {p:.4f} (target {t}) {s} | "
                f"Energia: {en:.1f} | Cristais: [{crystal_detail}]"
            )

    return correct, best_loss, history_loss, history_preds, elapsed, best_damping, best_dissipation


# ============================================
# EXECUÇÃO — PARIDADE 3 BITS
# ============================================

def run_batched_parity3(emitter, decoder, damping, dissipation,
                        time_cfg: TimeConfig,
                        disable_crystals=False):
    psi = PsiField(batch_size=8, damping=damping, dissipation=dissipation,
                   disable_crystals=disable_crystals)

    s1_bits = PARITY_3BIT_DATA[:, 0]
    s2_bits = PARITY_3BIT_DATA[:, 1]
    s3_bits = PARITY_3BIT_DATA[:, 2]

    s1_params, s1_mask = emitter.get_waves_for_batch(0, s1_bits)
    s2_params, s2_mask = emitter.get_waves_for_batch(1, s2_bits)
    s3_params, s3_mask = emitter.get_waves_for_batch(2, s3_bits)
    silence_mask = torch.zeros(8, 2, 1, 1, device=DEVICE)

    crystal_logs = [{} for _ in range(8)]

    for step in range(time_cfg.read):
        if time_cfg.s1_start <= step <= time_cfg.s1_end:
            psi.wave_params = s1_params
            psi.wave_mask = s1_mask
        elif time_cfg.s2_start <= step <= time_cfg.s2_end:
            psi.wave_params = s2_params
            psi.wave_mask = s2_mask
        elif time_cfg.s3_start <= step <= time_cfg.s3_end:
            psi.wave_params = s3_params
            psi.wave_mask = s3_mask
        else:
            psi.wave_mask = silence_mask

        psi.step()

        if step in time_cfg.monitor_steps:
            for b in range(8):
                crystal_logs[b][step] = psi.crystal_count[b]

    preds = decoder.predict(psi)   # (8,)
    energies = torch.sum(psi.field**2, dim=(1, 2))
    return preds, crystal_logs, energies


def evaluate_parity3(params, n_emitter, n_decoder,
                     time_cfg: TimeConfig,
                     disable_crystals=False):
    emitter = TemporalEmitter()
    decoder = TemporalDecoder()

    emitter.set_flat_params(params[:n_emitter].copy())
    decoder.set_flat_params(params[n_emitter:n_emitter + n_decoder].copy())

    raw_damping = float(params[n_emitter + n_decoder])
    raw_dissipation = float(params[n_emitter + n_decoder + 1])

    damping = 0.005 + (0.15 - 0.005) * sigmoid(raw_damping)
    dissipation = 0.001 + (0.02 - 0.001) * sigmoid(raw_dissipation)

    preds, crystal_logs, energies = run_batched_parity3(
        emitter, decoder, damping, dissipation,
        time_cfg=time_cfg, disable_crystals=disable_crystals,
    )

    targets = PARITY_3BIT_DATA[:, 3]
    total_loss = bce_loss_batched(preds, targets)

    return (total_loss.item(), preds.detach().cpu().numpy().tolist(),
            damping, dissipation, crystal_logs,
            energies.detach().cpu().numpy().tolist())


def train_parity3(label="", max_gen=300,
                  time_cfg: TimeConfig = None,
                  disable_crystals=False):
    if time_cfg is None:
        time_cfg = PARITY_3BIT_TIME

    n_emitter = TemporalEmitter().num_params()  # 24
    n_decoder = TemporalDecoder().num_params()     # 8
    n_physics = 2
    n_total = n_emitter + n_decoder + n_physics

    crystal_tag = " [SEM CRISTAIS]" if disable_crystals else ""
    print(f"  Params: {n_emitter}(E) + {n_decoder}(D) + {n_physics}(Física) = {n_total}{crystal_tag}")
    print(f"  Campo: 48×48 | Batch: 8 combinações simultâneas")
    print(f"  S1[{time_cfg.s1_start}-{time_cfg.s1_end}] → "
          f"Silêncio → S2[{time_cfg.s2_start}-{time_cfg.s2_end}] → "
          f"Silêncio → S3[{time_cfg.s3_start}-{time_cfg.s3_end}] → "
          f"Leitura[{time_cfg.read}]")
    print()

    x0 = np.random.randn(n_total) * 0.3
    es = cma.CMAEvolutionStrategy(
        x0, 0.5,
        {'popsize': 30, 'seed': 42, 'maxiter': max_gen, 'verbose': -9}
    )

    start = time.time()
    best_loss = float('inf')
    best_preds = None
    best_params = None
    best_damping = 0
    best_dissipation = 0
    history_loss = []
    history_preds = []

    gen = 0
    while not es.stop():
        gen += 1
        candidates = es.ask()
        fitnesses = []
        gen_best_loss = float('inf')
        gen_best_preds = None
        gen_best_damping = 0
        gen_best_dissipation = 0
        gen_best_crystals = []

        for c in candidates:
            (loss, preds, damp, diss,
             crystal_logs, energies) = evaluate_parity3(
                c, n_emitter, n_decoder,
                time_cfg=time_cfg,
                disable_crystals=disable_crystals,
            )
            fitnesses.append(loss)
            if loss < gen_best_loss:
                gen_best_loss = loss
                gen_best_preds = preds
                gen_best_damping = damp
                gen_best_dissipation = diss
                gen_best_crystals = crystal_logs

        es.tell(candidates, fitnesses)
        history_loss.append(gen_best_loss)
        history_preds.append(gen_best_preds)

        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_preds = gen_best_preds
            best_params = candidates[np.argmin(fitnesses)].copy()
            best_damping = gen_best_damping
            best_dissipation = gen_best_dissipation

        if gen % 20 == 0 or gen == 1:
            correct = sum(
                1 for p, (*_, t) in zip(gen_best_preds, PARITY_3BIT)
                if (p > 0.5) == (t > 0.5)
            )
            pstr = ", ".join([f"{p:.2f}" for p in gen_best_preds])
            elapsed = time.time() - start

            if gen_best_crystals and not disable_crystals:
                nc_str = " | ".join(
                    f"t{s}:{gen_best_crystals[0].get(s, 0)}"
                    for s in time_cfg.monitor_steps
                )
            else:
                nc_str = "desativados" if disable_crystals else "n/a"

            print(
                f"  Gen {gen:>4} | Loss: {gen_best_loss:.4f} | {correct}/8 | "
                f"γ={gen_best_damping:.4f} β={gen_best_dissipation:.4f} | "
                f"Cristais({nc_str}) | {elapsed:.0f}s"
            )

        if best_loss < 1.0:
            correct = sum(
                1 for p, (*_, t) in zip(best_preds, PARITY_3BIT)
                if (p > 0.5) == (t > 0.5)
            )
            if correct == 8:
                print(f"\n  → Resolvido na geração {gen}!")
                break

    elapsed = time.time() - start
    correct = sum(
        1 for p, (*_, t) in zip(best_preds, PARITY_3BIT)
        if (p > 0.5) == (t > 0.5)
    )

    print(f"\n  RESULTADO {label}:")
    print(f"  Loss: {best_loss:.4f} | {correct}/8 | {elapsed:.1f}s")
    print(f"  Damping aprendido: γ = {best_damping:.4f}")
    print(f"  Dissipação aprendida: β = {best_dissipation:.4f}")

    if best_params is not None:
        print(f"\n  Detalhes por input:")
        (_, final_preds, _, _, final_crystals, final_energies) = evaluate_parity3(
            best_params, n_emitter, n_decoder,
            time_cfg=time_cfg,
            disable_crystals=disable_crystals,
        )
        for (s1, s2, s3, t), p, cl, en in zip(
                PARITY_3BIT, final_preds, final_crystals, final_energies):
            b = 1 if p > 0.5 else 0
            s = "✓" if b == int(t) else "✗"
            crystal_detail = ", ".join(
                f"t{step}:{cl.get(step, 0)}" for step in time_cfg.monitor_steps
            )
            print(
                f"    S1={s1} S2={s2} S3={s3} → {p:.4f} (target {t}) {s} | "
                f"Energia: {en:.1f} | Cristais: [{crystal_detail}]"
            )

    return correct, best_loss, history_loss, history_preds, elapsed, best_damping, best_dissipation


# ============================================
# EXECUÇÃO — PARIDADE 4 BITS
# ============================================

def run_batched_parity4(emitter, decoder, damping, dissipation,
                        time_cfg: TimeConfig,
                        disable_crystals=False):
    psi = PsiField(batch_size=16, damping=damping, dissipation=dissipation,
                   disable_crystals=disable_crystals)

    s1_bits = PARITY_4BIT_DATA[:, 0]
    s2_bits = PARITY_4BIT_DATA[:, 1]
    s3_bits = PARITY_4BIT_DATA[:, 2]
    s4_bits = PARITY_4BIT_DATA[:, 3]

    s1_params, s1_mask = emitter.get_waves_for_batch(0, s1_bits)
    s2_params, s2_mask = emitter.get_waves_for_batch(1, s2_bits)
    s3_params, s3_mask = emitter.get_waves_for_batch(2, s3_bits)
    s4_params, s4_mask = emitter.get_waves_for_batch(3, s4_bits)
    silence_mask = torch.zeros(16, 2, 1, 1, device=DEVICE)

    crystal_logs = [{} for _ in range(16)]

    for step in range(time_cfg.read):
        if time_cfg.s1_start <= step <= time_cfg.s1_end:
            psi.wave_params = s1_params
            psi.wave_mask = s1_mask
        elif time_cfg.s2_start <= step <= time_cfg.s2_end:
            psi.wave_params = s2_params
            psi.wave_mask = s2_mask
        elif time_cfg.s3_start <= step <= time_cfg.s3_end:
            psi.wave_params = s3_params
            psi.wave_mask = s3_mask
        elif time_cfg.s4_start <= step <= time_cfg.s4_end:
            psi.wave_params = s4_params
            psi.wave_mask = s4_mask
        else:
            psi.wave_mask = silence_mask

        psi.step()

        if step in time_cfg.monitor_steps:
            for b in range(16):
                crystal_logs[b][step] = psi.crystal_count[b]

    preds = decoder.predict(psi)   # (16,)
    energies = torch.sum(psi.field**2, dim=(1, 2))
    return preds, crystal_logs, energies


def evaluate_parity4(params, n_emitter, n_decoder,
                     time_cfg: TimeConfig,
                     disable_crystals=False):
    emitter = TemporalEmitter()
    decoder = TemporalDecoder()

    emitter.set_flat_params(params[:n_emitter].copy())
    decoder.set_flat_params(params[n_emitter:n_emitter + n_decoder].copy())

    raw_damping = float(params[n_emitter + n_decoder])
    raw_dissipation = float(params[n_emitter + n_decoder + 1])

    damping = 0.005 + (0.15 - 0.005) * sigmoid(raw_damping)
    dissipation = 0.001 + (0.02 - 0.001) * sigmoid(raw_dissipation)

    preds, crystal_logs, energies = run_batched_parity4(
        emitter, decoder, damping, dissipation,
        time_cfg=time_cfg, disable_crystals=disable_crystals,
    )

    targets = PARITY_4BIT_DATA[:, 4]
    total_loss = bce_loss_batched(preds, targets)

    return (total_loss.item(), preds.detach().cpu().numpy().tolist(),
            damping, dissipation, crystal_logs,
            energies.detach().cpu().numpy().tolist())


def train_parity4(label="", max_gen=400,
                  time_cfg: TimeConfig = None,
                  disable_crystals=False):
    if time_cfg is None:
        time_cfg = PARITY_4BIT_TIME

    n_emitter = TemporalEmitter().num_params()  # 32
    n_decoder = TemporalDecoder().num_params()     # 8
    n_physics = 2
    n_total = n_emitter + n_decoder + n_physics

    crystal_tag = " [SEM CRISTAIS]" if disable_crystals else ""
    print(f"  Params: {n_emitter}(E) + {n_decoder}(D) + {n_physics}(Física) = {n_total}{crystal_tag}")
    print(f"  Campo: 48×48 | Batch: 16 combinações simultâneas")
    print(f"  S1[{time_cfg.s1_start}-{time_cfg.s1_end}] → "
          f"Silêncio → S2[{time_cfg.s2_start}-{time_cfg.s2_end}] → "
          f"Silêncio → S3[{time_cfg.s3_start}-{time_cfg.s3_end}] → "
          f"Silêncio → S4[{time_cfg.s4_start}-{time_cfg.s4_end}] → "
          f"Leitura[{time_cfg.read}]")
    print()

    x0 = np.random.randn(n_total) * 0.3
    es = cma.CMAEvolutionStrategy(
        x0, 0.5,
        {'popsize': 32, 'seed': 42, 'maxiter': max_gen, 'verbose': -9}
    )

    start = time.time()
    best_loss = float('inf')
    best_preds = None
    best_params = None
    best_damping = 0
    best_dissipation = 0
    history_loss = []
    history_preds = []

    gen = 0
    while not es.stop():
        gen += 1
        candidates = es.ask()
        fitnesses = []
        gen_best_loss = float('inf')
        gen_best_preds = None
        gen_best_damping = 0
        gen_best_dissipation = 0
        gen_best_crystals = []

        for c in candidates:
            (loss, preds, damp, diss,
             crystal_logs, energies) = evaluate_parity4(
                c, n_emitter, n_decoder,
                time_cfg=time_cfg,
                disable_crystals=disable_crystals,
            )
            fitnesses.append(loss)
            if loss < gen_best_loss:
                gen_best_loss = loss
                gen_best_preds = preds
                gen_best_damping = damp
                gen_best_dissipation = diss
                gen_best_crystals = crystal_logs

        es.tell(candidates, fitnesses)
        history_loss.append(gen_best_loss)
        history_preds.append(gen_best_preds)

        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_preds = gen_best_preds
            best_params = candidates[np.argmin(fitnesses)].copy()
            best_damping = gen_best_damping
            best_dissipation = gen_best_dissipation

        if gen % 20 == 0 or gen == 1:
            correct = sum(
                1 for p, (*_, t) in zip(gen_best_preds, PARITY_4BIT)
                if (p > 0.5) == (t > 0.5)
            )
            pstr = ", ".join([f"{p:.2f}" for p in gen_best_preds])
            elapsed = time.time() - start

            if gen_best_crystals and not disable_crystals:
                nc_str = " | ".join(
                    f"t{s}:{gen_best_crystals[0].get(s, 0)}"
                    for s in time_cfg.monitor_steps
                )
            else:
                nc_str = "desativados" if disable_crystals else "n/a"

            print(
                f"  Gen {gen:>4} | Loss: {gen_best_loss:.4f} | {correct}/16 | "
                f"γ={gen_best_damping:.4f} β={gen_best_dissipation:.4f} | "
                f"Cristais({nc_str}) | {elapsed:.0f}s"
            )

        if best_loss < 2.0:
            correct = sum(
                1 for p, (*_, t) in zip(best_preds, PARITY_4BIT)
                if (p > 0.5) == (t > 0.5)
            )
            if correct == 16:
                print(f"\n  → Resolvido na geração {gen}!")
                break

    elapsed = time.time() - start
    correct = sum(
        1 for p, (*_, t) in zip(best_preds, PARITY_4BIT)
        if (p > 0.5) == (t > 0.5)
    )

    print(f"\n  RESULTADO {label}:")
    print(f"  Loss: {best_loss:.4f} | {correct}/16 | {elapsed:.1f}s")
    print(f"  Damping aprendido: γ = {best_damping:.4f}")
    print(f"  Dissipação aprendida: β = {best_dissipation:.4f}")

    if best_params is not None:
        print(f"\n  Detalhes por input:")
        (_, final_preds, _, _, final_crystals, final_energies) = evaluate_parity4(
            best_params, n_emitter, n_decoder,
            time_cfg=time_cfg,
            disable_crystals=disable_crystals,
        )
        for (s1, s2, s3, s4, t), p, cl, en in zip(
                PARITY_4BIT, final_preds, final_crystals, final_energies):
            b = 1 if p > 0.5 else 0
            s = "✓" if b == int(t) else "✗"
            crystal_detail = ", ".join(
                f"t{step}:{cl.get(step, 0)}" for step in time_cfg.monitor_steps
            )
            print(
                f"    S1={s1} S2={s2} S3={s3} S4={s4} → {p:.4f} (target {t}) {s} | "
                f"Energia: {en:.1f} | Cristais: [{crystal_detail}]"
            )

    return correct, best_loss, history_loss, history_preds, elapsed, best_damping, best_dissipation


# ============================================
# MAIN
# ============================================

def main():
    total_start = time.time()

    print("╔" + "═" * 62 + "╗")
    print("║   ResNet-Ψ — XOR TEMPORAL + ABLAÇÕES + PARIDADE 4-BIT     ║")
    print("╚" + "═" * 62 + "╝\n")

    # ──────────────────────────────────────────────
    # ORIGINAL — XOR Temporal v2 (referência)
    # ──────────────────────────────────────────────
    print("=" * 58)
    print("  ORIGINAL — COM Ψ-FIELD (referência)")
    print("=" * 58)
    (f_correct, f_loss, f_hist_loss, f_hist_preds,
     f_time, f_damp, f_diss) = train_temporal(
        use_field=True, label="COM CAMPO", max_gen=200
    )

    print("\n" + "=" * 58)
    print("  ORIGINAL — SEM Ψ-FIELD (controle)")
    print("=" * 58)
    (c_correct, c_loss, c_hist_loss, c_hist_preds,
     c_time, c_damp, c_diss) = train_temporal(
        use_field=False, label="SEM CAMPO", max_gen=200
    )

    # ──────────────────────────────────────────────
    # TESTE 1 — Ablação de Cristais
    # ──────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  TESTE 1 — ABLAÇÃO DE CRISTAIS (campo sem cristais)")
    print("=" * 58)
    (a_correct, a_loss, a_hist_loss, a_hist_preds,
     a_time, a_damp, a_diss) = train_temporal(
        use_field=True, disable_crystals=True,
        label="ABLAÇÃO CRISTAIS", max_gen=200
    )

    # ──────────────────────────────────────────────
    # TESTE 2 — Estresse de Silêncio (300 steps)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  TESTE 2 — ESTRESSE DE SILÊNCIO (300 steps)")
    print("=" * 58)
    (ss_correct, ss_loss, ss_hist_loss, ss_hist_preds,
     ss_time, ss_damp, ss_diss) = train_temporal(
        use_field=True, time_cfg=SILENCE_STRESS_CONFIG,
        label="SILÊNCIO 300 STEPS", max_gen=200
    )

    # ──────────────────────────────────────────────
    # TESTE 3 — Paridade 3 Bits
    # ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  TESTE 3 — PARIDADE 3 BITS")
    print("=" * 62)
    (p3_correct, p3_loss, p3_hist_loss, p3_hist_preds,
     p3_elapsed, p3_damp, p3_diss) = train_parity3(
        time_cfg=PARITY_3BIT_TIME,
        label="PARIDADE 3 BITS", max_gen=300
    )

    # ──────────────────────────────────────────────
    # TESTE 4 — Paridade 4 Bits (Escudo de Cristal)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  TESTE 4 — PARIDADE 4 BITS (Escudo de Cristal + Janelas Espaçadas)")
    print("=" * 62)
    (p4_correct, p4_loss, p4_hist_loss, p4_hist_preds,
     p4_elapsed, p4_damp, p4_diss) = train_parity4(
        time_cfg=PARITY_4BIT_TIME,
        label="PARIDADE 4 BITS", max_gen=400
    )

    total = time.time() - total_start

    # ──────────────────────────────────────────────
    # RELATÓRIO CONSOLIDADO
    # ──────────────────────────────────────────────
    print("\n")
    print("╔" + "═" * 62 + "╗")
    print("║        RELATÓRIO CONSOLIDADO — ResNet-Ψ                  ║")
    print("╠" + "═" * 62 + "╣")

    def _row(label, correct, total_cases, loss, extra=""):
        s = "✓" if correct == total_cases else "✗"
        line = f"  {s} {label:<30} {correct}/{total_cases}  Loss: {loss:.4f}{extra}"
        print(f"║{line:<62}║")

    print(f"║{'':62}║")
    _row("Original com campo",       f_correct,  4,  f_loss)
    _row("Original sem campo",       c_correct,  4,  c_loss)
    print(f"║{'':62}║")
    _row("Teste 1: ablação cristais",   a_correct,  4,  a_loss)
    _row("Teste 2: silêncio 300s",      ss_correct, 4,  ss_loss)
    _row("Teste 3: paridade 3 bits",    p3_correct, 8,  p3_loss)
    _row("Teste 4: paridade 4 bits",    p4_correct, 16, p4_loss)
    print(f"║{'':62}║")

    # Interpretação
    if f_correct == 4 and c_correct < 4:
        print(f"║  → Campo é necessário para XOR temporal                 ║")
    if a_correct == 4:
        print(f"║  → Ondas sozinhas retêm memória por 40 steps            ║")
    elif a_correct < 4 and f_correct == 4:
        print(f"║  → Cristais são obrigatórios para retenção              ║")
    if ss_correct == 4:
        print(f"║  → Campo retém memória por 300 steps de silêncio        ║")
    else:
        print(f"║  → 300 steps dissipam — cristais não compensaram        ║")
    if p3_correct == 8:
        print(f"║  → Campo suporta 3 memórias temporais consecutivas      ║")
    else:
        print(f"║  → Saturação detectada em paridade 3-bit                ║")
    if p4_correct == 16:
        print(f"║  → CAMPO RESOLVE 4-BIT! Escudo de cristal funciona      ║")
    elif p4_correct >= 12:
        print(f"║  → Campo aproxima 4-bit ({p4_correct}/16) — escudo parcial       ║")
    else:
        print(f"║  → 4-bit ainda difícil — mais capacidade necessária     ║")

    print(f"║{'':62}║")
    mins = total / 60
    print(f"║  Tempo total: {total:.0f}s ({mins:.1f} min)"
          + " " * max(0, 35 - len(f"{total:.0f}")) + "║")
    print("╚" + "═" * 62 + "╝")

    # ──────────────────────────────────────────────
    # VISUALIZAÇÃO
    # ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("ResNet-Ψ — Ablações, Paridade 3-bit e 4-bit (CUDA)", fontsize=14, fontweight='bold')

    def plot_loss(ax, histories, labels, colors, title):
        ax.set_title(title)
        for hist, lbl, col in zip(histories, labels, colors):
            ax.plot(hist, color=col, linewidth=1.5, label=lbl)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel("Geração")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        all_l = [v for h in histories for v in h]
        if all_l and max(all_l) > 5 * (min(all_l) + 1e-8):
            ax.set_yscale('log')

    plot_loss(axes[0, 0],
              [f_hist_loss, c_hist_loss],
              [f"Com campo ({f_correct}/4)", f"Sem campo ({c_correct}/4)"],
              ['#00d4aa', '#ff6b35'],
              "Original: Com vs Sem Campo")

    plot_loss(axes[0, 1],
              [f_hist_loss, a_hist_loss],
              [f"Com cristais ({f_correct}/4)", f"Ablação ({a_correct}/4)"],
              ['#00d4aa', '#9b59b6'],
              "Teste 1: Ablação de Cristais")

    plot_loss(axes[0, 2],
              [f_hist_loss, ss_hist_loss],
              [f"40 steps ({f_correct}/4)", f"300 steps ({ss_correct}/4)"],
              ['#00d4aa', '#e74c3c'],
              "Teste 2: Estresse de Silêncio")

    # Predições — original
    ax = axes[1, 0]
    ax.set_title(f"Predições Original COM Campo ({f_correct}/4)")
    if f_hist_preds:
        pred_arr = np.array(f_hist_preds)
        for i, (s1, s2, t) in enumerate(TEMPORAL_XOR):
            color = '#00d4aa' if t == 1 else '#ff6b35'
            ax.plot(pred_arr[:, i], color=color, alpha=0.7,
                    linestyle='-' if t == 1 else '--',
                    label=f"({s1},{s2})→{t}")
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8)

    # Predições — paridade 3 bits
    ax = axes[1, 1]
    ax.set_title(f"Predições Paridade 3 Bits ({p3_correct}/8)")
    if p3_hist_preds:
        pred_arr = np.array(p3_hist_preds)
        colors8 = ['#00d4aa','#ff6b35','#9b59b6','#e74c3c',
                   '#3498db','#f39c12','#1abc9c','#e67e22']
        for i, (s1, s2, s3, t) in enumerate(PARITY_3BIT):
            ax.plot(pred_arr[:, i], color=colors8[i], alpha=0.7,
                    label=f"({s1},{s2},{s3})→{t}")
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=7, ncol=2)

    # Predições — paridade 4 bits
    ax = axes[1, 2]
    ax.set_title(f"Predições Paridade 4 Bits ({p4_correct}/16)")
    if p4_hist_preds:
        pred_arr = np.array(p4_hist_preds)
        colors16 = [
            '#00d4aa','#ff6b35','#9b59b6','#e74c3c',
            '#3498db','#f39c12','#1abc9c','#e67e22',
            '#2ecc71','#e91e63','#00bcd4','#ff5722',
            '#607d8b','#8bc34a','#673ab7','#ff9800',
        ]
        for i, (s1, s2, s3, s4, t) in enumerate(PARITY_4BIT):
            ax.plot(pred_arr[:, i], color=colors16[i], alpha=0.6,
                    linewidth=0.9, label=f"({s1},{s2},{s3},{s4})→{t}")
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.show()


# RODAR
main()
