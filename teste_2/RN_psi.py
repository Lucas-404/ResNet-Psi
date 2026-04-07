import torch
import torch.nn.functional as F
import numpy as np
import time
import cma
from dataclasses import dataclass
from typing import Optional, List

import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" usando dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f" CUDA version: {torch.version.cuda}")
else:
    print(" CUDA não disponível — rodando em CPU")

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
    read:          int = 200

    @property
    def monitor_steps(self) -> List[int]:
        steps = [self.s1_end, self.silence_end, self.s2_end]
        if self.s3_end is not None:
            steps.append(self.s3_end)
        steps.append(self.read)
        return steps

# 40 steps de silêncio entre cada bit — mesmo padrão do 4-bit
PARITY_3BIT_TIME = TimeConfig(
    s1_start=0,    s1_end=40,
    silence_start=41, silence_end=79,
    s2_start=80,   s2_end=120,
    s3_start=160,  s3_end=200,
    read=270,
)

# Paridade 4-bit: 40 steps de silêncio entre cada bit para cristalização
PARITY_4BIT_TIME = TimeConfig(
    s1_start=0,    s1_end=40,
    silence_start=41, silence_end=79,
    s2_start=80,   s2_end=120,
    s3_start=160,  s3_end=200,
    read=350,
)
# S4 injetado em t=240-280, leitura em t=350
PARITY_4BIT_S4_START = 240
PARITY_4BIT_S4_END   = 280

# ============================================
# DATASETS
# ============================================

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

PARITY_4BIT_DATA = torch.tensor([
    [0,0,0,0, 0], [0,0,0,1, 1], [0,0,1,0, 1], [0,0,1,1, 0],
    [0,1,0,0, 1], [0,1,0,1, 0], [0,1,1,0, 0], [0,1,1,1, 1],
    [1,0,0,0, 1], [1,0,0,1, 0], [1,0,1,0, 0], [1,0,1,1, 1],
    [1,1,0,0, 0], [1,1,0,1, 1], [1,1,1,0, 1], [1,1,1,1, 0],
], dtype=torch.float32, device=DEVICE)

PARITY_4BIT = [(int(r[0]),int(r[1]),int(r[2]),int(r[3]),int(r[4])) for r in PARITY_4BIT_DATA]

PSI_GAMMA_3BIT = 0.085421  # damping
PSI_BETA_3BIT  = 0.007453  # dissipation
PSI_SIGMA_3BIT = 0.632152  # largura gaussiana de ressonância

class PsiField:
    def __init__(self, batch_size=4, size=48, damping=0.03,
                 dissipation=0.005, disable_crystals=False, sigma=0.8):
        self.batch_size = batch_size
        self.size = size
        self.disable_crystals = disable_crystals

        # sigma: escalar ou tensor (B,) — armazenado como (B,) para broadcasting em step()
        if isinstance(sigma, torch.Tensor):
            self.sigma = sigma.to(DEVICE)  # (B,)
        else:
            self.sigma = torch.full((batch_size,), float(sigma), device=DEVICE)  # (B,)

        # Conversão vetorial — broadcasting (B,1,1) funciona nativamente no step()
        if isinstance(damping, (float, int)):
            self.damping = torch.full((batch_size, 1, 1), float(damping), device=DEVICE)
        else:
            self.damping = damping.view(batch_size, 1, 1).clone()

        if isinstance(dissipation, (float, int)):
            self.mu_base = torch.full((batch_size, 1, 1), float(dissipation), device=DEVICE)
        else:
            self.mu_base = dissipation.view(batch_size, 1, 1).clone()

        # Rigidez local — EMA da amplitude instantânea (B, S, S)
        self.rigidity  = torch.zeros(batch_size, size, size, device=DEVICE)
        self.alpha     = 0.95   # coeficiente EMA: janela efetiva ~20 steps
        self.mu_extra  = 0.5    # dissipação adicional máxima sobre cristais rígidos

        self.field    = torch.zeros(batch_size, size, size, device=DEVICE)
        self.velocity = torch.zeros(batch_size, size, size, device=DEVICE)

        coords = torch.linspace(0, 1, size, device=DEVICE)
        self.x_grid, self.y_grid = torch.meshgrid(coords, coords, indexing='ij')

        # Ondas: (B, W, 6) = [amp, freq, phase, decay, pos_x, pos_y]
        self.wave_params = torch.zeros(batch_size, 2, 6, device=DEVICE)
        self.wave_mask   = torch.zeros(batch_size, 2, 1, 1, device=DEVICE)

        self.step_count = 0

        # Cristais
        self.max_crystals = 0 if disable_crystals else 100
        self.crystal_radius = 5
        self.cpat_size = 2 * self.crystal_radius + 1  # 11
        self.crystal_positions = torch.zeros(batch_size, self.max_crystals, 2, dtype=torch.long, device=DEVICE)
        self.crystal_patterns  = torch.zeros(batch_size, self.max_crystals, self.cpat_size, self.cpat_size, device=DEVICE)
        self.crystal_active    = torch.zeros(batch_size, self.max_crystals, device=DEVICE)
        self.crystal_freqs     = torch.zeros(batch_size, self.max_crystals, device=DEVICE)  # freq dominante na cristalização
        self.crystal_count     = [0] * batch_size

        ps = self.cpat_size
        self._off_x = torch.arange(ps, device=DEVICE).unsqueeze(1).expand(ps, ps).reshape(-1)
        self._off_y = torch.arange(ps, device=DEVICE).unsqueeze(0).expand(ps, ps).reshape(-1)
        decay_coords = torch.arange(ps, dtype=torch.float32, device=DEVICE) - self.crystal_radius
        dx, dy = torch.meshgrid(decay_coords, decay_coords, indexing='ij')
        self.crystal_reemit_decay = 1.0 / (1.0 + torch.sqrt(dx**2 + dy**2))

        # Envelope tracking
        self.envelope_window      = 25
        self.envelope_num_windows = 6
        self.current_window_max   = torch.zeros(batch_size, size, size, device=DEVICE)
        self.envelope_history     = torch.zeros(batch_size, self.envelope_num_windows, size, size, device=DEVICE)
        self.window_step_count    = 0
        self.completed_windows    = 0

        # Laplaciano
        self.laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)

    def emit_all_waves(self, t):
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

        distance = torch.sqrt((xg - px)**2 + (yg - py)**2 + 1e-8)

        f = freq.unsqueeze(-1).unsqueeze(-1)
        p = phase.unsqueeze(-1).unsqueeze(-1)
        d = decay.unsqueeze(-1).unsqueeze(-1)
        a = amp.unsqueeze(-1).unsqueeze(-1)

        oscillation    = torch.sin(f * t + p - f * distance)
        temporal_decay = torch.exp(-d * t)
        spatial_decay  = 1.0 / (1.0 + distance)

        per_wave = a * oscillation * temporal_decay * spatial_decay * self.wave_mask
        return per_wave.sum(dim=1)  # (B, S, S)

    def step(self):
        dt = 0.05
        t  = self.step_count * dt

        emission   = self.emit_all_waves(t)
        self.field = self.field + emission * dt * 0.1

        input_tensor = self.field.unsqueeze(1)
        padded       = F.pad(input_tensor, (1, 1, 1, 1), mode='circular')
        laplacian    = F.conv2d(padded, self.laplacian_kernel).squeeze(1)

        nonlinear = 0.04 * torch.tanh(self.field) * self.field

        # Rigidez EMA: R_t = α·R_{t-1} + (1-α)·|Ψ_t|
        self.rigidity = self.rigidity * self.alpha + torch.abs(self.field) * (1.0 - self.alpha)

        # Dissipação efetiva: μ_eff = μ_base + R·μ_extra
        # mu_base é (B,1,1), rigidity é (B,S,S) — broadcasting automático
        mu_eff = self.mu_base + self.rigidity * self.mu_extra

        # damping é (B,1,1), mu_eff é (B,S,S) — broadcasting automático
        acceleration = (
            0.3 * laplacian
            - self.damping * self.velocity
            + nonlinear
            - mu_eff * self.field * (self.field ** 2)
        )

        # Re-emissão de cristais
        if not self.disable_crystals and self.crystal_active.any():
            r  = self.crystal_radius
            pw = self.size + 2 * r

            padded_c    = F.pad(self.field.unsqueeze(1), (r,r,r,r), mode='constant', value=0).squeeze(1)
            padded_flat = padded_c.reshape(self.batch_size, -1)

            cx = self.crystal_positions[:, :, 0]
            cy = self.crystal_positions[:, :, 1]
            lin_idx = (
                (cx.unsqueeze(-1) + self._off_x.unsqueeze(0).unsqueeze(0)) * pw +
                (cy.unsqueeze(-1) + self._off_y.unsqueeze(0).unsqueeze(0))
            )

            regions  = torch.gather(
                padded_flat.unsqueeze(1).expand(-1, self.max_crystals, -1), 2, lin_idx
            )

            pat_flat = self.crystal_patterns.reshape(self.batch_size, self.max_crystals, -1)
            p_norm   = pat_flat / (torch.norm(pat_flat, dim=-1, keepdim=True) + 1e-8)
            r_norm   = regions  / (torch.norm(regions,  dim=-1, keepdim=True) + 1e-8)
            corr     = (p_norm * r_norm).sum(dim=-1)

            # Equação 6 — R_freq: peso gaussiano pela distância espectral entre
            # a frequência instantânea das ondas ativas e a frequência gravada de cada cristal.
            # self.crystal_freqs: (B, max_crystals) — freq dominante no momento da cristalização
            f_now   = self.wave_params[:, :, 1].mean(dim=1, keepdim=True)           # (B, 1)
            delta_f = (f_now.expand(-1, self.max_crystals) - self.crystal_freqs).abs()  # (B, max_crystals)
            R_freq  = torch.exp(-(delta_f ** 2) / (2.0 * self.sigma.unsqueeze(1) ** 2))  # (B, max_crystals)

            mask  = self.crystal_active * (corr.abs() > 0.1).float()
            scale = corr * 0.35 * mask * R_freq
            contrib = (
                self.crystal_patterns
                * self.crystal_reemit_decay.unsqueeze(0).unsqueeze(0)
                * scale.unsqueeze(-1).unsqueeze(-1)
            )
            contrib_flat = contrib.reshape(self.batch_size, -1)
            idx_flat     = lin_idx.reshape(self.batch_size, -1)

            padded_accum = torch.zeros_like(padded_flat)
            padded_accum.scatter_add_(1, idx_flat, contrib_flat)
            accum      = padded_accum.reshape(self.batch_size, pw, pw)
            self.field = self.field + accum[:, r:r+self.size, r:r+self.size]

        self.velocity = self.velocity + acceleration * dt
        self.field    = self.field    + self.velocity * dt

        self.field    = torch.clamp(self.field,    -10.0, 10.0)
        self.velocity = torch.clamp(self.velocity,  -5.0,  5.0)

        # Envelope tracking
        self.current_window_max = torch.max(self.current_window_max, torch.abs(self.field))
        self.window_step_count += 1

        if self.window_step_count >= self.envelope_window:
            idx = self.completed_windows % self.envelope_num_windows
            self.envelope_history[:, idx] = self.current_window_max.clone()
            self.completed_windows += 1
            self.current_window_max = torch.zeros_like(self.current_window_max)
            self.window_step_count  = 0

            if not self.disable_crystals and self.completed_windows >= self.envelope_num_windows:
                self.try_crystallize()

        self.step_count += 1

    def try_crystallize(self):
        env_mean = torch.mean(self.envelope_history, dim=1)   # (B, S, S)
        env_std  = torch.std(self.envelope_history,  dim=1)
        cv       = env_std / (env_mean + 1e-8)

        candidates_mask = (env_mean > 0.15) & (cv < 0.18) & (env_mean < 9.0)  # (B, S, S)

        r  = self.crystal_radius
        ps = self.cpat_size

        occupied = torch.zeros(self.batch_size, 1, self.size, self.size, device=DEVICE)
        for b in range(self.batch_size):
            cc = self.crystal_count[b]
            if cc > 0:
                px = self.crystal_positions[b, :cc, 0]
                py = self.crystal_positions[b, :cc, 1]
                occupied[b, 0, px, py] = 1.0

        # Dilatar raio r ao redor de cada cristal — uma única convolução bloqueia a vizinhança
        dil_kernel = torch.ones(1, 1, 2*r+1, 2*r+1, device=DEVICE)
        blocked = F.conv2d(occupied, dil_kernel, padding=r).squeeze(1) > 0  # (B, S, S)

        # Margem de borda: pixels muito próximos da borda não podem hospedar cristais
        border_mask = torch.zeros(self.batch_size, self.size, self.size, dtype=torch.bool, device=DEVICE)
        border_mask[:, :r, :]      = True
        border_mask[:, self.size-r:, :] = True
        border_mask[:, :, :r]      = True
        border_mask[:, :, self.size-r:] = True

        # Score válido: candidato E não bloqueado E dentro da borda
        valid_scores = env_mean * candidates_mask.float() * (~blocked).float() * (~border_mask).float()
        flat_scores  = valid_scores.reshape(self.batch_size, -1)  # (B, S*S)

        any_valid = flat_scores.any(dim=1)
        if not any_valid.any():
            return

        # Top-K global por batch — completamente na GPU
        MAX_NEW = 10
        k_query  = min(MAX_NEW * 4, flat_scores.shape[1])  # margem para iteração greedy
        topk_vals, topk_idx = torch.topk(flat_scores, k=k_query, dim=1, largest=True, sorted=True)

        # Gravar padrões na GPU
        pw = self.size + 2 * r
        padded      = F.pad(self.field.unsqueeze(1), (r, r, r, r),
                            mode='constant', value=0).squeeze(1)   # (B, pw, pw)
        padded_flat = padded.reshape(self.batch_size, -1)           # (B, pw*pw)

        topk_idx_cpu  = topk_idx.cpu().numpy()   # (B, k_query)
        topk_vals_cpu = topk_vals.cpu().numpy()  # (B, k_query)

        for b in range(self.batch_size):
            if not any_valid[b]:
                continue

            cc      = self.crystal_count[b]
            formed  = 0
            # occupied_local rastreia posições aceitas nesta rodada (para exclusão entre novos)
            new_positions: list[tuple[int,int]] = []

            for k in range(k_query):
                if topk_vals_cpu[b, k] <= 0:
                    break
                if formed >= MAX_NEW or cc >= self.max_crystals:
                    break

                lin    = int(topk_idx_cpu[b, k])
                px, py = divmod(lin, self.size)

                # Verificar exclusão em relação aos cristais aceitos nesta rodada
                too_close = any(
                    (px - qx)**2 + (py - qy)**2 < r*r
                    for qx, qy in new_positions
                )
                if too_close:
                    continue

                # Gravar cristal
                slot = cc + formed
                lin_idx = (px + self._off_x.cpu()) * pw + (py + self._off_y.cpu())
                patch   = padded_flat[b, lin_idx.to(DEVICE)]

                self.crystal_patterns[b, slot]     = patch.reshape(ps, ps)
                self.crystal_positions[b, slot, 0] = px
                self.crystal_positions[b, slot, 1] = py
                self.crystal_active[b, slot]        = 1.0
                self.crystal_freqs[b, slot]         = self.wave_params[b, :, 1].mean()

                new_positions.append((px, py))
                formed += 1

            self.crystal_count[b] = cc + formed

    def read_at(self, positions):
        pos_t = torch.tensor(positions, dtype=torch.float32)
        ix = torch.clamp((pos_t[:, 0] * (self.size - 1)).long(), 0, self.size - 1)
        iy = torch.clamp((pos_t[:, 1] * (self.size - 1)).long(), 0, self.size - 1)
        return self.field[:, ix, iy]  # (B, P)

READ_POSITIONS = [
    (0.2, 0.3), (0.2, 0.5), (0.2, 0.7),
    (0.4, 0.3), (0.4, 0.5), (0.4, 0.7),
    (0.6, 0.3), (0.6, 0.5), (0.6, 0.7),
    (0.8, 0.3), (0.8, 0.5), (0.8, 0.7),
    (0.5, 0.2), (0.5, 0.5), (0.5, 0.8),
    (0.7, 0.4),  # assimétrico — discrimina 1011 vs 1110
]
N_READ = len(READ_POSITIONS)  # 16


def _build_wave_params(emitter_exp, stimulus_idx, bit_values,
                       pos_x, pos_y_0, pos_y_1, total_batch):
    selected = emitter_exp[torch.arange(total_batch, device=DEVICE), stimulus_idx, bit_values]
    # selected: (total_batch, 2, 2)

    amp  = torch.abs(selected[:, :, 0]) * 1.5 + 2.0
    freq = (torch.abs(selected[:, :, 1]) * 2.0
            + torch.where(bit_values.unsqueeze(1) == 0,
                          torch.tensor(2.0, device=DEVICE),
                          torch.tensor(7.0, device=DEVICE)))

    wp = torch.zeros(total_batch, 2, 6, device=DEVICE)
    wp[:, :, 0] = amp
    wp[:, :, 1] = freq
    wp[:, :, 3] = 0.001
    wp[:, :, 4] = pos_x
    wp[:, 0, 5] = pos_y_0
    wp[:, 1, 5] = pos_y_1
    return wp

def evaluate_population_parity3(candidates, n_emitter, n_decoder,
                                 disable_crystals=False,
                                 time_cfg: TimeConfig = None):
    if time_cfg is None:
        time_cfg = PARITY_3BIT_TIME

    pop_size = candidates.shape[0]
    n_inputs = 8

    # Constantes físicas fixas — fazem parte da arquitetura
    emitter_all = candidates[:, :n_emitter].view(pop_size, 3, 2, 2, 2)  # (P, 3, 2, 2, 2)

    all_losses = np.zeros(pop_size, dtype=np.float32)
    all_preds  = np.zeros((pop_size, n_inputs), dtype=np.float32)

    targets_gpu = PARITY_3BIT_DATA[:, 3]  # (8,)
    on_mask      = torch.ones(n_inputs, 2, 1, 1, device=DEVICE)
    silence_mask = torch.zeros(n_inputs, 2, 1, 1, device=DEVICE)

    for i in range(pop_size):
        # Constantes físicas fixas da arquitetura 3-bit
        psi = PsiField(batch_size=n_inputs,
                       damping=PSI_GAMMA_3BIT,
                       dissipation=PSI_BETA_3BIT,
                       disable_crystals=disable_crystals,
                       sigma=PSI_SIGMA_3BIT)

        emitter_i = emitter_all[i].unsqueeze(0).expand(n_inputs, -1, -1, -1, -1)  # (8, 3, 2, 2, 2)

        s1_bits = PARITY_3BIT_DATA[:, 0].long()
        s2_bits = PARITY_3BIT_DATA[:, 1].long()
        s3_bits = PARITY_3BIT_DATA[:, 2].long()

        s1_wp = _build_wave_params(emitter_i, 0, s1_bits, 0.2, 0.4, 0.6, n_inputs)
        s2_wp = _build_wave_params(emitter_i, 1, s2_bits, 0.5, 0.4, 0.6, n_inputs)
        s3_wp = _build_wave_params(emitter_i, 2, s3_bits, 0.8, 0.4, 0.6, n_inputs)

        for step in range(time_cfg.read):
            if time_cfg.s1_start <= step <= time_cfg.s1_end:
                psi.wave_params = s1_wp;  psi.wave_mask = on_mask
            elif time_cfg.s2_start <= step <= time_cfg.s2_end:
                psi.wave_params = s2_wp;  psi.wave_mask = on_mask
            elif time_cfg.s3_start <= step <= time_cfg.s3_end:
                psi.wave_params = s3_wp;  psi.wave_mask = on_mask
            else:
                psi.wave_mask = silence_mask
            psi.step()

        weights_i = candidates[i, n_emitter:n_emitter + n_decoder].unsqueeze(0).expand(n_inputs, -1)
        values    = psi.read_at(READ_POSITIONS)             # (8, N_READ)
        logits    = torch.sum(values * weights_i, dim=1)    # (8,)
        preds     = torch.sigmoid(logits)                   # (8,)

        eps = 1e-7
        p   = torch.clamp(preds, eps, 1 - eps)
        bce = -(targets_gpu * torch.log(p) + (1 - targets_gpu) * torch.log(1 - p))

        all_losses[i] = bce.sum().item()
        all_preds[i]  = preds.detach().cpu().numpy()

    return all_losses, all_preds

def evaluate_emitter_only(emitter_params, n_emitter, time_cfg, disable_crystals=False):

    n_inputs = 8
    candidates = torch.tensor(emitter_params[np.newaxis], dtype=torch.float32, device=DEVICE)
    emitter_i  = candidates[:, :n_emitter].view(1, 3, 2, 2, 2)
    emitter_i  = emitter_i.expand(n_inputs, -1, -1, -1, -1)

    psi = PsiField(batch_size=n_inputs,
                   damping=PSI_GAMMA_3BIT,
                   dissipation=PSI_BETA_3BIT,
                   disable_crystals=disable_crystals,
                   sigma=PSI_SIGMA_3BIT)

    on_mask      = torch.ones(n_inputs, 2, 1, 1, device=DEVICE)
    silence_mask = torch.zeros(n_inputs, 2, 1, 1, device=DEVICE)

    s1_bits = PARITY_3BIT_DATA[:, 0].long()
    s2_bits = PARITY_3BIT_DATA[:, 1].long()
    s3_bits = PARITY_3BIT_DATA[:, 2].long()

    s1_wp = _build_wave_params(emitter_i, 0, s1_bits, 0.2, 0.4, 0.6, n_inputs)
    s2_wp = _build_wave_params(emitter_i, 1, s2_bits, 0.5, 0.4, 0.6, n_inputs)
    s3_wp = _build_wave_params(emitter_i, 2, s3_bits, 0.8, 0.4, 0.6, n_inputs)

    for step in range(time_cfg.read):
        if time_cfg.s1_start <= step <= time_cfg.s1_end:
            psi.wave_params = s1_wp; psi.wave_mask = on_mask
        elif time_cfg.s2_start <= step <= time_cfg.s2_end:
            psi.wave_params = s2_wp; psi.wave_mask = on_mask
        elif time_cfg.s3_start <= step <= time_cfg.s3_end:
            psi.wave_params = s3_wp; psi.wave_mask = on_mask
        else:
            psi.wave_mask = silence_mask
        psi.step()

    # Lê os padrões de campo para as 8 combinações
    values = psi.read_at(READ_POSITIONS)  # (8, N_READ)

    # Métrica: maximizar separabilidade entre classes opostas (paridade 0 vs 1)
    targets = PARITY_3BIT_DATA[:, 3]  # (8,)
    pos_mask = targets > 0.5  # combinações com paridade 1
    neg_mask = ~pos_mask       # combinações com paridade 0

    pos_mean = values[pos_mask].mean(dim=0)   # (N_READ,)
    neg_mean = values[neg_mask].mean(dim=0)   # (N_READ,)

    # Distância entre centróides das duas classes — maximizar = minimizar negativo
    separation = torch.norm(pos_mean - neg_mean)

    # Variância intra-classe — minimizar
    pos_var = values[pos_mask].var(dim=0).mean()
    neg_var = values[neg_mask].var(dim=0).mean()

    # Loss: queremos separação grande e variância pequena
    loss = -separation.item() + (pos_var + neg_var).item()
    return float(loss), values.detach().cpu().numpy()


def train_parity3_vectorized(label="PARIDADE 3 BITS",
                              disable_crystals=False,
                              time_cfg: TimeConfig = None,
                              max_gen=400, pop_size=20):
    if time_cfg is None:
        time_cfg = PARITY_3BIT_TIME

    n_emitter = 24
    n_decoder = N_READ  # 16
    n_total   = n_emitter + n_decoder

    crystal_tag = " [SEM CRISTAIS]" if disable_crystals else ""
    print(f"\n{'='*55}")
    print(f"  {label}{crystal_tag}")
    print(f"{'='*55}")
    print(f"  Params: {n_emitter}(E) + {n_decoder}(D) = {n_total} [γ,β,σ fixos]")
    print(f"  γ={PSI_GAMMA_3BIT:.6f} β={PSI_BETA_3BIT:.6f} σ={PSI_SIGMA_3BIT:.6f}")
    print(f"  Otimização em 2 fases: Emitter → Decoder")
    print()

    # ── FASE 1: otimizar emitter ──────────────────────────────────────
    print(f"  FASE 1 — Emitter (separabilidade do campo)")
    print(f"  {'-'*45}")

    x0_e = np.random.randn(n_emitter) * 0.3
    es_e = cma.CMAEvolutionStrategy(
        x0_e, 0.5,
        {'popsize': pop_size, 'seed': 42, 'maxiter': 150, 'verbose': -9}
    )

    start         = time.time()
    best_e_loss   = float('inf')
    best_e_params = None

    gen = 0
    while not es_e.stop():
        gen += 1
        candidates_np = np.array(es_e.ask())
        losses = []
        for c in candidates_np:
            l, _ = evaluate_emitter_only(c, n_emitter, time_cfg, disable_crystals)
            losses.append(l)
        losses = np.array(losses, dtype=np.float32)
        es_e.tell(candidates_np.tolist(), losses.tolist())

        idx = int(np.argmin(losses))
        if losses[idx] < best_e_loss:
            best_e_loss   = float(losses[idx])
            best_e_params = candidates_np[idx].copy()

        if gen % 20 == 0 or gen == 1:
            elapsed = time.time() - start
            print(f"  Gen {gen:>4} | Sep-Loss: {best_e_loss:.4f} | {elapsed:.0f}s")

    print(f"\n  Emitter fixado. Loss separabilidade: {best_e_loss:.4f}")
    print(f"  Tempo fase 1: {time.time()-start:.0f}s")

    # ── FASE 2: otimizar decoder com emitter fixo ─────────────────────
    print(f"\n  FASE 2 — Decoder (classificação)")
    print(f"  {'-'*45}")

    targets_list = [t for (*_, t) in PARITY_3BIT]
    targets_gpu  = PARITY_3BIT_DATA[:, 3]
    n_inputs     = 8

    def eval_decoder(decoder_params):
        emitter_t = torch.tensor(best_e_params[np.newaxis], dtype=torch.float32, device=DEVICE)
        emitter_i = emitter_t[:, :n_emitter].view(1, 3, 2, 2, 2).expand(n_inputs, -1, -1, -1, -1)

        psi = PsiField(batch_size=n_inputs,
                       damping=PSI_GAMMA_3BIT,
                       dissipation=PSI_BETA_3BIT,
                       disable_crystals=disable_crystals,
                       sigma=PSI_SIGMA_3BIT)

        on_mask      = torch.ones(n_inputs, 2, 1, 1, device=DEVICE)
        silence_mask = torch.zeros(n_inputs, 2, 1, 1, device=DEVICE)

        s1_bits = PARITY_3BIT_DATA[:, 0].long()
        s2_bits = PARITY_3BIT_DATA[:, 1].long()
        s3_bits = PARITY_3BIT_DATA[:, 2].long()

        s1_wp = _build_wave_params(emitter_i, 0, s1_bits, 0.2, 0.4, 0.6, n_inputs)
        s2_wp = _build_wave_params(emitter_i, 1, s2_bits, 0.5, 0.4, 0.6, n_inputs)
        s3_wp = _build_wave_params(emitter_i, 2, s3_bits, 0.8, 0.4, 0.6, n_inputs)

        for step in range(time_cfg.read):
            if time_cfg.s1_start <= step <= time_cfg.s1_end:
                psi.wave_params = s1_wp; psi.wave_mask = on_mask
            elif time_cfg.s2_start <= step <= time_cfg.s2_end:
                psi.wave_params = s2_wp; psi.wave_mask = on_mask
            elif time_cfg.s3_start <= step <= time_cfg.s3_end:
                psi.wave_params = s3_wp; psi.wave_mask = on_mask
            else:
                psi.wave_mask = silence_mask
            psi.step()

        weights = torch.tensor(decoder_params, dtype=torch.float32, device=DEVICE)
        weights = weights.unsqueeze(0).expand(n_inputs, -1)
        values  = psi.read_at(READ_POSITIONS)
        logits  = torch.sum(values * weights, dim=1)
        preds   = torch.sigmoid(logits)

        eps = 1e-7
        p   = torch.clamp(preds, eps, 1 - eps)
        bce = -(targets_gpu * torch.log(p) + (1 - targets_gpu) * torch.log(1 - p))
        correct = sum(1 for pr, t in zip(preds.detach().cpu().numpy(), targets_list)
                      if (pr > 0.5) == (t > 0.5))
        return float(bce.sum().item()), preds.detach().cpu().numpy(), correct

    x0_d = np.random.randn(n_decoder) * 0.3
    es_d = cma.CMAEvolutionStrategy(
        x0_d, 0.5,
        {'popsize': pop_size, 'seed': 42, 'maxiter': max_gen, 'verbose': -9}
    )

    start2        = time.time()
    best_loss     = float('inf')
    best_preds    = None
    best_d_params = None
    history_correct = []
    refining      = False

    gen = 0
    while not es_d.stop():
        gen += 1
        candidates_np = np.array(es_d.ask())

        if refining and best_d_params is not None:
            candidates_np[-1] = best_d_params.copy()

        losses  = []
        preds_m = []
        corrects = []
        for c in candidates_np:
            l, p, cor = eval_decoder(c)
            losses.append(l)
            preds_m.append(p)
            corrects.append(cor)

        losses = np.array(losses, dtype=np.float32)
        es_d.tell(candidates_np.tolist(), losses.tolist())

        idx           = int(np.argmin(losses))
        gen_best_loss = float(losses[idx])
        correct       = corrects[idx]
        history_correct.append(correct)

        if gen_best_loss < best_loss:
            best_loss   = gen_best_loss
            best_preds  = preds_m[idx]
            if not refining or correct == 8:
                best_d_params = candidates_np[idx].copy()

        if correct == 8 and not refining:
            es_d.sigma *= 0.3
            es_d.mean   = best_d_params.copy()
            refining    = True
            print(f"\n  ★ 8/8 na gen {gen}! Refinamento ativado. σ → {es_d.sigma:.4f}")

        if gen % 10 == 0 or gen == 1:
            elapsed = time.time() - start2
            mode_tag = f" [refinando σ={es_d.sigma:.3f}]" if refining else ""
            print(f"  Gen {gen:>4} | Loss: {gen_best_loss:.4f} | {correct}/8{mode_tag} | {elapsed:.0f}s")

        if refining and len(history_correct) >= 10:
            if all(c == 8 for c in history_correct[-10:]):
                print(f"\n  → Solução estável por 10 gerações. Convergido.")
                break

    elapsed = time.time() - start
    correct = sum(1 for p, t in zip(best_preds, targets_list)
                  if (p > 0.5) == (t > 0.5))

    print(f"\n  RESULTADO {label}:")
    print(f"  Loss: {best_loss:.4f} | {correct}/8 | Total: {elapsed:.1f}s")

    if best_d_params is not None:
        full_params = np.concatenate([best_e_params, best_d_params])
        _, preds_final = evaluate_population_parity3(
            torch.tensor(full_params[np.newaxis], dtype=torch.float32, device=DEVICE),
            n_emitter, n_decoder,
            disable_crystals=disable_crystals, time_cfg=time_cfg,
        )
        fp = preds_final[0]
        print(f"  Detalhes:")
        for (s1, s2, s3, t), p in zip(PARITY_3BIT, fp):
            s = "✓" if (p > 0.5) == (t > 0.5) else "✗"
            print(f"    S1={s1} S2={s2} S3={s3} → {p:.4f} (target {t}) {s}")

    return correct, best_loss, [], [], elapsed

def evaluate_population_parity4(candidates, n_emitter, n_decoder,
                                  disable_crystals=False):

    pop_size = candidates.shape[0]
    n_inputs = 16

    emitter_all = candidates[:, :n_emitter].view(pop_size, 4, 2, 2, 2)  # (P, 4, 2, 2, 2)

    all_losses = np.zeros(pop_size, dtype=np.float32)
    all_preds  = np.zeros((pop_size, n_inputs), dtype=np.float32)

    targets_gpu  = PARITY_4BIT_DATA[:, 4]  # (16,)
    on_mask      = torch.ones(n_inputs, 2, 1, 1, device=DEVICE)
    silence_mask = torch.zeros(n_inputs, 2, 1, 1, device=DEVICE)

    bits_per_stimulus = [
        PARITY_4BIT_DATA[:, 0].long(),
        PARITY_4BIT_DATA[:, 1].long(),
        PARITY_4BIT_DATA[:, 2].long(),
        PARITY_4BIT_DATA[:, 3].long(),
    ]
    y_positions = [0.2, 0.4, 0.6, 0.8]

    STIM_ON      = 40
    STIM_SILENCE = 40
    STIM_TOTAL   = STIM_ON + STIM_SILENCE  # 80 steps por bit
    READ_STEP    = 40  # steps extras de silêncio após S4 antes da leitura

    for i in range(pop_size):
        emitter_i = emitter_all[i].unsqueeze(0).expand(n_inputs, -1, -1, -1, -1)  # (16, 4, 2, 2, 2)
        weights_i = candidates[i, n_emitter:n_emitter + n_decoder].unsqueeze(0).expand(n_inputs, -1)

        # Campo único persistente — NÃO reseta entre estímulos
        psi = PsiField(batch_size=n_inputs,
                       damping=PSI_GAMMA_3BIT,
                       dissipation=PSI_BETA_3BIT,
                       disable_crystals=disable_crystals,
                       sigma=PSI_SIGMA_3BIT)

        for k in range(4):
            wp = _build_wave_params(emitter_i, k, bits_per_stimulus[k],
                                    y_positions[k], 0.425, 0.575, n_inputs)
            for step in range(STIM_TOTAL):
                if step < STIM_ON:
                    psi.wave_params = wp
                    psi.wave_mask   = on_mask
                else:
                    psi.wave_mask = silence_mask
                psi.step()

        # Silêncio extra antes da leitura
        for _ in range(READ_STEP):
            psi.wave_mask = silence_mask
            psi.step()

        values = psi.read_at(READ_POSITIONS)
        logits = torch.sum(values * weights_i, dim=1)
        preds  = torch.sigmoid(logits)

        eps = 1e-7
        p   = torch.clamp(preds, eps, 1 - eps)
        bce = -(targets_gpu * torch.log(p) + (1 - targets_gpu) * torch.log(1 - p))

        all_losses[i] = bce.sum().item()
        all_preds[i]  = preds.detach().cpu().numpy()

    return all_losses, all_preds


def train_parity4_vectorized(label="PARIDADE 4 BITS",
                              disable_crystals=False,
                              max_gen=400, pop_size=40):
    n_emitter = 32      # (4, 2, 2, 2)
    n_decoder = N_READ  # 16
    n_total   = n_emitter + n_decoder  # física fixada nas constantes 3-bit

    crystal_tag = " [SEM CRISTAIS]" if disable_crystals else ""
    print(f"\n{'='*58}")
    print(f"  {label}{crystal_tag}")
    print(f"{'='*58}")
    print(f"  Params: {n_emitter}(E) + {n_decoder}(D) = {n_total}")
    print(f"  Física: γ={PSI_GAMMA_3BIT} β={PSI_BETA_3BIT} σ={PSI_SIGMA_3BIT} [fixos]")
    print(f"  Modo: campo PERSISTENTE entre S1→S2→S3→S4, leitura após 40 steps silêncio")
    print(f"  Batch efetivo por geração: {pop_size} cand × 16 × 4 estímulos")
    print()

    x0 = np.random.randn(n_total) * 0.3
    es = cma.CMAEvolutionStrategy(
        x0, 0.5,
        {'popsize': pop_size, 'seed': 42, 'maxiter': max_gen, 'verbose': -9}
    )

    start           = time.time()
    best_loss       = float('inf')
    best_preds      = None
    best_params     = None
    history_loss    = []
    history_preds   = []
    history_correct = []
    targets_list    = [t for (*_, t) in PARITY_4BIT]
    refining        = False

    gen = 0
    while not es.stop():
        gen += 1
        candidates_np = np.array(es.ask())

        # ELITISMO: só ativo após refinamento (16/16 atingido)
        if refining and best_params is not None:
            candidates_np[-1] = best_params.copy()

        candidates_t = torch.tensor(candidates_np, dtype=torch.float32, device=DEVICE)

        losses, preds_matrix = evaluate_population_parity4(
            candidates_t, n_emitter, n_decoder,
            disable_crystals=disable_crystals,
        )

        es.tell(candidates_np.tolist(), losses.tolist())

        idx            = int(np.argmin(losses))
        gen_best_loss  = float(losses[idx])
        gen_best_preds = preds_matrix[idx]
        correct        = sum(1 for p, t in zip(gen_best_preds, targets_list)
                             if (p > 0.5) == (t > 0.5))

        history_loss.append(gen_best_loss)
        history_preds.append(gen_best_preds.tolist())
        history_correct.append(correct)

        if gen_best_loss < best_loss:
            best_loss  = gen_best_loss
            best_preds = gen_best_preds
            # Após refinamento: só substitui best_params se mantiver 16/16
            if not refining or correct == 16:
                best_params = candidates_np[idx].copy()

        # 3. GATILHO DE REFINAMENTO: primeira vez que atinge 16/16
        if correct == 16 and not refining:
            es.sigma *= 0.3
            es.mean   = best_params.copy()
            refining  = True
            print(f"\n  ★ 16/16 na gen {gen}! Refinamento ativado.")
            print(f"    σ → {es.sigma:.4f} | média ancorada no vetor vencedor")

        if gen % 20 == 0 or gen == 1:
            elapsed  = time.time() - start
            mode_tag = f" [refinando σ={es.sigma:.3f}]" if refining else ""
            print(f"  Gen {gen:>4} | Loss: {gen_best_loss:.4f} | "
                  f"{correct}/16{mode_tag} | {elapsed:.0f}s")

        # 4. CRITÉRIO DE CONVERGÊNCIA: 10 gerações consecutivas com 16/16
        if refining and len(history_correct) >= 10:
            if all(c == 16 for c in history_correct[-10:]):
                print(f"\n  → Solução estável por 10 gerações consecutivas. Convergido.")
                break

    elapsed = time.time() - start
    correct = sum(1 for p, t in zip(best_preds, targets_list)
                  if (p > 0.5) == (t > 0.5))

    print(f"\n  RESULTADO {label}:")
    print(f"  Loss: {best_loss:.4f} | {correct}/16 | {elapsed:.1f}s")

    if best_params is not None:
        _, preds_final = evaluate_population_parity4(
            torch.tensor(best_params[np.newaxis], dtype=torch.float32, device=DEVICE),
            n_emitter, n_decoder,
            disable_crystals=disable_crystals,
        )
        fp = preds_final[0]
        print(f"  Detalhes:")
        for (s1, s2, s3, s4, t), p in zip(PARITY_4BIT, fp):
            s = "✓" if (p > 0.5) == (t > 0.5) else "✗"
            print(f"    {s1}{s2}{s3}{s4}→{t}  {p:.3f} {s}")

    return correct, best_loss, history_loss, history_preds, elapsed

def log_crystals_4bit(emitter_params, decoder_params, n_emitter, n_decoder):
    """
    Roda o campo com o melhor candidato e loga os cristais formados.
    Retorna dict com estatísticas dos cristais.
    """
    n_inputs = 16
    STIM_ON    = 40
    STIM_TOTAL = 80
    READ_STEP  = 40

    emitter_t = torch.tensor(emitter_params[np.newaxis], dtype=torch.float32, device=DEVICE)
    emitter_i = emitter_t[:, :n_emitter].view(1, 4, 2, 2, 2).expand(n_inputs, -1, -1, -1, -1)
    weights   = torch.tensor(decoder_params, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(n_inputs, -1)

    bits_per_stimulus = [
        PARITY_4BIT_DATA[:, 0].long(),
        PARITY_4BIT_DATA[:, 1].long(),
        PARITY_4BIT_DATA[:, 2].long(),
        PARITY_4BIT_DATA[:, 3].long(),
    ]
    y_positions = [0.2, 0.4, 0.6, 0.8]

    on_mask      = torch.ones(n_inputs, 2, 1, 1, device=DEVICE)
    silence_mask = torch.zeros(n_inputs, 2, 1, 1, device=DEVICE)

    psi = PsiField(batch_size=n_inputs,
                   damping=PSI_GAMMA_3BIT,
                   dissipation=PSI_BETA_3BIT,
                   sigma=PSI_SIGMA_3BIT)

    for k in range(4):
        wp = _build_wave_params(emitter_i, k, bits_per_stimulus[k],
                                y_positions[k], 0.425, 0.575, n_inputs)
        for step in range(STIM_TOTAL):
            if step < STIM_ON:
                psi.wave_params = wp
                psi.wave_mask   = on_mask
            else:
                psi.wave_mask = silence_mask
            psi.step()

    for _ in range(READ_STEP):
        psi.wave_mask = silence_mask
        psi.step()

    # Coleta cristais de todos os itens do batch
    total_crystals = sum(psi.crystal_count)
    all_positions  = []
    for b in range(n_inputs):
        cc = psi.crystal_count[b]
        for c in range(cc):
            px = int(psi.crystal_positions[b, c, 0].item())
            py = int(psi.crystal_positions[b, c, 1].item())
            all_positions.append((b, px, py))

    print(f"\n  CRISTAIS FORMADOS:")
    print(f"  Total: {total_crystals} cristais em {n_inputs} combinações de entrada")
    print(f"  Média: {total_crystals/n_inputs:.1f} cristais por combinação")
    counts = [psi.crystal_count[b] for b in range(n_inputs)]
    print(f"  Min: {min(counts)}  Max: {max(counts)}  por combinação")
    print(f"\n  Distribuição por combinação:")
    for b, (s1, s2, s3, s4, t) in enumerate(PARITY_4BIT):
        print(f"    {s1}{s2}{s3}{s4}→{t}: {counts[b]} cristais")

    # Mapa de densidade no grid 48x48
    grid = np.zeros((48, 48), dtype=int)
    for (b, px, py) in all_positions:
        grid[px, py] += 1

    occupied_cells = int((grid > 0).sum())
    print(f"\n  Células ocupadas no grid 48×48: {occupied_cells}/2304 ({occupied_cells/2304*100:.1f}%)")

    return {
        'total': total_crystals,
        'media': total_crystals / n_inputs,
        'counts': counts,
        'positions': all_positions,
        'grid': grid,
        'occupied_cells': occupied_cells,
    }


def evaluate_emitter_only_4bit(emitter_params, n_emitter, disable_crystals=False):
    """
    Fase 1 do 4-bit: avalia separabilidade do emitter para 16 combinações.
    Campo persistente entre os 4 estímulos.
    """
    n_inputs   = 16
    STIM_ON    = 40
    STIM_TOTAL = 80
    READ_STEP  = 40

    emitter_t = torch.tensor(emitter_params[np.newaxis], dtype=torch.float32, device=DEVICE)
    emitter_i = emitter_t[:, :n_emitter].view(1, 4, 2, 2, 2).expand(n_inputs, -1, -1, -1, -1)

    bits_per_stimulus = [
        PARITY_4BIT_DATA[:, 0].long(),
        PARITY_4BIT_DATA[:, 1].long(),
        PARITY_4BIT_DATA[:, 2].long(),
        PARITY_4BIT_DATA[:, 3].long(),
    ]
    y_positions = [0.2, 0.4, 0.6, 0.8]

    on_mask      = torch.ones(n_inputs, 2, 1, 1, device=DEVICE)
    silence_mask = torch.zeros(n_inputs, 2, 1, 1, device=DEVICE)

    psi = PsiField(batch_size=n_inputs,
                   damping=PSI_GAMMA_3BIT,
                   dissipation=PSI_BETA_3BIT,
                   disable_crystals=disable_crystals,
                   sigma=PSI_SIGMA_3BIT)

    for k in range(4):
        wp = _build_wave_params(emitter_i, k, bits_per_stimulus[k],
                                y_positions[k], 0.425, 0.575, n_inputs)
        for step in range(STIM_TOTAL):
            if step < STIM_ON:
                psi.wave_params = wp
                psi.wave_mask   = on_mask
            else:
                psi.wave_mask = silence_mask
            psi.step()

    for _ in range(READ_STEP):
        psi.wave_mask = silence_mask
        psi.step()

    values  = psi.read_at(READ_POSITIONS)  # (16, N_READ)
    targets = PARITY_4BIT_DATA[:, 4]
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask

    pos_mean = values[pos_mask].mean(dim=0)
    neg_mean = values[neg_mask].mean(dim=0)
    separation = torch.norm(pos_mean - neg_mean)
    pos_var = values[pos_mask].var(dim=0).mean()
    neg_var = values[neg_mask].var(dim=0).mean()

    loss = -separation.item() + (pos_var + neg_var).item()
    return float(loss)


def train_parity4_2fases(label="PARIDADE 4 BITS — 2 Fases",
                          disable_crystals=False,
                          max_gen=300, pop_size=20):
    n_emitter = 32
    n_decoder = N_READ  # 16

    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  Params: {n_emitter}(E) + {n_decoder}(D) = {n_emitter+n_decoder}")
    print(f"  Física: γ={PSI_GAMMA_3BIT} β={PSI_BETA_3BIT} σ={PSI_SIGMA_3BIT} [fixos]")
    print(f"  Modo: campo PERSISTENTE + otimização em 2 fases")
    print()

    # ── FASE 1: emitter ──────────────────────────────────────────────
    print(f"  FASE 1 — Emitter (separabilidade do campo, 16 classes)")
    print(f"  {'-'*50}")

    x0_e = np.random.randn(n_emitter) * 0.3
    es_e = cma.CMAEvolutionStrategy(
        x0_e, 0.5,
        {'popsize': pop_size, 'seed': 42, 'maxiter': 150, 'verbose': -9}
    )

    start         = time.time()
    best_e_loss   = float('inf')
    best_e_params = None

    gen = 0
    while not es_e.stop():
        gen += 1
        candidates_np = np.array(es_e.ask())
        losses = [evaluate_emitter_only_4bit(c, n_emitter, disable_crystals) for c in candidates_np]
        losses = np.array(losses, dtype=np.float32)
        es_e.tell(candidates_np.tolist(), losses.tolist())

        idx = int(np.argmin(losses))
        if losses[idx] < best_e_loss:
            best_e_loss   = float(losses[idx])
            best_e_params = candidates_np[idx].copy()

        if gen % 20 == 0 or gen == 1:
            print(f"  Gen {gen:>4} | Sep-Loss: {best_e_loss:.4f} | {time.time()-start:.0f}s")

    print(f"\n  Emitter fixado. Sep-Loss: {best_e_loss:.4f} | Tempo: {time.time()-start:.0f}s")

    # ── FASE 2: decoder ───────────────────────────────────────────────
    print(f"\n  FASE 2 — Decoder (classificação, 16 combinações)")
    print(f"  {'-'*50}")

    targets_list = [t for (*_, t) in PARITY_4BIT]
    targets_gpu  = PARITY_4BIT_DATA[:, 4]
    n_inputs     = 16
    STIM_ON      = 40
    STIM_TOTAL   = 80
    READ_STEP    = 40

    bits_per_stimulus = [
        PARITY_4BIT_DATA[:, 0].long(),
        PARITY_4BIT_DATA[:, 1].long(),
        PARITY_4BIT_DATA[:, 2].long(),
        PARITY_4BIT_DATA[:, 3].long(),
    ]
    y_positions  = [0.2, 0.4, 0.6, 0.8]
    on_mask      = torch.ones(n_inputs, 2, 1, 1, device=DEVICE)
    silence_mask = torch.zeros(n_inputs, 2, 1, 1, device=DEVICE)
    emitter_t    = torch.tensor(best_e_params[np.newaxis], dtype=torch.float32, device=DEVICE)
    emitter_i    = emitter_t[:, :n_emitter].view(1, 4, 2, 2, 2).expand(n_inputs, -1, -1, -1, -1)

    def eval_decoder_4bit(decoder_params):
        psi = PsiField(batch_size=n_inputs,
                       damping=PSI_GAMMA_3BIT,
                       dissipation=PSI_BETA_3BIT,
                       disable_crystals=disable_crystals,
                       sigma=PSI_SIGMA_3BIT)

        for k in range(4):
            wp = _build_wave_params(emitter_i, k, bits_per_stimulus[k],
                                    y_positions[k], 0.425, 0.575, n_inputs)
            for step in range(STIM_TOTAL):
                if step < STIM_ON:
                    psi.wave_params = wp
                    psi.wave_mask   = on_mask
                else:
                    psi.wave_mask = silence_mask
                psi.step()

        for _ in range(READ_STEP):
            psi.wave_mask = silence_mask
            psi.step()

        weights = torch.tensor(decoder_params, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(n_inputs, -1)
        values  = psi.read_at(READ_POSITIONS)
        logits  = torch.sum(values * weights, dim=1)
        preds   = torch.sigmoid(logits)

        eps = 1e-7
        p   = torch.clamp(preds, eps, 1 - eps)
        bce = -(targets_gpu * torch.log(p) + (1 - targets_gpu) * torch.log(1 - p))
        correct = sum(1 for pr, t in zip(preds.detach().cpu().numpy(), targets_list)
                      if (pr > 0.5) == (t > 0.5))
        return float(bce.sum().item()), preds.detach().cpu().numpy(), correct

    x0_d = np.random.randn(n_decoder) * 0.3
    es_d = cma.CMAEvolutionStrategy(
        x0_d, 0.5,
        {'popsize': pop_size, 'seed': 42, 'maxiter': max_gen, 'verbose': -9}
    )

    start2          = time.time()
    best_loss       = float('inf')
    best_preds      = None
    best_d_params   = None
    history_correct = []
    refining        = False

    gen = 0
    while not es_d.stop():
        gen += 1
        candidates_np = np.array(es_d.ask())

        if refining and best_d_params is not None:
            candidates_np[-1] = best_d_params.copy()

        losses, preds_m, corrects = [], [], []
        for c in candidates_np:
            l, p, cor = eval_decoder_4bit(c)
            losses.append(l); preds_m.append(p); corrects.append(cor)

        losses = np.array(losses, dtype=np.float32)
        es_d.tell(candidates_np.tolist(), losses.tolist())

        idx           = int(np.argmin(losses))
        gen_best_loss = float(losses[idx])
        correct       = corrects[idx]
        history_correct.append(correct)

        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_preds = preds_m[idx]
            if not refining or correct == 16:
                best_d_params = candidates_np[idx].copy()

        if correct == 16 and not refining:
            es_d.sigma *= 0.3
            es_d.mean   = best_d_params.copy()
            refining    = True
            print(f"\n  ★ 16/16 na gen {gen}! Refinamento ativado. σ → {es_d.sigma:.4f}")

        if gen % 20 == 0 or gen == 1:
            elapsed  = time.time() - start2
            mode_tag = f" [refinando σ={es_d.sigma:.3f}]" if refining else ""
            print(f"  Gen {gen:>4} | Loss: {gen_best_loss:.4f} | {correct}/16{mode_tag} | {elapsed:.0f}s")

        if refining and len(history_correct) >= 10:
            if all(c == 16 for c in history_correct[-10:]):
                print(f"\n  → Solução estável por 10 gerações. Convergido.")
                break

    elapsed = time.time() - start
    correct = sum(1 for p, t in zip(best_preds, targets_list)
                  if (p > 0.5) == (t > 0.5))

    print(f"\n  RESULTADO {label}:")
    print(f"  Loss: {best_loss:.4f} | {correct}/16 | Total: {elapsed:.1f}s")
    print(f"  Detalhes:")
    for (s1, s2, s3, s4, t), p in zip(PARITY_4BIT, best_preds):
        s = "✓" if (p > 0.5) == (t > 0.5) else "✗"
        print(f"    {s1}{s2}{s3}{s4}→{t}  {p:.3f} {s}")

    # Log cristais do melhor candidato
    if best_d_params is not None:
        crystal_stats = log_crystals_4bit(best_e_params, best_d_params, n_emitter, n_decoder)
    else:
        crystal_stats = None

    return correct, best_loss, elapsed, crystal_stats


def main():
    total_start = time.time()

    print("╔" + "═"*55 + "╗")
    print("║  ResNet-Ψ — Pipeline Vetorizado (RN_psi)              ║")
    print("║  Toda a população avaliada em um único forward pass   ║")
    print("╚" + "═"*55 + "╝")

    # Paridade 4 bits — 2 fases + logging de cristais
    r4, l4, t4, crystals = train_parity4_2fases(
        label="PARIDADE 4 BITS — 2 Fases", max_gen=300
    )

    total = time.time() - total_start

    print("\n\n╔" + "═"*58 + "╗")
    print("║         RELATÓRIO FINAL — RN_psi                        ║")
    print("╠" + "═"*58 + "╣")

    ok   = "✓" if r4 == 16 else "✗"
    line = f"  {ok} Paridade 4 bits (2 fases)    {r4}/16  Loss: {l4:.4f}"
    print(f"║{line:<58}║")
    print(f"║{'':58}║")

    if crystals:
        line2 = f"  Cristais: {crystals['total']} total | {crystals['media']:.1f}/combinação | {crystals['occupied_cells']} células"
        print(f"║{line2:<58}║")

    print(f"║{'':58}║")
    print(f"║  Tempo total: {total:.0f}s ({total/60:.1f} min)"
          + " " * max(0, 31 - len(f"{total:.0f}")) + "║")
    print("╚" + "═"*58 + "╝")

if __name__ == "__main__":
    main()