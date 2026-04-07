"""
ResNet-Ψ — Classificação de Pontos 2D

Teste com dados contínuos: vetores (x, y) → 4 classes (quadrantes).
Demonstra que o emitter generaliza além de dados binários.

Arquitetura:
  - Emitter: recebe (x, y) ∈ [-1,1]² → parâmetros de onda
  - PsiField: campo 48×48 com física fixada (constantes 3-bit)
  - Decoder: 16 pesos lineares → logits para 4 classes (softmax)

Otimização em 2 fases:
  1. Emitter: maximiza separabilidade entre 4 classes no campo
  2. Decoder: minimiza cross-entropy com emitter fixo
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import cma

import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" usando dispositivo: {DEVICE}")

# ============================================
# IMPORTA O PSIFIELD E CONSTANTES DO PRINCIPAL
# ============================================

from RN_psi import (
    PsiField,
    PSI_GAMMA_3BIT, PSI_BETA_3BIT, PSI_SIGMA_3BIT,
    READ_POSITIONS, N_READ,
)

# ============================================
# DATASET — pontos 2D, 4 quadrantes
# ============================================

N_SAMPLES = 20  # 5 pontos por classe

np.random.seed(0)
def make_dataset():
    X, Y = [], []
    for cls in range(4):
        for _ in range(N_SAMPLES // 4):
            # Quadrante de cada classe
            sx = 1.0 if cls in [1, 3] else -1.0
            sy = 1.0 if cls in [2, 3] else -1.0
            x  = sx * np.random.uniform(0.2, 0.9)
            y  = sy * np.random.uniform(0.2, 0.9)
            X.append([x, y])
            Y.append(cls)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

X_data, Y_data = make_dataset()
N_CLASSES  = 4
N_INPUTS   = len(X_data)  # 20

print(f" Dataset: {N_INPUTS} pontos, {N_CLASSES} classes")
print(f" Distribuição: {[(Y_data==c).sum() for c in range(N_CLASSES)]}")

# ============================================
# EMITTER — converte (x, y) em parâmetros de onda
#
# Parâmetros do emitter: (N_PARAMS_E,)
# Para cada ponto (x, y), gera:
#   - 2 ondas (como no sistema de bits)
#   - Cada onda: amp, freq, fase, pos_x, pos_y
#
# Estrutura: emitter_params = (n_wave_params,)
# n_wave_params = 2 ondas × 5 coeficientes × 2 entradas (x, y) = 20
# Para cada onda w e coeficiente c:
#   param_w_c = base_w_c + coef_x_w_c * x + coef_y_w_c * y
# ============================================

N_WAVES     = 2
N_COEFS     = 5   # amp, freq, fase, pos_x, pos_y
N_PARAMS_E  = N_WAVES * N_COEFS * 3  # base + coef_x + coef_y = 30 params


def build_wave_params_2d(emitter_params, x_vals, y_vals):
    """
    Converte (x, y) em parâmetros de onda usando o emitter treinável.

    emitter_params: (N_PARAMS_E,) numpy
    x_vals, y_vals: (B,) tensors
    Retorna: wave_params (B, 2, 6), wave_mask (B, 2, 1, 1)
    """
    B = len(x_vals)
    ep = torch.tensor(emitter_params, dtype=torch.float32, device=DEVICE)

    # Reshape: (N_WAVES, N_COEFS, 3) — 3 = [base, coef_x, coef_y]
    ep = ep.view(N_WAVES, N_COEFS, 3)

    x = torch.tensor(x_vals, dtype=torch.float32, device=DEVICE)  # (B,)
    y = torch.tensor(y_vals, dtype=torch.float32, device=DEVICE)  # (B,)

    # Para cada onda e coeficiente: param = base + coef_x*x + coef_y*y
    # ep[:, :, 0] = base   (N_WAVES, N_COEFS)
    # ep[:, :, 1] = coef_x
    # ep[:, :, 2] = coef_y

    base   = ep[:, :, 0]  # (N_WAVES, N_COEFS)
    coef_x = ep[:, :, 1]
    coef_y = ep[:, :, 2]

    # (B, N_WAVES, N_COEFS)
    params = (base.unsqueeze(0)
              + coef_x.unsqueeze(0) * x.view(B, 1, 1)
              + coef_y.unsqueeze(0) * y.view(B, 1, 1))

    amp   = torch.abs(params[:, :, 0]) * 1.5 + 2.0          # (B, N_WAVES)
    freq  = torch.abs(params[:, :, 1]) * 3.0 + 2.0
    phase = params[:, :, 2]
    decay = torch.full((B, N_WAVES), 0.001, device=DEVICE)
    pos_x = torch.sigmoid(params[:, :, 3])                   # normaliza 0-1
    pos_y = torch.sigmoid(params[:, :, 4])

    wp = torch.zeros(B, N_WAVES, 6, device=DEVICE)
    wp[:, :, 0] = amp
    wp[:, :, 1] = freq
    wp[:, :, 2] = phase
    wp[:, :, 3] = decay
    wp[:, :, 4] = pos_x
    wp[:, :, 5] = pos_y

    mask = torch.ones(B, N_WAVES, 1, 1, device=DEVICE)
    return wp, mask


# ============================================
# SIMULAÇÃO DO CAMPO — 1 ponto por vez (batch=N_INPUTS)
# ============================================

STIM_ON    = 40
STIM_TOTAL = 80  # 40 on + 40 silêncio


def run_field(emitter_params, x_vals, y_vals):
    """
    Roda o campo para todos os pontos em paralelo (batch=N_INPUTS).
    Retorna values (N_INPUTS, N_READ).
    """
    B            = len(x_vals)
    silence_mask = torch.zeros(B, N_WAVES, 1, 1, device=DEVICE)

    wp, on_mask = build_wave_params_2d(emitter_params, x_vals, y_vals)

    psi = PsiField(batch_size=B,
                   damping=PSI_GAMMA_3BIT,
                   dissipation=PSI_BETA_3BIT,
                   sigma=PSI_SIGMA_3BIT)

    for step in range(STIM_TOTAL):
        if step < STIM_ON:
            psi.wave_params = wp
            psi.wave_mask   = on_mask
        else:
            psi.wave_mask = silence_mask
        psi.step()

    return psi.read_at(READ_POSITIONS)  # (B, N_READ)


# ============================================
# FASE 1 — separabilidade do emitter
# ============================================

def evaluate_emitter_2d(emitter_params):
    values  = run_field(emitter_params, X_data[:, 0], X_data[:, 1])  # (N, N_READ)
    targets = torch.tensor(Y_data, device=DEVICE)

    # Centróide de cada classe
    centroids = []
    for c in range(N_CLASSES):
        mask = targets == c
        centroids.append(values[mask].mean(dim=0))

    # Separação inter-classe: soma das distâncias entre pares de centróides
    separation = 0.0
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            separation += torch.norm(centroids[i] - centroids[j])

    # Variância intra-classe
    intra_var = 0.0
    for c in range(N_CLASSES):
        mask = targets == c
        intra_var += values[mask].var(dim=0).mean()

    loss = -separation.item() + intra_var.item()
    return float(loss)


# ============================================
# FASE 2 — decoder multi-classe (softmax)
# ============================================

def evaluate_decoder_2d(decoder_params, emitter_params):
    """
    decoder_params: (N_READ * N_CLASSES,) — matriz de pesos 16×4
    """
    values  = run_field(emitter_params, X_data[:, 0], X_data[:, 1])  # (N, N_READ)
    targets = torch.tensor(Y_data, dtype=torch.long, device=DEVICE)

    W = torch.tensor(decoder_params, dtype=torch.float32, device=DEVICE)
    W = W.view(N_READ, N_CLASSES)  # (16, 4)

    logits = values @ W             # (N, 4)
    loss   = F.cross_entropy(logits, targets)
    preds  = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()

    return float(loss.item()), int(correct)


# ============================================
# TREINO — 2 fases
# ============================================

def train_2d(max_gen=200, pop_size=20):
    N_DECODER = N_READ * N_CLASSES  # 16 × 4 = 64

    print(f"\n{'='*55}")
    print(f"  ResNet-Ψ — Classificação 2D (4 quadrantes)")
    print(f"{'='*55}")
    print(f"  Emitter: {N_PARAMS_E} params | Decoder: {N_DECODER} params")
    print(f"  Dataset: {N_INPUTS} pontos × {N_CLASSES} classes")
    print(f"  Física: γ={PSI_GAMMA_3BIT} β={PSI_BETA_3BIT} σ={PSI_SIGMA_3BIT}")
    print()

    # ── FASE 1 ──────────────────────────────────────────────────────
    print(f"  FASE 1 — Emitter (separabilidade)")
    print(f"  {'-'*45}")

    x0_e = np.random.randn(N_PARAMS_E) * 0.3
    es_e = cma.CMAEvolutionStrategy(
        x0_e, 0.5,
        {'popsize': pop_size, 'maxiter': 150, 'verbose': -9}
    )

    start       = time.time()
    best_e_loss = float('inf')
    best_e_p    = None

    gen = 0
    while not es_e.stop():
        gen += 1
        cands = np.array(es_e.ask())
        losses = [evaluate_emitter_2d(c) for c in cands]
        losses = np.array(losses, dtype=np.float32)
        es_e.tell(cands.tolist(), losses.tolist())

        idx = int(np.argmin(losses))
        if losses[idx] < best_e_loss:
            best_e_loss = float(losses[idx])
            best_e_p    = cands[idx].copy()

        if gen % 20 == 0 or gen == 1:
            print(f"  Gen {gen:>4} | Sep-Loss: {best_e_loss:.4f} | {time.time()-start:.0f}s")

    print(f"\n  Emitter fixado. Sep-Loss: {best_e_loss:.4f} | {time.time()-start:.0f}s")

    # ── FASE 2 ──────────────────────────────────────────────────────
    print(f"\n  FASE 2 — Decoder (classificação)")
    print(f"  {'-'*45}")

    x0_d = np.random.randn(N_DECODER) * 0.1
    es_d = cma.CMAEvolutionStrategy(
        x0_d, 0.5,
        {'popsize': pop_size, 'maxiter': max_gen, 'verbose': -9}
    )

    start2        = time.time()
    best_d_loss   = float('inf')
    best_d_p      = None
    best_correct  = 0
    history       = []
    refining      = False

    gen = 0
    while not es_d.stop():
        gen += 1
        cands = np.array(es_d.ask())

        if refining and best_d_p is not None:
            cands[-1] = best_d_p.copy()

        losses, corrects = [], []
        for c in cands:
            l, cor = evaluate_decoder_2d(c, best_e_p)
            losses.append(l)
            corrects.append(cor)

        losses = np.array(losses, dtype=np.float32)
        es_d.tell(cands.tolist(), losses.tolist())

        idx     = int(np.argmin(losses))
        gl      = float(losses[idx])
        correct = corrects[idx]
        history.append(correct)

        if gl < best_d_loss:
            best_d_loss  = gl
            best_correct = correct
            if not refining or correct == N_INPUTS:
                best_d_p = cands[idx].copy()

        if correct == N_INPUTS and not refining:
            es_d.sigma *= 0.3
            es_d.mean   = best_d_p.copy()
            refining    = True
            print(f"\n  ★ {N_INPUTS}/{N_INPUTS} na gen {gen}! Refinamento ativado.")

        if gen % 10 == 0 or gen == 1:
            elapsed  = time.time() - start2
            mode_tag = f" [refinando]" if refining else ""
            print(f"  Gen {gen:>4} | Loss: {gl:.4f} | {correct}/{N_INPUTS}{mode_tag} | {elapsed:.0f}s")

        if refining and len(history) >= 10:
            if all(c == N_INPUTS for c in history[-10:]):
                print(f"\n  → Convergido.")
                break

    # ── RESULTADO ────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n  RESULTADO: {best_correct}/{N_INPUTS} | Loss: {best_d_loss:.4f} | {elapsed:.1f}s")

    if best_d_p is not None:
        values  = run_field(best_e_p, X_data[:, 0], X_data[:, 1])
        W       = torch.tensor(best_d_p, dtype=torch.float32, device=DEVICE).view(N_READ, N_CLASSES)
        logits  = values @ W
        preds   = logits.argmax(dim=1).cpu().numpy()
        targets = Y_data

        nomes = ["Q3(−,−)", "Q4(+,−)", "Q1(−,+)", "Q2(+,+)"]
        print(f"\n  Detalhes por ponto:")
        print(f"  {'x':>6} {'y':>6} {'classe':>8} {'pred':>6} {'ok':>4}")
        for i in range(N_INPUTS):
            ok = "✓" if preds[i] == targets[i] else "✗"
            print(f"  {X_data[i,0]:>6.2f} {X_data[i,1]:>6.2f} "
                  f"{nomes[targets[i]]:>10} {nomes[preds[i]]:>10} {ok:>4}")

    return best_correct, best_d_loss, elapsed


if __name__ == "__main__":
    train_2d()
