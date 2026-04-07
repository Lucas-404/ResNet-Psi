"""
ResNet-Ψ — Classificação de Vetores 8D

Teste com dados contínuos de alta dimensão: vetores 8D → 8 classes.
Cada classe é um cluster gaussiano em espaço 8D.

Demonstra que o emitter generaliza para dimensões maiores —
passo no caminho para embeddings de texto reais (768D).

Arquitetura:
  - Emitter: recebe vetor 8D → parâmetros de onda
    param_w_c = base_w_c + Σ coef_d_w_c * x_d  (combinação linear das 8 dims)
  - PsiField: campo 48×48 com física fixada
  - Decoder: matriz 16×8 → logits para 8 classes (softmax)

Otimização em 2 fases:
  1. Emitter: maximiza separabilidade entre 8 classes no campo
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

from RN_psi import (
    PsiField,
    PSI_GAMMA_3BIT, PSI_BETA_3BIT, PSI_SIGMA_3BIT,
    READ_POSITIONS, N_READ,
)

# ============================================
# DATASET — vetores 8D, 8 classes (clusters gaussianos)
# ============================================

N_DIM      = 8
N_CLASSES  = 8
N_PER_CLASS = 8   # 8 pontos por classe = 64 total
N_SAMPLES  = N_CLASSES * N_PER_CLASS

np.random.seed(42)

def make_dataset_8d():
    """
    8 clusters gaussianos em espaço 8D.
    Cada cluster tem centróide em um vértice do hipercubo {-1,+1}^8.
    """
    centroids = np.array([
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [+1,-1,-1,-1,-1,-1,-1,-1],
        [-1,+1,-1,-1,-1,-1,-1,-1],
        [+1,+1,-1,-1,-1,-1,-1,-1],
        [-1,-1,+1,-1,-1,-1,-1,-1],
        [+1,-1,+1,-1,-1,-1,-1,-1],
        [-1,+1,+1,-1,-1,-1,-1,-1],
        [+1,+1,+1,-1,-1,-1,-1,-1],
    ], dtype=np.float32)

    X, Y = [], []
    for cls in range(N_CLASSES):
        for _ in range(N_PER_CLASS):
            noise = np.random.randn(N_DIM).astype(np.float32) * 0.3
            X.append(centroids[cls] + noise)
            Y.append(cls)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

X_data, Y_data = make_dataset_8d()
print(f" Dataset: {N_SAMPLES} pontos × {N_DIM}D → {N_CLASSES} classes")
print(f" Distribuição: {[(Y_data==c).sum() for c in range(N_CLASSES)]}")

# ============================================
# EMITTER — converte vetor 8D em parâmetros de onda
#
# Para cada onda w e coeficiente c:
#   param_w_c = base_w_c + Σ_{d=0}^{7} coef_d_w_c * x_d
#
# N_WAVES = 2, N_COEFS = 5 (amp, freq, fase, pos_x, pos_y)
# Params por coeficiente: 1 base + 8 coefs = 9
# Total: 2 × 5 × 9 = 90 params
# ============================================

N_WAVES    = 2
N_COEFS    = 5
N_PARAMS_E = N_WAVES * N_COEFS * (1 + N_DIM)  # 90 params


def build_wave_params_8d(emitter_params, x_batch):
    """
    emitter_params: (N_PARAMS_E,) numpy
    x_batch: (B, N_DIM) tensor
    Retorna: wave_params (B, N_WAVES, 6), wave_mask (B, N_WAVES, 1, 1)
    """
    B  = x_batch.shape[0]
    ep = torch.tensor(emitter_params, dtype=torch.float32, device=DEVICE)

    # Reshape: (N_WAVES, N_COEFS, 1+N_DIM)
    ep = ep.view(N_WAVES, N_COEFS, 1 + N_DIM)

    base  = ep[:, :, 0]        # (N_WAVES, N_COEFS)
    coefs = ep[:, :, 1:]       # (N_WAVES, N_COEFS, N_DIM)

    # (B, N_WAVES, N_COEFS) = base + x @ coefs^T
    # x_batch: (B, N_DIM)
    # coefs: (N_WAVES, N_COEFS, N_DIM) → (N_WAVES, N_COEFS, N_DIM)
    linear = torch.einsum('bd,wcd->bwc', x_batch, coefs)  # (B, N_WAVES, N_COEFS)
    params = base.unsqueeze(0) + linear                    # (B, N_WAVES, N_COEFS)

    amp   = torch.abs(params[:, :, 0]) * 1.5 + 2.0
    freq  = torch.abs(params[:, :, 1]) * 3.0 + 2.0
    phase = params[:, :, 2]
    decay = torch.full((B, N_WAVES), 0.001, device=DEVICE)
    pos_x = torch.sigmoid(params[:, :, 3])
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
# SIMULAÇÃO DO CAMPO
# ============================================

STIM_ON    = 40
STIM_TOTAL = 80


def run_field_8d(emitter_params, x_batch_np):
    B            = len(x_batch_np)
    x_batch      = torch.tensor(x_batch_np, dtype=torch.float32, device=DEVICE)
    silence_mask = torch.zeros(B, N_WAVES, 1, 1, device=DEVICE)

    wp, on_mask = build_wave_params_8d(emitter_params, x_batch)

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
# FASE 1 — separabilidade
# ============================================

def evaluate_emitter_8d(emitter_params):
    values  = run_field_8d(emitter_params, X_data)  # (N, N_READ)
    targets = torch.tensor(Y_data, device=DEVICE)

    centroids = []
    for c in range(N_CLASSES):
        mask = targets == c
        centroids.append(values[mask].mean(dim=0))

    separation = 0.0
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            separation += torch.norm(centroids[i] - centroids[j])

    intra_var = 0.0
    for c in range(N_CLASSES):
        mask = targets == c
        intra_var += values[mask].var(dim=0).mean()

    loss = -separation.item() + intra_var.item()
    return float(loss)


# ============================================
# FASE 2 — decoder
# ============================================

def evaluate_decoder_8d(decoder_params, emitter_params):
    values  = run_field_8d(emitter_params, X_data)
    targets = torch.tensor(Y_data, dtype=torch.long, device=DEVICE)

    W      = torch.tensor(decoder_params, dtype=torch.float32, device=DEVICE)
    W      = W.view(N_READ, N_CLASSES)
    logits = values @ W
    loss   = F.cross_entropy(logits, targets)
    preds  = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()

    return float(loss.item()), int(correct)


# ============================================
# TREINO — 2 fases
# ============================================

def train_8d(max_gen=700, pop_size=20):
    N_DECODER = N_READ * N_CLASSES  # 16 × 8 = 128

    print(f"\n{'='*58}")
    print(f"  ResNet-Ψ — Classificação 8D ({N_CLASSES} classes)")
    print(f"{'='*58}")
    print(f"  Emitter: {N_PARAMS_E} params | Decoder: {N_DECODER} params")
    print(f"  Dataset: {N_SAMPLES} pontos × {N_DIM}D × {N_CLASSES} classes")
    print(f"  Física: γ={PSI_GAMMA_3BIT} β={PSI_BETA_3BIT} σ={PSI_SIGMA_3BIT}")
    print(f"  Clusters gaussianos nos vértices do hipercubo {{-1,+1}}^8")
    print()

    # ── FASE 1 ──────────────────────────────────────────────────────
    print(f"  FASE 1 — Emitter (separabilidade, {N_CLASSES} classes em 8D)")
    print(f"  {'-'*50}")

    x0_e = np.random.randn(N_PARAMS_E) * 0.1
    es_e = cma.CMAEvolutionStrategy(
        x0_e, 0.3,
        {'popsize': pop_size, 'maxiter': 700, 'verbose': -9}
    )

    start       = time.time()
    best_e_loss = float('inf')
    best_e_p    = None

    gen = 0
    while not es_e.stop():
        gen += 1
        cands  = np.array(es_e.ask())
        losses = [evaluate_emitter_8d(c) for c in cands]
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
    print(f"\n  FASE 2 — Decoder ({N_CLASSES} classes)")
    print(f"  {'-'*50}")

    x0_d = np.random.randn(N_DECODER) * 0.1
    es_d = cma.CMAEvolutionStrategy(
        x0_d, 0.5,
        {'popsize': pop_size, 'maxiter': max_gen, 'verbose': -9}
    )

    start2       = time.time()
    best_d_loss  = float('inf')
    best_d_p     = None
    best_correct = 0
    history      = []
    refining     = False

    gen = 0
    while not es_d.stop():
        gen += 1
        cands = np.array(es_d.ask())

        if refining and best_d_p is not None:
            cands[-1] = best_d_p.copy()

        losses, corrects = [], []
        for c in cands:
            l, cor = evaluate_decoder_8d(c, best_e_p)
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
            if not refining or correct == N_SAMPLES:
                best_d_p = cands[idx].copy()

        if correct == N_SAMPLES and not refining:
            es_d.sigma *= 0.3
            es_d.mean   = best_d_p.copy()
            refining    = True
            print(f"\n  ★ {N_SAMPLES}/{N_SAMPLES} na gen {gen}! Refinamento ativado.")

        if gen % 10 == 0 or gen == 1:
            elapsed  = time.time() - start2
            mode_tag = " [refinando]" if refining else ""
            print(f"  Gen {gen:>4} | Loss: {gl:.4f} | {correct}/{N_SAMPLES}{mode_tag} | {elapsed:.0f}s")

        if refining and len(history) >= 10:
            if all(c == N_SAMPLES for c in history[-10:]):
                print(f"\n  → Convergido.")
                break

    # ── RESULTADO ────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n  RESULTADO: {best_correct}/{N_SAMPLES} | Loss: {best_d_loss:.4f} | {elapsed:.1f}s")

    if best_d_p is not None:
        x_t     = torch.tensor(X_data, dtype=torch.float32, device=DEVICE)
        values  = run_field_8d(best_e_p, X_data)
        W       = torch.tensor(best_d_p, dtype=torch.float32, device=DEVICE).view(N_READ, N_CLASSES)
        logits  = values @ W
        preds   = logits.argmax(dim=1).cpu().numpy()

        print(f"\n  Acurácia por classe:")
        for c in range(N_CLASSES):
            mask    = Y_data == c
            correct_c = (preds[mask] == Y_data[mask]).sum()
            total_c   = mask.sum()
            print(f"    Classe {c}: {correct_c}/{total_c}")

    return best_correct, best_d_loss, elapsed


if __name__ == "__main__":
    train_8d()
