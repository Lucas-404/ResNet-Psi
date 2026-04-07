"""
ResNet-Ψ — Classificação 8D v5 (Bottleneck Direto, sem Fourier)

Emitter simplificado:
  x (8D) → W_bottleneck (8×16) → z (16D) → amp, freq, fase, pos_x, pos_y

Sem Fourier no meio — o bottleneck aprende o mapeamento direto para
os parâmetros de onda que o campo precisa.

Hipótese: W_bottleneck converge para Sep-Loss comparável ao v1 (-214)
e depois vira constante, assim como γ, β, σ.

Split: 6/2 por classe → 48 treino, 16 teste
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
)

# 32 posições de leitura — grid 4×8 + extras assimétricos
# 16 era insuficiente para 8 classes (só 2 sinais/classe em média)
READ_POSITIONS = [
    # grade 4×6 uniforme
    (0.15, 0.2), (0.15, 0.4), (0.15, 0.6), (0.15, 0.8),
    (0.35, 0.2), (0.35, 0.4), (0.35, 0.6), (0.35, 0.8),
    (0.55, 0.2), (0.55, 0.4), (0.55, 0.6), (0.55, 0.8),
    (0.75, 0.2), (0.75, 0.4), (0.75, 0.6), (0.75, 0.8),
    # faixa central horizontal
    (0.25, 0.3), (0.25, 0.5), (0.25, 0.7),
    (0.45, 0.3), (0.45, 0.5), (0.45, 0.7),
    (0.65, 0.3), (0.65, 0.5), (0.65, 0.7),
    (0.85, 0.3), (0.85, 0.5), (0.85, 0.7),
    # pontos assimétricos para quebrar degenerescências
    (0.1, 0.45), (0.9, 0.55),
    (0.5, 0.1),  (0.5, 0.9),
]
N_READ = len(READ_POSITIONS)  # 32

# ============================================
# DATASET
# ============================================

N_DIM       = 8
N_CLASSES   = 8
N_TRAIN_CLS = 25
N_TEST_CLS  = 10
N_TRAIN     = N_CLASSES * N_TRAIN_CLS   # 200
N_TEST      = N_CLASSES * N_TEST_CLS    # 80

np.random.seed(42)

CENTROIDS = np.array([
    [-1,-1,-1,-1,-1,-1,-1,-1],  # 0: todos -1
    [+1,+1,-1,-1,-1,-1,-1,-1],  # 1: dim 0,1 ativos
    [-1,-1,+1,+1,-1,-1,-1,-1],  # 2: dim 2,3 ativos
    [+1,+1,+1,+1,-1,-1,-1,-1],  # 3: dim 0,1,2,3 ativos
    [-1,-1,-1,-1,+1,+1,-1,-1],  # 4: dim 4,5 ativos
    [+1,+1,-1,-1,+1,+1,-1,-1],  # 5: dim 0,1,4,5 ativos
    [-1,-1,+1,+1,+1,+1,-1,-1],  # 6: dim 2,3,4,5 ativos
    [+1,+1,+1,+1,+1,+1,-1,-1],  # 7: dim 0..5 ativos
], dtype=np.float32)


def make_dataset_split():
    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    for cls in range(N_CLASSES):
        for i in range(N_TRAIN_CLS + N_TEST_CLS):
            noise = np.random.randn(N_DIM).astype(np.float32) * 0.3
            p = CENTROIDS[cls] + noise
            if i < N_TRAIN_CLS:
                X_tr.append(p); Y_tr.append(cls)
            else:
                X_te.append(p); Y_te.append(cls)
    return (np.array(X_tr, dtype=np.float32), np.array(Y_tr, dtype=np.int64),
            np.array(X_te, dtype=np.float32), np.array(Y_te, dtype=np.int64))


X_train, Y_train, X_test, Y_test = make_dataset_split()
print(f" Treino: {N_TRAIN} | Teste: {N_TEST} | {N_DIM}D -> {N_CLASSES} classes")

# ============================================
# EMITTER — bottleneck direto (sem Fourier)
#
# W: (N_DIM, N_WAVES * 5) = (8, 10) = 80 params
#   z = tanh(x @ W)   shape: (B, N_WAVES * 5)
#   z contém direto: amp, freq, fase, pos_x, pos_y para cada onda
#
# N_WAVES = 2 (igual ao v1 original)
# Total params: 8 × 10 = 80
# ============================================

N_WAVES    = 6
N_COEFS    = 5   # amp, freq, fase, pos_x, pos_y
N_PARAMS_W = N_DIM * (N_WAVES * N_COEFS)  # 8 × 30 = 240 params


def emitter(W_flat, x_batch):
    """
    W_flat:  (N_PARAMS_W,) — pesos do bottleneck
    x_batch: (B, N_DIM) tensor
    Retorna: wave_params (B, N_WAVES, 6), wave_mask (B, N_WAVES, 1, 1)
    """
    B = x_batch.shape[0]

    if not isinstance(W_flat, torch.Tensor):
        W = torch.tensor(W_flat, dtype=torch.float32, device=DEVICE)
    else:
        W = W_flat.to(DEVICE)

    W = W.view(N_DIM, N_WAVES * N_COEFS)

    # Projeção direta: x (B, 8) @ W (8, 10) → z (B, 10)
    z = x_batch @ W  # sem ativação — mapeamento linear direto
    z = z.view(B, N_WAVES, N_COEFS)  # (B, N_WAVES, N_COEFS)

    amp   = torch.abs(z[:, :, 0]) * 1.5 + 2.0
    freq  = torch.abs(z[:, :, 1]) * 3.0 + 2.0
    phase = z[:, :, 2]
    decay = torch.full((B, N_WAVES), 0.001, device=DEVICE)
    pos_x = torch.sigmoid(z[:, :, 3])
    pos_y = torch.sigmoid(z[:, :, 4])

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


def run_field(W_flat, x_batch_np):
    B            = len(x_batch_np)
    x_batch      = torch.tensor(x_batch_np, dtype=torch.float32, device=DEVICE)
    silence_mask = torch.zeros(B, N_WAVES, 1, 1, device=DEVICE)

    wp, on_mask = emitter(W_flat, x_batch)

    psi = PsiField(batch_size=B,
                   damping=PSI_GAMMA_3BIT,
                   dissipation=PSI_BETA_3BIT,
                   sigma=PSI_SIGMA_3BIT)

    for step in range(STIM_TOTAL):
        psi.wave_params = wp
        psi.wave_mask   = on_mask if step < STIM_ON else silence_mask
        psi.step()

    return psi.read_at(READ_POSITIONS)  # (B, N_READ)


def run_field_batch_candidates(W_candidates, x_batch_np):
    """
    Avalia todos os candidatos CMA-ES em paralelo num único PsiField gigante.

    W_candidates: (POP, N_PARAMS_W) numpy array
    x_batch_np:   (N_TRAIN, N_DIM) numpy array

    Estratégia: empilha POP × N_TRAIN amostras no batch do PsiField.
    Retorna: (POP, N_TRAIN, N_READ)
    """
    POP    = len(W_candidates)
    N_DATA = len(x_batch_np)
    B      = POP * N_DATA  # batch total

    # Repete cada candidato N_DATA vezes e cada amostra POP vezes
    W_rep = np.repeat(W_candidates, N_DATA, axis=0)          # (B, N_PARAMS_W)
    x_rep = np.tile(x_batch_np, (POP, 1))                    # (B, N_DIM)

    x_tensor     = torch.tensor(x_rep, dtype=torch.float32, device=DEVICE)
    silence_mask = torch.zeros(B, N_WAVES, 1, 1, device=DEVICE)

    # Emitter vetorizado: cada linha de W_rep é um candidato diferente
    W_t = torch.tensor(W_rep, dtype=torch.float32, device=DEVICE)  # (B, N_PARAMS_W)
    W_t = W_t.view(B, N_DIM, N_WAVES * N_COEFS)
    z   = torch.bmm(x_tensor.unsqueeze(1), W_t).squeeze(1)         # (B, N_WAVES*N_COEFS)
    z   = z.view(B, N_WAVES, N_COEFS)

    amp   = torch.abs(z[:, :, 0]) * 1.5 + 2.0
    freq  = torch.abs(z[:, :, 1]) * 3.0 + 2.0
    phase = z[:, :, 2]
    decay = torch.full((B, N_WAVES), 0.001, device=DEVICE)
    pos_x = torch.sigmoid(z[:, :, 3])
    pos_y = torch.sigmoid(z[:, :, 4])

    wp = torch.zeros(B, N_WAVES, 6, device=DEVICE)
    wp[:, :, 0] = amp
    wp[:, :, 1] = freq
    wp[:, :, 2] = phase
    wp[:, :, 3] = decay
    wp[:, :, 4] = pos_x
    wp[:, :, 5] = pos_y

    on_mask = torch.ones(B, N_WAVES, 1, 1, device=DEVICE)

    psi = PsiField(batch_size=B,
                   damping=PSI_GAMMA_3BIT,
                   dissipation=PSI_BETA_3BIT,
                   sigma=PSI_SIGMA_3BIT)

    for step in range(STIM_TOTAL):
        psi.wave_params = wp
        psi.wave_mask   = on_mask if step < STIM_ON else silence_mask
        psi.step()

    values = psi.read_at(READ_POSITIONS)          # (B, N_READ)
    return values.view(POP, N_DATA, N_READ)        # (POP, N_TRAIN, N_READ)


# ============================================
# FASE 1 — CMA-ES treina W_bottleneck
# ============================================

def evaluate_bottleneck_batch(W_candidates):
    """Avalia todos os candidatos de uma vez — um único PsiField gigante."""
    all_values = run_field_batch_candidates(W_candidates, X_train)
    # all_values: (POP, N_TRAIN, N_READ)

    targets = torch.tensor(Y_train, device=DEVICE)
    losses  = []

    for values in all_values:  # values: (N_TRAIN, N_READ)
        centroids = [values[targets == c].mean(dim=0) for c in range(N_CLASSES)]

        separation = sum(
            torch.norm(centroids[i] - centroids[j])
            for i in range(N_CLASSES)
            for j in range(i + 1, N_CLASSES)
        )

        intra_var = sum(
            values[targets == c].var(dim=0).mean()
            for c in range(N_CLASSES)
        )

        losses.append(float((-separation + intra_var).item()))

    return losses


def train_bottleneck(max_gen=700, pop_size=20):
    print(f"  FASE 1 - W_bottleneck direto (CMA-ES, {N_PARAMS_W} params) [batch paralelo]")
    print(f"  {'-'*50}")

    x0 = np.random.randn(N_PARAMS_W) * 0.1
    es = cma.CMAEvolutionStrategy(
        x0, 0.3,
        {'popsize': pop_size, 'maxiter': max_gen, 'verbose': -9}
    )

    start     = time.time()
    best_loss = float('inf')
    best_W    = None
    gen       = 0

    while not es.stop():
        gen += 1
        cands     = np.array(es.ask())
        losses    = evaluate_bottleneck_batch(cands)       # todos de uma vez
        losses_np = np.array(losses, dtype=np.float32)
        es.tell(cands.tolist(), losses_np.tolist())

        idx = int(np.argmin(losses_np))
        if losses_np[idx] < best_loss:
            best_loss = float(losses_np[idx])
            best_W    = cands[idx].copy()

        if gen % 20 == 0 or gen == 1:
            print(f"  Gen {gen:>4} | Sep-Loss: {best_loss:.4f} | {time.time()-start:.0f}s")

    print(f"\n  W_bottleneck fixado. Sep-Loss: {best_loss:.4f} | {time.time()-start:.0f}s")
    return best_W


# ============================================
# FASE 2 — CMA-ES treina Decoder
# ============================================

def evaluate_decoder(decoder_params, W_flat, x_data, y_data):
    """Avalia um único decoder — usado só no teste final."""
    values  = run_field(W_flat, x_data)
    targets = torch.tensor(y_data, dtype=torch.long, device=DEVICE)

    W_dec   = torch.tensor(decoder_params, dtype=torch.float32, device=DEVICE)
    W_dec   = W_dec.view(N_READ, N_CLASSES)
    logits  = values @ W_dec
    loss    = F.cross_entropy(logits, targets)
    correct = (logits.argmax(dim=1) == targets).sum().item()

    return float(loss.item()), int(correct)


def evaluate_decoder_batch(decoder_candidates, field_values, y_data):
    """
    Avalia todos os candidatos decoder de uma vez.
    field_values: (N_TRAIN, N_READ) — campo já computado (fixo na geração)
    decoder_candidates: (POP, N_DECODER)
    Retorna: losses (POP,), corrects (POP,)
    """
    targets = torch.tensor(y_data, dtype=torch.long, device=DEVICE)
    W_decs  = torch.tensor(decoder_candidates, dtype=torch.float32, device=DEVICE)
    W_decs  = W_decs.view(-1, N_READ, N_CLASSES)          # (POP, N_READ, N_CLASSES)

    # field_values: (N_TRAIN, N_READ) → (1, N_TRAIN, N_READ)
    fv      = field_values.unsqueeze(0)                    # (1, N_TRAIN, N_READ)
    logits  = torch.bmm(fv.expand(W_decs.shape[0], -1, -1), W_decs)  # (POP, N_TRAIN, N_CLASSES)

    losses   = []
    corrects = []
    for i in range(W_decs.shape[0]):
        loss    = F.cross_entropy(logits[i], targets)
        correct = (logits[i].argmax(dim=1) == targets).sum().item()
        losses.append(float(loss.item()))
        corrects.append(int(correct))

    return losses, corrects


def train_decoder(W_bottleneck, max_gen=700, pop_size=20):
    N_DECODER = N_READ * N_CLASSES  # 256 params (32 leituras × 8 classes)

    print(f"\n  FASE 2 - Decoder (CMA-ES, {N_DECODER} params) [batch paralelo]")
    print(f"  {'-'*50}")

    # Campo fixo para todo o treino da Fase 2 (W_bottleneck não muda)
    field_values = run_field(W_bottleneck, X_train)  # (N_TRAIN, N_READ)

    x0_d = np.random.randn(N_DECODER) * 0.1
    es_d = cma.CMAEvolutionStrategy(
        x0_d, 0.5,
        {'popsize': pop_size, 'maxiter': max_gen, 'verbose': -9}
    )

    start        = time.time()
    best_d_loss  = float('inf')
    best_d_p     = None
    best_correct = 0
    history      = []
    refining     = False
    gen          = 0

    while not es_d.stop():
        gen += 1
        cands = np.array(es_d.ask())

        if refining and best_d_p is not None:
            cands[-1] = best_d_p.copy()

        losses, corrects = evaluate_decoder_batch(cands, field_values, Y_train)

        losses_np = np.array(losses, dtype=np.float32)
        es_d.tell(cands.tolist(), losses_np.tolist())

        idx     = int(np.argmin(losses_np))
        gl      = float(losses_np[idx])
        correct = corrects[idx]
        history.append(correct)

        if gl < best_d_loss:
            best_d_loss  = gl
            best_correct = correct
            if not refining or correct == N_TRAIN:
                best_d_p = cands[idx].copy()

        if correct == N_TRAIN and not refining:
            es_d.sigma *= 0.3
            es_d.mean   = best_d_p.copy()
            refining    = True
            print(f"\n  * {N_TRAIN}/{N_TRAIN} treino na gen {gen}! Refinamento ativado.")

        if gen % 10 == 0 or gen == 1:
            elapsed  = time.time() - start
            mode_tag = " [refinando]" if refining else ""
            print(f"  Gen {gen:>4} | Loss: {gl:.4f} | {correct}/{N_TRAIN}{mode_tag} | {elapsed:.0f}s")

        if refining and len(history) >= 10:
            if all(c == N_TRAIN for c in history[-10:]):
                print(f"\n  -> Convergido.")
                break

    return best_d_p, best_d_loss, best_correct


# ============================================
# TREINO COMPLETO
# ============================================

def train_v5(max_gen_w=700, max_gen_d=700, pop_size=20):
    print(f"\n{'='*58}")
    print(f"  ResNet-Psi v5 - Bottleneck Direto (sem Fourier)")
    print(f"{'='*58}")
    print(f"  Emitter: W {N_DIM}x{N_WAVES*N_COEFS} = {N_PARAMS_W} params -> direto para ondas")
    print(f"  Decoder: {N_READ}x{N_CLASSES} = {N_READ*N_CLASSES} params")
    print(f"  Treino: {N_TRAIN} | Teste: {N_TEST}")
    print(f"  Fisica: gamma={PSI_GAMMA_3BIT} beta={PSI_BETA_3BIT} sigma={PSI_SIGMA_3BIT}")
    print()

    start_total = time.time()

    # ── FASE 1 ──────────────────────────────────────────────────────
    W_bottleneck = train_bottleneck(max_gen=max_gen_w, pop_size=pop_size)
    np.save("W_BOTTLENECK_8D_v5.npy", W_bottleneck)
    print(f"\n  -> W_bottleneck salvo em W_BOTTLENECK_8D_v5.npy")

    # ── FASE 2 ──────────────────────────────────────────────────────
    best_d_p, best_d_loss, best_correct = train_decoder(
        W_bottleneck, max_gen=max_gen_d, pop_size=pop_size
    )

    # ── RESULTADO ────────────────────────────────────────────────────
    elapsed = time.time() - start_total

    print(f"\n  {'='*50}")
    print(f"  TREINO: {best_correct}/{N_TRAIN} | Loss: {best_d_loss:.4f}")

    if best_d_p is not None:
        _, test_correct = evaluate_decoder(best_d_p, W_bottleneck, X_test, Y_test)
        print(f"  TESTE:  {test_correct}/{N_TEST}  (dados nunca vistos)")
        print(f"  Tempo total: {elapsed:.1f}s")

        print(f"\n  Acurácia por classe (teste):")
        values_te = run_field(W_bottleneck, X_test)
        W_dec     = torch.tensor(best_d_p, dtype=torch.float32, device=DEVICE).view(N_READ, N_CLASSES)
        preds_te  = (values_te @ W_dec).argmax(dim=1).cpu().numpy()

        for c in range(N_CLASSES):
            mask      = Y_test == c
            correct_c = (preds_te[mask] == Y_test[mask]).sum()
            total_c   = mask.sum()
            print(f"    Classe {c}: {correct_c}/{total_c}")

        print(f"\n  Constantes do sistema:")
        print(f"  PSI_GAMMA        = {PSI_GAMMA_3BIT}")
        print(f"  PSI_BETA         = {PSI_BETA_3BIT}")
        print(f"  PSI_SIGMA        = {PSI_SIGMA_3BIT}")
        print(f"  W_BOTTLENECK_8D  -> W_BOTTLENECK_8D_v5.npy ({N_PARAMS_W} valores)")

    return best_correct, best_d_loss, elapsed


if __name__ == "__main__":
    train_v5()
