"""
ResNet-Ψ — MNIST sem emitter.

Pixels injetados diretamente como perturbação gaussiana no campo.
Zero parâmetros no processamento da entrada.
Apenas o decoder linear treina (23040 parâmetros).

Pipeline:
  imagem 28×28 → perturbação gaussiana no campo 48×48
  campo evolui (física pura)
  cristais emergem
  crystal_map 48×48 → decoder linear → 10 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Device ────────────────────────────────────────────────────────────────────
from RN_psi_mnist import (
    psi_step, CrystalMemory,
    FIELD_SIZE, STIM_ON, STIM_TOTAL,
    DEVICE,
    PSI_C2, PSI_GAMMA, PSI_ALPHA, PSI_BETA, PSI_DT,
)

print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 50
PATIENCE     = 10

USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ── MNIST ─────────────────────────────────────────────────────────────────────
from torchvision import datasets, transforms

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_raw   = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Imagens como (N, 28, 28) — mantém estrutura espacial
    X_train = torch.stack([train_full[i][0].squeeze(0) for i in range(len(train_full))])
    Y_train = torch.tensor([train_full[i][1] for i in range(len(train_full))], dtype=torch.long)
    X_test  = torch.stack([test_raw[i][0].squeeze(0)   for i in range(len(test_raw))])
    Y_test  = torch.tensor([test_raw[i][1]  for i in range(len(test_raw))],   dtype=torch.long)

    # Split treino/val
    torch.manual_seed(42)
    perm  = torch.randperm(len(X_train))
    n_val = 10000
    X_val, Y_val     = X_train[perm[-n_val:]], Y_train[perm[-n_val:]]
    X_train, Y_train = X_train[perm[:-n_val]], Y_train[perm[:-n_val]]

    print(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}")
    return (X_train.to(DEVICE), Y_train.to(DEVICE),
            X_val.to(DEVICE),   Y_val.to(DEVICE),
            X_test.to(DEVICE),  Y_test.to(DEVICE))

# ── Encoder físico: imagem 28×28 → perturbação no campo 48×48 ─────────────────
#
# Cada pixel (i,j) da imagem 28×28 tem uma posição fixa no campo 48×48.
# A posição é determinística: mapeia linearmente [0,27] → [0.1, 0.9] do grid.
# A amplitude é o valor do pixel.
# A perturbação é uma gaussiana centrada nessa posição.
#
# Pré-computamos as gaussianas de todos os 784 pixels UMA VEZ.
# No forward: perturbação = imagem_flat @ gaussianas_flat  (matmul, O(1))

SIGMA_PIX = 0.04   # largura gaussiana (~2px no grid 48)

def _build_pixel_gaussians():
    """
    Retorna tensor (784, FIELD_SIZE, FIELD_SIZE):
    gaussian[p] = gaussiana do pixel p no campo.
    """
    coords = torch.linspace(0., 1., FIELD_SIZE, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')  # (H, W)

    gaussians = []
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            g  = torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * SIGMA_PIX**2))
            gaussians.append(g)

    return torch.stack(gaussians)   # (784, H, W)

print("Pré-computando gaussianas dos pixels...")
_PIXEL_GAUSSIANS = _build_pixel_gaussians()   # (784, 48, 48)
_PG_FLAT = _PIXEL_GAUSSIANS.view(784, -1)     # (784, 2304) — para matmul rápido
print(f"  Gaussianas: {_PIXEL_GAUSSIANS.shape}  |  mem: {_PIXEL_GAUSSIANS.nbytes/1e6:.1f} MB")


def images_to_field_perturbation(images_batch):
    """
    images_batch: (B, 28, 28) — pixels normalizados
    Retorna: (B, FIELD_SIZE, FIELD_SIZE) — perturbação inicial do campo

    Operação: soma ponderada de gaussianas pelos valores dos pixels.
    É uma convolução linear: cada pixel contribui com sua gaussiana escalada.
    Zero parâmetros — puramente determinístico.
    """
    B    = images_batch.shape[0]
    flat = images_batch.view(B, 784)                   # (B, 784)
    # matmul: (B, 784) @ (784, 2304) → (B, 2304)
    pert = flat @ _PG_FLAT.to(flat.dtype)
    return pert.view(B, FIELD_SIZE, FIELD_SIZE)        # (B, 48, 48)


def run_field_batch(images_batch):
    """
    Roda o campo para um batch de imagens.
    Sem emitter, sem parâmetros — física pura.
    Retorna crystal_map (B, FIELD_SIZE*FIELD_SIZE).
    """
    with torch.no_grad():
        init_pert = images_to_field_perturbation(images_batch)   # (B, H, W)
        field     = init_pert.clone()
        velocity  = torch.zeros_like(field)
        memory    = CrystalMemory(images_batch.shape[0], dtype=field.dtype)

        for s in range(STIM_TOTAL):
            active          = s < STIM_ON
            field, velocity = psi_step(field, velocity, init_pert, active)
            memory.update_envelope(field)
            if memory.window_idx > 0:
                memory.try_crystallize(field)
            field = memory.remit(field, None)

    return memory.crystal_map.view(images_batch.shape[0], -1).float()   # (B, 2304)


# ── Decoder — único componente treinável ──────────────────────────────────────

class PureFieldDecoder(nn.Module):
    """
    Lê o crystal_map e classifica.
    É o ÚNICO componente com parâmetros treináveis.
    23040 parâmetros no total (2304 × 10).
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FIELD_SIZE * FIELD_SIZE, 10)

    def forward(self, crystal_map_flat):
        return self.fc(crystal_map_flat)


# ── Pré-computar crystal_maps de todo o dataset ───────────────────────────────
# Como o campo não tem parâmetros, os crystal_maps são fixos.
# Pré-computar uma vez → treinamento do decoder é trivialmente rápido.

def precompute_crystal_maps(X, desc, batch_size=128):
    """Processa o dataset inteiro e retorna crystal_maps pré-computados."""
    N = len(X)
    all_cmaps = []
    t0 = time.time()
    for i in range(0, N, batch_size):
        batch = X[i:i+batch_size]
        cmap  = run_field_batch(batch)
        all_cmaps.append(cmap.cpu())
        if (i // batch_size) % 20 == 0:
            pct = min(100, i / N * 100)
            elapsed = time.time() - t0
            print(f"  {desc}: {i}/{N} ({pct:.0f}%)  {elapsed:.0f}s", end='\r')
    print(f"  {desc}: {N}/{N} (100%)  {time.time()-t0:.1f}s")
    return torch.cat(all_cmaps, dim=0)   # (N, 2304)


# ── Treino ────────────────────────────────────────────────────────────────────

def train():
    print("\nCarregando MNIST...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()

    print("\nPré-computando crystal_maps (campo sem parâmetros — feito uma vez)...")
    t0 = time.time()
    CM_train = precompute_crystal_maps(X_train, "Treino").to(DEVICE)
    CM_val   = precompute_crystal_maps(X_val,   "Val"  ).to(DEVICE)
    CM_test  = precompute_crystal_maps(X_test,  "Teste").to(DEVICE)
    print(f"Pré-computação total: {time.time()-t0:.1f}s")
    print(f"Crystal maps — treino: {CM_train.shape}  val: {CM_val.shape}")

    # Estatísticas do campo
    n_crys_mean = (CM_train > 0.01).float().sum(dim=1).mean().item()
    print(f"Cristais por imagem (média): {n_crys_mean:.1f} / {FIELD_SIZE*FIELD_SIZE} pixels")

    # Decoder
    decoder   = PureFieldDecoder().to(DEVICE)
    n_params  = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder: {n_params} parâmetros")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_cnt = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}

    print(f"\n{'Época':>5}  {'Loss':>8}  {'Treino':>8}  {'Val':>8}  {'Tempo':>6}")
    print("-" * 45)

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep = time.time()
        decoder.train()

        # Shuffle
        perm   = torch.randperm(len(CM_train), device=DEVICE)
        CM_sh  = CM_train[perm]
        Y_sh   = Y_train[perm]

        total_loss = 0.0
        correct    = 0

        for i in range(0, len(CM_sh), BATCH_SIZE):
            cm = CM_sh[i:i+BATCH_SIZE]
            y  = Y_sh[i:i+BATCH_SIZE]

            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    logits = decoder(cm)
                    loss   = criterion(logits, y)
            else:
                logits = decoder(cm)
                loss   = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()

        scheduler.step()

        train_acc  = correct / len(CM_sh) * 100
        train_loss = total_loss / len(CM_sh)

        # Validação
        decoder.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(0, len(CM_val), 1024):
                val_logits.append(decoder(CM_val[i:i+1024]))
            val_logits = torch.cat(val_logits)
            val_acc    = (val_logits.argmax(1) == Y_val).float().mean().item() * 100

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)

        elapsed = time.time() - t_ep
        print(f"{epoch:>5}  {train_loss:>8.4f}  {train_acc:>7.2f}%  {val_acc:>7.2f}%  {elapsed:>5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(decoder.state_dict(), 'pure_field_decoder_best.pt')
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping na época {epoch} (melhor val: {best_val_acc:.2f}%)")
                break

    # Teste final
    decoder.load_state_dict(torch.load('pure_field_decoder_best.pt', weights_only=True))
    decoder.eval()
    with torch.no_grad():
        test_logits = []
        for i in range(0, len(CM_test), 1024):
            test_logits.append(decoder(CM_test[i:i+1024]))
        test_logits = torch.cat(test_logits)
        test_acc    = (test_logits.argmax(1) == Y_test).float().mean().item() * 100

    print(f"\n{'='*45}")
    print(f"RESULTADO FINAL")
    print(f"  Parâmetros treináveis : {n_params}")
    print(f"  Melhor val accuracy   : {best_val_acc:.2f}%")
    print(f"  Teste accuracy        : {test_acc:.2f}%")
    print(f"{'='*45}")

    # Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'MNIST — Campo Puro (zero parâmetros na entrada)\nTeste: {test_acc:.2f}%  |  Params decoder: {n_params}', fontsize=11)

    axes[0].plot(history['train_acc'], label='Treino')
    axes[0].plot(history['val_acc'],   label='Val')
    axes[0].set_xlabel('Época'); axes[0].set_ylabel('Acurácia (%)')
    axes[0].set_title('Acurácia'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_loss'])
    axes[1].set_xlabel('Época'); axes[1].set_ylabel('Cross-Entropy Loss')
    axes[1].set_title('Loss de Treino'); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('viz_mnist_pure_field.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("-> viz_mnist_pure_field.png")

    return test_acc


if __name__ == '__main__':
    train()
