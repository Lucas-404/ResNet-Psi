"""
ResNet-Ψ — MNIST (784D → 10 classes) — versão TPU (Google Colab v6e-1)

Diferenças em relação ao RN_psi_mnist.py:
  - Device via torch_xla (xla:0)
  - AMP desabilitado (TPU já opera em bfloat16 internamente)
  - autocast substituído por no-op
  - xm.mark_step() após cada batch (força execução do grafo XLA)
  - optimizer.fused=False (não suportado no XLA)
  - Branches Python no loop físico eliminados — substituídos por máscaras tensoriais
    (evita recompilação do grafo XLA a cada step)

Setup no Colab (célula 1):
  !pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html -q

Setup no Colab (célula 2):
  import os
  os.environ['PJRT_DEVICE'] = 'TPU'
  !python RN_psi_mnist_tpu.py
"""

import os
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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ── Device TPU ─────────────────────────────────────────────────────────────
import torch_xla.core.xla_model as xm

DEVICE = xm.xla_device()
print(f"Dispositivo: {DEVICE}")

# ── Constantes físicas (Seção 2.2 do paper) ────────────────────────────────
PSI_C2    = 0.3
PSI_GAMMA = 0.06
PSI_ALPHA = 0.04
PSI_BETA  = 0.005
PSI_DT    = 0.05

# ── Parâmetros do campo ────────────────────────────────────────────────────
FIELD_SIZE = 48
N_WAVES    = 16
STIM_ON    = 40
STIM_TOTAL = 80

# ── Cristais ───────────────────────────────────────────────────────────────
CRYSTAL_W       = 20
CRYSTAL_K       = 3
CRYSTAL_A_MIN   = 0.3
CRYSTAL_CV_MAX  = 0.15
CRYSTAL_SEP     = 5
CRYSTAL_PATTERN = 5
CRYSTAL_MAX     = 80
CRYSTAL_REMIT   = 0.05

# ── Hiperparâmetros de treino ──────────────────────────────────────────────
HIDDEN_DIM   = 256
BATCH_SIZE   = 2048   # TPU v6e-1: batch maior = menos compilações por epoch
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 15

N_DIM     = 784
N_CLASSES = 10

# ── AMP — no-op na TPU (bfloat16 é nativo) ────────────────────────────────
from contextlib import contextmanager

@contextmanager
def autocast():
    yield

# ── Máscaras de step pré-computadas (elimina branches Python no XLA) ───────
# active_masks[s] = 1.0 se s < STIM_ON, 0.0 caso contrário
# window_masks[s] = 1.0 se janela deve ser fechada neste step
_active_masks  = torch.tensor(
    [1.0 if s < STIM_ON else 0.0 for s in range(STIM_TOTAL)],
    device=DEVICE
)

# Pré-computa em qual step cada janela é fechada
# window_idx avança quando window_step atinge CRYSTAL_W
_window_close  = torch.tensor(
    [1.0 if (s + 1) % CRYSTAL_W == 0 else 0.0 for s in range(STIM_TOTAL)],
    device=DEVICE
)
_window_idx_at = torch.tensor(
    [((s + 1) // CRYSTAL_W - 1) % CRYSTAL_K if (s + 1) % CRYSTAL_W == 0 else -1
     for s in range(STIM_TOTAL)],
    dtype=torch.int32   # int32 — TPU v6e não suporta int64
)
_has_window    = torch.tensor(
    [1.0 if (s + 1) // CRYSTAL_W > 0 else 0.0 for s in range(STIM_TOTAL)],
    device=DEVICE
)

# ── Dataset MNIST ──────────────────────────────────────────────────────────
from torchvision import datasets, transforms


class TPUTensorDataset:
    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = X.to(DEVICE)
        self.Y = Y.to(DEVICE)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.n          = len(X)

    def __iter__(self):
        # int32 — TPU v6e não suporta int64 (X64 proibido)
        if self.shuffle:
            idx = torch.randperm(self.n, dtype=torch.int32, device=DEVICE)
        else:
            idx = torch.arange(self.n, dtype=torch.int32, device=DEVICE)
        for i in range(0, self.n, self.batch_size):
            b = idx[i:i+self.batch_size].long()  # converte só na hora do index
            yield self.X[b], self.Y[b]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_full = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_raw   = datasets.MNIST('./data', train=False, download=True, transform=transform)

    X_all = torch.stack([train_full[i][0] for i in range(len(train_full))])
    Y_all = torch.tensor([train_full[i][1] for i in range(len(train_full))], dtype=torch.int32)
    X_te  = torch.stack([test_raw[i][0]   for i in range(len(test_raw))])
    Y_te  = torch.tensor([test_raw[i][1]  for i in range(len(test_raw))],   dtype=torch.int32)

    torch.manual_seed(42)
    perm    = torch.randperm(len(X_all))
    n_val   = 10000
    n_train = len(X_all) - n_val
    X_train, Y_train = X_all[perm[:n_train]], Y_all[perm[:n_train]]
    X_val,   Y_val   = X_all[perm[n_train:]], Y_all[perm[n_train:]]

    print(f"Treino: {n_train} | Val: {n_val} | Teste: {len(X_te)}")
    return (TPUTensorDataset(X_train, Y_train, BATCH_SIZE, shuffle=True),
            TPUTensorDataset(X_val,   Y_val,   2048,       shuffle=False),
            TPUTensorDataset(X_te,    Y_te,    2048,       shuffle=False))


# ── Tensores físicos pré-computados ────────────────────────────────────────
_LAP_KERNEL = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                             device=DEVICE).view(1,1,3,3)
_coords = torch.linspace(0., 1., FIELD_SIZE, device=DEVICE)
_XG, _YG = torch.meshgrid(_coords, _coords, indexing='ij')
_XG = _XG.unsqueeze(0).unsqueeze(0)
_YG = _YG.unsqueeze(0).unsqueeze(0)

_DT    = torch.tensor(PSI_DT,    device=DEVICE)
_GAMMA = torch.tensor(PSI_GAMMA, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_BETA  = torch.tensor(PSI_BETA,  device=DEVICE)
_C2    = torch.tensor(PSI_C2,    device=DEVICE)


# ── Física ─────────────────────────────────────────────────────────────────

def psi_step(field, velocity, wave_sources, active_mask):
    """Step Verlet sem branch Python — active_mask é tensor 0.0 ou 1.0."""
    field = field + wave_sources * (_DT * 0.1) * active_mask

    inp    = field.unsqueeze(1)
    padded = F.pad(inp, (1,1,1,1), mode='circular')
    lap    = F.conv2d(padded, _LAP_KERNEL.to(field.dtype)).squeeze(1)

    nonlinear = _ALPHA * torch.tanh(field) * field
    dissip    = _BETA  * field * field**2
    acc       = _C2 * lap - _GAMMA * velocity + nonlinear - dissip

    velocity  = torch.clamp(velocity + acc * _DT, -5., 5.)
    field     = torch.clamp(field    + velocity * _DT, -10., 10.)
    return field, velocity


def emit_waves(wave_params, t_scalar):
    amp   = wave_params[:,:,0].unsqueeze(-1).unsqueeze(-1)
    freq  = wave_params[:,:,1].unsqueeze(-1).unsqueeze(-1)
    phase = wave_params[:,:,2].unsqueeze(-1).unsqueeze(-1)
    decay = wave_params[:,:,3].unsqueeze(-1).unsqueeze(-1)
    pos_x = wave_params[:,:,4].unsqueeze(-1).unsqueeze(-1)
    pos_y = wave_params[:,:,5].unsqueeze(-1).unsqueeze(-1)

    xg = _XG.to(wave_params.dtype)
    yg = _YG.to(wave_params.dtype)

    dist    = torch.sqrt((xg-pos_x)**2 + (yg-pos_y)**2 + 1e-8)
    osc     = torch.sin(freq*t_scalar + phase - freq*dist)
    temp    = torch.exp(-decay * t_scalar)
    spatial = 1.0 / (1.0 + dist)

    return (amp * osc * temp * spatial).sum(dim=1)


# ── Cristais (TPU XLA safe) ────────────────────────────────────────────────

class CrystalMemory:
    def __init__(self, B, dtype=torch.float32):
        self.B     = B
        self.dtype = dtype

        self.crystal_map = torch.zeros(B, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)
        self.window_max  = torch.zeros(B, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)

        ks = 2*CRYSTAL_SEP+1
        self._dilate = torch.ones(1, 1, ks, ks, device=DEVICE, dtype=dtype)

    def update_envelope(self, field, close_mask, window_idx):
        """
        XLA safe: sem 'if' Python. Atualiza com torch.where e tensores.
        close_mask: tensor escalar 0.0 ou 1.0
        window_idx: tensor escalar int32
        """
        self.window_max = torch.max(self.window_max, field.abs())

        # Máscara shape (1, K, 1, 1): 1.0 no slot correto, 0.0 nos demais
        idx_tensor = torch.arange(CRYSTAL_K, device=DEVICE, dtype=window_idx.dtype)
        match_idx  = (idx_tensor == window_idx).float().view(1, CRYSTAL_K, 1, 1)
        update_mask = match_idx * close_mask.view(1, 1, 1, 1).float()

        # Atualiza env_buffer sem inplace: mistura valor antigo e window_max
        window_max_expanded = self.window_max.unsqueeze(1).expand(-1, CRYSTAL_K, -1, -1)
        self.env_buffer = self.env_buffer * (1.0 - update_mask) + window_max_expanded * update_mask

        # Reseta window_max onde close_mask == 1.0
        self.window_max = self.window_max * (1.0 - close_mask.float())

    def try_crystallize(self, field, has_window_mask):
        """
        XLA safe: sem 'if' Python.
        has_window_mask: tensor escalar 0.0 ou 1.0 — zera scored se K janelas não completaram.
        """
        env  = self.env_buffer
        mean = env.mean(dim=1)
        std  = env.std(dim=1)
        cv   = std / (mean + 1e-8)

        crit_amp   = (mean > CRYSTAL_A_MIN).float()
        crit_cv    = (cv   < CRYSTAL_CV_MAX).float()
        crit_sat   = (mean < 8.0).float()
        candidates = crit_amp * crit_cv * crit_sat

        weight_kernel = self._dilate.to(self.crystal_map.dtype)
        occupied = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0, 1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            weight_kernel
        ).squeeze(1).clamp(0, 1)

        # has_window_mask anula scored se K janelas ainda não completaram
        scored = candidates * (1.0 - occupied) * field.abs() * has_window_mask.float()
        self.crystal_map = torch.clamp(self.crystal_map + scored, 0, 10.)

    def remit(self, field):
        remit_strength = self.crystal_map * CRYSTAL_REMIT
        field = field + remit_strength * torch.sign(field)
        return torch.clamp(field, -10., 10.)


# ── Simulação do campo (TPU XLA safe) ──────────────────────────────────────

def run_psi_field(wave_params):
    B     = wave_params.shape[0]
    dtype = wave_params.dtype
    wp    = wave_params.detach()

    with torch.no_grad():
        field    = torch.zeros(B, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)
        velocity = torch.zeros_like(field)
        memory   = CrystalMemory(B, dtype=dtype)

        for s in range(STIM_TOTAL):
            t           = s * float(PSI_DT)
            active_mask = _active_masks[s]
            close_mask  = _window_close[s]   # tensor escalar, sem .item()
            win_idx     = _window_idx_at[s]  # tensor int32 escalar
            has_window  = _has_window[s]     # tensor escalar

            sources = emit_waves(wp, t)
            field, velocity = psi_step(field, velocity, sources, active_mask)

            # Execução incondicional — máscaras anulam efeito quando necessário
            memory.update_envelope(field, close_mask, win_idx)
            memory.try_crystallize(field, has_window)
            field = memory.remit(field)

        crystal_map = memory.crystal_map

    return crystal_map.view(B, -1).float()


# ── Modelo ─────────────────────────────────────────────────────────────────

class PsiFieldNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.emitter = nn.Sequential(
            nn.Linear(N_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM // 2, N_WAVES * 6),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * FIELD_SIZE * FIELD_SIZE, N_CLASSES),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        raw = self.emitter(x).view(-1, N_WAVES, 6)

        amp   = torch.abs(raw[:,:,0]) * 1.5 + 2.0
        freq  = torch.abs(raw[:,:,1]) * 3.0 + 2.0
        phase = raw[:,:,2]
        decay = torch.abs(raw[:,:,3]) * 0.01 + 0.001
        pos_x = 0.2 + 0.6 * torch.sigmoid(raw[:,:,4])
        pos_y = 0.2 + 0.6 * torch.sigmoid(raw[:,:,5])

        wave_params = torch.stack([amp, freq, phase, decay, pos_x, pos_y], dim=-1)
        crystal_map = run_psi_field(wave_params)
        crystal_img = crystal_map.view(-1, 1, FIELD_SIZE, FIELD_SIZE)

        return self.decoder(crystal_img)


# ── Avaliação ──────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, total = 0., 0, 0
    with torch.no_grad():
        for x, y in loader:
            with autocast():
                logits = model(x)
            xm.mark_step()
            total_loss    += F.cross_entropy(logits.float(), y.long(), reduction='sum').item()
            total_correct += (logits.argmax(dim=1) == y.long()).sum().item()
            total         += len(y)
    return total_loss / total, total_correct / total * 100


# ── Treino ─────────────────────────────────────────────────────────────────

def train():
    train_loader, val_loader, test_loader = load_mnist()

    model = PsiFieldNet().to(DEVICE)

    # fused=True não é suportado no XLA
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo: {n_params} parâmetros")
    print(f"Emitter: MLP({N_DIM}→{HIDDEN_DIM}→{HIDDEN_DIM//2}→{N_WAVES*6})")
    print(f"Campo: {FIELD_SIZE}×{FIELD_SIZE} | N_WAVES: {N_WAVES}")
    print(f"Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"{'='*65}")

    best_val_acc = 0.
    best_state   = None
    no_improve   = 0

    hist_train_loss, hist_val_loss, hist_val_acc = [], [], []
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_corr, epoch_total = 0., 0, 0

        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)

            with autocast():
                logits = model(x)
                loss   = F.cross_entropy(logits, y.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            xm.mark_step()  # força execução do grafo XLA

            epoch_loss  += loss.item() * len(y)
            epoch_corr  += (logits.detach().argmax(dim=1) == y.long()).sum().item()
            epoch_total += len(y)

        sched.step()

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_corr / epoch_total * 100
        val_loss, val_acc = evaluate(model, val_loader)

        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_val_acc.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve  += 1

        elapsed = time.time() - start
        print(f"Epoch {epoch:>3} | "
              f"Train {train_acc:.1f}% ({train_loss:.4f}) | "
              f"Val {val_acc:.1f}% ({val_loss:.4f}) | "
              f"LR {sched.get_last_lr()[0]:.2e} | "
              f"{elapsed:.0f}s")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping — val_acc estagnada por {PATIENCE} epochs")
            break

    if best_state:
        model.load_state_dict(best_state)

    print(f"\n{'='*65}")
    _, train_acc = evaluate(model, train_loader)
    _, val_acc   = evaluate(model, val_loader)
    _, test_acc  = evaluate(model, test_loader)
    print(f"FINAL | Treino: {train_acc:.2f}% | Val: {val_acc:.2f}% | Teste: {test_acc:.2f}%")

    xm.save(best_state, 'psi_field_mnist_tpu.pt')
    print("Modelo salvo em psi_field_mnist_tpu.pt")

    generate_plots(model, test_loader, hist_train_loss, hist_val_loss, hist_val_acc)
    return model


# ── Visualizações ──────────────────────────────────────────────────────────

COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
          '#42d4f4','#f032e6','#bfef45','#000075','#a9a9a9']


def generate_plots(model, test_loader, hist_train_loss, hist_val_loss, hist_val_acc):
    print("\nGerando visualizações...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(hist_train_loss)+1)
    ax1.plot(epochs, hist_train_loss, label='Treino', color='#3cb44b')
    ax1.plot(epochs, hist_val_loss,   label='Val',    color='#e6194b')
    ax1.set_title('Loss — MNIST (ResNet-Ψ TPU)'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.2)
    ax2.plot(epochs, hist_val_acc, color='#4363d8')
    ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax2.axhline(y=87, color='gold', linestyle='--', alpha=0.6, label='87% (baseline)')
    ax2.set_title('Acurácia Val (%)'); ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('viz_mnist_curves_tpu.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  -> viz_mnist_curves_tpu.png')

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            xm.mark_step()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Predito'); ax.set_ylabel('Real')
    correct = sum(p==l for p,l in zip(all_preds, all_labels))
    ax.set_title(f'Confusão MNIST Teste ({correct}/10000) — {correct/100:.2f}%')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('viz_mnist_confusion_tpu.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  -> viz_mnist_confusion_tpu.png')

    model.eval()
    readings_list, labels_list = [], []
    count = 0
    with torch.no_grad():
        for x, y in test_loader:
            raw   = model.emitter(x).view(-1, N_WAVES, 6)
            amp   = torch.abs(raw[:,:,0])*1.5+2.0
            freq  = torch.abs(raw[:,:,1])*3.0+2.0
            phase = raw[:,:,2]
            decay = torch.abs(raw[:,:,3])*0.01+0.001
            pos_x = 0.2+0.6*torch.sigmoid(raw[:,:,4])
            pos_y = 0.2+0.6*torch.sigmoid(raw[:,:,5])
            wp    = torch.stack([amp,freq,phase,decay,pos_x,pos_y], dim=-1)
            xm.mark_step()
            r = run_psi_field(wp).cpu().float().numpy()
            readings_list.append(r)
            labels_list.extend(y.cpu().numpy())
            count += len(y)
            if count >= 2000:
                break

    readings = np.vstack(readings_list)[:2000]
    labels   = np.array(labels_list)[:2000]
    pca      = PCA(n_components=2)
    proj     = pca.fit_transform(readings)

    fig, ax = plt.subplots(figsize=(9, 8))
    for cls in range(10):
        mask = labels == cls
        ax.scatter(proj[mask,0], proj[mask,1], c=COLORS[cls], alpha=0.4, s=15, label=str(cls))
        cx, cy = proj[mask,0].mean(), proj[mask,1].mean()
        ax.scatter(cx, cy, c=COLORS[cls], s=200, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=5)
    ax.set_title(f'PCA Mapa de Cristais — MNIST TPU\n'
                 f'PC1={pca.explained_variance_ratio_[0]*100:.1f}% '
                 f'PC2={pca.explained_variance_ratio_[1]*100:.1f}%')
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('viz_mnist_pca_tpu.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  -> viz_mnist_pca_tpu.png')

    print('Visualizações salvas.')


if __name__ == '__main__':
    train()
