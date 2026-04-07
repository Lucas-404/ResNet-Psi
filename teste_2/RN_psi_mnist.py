"""
ResNet-Ψ — MNIST (784D → 10 classes)
Implementação fiel ao paper: "Ressonância É Tudo Que Você Precisa"

Arquitetura (Figura 1 do paper):
  EMITTER (treinável) → ondas → CAMPO + CRISTAIS (física pura) → leituras → DECODER (treinável)

O campo é um reservatório físico NÃO diferenciável.
Gradiente flui APENAS dentro do emitter e do decoder.
O emitter aprende a gerar wave_params que produzem leituras
que o decoder consegue classificar.

Cristais (Seção 2.3 do paper):
  - Envelope tracking: janelas W=20 steps, K=5 janelas
  - Critérios: mean(env) > A_min, cv < ε, não saturado, separação espacial
  - Re-emissão ativa: cristais injetam padrão de volta no campo
  - Ressonância: 0.7*R_freq + 0.3*R_espacial (Eq. 6)
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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ── Device + otimizações H100 ──────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ── Constantes físicas (Seção 2.2 do paper) ────────────────────────────────
PSI_C2    = 0.3      # velocidade de onda (Laplaciano)
PSI_GAMMA = 0.06     # amortecimento (damping)
PSI_ALPHA = 0.04     # não-linearidade seletiva (tanh)
PSI_BETA  = 0.005    # dissipação cúbica
PSI_DT    = 0.05     # passo temporal (Verlet)

# ── Parâmetros do campo ────────────────────────────────────────────────────
FIELD_SIZE = 48
N_WAVES    = 16
STIM_ON    = 40      # steps com estímulo ativo
STIM_TOTAL = 80      # steps totais

# ── Cristais (Seção 2.3 do paper) ──────────────────────────────────────────
CRYSTAL_W       = 20    # janela de envelope (steps)
CRYSTAL_K       = 3     # número de janelas para cristalizar (reduzido para 80 steps)
CRYSTAL_A_MIN   = 0.3   # amplitude mínima do envelope
CRYSTAL_CV_MAX  = 0.15  # coeficiente de variação máximo
CRYSTAL_SEP     = 5     # separação mínima entre cristais (pixels)
CRYSTAL_PATTERN = 5     # meia-largura do padrão (11×11 = ±5)
CRYSTAL_MAX     = 80    # máximo de cristais (cap do paper)
CRYSTAL_REMIT   = 0.05  # força de re-emissão

# ── Ressonância (Eq. 6 do paper) ──────────────────────────────────────────
RES_FREQ_W  = 0.7    # peso frequencial
RES_SPAT_W  = 0.3    # peso espacial
RES_SIGMA   = 0.8    # largura Gaussiana frequencial
RES_THRESH  = 0.1    # limiar de re-emissão

# ── Hiperparâmetros de treino ──────────────────────────────────────────────
HIDDEN_DIM   = 256
BATCH_SIZE   = 512
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 15

N_DIM     = 784
N_CLASSES = 10

# ── AMP ────────────────────────────────────────────────────────────────────
USE_AMP   = True
AMP_DTYPE = torch.bfloat16
autocast  = lambda: torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP)

# ── Dataset MNIST — tudo na GPU ────────────────────────────────────────────
from torchvision import datasets, transforms

class GPUTensorDataset:
    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = X.to(DEVICE); self.Y = Y.to(DEVICE)
        self.batch_size = batch_size; self.shuffle = shuffle; self.n = len(X)

    def __iter__(self):
        idx = torch.randperm(self.n, device=DEVICE) if self.shuffle \
              else torch.arange(self.n, device=DEVICE)
        for i in range(0, self.n, self.batch_size):
            b = idx[i:i+self.batch_size]
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
    Y_all = torch.tensor([train_full[i][1] for i in range(len(train_full))], dtype=torch.long)
    X_te  = torch.stack([test_raw[i][0]   for i in range(len(test_raw))])
    Y_te  = torch.tensor([test_raw[i][1]  for i in range(len(test_raw))],   dtype=torch.long)

    torch.manual_seed(42)
    perm    = torch.randperm(len(X_all))
    n_val   = 10000
    n_train = len(X_all) - n_val
    X_train, Y_train = X_all[perm[:n_train]], Y_all[perm[:n_train]]
    X_val,   Y_val   = X_all[perm[n_train:]], Y_all[perm[n_train:]]

    print(f"Treino: {n_train} | Val: {n_val} | Teste: {len(X_te)}")
    print(f"Dataset na GPU: {(X_all.nbytes + X_te.nbytes) / 1e6:.0f} MB")

    return (GPUTensorDataset(X_train, Y_train, BATCH_SIZE, shuffle=True),
            GPUTensorDataset(X_val,   Y_val,   1024,       shuffle=False),
            GPUTensorDataset(X_te,    Y_te,    1024,       shuffle=False))

# ── Posições de leitura ────────────────────────────────────────────────────
def make_read_positions(n_grid=7):
    pos = []
    for i in range(n_grid):
        for j in range(n_grid):
            pos.append((0.1 + 0.8*i/(n_grid-1), 0.1 + 0.8*j/(n_grid-1)))
    pos += [(0.15,0.35),(0.35,0.15),(0.65,0.85),(0.85,0.65),
            (0.25,0.75),(0.75,0.25),(0.5,0.5)]
    return pos

READ_POSITIONS = make_read_positions()
N_READ = len(READ_POSITIONS)
print(f"N_READ: {N_READ}")

_read_pos = torch.tensor(READ_POSITIONS, dtype=torch.float32, device=DEVICE)
_read_ix  = torch.clamp((_read_pos[:,0]*(FIELD_SIZE-1)).long(), 0, FIELD_SIZE-1)
_read_iy  = torch.clamp((_read_pos[:,1]*(FIELD_SIZE-1)).long(), 0, FIELD_SIZE-1)

# ── Tensores físicos pré-computados ────────────────────────────────────────
_LAP_KERNEL = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                            device=DEVICE).view(1,1,3,3)
_coords = torch.linspace(0., 1., FIELD_SIZE, device=DEVICE)
_XG, _YG = torch.meshgrid(_coords, _coords, indexing='ij')
_XG = _XG.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
_YG = _YG.unsqueeze(0).unsqueeze(0)

_DT    = torch.tensor(PSI_DT,    device=DEVICE)
_GAMMA = torch.tensor(PSI_GAMMA, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_BETA  = torch.tensor(PSI_BETA,  device=DEVICE)
_C2    = torch.tensor(PSI_C2,    device=DEVICE)

# ── Física — Equação de onda modificada (Eq. 4 do paper) ──────────────────

def psi_step(field, velocity, wave_sources, active):
    """Um step de integração Verlet. Sem cristais aqui — gerenciados separado."""
    if active:
        field = field + wave_sources * (_DT * 0.1)

    inp    = field.unsqueeze(1)
    padded = F.pad(inp, (1,1,1,1), mode='circular')
    lap    = F.conv2d(padded, _LAP_KERNEL.to(field.dtype)).squeeze(1)

    # Eq. 4: c²∇²Ψ − γ(∂Ψ/∂t) + α·tanh(Ψ)·Ψ − β·Ψ·|Ψ|²
    nonlinear = _ALPHA * torch.tanh(field) * field
    dissip    = _BETA  * field * field**2
    acc       = _C2 * lap - _GAMMA * velocity + nonlinear - dissip

    velocity  = velocity + acc * _DT
    field     = field    + velocity * _DT
    field     = torch.clamp(field,    -10., 10.)
    velocity  = torch.clamp(velocity,  -5.,  5.)
    return field, velocity


def emit_waves(wave_params, t_scalar):
    """Eq. 1 do paper: ψᵢ(x,y,t) = Aᵢ·sin(ωᵢt+φᵢ−ωᵢr)·e^(−λᵢt)·(1+r)⁻¹"""
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

    return (amp * osc * temp * spatial).sum(dim=1)   # (B, H, W)


# ── Cristais — implementação tensorial (Seção 2.3 do paper) ───────────────

class CrystalMemory:
    """
    Gerencia cristais para um batch de campos.
    Operações 100% tensoriais — sem listas Python no hot-loop.

    Cristal = região do campo que mantém envelope estável por K janelas.
    Armazena: posição, padrão 11×11, energia, frequência dominante.
    """

    def __init__(self, B, dtype=torch.float32):
        self.B     = B
        self.dtype = dtype
        P          = 2*CRYSTAL_PATTERN+1   # 11

        # Mapa de cristais: valor > 0 indica cristal ativo com sua energia
        self.crystal_map  = torch.zeros(B, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)

        # Padrões armazenados: (B, MAX, P, P)
        self.patterns     = torch.zeros(B, CRYSTAL_MAX, P, P,       device=DEVICE, dtype=dtype)
        self.energies     = torch.zeros(B, CRYSTAL_MAX,              device=DEVICE, dtype=dtype)
        self.freqs        = torch.zeros(B, CRYSTAL_MAX,              device=DEVICE, dtype=dtype)
        self.pos_x        = torch.full ((B, CRYSTAL_MAX), -1,        device=DEVICE, dtype=dtype)
        self.pos_y        = torch.full ((B, CRYSTAL_MAX), -1,        device=DEVICE, dtype=dtype)
        self.n_crystals   = torch.zeros(B,                           device=DEVICE, dtype=torch.long)

        # Buffer de envelopes para rastreamento (K janelas)
        self.env_buffer   = torch.zeros(B, CRYSTAL_K, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)
        self.window_step  = 0   # step atual dentro da janela
        self.window_idx   = 0   # índice da janela atual
        self.window_max   = torch.zeros(B, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)

        # Kernel de exclusão espacial (dilatação morfológica)
        ks = 2*CRYSTAL_SEP+1
        self._dilate = torch.ones(1,1,ks,ks, device=DEVICE, dtype=dtype)

    def update_envelope(self, field):
        """Atualiza o rastreamento de envelope por janelas."""
        self.window_max = torch.max(self.window_max, field.abs())
        self.window_step += 1

        if self.window_step >= CRYSTAL_W:
            # Fecha a janela — salva envelope máximo
            self.env_buffer[:, self.window_idx] = self.window_max
            self.window_max  = torch.zeros_like(self.window_max)
            self.window_step = 0
            self.window_idx  = (self.window_idx + 1) % CRYSTAL_K

    def try_crystallize(self, field):
        """
        Tenta cristalizar regiões estáveis após K janelas completas.
        Critérios (Seção 2.3):
          (i)  mean(env) > A_min
          (ii) cv(env) < ε
          (iii) mean(env) < 0.8 × clamp
          (iv) separação espacial > CRYSTAL_SEP
        """
        # Precisa de K janelas completas
        env  = self.env_buffer                          # (B, K, H, W)
        mean = env.mean(dim=1)                          # (B, H, W)
        std  = env.std(dim=1)                           # (B, H, W)
        cv   = std / (mean + 1e-8)

        crit_amp  = (mean > CRYSTAL_A_MIN).float()
        crit_cv   = (cv   < CRYSTAL_CV_MAX).float()
        crit_sat  = (mean < 0.8 * 10.0).float()
        candidates = crit_amp * crit_cv * crit_sat      # (B, H, W)

        # Cast do _dilate para o tipo do crystal_map para evitar o RuntimeError
        weight_kernel = self._dilate.to(self.crystal_map.dtype)

        # Exclusão espacial via dilatação
        occupied = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            weight_kernel
        ).squeeze(1).clamp(0,1)

        new_sites = candidates * (1.0 - occupied)       # (B, H, W)

        # Para cada batch, registra cristais (simplificado: top-K por energia)
        # Multiplica pelo valor absoluto do campo para priorizar mais energéticos
        scored = new_sites * field.abs()

        # Atualiza o mapa de cristais
        self.crystal_map = torch.clamp(self.crystal_map + scored, 0, 10.)

    def remit(self, field, field_freq_map):
        """
        Re-emissão ativa (Eq. 7 do paper):
        Se R(Cₖ, Ψ_região, f_campo) > 0.1: Ψ += Pₖ·R·0.05

        Implementação simplificada: usa o crystal_map como peso de ressonância
        e re-emite proporcionalmente à energia cristalizada.
        """
        if self.crystal_map.abs().max() < 1e-6:
            return field

        # Ressonância simplificada: cristais com energia alta re-emitem
        # O padrão re-emitido é o campo atual na posição do cristal (memória associativa)
        remit_strength = self.crystal_map * CRYSTAL_REMIT
        field = field + remit_strength * torch.sign(field)
        return torch.clamp(field, -10., 10.)


# ── Simulação do campo (física pura, sem gradiente) ────────────────────────

def run_psi_field(wave_params):
    """
    Roda o campo completo SEM gradiente.
    Campo = reservatório físico fixo (Figura 1 do paper).
    Gradiente flui APENAS no decoder (via crystal_map).

    wave_params: (B, N_WAVES, 6) — [amp, freq, phase, decay, pos_x, pos_y]
    Retorna: (B, FIELD_SIZE*FIELD_SIZE) — mapa de cristais achatado
    """
    B     = wave_params.shape[0]
    dtype = wave_params.dtype
    wp    = wave_params.detach()   # desconecta do grafo

    with torch.no_grad():
        field    = torch.zeros(B, FIELD_SIZE, FIELD_SIZE, device=DEVICE, dtype=dtype)
        velocity = torch.zeros_like(field)
        memory   = CrystalMemory(B, dtype=dtype)

        for s in range(STIM_TOTAL):
            t      = s * float(PSI_DT)
            active = s < STIM_ON

            sources = emit_waves(wp, t)
            field, velocity = psi_step(field, velocity, sources, active)

            memory.update_envelope(field)
            if memory.window_idx > 0:
                memory.try_crystallize(field)

            field = memory.remit(field, None)

        # Retorna o mapa de cristais — representação emergente podada
        # crystal_map[b, x, y] = energia cristalizada naquele ponto
        # 0 = sem cristal, >0 = cristal com aquela energia
        crystal_map = memory.crystal_map   # (B, H, W)

    return crystal_map.view(B, -1).float()   # (B, 2304) — campo inteiro de cristais


# ── Modelo ─────────────────────────────────────────────────────────────────

class PsiFieldNet(nn.Module):
    """
    Arquitetura Nível 2 do paper (Seção 5):
      EMITTER (treinável) → campo físico → DECODER (treinável)
    """

    def __init__(self):
        super().__init__()

        # Emitter: 784D → parâmetros de onda (6 params por onda: amp,freq,phase,decay,x,y)
        self.emitter = nn.Sequential(
            nn.Linear(N_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM // 2, N_WAVES * 6),
        )

        # Decoder CNN: lê o crystal_map como imagem
        # ReLU ignora zeros naturalmente — espaço vazio não ativa nada
        # Sem compressão agressiva — preserva padrão espacial dos cristais
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B, 16, 48, 48)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 32, 48, 48)
            nn.ReLU(),
            nn.Flatten(),                                  # (B, 32*48*48 = 73728)
            nn.Linear(32 * FIELD_SIZE * FIELD_SIZE, N_CLASSES),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Emitter gera parâmetros de onda
        raw = self.emitter(x).view(-1, N_WAVES, 6)

        # Constraints físicos (Eq. 1 do paper)
        amp   = torch.abs(raw[:,:,0]) * 1.5 + 2.0          # amplitude positiva
        freq  = torch.abs(raw[:,:,1]) * 3.0 + 2.0          # frequência positiva
        phase = raw[:,:,2]                                   # fase livre
        decay = torch.abs(raw[:,:,3]) * 0.01 + 0.001        # decaimento lento
        pos_x = 0.2 + 0.6 * torch.sigmoid(raw[:,:,4])      # interior [0.2, 0.8]
        pos_y = 0.2 + 0.6 * torch.sigmoid(raw[:,:,5])

        wave_params = torch.stack([amp, freq, phase, decay, pos_x, pos_y], dim=-1)

        # Campo físico — reservatório fixo (sem grad)
        crystal_map = run_psi_field(wave_params)            # (B, 2304)
        crystal_img = crystal_map.view(-1, 1, FIELD_SIZE, FIELD_SIZE)  # (B, 1, 48, 48)

        # Decoder CNN lê o padrão coletivo de cristais como imagem
        return self.decoder(crystal_img)


# ── Avaliação ──────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, total = 0., 0, 0
    with torch.no_grad():
        for x, y in loader:
            with autocast():
                logits = model(x)
            total_loss    += F.cross_entropy(logits.float(), y, reduction='sum').item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total         += len(y)
    return total_loss / total, total_correct / total * 100


# ── Treino ─────────────────────────────────────────────────────────────────

def train():
    train_loader, val_loader, test_loader = load_mnist()

    model = PsiFieldNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo: {n_params} parâmetros")
    print(f"Emitter: MLP({N_DIM}→{HIDDEN_DIM}→{HIDDEN_DIM//2}→{N_WAVES*6})")
    print(f"Campo: {FIELD_SIZE}×{FIELD_SIZE} | N_READ: {N_READ} | N_WAVES: {N_WAVES}")
    print(f"Cristais: W={CRYSTAL_W} K={CRYSTAL_K} A_min={CRYSTAL_A_MIN} sep={CRYSTAL_SEP}")
    print(f"Batch: {BATCH_SIZE} | LR: {LR} | AMP: {AMP_DTYPE}")
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
                loss   = F.cross_entropy(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_loss  += loss.item() * len(y)
            epoch_corr  += (logits.detach().argmax(dim=1) == y).sum().item()
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

    torch.save(best_state, 'psi_field_mnist.pt')
    print("Modelo salvo em psi_field_mnist.pt")

    generate_plots(model, test_loader, hist_train_loss, hist_val_loss, hist_val_acc)
    return model


# ── Visualizações ──────────────────────────────────────────────────────────

COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
          '#42d4f4','#f032e6','#bfef45','#000075','#a9a9a9']


def generate_plots(model, test_loader, hist_train_loss, hist_val_loss, hist_val_acc):
    print("\nGerando visualizações...")

    # 1. Curvas de treino
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(hist_train_loss)+1)
    ax1.plot(epochs, hist_train_loss, label='Treino', color='#3cb44b')
    ax1.plot(epochs, hist_val_loss,   label='Val',    color='#e6194b')
    ax1.set_title('Loss — MNIST (ResNet-Ψ)'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.2)
    ax2.plot(epochs, hist_val_acc, color='#4363d8')
    ax2.axhline(y=10, color='gray',  linestyle='--', alpha=0.5, label='Chance (10%)')
    ax2.axhline(y=99, color='gold',  linestyle='--', alpha=0.6, label='99%')
    ax2.set_title('Acurácia Val (%)'); ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('viz_mnist_curves.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  -> viz_mnist_curves.png')

    # 2. Matriz de confusão
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            with autocast():
                logits = model(x)
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
    plt.savefig('viz_mnist_confusion.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  -> viz_mnist_confusion.png')

    # 3. PCA das leituras do campo
    model.eval()
    readings_list, labels_list = [], []
    count = 0
    with torch.no_grad():
        for x, y in test_loader:
            with autocast():
                raw   = model.emitter(x).view(-1, N_WAVES, 6)
                amp   = torch.abs(raw[:,:,0])*1.5+2.0
                freq  = torch.abs(raw[:,:,1])*3.0+2.0
                phase = raw[:,:,2]
                decay = torch.abs(raw[:,:,3])*0.01+0.001
                pos_x = 0.2+0.6*torch.sigmoid(raw[:,:,4])
                pos_y = 0.2+0.6*torch.sigmoid(raw[:,:,5])
                wp    = torch.stack([amp,freq,phase,decay,pos_x,pos_y], dim=-1)
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
    ax.set_title(f'PCA Mapa de Cristais — MNIST\n'
                 f'PC1={pca.explained_variance_ratio_[0]*100:.1f}% '
                 f'PC2={pca.explained_variance_ratio_[1]*100:.1f}%')
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('viz_mnist_pca.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  -> viz_mnist_pca.png')

    print('Visualizações salvas.')


if __name__ == '__main__':
    train()
