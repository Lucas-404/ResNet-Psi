"""
ResNet-Ψ — Curva de Escala: B(N) → Acurácia

Campos de 48x48 até 683x683 (equivalente a 512 tokens de contexto).
Mesmo dado (MNIST), mesmo decoder linear, só o campo muda.
Mede: bits físicos vs acurácia — lei de escala empírica do PsiField.
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

from torchvision import datasets, transforms

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ── Constantes físicas ────────────────────────────────────────────────────────
PSI_DT     = 0.05
PSI_GAMMA  = 0.06
PSI_ALPHA  = 0.04
PSI_BETA   = 0.005
PSI_C2     = 0.3
STIM_ON    = 40
STIM_TOTAL = 80

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

# Lei de escala medida
BITS_PER_CRYSTAL = 2.32
SCALE_A          = 0.0024
SCALE_B          = 2.171

def estimated_bits(N):
    return BITS_PER_CRYSTAL * SCALE_A * (N ** SCALE_B)

# ── Física ────────────────────────────────────────────────────────────────────

class CrystalMem:
    def __init__(self, B, FS, dtype=torch.float32):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE, dtype=dtype)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE, dtype=dtype)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE, dtype=dtype)
        ks = 2 * CRYSTAL_SEP + 1
        self._dilate = torch.ones(1, 1, ks, ks, device=DEVICE, dtype=dtype)

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
        cand = ((mean > CRYSTAL_A_MIN) & (cv < CRYSTAL_CV_MAX) & (mean < 8.0)).float()
        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0,1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0,1)
        self.crystal_map = torch.clamp(
            self.crystal_map + cand*(1.0-occ)*field.abs(), 0, 10.)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


def psi_step(field, velocity, sources, active):
    lap_k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                          device=DEVICE).view(1,1,3,3).to(field.dtype)
    if active:
        field = field + sources * (_DT * 0.1)
    lap      = F.conv2d(F.pad(field.unsqueeze(1),(1,1,1,1),mode='circular'), lap_k).squeeze(1)
    acc      = _C2*lap - _GAMMA*velocity + _ALPHA*torch.tanh(field)*field - _BETA*field*field**2
    velocity = torch.clamp(velocity + acc*_DT, -5., 5.)
    field    = torch.clamp(field + velocity*_DT, -10., 10.)
    return field, velocity


# ── Encoder ───────────────────────────────────────────────────────────────────

def build_gaussians(field_size, sigma=0.04):
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []
    for pi in range(28):
        for pj in range(28):
            cx = 0.1 + 0.8 * pi / 27.0
            cy = 0.1 + 0.8 * pj / 27.0
            gs.append(torch.exp(-((xg-cx)**2+(yg-cy)**2)/(2*sigma**2)))
    return torch.stack(gs).view(784, -1)   # (784, FS²)


def precompute(X, PG, FS, desc, bs=64):
    """
    Pré-computa crystal_maps e guarda em float16 na GPU.
    float16 vs float32: metade da memória, sem perda relevante para decoder linear.
    """
    N, out, t0 = len(X), [], time.time()
    for i in range(0, N, bs):
        B    = X[i:i+bs]
        pert = (B.view(len(B),784) @ PG.to(B.dtype)).view(len(B), FS, FS)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem  = CrystalMem(len(B), FS, dtype=f.dtype)
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                f, v = psi_step(f, v, pert, s < STIM_ON)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)
        # float16 na GPU — metade da RAM, sem sair da VRAM
        out.append(mem.crystal_map.view(len(B),-1).half())
        if (i//bs) % 10 == 0:
            print(f"  {desc}: {min(i+bs,N)}/{N} ({min(i+bs,N)/N*100:.0f}%)  {time.time()-t0:.0f}s", end='\r')
    print(f"  {desc}: {N}/{N} (100%)  {time.time()-t0:.1f}s      ")
    return torch.cat(out, dim=0)   # (N, FS²) float16 na GPU


# ── MNIST ─────────────────────────────────────────────────────────────────────

def load_mnist():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,),(0.3081,))])
    tr = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    te = datasets.MNIST('./data', train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0].squeeze(0) for i in range(len(tr))])
    Ytr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    Xte = torch.stack([te[i][0].squeeze(0) for i in range(len(te))])
    Yte = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)
    torch.manual_seed(42)
    perm = torch.randperm(len(Xtr))
    Xva, Yva = Xtr[perm[-10000:]], Ytr[perm[-10000:]]
    Xtr, Ytr = Xtr[perm[:-10000]], Ytr[perm[:-10000]]
    print(f"Treino:{len(Xtr)} Val:{len(Xva)} Teste:{len(Xte)}")
    return Xtr.to(DEVICE), Ytr, Xva.to(DEVICE), Yva, Xte.to(DEVICE), Yte


# ── Treino do decoder ─────────────────────────────────────────────────────────

def train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte, FS,
                  max_ep=60, patience=10, bs=512, lr=1e-3):
    dec = nn.Linear(FS*FS, 10).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    crit = nn.CrossEntropyLoss()

    # crystal_maps ficam em float16 na GPU — converte para float32 por batch
    Ytr_d = Ytr.to(DEVICE)
    Yva_d = Yva.to(DEVICE)
    Yte_d = Yte.to(DEVICE)

    best_val, best_sd, pat = 0.0, None, 0
    for ep in range(1, max_ep+1):
        dec.train()
        perm = torch.randperm(len(CMtr), device=DEVICE)
        for i in range(0, len(CMtr), bs):
            idx = perm[i:i+bs]
            opt.zero_grad(set_to_none=True)
            crit(dec(CMtr[idx].float()), Ytr_d[idx]).backward()
            opt.step()
        sch.step()
        dec.eval()
        with torch.no_grad():
            va_correct = 0
            for i in range(0, len(CMva), bs):
                va_correct += (dec(CMva[i:i+bs].float()).argmax(1) == Yva_d[i:i+bs]).sum().item()
            va = va_correct / len(CMva) * 100
        if va > best_val:
            best_val = va
            best_sd  = {k:v.clone() for k,v in dec.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break
    dec.load_state_dict(best_sd)
    dec.eval()
    with torch.no_grad():
        te_correct = 0
        for i in range(0, len(CMte), bs):
            te_correct += (dec(CMte[i:i+bs].float()).argmax(1) == Yte_d[i:i+bs]).sum().item()
        te_acc = te_correct / len(CMte) * 100
    return best_val, te_acc, ep


# ── Treino on-the-fly para campos grandes (sem pré-computar tudo) ─────────────

def field_batch(X_batch, PG, FS):
    """Computa crystal_map de um batch sem guardar estado."""
    flat = X_batch.view(len(X_batch), 784)
    pert = (flat @ PG.to(flat.dtype)).view(len(X_batch), FS, FS)
    f, v = pert.clone(), torch.zeros_like(pert)
    mem  = CrystalMem(len(X_batch), FS, dtype=f.dtype)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            f, v = psi_step(f, v, pert, s < STIM_ON)
            mem.update_envelope(f)
            if mem.window_idx > 0:
                mem.try_crystallize(f)
            f = mem.remit(f)
    return mem.crystal_map.view(len(X_batch), -1).float()


def train_onthefly(Xtr, Ytr, Xva, Yva, Xte, Yte, PG, FS,
                   max_ep=60, patience=10, bs=32, lr=1e-3):
    """
    Treina decoder processando campo on-the-fly por batch.
    RAM constante = bs × FS² × 4 bytes independente do dataset.
    """
    dec  = nn.Linear(FS*FS, 10).to(DEVICE)   # float32
    opt  = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    crit = nn.CrossEntropyLoss()

    Ytr_d = Ytr.to(DEVICE)
    Yva_d = Yva.to(DEVICE)
    Yte_d = Yte.to(DEVICE)

    best_val, best_sd, pat = 0.0, None, 0
    n_crys_acc, n_crys_n   = 0.0, 0

    for ep in range(1, max_ep+1):
        dec.train()
        perm = torch.randperm(len(Xtr))
        ep_loss = 0.0
        for i in range(0, len(Xtr), bs):
            idx  = perm[i:i+bs]
            cm   = field_batch(Xtr[idx], PG, FS)   # (bs, FS²) — computa agora
            if ep == 1:   # conta cristais só na 1ª época
                n_crys_acc += (cm > 0.01).float().sum(dim=1).mean().item()
                n_crys_n   += 1
            opt.zero_grad(set_to_none=True)
            crit(dec(cm.to(DEVICE)), Ytr_d[idx]).backward()
            opt.step()
            del cm

        sch.step()

        # Validação on-the-fly
        dec.eval()
        correct, total = 0, 0
        for i in range(0, len(Xva), bs):
            cm = field_batch(Xva[i:i+bs], PG, FS)
            with torch.no_grad():
                correct += (dec(cm.to(DEVICE)).argmax(1) == Yva_d[i:i+bs]).sum().item()
            total += len(Xva[i:i+bs])
            del cm
        va = correct / total * 100
        print(f"    ep {ep:>3}: val={va:.2f}%", end='\r')

        if va > best_val:
            best_val = va
            best_sd  = {k:v.clone() for k,v in dec.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break

    print()
    dec.load_state_dict(best_sd)
    dec.eval()
    correct, total = 0, 0
    for i in range(0, len(Xte), bs):
        cm = field_batch(Xte[i:i+bs], PG, FS)
        with torch.no_grad():
            correct += (dec(cm.to(DEVICE)).argmax(1) == Yte_d[i:i+bs]).sum().item()
        total += len(Xte[i:i+bs])
        del cm
    te_acc = correct / total * 100

    n_crys = n_crys_acc / max(n_crys_n, 1)
    bits_r = BITS_PER_CRYSTAL * n_crys
    print(f"  Cristais: {n_crys:.1f} | Bits reais: {bits_r:.1f}")
    return best_val, te_acc, ep, n_crys, bits_r


# ── Experimento ───────────────────────────────────────────────────────────────

# crystal_maps em float16 (2 bytes/posição).
# Limiar dinâmico: usa VRAM disponível para decidir se pré-computa ou vai on-the-fly.
# N²×70000×2 bytes < 0.7×VRAM_livre → pré-computa; senão on-the-fly.
import os

# Com float16 na GPU: N²×70000×2 bytes
# RTX PRO 6000 Blackwell (96GB VRAM): 683×683×70000×2 = 65GB → cabe
# Limiar alto: pré-computa tudo até 683
RAM_LIMIT_FS = 700  # pré-computa todos os tamanhos testados
print(f"Limite para pré-computação: {RAM_LIMIT_FS} (VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)" if torch.cuda.is_available() else f"Limite CPU: {RAM_LIMIT_FS}")

FIELD_SIZES = [48, 64, 96, 128, 192, 256, 384, 512, 683]

print("\nCarregando MNIST...")
Xtr, Ytr, Xva, Yva, Xte, Yte = load_mnist()

print(f"\n{'='*65}")
print("CURVA DE ESCALA: B(N) → Acurácia MNIST")
print(f"Campos: {FIELD_SIZES}")
print(f"{'='*65}\n")

results = []

for fs in FIELD_SIZES:
    bits_est  = estimated_bits(fs)
    mem_gb    = fs*fs * 60000 * 4 / 1e9
    on_the_fly = fs > RAM_LIMIT_FS
    print(f"━━ Campo {fs}×{fs}  |  ~{bits_est:.0f} bits  |  RAM estimada: {mem_gb:.1f}GB  |  {'on-the-fly' if on_the_fly else 'pré-computa'}")

    PG = build_gaussians(fs)

    if not on_the_fly:
        # Pré-computa crystal_maps (campos pequenos)
        t0   = time.time()
        CMtr = precompute(Xtr, PG, fs, "Treino", bs=128)
        CMva = precompute(Xva, PG, fs, "Val",    bs=256)
        CMte = precompute(Xte, PG, fs, "Teste",  bs=256)
        t_field = time.time() - t0
        n_crys  = (CMva > 0.01).float().sum(dim=1).mean().item()
        bits_r  = BITS_PER_CRYSTAL * n_crys
        print(f"  Campo: {t_field:.1f}s | Cristais: {n_crys:.1f} | Bits reais: {bits_r:.1f}")

        t1 = time.time()
        val_acc, test_acc, ep = train_decoder(CMtr, Ytr, CMva, Yva, CMte, Yte, fs)
        del CMtr, CMva, CMte
        torch.cuda.empty_cache()
        print(f"  Val: {val_acc:.2f}%  Teste: {test_acc:.2f}%  ({ep} épocas, {time.time()-t1:.1f}s)\n")

    else:
        # On-the-fly: computa crystal_map por batch, treina imediatamente
        # Não guarda nada na RAM além do batch atual
        val_acc, test_acc, ep, n_crys, bits_r = train_onthefly(
            Xtr, Ytr, Xva, Yva, Xte, Yte, PG, fs)
        print(f"  Val: {val_acc:.2f}%  Teste: {test_acc:.2f}%  ({ep} épocas)\n")

    results.append({
        'fs': fs, 'n2': fs*fs,
        'bits_real': bits_r, 'n_crys': n_crys,
        'val_acc': val_acc, 'test_acc': test_acc,
    })
    # Libera memória entre tamanhos
    del PG
    torch.cuda.empty_cache()

# ── Resumo ────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESULTADO FINAL")
print(f"{'='*70}")
print(f"\n{'Campo':>8}  {'Bits reais':>11}  {'Cristais':>9}  {'Val%':>7}  {'Teste%':>7}  {'Tokens eq.':>11}")
print("-"*65)
for r in results:
    tokens = r['bits_real'] / np.log2(50000)
    print(f"  {r['fs']:>3}×{r['fs']:<3}  "
          f"{r['bits_real']:>11.1f}  "
          f"{r['n_crys']:>9.1f}  "
          f"{r['val_acc']:>7.2f}%  "
          f"{r['test_acc']:>7.2f}%  "
          f"{tokens:>11.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('ResNet-Ψ — Lei de Escala: Bits Físicos vs Acurácia MNIST', fontsize=13)

bits  = [r['bits_real']  for r in results]
accs  = [r['test_acc']   for r in results]
crys  = [r['n_crys']     for r in results]
n2s   = [r['n2']         for r in results]
toks  = [r['bits_real']/np.log2(50000) for r in results]

# 1. Bits → Acurácia
axes[0].plot(bits, accs, 'o-', color='#e6194b', lw=2.5, ms=8)
for r in results:
    axes[0].annotate(f"{r['fs']}×{r['fs']}",
                     (r['bits_real'], r['test_acc']),
                     textcoords="offset points", xytext=(5,-12), fontsize=7)
axes[0].axhline(92, color='gray', ls='--', alpha=0.5, label='Reg. logística ~92%')
axes[0].set_xlabel('Bits físicos B(N)')
axes[0].set_ylabel('Acurácia no teste (%)')
axes[0].set_title('Lei de Escala: Bits → Acurácia')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# 2. Cristais vs N² (log-log)
axes[1].loglog(n2s, [max(c,1) for c in crys], 'o-', color='#3cb44b', lw=2, ms=7)
log_n2 = np.log(n2s)
log_c  = np.log([max(c,1) for c in crys])
b2, la2 = np.polyfit(log_n2, log_c, 1)
n2r = np.array(n2s)
axes[1].loglog(n2r, np.exp(la2)*n2r**b2, '--', color='gray', label=f'N^{b2:.2f}')
for r in results:
    axes[1].annotate(f"{r['fs']}",
                     (r['n2'], max(r['n_crys'],1)),
                     textcoords="offset points", xytext=(3,3), fontsize=7)
axes[1].set_xlabel('N² (área do campo)')
axes[1].set_ylabel('Cristais')
axes[1].set_title('Cristais vs Área (log-log)')
axes[1].legend(); axes[1].grid(alpha=0.3, which='both')

# 3. Tokens → Acurácia
axes[2].plot(toks, accs, 'o-', color='#4363d8', lw=2.5, ms=8)
for r, t in zip(results, toks):
    axes[2].annotate(f"{r['fs']}×{r['fs']}",
                     (t, r['test_acc']),
                     textcoords="offset points", xytext=(5,-12), fontsize=7)
axes[2].axhline(92, color='gray', ls='--', alpha=0.5, label='Reg. logística ~92%')
axes[2].axvline(512, color='orange', ls=':', alpha=0.7, label='512 tokens')
axes[2].set_xlabel('Contexto equivalente (tokens, vocab 50k)')
axes[2].set_ylabel('Acurácia no teste (%)')
axes[2].set_title('Tokens de Contexto vs Acurácia')
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz_scaling_curve.png', dpi=130, bbox_inches='tight')
plt.close()
print("\n-> viz_scaling_curve.png")
print("Pronto.")
