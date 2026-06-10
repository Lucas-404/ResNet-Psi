import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO E CONSTANTES FÍSICAS
# ══════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

FIELD_SIZE     = 48
PSI_DT         = 0.05
PSI_GAMMA      = 0.06
PSI_ALPHA      = 0.04
PSI_BETA       = 0.005
PSI_C2         = 0.3
STIM_ON        = 80
STIM_TOTAL     = 160

CRYSTAL_W      = 20       
CRYSTAL_K      = 3        
CRYSTAL_A_MIN  = 0.05
CRYSTAL_CV_MAX = 0.15     
CRYSTAL_SEP    = 5        
CRYSTAL_REMIT  = 0.05     
CRYSTAL_LAM    = 0.03     

_DT    = torch.tensor(PSI_DT, device=DEVICE)
_GAMMA = torch.tensor(PSI_GAMMA, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_BETA  = torch.tensor(PSI_BETA, device=DEVICE)
_C2    = torch.tensor(PSI_C2, device=DEVICE)

# Kernels
_LAP_K   = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=DEVICE).view(1, 1, 3, 3)
_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=DEVICE).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=DEVICE).view(1, 1, 3, 3)

# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS: TEXTO E PARIDADE
# ══════════════════════════════════════════════════════════════════════════════
def texts_to_spatial_tensor(texts, field_size=FIELD_SIZE):
    """V1: byte linear em grid (trunca em FS²)."""
    N = len(texts)
    max_len = field_size * field_size
    tensor = torch.zeros((N, field_size, field_size), device=DEVICE)

    for i, text in enumerate(texts):
        bytes_arr = list(str(text).encode('utf-8', 'ignore'))[:max_len]
        if not bytes_arr:
            continue
        vals = torch.tensor(bytes_arr, dtype=torch.float32, device=DEVICE) / 127.5 - 1.0
        flat_len = len(vals)
        tensor[i].view(-1)[:flat_len] = vals

    return tensor


def texts_to_spatial_tensor_v3(texts, field_size=FIELD_SIZE):
    """
    V3: Byte-Bigram Hash (densidade espacial).
    - Hash (b1*257 + b2) mod FS²  (257 coprime com 2304=2^8·3²)
    - Vetorizado com scatter_add_ (1 kernel CUDA por texto)
    - Acumula 0.5 por bigrama, clamp max=2.0 (preserva frequência, evita saturação)
    Resolve simultaneamente grid-vazio e truncamento.
    """
    N = len(texts)
    FS2 = field_size * field_size
    tensor = torch.zeros((N, field_size, field_size), device=DEVICE)

    for i, text in enumerate(texts):
        raw = str(text).encode('utf-8', 'ignore')
        n = len(raw)
        if n == 0:
            continue
        if n == 1:
            h = (raw[0] * 257) % FS2
            tensor[i].view(-1)[h] = 1.0
            continue

        b = torch.tensor(list(raw), device=DEVICE, dtype=torch.long)
        h = (b[:-1] * 257 + b[1:]) % FS2
        flat = tensor[i].view(-1)
        flat.scatter_add_(0, h, torch.full((n - 1,), 0.5, device=DEVICE, dtype=torch.float32))
        flat.clamp_(max=2.0)

    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# V4: Projeção semântica (MiniLM + JL matrix + Top-K esparso)
# ──────────────────────────────────────────────────────────────────────────────
# Pipeline:  T → MiniLM (384-d) → W·e (JL, 2304-d) → Top-K → grid 48×48
# Preserva similaridade semântica (JL), mantém esparsidade pro campo respirar.
# ──────────────────────────────────────────────────────────────────────────────
_MINILM = None
_W_JL   = None
V4_TOPK = 50  # pode ser sobrescrito antes de chamar

def _get_minilm():
    global _MINILM
    if _MINILM is None:
        from sentence_transformers import SentenceTransformer
        _MINILM = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',
                                       device=str(DEVICE))
    return _MINILM

def _get_jl_matrix(d_in, d_out):
    global _W_JL
    if _W_JL is None or _W_JL.shape != (d_out, d_in):
        g = torch.Generator(device=DEVICE).manual_seed(42)
        _W_JL = torch.randn(d_out, d_in, generator=g, device=DEVICE) / (d_in ** 0.5)
    return _W_JL

def texts_to_spatial_tensor_v4(texts, field_size=FIELD_SIZE, topk=None):
    """
    V4: MiniLM → JL → Top-K esparso → grid FS×FS, valores positivos [0, 2].
    Preserva similaridade semântica. Esparso pra não saturar o campo.
    """
    k = topk if topk is not None else V4_TOPK
    model = _get_minilm()
    emb = model.encode(list(texts), convert_to_tensor=True, show_progress_bar=False,
                       device=str(DEVICE))                     # (N, 384)
    N, d = emb.shape
    FS2 = field_size * field_size
    W = _get_jl_matrix(d, FS2)                                  # (FS2, 384)
    S = torch.abs(emb @ W.T)                                    # (N, FS2)

    # Top-K por linha, zera o resto
    top_vals, top_idx = torch.topk(S, k, dim=1)                 # (N, K)
    grid = torch.zeros_like(S)
    grid.scatter_(1, top_idx, top_vals)

    # Normaliza por texto pra faixa [0, 2]
    row_max = grid.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    grid = grid / row_max * 2.0
    return grid.view(N, field_size, field_size)


PROJECTIONS = {
    'v1': texts_to_spatial_tensor,
    'v3': texts_to_spatial_tensor_v3,
    'v4': texts_to_spatial_tensor_v4,
}

def mask_parity_8bit(tensor):
    """Filtra blocos baseados na verificação de paridade par em 8 bits."""
    q_tensor = (tensor * 127).to(torch.int32)
    parity = torch.zeros_like(q_tensor)
    for i in range(8):
        parity += (q_tensor >> i) & 1
    return (parity % 2 == 0).float()

# ══════════════════════════════════════════════════════════════════════════════
# CRISTALIZAÇÃO COMPETITIVA (ESTRITA / BOOLEANA)
# ══════════════════════════════════════════════════════════════════════════════
class CrystalCompetitivo:
    def __init__(self, B, FS=FIELD_SIZE, decay=0.02, ressonance_boost=0.1):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
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

        # Thresholds lógicos (Heaviside Theta)
        amp_mask = (mean > CRYSTAL_A_MIN).float()
        cv_mask  = (cv < CRYSTAL_CV_MAX).float()
        sat_mask = (mean < 8.0).float()
        cand = amp_mask * cv_mask * sat_mask

        occ = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0, 1), (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0, 1)

        new_crystals = cand * (1.0 - occ) * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + new_crystals, 0, 10.)

        self.crystal_hp = torch.where(new_crystals > 0.01, torch.clamp(self.crystal_hp + 1.0, 0, 5.0), self.crystal_hp)
        ressonance = field.abs() * (self.crystal_map > 0.01).float()
        self.crystal_hp = self.crystal_hp + ressonance * self.ressonance_boost - self.decay

        alive = (self.crystal_hp > 0).float()
        self.crystal_map = self.crystal_map * alive
        self.crystal_hp  = torch.clamp(self.crystal_hp * alive, 0, 5.0)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6: return field
        return torch.clamp(field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)

# ══════════════════════════════════════════════════════════════════════════════
# EQUAÇÃO DE ONDA + 4 MODOS DE ATENÇÃO
# ══════════════════════════════════════════════════════════════════════════════
#   'none'   — só física (baseline)
#   'sobel'  — Ressonância guiada pelo gradiente (Sobel)  [gate: crystal_map]
#   'energy' — Equilíbrio energético  Q=|v|·dt, K=|Ψ|     [sem gate, element-wise]
#   'ngram'  — Autocorrelação espacial via torch.roll     [sem gate, detecta repetição]
# ══════════════════════════════════════════════════════════════════════════════
def psi_step(field, velocity, source_term, crystal_map, lam, mode='sobel'):
    field = field + source_term
    field_pad = F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular')

    # Física local
    lap = F.conv2d(field_pad, _LAP_K.to(field.dtype)).squeeze(1)
    acc = (_C2 * lap - _GAMMA * velocity + _ALPHA * torch.tanh(field) * field - _BETA * field ** 3)

    # Value filtrado por paridade (usado em todos os modos de atenção)
    V_masked = field * mask_parity_8bit(field)

    if mode == 'sobel':
        grad_x = F.conv2d(field_pad, _SOBEL_X.to(field.dtype)).squeeze(1)
        grad_y = F.conv2d(field_pad, _SOBEL_Y.to(field.dtype)).squeeze(1)
        Q = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        K = lap.abs()
        R = torch.exp(-0.1 * (Q - K) ** 2)
        coupling = lam * crystal_map * R * V_masked

    elif mode == 'energy':
        Q = velocity.abs() * _DT
        K = field.abs()
        R = torch.exp(-0.1 * (Q - K).abs())
        coupling = lam * R * V_masked

    elif mode == 'ngram':
        Kx = torch.roll(V_masked, shifts=1, dims=-1)
        Ky = torch.roll(V_masked, shifts=1, dims=-2)
        R = V_masked * Kx + V_masked * Ky
        coupling = lam * R

    else:  # 'none'
        coupling = torch.zeros_like(field)

    acc = acc + coupling
    velocity = torch.clamp(velocity + acc * _DT, -5., 5.)
    field    = torch.clamp(field + velocity * _DT, -10., 10.)
    return field, velocity, coupling.abs().max()

# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO E CLASSIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def compute_crystal_maps_text(texts, bs=256, lam=None, mode='sobel',
                              projection='v3', verbose=False):
    X = PROJECTIONS[projection](texts)
    N = len(X)
    out = []
    lam_t = torch.tensor(CRYSTAL_LAM if lam is None else lam,
                         device=DEVICE, dtype=torch.float32)

    stats = {'coup_max_global': 0.0, 'coup_max_mean': 0.0, 'n_batches': 0}

    for i in range(0, N, bs):
        pert = X[i:i+bs]
        B = len(pert)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(B, FIELD_SIZE)

        src_on  = pert * 0.5
        src_off = torch.zeros_like(src_on)

        coup_hist = []
        with torch.no_grad():
            for s in range(STIM_TOTAL):
                src = src_on if s < STIM_ON else src_off
                f, v, coup_max = psi_step(f, v, src, mem.crystal_map, lam_t, mode=mode)
                coup_hist.append(coup_max.item())
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)

        stats['coup_max_global'] = max(stats['coup_max_global'], max(coup_hist))
        stats['coup_max_mean'] += sum(coup_hist) / len(coup_hist)
        stats['n_batches'] += 1
        out.append(mem.crystal_map)

    if verbose and stats['n_batches'] > 0:
        print(f"  [proj={projection} mode={mode} lam={lam_t.item():.3f}] "
              f"coup.max_global={stats['coup_max_global']:.4f}  "
              f"coup.max_mean={stats['coup_max_mean']/stats['n_batches']:.4f}  "
              f"src.max={X.max().item():.2f}  src.mean={X.mean().item():.3f}")
    return torch.cat(out, dim=0)

def classify_euclidean(crystal_maps, prototypes):
    n_classes = len(prototypes)
    N = len(crystal_maps)
    proto_stack = torch.stack([prototypes[c].flatten() for c in range(n_classes)])
    cmaps_flat = crystal_maps.view(N, -1)
    dists = torch.cdist(cmaps_flat.unsqueeze(0), proto_stack.unsqueeze(0)).squeeze(0)
    return dists.argmin(dim=1).cpu().numpy()

class ResNetPsiText:
    def __init__(self, n_classes, lam=None, mode='sobel', projection='v3'):
        self.n_classes = n_classes
        self.prototypes = {}
        self.lam = lam
        self.mode = mode
        self.projection = projection

    def extract(self, texts, bs=64, verbose=False):
        return compute_crystal_maps_text(texts, bs, lam=self.lam,
                                         mode=self.mode, projection=self.projection,
                                         verbose=verbose)

    def fit(self, train_texts, train_labels, bs=64):
        cmaps = self.extract(train_texts, bs, verbose=True)
        labels_np = np.array(train_labels)
        
        for cls in range(self.n_classes):
            mask = (labels_np == cls)
            if mask.any():
                self.prototypes[cls] = cmaps[mask].mean(dim=0)
        return self

    def predict(self, test_texts, bs=64):
        cmaps = self.extract(test_texts, bs)
        return classify_euclidean(cmaps, self.prototypes)