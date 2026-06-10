"""
ResNet-Ψ — Base para projetos futuros

Dinâmica ondulatória + cristalização competitiva produz representações
que classificam dados sem nenhum parâmetro treinado.
Os cristais são os pesos: cada classe deixa uma assinatura física distinta no campo.

Resultados validados:
  - MNIST:         77.4% (zero treino) | 88.1% (linear) | 93.1% (MLP)
  - Fashion-MNIST: 67.0% (zero treino)

Uso rápido (zero treino):
    from resnet_psi import ResNetPsi
    rn = ResNetPsi()
    rn.fit(train_images, train_labels)       # só computa protótipos (sem treino)
    preds = rn.predict(test_images)          # classifica por distância euclidiana
    acc = (preds == test_labels).mean()

Uso com decoder (treino só no decoder):
    rn = ResNetPsi()
    cmaps_train = rn.extract(train_images)   # crystal_maps (B, 48, 48)
    cmaps_test  = rn.extract(test_images)
    # treinar qualquer classificador em cima dos cmaps

Domínio: dados com estrutura geométrica (contornos, formas, silhuetas).
Não funciona para imagens naturais complexas (CIFAR-10 = 18.7%).

Acoplamento cristalino (crystal attention):
    Cristais mediam comunicação global entre regiões do campo.
    acc[i] += λ × crystal[i] × (Σ_j crystal[j]×field[j] − field[i] × Σ_j crystal[j])
    Factoriza O(N²) → O(N): dois sums globais + elementwise. Completamente paralelo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS
# ══════════════════════════════════════════════════════════════════════════════

FIELD_SIZE     = 48       # campo 48×48
PSI_DT         = 0.05     # passo temporal
PSI_GAMMA      = 0.06     # amortecimento (damping)
PSI_ALPHA      = 0.04     # não-linearidade (tanh)
PSI_BETA       = 0.005    # dissipação cúbica
PSI_C2         = 0.3      # velocidade de onda (Laplaciano)
STIM_ON        = 40       # steps com estímulo ativo
STIM_TOTAL     = 80       # steps totais

# Cristalização
CRYSTAL_W      = 20       # janela de envelope (steps)
CRYSTAL_K      = 3        # janelas para cristalizar
CRYSTAL_A_MIN  = 0.3      # amplitude mínima
CRYSTAL_CV_MAX = 0.15     # coeficiente de variação máximo
CRYSTAL_SEP    = 5        # separação mínima entre cristais (pixels)
CRYSTAL_REMIT  = 0.05     # força de re-emissão
CRYSTAL_LAM    = 0.03     # acoplamento cristalino (crystal attention)

# Tensores pré-computados (fora dos steps — não recriar a cada chamada)
_DT    = torch.tensor(PSI_DT,    device=DEVICE)
_GAMMA = torch.tensor(PSI_GAMMA, device=DEVICE)
_ALPHA = torch.tensor(PSI_ALPHA, device=DEVICE)
_BETA  = torch.tensor(PSI_BETA,  device=DEVICE)
_C2    = torch.tensor(PSI_C2,    device=DEVICE)
_LAP_K = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                        device=DEVICE).view(1, 1, 3, 3)


# ══════════════════════════════════════════════════════════════════════════════
# CRISTALIZAÇÃO COMPETITIVA
# ══════════════════════════════════════════════════════════════════════════════

class CrystalCompetitivo:
    """
    Cristalização competitiva: o mecanismo que funciona.

    - Thresholds suaves (sigmoid) ao invés de duros
    - Cristais ganham HP quando ressoam com a onda
    - Cristais perdem HP por decay constante
    - Cristais com HP <= 0 morrem (seleção natural)
    - Exclusão espacial: cristais não podem nascer perto de outros
    """

    def __init__(self, B, FS=FIELD_SIZE, sharpness=5.0, decay=0.02, ressonance_boost=0.1):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.crystal_hp  = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
        self.sharpness = sharpness
        self.decay = decay
        self.ressonance_boost = ressonance_boost
        ks = 2 * CRYSTAL_SEP + 1
        self._dilate = torch.ones(1, 1, ks, ks, device=DEVICE)

    def update_envelope(self, field):
        """Rastreia envelope de amplitude por janelas temporais."""
        self.window_max = torch.max(self.window_max, field.abs())
        self.window_step += 1
        if self.window_step >= CRYSTAL_W:
            self.env_buffer[:, self.window_idx] = self.window_max
            self.window_max  = torch.zeros_like(self.window_max)
            self.window_step = 0
            self.window_idx  = (self.window_idx + 1) % CRYSTAL_K

    def try_crystallize(self, field):
        """Tenta cristalizar regiões estáveis. Cristais competem por sobrevivência."""
        env  = self.env_buffer
        mean = env.mean(dim=1)
        cv   = env.std(dim=1) / (mean + 1e-8)

        # Thresholds suaves via sigmoid
        amp_score = torch.sigmoid(self.sharpness * (mean - CRYSTAL_A_MIN))
        cv_score  = torch.sigmoid(self.sharpness * (CRYSTAL_CV_MAX - cv))
        sat_score = torch.sigmoid(self.sharpness * (8.0 - mean))
        cand = amp_score * cv_score * sat_score

        # Exclusão espacial via dilatação morfológica
        occ = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0, 1),
                  (CRYSTAL_SEP,)*4, mode='circular'),
            self._dilate).squeeze(1).clamp(0, 1)

        # Novos cristais
        new_crystals = cand * (1.0 - occ) * field.abs()
        self.crystal_map = torch.clamp(self.crystal_map + new_crystals, 0, 10.)

        # Novos cristais nascem com HP = 1
        self.crystal_hp = torch.where(
            new_crystals > 0.01,
            torch.clamp(self.crystal_hp + 1.0, 0, 5.0),
            self.crystal_hp)

        # Competição: ressonância = onda forte onde há cristal
        ressonance = field.abs() * (self.crystal_map > 0.01).float()
        self.crystal_hp = self.crystal_hp + ressonance * self.ressonance_boost
        self.crystal_hp = self.crystal_hp - self.decay

        # Cristais com HP <= 0 morrem
        alive = (self.crystal_hp > 0).float()
        self.crystal_map = self.crystal_map * alive
        self.crystal_hp  = torch.clamp(self.crystal_hp * alive, 0, 5.0)

    def remit(self, field):
        """Re-emissão: cristais injetam energia de volta no campo."""
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


# ══════════════════════════════════════════════════════════════════════════════
# EQUAÇÃO DE ONDA
# ══════════════════════════════════════════════════════════════════════════════

@torch.compile(mode='default')
def psi_step(field, velocity, source_term, crystal_map):
    """
    Um step da equação de onda + acoplamento cristalino.

        acc = c²∇²Ψ − γ·v + α·tanh(Ψ)·Ψ − β·Ψ³
            + λ · crystal · (Σ_j crystal_j·field_j − field · Σ_j crystal_j)
        v += acc·dt
        Ψ += v·dt

    Args:
        field:       (B, H, W)
        velocity:    (B, H, W)
        source_term: (B, H, W) — perturbação pré-multiplicada por dt*0.1 (zeros se inativo)
        crystal_map: (B, H, W) — mapa de cristais atual (zeros antes de cristalizar)

    O acoplamento cristalino é O(N) e completamente paralelo:
        S_cf = Σ_j crystal_j × field_j   (sum global, shape (B,1))
        S_c  = Σ_j crystal_j              (sum global, shape (B,1))
        coupling[i] = crystal[i] × (S_cf − field[i] × S_c)
    Regiões cristalizadas comunicam-se diretamente, independente de distância.
    """
    B, H, W = field.shape
    N = H * W

    # Estímulo (zero quando inativo — passado como zeros pelo chamador)
    field = field + source_term

    # ── Física local: Laplaciano (propagação de vizinhos) ────────────────
    lap = F.conv2d(
        F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular'),
        _LAP_K.to(field.dtype),
    ).squeeze(1)

    acc = (_C2 * lap
           - _GAMMA * velocity
           + _ALPHA * torch.tanh(field) * field
           - _BETA  * field * field ** 2)

    # ── Acoplamento cristalino: comunicação global via cristais ──────────
    # crystal_map = zeros antes de qualquer cristal existir → coupling = 0
    f_flat = field.view(B, N)
    c_flat = crystal_map.view(B, N)
    S_cf   = (c_flat * f_flat).sum(dim=1, keepdim=True)   # (B, 1)
    S_c    =  c_flat.sum(dim=1, keepdim=True)              # (B, 1)
    coupling = (CRYSTAL_LAM * c_flat * (S_cf - f_flat * S_c)).view(B, H, W)
    acc = acc + coupling

    velocity = torch.clamp(velocity + acc * _DT, -5., 5.)
    field    = torch.clamp(field    + velocity * _DT, -10., 10.)
    return field, velocity


# ══════════════════════════════════════════════════════════════════════════════
# PROJEÇÃO: PIXELS → CAMPO
# ══════════════════════════════════════════════════════════════════════════════

def build_gaussians(input_shape, field_size=FIELD_SIZE, sigma=0.04):
    """
    Matriz de projecao generica: cada elemento da entrada vira uma gaussiana no campo 2D.

    input_shape pode ser:
      - int ou (int,)      -> entrada 1D (sinal temporal, audio, ECG)
                             gaussianas distribuidas numa linha horizontal
      - (int, int)         -> entrada 2D (imagem)
                             gaussianas distribuidas num grid

    Retorna: (n_elementos, field_size²)
    """
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gs = []

    if len(input_shape) == 1:
        # Entrada 1D: distribui ao longo da linha central do campo
        N = input_shape[0]
        cy = 0.5  # linha central (eixo y fixo)
        for i in range(N):
            cx = 0.1 + 0.8 * i / max(N - 1, 1)
            gs.append(torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2)))

    elif len(input_shape) == 2:
        # Entrada 2D: distribui num grid
        H, W = input_shape
        for pi in range(H):
            for pj in range(W):
                cx = 0.1 + 0.8 * pi / max(H - 1, 1)
                cy = 0.1 + 0.8 * pj / max(W - 1, 1)
                gs.append(torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2)))

    else:
        raise ValueError(f"input_shape deve ser 1D ou 2D, recebeu: {input_shape}")

    n_elementos = 1
    for d in input_shape:
        n_elementos *= d
    return torch.stack(gs).view(n_elementos, -1)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE: ENTRADA → CRYSTAL MAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_crystal_maps(X, PG, field_size=FIELD_SIZE, bs=256):
    """
    Pipeline completo: entrada → crystal_maps.

    Args:
        X:  (N, ...) tensor — qualquer forma: (N, L) para 1D, (N, H, W) para 2D
        PG: matriz de projeção gaussiana (n_elementos, field_size²)
        field_size: tamanho do campo
        bs: batch size

    Returns:
        (N, field_size, field_size) tensor de crystal_maps
    """
    N = len(X)
    n_pixels = PG.shape[0]
    out = []

    for i in range(0, N, bs):
        batch = X[i:i+bs]
        B = len(batch)
        pert = (batch.view(B, n_pixels) @ PG.to(batch.dtype)).view(B, field_size, field_size)
        f, v = pert.clone(), torch.zeros_like(pert)
        mem = CrystalCompetitivo(B, field_size)

        # Pré-computa source_term para cada fase (evita branch dentro do loop)
        src_on  = pert * (_DT * 0.1)          # (B, H, W) — ativo
        src_off = torch.zeros_like(src_on)     # (B, H, W) — inativo

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                src = src_on if s < STIM_ON else src_off
                f, v = psi_step(f, v, src, mem.crystal_map)
                mem.update_envelope(f)
                if mem.window_idx > 0:
                    mem.try_crystallize(f)
                f = mem.remit(f)

        out.append(mem.crystal_map)

    return torch.cat(out, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO POR PROTÓTIPOS (ZERO TREINO)
# ══════════════════════════════════════════════════════════════════════════════

def build_prototypes(crystal_maps, labels, n_classes):
    """
    Constrói protótipos por classe: média dos crystal_maps.

    Args:
        crystal_maps: (N, FS, FS)
        labels: (N,) array/tensor de labels inteiros
        n_classes: número de classes

    Returns:
        prototypes: dict {cls: (FS, FS)} — protótipo médio por classe
    """
    prototypes = {}
    for cls in range(n_classes):
        if isinstance(labels, np.ndarray):
            mask = labels == cls
        else:
            mask = (labels == cls)
        prototypes[cls] = crystal_maps[mask].mean(dim=0)
    return prototypes


def classify_euclidean(crystal_maps, prototypes):
    """
    Classifica por distância euclidiana ao protótipo mais próximo.

    Args:
        crystal_maps: (N, FS, FS) — crystal_maps das amostras de teste
        prototypes: dict {cls: (FS, FS)} — protótipos

    Returns:
        predictions: (N,) array de inteiros
    """
    n_classes = len(prototypes)
    N = len(crystal_maps)

    # Empilha protótipos: (n_classes, FS*FS)
    proto_stack = torch.stack([prototypes[c].flatten() for c in range(n_classes)])

    # Crystal maps: (N, FS*FS)
    cmaps_flat = crystal_maps.view(N, -1)

    # Distância euclidiana: (N, n_classes)
    dists = torch.cdist(cmaps_flat.unsqueeze(0), proto_stack.unsqueeze(0)).squeeze(0)

    return dists.argmin(dim=1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# DECODER TREINÁVEL (OPCIONAL — para 88-93%)
# ══════════════════════════════════════════════════════════════════════════════

def train_decoder(cmaps_train, labels_train, cmaps_val, labels_val,
                  cmaps_test, labels_test, n_classes=10,
                  nonlinear=False, lr=1e-3, epochs=60, patience=10,
                  batch_size=512, seed=0):
    """
    Treina um decoder linear ou MLP em cima dos crystal_maps.

    A física (campo + cristalização) NÃO é treinada — só o decoder.

    Args:
        cmaps_*:  (N, FS, FS) crystal_maps já computados
        labels_*: (N,) labels
        nonlinear: False = linear (88%), True = MLP (93%)

    Returns:
        (test_accuracy, n_params, decoder)
    """
    torch.manual_seed(seed)
    input_dim = cmaps_train.shape[1] * cmaps_train.shape[2]

    # Achata
    X_tr = cmaps_train.view(len(cmaps_train), -1).float()
    X_va = cmaps_val.view(len(cmaps_val), -1).float()
    X_te = cmaps_test.view(len(cmaps_test), -1).float()
    Y_tr = labels_train if isinstance(labels_train, torch.Tensor) else torch.tensor(labels_train, dtype=torch.long, device=DEVICE)
    Y_va = labels_val if isinstance(labels_val, torch.Tensor) else torch.tensor(labels_val, dtype=torch.long, device=DEVICE)
    Y_te = labels_test if isinstance(labels_test, torch.Tensor) else torch.tensor(labels_test, dtype=torch.long, device=DEVICE)

    if Y_tr.device != DEVICE: Y_tr = Y_tr.to(DEVICE)
    if Y_va.device != DEVICE: Y_va = Y_va.to(DEVICE)
    if Y_te.device != DEVICE: Y_te = Y_te.to(DEVICE)

    if nonlinear:
        dec = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        ).to(DEVICE)
    else:
        dec = nn.Linear(input_dim, n_classes).to(DEVICE)

    n_params = sum(p.numel() for p in dec.parameters())
    opt  = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best_val, best_sd, pat = 0.0, None, 0

    for ep in range(1, epochs + 1):
        dec.train()
        perm = torch.randperm(len(X_tr), device=DEVICE)
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad(set_to_none=True)
            crit(dec(X_tr[idx]), Y_tr[idx]).backward()
            opt.step()
        sch.step()

        dec.eval()
        with torch.no_grad():
            c = 0
            for i in range(0, len(X_va), 1024):
                c += (dec(X_va[i:i+1024]).argmax(1) == Y_va[i:i+1024]).sum().item()
            va = c / len(X_va) * 100

        if va > best_val:
            best_val = va
            best_sd = {k: v.clone() for k, v in dec.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    dec.load_state_dict(best_sd)
    dec.eval()
    with torch.no_grad():
        c = 0
        for i in range(0, len(X_te), 1024):
            c += (dec(X_te[i:i+1024]).argmax(1) == Y_te[i:i+1024]).sum().item()
        te_acc = c / len(X_te) * 100

    return te_acc, n_params, dec


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL — API SIMPLES
# ══════════════════════════════════════════════════════════════════════════════

class ResNetPsi:
    """
    ResNet-Ψ: classificação via física ondulatória + cristalização competitiva.

    Exemplo (zero treino):
        rn = ResNetPsi()
        rn.fit(train_images, train_labels)
        preds = rn.predict(test_images)

    Exemplo (com decoder):
        rn = ResNetPsi()
        cmaps = rn.extract(images)
        # usar cmaps como features para qualquer classificador
    """

    def __init__(self, input_shape=28, field_size=FIELD_SIZE, n_classes=10):
        """
        input_shape: int ou (int,) para 1D, (H, W) para 2D
        """
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)  # backward compat: int = imagem quadrada
        self.input_shape = input_shape
        self.field_size = field_size
        self.n_classes = n_classes
        self.PG = build_gaussians(input_shape, field_size)
        self.prototypes = None

    def extract(self, images, bs=64):
        """
        Extrai crystal_maps de um conjunto de imagens.

        Args:
            images: (N, H, W) tensor na GPU, normalizado
            bs: batch size

        Returns:
            (N, field_size, field_size) crystal_maps
        """
        images = images.to(DEVICE)
        return compute_crystal_maps(images, self.PG, self.field_size, bs)

    def fit(self, train_images, train_labels, bs=64):
        """
        Constrói protótipos por classe (sem treino — só computa e faz média).

        Args:
            train_images: (N, H, W) tensor normalizado
            train_labels: (N,) array ou tensor de labels
        """
        print(f"Extraindo crystal_maps de {len(train_images)} imagens...")
        t0 = time.time()
        cmaps = self.extract(train_images, bs)
        print(f"  Pronto: {time.time()-t0:.0f}s")

        if isinstance(train_labels, torch.Tensor):
            labels_np = train_labels.cpu().numpy()
        else:
            labels_np = np.array(train_labels)

        self.prototypes = build_prototypes(cmaps, labels_np, self.n_classes)
        return self

    def predict(self, test_images, bs=64):
        """
        Classifica imagens por distância euclidiana ao protótipo.

        Args:
            test_images: (N, H, W) tensor normalizado

        Returns:
            (N,) array de predições
        """
        cmaps = self.extract(test_images, bs)
        return classify_euclidean(cmaps, self.prototypes)

    def score(self, test_images, test_labels, bs=64):
        """
        Retorna acurácia (%).

        Args:
            test_images: (N, H, W) tensor normalizado
            test_labels: (N,) array ou tensor

        Returns:
            float: acurácia em porcentagem
        """
        preds = self.predict(test_images, bs)
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.cpu().numpy()
        else:
            test_labels = np.array(test_labels)
        return (preds == test_labels).mean() * 100


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — executa se rodar direto
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from torchvision import datasets, transforms

    print(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Carrega MNIST ─────────────────────────────────────────────────────
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

    train_imgs   = torch.stack([img.squeeze(0) for img, _ in train_ds]).to(DEVICE)
    train_labels = np.array([label for _, label in train_ds])
    test_imgs    = torch.stack([img.squeeze(0) for img, _ in test_ds]).to(DEVICE)
    test_labels  = np.array([label for _, label in test_ds])

    # ── Extrai crystal_maps uma vez (reusado nos dois modos) ─────────────
    rn = ResNetPsi()
    print("Extraindo crystal_maps (treino)...")
    t0 = time.time()
    cmaps_all = compute_crystal_maps(train_imgs, rn.PG)
    print(f"  Pronto: {time.time()-t0:.0f}s")

    print("Extraindo crystal_maps (teste)...")
    t0 = time.time()
    cmaps_test = compute_crystal_maps(test_imgs, rn.PG)
    print(f"  Pronto: {time.time()-t0:.0f}s")

    # ── Modo 1: Zero treino (protótipos) ─────────────────────────────────
    print("\n" + "="*60)
    print("MODO 1: Zero treino (protótipos cristalinos)")
    print("="*60)

    protos = build_prototypes(cmaps_all, train_labels, n_classes=10)
    preds  = classify_euclidean(cmaps_test, protos)
    acc    = (preds == test_labels).mean() * 100
    print(f"\n  Acurácia: {acc:.1f}%")
    print(f"  Referência: 77.4% (500 teste) / ~76% (10k teste)")

    # ── Modo 2: Com decoder linear ───────────────────────────────────────
    print("\n" + "="*60)
    print("MODO 2: Com decoder linear (treina só o decoder)")
    print("="*60)

    # Split treino/val
    torch.manual_seed(42)
    perm = torch.randperm(len(train_imgs))
    val_idx   = perm[-10000:]
    train_idx = perm[:-10000]

    cmaps_tr = cmaps_all[train_idx].to(DEVICE)
    cmaps_va = cmaps_all[val_idx].to(DEVICE)
    labels_tr = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)[train_idx]
    labels_va = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)[val_idx]
    labels_te = torch.tensor(test_labels, dtype=torch.long, device=DEVICE)

    acc_l, np_l, _ = train_decoder(cmaps_tr, labels_tr, cmaps_va, labels_va,
                                    cmaps_test.to(DEVICE), labels_te, nonlinear=False)
    print(f"\n  Linear: {acc_l:.1f}% ({np_l} params)")
    print(f"  Referência: 88.1%")

    acc_m, np_m, _ = train_decoder(cmaps_tr, labels_tr, cmaps_va, labels_va,
                                    cmaps_test.to(DEVICE), labels_te, nonlinear=True)
    print(f"  MLP:    {acc_m:.1f}% ({np_m} params)")
    print(f"  Referência: 93.1%")

    print("\nPronto.")
