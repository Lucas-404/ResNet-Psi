"""
Encoder sequencial para o PsiField — versão por PALAVRA.

Unidade de encoding: palavra (ou token), não caractere.
- Eixo X: hash da palavra inteira (o QUE é)
- Eixo Y: posição na sequência (QUANDO aparece)

"gato come rato" != "rato come gato" porque:
- "gato" tem X fixo, Y=0.1
- "rato" tem X fixo, Y=0.9
Invertendo a frase, os Y trocam — completamente distinguível.

Funciona para:
- Palavras: "gato", "rato", "amor"
- Números: "123", "456"
- Caracteres únicos: "a", "1"
- Frases: tokeniza por espaço
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RN_psi_mnist import (
    psi_step,
    STIM_ON, STIM_TOTAL, DEVICE,
    CRYSTAL_W, CRYSTAL_K, CRYSTAL_A_MIN, CRYSTAL_CV_MAX,
    CRYSTAL_SEP, CRYSTAL_REMIT,
)

FIELD_SIZE = 128

# ── CrystalMemory escalável ───────────────────────────────────────────────────

class CrystalMem:
    def __init__(self, B, FS):
        self.crystal_map = torch.zeros(B, FS, FS, device=DEVICE)
        self.env_buffer  = torch.zeros(B, CRYSTAL_K, FS, FS, device=DEVICE)
        self.window_step = 0
        self.window_idx  = 0
        self.window_max  = torch.zeros(B, FS, FS, device=DEVICE)
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
        cand = ((mean > CRYSTAL_A_MIN) & (cv < CRYSTAL_CV_MAX) & (mean < 8.0)).float()
        occ  = F.conv2d(
            F.pad(self.crystal_map.unsqueeze(1).clamp(0, 1),
                  (CRYSTAL_SEP,) * 4, mode='circular'),
            self._dilate).squeeze(1).clamp(0, 1)
        self.crystal_map = torch.clamp(
            self.crystal_map + cand * (1.0 - occ) * field.abs(), 0, 10.)

    def remit(self, field):
        if self.crystal_map.abs().max() < 1e-6:
            return field
        return torch.clamp(
            field + self.crystal_map * CRYSTAL_REMIT * torch.sign(field), -10., 10.)


# ── Encoder por palavra ───────────────────────────────────────────────────────

def word_hash(word):
    """
    Hash determinístico de uma palavra → posição X no campo [0, 1].
    Palavras diferentes → posições X diferentes.
    Mesma palavra → sempre mesma posição X.
    """
    # Hash polinomial simples sobre os caracteres da palavra
    h = 0
    for c in word:
        h = (h * 31 + ord(c)) % 10007
    return h / 10007.0  # [0, 1]


def word_amplitude(word):
    """Amplitude da gaussiana baseada no comprimento e conteúdo da palavra."""
    v = sum(ord(c) for c in word) / (len(word) * 127.0)
    return 1.5 + v * 2.5


def word_to_gaussian(word, seq_idx, seq_len, field_size=FIELD_SIZE):
    """
    Palavra + posição na sequência → gaussiana no campo.
    Eixo X: hash da palavra (identidade)
    Eixo Y: posição na sequência (ordem)
    """
    cx    = word_hash(word)
    cy    = 0.1 + 0.8 * (seq_idx / max(seq_len - 1, 1))
    amp   = word_amplitude(word)
    sigma = 0.04

    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gauss  = amp * torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))
    return gauss


def tokenize(text):
    """Tokeniza por espaço. Caractere único retorna lista com 1 elemento."""
    return text.split()


def seq_to_field(text, field_size=FIELD_SIZE):
    """Texto → perturbação do campo. Unidade = palavra."""
    tokens = tokenize(text)
    field  = torch.zeros(field_size, field_size, device=DEVICE)
    n = len(tokens)
    for i, word in enumerate(tokens):
        field = field + word_to_gaussian(word, i, n, field_size)
    return field.unsqueeze(0)


def run_field(text, field_size=FIELD_SIZE):
    pert     = seq_to_field(text, field_size)
    field    = pert.clone()
    velocity = torch.zeros_like(field)
    memory   = CrystalMem(1, field_size)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            active = s < STIM_ON
            field, velocity = psi_step(field, velocity, pert, active)
            memory.update_envelope(field)
            if memory.window_idx > 0:
                memory.try_crystallize(field)
            field = memory.remit(field)
    return memory.crystal_map.squeeze(0).cpu().numpy()


def iou(a, b, thr=0.01):
    ma, mb = a > thr, b > thr
    return float((ma & mb).sum()) / float((ma | mb).sum() + 1e-8)

def overlap(a, b, thr=0.01):
    ma, mb = a > thr, b > thr
    return float((ma & mb).sum()) / float(max(ma.sum(), mb.sum()) + 1e-8)


# ── Experimento A: frases com ordem diferente ─────────────────────────────────

print("=" * 60)
print(f"Campo: {FIELD_SIZE}×{FIELD_SIZE} | Encoder: por PALAVRA")
print("=" * 60)

print("\nEXPERIMENTO A: Frases — ordem das palavras")
frases = [
    ("gato come rato", "gato come rato"),    # idêntico
    ("gato come rato", "rato come gato"),    # invertido
    ("gato come rato", "rato gato come"),    # permutado
    ("gato come rato", "gato bebe leite"),   # palavras diferentes
    ("o sol nasce", "nasce o sol"),          # invertido
    ("eu amo voce", "voce ama eu"),          # similar mas diferente
]

cmaps = {}
for w1, w2 in frases:
    for w in [w1, w2]:
        if w not in cmaps:
            cmaps[w] = run_field(w)

print(f"\n  {'Par':40s}  {'IoU':>8}  {'Overlap':>8}")
print("  " + "-" * 60)
for w1, w2 in frases:
    print(f"  {w1+' vs '+w2:40s}  {iou(cmaps[w1],cmaps[w2]):>8.4f}  {overlap(cmaps[w1],cmaps[w2]):>8.4f}")


# ── Experimento B: palavras individuais ──────────────────────────────────────

print("\nEXPERIMENTO B: Palavras individuais")
palavras = [
    ("gato", "gato"),     # idêntico
    ("gato", "rato"),     # diferentes
    ("gato", "gatos"),    # singular vs plural
    ("amor", "amor"),     # idêntico
    ("amor", "amora"),    # subpalavra
    ("123", "123"),       # número idêntico
    ("123", "321"),       # número diferente
    ("123", "456"),       # números distintos
]

cmaps_p = {}
for w1, w2 in palavras:
    for w in [w1, w2]:
        if w not in cmaps_p:
            cmaps_p[w] = run_field(w)

print(f"\n  {'Par':25s}  {'IoU':>8}  {'Overlap':>8}")
print("  " + "-" * 45)
for w1, w2 in palavras:
    print(f"  {w1+' vs '+w2:25s}  {iou(cmaps_p[w1],cmaps_p[w2]):>8.4f}  {overlap(cmaps_p[w1],cmaps_p[w2]):>8.4f}")


# ── Experimento C: similaridade semântica ────────────────────────────────────

print("\nEXPERIMENTO C: Similaridade — palavras relacionadas")
semantico = [
    ("gato", "gato"),         # idêntico
    ("gato", "gata"),         # masculino/feminino
    ("gato", "cachorro"),     # animal diferente
    ("gato", "felino"),       # sinônimo
    ("gato", "computador"),   # sem relação
    ("correr", "correndo"),   # forma verbal
    ("correr", "andar"),      # ação similar
]

cmaps_s = {}
for w1, w2 in semantico:
    for w in [w1, w2]:
        if w not in cmaps_s:
            cmaps_s[w] = run_field(w)

print(f"\n  {'Par':30s}  {'IoU':>8}  {'Overlap':>8}")
print("  " + "-" * 50)
for w1, w2 in semantico:
    print(f"  {w1+' vs '+w2:30s}  {iou(cmaps_s[w1],cmaps_s[w2]):>8.4f}  {overlap(cmaps_s[w1],cmaps_s[w2]):>8.4f}")


# ── Visualização ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Encoder por Palavra {FIELD_SIZE}×{FIELD_SIZE}', fontsize=12)

viz = ["gato come rato", "rato come gato", "rato gato come", "gato bebe leite",
       "gato", "rato", "123", "321"]
for ax, text in zip(axes.flatten(), viz):
    cmap = run_field(text)
    ax.imshow(cmap, cmap='inferno', interpolation='nearest', origin='upper')
    nc = int((cmap > 0.01).sum())
    ax.set_title(f'"{text}"\n{nc} cristais', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('viz_encoder_seq.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"\n-> viz_encoder_seq.png")
print("Pronto.")


# ── Experimento D: Predição por ressonância ───────────────────────────────────

def build_vocab(words):
    vocab = {w: word_hash(w) for w in words}
    return vocab

def predict_next(context, vocab):
    """Injeta contexto no campo e vê qual palavra do vocab ressoa mais."""
    context_map = run_field(context)
    context_tensor = torch.tensor(context_map, device=DEVICE)

    best_word = None
    best_score = -1

    for word, wx in vocab.items():
        # Posição X esperada da palavra no campo
        px = int(wx * FIELD_SIZE)
        # Energia na coluna X do crystal_map
        score = float(context_tensor[:, px].sum())
        if score > best_score:
            best_score = score
            best_word = word

    return best_word, best_score

# Vocabulário de teste
vocab = build_vocab(["rato", "leite", "sol", "gato", "come", "bebe", "nasce"])

print("\nEXPERIMENTO D: Predição por ressonância")
testes = [
    "gato come",
    "gato bebe",
    "o sol",
    "rato come",
]

for ctx in testes:
    pred, score = predict_next(ctx, vocab)
    print(f"  '{ctx}' → '{pred}' (score: {score:.4f})")


# ── Experimento D v2: Predição por similaridade de crystal_map ───────────────

def score_match(context_map, frase_map, context, frase):
    """Overlap físico + bônus por palavras do contexto presentes na frase."""
    base = overlap(context_map, frase_map)

    ctx_tokens = tokenize(context)
    frase_tokens = tokenize(frase)

    # Quantas palavras do contexto estão na frase
    matches = sum(1 for w in ctx_tokens if w in frase_tokens)
    bonus = matches / len(ctx_tokens)

    return base * 0.5 + bonus * 0.5  # 50% física, 50% léxico


def predict_next_v2(context, corpus):
    context_map = run_field(context)

    best_match = None
    best_score = -1

    for frase in corpus:
        frase_map = run_field(frase)
        score = score_match(context_map, frase_map, context, frase)
        if score > best_score:
            best_score = score
            best_match = frase

    ctx_tokens = set(tokenize(context))
    match_tokens = tokenize(best_match)
    continuacao = [w for w in match_tokens if w not in ctx_tokens]

    return best_match, continuacao, best_score


# Corpus de 25 frases — 5 domínios temáticos
corpus = [
    # animais
    "gato come rato",
    "gato bebe leite",
    "gato dorme muito",
    "cachorro corre rapido",
    "cachorro bebe agua",
    "rato come queijo",
    "passaro voa alto",
    "peixe nada rio",
    # natureza
    "o sol nasce cedo",
    "sol nasce todo dia",
    "chuva cai forte",
    "vento sopra frio",
    "lua brilha noite",
    "rio corre mar",
    # comida
    "crianca come fruta",
    "homem bebe cafe",
    "mulher come pao",
    "menino bebe suco",
    # acao
    "crianca corre parque",
    "homem dorme cedo",
    "mulher trabalha muito",
    "menino estuda escola",
    # tempo
    "dia nasce cedo",
    "noite cai fria",
    "tempo passa rapido",
]

# Contextos de teste — varia domínio e comprimento
testes = [
    # 1 palavra
    "gato",
    "cachorro",
    "sol",
    "crianca",
    # 2 palavras — continuação direta
    "gato come",
    "gato bebe",
    "cachorro bebe",
    "rato come",
    "o sol",
    "homem bebe",
    "mulher come",
    "menino bebe",
    # 2 palavras — cruzamento de domínio
    "gato corre",
    "passaro come",
    "chuva cai",
    "tempo passa",
]

print("\nEXPERIMENTO D v2: Predição por similaridade física (corpus 25 frases)")
print(f"  {'Contexto':20s}  {'Continuação':20s}  {'Match':25s}  Score")
print("  " + "-" * 80)

acertos = 0
for ctx in testes:
    match, cont, score = predict_next_v2(ctx, corpus)
    cont_str = ' '.join(cont) if cont else '(vazio)'
    # Acerto: pelo menos 1 palavra de continuação faz sentido com o contexto
    print(f"  {ctx:20s}  {cont_str:20s}  {match:25s}  {score:.4f}")

print(f"\nTotal testado: {len(testes)} contextos | Corpus: {len(corpus)} frases")
