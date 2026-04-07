"""
Auditoria 3: Ablação da completação de frases

Testa separadamente:
  a) 100% física (overlap) + 0% léxico
  b) 70/30, 50/50, 30/70
  c) 0% física + 100% léxico (baseline string matching)
  d) Baseline TF-IDF + cosseno

Também testa com corpus expandido (100 frases) para ver se escala.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from RN_psi_mnist import (
    psi_step,
    STIM_ON, STIM_TOTAL, DEVICE,
    CRYSTAL_W, CRYSTAL_K, CRYSTAL_A_MIN, CRYSTAL_CV_MAX,
    CRYSTAL_SEP, CRYSTAL_REMIT,
)

FIELD_SIZE = 128

# ── CrystalMemory ──────────────────────────────────────────────────────────

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


# ── Encoder por palavra (mesmo do RN_psi_encoder_seq.py) ───────────────────

def word_hash(word):
    h = 0
    for c in word:
        h = (h * 31 + ord(c)) % 10007
    return h / 10007.0

def word_amplitude(word):
    v = sum(ord(c) for c in word) / (len(word) * 127.0)
    return 1.5 + v * 2.5

def word_to_gaussian(word, seq_idx, seq_len, field_size=FIELD_SIZE):
    cx    = word_hash(word)
    cy    = 0.1 + 0.8 * (seq_idx / max(seq_len - 1, 1))
    amp   = word_amplitude(word)
    sigma = 0.04
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    return amp * torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))

def tokenize(text):
    return text.split()

def run_field(text, field_size=FIELD_SIZE):
    tokens = tokenize(text)
    field  = torch.zeros(field_size, field_size, device=DEVICE)
    n = len(tokens)
    for i, word in enumerate(tokens):
        field = field + word_to_gaussian(word, i, n, field_size)
    pert = field.unsqueeze(0)
    field_state = pert.clone()
    velocity    = torch.zeros_like(field_state)
    memory      = CrystalMem(1, field_size)
    with torch.no_grad():
        for s in range(STIM_TOTAL):
            active = s < STIM_ON
            field_state, velocity = psi_step(field_state, velocity, pert, active)
            memory.update_envelope(field_state)
            if memory.window_idx > 0:
                memory.try_crystallize(field_state)
            field_state = memory.remit(field_state)
    return memory.crystal_map.squeeze(0).cpu().numpy()

def overlap(a, b, thr=0.01):
    ma, mb = a > thr, b > thr
    return float((ma & mb).sum()) / float(max(ma.sum(), mb.sum()) + 1e-8)


# ── Métricas de score ───────────────────────────────────────────────────────

def score_physics_only(ctx_map, frase_map, ctx, frase):
    """100% overlap físico."""
    return overlap(ctx_map, frase_map)

def score_lexical_only(ctx_map, frase_map, ctx, frase):
    """100% string matching (quantas palavras do contexto estão na frase)."""
    ctx_tokens = set(tokenize(ctx))
    frase_tokens = set(tokenize(frase))
    if len(ctx_tokens) == 0:
        return 0.0
    return len(ctx_tokens & frase_tokens) / len(ctx_tokens)

def score_hybrid(ctx_map, frase_map, ctx, frase, phys_weight=0.5):
    """Híbrido: phys_weight × física + (1-phys_weight) × léxico."""
    phys = overlap(ctx_map, frase_map)
    lex  = score_lexical_only(ctx_map, frase_map, ctx, frase)
    return phys * phys_weight + lex * (1 - phys_weight)

def score_tfidf(ctx, frase, idf_dict):
    """TF-IDF cosseno simplificado."""
    ctx_tokens = tokenize(ctx)
    frase_tokens = tokenize(frase)
    all_words = set(ctx_tokens) | set(frase_tokens)
    if not all_words:
        return 0.0

    def tfidf_vec(tokens):
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        vec = []
        for w in sorted(all_words):
            vec.append(tf.get(w, 0) * idf_dict.get(w, 1.0))
        return np.array(vec)

    v1 = tfidf_vec(ctx_tokens)
    v2 = tfidf_vec(frase_tokens)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ── Corpus original (25 frases) ────────────────────────────────────────────

corpus_25 = [
    "gato come rato", "gato bebe leite", "gato dorme muito",
    "cachorro corre rapido", "cachorro bebe agua",
    "rato come queijo", "passaro voa alto", "peixe nada rio",
    "o sol nasce cedo", "sol nasce todo dia", "chuva cai forte",
    "vento sopra frio", "lua brilha noite", "rio corre mar",
    "crianca come fruta", "homem bebe cafe", "mulher come pao", "menino bebe suco",
    "crianca corre parque", "homem dorme cedo", "mulher trabalha muito", "menino estuda escola",
    "dia nasce cedo", "noite cai fria", "tempo passa rapido",
]

# ── Corpus expandido (100 frases) ──────────────────────────────────────────

corpus_100 = corpus_25 + [
    # animais expandido
    "gato pula alto", "gato mia noite", "cachorro late forte",
    "cachorro come osso", "rato foge rapido", "rato esconde buraco",
    "passaro canta manha", "passaro come semente", "peixe come minhoca",
    "peixe pula agua", "cavalo corre campo", "vaca come capim",
    "galinha bota ovo", "cobra rasteja chao", "borboleta voa flor",
    # natureza expandido
    "chuva molha terra", "chuva cai leve", "sol brilha forte",
    "sol esquenta praia", "vento leva folha", "vento sopra forte",
    "lua ilumina caminho", "estrela brilha ceu", "rio desce montanha",
    "mar bate rocha", "neve cai inverno", "trovao assusta crianca",
    # comida expandido
    "crianca come bolo", "crianca bebe suco", "homem come arroz",
    "homem bebe cerveja", "mulher come salada", "mulher bebe vinho",
    "menino come chocolate", "menina come pipoca", "bebe bebe leite",
    # acao expandido
    "crianca brinca parque", "crianca estuda escola", "homem trabalha escritorio",
    "homem corre praia", "mulher corre parque", "mulher le livro",
    "menino joga bola", "menino nada piscina", "menina danca sala",
    # tempo expandido
    "tempo voa rapido", "tempo muda sempre", "hora passa devagar",
    "minuto parece eterno", "segundo vale ouro", "manha chega cedo",
    "tarde passa lento", "noite chega rapido", "inverno chega frio",
    # abstrato
    "vida passa rapido", "amor cura tudo", "paz traz felicidade",
    "verdade liberta alma", "musica acalma coracao", "arte transforma mundo",
    # ambíguo (compartilha palavras entre domínios)
    "gato come peixe", "peixe come rato", "cachorro bebe leite",
    "crianca come rato", "homem corre rapido", "passaro come peixe",
    "rato bebe agua", "menino come queijo", "mulher bebe agua",
    "cavalo bebe agua", "gato bebe agua", "crianca bebe agua",
]

# ── Contextos de teste ──────────────────────────────────────────────────────

# Original (16)
testes_16 = [
    "gato", "cachorro", "sol", "crianca",
    "gato come", "gato bebe", "rato come", "o sol",
    "homem bebe", "chuva cai", "tempo passa",
    "gato corre", "passaro come",
    "menino bebe", "mulher come", "cachorro bebe",
]

# Expandido (30) — inclui casos ambíguos
testes_30 = testes_16 + [
    "peixe come",       # peixe come minhoca ou peixe come rato?
    "crianca come",     # fruta, bolo, rato?
    "homem corre",      # praia?
    "gato bebe",        # leite ou agua?
    "cachorro come",    # osso?
    "passaro voa",      # alto?
    "rio corre",        # mar?
    "menino come",      # chocolate? queijo?
    "mulher bebe",      # vinho? agua?
    "cavalo corre",     # campo?
    "vida passa",       # rapido?
    "amor cura",        # tudo?
    "chuva molha",      # terra?
    "sol esquenta",     # praia?
]


# ── Função de avaliação ────────────────────────────────────────────────────

def evaluate_completion(corpus, testes, score_fn, cache=None):
    """
    Para cada contexto, encontra a frase do corpus com maior score.
    Retorna lista de (contexto, match, continuação, score).
    """
    # Cache de crystal maps
    if cache is None:
        cache = {}

    results = []
    for ctx in testes:
        if ctx not in cache:
            cache[ctx] = run_field(ctx)

        best_match = None
        best_score = -1

        for frase in corpus:
            if frase not in cache:
                cache[frase] = run_field(frase)

            score = score_fn(cache[ctx], cache[frase], ctx, frase)
            if score > best_score:
                best_score = score
                best_match = frase

        ctx_tokens = set(tokenize(ctx))
        match_tokens = tokenize(best_match)
        continuacao = [w for w in match_tokens if w not in ctx_tokens]

        results.append({
            'ctx': ctx,
            'match': best_match,
            'cont': ' '.join(continuacao) if continuacao else '(vazio)',
            'score': best_score,
        })

    return results, cache


def is_reasonable_match(ctx, match):
    """Verifica se o match contém todas as palavras do contexto."""
    ctx_tokens = set(tokenize(ctx))
    match_tokens = set(tokenize(match))
    return ctx_tokens.issubset(match_tokens)


# ── Experimento A: Ablação no corpus de 25 ─────────────────────────────────

print("="*70)
print("AUDITORIA 3a: Ablação — Corpus 25, 16 contextos")
print("="*70)

cache = {}
proportions = [
    ("100% física",     lambda cm, fm, c, f: score_physics_only(cm, fm, c, f)),
    ("70% fís + 30% lex", lambda cm, fm, c, f: score_hybrid(cm, fm, c, f, 0.7)),
    ("50% fís + 50% lex", lambda cm, fm, c, f: score_hybrid(cm, fm, c, f, 0.5)),
    ("30% fís + 70% lex", lambda cm, fm, c, f: score_hybrid(cm, fm, c, f, 0.3)),
    ("100% léxico",     lambda cm, fm, c, f: score_lexical_only(cm, fm, c, f)),
]

print(f"\n{'Método':>22}  {'Acertos':>8}  {'% exata':>8}  {'% razoável':>11}")
print("-"*55)

for name, fn in proportions:
    results, cache = evaluate_completion(corpus_25, testes_16, fn, cache)
    exact   = sum(1 for r in results if is_reasonable_match(r['ctx'], r['match']))
    n_total = len(results)
    print(f"  {name:>20}  {exact:>4}/{n_total:<3}  {exact/n_total*100:>7.1f}%  —")


# ── Experimento B: TF-IDF baseline ──────────────────────────────────────────

print(f"\n{'='*70}")
print("AUDITORIA 3b: Baseline TF-IDF — Corpus 25, 16 contextos")
print("="*70)

# Calcula IDF do corpus
all_words = set()
for frase in corpus_25:
    all_words.update(tokenize(frase))

doc_freq = {}
for w in all_words:
    doc_freq[w] = sum(1 for f in corpus_25 if w in tokenize(f))

idf = {w: np.log(len(corpus_25) / (df + 1)) for w, df in doc_freq.items()}

tfidf_results = []
for ctx in testes_16:
    best_match = None
    best_score = -1
    for frase in corpus_25:
        s = score_tfidf(ctx, frase, idf)
        if s > best_score:
            best_score = s
            best_match = frase
    ctx_tokens = set(tokenize(ctx))
    cont = [w for w in tokenize(best_match) if w not in ctx_tokens]
    tfidf_results.append({
        'ctx': ctx, 'match': best_match,
        'cont': ' '.join(cont) if cont else '(vazio)', 'score': best_score,
    })

exact_tfidf = sum(1 for r in tfidf_results if is_reasonable_match(r['ctx'], r['match']))
print(f"  TF-IDF cosseno: {exact_tfidf}/{len(tfidf_results)} ({exact_tfidf/len(tfidf_results)*100:.1f}%)")

print(f"\n  {'Contexto':>20}  {'Match TF-IDF':>25}  {'Score':>6}")
print("  " + "-"*55)
for r in tfidf_results:
    print(f"  {r['ctx']:>20}  {r['match']:>25}  {r['score']:.4f}")


# ── Experimento C: Corpus expandido (100 frases) ───────────────────────────

print(f"\n{'='*70}")
print(f"AUDITORIA 3c: Corpus 100 frases, 30 contextos (inclui ambíguos)")
print(f"{'='*70}")

cache_100 = {}

for name, fn in proportions:
    results, cache_100 = evaluate_completion(corpus_100, testes_30, fn, cache_100)
    exact = sum(1 for r in results if is_reasonable_match(r['ctx'], r['match']))
    n_total = len(results)
    print(f"  {name:>20}  {exact:>4}/{n_total:<3}  ({exact/n_total*100:.1f}%)")

# TF-IDF no corpus 100
all_words_100 = set()
for frase in corpus_100:
    all_words_100.update(tokenize(frase))
doc_freq_100 = {}
for w in all_words_100:
    doc_freq_100[w] = sum(1 for f in corpus_100 if w in tokenize(f))
idf_100 = {w: np.log(len(corpus_100) / (df + 1)) for w, df in doc_freq_100.items()}

tfidf_100 = []
for ctx in testes_30:
    best_match, best_score = None, -1
    for frase in corpus_100:
        s = score_tfidf(ctx, frase, idf_100)
        if s > best_score:
            best_score = s
            best_match = frase
    tfidf_100.append({'ctx': ctx, 'match': best_match})

exact_tfidf_100 = sum(1 for r in tfidf_100 if is_reasonable_match(r['ctx'], r['match']))
print(f"  {'TF-IDF cosseno':>20}  {exact_tfidf_100:>4}/{len(tfidf_100):<3}  ({exact_tfidf_100/len(tfidf_100)*100:.1f}%)")


# ── Tabela detalhada: física vs TF-IDF nos casos ambíguos ──────────────────

print(f"\n{'='*70}")
print("DETALHE: Casos ambíguos — 100% física vs TF-IDF (corpus 100)")
print(f"{'='*70}")

ambiguos = [t for t in testes_30 if t not in testes_16]

results_phys, _ = evaluate_completion(corpus_100, ambiguos,
    lambda cm, fm, c, f: score_physics_only(cm, fm, c, f), cache_100)

print(f"\n  {'Contexto':>18}  {'Física → Match':>30}  {'TF-IDF → Match':>30}")
print("  " + "-"*80)

tfidf_amb = {r['ctx']: r['match'] for r in tfidf_100 if r['ctx'] in ambiguos}
for r in results_phys:
    tf_match = tfidf_amb.get(r['ctx'], '?')
    marker = " ✓" if r['match'] == tf_match else " ≠"
    print(f"  {r['ctx']:>18}  {r['match']:>30}  {tf_match:>30}{marker}")

print(f"\nPronto.")
