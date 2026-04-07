"""
Teste de similaridade do PsiField com sequências de texto.

Converte texto → vetor numérico → wave_params → crystal_map.
Pergunta: o campo captura similaridade estrutural entre sequências?

Grupos testados:
  - Letras repetidas: "aaa", "bbb", "ccc", "aab", "abb"
  - Palavras similares: "gato", "gata", "rato", "pato"
  - Palavras opostas: "frio", "quente"
  - Anagramas: "amor", "mora", "roma", "armo"
  - Sequências numéricas: "111", "112", "123", "321"
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from RN_psi_mnist import (
    run_psi_field,
    FIELD_SIZE, N_WAVES, DEVICE
)

# ── Encoder: texto → vetor de wave_params ────────────────────────────────────

def text_to_wave_params(text, B=1):
    """
    Converte texto em wave_params deterministicamente.
    Cada caractere vira uma onda com freq/pos derivados do código ASCII.
    Preenche/trunca para N_WAVES ondas.
    """
    chars = [ord(c) for c in text]
    # normaliza para [0,1]
    vals = [c / 127.0 for c in chars]

    wp = torch.zeros(B, N_WAVES, 6, device=DEVICE)
    for w in range(N_WAVES):
        if w < len(vals):
            v = vals[w]
        else:
            # repete ciclicamente para preencher ondas restantes
            v = vals[w % len(vals)]

        wp[:, w, 0] = 2.0 + v * 2.0          # amp: [2.0, 4.0]
        wp[:, w, 1] = 1.0 + v * 5.0          # freq: [1.0, 6.0]
        wp[:, w, 2] = w * 0.4                 # phase: varia por índice
        wp[:, w, 3] = 0.001                   # decay: fixo
        wp[:, w, 4] = 0.2 + (w % 4) * 0.15   # pos_x: grade 4 colunas
        wp[:, w, 5] = 0.2 + (w // 4) * 0.2   # pos_y: grade 4 linhas
    return wp


# ── Grupos de sequências ──────────────────────────────────────────────────────

groups = {
    "Letras repetidas": ["aaa", "bbb", "ccc", "aab", "abb", "abc"],
    "Palavras similares": ["gato", "gata", "rato", "pato", "mato"],
    "Opostos": ["frio", "quente", "dia", "noite", "amor", "odio"],
    "Anagramas": ["amor", "mora", "roma", "armo", "omar"],
    "Números como texto": ["111", "112", "123", "321", "999"],
}

all_labels = []
all_cmaps  = []

print("Gerando crystal_maps...")
for group_name, words in groups.items():
    for word in words:
        wp   = text_to_wave_params(word)
        cmap = run_psi_field(wp).view(FIELD_SIZE, FIELD_SIZE).cpu().numpy()
        all_labels.append(f"{group_name}|{word}")
        all_cmaps.append(cmap)
        print(f"  {word:10s} → cristais: {int((cmap > 0.01).sum()):4d}  energia: {cmap.sum():.2f}")

N = len(all_labels)
words_only = [l.split("|")[1] for l in all_labels]

# ── Matriz de similaridade ────────────────────────────────────────────────────

def similarity(a, b):
    af, bf = a.flatten(), b.flatten()
    sa, sb = af.std(), bf.std()
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])

def overlap(a, b, thr=0.01):
    ma, mb = a > thr, b > thr
    inter = (ma & mb).sum()
    union = (ma | mb).sum()
    return float(inter / union) if union > 0 else 0.0

print("\nCalculando matriz de similaridade...")
sim_matrix  = np.zeros((N, N))
over_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        sim_matrix[i, j]  = similarity(all_cmaps[i], all_cmaps[j])
        over_matrix[i, j] = overlap(all_cmaps[i], all_cmaps[j])

# ── Estatísticas por grupo ────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SIMILARIDADE INTRA-GRUPO vs INTER-GRUPO")

group_indices = {}
start = 0
for gname, words in groups.items():
    group_indices[gname] = list(range(start, start + len(words)))
    start += len(words)

for gname, idxs in group_indices.items():
    # intra: pares dentro do grupo
    intra_sims = []
    for i in idxs:
        for j in idxs:
            if i < j:
                intra_sims.append(sim_matrix[i, j])

    # inter: pares entre este grupo e os outros
    other_idxs = [k for k in range(N) if k not in idxs]
    inter_sims = []
    for i in idxs:
        for j in other_idxs:
            inter_sims.append(sim_matrix[i, j])

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    print(f"\n  [{gname}]")
    print(f"    Intra-grupo : {intra_mean:.4f}  (esperado: alto)")
    print(f"    Inter-grupo : {inter_mean:.4f}  (esperado: baixo)")
    print(f"    Razão intra/inter: {intra_mean/inter_mean:.2f}x" if inter_mean > 1e-6 else "    Razão: inf")

# Anagramas em detalhe
print(f"\n{'='*60}")
print("ANAGRAMAS — pares de similaridade:")
ana_idxs = group_indices["Anagramas"]
for i in ana_idxs:
    for j in ana_idxs:
        if i < j:
            w1 = words_only[i]
            w2 = words_only[j]
            s  = sim_matrix[i, j]
            o  = over_matrix[i, j] * 100
            print(f"  {w1:6s} vs {w2:6s} : corr={s:.4f}  overlap={o:.1f}%")

# ── Visualizações ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Similaridade de Crystal Maps por Sequência de Texto', fontsize=13)

# Heatmap de correlação
im0 = axes[0].imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(N)); axes[0].set_xticklabels(words_only, rotation=45, ha='right', fontsize=8)
axes[0].set_yticks(range(N)); axes[0].set_yticklabels(words_only, fontsize=8)
axes[0].set_title('Correlação de Pearson entre crystal_maps')
plt.colorbar(im0, ax=axes[0])

# Separadores de grupo
pos = 0
for gname, words in groups.items():
    n = len(words)
    for ax in axes[:2]:
        ax.axhline(pos - 0.5, color='black', lw=1.5)
        ax.axvline(pos - 0.5, color='black', lw=1.5)
    pos += n

# Heatmap de overlap IoU
im1 = axes[1].imshow(over_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
axes[1].set_xticks(range(N)); axes[1].set_xticklabels(words_only, rotation=45, ha='right', fontsize=8)
axes[1].set_yticks(range(N)); axes[1].set_yticklabels(words_only, fontsize=8)
axes[1].set_title('Sobreposição de Cristais (IoU)')
plt.colorbar(im1, ax=axes[1])

# anotação de valores na diagonal
for ax, mat in [(axes[0], sim_matrix), (axes[1], over_matrix)]:
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center',
                    fontsize=5, color='black' if abs(mat[i,j]) < 0.7 else 'white')

plt.tight_layout()
plt.savefig('viz_similarity_matrix.png', dpi=130, bbox_inches='tight')
plt.close()

# ── Crystal maps lado a lado por grupo ───────────────────────────────────────

for gname, idxs in group_indices.items():
    n = len(idxs)
    fig, axes2 = plt.subplots(1, n, figsize=(n * 3.5, 3.5))
    if n == 1:
        axes2 = [axes2]
    fig.suptitle(f'Crystal Maps — {gname}', fontsize=11)

    vmax = max(all_cmaps[i].max() for i in idxs) + 1e-6
    for ax, idx in zip(axes2, idxs):
        word   = words_only[idx]
        cmap_i = all_cmaps[idx]
        n_crys = int((cmap_i > 0.01).sum())
        ax.imshow(cmap_i, cmap='inferno', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_title(f'"{word}"\n{n_crys} cristais', fontsize=9)
        ax.axis('off')

    fname = f"viz_group_{gname.replace(' ', '_').replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"-> {fname}")

print("-> viz_similarity_matrix.png")
print("Pronto.")
