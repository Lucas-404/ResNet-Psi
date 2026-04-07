"""
Teste: PsiField como memória associativa pura.

Sem emitter. Sem pesos. Sem rede neural.

O texto é injetado diretamente como perturbação gaussiana no campo.
Cada caractere perturba uma região fixa e determinística do grid.
As ondas se propagam, interferem, e os cristais emergem da física bruta.

Hipótese: textos com caracteres similares → padrões de cristal similares.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RN_psi_mnist import (
    psi_step, CrystalMemory,
    FIELD_SIZE, STIM_ON, STIM_TOTAL, PSI_DT,
    DEVICE
)

# ── Encoder puramente físico: texto → perturbação inicial do campo ────────────

def char_to_gaussian(c, field_size=FIELD_SIZE):
    """
    Cada caractere define uma gaussiana no campo.
    Posição: determinística pelo valor ASCII, distribuída no grid.
    Amplitude: determinística pelo valor ASCII.
    """
    v = ord(c) / 127.0   # [0, 1]

    # Posição: mapeamento de Hilbert-like simples — distribui uniformemente
    # Divide o grid em células pela posição do ASCII
    n_cells = 8  # 8x8 = 64 células — suficiente para o alfabeto
    cell_x  = ord(c) % n_cells
    cell_y  = (ord(c) // n_cells) % n_cells
    cx      = (cell_x + 0.5) / n_cells   # centro da célula em [0,1]
    cy      = (cell_y + 0.5) / n_cells

    # Amplitude baseada no valor ASCII — letras minúsculas têm boa amplitude
    amp   = 1.5 + v * 2.5   # [1.5, 4.0]
    sigma = 0.06             # largura gaussiana (~3px no grid 48)

    # Gera gaussiana no campo
    coords = torch.linspace(0., 1., field_size, device=DEVICE)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    gauss  = amp * torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))
    return gauss


def text_to_field(text):
    """
    Converte texto em campo inicial.
    Cada caractere contribui com uma gaussiana — soma linear.
    Resultado: perturbação espacial determinística, sem ondas, sem rede.
    """
    field = torch.zeros(FIELD_SIZE, FIELD_SIZE, device=DEVICE)
    for i, c in enumerate(text):
        gauss = char_to_gaussian(c)
        # Fase temporal via índice do caractere — cria diferença de fase por posição
        phase_shift = float(i) * 0.3
        field = field + gauss * np.cos(phase_shift)
    return field


def run_associative(text):
    """
    Roda o campo com perturbação inicial derivada do texto.
    Sem emitter. Campo evolui livremente após a perturbação.
    """
    # Campo inicial = texto codificado como perturbação gaussiana
    init_field = text_to_field(text).unsqueeze(0)   # (1, H, W)
    field      = init_field.clone()
    velocity   = torch.zeros_like(field)
    memory     = CrystalMemory(1)

    for s in range(STIM_TOTAL):
        # Durante STIM_ON: reinjetar perturbação inicial (como "segurar" a entrada)
        # Após STIM_ON: campo evolui livremente
        active = s < STIM_ON
        # Sem emit_waves — a perturbação É o campo inicial, não ondas pontuais
        field, velocity = psi_step(field, velocity, init_field, active)
        memory.update_envelope(field)
        if memory.window_idx > 0:
            memory.try_crystallize(field)
        field = memory.remit(field, None)

    return memory.crystal_map.squeeze(0).cpu().numpy()   # (H, W)


# ── Grupos de teste ───────────────────────────────────────────────────────────

groups = {
    "Letras repetidas": ["aaa", "bbb", "ccc", "aab", "abb", "abc"],
    "Palavras similares": ["gato", "gata", "rato", "pato", "mato"],
    "Opostos": ["frio", "quente", "dia", "noite"],
    "Anagramas": ["amor", "mora", "roma", "armo", "omar"],
    "Frases curtas": ["oi", "ola", "oi!", "ola!", "boa"],
    "Números": ["111", "112", "123", "321", "999"],
}

all_labels = []
all_cmaps  = []

print("Gerando crystal_maps (sem emitter, sem rede)...")
for gname, words in groups.items():
    for word in words:
        cmap = run_associative(word)
        n_crys = int((cmap > 0.01).sum())
        energia = float(cmap.sum())
        all_labels.append(f"{gname}|{word}")
        all_cmaps.append(cmap)
        print(f"  {word:10s} → cristais: {n_crys:4d}  energia: {energia:.2f}")

N = len(all_labels)
words_only  = [l.split("|")[1] for l in all_labels]
groups_only = [l.split("|")[0] for l in all_labels]

# ── Métricas ──────────────────────────────────────────────────────────────────

def similarity(a, b):
    af, bf = a.flatten(), b.flatten()
    sa, sb = af.std(), bf.std()
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])

def iou(a, b, thr=0.01):
    ma, mb = a > thr, b > thr
    inter  = (ma & mb).sum()
    union  = (ma | mb).sum()
    return float(inter / union) if union > 0 else 0.0

print("\nCalculando matrizes de similaridade...")
sim_mat  = np.zeros((N, N))
iou_mat  = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        sim_mat[i, j] = similarity(all_cmaps[i], all_cmaps[j])
        iou_mat[i, j] = iou(all_cmaps[i], all_cmaps[j])

# ── Análise intra vs inter grupo ──────────────────────────────────────────────

group_indices = {}
start = 0
for gname, words in groups.items():
    group_indices[gname] = list(range(start, start + len(words)))
    start += len(words)

print(f"\n{'='*60}")
print("SIMILARIDADE INTRA vs INTER GRUPO")
for gname, idxs in group_indices.items():
    intra = [sim_mat[i,j] for i in idxs for j in idxs if i < j]
    other = [k for k in range(N) if k not in idxs]
    inter = [sim_mat[i,j] for i in idxs for j in other]
    im = np.mean(intra) if intra else 0
    ie = np.mean(inter) if inter else 0
    ratio = im/ie if ie > 1e-6 else float('inf')
    print(f"\n  [{gname}]")
    print(f"    Intra: {im:.4f}  Inter: {ie:.4f}  Razão: {ratio:.2f}x")

print(f"\n{'='*60}")
print("ANAGRAMAS — pares:")
for i in group_indices["Anagramas"]:
    for j in group_indices["Anagramas"]:
        if i < j:
            print(f"  {words_only[i]:6s} vs {words_only[j]:6s} : "
                  f"corr={sim_mat[i,j]:.4f}  iou={iou_mat[i,j]*100:.1f}%")

# ── Visualizações ─────────────────────────────────────────────────────────────

# Matriz de similaridade
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Memória Associativa Pura — Sem Emitter, Sem Rede Neural', fontsize=12)

im0 = axes[0].imshow(sim_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(N)); axes[0].set_xticklabels(words_only, rotation=45, ha='right', fontsize=8)
axes[0].set_yticks(range(N)); axes[0].set_yticklabels(words_only, fontsize=8)
axes[0].set_title('Correlação de Pearson')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(iou_mat, cmap='Blues', vmin=0, vmax=1, aspect='auto')
axes[1].set_xticks(range(N)); axes[1].set_xticklabels(words_only, rotation=45, ha='right', fontsize=8)
axes[1].set_yticks(range(N)); axes[1].set_yticklabels(words_only, fontsize=8)
axes[1].set_title('Sobreposição IoU')
plt.colorbar(im1, ax=axes[1])

# Separadores de grupo
pos = 0
for gname, words in groups.items():
    for ax in axes:
        ax.axhline(pos - 0.5, color='black', lw=1.5)
        ax.axvline(pos - 0.5, color='black', lw=1.5)
    pos += len(words)

# Valores na matriz
for ax, mat in [(axes[0], sim_mat), (axes[1], iou_mat)]:
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center',
                    fontsize=5, color='black' if abs(mat[i,j]) < 0.7 else 'white')

plt.tight_layout()
plt.savefig('viz_associative_matrix.png', dpi=130, bbox_inches='tight')
plt.close()

# Crystal maps por grupo
for gname, idxs in group_indices.items():
    n = len(idxs)
    fig, axes2 = plt.subplots(1, n, figsize=(n * 3.5, 3.8))
    if n == 1: axes2 = [axes2]
    fig.suptitle(f'Crystal Maps — {gname}\n(perturbação direta, sem rede)', fontsize=10)
    vmax = max(all_cmaps[i].max() for i in idxs) + 1e-6
    for ax, idx in zip(axes2, idxs):
        n_crys = int((all_cmaps[idx] > 0.01).sum())
        ax.imshow(all_cmaps[idx], cmap='inferno', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_title(f'"{words_only[idx]}"\n{n_crys} cristais', fontsize=9)
        ax.axis('off')
    fname = f"viz_assoc_{gname.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"-> {fname}")

# Visualiza as gaussianas de caracteres individuais
print("\nVisualizando posições de caracteres no grid...")
fig, axes3 = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('Distribuição espacial de caracteres no campo\n(posição determinística por ASCII)', fontsize=11)
chars_demo = list("abcdefghijklmnop")
for ax, c in zip(axes3.flat, chars_demo):
    g = char_to_gaussian(c).cpu().numpy()
    ax.imshow(g, cmap='hot', interpolation='nearest')
    ax.set_title(f"'{c}' (ASCII={ord(c)})", fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('viz_assoc_char_positions.png', dpi=120, bbox_inches='tight')
plt.close()
print("-> viz_assoc_char_positions.png")

print("-> viz_associative_matrix.png")
print("\nPronto.")
