"""
Teste de saturação e capacidade de memória do PsiField.

Pergunta central: quantas entradas sequenciais o campo aguenta
antes de saturar — e o que acontece quando satura?

Experimento:
  1. Campo persiste (NÃO reseta entre entradas)
  2. Injeta N palavras/padrões sequencialmente
  3. Após cada injeção: mede cristais ativos, energia, e quanto
     do padrão da PRIMEIRA entrada ainda é recuperável
  4. Compara com vetor de memória convencional do mesmo tamanho (2304 floats)

Métricas:
  - n_cristais: quantos cristais ativos no campo
  - ocupacao: % do campo com cristal
  - retencao_1: IoU do crystal_map atual com o da primeira entrada
  - interferencia: quanto cada nova entrada apaga das anteriores
  - saturacao: ponto onde n_cristais para de crescer
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RN_psi_mnist import (
    psi_step, CrystalMemory,
    FIELD_SIZE, STIM_ON, STIM_TOTAL,
    DEVICE,
    CRYSTAL_SEP, CRYSTAL_MAX,
)

# ── Encoder físico (mesmo do teste associativo) ───────────────────────────────

def char_to_gaussian(c):
    v      = ord(c) / 127.0
    n_cells = 8
    cell_x  = ord(c) % n_cells
    cell_y  = (ord(c) // n_cells) % n_cells
    cx      = (cell_x + 0.5) / n_cells
    cy      = (cell_y + 0.5) / n_cells
    amp     = 1.5 + v * 2.5
    sigma   = 0.06
    coords  = torch.linspace(0., 1., FIELD_SIZE, device=DEVICE)
    xg, yg  = torch.meshgrid(coords, coords, indexing='ij')
    return amp * torch.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))

def text_to_perturbation(text):
    field = torch.zeros(FIELD_SIZE, FIELD_SIZE, device=DEVICE)
    for i, c in enumerate(text):
        field = field + char_to_gaussian(c) * float(np.cos(i * 0.3))
    return field.unsqueeze(0)   # (1, H, W)


# ── Estado persistente do campo ───────────────────────────────────────────────

class PersistentField:
    """
    Campo que NÃO reseta entre entradas.
    Acumula cristais de todas as entradas anteriores.
    """
    def __init__(self):
        self.field    = torch.zeros(1, FIELD_SIZE, FIELD_SIZE, device=DEVICE)
        self.velocity = torch.zeros(1, FIELD_SIZE, FIELD_SIZE, device=DEVICE)
        self.memory   = CrystalMemory(1)
        self.n_inputs = 0

    def inject(self, text):
        """Injeta uma entrada no campo persistente e retorna o crystal_map atual."""
        pert = text_to_perturbation(text)

        with torch.no_grad():
            for s in range(STIM_TOTAL):
                active = s < STIM_ON
                self.field, self.velocity = psi_step(
                    self.field, self.velocity, pert, active
                )
                self.memory.update_envelope(self.field)
                if self.memory.window_idx > 0:
                    self.memory.try_crystallize(self.field)
                self.field = self.memory.remit(self.field, None)

        self.n_inputs += 1
        return self.memory.crystal_map.squeeze(0).cpu().numpy()   # (H, W)

    @property
    def crystal_map(self):
        return self.memory.crystal_map.squeeze(0).cpu().numpy()


def iou(a, b, thr=0.01):
    ma, mb = a > thr, b > thr
    inter  = (ma & mb).sum()
    union  = (ma | mb).sum()
    return float(inter / union) if union > 0 else 0.0

def n_crystals(cmap, thr=0.01):
    return int((cmap > thr).sum())

def energia(cmap):
    return float(cmap.sum())


# ── Sequência de teste ────────────────────────────────────────────────────────

# 30 palavras variadas para injetar sequencialmente
sequence = [
    "amor", "casa", "vida", "sol", "lua",
    "mar", "rio", "flor", "vento", "fogo",
    "agua", "terra", "ar", "luz", "sombra",
    "tempo", "espaco", "mente", "corpo", "alma",
    "paz", "guerra", "bem", "mal", "verdade",
    "erro", "caminho", "porta", "janela", "chave",
]

print(f"Sequência: {len(sequence)} entradas")
print(f"Campo: {FIELD_SIZE}×{FIELD_SIZE} = {FIELD_SIZE**2} posições")
print(f"Separação mínima cristal: {CRYSTAL_SEP}px")
print(f"Limite teórico (não-sobrepostos): {FIELD_SIZE**2 // (2*CRYSTAL_SEP+1)**2} cristais")
print()

# ── Experimento 1: campo persistente ─────────────────────────────────────────

print("="*55)
print("EXPERIMENTO 1: Campo Persistente (sem reset)")
print("="*55)

pfield = PersistentField()

# Referências: crystal_map isolado de cada palavra
print("Computando crystal_maps isolados (referência)...")
isolated_cmaps = {}
for word in sequence:
    pf_iso = PersistentField()
    isolated_cmaps[word] = pf_iso.inject(word)

# Injeta sequencialmente no campo persistente
history = []
cmap_after_first = None

print("\nInjetando sequencialmente no campo persistente...")
print(f"{'N':>4}  {'Palavra':>10}  {'Cristais':>9}  {'Ocup%':>6}  {'Energia':>8}  {'Ret_1':>7}  {'Ret_self':>9}")
print("-" * 65)

for i, word in enumerate(sequence):
    cmap_current = pfield.inject(word)

    if i == 0:
        cmap_after_first = cmap_current.copy()

    nc    = n_crystals(cmap_current)
    occ   = nc / (FIELD_SIZE * FIELD_SIZE) * 100
    eng   = energia(cmap_current)
    ret1  = iou(cmap_current, cmap_after_first) * 100      # retenção da 1ª entrada
    rets  = iou(cmap_current, isolated_cmaps[word]) * 100  # quanto desta entrada está presente

    history.append({
        'n': i+1, 'word': word,
        'n_crystals': nc, 'ocupacao': occ, 'energia': eng,
        'retencao_1': ret1, 'retencao_self': rets,
    })

    print(f"{i+1:>4}  {word:>10}  {nc:>9}  {occ:>5.1f}%  {eng:>8.1f}  {ret1:>6.1f}%  {rets:>8.1f}%")

# ── Experimento 2: vetor de memória convencional ──────────────────────────────

print()
print("="*55)
print("EXPERIMENTO 2: Vetor convencional (mesmo tamanho)")
print("="*55)

# Vetor de memória = média acumulada de crystal_maps isolados
# Representa um "buffer de memória" do mesmo tamanho que o campo
mem_vector  = np.zeros(FIELD_SIZE * FIELD_SIZE)
conv_history = []
iso_flat_1  = isolated_cmaps[sequence[0]].flatten()

for i, word in enumerate(sequence):
    iso_flat = isolated_cmaps[word].flatten()

    # Memória convencional: média acumulada (EMA com alpha=0.3)
    alpha      = 0.3
    mem_vector = (1 - alpha) * mem_vector + alpha * iso_flat

    # Retenção da 1ª entrada no vetor
    def vec_similarity(a, b):
        sa, sb = a.std(), b.std()
        if sa < 1e-8 or sb < 1e-8: return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    ret1_conv = vec_similarity(mem_vector, iso_flat_1) * 100
    rets_conv = vec_similarity(mem_vector, iso_flat) * 100

    conv_history.append({
        'n': i+1, 'retencao_1': ret1_conv, 'retencao_self': rets_conv
    })

# ── Análise comparativa ───────────────────────────────────────────────────────

print()
print("="*55)
print("COMPARAÇÃO: Campo Físico vs Vetor Convencional")
print("="*55)
print(f"\n{'N':>4}  {'Ret_1 Campo':>12}  {'Ret_1 Vetor':>12}  {'Vantagem':>10}")
print("-" * 45)
for h, c in zip(history, conv_history):
    vantagem = h['retencao_1'] - c['retencao_1']
    print(f"{h['n']:>4}  {h['retencao_1']:>11.1f}%  {c['retencao_1']:>11.1f}%  {vantagem:>+9.1f}%")

# Ponto de saturação
print()
ns     = [h['n_crystals'] for h in history]
deltas = [ns[i] - ns[i-1] for i in range(1, len(ns))]
sat_pt = next((i+2 for i, d in enumerate(deltas) if d < 10), None)
if sat_pt:
    print(f"Ponto de saturação estimado: entrada #{sat_pt} ({sequence[sat_pt-1]})")
    print(f"  Cristais no ponto de saturação: {ns[sat_pt-1]}")
    print(f"  Ocupação: {ns[sat_pt-1]/(FIELD_SIZE**2)*100:.1f}%")
else:
    print("Campo ainda não saturou após todas as entradas.")
    print(f"  Cristais finais: {ns[-1]} / {FIELD_SIZE**2} ({ns[-1]/(FIELD_SIZE**2)*100:.1f}%)")

# ── Visualizações ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Saturação e Capacidade de Memória do PsiField', fontsize=13)

ns_plot  = [h['n'] for h in history]
nc_plot  = [h['n_crystals'] for h in history]
occ_plot = [h['ocupacao'] for h in history]
ret1_plot = [h['retencao_1'] for h in history]
rets_plot = [h['retencao_self'] for h in history]
ret1_conv = [c['retencao_1'] for c in conv_history]
rets_conv = [c['retencao_self'] for c in conv_history]

# 1. Crescimento de cristais
axes[0,0].plot(ns_plot, nc_plot, 'o-', color='#e6194b', linewidth=2)
axes[0,0].axhline(FIELD_SIZE**2 // (2*CRYSTAL_SEP+1)**2, color='gray',
                   linestyle='--', label=f'Limite teórico (~{FIELD_SIZE**2//(2*CRYSTAL_SEP+1)**2})')
axes[0,0].set_xlabel('N entradas injetadas')
axes[0,0].set_ylabel('Cristais ativos')
axes[0,0].set_title('Crescimento de Cristais no Campo Persistente')
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
words_ticks = [h['word'] for h in history]
axes[0,0].set_xticks(ns_plot[::3])
axes[0,0].set_xticklabels(words_ticks[::3], rotation=35, ha='right', fontsize=8)

# 2. Ocupação do campo
axes[0,1].fill_between(ns_plot, occ_plot, alpha=0.4, color='#3cb44b')
axes[0,1].plot(ns_plot, occ_plot, 'o-', color='#3cb44b', linewidth=2)
axes[0,1].set_xlabel('N entradas injetadas')
axes[0,1].set_ylabel('Ocupação (%)')
axes[0,1].set_title('Ocupação do Campo (% com cristal)')
axes[0,1].set_ylim(0, 100); axes[0,1].grid(alpha=0.3)
axes[0,1].set_xticks(ns_plot[::3])
axes[0,1].set_xticklabels(words_ticks[::3], rotation=35, ha='right', fontsize=8)

# 3. Retenção da 1ª entrada
axes[1,0].plot(ns_plot, ret1_plot, 'o-', color='#4363d8', linewidth=2, label='Campo Físico')
axes[1,0].plot(ns_plot, ret1_conv, 's--', color='#f58231', linewidth=2, label='Vetor Convencional')
axes[1,0].set_xlabel('N entradas injetadas')
axes[1,0].set_ylabel('IoU com 1ª entrada (%)')
axes[1,0].set_title('Retenção da 1ª Entrada após N injeções')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
axes[1,0].set_xticks(ns_plot[::3])
axes[1,0].set_xticklabels(words_ticks[::3], rotation=35, ha='right', fontsize=8)

# 4. Retenção da entrada atual
axes[1,1].plot(ns_plot, rets_plot, 'o-', color='#911eb4', linewidth=2, label='Campo Físico')
axes[1,1].plot(ns_plot, rets_conv, 's--', color='#f58231', linewidth=2, label='Vetor Convencional')
axes[1,1].set_xlabel('N entradas injetadas')
axes[1,1].set_ylabel('IoU entrada atual (%)')
axes[1,1].set_title('Presença da Entrada Atual no Campo')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
axes[1,1].set_xticks(ns_plot[::3])
axes[1,1].set_xticklabels(words_ticks[::3], rotation=35, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('viz_saturation.png', dpi=130, bbox_inches='tight')
plt.close()
print("\n-> viz_saturation.png")

# Crystal maps em 5 momentos: entrada 1, 5, 10, 20, 30
fig2, axes2 = plt.subplots(1, 5, figsize=(17, 3.5))
fig2.suptitle('Crystal Map do Campo Persistente após N entradas', fontsize=11)
checkpoints = [0, 4, 9, 19, 29]
pfield2     = PersistentField()
snapshots   = {}
for i, word in enumerate(sequence):
    pfield2.inject(word)
    if i in checkpoints:
        snapshots[i] = pfield2.crystal_map.copy()

vmax = max(s.max() for s in snapshots.values()) + 1e-6
for ax, cp in zip(axes2, checkpoints):
    nc = n_crystals(snapshots[cp])
    ax.imshow(snapshots[cp], cmap='inferno', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(f'Após {cp+1} entrada(s)\n"{sequence[cp]}"\n{nc} cristais', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('viz_saturation_snapshots.png', dpi=130, bbox_inches='tight')
plt.close()
print("-> viz_saturation_snapshots.png")
print("\nPronto.")
