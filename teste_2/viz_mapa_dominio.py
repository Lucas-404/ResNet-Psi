"""
Figura resumo: Mapa de Domínio da ResNet-Psi

Todos os datasets testados num único gráfico.
Separa funciona / não funciona pela linha de ratio 3x.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# Dados consolidados (todos os audits)
# ======================================================================

datasets = [
    # (nome, acc, chance, n_classes, tipo)
    # Funciona
    ("EMNIST Digits",   77.4, 10.0,  10, "digitos"),
    ("MNIST",           69.7, 10.0,  10, "digitos"),
    ("Pneumonia",       69.2, 50.0,   2, "medico"),
    ("Breast",          66.7, 50.0,   2, "medico"),
    ("Fashion-MNIST",   62.4, 10.0,  10, "silhuetas"),
    ("BloodMNIST",      55.0, 12.5,   8, "medico"),
    ("OrganAMNIST",     47.7,  9.1,  11, "medico"),
    ("PathMNIST",       43.0, 11.1,   9, "medico"),
    ("EMNIST Letters",  40.2,  3.8,  26, "letras"),
    # Fraco / Falha
    ("RetinaMNIST",     35.2, 20.0,   5, "medico"),
    ("OCTMNIST",        25.1, 25.0,   4, "medico"),
    ("CIFAR-10",        18.7, 10.0,  10, "natural"),
    ("DermaMNIST",      16.1, 14.3,   7, "textura"),
    ("SVHN",             9.8, 10.0,  10, "natural"),
]

nomes   = [d[0] for d in datasets]
accs    = [d[1] for d in datasets]
chances = [d[2] for d in datasets]
n_cls   = [d[3] for d in datasets]
tipos   = [d[4] for d in datasets]
ratios  = [a / c for a, c in zip(accs, chances)]

# Cores por tipo
cor_map = {
    "digitos": "#2196F3",
    "silhuetas": "#FF9800",
    "letras": "#9C27B0",
    "medico": "#4CAF50",
    "natural": "#F44336",
    "textura": "#F44336",
}
cores = [cor_map[t] for t in tipos]

# ======================================================================
# Figura 1: Barras horizontais — Acurácia + Chance
# ======================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('ResNet-Ψ — Mapa de Domínio: 14 Datasets, Zero Treino',
             fontsize=14, fontweight='bold')

# --- Gráfico 1: Acurácia ---
ax = axes[0]
y_pos = np.arange(len(nomes))

bars = ax.barh(y_pos, accs, height=0.6, color=cores, alpha=0.85, label='ResNet-Ψ')
ax.barh(y_pos, chances, height=0.6, color='lightgray', alpha=0.4, label='Chance')

# Linha divisória
ax.axvline(x=0, color='black', linewidth=0.5)

for i, (acc, chance, ratio) in enumerate(zip(accs, chances, ratios)):
    ax.text(acc + 1, i, f'{acc:.1f}% ({ratio:.1f}x)', va='center', fontsize=8, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(nomes, fontsize=10)
ax.set_xlabel('Acurácia (%)', fontsize=11)
ax.set_title('Acurácia Zero Treino', fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.legend(loc='lower right', fontsize=9)

# Separador visual entre funciona / não funciona
ax.axhline(y=8.5, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
ax.text(50, 8.7, '— linha divisória —', ha='center', fontsize=8, color='red', alpha=0.7)

# --- Gráfico 2: Ratio (acc/chance) ---
ax2 = axes[1]

bars2 = ax2.barh(y_pos, ratios, height=0.6, color=cores, alpha=0.85)
ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=3, color='red', linestyle='--', alpha=0.5, linewidth=1)

for i, ratio in enumerate(ratios):
    ax2.text(ratio + 0.1, i, f'{ratio:.1f}x', va='center', fontsize=8, fontweight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(nomes, fontsize=10)
ax2.set_xlabel('Ratio (Acurácia / Chance)', fontsize=11)
ax2.set_title('Quanto acima da chance?', fontsize=12)
ax2.invert_yaxis()

# Anotações
ax2.text(1.1, 13.5, 'chance (1x)', fontsize=8, color='gray', alpha=0.7)
ax2.text(3.1, 13.5, 'limiar (3x)', fontsize=8, color='red', alpha=0.7)

ax2.axhline(y=8.5, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

# Legenda de tipos
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2196F3', alpha=0.85, label='Dígitos'),
    Patch(facecolor='#FF9800', alpha=0.85, label='Silhuetas'),
    Patch(facecolor='#9C27B0', alpha=0.85, label='Letras'),
    Patch(facecolor='#4CAF50', alpha=0.85, label='Médico'),
    Patch(facecolor='#F44336', alpha=0.85, label='Natural/Textura (falha)'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('viz_mapa_dominio.png', dpi=150, bbox_inches='tight')
plt.close()

# ======================================================================
# Figura 2: Scatter — Acurácia vs Número de classes
# ======================================================================

fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))
fig2.suptitle('ResNet-Ψ — Acurácia vs Complexidade (Zero Treino)',
              fontsize=13, fontweight='bold')

for i in range(len(datasets)):
    marker = 'o' if ratios[i] >= 3 else 'x'
    size = 120 if ratios[i] >= 3 else 80
    ax3.scatter(n_cls[i], accs[i], c=cores[i], s=size, marker=marker,
                edgecolors='black', linewidth=0.5, zorder=3)
    ax3.annotate(nomes[i], (n_cls[i], accs[i]),
                 textcoords="offset points", xytext=(8, 4), fontsize=7)

# Linha de chance
x_range = np.arange(2, 28)
ax3.plot(x_range, 100.0 / x_range, 'k--', alpha=0.3, label='Chance')
ax3.plot(x_range, 3 * 100.0 / x_range, 'r--', alpha=0.3, label='3x chance')

ax3.set_xlabel('Número de classes', fontsize=11)
ax3.set_ylabel('Acurácia (%)', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_xlim(1, 28)
ax3.set_ylim(0, 85)

plt.tight_layout()
plt.savefig('viz_mapa_dominio_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("-> viz_mapa_dominio.png")
print("-> viz_mapa_dominio_scatter.png")
print("Pronto.")
