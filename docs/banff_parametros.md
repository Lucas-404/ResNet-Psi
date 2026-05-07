# BANFF — Parâmetros e Resultado

## Arquitetura

| Componente | Valor | Status |
|---|---|---|
| Neurônios no reservatório (N_RES) | 1000 | Fixo |
| Esparsidade das conexões | 10% | Fixo |
| Raio espectral (W_res) | 0.9 | Fixo |
| Pesos de entrada W_in | aleatórios × 0.1 | Fixo |
| Pesos de saída W_out | aleatórios × 0.1 | Fixo |
| Bias | zeros (inicialização) | **Único parâmetro treinável** |

## Dinâmica do reservatório

```
h(t+1) = tanh( h(t) @ W_res.T + x @ W_in.T + bias )
```

- Steps por amostra: 10
- Estado inicial: zeros

## Treino do bias

| Parâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Learning rate | 0.05 |
| Epochs | 30 |
| Batch size | 256 |
| Loss | Cross-entropy |
| Parâmetros ajustados | 1000 (só bias) |

## Resultados por tamanho de dataset

| Sistema | 5k treino / 1k teste | 60k treino / 10k teste |
|---|---|---|
| BANFF | 75.4% | **81.6%** |
| ResNet-Ψ | **79.9%** | 79.5% |

### Dataset subconjunto (5k/1k)
- Batch size: 256 | Seed: 42

### Dataset completo (60k/10k — MNIST padrão)
- Batch size: 512

## Observação: escalabilidade

BANFF escala com mais dados (+6.2% de 5k para 60k) — gradiente se beneficia de mais exemplos.
ResNet-Ψ não muda (-0.4%) — o protótipo (média por classe) converge com ~500 amostras e não melhora com 6.000.

## Conclusão

Com dataset completo (60k/10k), BANFF supera ResNet-Ψ por 2.1% (81.6% vs 79.5%).
Com poucos dados (5k/1k), ResNet-Ψ supera BANFF por 4.5% (79.9% vs 75.4%).

ResNet-Ψ é data-efficient: atinge performance similar com muito menos dados, sem nenhum ajuste automático.
