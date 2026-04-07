# ResNet-Ψ — Otimização em 2 Fases: Resultados

## Motivação

A abordagem original otimizava emitter (24 params) e decoder (16 params) juntos via CMA-ES num espaço de 40 dimensões. O resultado era inconsistente — o otimizador se perdia no espaço de busca e a convergência dependia da sorte do ponto inicial.

**Taxa de convergência original (CMA-ES conjunto):** ~83% (5/6 runs observadas)

---

## A Hipótese

O decoder é apenas 16 pesos lineares. Se o campo físico for **linearmente separável** — ou seja, se as 8 combinações de entrada produzirem padrões distinguíveis nas posições de leitura — o decoder trivialmente encontra os pesos certos.

O problema não é o decoder. É o emitter não estar treinado para criar padrões distinguíveis.

**Solução:** separar a otimização em duas fases sequenciais.

---

## Arquitetura das 2 Fases

### Fase 1 — Emitter (separabilidade do campo)

- Otimiza apenas os 24 parâmetros do emitter
- Métrica: **separabilidade entre classes** no campo físico
  - Maximiza distância entre centróide das combinações com paridade=1 e paridade=0 nas posições de leitura
  - Minimiza variância intra-classe
- Loss: `−separação + (var_pos + var_neg)`
- CMA-ES com 150 gerações máximas, pop_size=20

### Fase 2 — Decoder (classificação)

- Emitter **fixo** (congelado após Fase 1)
- Otimiza apenas os 16 pesos do decoder
- Métrica: Binary Cross-Entropy padrão
- CMA-ES com 300 gerações máximas, pop_size=20
- Elitismo + refinamento ao atingir 8/8

---

## Resultados — 5 Runs Independentes (CPU, sem GPU)

| Run | Convergência | Loss final | Fase 2 (gens até 8/8) | Tempo total |
|-----|-------------|------------|----------------------|-------------|
| 1   | ✓ 8/8       | 0.8304     | gen 65               | 1618s       |
| 2   | ✓ 8/8       | 1.7923     | gen 50               | 1269s       |
| 3   | ✓ 8/8       | 0.3615     | gen 15               | 1051s       |
| 4   | ✓ 8/8       | 0.6234     | gen 19               | 1177s       |
| 5   | ✓ 8/8       | 1.1984     | gen 8                | 1098s       |

**Taxa de convergência: 5/5 (100%)**
**Tempo médio: 1242s (~21 min em CPU)**
**Loss médio: 0.9612**

---

## Predições — Exemplo Run 3 (melhor loss: 0.3615)

| Entrada       | Predição | Target | Resultado |
|---------------|----------|--------|-----------|
| S1=0 S2=0 S3=0 | 0.0000  | 0      | ✓         |
| S1=0 S2=0 S3=1 | 1.0000  | 1      | ✓         |
| S1=0 S2=1 S3=0 | 1.0000  | 1      | ✓         |
| S1=0 S2=1 S3=1 | 0.0000  | 0      | ✓         |
| S1=1 S2=0 S3=0 | 0.8856  | 1      | ✓         |
| S1=1 S2=0 S3=1 | 0.1477  | 0      | ✓         |
| S1=1 S2=1 S3=0 | 0.0768  | 0      | ✓         |
| S1=1 S2=1 S3=1 | 0.9998  | 1      | ✓         |

Predições saturadas (0.0000 e 1.0000) indicam alta confiança — o campo está bem separado.

---

## O Que Isso Significa

### 1. A física faz o trabalho não-linear

O decoder são 16 pesos lineares — sem camadas ocultas, sem não-linearidade. Se o sistema resolve paridade XOR (que é não-linear) com um decoder linear, significa que **o campo físico realizou a transformação não-linear**. A propagação de ondas, interferência e cristalização projetam os dados num espaço onde as classes são linearmente separáveis.

Analogia: kernels em SVMs fazem isso explicitamente. Aqui, **a física é o kernel**.

### 2. Separação de fases resolve o problema de convergência

CMA-ES conjunto (40 dims) → ~83% de convergência, margem fraca
CMA-ES em 2 fases (24 + 16 dims) → **100% de convergência**, predições saturadas

Reduzir a dimensionalidade e estruturar o problema em etapas sequenciais elimina a dependência de sorte do ponto inicial.

### 3. Memória de tamanho fixo

O campo 48×48 = 2304 pixels processa sequências de 3 estímulos sem crescer. A memória é constante independente do número de tokens processados — ao contrário de Transformers onde a memória cresce com O(n²) no contexto.

---

## Comparação com Abordagem Original

| Métrica                  | CMA-ES Conjunto | 2 Fases      |
|--------------------------|-----------------|--------------|
| Dimensões otimizadas     | 40 (simultâneo) | 24 → 16 (sequencial) |
| Taxa de convergência     | ~83%            | **100%**     |
| Gens Fase 2 até 8/8      | 67-92           | **8-65**     |
| Predições típicas        | 0.39-0.66       | **0.00-1.00** |
| Dependência do seed      | Alta            | **Baixa**    |

---

## Próximos Passos

1. **Aplicar 2 fases ao 4-bit** (requer GPU — T4 ou superior)
2. **Medir cristais formados** durante runs bem-sucedidas
3. **Atualizar o paper** com seção de resultados de aprendizado
