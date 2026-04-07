# Seção: Resultados de Aprendizado — ResNet-Ψ

## 6. Validação do Aprendizado: Da Física à Computação

### 6.1 Contexto

O paper original (Seções 1-5) valida o Ψ-Field como substrato computacional — cristais emergem, re-emitem e são seletivos em frequência. A Seção 5 propõe a arquitetura Emitter→Campo→Decoder como caminho para aprendizado, deixando a validação experimental como trabalho futuro.

Esta seção apresenta os resultados dessa validação.

---

### 6.2 O Problema da Otimização Conjunta

A primeira abordagem otimizava emitter e decoder simultaneamente via CMA-ES num espaço de 40 dimensões (3-bit) ou 48 dimensões (4-bit).

**Resultado:** convergência inconsistente (~83%), predições com margem fraca (0.39–0.66), forte dependência do ponto inicial.

**Diagnóstico:** o otimizador tentava resolver dois problemas conflitantes ao mesmo tempo:
- *Como injetar ondas que criem padrões distinguíveis no campo?* (responsabilidade do emitter)
- *Como ler o campo para classificar corretamente?* (responsabilidade do decoder)

Quando o emitter muda, o campo muda, e os pesos do decoder que funcionavam deixam de funcionar. O espaço de busca conjunto é caótico.

---

### 6.3 Solução: Otimização em 2 Fases com Aproximação de Campos

A solução foi separar as responsabilidades em duas fases sequenciais, com o emitter treinado explicitamente para **aproximar campos de entradas similares e afastar campos de entradas diferentes**.

#### Fase 1 — Treinamento do Emitter (Separabilidade do Campo)

O emitter é otimizado com uma métrica de separabilidade geométrica no espaço de leituras do campo:

```
Loss_emitter = −separação_inter_classe + variância_intra_classe

onde:
  separação_inter_classe = ‖centróide(classe=1) − centróide(classe=0)‖
  variância_intra_classe = var(leituras | classe=0) + var(leituras | classe=1)
```

**O que isso faz:** força o campo físico a organizar as combinações de entrada de forma que classes diferentes ocupem regiões distintas no espaço de leituras. O campo — através de propagação, interferência e cristalização — realiza a transformação não-linear necessária.

**Por que isso funciona:** o campo físico é um sistema dinâmico rico. O emitter não precisa programar a separação — só precisa encontrar a "sintonia" de ondas que deixa a física organizar a informação. A física faz o trabalho não-linear.

#### Fase 2 — Treinamento do Decoder (Classificação)

Com o emitter fixo e o campo já organizado de forma separável, o decoder — uma simples transformação linear — encontra os pesos de leitura corretos rapidamente.

```
logit = Σ field(posição_i) × peso_i
pred  = sigmoid(logit)
```

---

### 6.4 Resultados — Paridade 3-bit

**Configuração:** 8 combinações, XOR(S1, S2, S3), campo persistente entre estímulos, 5 runs independentes com seeds aleatórias.

| Run | Convergência | Loss final | Fase 2 (gens até 8/8) | Tempo total |
|-----|-------------|------------|----------------------|-------------|
| 1   | ✓ 8/8       | 0.8304     | gen 65               | 1618s       |
| 2   | ✓ 8/8       | 1.7923     | gen 50               | 1269s       |
| 3   | ✓ 8/8       | 0.3615     | gen 15               | 1051s       |
| 4   | ✓ 8/8       | 0.6234     | gen 19               | 1177s       |
| 5   | ✓ 8/8       | 1.1984     | gen 8                | 1098s       |

**Taxa de convergência: 5/5 (100%)**
**Comparação com otimização conjunta: ~83% → 100%**

Predições típicas após convergência:
```
S1=0 S2=0 S3=0 → 0.0000 (target 0) ✓
S1=0 S2=0 S3=1 → 1.0000 (target 1) ✓
S1=0 S2=1 S3=0 → 1.0000 (target 1) ✓
S1=0 S2=1 S3=1 → 0.0000 (target 0) ✓
S1=1 S2=0 S3=0 → 0.8856 (target 1) ✓
S1=1 S2=0 S3=1 → 0.1477 (target 0) ✓
S1=1 S2=1 S3=0 → 0.0768 (target 0) ✓
S1=1 S2=1 S3=1 → 0.9998 (target 1) ✓
```

Predições saturadas (0.0000 e 1.0000) indicam que o campo está bem separado — o decoder linear encontrou uma solução de alta margem.

**Observação crítica:** o decoder é uma transformação linear. Se o sistema resolve paridade XOR (função não-linear) com decoder linear, a não-linearidade foi realizada inteiramente pelo campo físico. A física é o kernel.

---

### 6.5 Resultados — Paridade 4-bit

**Configuração:** 16 combinações, XOR(S1,S2,S3,S4), campo persistente entre os 4 estímulos sequenciais, 2 runs (GPU T4 + CPU).

| Run | Hardware | Convergência | Loss final | Fase 2 (gens até 16/16) | Tempo total |
|-----|----------|-------------|------------|------------------------|-------------|
| 1   | GPU T4   | ✓ 16/16     | 0.2288     | gen 7                  | 1869s       |
| 2   | CPU      | ✓ 16/16     | 0.1585     | gen 9                  | 2265s       |

**Destaque:** o par `1011` e `1110` — que antes era indistinguível com campo resetado por estímulo — foi resolvido com predições `1.000` e `0.998`. A ordem dos estímulos foi preservada pelo campo persistente.

Predições (Run 2, Loss 0.1585):
```
0000→0  0.000 ✓    1000→1  0.997 ✓
0001→1  1.000 ✓    1001→0  0.000 ✓
0010→1  1.000 ✓    1010→0  0.003 ✓
0011→0  0.000 ✓    1011→1  1.000 ✓
0100→1  1.000 ✓    1100→0  0.117 ✓
0101→0  0.005 ✓    1101→1  0.999 ✓
0110→0  0.000 ✓    1110→1  0.998 ✓
0111→1  0.999 ✓    1111→0  0.018 ✓
```

#### 6.5.1 Cristais Formados

| Métrica | Run 1 (GPU) | Run 2 (CPU) |
|---------|-------------|-------------|
| Total de cristais | 147 | 191 |
| Média por combinação | 9.2 | 11.9 |
| Células ocupadas | 114/2304 | 109/2304 |
| Ocupação do campo | **4.9%** | **4.7%** |

**Observação fundamental:** o campo 48×48 usou menos de 5% de sua capacidade para armazenar e computar paridade de 4 bits sequencial. 95% do campo permanece disponível — a capacidade de memória não foi esgotada.

A distribuição de cristais reflete a complexidade de cada combinação:
- `0000→0`: 1–23 cristais (combinação mais simples — todos zeros)
- `1011→1`: 6–19 cristais (combinação com sequência complexa)

Os cristais emergem espontaneamente da dinâmica do campo — não são programados. A física decide onde e quantos cristais se formam com base na estrutura das ondas injetadas.

---

### 6.6 Generalização para Dados Contínuos

Para validar que a arquitetura não é restrita a dados binários, testamos classificação de pontos 2D com valores contínuos.

**Configuração:**
- 20 pontos `(x, y)` com `x, y ∈ [-1, 1]`
- 4 classes (quadrantes do plano)
- Emitter: 30 parâmetros — aprende `param = base + coef_x·x + coef_y·y`
- Decoder: matriz 16×4 (64 parâmetros)

**Resultado: 20/20 (100%)**, Loss 0.0466

```
(-0.58, -0.70) → Q3(−,−) ✓    (0.75, -0.57) → Q4(+,−) ✓
(-0.89,  0.76) → Q1(−,+) ✓    (0.39,  0.74) → Q2(+,+) ✓
... (todos os 20 pontos classificados corretamente)
```

A Fase 1 atingiu Sep-Loss de −34.17 — separabilidade muito maior que nos dados binários (−2.5), refletindo que dados contínuos criam padrões mais ricos no campo.

**Significado:** o emitter generaliza além de dados binários. Qualquer dado numérico contínuo pode ser convertido em ondas. A física do campo organiza a informação independentemente do tipo de entrada.

---

### 6.7 Análise: Por Que a Separação de Fases Funciona

A chave é que o campo físico realiza uma **transformação de kernel implícita**.

Em SVMs, um kernel mapeia dados para um espaço de alta dimensão onde são linearmente separáveis. Aqui, o campo de ondas realiza essa transformação através de física — propagação, interferência e cristalização criam representações ricas a partir de entradas simples.

O emitter (Fase 1) aprende a "sintonia" que ativa esse kernel físico de forma ótima para o problema. O decoder (Fase 2) opera no espaço já transformado — trivialmente linear.

```
Entrada → [Emitter] → ondas → [Campo Físico] → padrão cristalizado → [Decoder] → saída
           treinável           kernel físico                           linear
```

A separação de fases é natural porque emitter e decoder operam em espaços diferentes:
- Emitter: espaço de parâmetros de onda
- Decoder: espaço de leituras do campo já transformado

---

### 6.8 Comparação com Transformers

| Métrica | Transformer (mínimo) | ResNet-Ψ (4-bit) |
|---------|---------------------|------------------|
| Parâmetros treináveis | ~millions | **48** |
| Memória durante inferência | O(n²) com contexto | **O(1) fixo** |
| Memória para 4 tokens | 4×4 = 16 (attention) | **2304 pixels** |
| Memória para 400 tokens | 400×400 = 160.000 | **2304 pixels** |
| Ocupação de memória usada | proporcional ao contexto | **< 5% do campo** |
| Tipo de aprendizado | gradiente estatístico | **física emergente** |

A memória do campo é estática independente do comprimento da sequência. Os cristais acumulam a história da sequência sem alocar memória adicional.

---

### 6.9 Questão Aberta: Relação Campo × Cristais

O resultado de 4.7–4.9% de ocupação levanta a questão central para escalabilidade:

> *"Qual é a relação entre a complexidade da tarefa e a ocupação do campo?"*

Se a ocupação crescer **linearmente** com a complexidade (e não quadraticamente), o campo pode ser dimensionado de forma eficiente para tarefas mais complexas.

Hipótese a testar:

```
4-bit   → 16 classes  → ~5% ocupado    ← demonstrado
8-bit   → 256 classes → ~?% ocupado    ← próximo experimento
16-bit  → 65536 classes → ~?% ocupado  ← experimento futuro
```

Se um campo 128×128 resolver 8-bit com ~5% de ocupação, confirma que a relação é linear — e que escalar o campo linearmente habilita capacidade quadraticamente maior.

---

### 6.10 Resumo dos Resultados

| Experimento | Resultado | Params treináveis | Ocupação do campo |
|-------------|-----------|-------------------|-------------------|
| Paridade 3-bit (5 runs) | **5/5 (100%)** | 40 | — |
| Paridade 4-bit (GPU) | **16/16** | 48 | 4.9% |
| Paridade 4-bit (CPU) | **16/16** | 48 | 4.7% |
| Classificação 2D contínua | **20/20** | 94 | — |

A otimização em 2 fases com aproximação de campos via métrica de separabilidade transforma um problema de otimização caótico (40-48 dimensões acopladas) em dois problemas estruturados menores, onde o campo físico realiza o trabalho não-linear e o decoder opera num espaço linearmente separável.
