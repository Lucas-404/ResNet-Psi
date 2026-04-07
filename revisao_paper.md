# Revisão do Paper — ResNet-Ψ v2 → v3

Baseado nos resultados dos 12 audits experimentais realizados após a publicação do paper v2.
Cada seção do paper é analisada: o que está certo, o que está errado, e o que precisa mudar.

---

## 1. RESUMO (Página 1)

### Claims atuais do resumo:
> "Validamos a fundação da arquitetura através de quatro testes rigorosos: robustez de cristalização (100%), re-emissão de cristais (razão de energia 136.8×), seletividade de frequência (1.57×) e estrutura espectral (4.56× acima de ruído aleatório)."

**Status**: CORRETO — os 4 testes de fundação não foram invalidados.

---

> "Demonstramos que dados podem ser injetados diretamente no campo sem módulo intermediário treinável — e que adicionar um emitter MLP (988k parâmetros) degrada a acurácia de 87.2% para ~40% no MNIST, provando que a física do campo é o transformador."

**Status**: CORRETO — Audit 5 confirmou que injeção direta supera emitter.

**Atualização necessária**: A acurácia de referência mudou. Com cristalização competitiva (Audit 7), injeção direta + decoder linear = **88.1%**, não 80.8%. Com MLP decoder = **93.1%**. A comparação fica ainda mais forte: 93.1% sem emitter vs ~40% com emitter.

---

> "Identificamos uma constante de informação universal: βc = 2.32 bits por cristal, determinada pelas constantes físicas do sistema."

**Status**: ERRADO — Audit 1 provou que βc = 2.32 é tautológico.

**O que aconteceu**: βc = log₂(1/α) = log₂(1/0.20) = log₂(5) ≈ 2.32. O valor não foi "medido" — foi calculado diretamente do threshold α = 0.20 que foi escolhido arbitrariamente. Qualquer α produziria um βc diferente. Não é propriedade do sistema.

**O que o Audit 1 encontrou**: MI real ≈ 4.75 bits via clustering (KMeans + mutual_info_score), e essa MI é **constante** entre diferentes configurações de (γ, β). Isso sim é propriedade intrínseca do sistema.

**Correção**: Remover βc = 2.32 completamente. Substituir por:
- MI ≈ 4.75 bits (medido via clustering, não via threshold)
- MI é constante entre configurações físicas diferentes
- A capacidade informacional é propriedade intrínseca do campo, não dos parâmetros

---

> "A lei de escala C(N) = 0.0024 × N2.171 descreve a capacidade do campo em função do tamanho."

**Status**: PARCIALMENTE CORRETO — A lei de escala existe, mas os coeficientes foram medidos com cristalização original. Com cristalização competitiva os números mudam.

**O que o Audit 2 confirmou**: Acurácia escala com tamanho do campo: 48² (80.8%) → 192² (87.5%) com decoder linear. Scaling real confirmado com 10 seeds e barras de erro.

**O que o Audit 11 revelou**: O scaling funciona com decoder, mas NÃO funciona com protótipos (zero treino). Campo 96² deu 77.2% vs 76.4% no 48² — diferença insignificante. O benefício do campo maior é capturado pelo decoder, não pela métrica de distância simples.

**Correção**: Manter a lei de escala mas:
- Re-medir coeficientes com cristalização competitiva
- Explicitar que o scaling beneficia o decoder, não os protótipos
- Remover extrapolações para campos 683+ (OOM na T4, não testado com v2)

---

> "Por fim, demonstramos completação de frases por física pura — sem embeddings, atenção ou pesos treinados — com 16/16 acertos (100%) em corpus de 25 frases."

**Status**: ERRADO — Audit 3 e Audit 12 invalidaram completamente.

**O que aconteceu**: O score do paper usava 50% overlap léxico + 50% overlap físico. O 100% de acerto vinha da parte léxica (comparação de strings), não da física. A física sozinha dá 31%.

**O que os Audits 3 e 12 mostraram**:
- Audit 3: Física pura = 31%. O resultado era artefato da métrica híbrida léxica.
- Audit 12 (hash): 20% (aleatório) — hash não carrega semântica, física não consegue diferenciar.
- Audit 12 (embedding semântico manual): 47.5% protótipos, 70% multiple choice — quando a entrada carrega estrutura semântica, a física captura. Mas os embeddings são manuais (hardcoded), o que é treino disfarçado.

**Correção**: Remover a seção de completação de frases do paper, ou reescrevê-la honestamente:
- "A física sozinha não consegue processar texto quando a entrada não carrega estrutura semântica"
- "Quando embeddings semânticos são fornecidos, o mecanismo captura relações (70% em 5 classes) — mas a ResNet-Ψ depende de que a similaridade da entrada reflita a similaridade semântica"
- "O mecanismo é um amplificador de estrutura: preserva relações que já existem na entrada, não inventa relações novas"

---

## 2. ARQUITETURA (Páginas 2-4)

### 2.1 Representação de Onda
**Status**: CORRETO — A equação de onda não mudou.

### 2.2 Injeção Direta de Dados
**Status**: CORRETO — Confirmado por todos os audits MNIST. A projeção gaussiana funciona.

**Adição**: Explicitar que a projeção gaussiana preserva similaridade geométrica da entrada. Isso é o motivo pelo qual funciona para imagens (pixels de "7" → perturbação similar → cristais similares) e não funciona para texto com hash (palavras semanticamente próximas → perturbações aleatoriamente diferentes → cristais diferentes).

### 2.3 Dinâmica do Ψ-Field
**Status**: CORRETO — Constantes físicas confirmadas. Tabela 1 está correta.

### 2.4 Formação de Cristais (SEÇÃO CRÍTICA — REESCREVER)

**Status**: ERRADO — O paper descreve cristalização com thresholds duros. Audit 4 provou que essa cristalização piora a acurácia em 10% vs campo bruto. Audit 7 provou que cristalização competitiva é o mecanismo correto.

**O que está no paper (cristalização original)**:
```
Cristal se forma se:
  (i)   mean(env) > A_min = 0.3        ← threshold duro
  (ii)  cv(env) < ε = 0.15             ← threshold duro
  (iii) mean(env) < 8.0                ← threshold duro
  (iv)  separação > 5 pixels           ← exclusão espacial
```

**O que deveria estar (cristalização competitiva)**:
```
Cristal se forma com score contínuo:
  amp_score = sigmoid(sharpness × (mean - A_min))     ← threshold suave
  cv_score  = sigmoid(sharpness × (CV_max - cv))      ← threshold suave
  sat_score = sigmoid(sharpness × (8.0 - mean))       ← threshold suave
  candidato = amp_score × cv_score × sat_score         ← score contínuo [0, 1]

Competição (mecanismo novo):
  - Cada cristal tem HP (hit points / vida)
  - Cristais nascem com HP = 1
  - Ressonância: se onda é forte onde há cristal → HP += ressonance_boost (0.1)
  - Decay: todos cristais perdem HP -= 0.02 por step
  - Morte: cristais com HP ≤ 0 são removidos (crystal_map × alive)

Parâmetros novos:
  sharpness = 5.0       (suavidade do sigmoid)
  decay = 0.02          (perda de vida por step)
  ressonance_boost = 0.1 (ganho de vida por ressonância)
```

**Por que é melhor** (Audit 7 — 5 seeds):

| Variante | Linear | MLP | MI (bits) |
|----------|--------|-----|-----------|
| Original (thresholds duros) | 80.9% | 88.4% | 1.26 |
| Suave (sigmoid) | 84.0% | 90.2% | 1.78 |
| **Competitivo (sigmoid + HP)** | **88.1%** | **93.1%** | **2.49** |

A competição faz seleção natural: cristais que não ressoam com a onda morrem, cristais que ressoam sobrevivem. Isso remove ruído e mantém só os cristais informativos. Resultado: +7.2% linear, +4.7% MLP, +97% MI.

**Correção**: Reescrever a seção 2.4 inteira com o mecanismo competitivo. Mover cristalização original para "trabalho anterior" ou apêndice.

---

### 2.5 Mapa de Frequência Temporal
**Status**: NÃO TESTADO nos audits. O frequency_map existe no código original mas não foi usado na cristalização competitiva (que usa envelope puro). Pode ser removido ou mantido como feature secundária.

### 2.6 Função de Ressonância e Re-emissão
**Status**: SIMPLIFICADO — A re-emissão nos audits usa versão mais simples:
```python
field += crystal_map × CRYSTAL_REMIT × sign(field)
```
Sem R_freq, sem R_espacial, sem matching harmônico. A versão simplificada funciona melhor nos benchmarks. A equação complexa do paper (0.7·R_freq + 0.3·R_espacial) pode ser mantida como formulação teórica mas precisa notar que a implementação que produz os melhores resultados usa re-emissão direta.

---

## 3. VALIDAÇÃO NÍVEL 1 (Páginas 4-5)

### 3.1 Robustez de Cristalização (100%)
**Status**: CORRETO — Não invalidado.

### 3.2 Re-emissão 136.8×
**Status**: CORRETO — Não invalidado. Mas foi medido com cristalização original. Pode ser re-medido com competitiva.

### 3.3 Seletividade de Frequência 1.57×
**Status**: CORRETO — Não invalidado. O paper já nota que é "modesto comparado ao attention".

### 3.4 Estrutura Espectral 4.56×
**Status**: CORRETO — Não invalidado.

**Nota**: Esses 4 testes de fundação são sólidos. Não precisam mudar.

---

## 4. VALIDAÇÃO NÍVEL 2 (Páginas 5-8) — MUDANÇAS PESADAS

### 4.1 Hash Físico Determinístico
**Status**: CORRETO — Confirmado. Mesma entrada → correlação 1.000.

### 4.2 Ablação Emitter MLP

**Status**: CORRETO na conclusão (injeção direta > emitter), mas DESATUALIZADO nos números.

**Tabela atual do paper**:
| Config | Acurácia |
|--------|----------|
| Injeção direta + linear | 80.8% |
| Emitter MLP | ~40% |

**Tabela corrigida** (com cristalização competitiva, Audit 7):
| Config | Acurácia | Params |
|--------|----------|--------|
| Injeção direta + linear (competitivo) | **88.1%** | 23k |
| Injeção direta + MLP (competitivo) | **93.1%** | ~25k |
| Emitter MLP | ~40% | 988k |

A diferença fica ainda mais dramática: 93.1% com zero params no encoder vs 40% com 988k params.

**Adicionar** (Audit 5 — ablação dos cristais):
| Config | Linear | MLP |
|--------|--------|-----|
| Campo bruto (sem cristais) | 91.7% | 96.5% |
| Cristalização original | 80.8% | 88.4% |
| Cristalização competitiva | **88.1%** | **93.1%** |

**Nota importante**: Campo bruto (sem cristais nenhum) dá 91.7% linear. Cristalização original PIORA pra 80.8%. Competitiva recupera pra 88.1% mas ainda perde pro bruto. Cristais não são necessariamente melhores que campo bruto pra acurácia com decoder — mas são essenciais pra o modo zero treino (protótipos).

---

### 4.3 Curva de Escala

**Status**: PARCIALMENTE CORRETO.

**O que o paper diz**: 48→683, satura em ~87%.

**O que os audits confirmam** (Audit 2, 10 seeds):
| Campo | Acurácia (linear) |
|-------|-------------------|
| 48×48 | 80.8% ± 0.3% |
| 96×96 | 85.9% ± 0.2% |
| 128×128 | 86.1% ± 0.3% |
| 192×192 | 87.5% ± 0.2% |
| 256×256 | OOM (T4 15GB) |

**O que precisa mudar**:
- Campos 384-683 não foram reproduzidos (OOM). Remover ou marcar como "run única sem barras de erro"
- Audit 2 confirma com 10 seeds até 192×192
- Adicionar que com cristalização competitiva os números podem mudar (não re-testado)

**Dado novo** (Audit 11): Scaling NÃO funciona pra protótipos:
| Campo | Protótipos (zero treino) |
|-------|------------------------|
| 48×48 | 76.4% |
| 96×96 | 77.2% |

Diferença insignificante. O scaling beneficia o decoder, não os protótipos.

---

### 4.4 Densidade de Informação: βc = 2.32

**Status**: ERRADO — Remover completamente.

**O que substituir** (Audit 1):
- MI ≈ 4.75 bits via KMeans clustering (n_clusters = [8, 16, 32, 64, 128])
- MI é constante entre diferentes (γ, β) — propriedade intrínseca
- A capacidade informacional do campo não depende dos parâmetros físicos

**O que substituir** (Audit 6 — densidade):
| Representação | MI (bits) | Mem (KB) | Densidade (bits/KB) |
|---------------|-----------|----------|---------------------|
| Cristais competitivos | 2.49 | ~4 KB | 0.63 |
| MLP random | 1.12 | 0.1 KB | 11.2 |
| MLP treinado | 3.8+ | 0.1 KB | 38+ |

**Conclusão honesta**: Cristais NÃO são mais densos que MLPs em bits/KB. O valor da cristalização não é densidade — é que funciona sem treino.

---

### 4.5 Lei de Escala de Cristais

**Status**: PARCIALMENTE CORRETO — O crescimento super-linear existe, mas os coeficientes foram medidos com cristalização original. Precisa re-medir com competitiva.

**Correção**: Manter a forma C(N) = a × Nᵇ mas notar que a e b precisam ser re-estimados.

---

### 4.6 Capacidade para Tarefas de Linguagem

**Status**: ERRADO — As extrapolações de campo mínimo pra tokens/vocabulário são baseadas em βc = 2.32 que é tautológico, e a completação de frases falhou.

**Correção**: Remover a tabela de "campo mínimo para N tokens". Substituir por:
- "O campo processa dados com estrutura geométrica. Para texto, a entrada precisa de embedding semântico externo."
- Adicionar resultados do Audit 12: hash = 20% (aleatório), embedding semântico = 70% (multiple choice, 5 classes)

---

### 4.7 Completação de Frases

**Status**: ERRADO — Remover ou reescrever completamente.

**O que estava errado**:
- Score = 50% overlap léxico + 50% overlap físico
- O 100% de acerto vinha da parte léxica (match de strings)
- Física pura = 31% (Audit 3)

**O que colocar no lugar** (Audit 12):
| Condição | Acurácia | Referência |
|----------|----------|------------|
| Hash (sem semântica na entrada) | 20.0% | Aleatório = 20% |
| Embedding semântico (protótipos) | 47.5% | 2.4× aleatório |
| Embedding semântico (multiple choice, max energy) | 70.0% | 3.5× aleatório |

**Conclusão correta**: "A ResNet-Ψ amplifica estrutura que já existe na entrada. Para imagens, essa estrutura é natural (pixels). Para texto, precisa de embedding externo. O mecanismo não inventa relações semânticas — preserva as que recebe."

---

## 5. DISCUSSÃO (Páginas 8-9)

### 5.1 O Campo como Kernel Físico
**Status**: CORRETO — Reforçado pelos audits. Os protótipos (Audit 8b) são a prova mais forte: o campo transforma imagens em representações que separam classes sem nenhum treino.

**Adição**: Citar o resultado de 77.4% zero treino como evidência principal. "O campo transforma imagens de forma que a simples média por classe já discrimina — demonstrando que a transformação física, por si só, organiza informação por classe."

### 5.2 Relação com Trabalho Existente

**Status**: INCOMPLETO — Precisa adicionar comparação com reservoir computing.

**Adição** (baseado na pesquisa web realizada):

> "Em Reservoir Computing clássico (Jaeger 2001, Maass 2002), o reservatório é fixo e apenas o readout é treinado. Physical Reservoir Computing (Hughes et al., 2019; Marcucci et al., 2020) usa dinâmica ondulatória como reservatório, mas sempre treina o readout."
>
> "A ResNet-Ψ se distingue em dois aspectos:
> 1. **Cristalização competitiva como mecanismo de memória** — cristais emergem, competem e morrem baseado em ressonância. Nenhum paper de reservoir computing propõe cristalização esparsa em campo de ondas.
> 2. **Classificação sem treinar o readout** — 77.4% no MNIST e 67% no Fashion-MNIST sem treinar nenhum parâmetro em lugar nenhum. Todo reservoir computing existente treina pelo menos o readout."

Papers relevantes:
- Hughes et al., "Wave physics as an analog recurrent neural network", Science Advances (2019)
- Marcucci et al., PRL (2020) — wave computing
- Tong & Tanaka, "Reservoir Computing with Untrained CNNs" (2018) — untrained reservoir + trained readout = 98.4% MNIST

### 5.3 Complexidade Computacional
**Status**: CORRETO — A análise de complexidade não mudou.

### 5.4 Limitações

**Status**: INCOMPLETO — Precisa adicionar limites descobertos nos audits.

**Adicionar**:

1. **CIFAR-10 falha (18.7%)** (Audit 11): "O mecanismo não funciona para imagens naturais complexas. A projeção gaussiana + dinâmica ondulatória não preserva informação discriminativa de textura e cor. Protótipos CIFAR ficam todos idênticos. O domínio do mecanismo é geometria (contornos, formas, silhuetas), não imagens naturais."

2. **Cristalização original piora acurácia** (Audit 4): "A cristalização com thresholds duros piora a acurácia em ~10% comparado ao campo bruto. A cristalização competitiva (v2) recupera parcialmente, mas campo bruto com decoder ainda supera cristais com decoder (91.7% vs 88.1%). O valor dos cristais é no modo zero treino (protótipos), onde são indispensáveis."

3. **Gargalo é a representação** (Audit 10): "Cinco métodos de leitura testados (Euclidiana, Coseno, Variância ponderada, Fisher, Mahalanobis diagonal) — nenhum melhora sobre Euclidiana simples. O limite de 77% sem decoder vem da qualidade da representação cristalina, não do método de comparação."

4. **Campo maior não ajuda protótipos** (Audit 11): "96×96 dá 77.2% vs 76.4% no 48×48. O benefício de campos maiores é capturado pelo decoder treinado, não pela métrica de distância dos protótipos."

5. **Texto precisa de embedding externo** (Audit 12): "Hash determinístico = 20% (aleatório). Embedding semântico manual = 47-70%. A ResNet-Ψ não cria relações semânticas — amplifica as que já existem na entrada."

6. **Densidade inferior a MLPs** (Audit 6): "Cristais: 0.63 bits/KB. MLP treinado: 38+ bits/KB. Cristais não são representação mais densa. O valor é que funcionam sem treino."

---

## 6. CONCLUSÃO (Página 9)

### Claims atuais vs status:

| Claim | Status |
|-------|--------|
| "(1) forma unidades de memória estáveis (cristais) de forma robusta e determinística" | **CORRETO** |
| "(2) opera como hash físico — representação única e consistente por padrão de entrada" | **CORRETO** |
| "(3) não requer módulo de encoding treinável" | **CORRETO** |
| "(4) possui capacidade de informação descrita por constante universal βc = 2.32 bits/cristal e lei de escala C(N) = 0.0024 × N2.171" | **βc ERRADO, lei de escala parcialmente correta** |
| "(5) habilita completação de frases por física pura, sem embeddings ou atenção" | **ERRADO** |

### Conclusão reescrita sugerida:

> "Apresentamos Resonance Networks (ResNet-Ψ), uma arquitetura neural na qual a computação emerge de dinâmica de ondas e cristalização competitiva. O Ψ-Field foi validado como substrato computacional que:
>
> (1) Forma unidades de memória estáveis (cristais) de forma robusta e determinística;
>
> (2) Opera como hash físico — representação única e consistente por padrão de entrada;
>
> (3) Não requer módulo de encoding treinável — a injeção direta supera emitters com 43× mais parâmetros;
>
> (4) **Classifica MNIST com 77.4% e Fashion-MNIST com 67.0% sem nenhum parâmetro treinado em lugar nenhum** — via protótipos cristalinos (média por classe + distância euclidiana). Nenhum sistema de reservoir computing existente classifica sem treinar pelo menos o readout;
>
> (5) Com decoder simples (treinado apenas no readout), atinge 88.1% (linear) e 93.1% (MLP) no MNIST;
>
> (6) A cristalização competitiva — onde cristais ganham vida por ressonância e morrem por decay — melhora todas as métricas: +7.2% acurácia linear, +4.7% MLP, +97% MI vs cristalização original;
>
> (7) O mecanismo funciona para dados com estrutura geométrica (contornos, silhuetas). Não funciona para imagens naturais complexas (CIFAR-10: 18.7%) nem para texto sem embedding semântico externo.
>
> A implicação central é que **informação pode emergir de dinâmica física sem otimização** — cristais são os pesos que o campo encontra sozinho. A ResNet-Ψ é um amplificador de estrutura: preserva e cristaliza relações que existem na entrada."

---

## 7. SEÇÕES NOVAS A ADICIONAR

### 7.1 Cristalização Competitiva (nova Seção 2.4)
Descrição completa do mecanismo: sigmoid, HP, decay, ressonância, morte.
Código de referência: `resnet_psi.py` → classe `CrystalCompetitivo`

### 7.2 Classificação por Protótipos (nova Seção 4.x)
- Pipeline: imagem → crystal_map → média por classe → protótipo
- Classificação: distância euclidiana ao protótipo mais próximo
- Resultados: 77.4% MNIST, 67% Fashion-MNIST
- Curva: 5 exemplos → 53%, 50 → 71%, 500 → 75.6%, 5000 → 77.4%
- Zero treino, zero backprop, zero gradiente, zero decoder
- Código de referência: `resnet_psi.py` → `ResNetPsi.fit()` / `ResNetPsi.predict()`

### 7.3 Domínio de Aplicação (nova Seção 5.x)
| Dataset | Tipo | Resultado | Funciona? |
|---------|------|-----------|-----------|
| MNIST | Formas simples, alto contraste | 77.4% (zero treino) / 93.1% (MLP) | Sim |
| Fashion-MNIST | Silhuetas de roupas | 67.0% (zero treino) | Sim |
| CIFAR-10 | Imagens naturais, cor, textura | 18.7% | Não |
| Texto (hash) | Sem estrutura semântica | 20.0% | Não |
| Texto (embedding) | Com estrutura semântica | 47-70% | Parcial |

### 7.4 Comparação com Reservoir Computing (nova Seção 5.x)
- Reservoir computing: reservatório fixo + readout treinado
- ResNet-Ψ: reservatório fixo + readout **opcional** (protótipos = zero treino)
- Cristalização competitiva como mecanismo novo (não existe em RC)
- Referências: Hughes 2019, Marcucci 2020, Jaeger 2001

---

## 8. TABELA RESUMO DE TODAS AS CORREÇÕES

| Seção | Ação | Prioridade |
|-------|------|------------|
| Resumo | Atualizar acurácias, remover βc e frases | ALTA |
| 2.4 Cristais | Reescrever com cristalização competitiva | ALTA |
| 2.5 Freq map | Notar que não é usado na v2 | BAIXA |
| 2.6 Re-emissão | Notar simplificação na implementação | MÉDIA |
| 4.2 Ablação | Atualizar números (88.1%/93.1%) | ALTA |
| 4.3 Scaling | Limitar a 48-192 com barras de erro | MÉDIA |
| 4.4 βc | REMOVER, substituir por MI ≈ 4.75 bits | ALTA |
| 4.5 Lei escala | Notar que precisa re-medir com v2 | MÉDIA |
| 4.6 Linguagem | REMOVER extrapolações de campo pra tokens | ALTA |
| 4.7 Frases | REESCREVER com resultados do Audit 12 | ALTA |
| 5.2 Trabalho existente | Adicionar comparação com RC | ALTA |
| 5.4 Limitações | Adicionar CIFAR, densidade, campo maior | ALTA |
| 6 Conclusão | Reescrever com claims corretos | ALTA |
| NOVA 2.4 | Cristalização competitiva | ALTA |
| NOVA 4.x | Protótipos zero treino (77.4%/67%) | ALTA |
| NOVA 5.x | Domínio de aplicação | ALTA |
| NOVA 5.x | Comparação com RC | ALTA |

---

## 9. ARQUIVOS DE REFERÊNCIA

Cada resultado citado nesta revisão pode ser reproduzido com os seguintes arquivos:

| Resultado | Arquivo |
|-----------|---------|
| MI ≈ 4.75 bits | `RN_psi_audit_1_beta_c.py` |
| Scaling 48-192 | `RN_psi_audit_2_scaling.py` |
| Frases 31% | `RN_psi_audit_3_frases.py` |
| Cristais pioram 10% | `RN_psi_audit_5_ablacao_baselines.py` |
| Baselines MNIST | `RN_psi_audit_5_baselines.py` |
| Densidade 0.63 bits/KB | `RN_psi_audit_6_densidade.py` |
| Competitivo 88.1%/93.1% | `RN_psi_audit_7_cristalizacao_v2.py` |
| Protótipos 77.4% | `RN_psi_audit_8b_prototipos_v2.py` |
| Fashion 67% | `RN_psi_audit_9_fashion.py` |
| 5 métodos leitura | `RN_psi_audit_10_leitura.py` |
| CIFAR 18.7% / campo 96 | `RN_psi_audit_11_campo96_cifar.py` |
| Texto MC 47-70% | `RN_psi_audit_12_frases_mc.py` |
| Biblioteca base | `resnet_psi.py` |
