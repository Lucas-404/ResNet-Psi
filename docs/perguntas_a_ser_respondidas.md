# Perguntas a Ser Respondidas — ResNet-Ψ

## Pergunta Central

**É possível criar representações úteis e classificar dados sem nenhum parâmetro treinado?**

A ResNet-Ψ demonstra que sim: dinâmica ondulatória + cristalização competitiva produz representações que classificam MNIST com 77.4% sem treino nenhum (nem encoder, nem decoder), e 93% com decoder simples.

---

## O que já foi provado (11 Audits)

| Audit | Resultado | Status |
|-------|-----------|--------|
| 1. β_c via MI | MI ≈ 4.75 bits, constante entre (γ,β). β_c original (2.32) era tautológico | Provado |
| 2. Scaling | Acurácia escala com campo: 80.8% (48²) → 87.5% (192²). Satura ~91% com MLP | Provado |
| 3. Frases | Física sozinha = 31%. Completação por frases é inviável | Provado (negativo) |
| 4. Ablação cristais | Cristais originais pioram ~10% vs campo bruto | Provado |
| 5. Baselines | ResNet-Ψ não compete em acurácia com CNN/MLP convencionais | Provado |
| 6. Densidade | Cristais: 0.63 bits/KB. MLP random: 11.2 bits/KB. Cristais NÃO são mais densos | Provado (negativo) |
| 7. Cristalização v2 | Competitivo: 88.1% linear (+7.2%), 93.1% MLP (+4.7%), MI 2.49 bits (+97%) | **Melhoria real** |
| 8. Protótipos | 77.4% sem decoder nenhum (5000 exemplos por protótipo). Zero treino total | **Resultado chave** |
| 9. Fashion-MNIST | 67.0% com protótipos cristalinos (zero treino). Mecanismo é genérico | **Generalização** |
| 10. Métodos de leitura | Euclidiana, Coseno, Variância ponderada, Fisher, Mahalanobis — nenhum melhora. Gargalo é representação | Provado |
| 11. Campo 96² + CIFAR | MNIST 96²: 77.2% (≈48²). CIFAR-10: 18.7% (falhou). Física não captura imagens naturais | **Limite encontrado** |

---

## Resultados chave consolidados

### O que funciona

1. **77.4% no MNIST sem nenhum parâmetro treinado** (Audit 8b)
   - Protótipos = média dos crystal_maps por classe
   - Classificação = distância euclidiana ao protótipo
   - Zero backprop, zero decoder, zero gradiente
   - Curva: 5 exemplos → 53%, 500 → 75.6%, 5000 → 77.4% (satura)

2. **93% com cristalização competitiva + MLP decoder** (Audit 7)
   - Cristais que não ressoam morrem (seleção natural)
   - +7% linear, +5% MLP, +97% MI vs cristalização original
   - A competição é o mecanismo que faltava

3. **MI constante ≈ 4.75 bits entre configurações físicas** (Audit 1)
   - Capacidade informacional é propriedade intrínseca do sistema
   - Não depende de γ ou β

4. **Scaling funciona** (Audit 2)
   - Campo maior → mais cristais → mais acurácia
   - 48² (80.8%) → 192² (87.5%) com linear
   - 10 seeds, barras de erro em todos os pontos

5. **Cristalização é mecanismo novo** (pesquisa confirmou)
   - Ninguém propôs cristalização de memória esparsa em campo de ondas
   - Marcucci et al. (PRL 2020) cobrem wave computing, mas sem cristalização

6. **67% no Fashion-MNIST sem nenhum parâmetro treinado** (Audit 9)
   - Mesmo mecanismo do MNIST, agora com roupas
   - Protótipos diferenciais mostram silhuetas reconhecíveis (pernas de calça, formato de bolsa)
   - Prova que o mecanismo é genérico, não específico pra dígitos

7. **Gargalo é a representação, não a leitura** (Audit 10)
   - 5 métodos testados: Euclidiana, Coseno, Variância ponderada, Fisher, Mahalanobis diagonal
   - Nenhum melhora sobre Euclidiana simples
   - Conclusão: a qualidade da crystal_map é o limite, não como você lê

### O que não funciona

1. **Completação de frases** — morto (31% física pura, artefato lexical)
2. **β_c = 2.32** — era tautológico (log₂(1/α))
3. **Densidade informacional superior** — MLPs são 10-35x mais densos em bits/KB
4. **Cristalização original** — piora acurácia em 10% vs campo bruto
5. **Competir em acurácia** — CNN 9k params = 98.5%, ResNet-Ψ 23k = 77-93%
6. **CIFAR-10** — 18.7% (quase aleatório). Física ondulatória não captura textura/cor de imagens naturais
7. **Campo maior pra protótipos** — 96² deu 77.2% vs 76.4% no 48². Sem ganho significativo (diferente do scaling com decoder)

---

## A narrativa do paper

### O que ResNet-Ψ NÃO é
- Não é competitor de Transformers/CNNs em acurácia
- Não é reservoir computing (reservatórios já existem)
- Não é sistema de NLP
- Não funciona pra imagens naturais complexas (CIFAR falhou)

### O que ResNet-Ψ É
- Um mecanismo onde **a física gera representações sem treino**
- **Os cristais são os pesos do modelo** — a informação tem assinatura física nos cristais
- 77% MNIST + 67% Fashion-MNIST sem nenhum parâmetro treinado em lugar nenhum
- A cristalização competitiva (v2) melhora tudo significativamente
- Os cristais funcionam como **protótipos de classe emergentes**
- Funciona para dados com **estrutura geométrica** (contornos, formas, silhuetas)

### Domínio de aplicação
O mecanismo captura **geometria** — contornos, formas, silhuetas. Quando a informação discriminativa está na forma (dígitos, roupas), funciona. Quando está em textura, cor, detalhes finos (imagens naturais), não funciona.

| Dataset | Tipo | Resultado | Funciona? |
|---------|------|-----------|-----------|
| MNIST | Formas simples, alto contraste | 77.4% | Sim |
| Fashion-MNIST | Silhuetas de roupas | 67.0% | Sim |
| CIFAR-10 | Imagens naturais, cor, textura | 18.7% | Não |

### A frase central
"Dinâmica ondulatória + cristalização competitiva produz representações que classificam dados com 77% de acurácia sem nenhum parâmetro treinado — a informação emerge da física. Os cristais são os pesos: cada classe deixa uma assinatura física distinta no campo."

---

## Perguntas abertas (futuro)

### 1. Cristalização diferenciável
- Substituir thresholds por sigmoid → gradiente flui → treino end-to-end
- Potencial: sistema que já começa com boa representação (77%) e refina com gradiente

### 2. Competição entre cristais como aprendizado
- Cristais que ressoam crescem, que não ressoam morrem
- Já funciona (Audit 7). Pode ser refinado com decay/boost adaptativos

### 3. Inferência puramente física
- Protótipos cristalinos já dão 77%
- Gap para decoder linear: 11%. Para MLP: 16%
- Possível melhorar com ponderação por importância dos cristais

### 4. Teto dos 77% sem decoder
- Saturou em 5000 exemplos
- Limitação: correlação coseno / distância euclidiana são métricas simples
- Possível melhorar com métrica de similaridade mais sofisticada (sem treino)
- Métodos de leitura alternativos NÃO ajudam (Audit 10) — gargalo é a representação

### 5. Campo maior não ajuda protótipos
- 96² deu 77.2% vs 76.4% no 48² (Audit 11)
- Com decoder, scaling funciona (48²→192²: +7%). Sem decoder, não
- O benefício do campo maior é capturado pelo decoder, não pela métrica de distância simples

### 6. Imagens naturais
- CIFAR-10: 18.7% — a projeção gaussiana + física ondulatória não preserva informação discriminativa de textura/cor
- Protótipos CIFAR ficam todos idênticos (blobs amarelos)
- Limite fundamental: o mecanismo é para geometria, não para imagens naturais

---

## Arquivos dos audits

| Arquivo | Descrição |
|---------|-----------|
| RN_psi_audit_1_beta_c.py | MI via clustering (β_c corrigido) |
| RN_psi_audit_2_scaling.py | Curva de escala 48-192, 10 seeds |
| RN_psi_audit_3_frases.py | Ablação de completação de frases |
| RN_psi_audit_5_ablacao_baselines.py | Ablação cristais + baselines MNIST |
| RN_psi_audit_6_densidade.py | Densidade informacional comparada |
| RN_psi_audit_7_cristalizacao_v2.py | Cristalização suave + competitiva |
| RN_psi_audit_8_prototipos.py | Protótipos v1 (campo acumulado — saturou) |
| RN_psi_audit_8b_prototipos_v2.py | Protótipos v2 (média + subtração — 77.4%) |
| RN_psi_audit_9_fashion.py | Fashion-MNIST protótipos (67.0%) |
| RN_psi_audit_10_leitura.py | 5 métodos de leitura (nenhum melhora) |
| RN_psi_audit_11_campo96_cifar.py | Campo 96² + CIFAR-10 (limite encontrado) |
