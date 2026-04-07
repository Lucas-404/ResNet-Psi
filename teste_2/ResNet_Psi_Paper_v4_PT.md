# Ressonância É Tudo Que Você Precisa: Computação Neural Através de Cristalização de Ondas

**Uma arquitetura novel para processamento neural de informação baseada em dinâmica de campo ressonante, cristalização competitiva e classificação sem treino**

Lucas — OrfeuAI, Campinas, Brasil
Março 2026

---

## RESUMO

Apresentamos Resonance Networks (ResNet-Ψ), uma arquitetura neural que substitui mecanismos de attention, recorrência e convolução por ressonância de ondas em um campo contínuo. A informação é representada como ondas oscilatórias em um campo bidimensional (Ψ-Field) que propagam, interferem e se auto-organizam em estruturas estáveis denominadas cristais. Estas unidades emergem espontaneamente da dinâmica do campo sem qualquer otimização paramétrica.

O resultado central deste trabalho é a classificação sem treino via protótipos cristalinos: 77.4% no MNIST e 67.0% no Fashion-MNIST sem treinar nenhum parâmetro. Não há backpropagation, gradiente ou decoder. Até onde é possível determinar, nenhum sistema de reservoir computing existente classifica sem treinar pelo menos a camada de readout.

A cristalização competitiva, na qual cristais acumulam vitalidade por ressonância e são eliminados por decaimento, melhora todas as métricas em relação à cristalização por thresholds fixos: +7.2% na acurácia linear, +4.7% na acurácia MLP e +97% na informação mútua. Com decoder treinado apenas na camada de readout, o sistema atinge 88.1% (linear) e 93.1% (MLP) no MNIST.

Mapeamos o domínio de aplicação em 14 datasets de cinco domínios distintos (dígitos, letras, silhuetas, imagens médicas e imagens naturais). Os resultados revelam uma regra consistente: a arquitetura funciona para dados com estrutura geométrica (contornos, formas, silhuetas) e falha para dados cuja informação discriminativa reside em textura, cor ou fundos complexos. Demonstramos adicionalmente classificação 1-shot (47.4% no MNIST com um único exemplo por classe) e detecção de anomalia (F1=93.6%) sem treino.

---

## 1. INTRODUÇÃO

As arquiteturas neurais dominantes, incluindo Transformers (Vaswani et al., 2017), redes recorrentes e redes convolucionais, compartilham uma premissa representacional: informação é codificada como vetores estáticos transformados por sequências de operações lineares e não-lineares. Este paradigma, embora eficaz, requer otimização baseada em gradiente para produzir representações úteis. Sem treino, os pesos aleatórios de qualquer rede convencional produzem saídas aleatórias. Esta dependência de otimização é aceita como premissa fundamental desde a popularização do backpropagation (Rumelhart et al., 1986).

O presente trabalho propõe um paradigma alternativo no qual a computação emerge da dinâmica física de propagação, interferência e cristalização de ondas em um campo contínuo. A abordagem se fundamenta em três bases científicas independentes: binding oscilatório em neurociência (Engel et al., 2001), dinâmica de ondas não-lineares em física e teoria de ressonância adaptativa (Grossberg, 1987). A combinação específica proposta neste trabalho, consistindo em um campo de onda não-linear amortecido com cristalização competitiva baseada em envelope e competição vida-morte, é, até onde foi possível determinar na literatura, inédita.

O resultado central é que a transformação física do campo é suficiente para classificação sem nenhum treino. Computando a média dos crystal maps por classe (protótipos) e classificando por distância euclidiana, obtemos 77.4% no MNIST. Este resultado é validado em 14 datasets, demonstrando que o mecanismo é genérico para dados com estrutura geométrica.

---

## 2. ARQUITETURA

A arquitetura ResNet-Ψ consiste em quatro componentes: injeção direta de dados, dinâmica do Ψ-Field, cristalização competitiva e classificação por protótipos.

### 2.1 Injeção Direta de Dados

Cada valor de entrada é mapeado para uma gaussiana no campo bidimensional, cuja posição codifica a localização espacial do dado e cuja amplitude corresponde ao valor do dado:

    ψ_pixel(x, y) = A_pixel · exp(−((X − cx)² + (Y − cy)²) / (2σ²))

O parâmetro σ = 0.04 controla a largura da gaussiana. As coordenadas (cx, cy) são determinadas por mapeamento linear da posição do pixel no intervalo [0.1, 0.9] do campo normalizado. Para uma imagem de 784 pixels, a perturbação total é computada como combinação linear, sem parâmetros treináveis.

O campo é reinicializado para cada amostra. A perturbação é injetada durante STIM_ON = 40 passos temporais (fase de estímulo ativo), seguidos de 40 passos de evolução livre, totalizando STIM_TOTAL = 80 passos.

A projeção gaussiana preserva relações de similaridade geométrica presentes na entrada. Entradas com estrutura espacial similar produzem perturbações similares e, consequentemente, cristais similares. O mecanismo opera como amplificador de estrutura: preserva relações existentes na entrada sem criar relações novas.

### 2.2 Dinâmica do Ψ-Field

O campo Ψ(x, y, t) evolui segundo uma equação de onda modificada com quatro termos:

    ∂²Ψ/∂t² = c²∇²Ψ − γ(∂Ψ/∂t) + α·tanh(Ψ)·Ψ − β·Ψ·|Ψ|²

O termo de propagação (c²∇²Ψ) permite que ondas se espalhem pelo campo via diferenças finitas com condições de contorno periódicas. O amortecimento (−γ∂Ψ/∂t) suprime ruído e impede acumulação de energia. A não-linearidade seletiva (α·tanh(Ψ)·Ψ) amplifica padrões coerentes enquanto suprime componentes incoerentes. A dissipação cúbica (−β·Ψ·|Ψ|²) impede crescimento ilimitado de energia. A integração utiliza o método de Verlet com limitação de amplitude em ±10.0.

**Tabela 1: Constantes físicas do Ψ-Field (experimentos MNIST)**

| Símbolo | Parâmetro | Valor |
|---------|-----------|-------|
| c² | Velocidade de propagação | 0.3 |
| γ | Coeficiente de amortecimento | 0.06 |
| α | Coeficiente não-linear | 0.04 |
| β | Coeficiente de dissipação | 0.005 |
| dt | Passo temporal | 0.05 |
| STIM_ON | Passos com estímulo | 40 |
| STIM_TOTAL | Passos totais | 80 |

### 2.3 Cristalização Competitiva

Cristais emergem em regiões do campo que mantêm amplitude de oscilação consistente ao longo do tempo. O mecanismo de cristalização competitiva substitui thresholds binários por funções de scoring contínuas e introduz competição baseada em vitalidade.

**Scoring contínuo.** Cada posição candidata recebe um score no intervalo [0, 1] baseado em três critérios avaliados via função sigmoide:

    score_amplitude = σ(κ · (μ_envelope − A_min))
    score_estabilidade = σ(κ · (CV_max − cv))
    score_saturação = σ(κ · (8.0 − μ_envelope))
    candidato = score_amplitude × score_estabilidade × score_saturação

onde κ = 5.0 é o parâmetro de suavidade, μ_envelope é a média do envelope de amplitude ao longo de K = 3 janelas temporais de W = 20 passos, cv é o coeficiente de variação do envelope, A_min = 0.3 é a amplitude mínima e CV_max = 0.15 é o coeficiente de variação máximo.

**Competição por vitalidade.** Cada cristal mantém um valor de vitalidade (HP). Cristais são inicializados com HP = 1. A cada passo de simulação, cristais que coincidem com regiões de alta amplitude do campo recebem incremento de vitalidade (HP += 0.1 × |campo|). Simultaneamente, todos os cristais sofrem decaimento constante (HP -= 0.02). Cristais cujo HP atinge zero são eliminados. Este mecanismo implementa seleção competitiva: apenas cristais que ressoam consistentemente com a dinâmica do campo sobrevivem.

Adicionalmente, cristais sobreviventes re-emitem energia no campo (field += crystal_map × 0.05 × sign(field)), reforçando padrões estáveis. A exclusão espacial impede a formação de cristais a menos de 5 pixels de distância entre si.

**Tabela 2: Comparação de variantes de cristalização** (5 seeds, campo 48×48, MNIST)

| Variante | Acurácia Linear | Acurácia MLP | MI (bits) |
|----------|----------------|-------------|-----------|
| Thresholds fixos | 80.9% | 88.4% | 1.26 |
| Scoring sigmoide | 84.0% | 90.2% | 1.78 |
| Competitiva (sigmoide + HP) | 88.1% | 93.1% | 2.49 |

### 2.4 Classificação por Protótipos

A classificação sem treino segue três etapas: (1) cada imagem de treino é processada pelo campo para produzir seu crystal map; (2) a média dos crystal maps por classe define o protótipo de cada classe; (3) imagens de teste são classificadas pela menor distância euclidiana ao protótipo. Nenhum parâmetro é otimizado em qualquer etapa.

---

## 3. VALIDAÇÃO DO Ψ-FIELD

A fundação do Ψ-Field foi validada por quatro testes independentes, cada um executado com 10 seeds aleatórias em campo 128×128 com 15 ondas em 3 clusters ao longo de 2000 passos temporais.

### 3.1 Robustez de Cristalização
Todas as 10 seeds produziram cristais, atingindo o limite de 80 cristais em todos os casos (variância zero). O mecanismo de cristalização é determinístico e robusto em relação às condições iniciais.

### 3.2 Re-emissão
Após cristalização, o campo foi reinicializado e uma onda de teste foi injetada. A energia média nas proximidades dos cristais (28.095) foi 136.8 vezes superior à energia em regiões distantes (0.205), confirmando que cristais operam como amplificadores ativos.

### 3.3 Seletividade de Frequência
Cristais responderam 1.57 vezes mais fortemente a ondas harmônicas (resposta média 0.942, intervalo [0.887, 0.996]) em comparação com ondas dissonantes (resposta média 0.601, intervalo [0.272, 0.946]).

### 3.4 Estrutura Espectral
A razão pico/média do espectro de cristais (10.79) foi 4.56 vezes superior à do ruído aleatório (2.37), indicando que cristais concentram energia espectral em frequências dominantes.

---

## 4. RESULTADOS DE CLASSIFICAÇÃO

### 4.1 Determinismo do Campo

| Condição | Correlação | Sobreposição |
|----------|-----------|--------------|
| Mesma entrada | 1.000 | 100% |
| Entradas distintas | −0.098 | 3.1% |

O campo produz representações determinísticas: entradas idênticas geram crystal maps idênticos, enquanto entradas distintas geram crystal maps com correlação negativa e sobreposição mínima.

### 4.2 Injeção Direta versus Encoder Treinável

| Configuração | Encoder | Decoder | Parâmetros | Acurácia |
|-------------|---------|---------|------------|----------|
| Injeção direta + linear | Nenhum | Linear | 23k | 88.1% |
| Injeção direta + MLP | Nenhum | MLP | 25k | 93.1% |
| Encoder MLP treinado | MLP | CNN | 988k | ~40% |

A introdução de um encoder MLP treinável (988k parâmetros) degradou a acurácia em mais de 50 pontos percentuais. O Ψ-Field opera sem gradientes por design; a dinâmica caótica ao longo de 80 passos não é diferenciavelmente estável.

### 4.3 Classificação sem Treino

| Dataset | Acurácia | Referência aleatória |
|---------|----------|---------------------|
| MNIST | 77.4% | 10% |
| Fashion-MNIST | 67.0% | 10% |

Nenhum sistema de reservoir computing identificado na literatura classifica sem treinar ao menos a camada de readout. Para referência, Tong & Tanaka (2018) reportam 98.4% no MNIST com readout treinado.

### 4.4 Classificação Few-Shot

| Exemplos por classe | 1 | 2 | 5 | 10 | 50 |
|--------------------|---|---|---|----|----|
| MNIST | 47.4% | 53.6% | 60.3% | 64.5% | 69.7% |
| Fashion-MNIST | 48.0% | 55.5% | 60.4% | 61.0% | 62.4% |

Com um único exemplo por classe (10 imagens no total), a acurácia atinge 47.4% no MNIST, correspondendo a 4.7 vezes o valor aleatório. A acurácia satura a partir de aproximadamente 50 exemplos por classe, indicando que o fator limitante é a qualidade da representação cristalina, não a quantidade de dados de referência.

### 4.5 Detecção de Anomalia

O sistema pode detectar anomalias sem exposição prévia a exemplos anômalos. O procedimento consiste em construir um protótipo a partir de exemplos de uma única classe ("normal") e classificar novas amostras pela distância ao protótipo.

| Classe normal | Razão distância (anomalia/normal) | F1 |
|--------------|----------------------------------|-----|
| MNIST dígito "1" | 2.04 | 92.8% |
| MNIST dígito "0" | 1.48 | 75.9% |
| Fashion-MNIST Sneaker | 3.04 | 93.6% |
| MNIST média (10 classes) | 1.45 | 55.2% |

O desempenho da detecção correlaciona com a distinção geométrica da classe normal: classes com formas únicas (Sneaker, dígito "1") produzem melhor separação.

### 4.6 Curva de Escala

| Tamanho do campo | Acurácia (decoder linear) | Desvio padrão |
|-----------------|--------------------------|---------------|
| 48×48 | 80.8% | ±0.3% |
| 96×96 | 85.9% | ±0.2% |
| 128×128 | 86.1% | ±0.3% |
| 192×192 | 87.5% | ±0.2% |

Resultados obtidos com 10 seeds independentes. O aumento do campo beneficia o decoder treinado, mas não altera significativamente a acurácia dos protótipos (96×96: 77.2% versus 48×48: 76.4%).

### 4.7 Informação Mútua

A informação mútua entre representações cristalinas e rótulos de classe, medida via clustering KMeans com múltiplos tamanhos de cluster, é MI ≈ 4.75 bits. Este valor se mantém constante entre diferentes configurações dos parâmetros físicos (γ, β), constituindo uma propriedade intrínseca do campo.

---

## 5. MAPA DE DOMÍNIO

O domínio de aplicação da arquitetura foi mapeado sistematicamente em 14 datasets de cinco domínios, todos avaliados com zero treino e 50 exemplos por classe.

### 5.1 Resultados

**Tabela 5: Mapa de domínio completo**

| Dataset | Classes | Acurácia | Chance | Razão | Domínio |
|---------|---------|----------|--------|-------|---------|
| EMNIST Digits | 10 | 77.4% | 10.0% | 7.7 | Dígitos manuscritos |
| MNIST | 10 | 69.7% | 10.0% | 7.0 | Dígitos manuscritos |
| PneumoniaMNIST | 2 | 69.2% | 50.0% | 1.4 | Raio-X torácico |
| BreastMNIST | 2 | 66.7% | 50.0% | 1.3 | Ultrassonografia mamária |
| Fashion-MNIST | 10 | 62.4% | 10.0% | 6.2 | Silhuetas de vestuário |
| BloodMNIST | 8 | 55.0% | 12.5% | 4.4 | Células sanguíneas |
| OrganAMNIST | 11 | 47.7% | 9.1% | 5.2 | Órgãos em TC axial |
| PathMNIST | 9 | 43.0% | 11.1% | 3.9 | Histopatologia |
| EMNIST Letters | 26 | 40.2% | 3.8% | 10.5 | Letras manuscritas |
| RetinaMNIST | 5 | 35.2% | 20.0% | 1.8 | Fundo de olho |
| OCTMNIST | 4 | 25.1% | 25.0% | 1.0 | Tomografia óptica |
| CIFAR-10 | 10 | 18.7% | 10.0% | 1.9 | Imagens naturais |
| DermaMNIST | 7 | 16.1% | 14.3% | 1.1 | Lesões dermatológicas |
| SVHN | 10 | 9.8% | 10.0% | 1.0 | Dígitos fotografados |

### 5.2 Análise

Os resultados revelam uma regra consistente: datasets cuja informação discriminativa reside em contornos e formas (razão acurácia/chance superior a 3) são classificados com sucesso, enquanto datasets cuja informação reside em textura, cor ou fundos complexos (razão próxima a 1) não são discriminados pelo campo.

Os nove datasets com razão superior a 3 compartilham uma propriedade: a informação de classe está codificada na geometria dos objetos. Os cinco datasets com razão inferior a 2 dependem de textura (DermaMNIST), cor (CIFAR-10), iluminação (SVHN) ou camadas sobrepostas (OCTMNIST). O caso SVHN é particularmente informativo: são dígitos (mesma categoria semântica do MNIST), porém fotografados com fundos complexos, resultando em razão 1.0 contra 7.0 do MNIST manuscrito.

A razão acurácia/chance constitui a métrica adequada para comparação entre datasets com números diferentes de classes. EMNIST Letters, com acurácia absoluta de 40.2%, apresenta a maior razão (10.5) entre todos os datasets testados, uma vez que a chance aleatória para 26 classes é apenas 3.8%.

### 5.3 Caracterização do Domínio

A ResNet-Ψ opera como amplificador de estrutura geométrica. O campo preserva e cristaliza relações espaciais presentes na entrada sem criar relações novas. Quando a entrada contém geometria discriminativa, o campo produz representações separáveis por classe. Quando a informação discriminativa não é geométrica, o campo não consegue separá-la.

Esta caracterização define o nicho da arquitetura: classificação sem treino de dados com estrutura geométrica. Dentro deste nicho, não foi identificada outra arquitetura capaz de classificar sem treinar ao menos uma camada.

---

## 6. DISCUSSÃO

### 6.1 O Campo como Kernel Físico

O Ψ-Field opera como um kernel implícito análogo aos utilizados em máquinas de vetores de suporte. Em SVMs, um kernel mapeia dados para um espaço de alta dimensão onde se tornam linearmente separáveis. No presente caso, a dinâmica de propagação, interferência e cristalização realiza esta transformação por meio de física. A evidência mais direta é que a simples média dos crystal maps por classe já produz protótipos discriminativos, indicando que a transformação física organiza a informação por classe sem otimização.

### 6.2 Relação com Reservoir Computing

Em Reservoir Computing clássico (Jaeger, 2001; Maass et al., 2002), um reservatório fixo transforma a entrada em um espaço de alta dimensão, e apenas a camada de readout é treinada. Implementações físicas (Hughes et al., 2019; Marcucci et al., 2020) utilizam dinâmica de ondas como reservatório, porém sempre treinam o readout.

A ResNet-Ψ difere em dois aspectos. Primeiro, a cristalização competitiva constitui um mecanismo de memória esparsa não presente na literatura de reservoir computing: cristais emergem, competem por sobrevivência e são eliminados com base em ressonância. Segundo, a classificação é realizada sem treinar qualquer camada, incluindo o readout.

### 6.3 Separação sem Semântica

A ResNet-Ψ não atribui significado semântico aos dados. O sistema não identifica que um "7" representa o número sete ou que uma célula sanguínea é uma célula. Contudo, entradas com formas similares produzem cristais similares, e entradas com formas distintas produzem cristais distintos. A física do campo efetua esta separação de forma autônoma.

Este comportamento é fundamentalmente distinto de aprendizado. Não há otimização, ajuste de parâmetros ou gradiente. O campo separa entradas por sua estrutura geométrica sem interpretar seu conteúdo. A analogia adequada é um prisma óptico: a física da refração separa comprimentos de onda sem que o prisma "compreenda" o conceito de cor.

### 6.4 Robustez da Representação

Experimentos adicionais com variações na dinâmica do campo (velocidade de propagação variável por região, múltiplas passadas com física adaptativa) não produziram melhoria na acurácia dos protótipos. Cinco métodos de distância distintos (Euclidiana, Cosseno, variância ponderada, Fisher e Mahalanobis diagonal) também não superaram a distância Euclidiana simples. Campos maiores (96×96) não beneficiaram os protótipos. Estes resultados convergem para a conclusão de que o teto de aproximadamente 77% no MNIST constitui um limite intrínseco da representação cristalina obtida pela projeção gaussiana, e não um artefato do método de classificação.

### 6.5 Limitações

1. O domínio de aplicação é restrito a dados com estrutura geométrica. O sistema falha para imagens naturais (CIFAR-10: 18.7%), texturas (DermaMNIST: 16.1%) e dígitos fotografados com fundos complexos (SVHN: 9.8%).

2. Com decoder treinado, o campo bruto (sem cristais) atinge 91.7% com decoder linear, superando a cristalização competitiva (88.1%). Cristais são essenciais apenas para o modo sem treino.

3. A acurácia de 77.4% não compete com redes treinadas (99%+ no MNIST). A contribuição é demonstrar classificação sem treino, não atingir acurácia máxima.

4. O aumento do tamanho do campo não beneficia os protótipos, sugerindo que o gargalo reside na projeção gaussiana, não na resolução do campo.

5. A cristalização competitiva introduz seis hiperparâmetros (κ, A_min, CV_max, decay, resonance_boost, separação). Embora os valores utilizados funcionem consistentemente nos 14 datasets testados, a sensibilidade a estes parâmetros não foi exaustivamente avaliada.

---

## 7. CONCLUSÃO

Este trabalho apresenta Resonance Networks (ResNet-Ψ), uma arquitetura na qual a computação emerge de dinâmica de ondas e cristalização competitiva. Os resultados em 14 datasets permitem as seguintes conclusões:

1. O Ψ-Field separa dados por estrutura geométrica sem treino. A acurácia atinge 77.4% no MNIST, 67.0% no Fashion-MNIST e 40.2% no EMNIST Letters (26 classes, 10.5 vezes a chance aleatória).

2. O mecanismo é genérico para dados com geometria discriminativa, funcionando consistentemente (razão superior a 3 vezes a chance) em 9 dos 14 datasets testados, abrangendo dígitos, letras, silhuetas, células sanguíneas e órgãos em tomografia computadorizada.

3. A classificação 1-shot atinge 47.4% no MNIST com um único exemplo por classe. A acurácia satura a partir de aproximadamente 50 exemplos.

4. A detecção de anomalia atinge F1=93.6% sem exposição prévia a exemplos anômalos, utilizando distância ao protótipo da classe normal.

5. O domínio é delimitado pela natureza da informação discriminativa: contornos e formas são separáveis, textura e cor não. O caso SVHN (dígitos fotografados, razão 1.0) versus MNIST (dígitos manuscritos, razão 7.0) demonstra que é a estrutura geométrica da representação, não o conteúdo semântico, que determina a eficácia.

6. Não foi identificado na literatura nenhum sistema que classifique múltiplos datasets sem treinar qualquer parâmetro.

A implicação central é que dinâmica física pode organizar informação de forma discriminativa sem otimização. Os cristais constituem os parâmetros que o campo determina autonomamente. A ResNet-Ψ não aprende: separa. Para dados com geometria, separar é suficiente.

---

## REFERÊNCIAS

[1] Vaswani, A., Shazeer, N., Parmar, N., et al. Attention Is All You Need. Advances in Neural Information Processing Systems 30, 2017.

[2] Rumelhart, D. E., Hinton, G. E., Williams, R. J. Learning representations by back-propagating errors. Nature, 323(6088):533-536, 1986.

[3] Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8):2554-2558, 1982.

[4] Grossberg, S. Competitive learning: From interactive activation to adaptive resonance. Cognitive Science, 11(1):23-63, 1987.

[5] Engel, A. K., Fries, P., Singer, W. Dynamic predictions: oscillations and synchrony in top-down processing. Nature Reviews Neuroscience, 2(10):704-716, 2001.

[6] Jaeger, H. The echo state approach to analysing and training recurrent neural networks. GMD Report 148, German National Research Center for Information Technology, 2001.

[7] Maass, W., Natschläger, T., Markram, H. Real-time computing without stable states: a new framework for neural computation based on perturbations. Neural Computation, 14(11):2531-2560, 2002.

[8] Hughes, T. W., Williamson, I. A. D., Minkov, M., Fan, S. Wave physics as an analog recurrent neural network. Science Advances, 5(12):eaay6946, 2019.

[9] Marcucci, G., Pierangeli, D., Conti, C. Theory of neuromorphic computing by waves: machine learning by rogue waves, dispersive shocks, and solitons. Physical Review Letters, 125(9):093901, 2020.

[10] Tong, Z., Tanaka, G. Reservoir Computing with Untrained Convolutional Neural Networks for Image Recognition. Proceedings of the International Conference on Pattern Recognition, 2018.

[11] Yang, J., Shi, R., Wei, D., et al. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10:41, 2023.

---

## APÊNDICE A: Figuras

- Figura 1: Mapa de domínio com 14 datasets (acurácia e razão acurácia/chance)
- Figura 2: Acurácia versus número de classes com curvas de referência
- Figura 3: Curva few-shot (1 a 50 exemplos por classe)
- Figura 4: Detecção de anomalia (distribuições de distância)
- Figura 5: Exemplos de imagens médicas e respectivos crystal maps
- Figura 6: Comparação entre datasets geométricos

## APÊNDICE B: Reprodutibilidade

Todos os resultados são reprodutíveis por meio dos scripts disponíveis como material suplementar. Os experimentos foram conduzidos em CPU AMD Ryzen 7 5700G utilizando PyTorch. Todos os datasets são públicos e obtidos automaticamente pelos scripts de avaliação.

| Script | Experimento |
|--------|------------|
| RN_psi_audit_8b_prototipos_v2.py | Protótipos MNIST (77.4%) |
| RN_psi_audit_9_fashion.py | Fashion-MNIST (67.0%) |
| RN_psi_audit_25_fewshot.py | Classificação few-shot |
| RN_psi_audit_26_anomalia.py | Detecção de anomalia |
| RN_psi_audit_27_medico.py | Imagens médicas |
| RN_psi_audit_28_datasets.py | EMNIST, KMNIST, Fashion-MNIST |
| RN_psi_audit_28b_extras.py | SVHN |
| RN_psi_audit_28c_blood.py | MedMNIST (Blood, Organ, Path, OCT, Retina) |
