# ResNet-Psi — CLAUDE.md

## O que é este projeto

ResNet-Psi é uma rede neural baseada em física de ondas e cristalização competitiva. Um campo 2D (Ψ-Field) 48×48 recebe dados como perturbações ondulatórias. As ondas propagam, interferem, e regiões estáveis cristalizam — formando uma representação esparsa (crystal_map) que classifica dados **sem nenhum parâmetro treinado**.

**Arquivo principal**: `C:\ResNet-Psi\resnet_psi.py` — biblioteca base com API simples (fit/predict/score)

---

## Resultado central

**79.9% no MNIST, 67% no Fashion-MNIST, e 65-82% em falha industrial (CWRU) — tudo sem treinar absolutamente nada.**

Zero parâmetros. Zero gradiente. Zero backprop. Zero decoder. Só física + média por classe + distância euclidiana. Nenhum sistema de reservoir computing existente classifica sem treinar pelo menos o readout.

Com decoder treinado (só no readout): 88.1% linear, 93.1% MLP.

**Few-shot industrial (CWRU):** com K ≤ 5 amostras por classe, ResNet-Ψ zero treino supera CNN 1D arquitetura-específica treinada do zero. Crossover em K=10.

---

## Arquitetura atual (pós-audits)

### Pipeline
```
Imagem 28×28 → projeção gaussiana → campo de ondas 48×48 → cristalização competitiva → crystal_map → classificação
```

### Constantes físicas (MNIST)
```python
PSI_C2    = 0.3      # velocidade de onda (Laplaciano)
PSI_GAMMA = 0.06     # amortecimento (damping)
PSI_ALPHA = 0.04     # não-linearidade seletiva (tanh)
PSI_BETA  = 0.005    # dissipação cúbica
PSI_DT    = 0.05     # passo temporal
STIM_ON   = 40       # steps com estímulo ativo
STIM_TOTAL = 80      # steps totais
```

### Cristalização Competitiva (mecanismo que funciona)
```python
# Thresholds SUAVES (sigmoid, não step)
amp_score = sigmoid(sharpness × (mean - 0.3))
cv_score  = sigmoid(sharpness × (0.15 - cv))
sat_score = sigmoid(sharpness × (8.0 - mean))
candidato = amp_score × cv_score × sat_score

# COMPETIÇÃO: cristais vivem ou morrem
ressonância = |campo| × (crystal_map > 0.01)
HP += ressonância × 0.1    # ressoam → ganham vida
HP -= 0.02                  # todos decaem
alive = (HP > 0)            # HP ≤ 0 → morre
```

### Modos de uso

**Modo 1 — Zero treino (protótipos)**:
```python
from resnet_psi import ResNetPsi
rn = ResNetPsi()
rn.fit(train_images, train_labels)   # só computa média por classe
preds = rn.predict(test_images)      # distância euclidiana ao protótipo
```

**Modo 2 — Com decoder (treina só o readout)**:
```python
from resnet_psi import ResNetPsi, train_decoder
rn = ResNetPsi()
cmaps = rn.extract(images)           # crystal_maps como features
# treinar qualquer classificador em cima
```

---

## Resultados dos 12 Audits

| Audit | Resultado | Status |
|-------|-----------|--------|
| 1. MI via clustering | MI ≈ 4.75 bits, constante entre (γ,β). βc=2.32 do paper era tautológico | Provado |
| 2. Scaling | Acurácia escala: 48² (80.8%) → 192² (87.5%) com decoder. 10 seeds | Provado |
| 3. Frases | Física pura = 31%. Completação por frases era artefato léxico | Provado (negativo) |
| 4. Ablação cristais | Cristais originais pioram ~10% vs campo bruto | Provado |
| 5. Baselines | ResNet-Ψ não compete em acurácia com CNN/MLP convencionais | Provado |
| 6. Densidade | Cristais: 0.63 bits/KB. MLP: 11.2 bits/KB. Cristais NÃO são mais densos | Provado (negativo) |
| 7. Cristalização v2 | Competitivo: 88.1% linear, 93.1% MLP, MI 2.49 bits (+97%) | **Melhoria real** |
| 8. Protótipos | 77.4% MNIST sem decoder nenhum. Zero treino total | **Resultado chave** |
| 9. Fashion-MNIST | 67.0% com protótipos cristalinos. Mecanismo é genérico | **Generalização** |
| 10. Métodos leitura | 5 métodos testados — nenhum melhora. Gargalo é representação | Provado |
| 11. Campo 96² + CIFAR | MNIST 96²: 77.2% (≈48²). CIFAR-10: 18.7% (falhou) | **Limite encontrado** |
| 12. Texto MC | Hash=20% (aleatório). Embedding semântico=47-70% | **Limite + insight** |
| 31. CWRU few-shot | K≤5: Psi ganha CNN 1D treinada. K=1: 65.7% vs 60.7%. Crossover K=10 | **Industrial real** |

---

## O que funciona e o que não funciona

### Funciona
- Imagens com estrutura geométrica (contornos, formas, silhuetas)
- MNIST: 79.9% zero treino, 93.1% com MLP decoder
- Fashion-MNIST: 67% zero treino
- Sinais de vibração industrial (CWRU): 65-82% zero treino, supera CNN treinada com K≤5 amostras
- Cristalização competitiva (sigmoid + HP + competição)
- **Few-shot**: data-efficient por construção — protótipo converge com ~5 amostras

### Não funciona
- Imagens naturais complexas (CIFAR-10: 18.7%)
- Texto sem embedding semântico externo (hash = 20%, aleatório)
- Competir em acurácia com CNNs/Transformers
- Campo maior pra protótipos (96² ≈ 48² sem decoder)
- Cristalização original (piora 10% vs campo bruto)

### Insight central
A ResNet-Ψ é um **amplificador de estrutura**: preserva e cristaliza relações que já existem na entrada. Se a entrada tem estrutura geométrica (pixels), funciona sozinha. Se não tem (texto com hash), não funciona.

---

## Domínio de aplicação

| Dataset | Tipo | Resultado | Funciona? |
|---------|------|-----------|-----------|
| MNIST | Formas simples, alto contraste | 79.9% (zero treino) / 93.1% (MLP) | Sim |
| Fashion-MNIST | Silhuetas de roupas | 67.0% (zero treino) | Sim |
| CWRU Bearing | Vibração industrial 1D | 65-82% zero treino; ganha CNN treinada com K≤5 | **Sim** |
| CIFAR-10 | Imagens naturais, cor, textura | 18.7% | Não |
| Texto (hash) | Sem estrutura semântica | 20.0% | Não |
| Texto (embedding) | Com estrutura semântica | 47-70% | Parcial |

---

## Arquivos do projeto

### Biblioteca principal
- `resnet_psi.py` — API limpa: ResNetPsi, CrystalCompetitivo, psi_step, compute_crystal_maps, train_decoder
- `wave_gpt_v8.py` — PsiGPT v8: campo Ψ recorrente substituindo atenção, com acoplamento cristalino
- `nano_transformer.py` — Transformer padrão com análise Chinchilla, streaming de datasets, baseline de comparação

### Audits (raiz)
| Arquivo | Descrição |
|---------|-----------|
| `RN_psi_audit_1_beta_c.py` | MI via clustering (βc corrigido) |
| `RN_psi_audit_2_scaling.py` | Curva de escala 48-192, 10 seeds |
| `RN_psi_audit_3_frases.py` | Ablação de completação de frases |
| `RN_psi_audit_5_ablacao_baselines.py` | Ablação cristais + baselines MNIST |
| `RN_psi_audit_5_baselines.py` | Baselines MNIST (10 seeds) |
| `RN_psi_audit_6_densidade.py` | Densidade informacional comparada |
| `RN_psi_audit_7_cristalizacao_v2.py` | Cristalização suave + competitiva |
| `RN_psi_audit_8_prototipos.py` | Protótipos v1 (saturou — abandonado) |
| `RN_psi_audit_8b_prototipos_v2.py` | Protótipos v2 (77.4%) |
| `RN_psi_audit_9_fashion.py` | Fashion-MNIST (67%) |
| `RN_psi_audit_10_leitura.py` | 5 métodos de leitura |
| `RN_psi_audit_11_campo96_cifar.py` | Campo 96² + CIFAR-10 |
| `RN_psi_audit_12_frases_mc.py` | Texto multiple choice |
| `RN_psi_audit_31_cwru_fewshot.py` | CWRU few-shot: CNN vs MLP vs Psi, K=1..50, 5 runs |

### Versões antigas (teste_2/)
Código exploratório anterior aos audits: 3-bit/4-bit CMA-ES, 2D, 8D, testes de campo. Usa cristalização original (obsoleta).

### Documentação
- `perguntas_a_ser_respondidas.md` — Perguntas centrais + resultados consolidados dos 12 audits
- `revisao_paper.md` — Revisão detalhada do paper v2, seção por seção, com correções necessárias
- `resultados_2fases.md` — Resultados antigos 3-bit/4-bit
- `resultados_experimentais.md` — Resultados de caracterização do campo

---

## PsiGPT — Direção atual (texto com memória constante)

### Motivação
Transformer attention escala O(N²) com contexto — impossível treinar modelos grandes num 4GB.
O campo Ψ tem memória **O(1)**: sempre 24×24 = 576 floats por camada, independente do tamanho do contexto.
Objetivo: substituir a atenção pelo campo de ondas, mantendo memória constante em qualquer escala.

### Acoplamento cristalino (crystal attention)
Mecanismo novo desenvolvido nesta sessão. Cristais mediam comunicação global no campo:
```
acc[i] += λ × (crystal[i]×vel[i]) × Σ_j (crystal[j]×vel[j]) × (field[j] - field[i])
```
- Cristais com mesma fase de velocidade se atraem (acoplamento construtivo)
- Cristais em fase oposta se repelem (acoplamento destrutivo)
- Factoriza O(N²) → O(N): dois sums globais + elementwise, zero matriz, zero loops extras
- `crystal_lam` é parâmetro treinável no v8

### Otimizações wave_gpt_v8.py
- `from_field` e `proj_out` saíram do loop → batcheados em (B×C, FS²), 1 chamada em vez de 64
- `torch.roll` substituiu `F.conv2d` no Laplaciano (menos overhead)
- `torch.compile` + `torch.set_float32_matmul_precision('high')`
- Resultado: loop sequencial ainda domina (~20s/100steps no RTX 3050 Laptop)

### nano_transformer.py (baseline)
- Transformer padrão com Flash Attention (F.scaled_dot_product_attention)
- Análise Chinchilla automática antes do treino
- Streaming de datasets HuggingFace (Wikipedia PT funcionando)
- Config `auto` detecta maior modelo que cabe na VRAM
- Mixed precision (AMP fp16)
- Em treino: small (25M params) no Wikipedia PT, ~20s/100steps, loss ~2.4 no step 500

### Próximo
- Terminar treino do nano_transformer (baseline de loss e tempo)
- Testar wave_gpt_v8 e comparar memória vs transformer
- O campo Ψ nunca escala memória com contexto — esse é o resultado a validar

---

## Estado do paper

O paper v2 (`ResNet_Psi_Paper_PT_v2.pdf`) precisa de revisão pesada. Documento `revisao_paper.md` detalha tudo. Principais correções:

1. **Remover** βc = 2.32 (tautológico) → substituir por MI ≈ 4.75 bits
2. **Remover/reescrever** completação de frases 100% (artefato léxico)
3. **Substituir** cristalização original por competitiva
4. **Adicionar** 77.4% zero treino como resultado central
5. **Adicionar** Fashion-MNIST 67%, CIFAR-10 18.7%, texto 47-70%
6. **Atualizar** acurácias: 80.8% → 88.1% linear, 93.1% MLP
7. **Adicionar** comparação com reservoir computing (ninguém faz zero treino no readout)
8. **Adicionar** limites honestos

---

## Trabalhos anteriores (3-bit/4-bit/8D)

Resultados anteriores de paridade de bits usando CMA-ES:
- **3-bit**: 5/5 runs (100% convergência) com otimização em 2 fases
- **4-bit**: 3/3 runs (100% convergência) com campo persistente
- **2D contínuo**: 20/20 (100%)
- **8D v5**: Abandonado — bottleneck linear sem ativação não separa classes compostas

Constantes extraídas do 3-bit (usadas nos experimentos CMA-ES, diferentes das MNIST):
```python
PSI_GAMMA_3BIT = 0.085421
PSI_BETA_3BIT  = 0.007453
PSI_SIGMA_3BIT = 0.632152
```


  --- Gerações (PsiGPT (campo 16×16, cristais, chunk=32)) ---
  > ROMEO:
I whone good for countale, bond the making so cith aplspans to the doou ble?

POCNIUTES:
Nos I hot, you a myou
SAAUIArr,yatn tworg!or

RIThITch
oeBBlRS:eRWul:aPpndN.E
 B kRTM
ei dhttteeurnpn,msn.fsia

  > To be, ort sintles'sve,
A all the make to with an to this stret thers!


GRUCATIO:
The fa botther cacuse, be lof heh mereads od gamedsientl IanDW:nnnl,snnhsncmotngWwh oeAAhtttteueuOieUeS oaoiehsdDgNLhgxn?tmnra

  > The king is leightings that the wen dusath
And homhis somwe deageth, il por bettis bots fan
surrot she sauch atenen 'piened man a awhnl atyet lfafntnler Ieiiiaa er oiaeS
aoe!iia IeoaaoOioeoeRmT IaaotGdS,sdN:iiei

  Estado final dos cristais:
  camada 0: field_amp=0.2452  crystal_cov=13.3%  crystal_max=0.758
  camada 1: field_amp=0.4918  crystal_cov=48.8%  crystal_max=0.796
  camada 2: field_amp=0.6031  crystal_cov=80.9%  crystal_max=0.900
  camada 3: field_amp=0.6521  crystal_cov=90.6%  crystal_max=0.926

=================================================================
RESULTADO FINAL
=================================================================
                                   Transformer        PsiGPT
  ------------------------------  ------------  ------------
  Parâmetros                           875,520     1,017,880
  Loss final                            1.3844        1.9736
  Mem peak (MB)                           1362           117
  Tempo (s)                                687          3352
  Estado recorrente                  N/A (KV$)        256 KB
=================================================================

  Campo Ψ recorrente: 16×16 = 256 floats por camada
  Causal por construção. Memória constante. Cristais persistentes.