# ResNet-Psi — CLAUDE.md

## O que é este projeto

ResNet-Psi é uma rede neural baseada em física de ondas e cristalização competitiva. Um campo 2D (Ψ-Field) 48×48 recebe dados como perturbações ondulatórias. As ondas propagam, interferem, e regiões estáveis cristalizam — formando uma representação esparsa (crystal_map) que classifica dados **sem nenhum parâmetro treinado**.

**Arquivo principal**: `C:\ResNet-Psi\resnet_psi.py` — biblioteca base com API simples (fit/predict/score)

---

## Resultado central

**77.4% no MNIST e 67% no Fashion-MNIST sem treinar absolutamente nada.**

Zero parâmetros. Zero gradiente. Zero backprop. Zero decoder. Só física + média por classe + distância euclidiana. Nenhum sistema de reservoir computing existente classifica sem treinar pelo menos o readout.

Com decoder treinado (só no readout): 88.1% linear, 93.1% MLP.

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

---

## O que funciona e o que não funciona

### Funciona
- Imagens com estrutura geométrica (contornos, formas, silhuetas)
- MNIST: 77.4% zero treino, 93.1% com MLP decoder
- Fashion-MNIST: 67% zero treino
- Cristalização competitiva (sigmoid + HP + competição)

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
| MNIST | Formas simples, alto contraste | 77.4% (zero treino) / 93.1% (MLP) | Sim |
| Fashion-MNIST | Silhuetas de roupas | 67.0% (zero treino) | Sim |
| CIFAR-10 | Imagens naturais, cor, textura | 18.7% | Não |
| Texto (hash) | Sem estrutura semântica | 20.0% | Não |
| Texto (embedding) | Com estrutura semântica | 47-70% | Parcial |

---

## Arquivos do projeto

### Biblioteca principal
- `resnet_psi.py` — API limpa: ResNetPsi, CrystalCompetitivo, psi_step, compute_crystal_maps, train_decoder

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

### Versões antigas (teste_2/)
Código exploratório anterior aos audits: 3-bit/4-bit CMA-ES, 2D, 8D, testes de campo. Usa cristalização original (obsoleta).

### Documentação
- `perguntas_a_ser_respondidas.md` — Perguntas centrais + resultados consolidados dos 12 audits
- `revisao_paper.md` — Revisão detalhada do paper v2, seção por seção, com correções necessárias
- `resultados_2fases.md` — Resultados antigos 3-bit/4-bit
- `resultados_experimentais.md` — Resultados de caracterização do campo

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
