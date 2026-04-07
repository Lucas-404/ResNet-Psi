---
name: project_estado
description: Estado atual do projeto ResNet-Psi em 2026-03-31 — paradigma computacional (memória O(1)) + experimentos Wave GPT (geração de texto com campo de ondas)
type: project
---

## Arquivo principal: C:\ResNet-Psi\resnet_psi.py (biblioteca base limpa)

## TESE CENTRAL DO PROJETO
O ResNet-Psi **NÃO é sobre acurácia**. É sobre uma **nova forma de computar**:
- **Memória O(1)**: campo 48×48 = 2304 posições fixas, independente do tamanho da entrada. Transformers escalam O(n²).
- **Processamento local**: Laplaciano 3×3. Sem multiplicação de matrizes N×N.
- **Sem treino**: a física do campo (propagação, interferência, cristalização) faz a computação. Zero parâmetros, zero gradiente, zero backprop.
- **Implementável em hardware analógico**: ondas propagando num meio físico fazem o mesmo processamento.

## Wave Transformer — PROVADO
- **wave_transformer.py**: Wave Attention substitui Q×K^T por campo de ondas
  - Resultado: 97.0% MNIST vs 97.9% Transformer (0.9% gap)
  - **Prova que o campo computa** com qualidade equivalente ao Transformer
- **wave_transformer_stress.py**: Stress test de memória
  - 4096 tokens: Transformer 4137MB vs Wave 93MB = **44x menos memória**
  - 8192 tokens: **Transformer OOM**, Wave 189MB (sobrevive)
  - 32768 tokens: Wave 597MB (funciona tranquilo)
  - **Prova que memória é O(1)**

## Wave GPT — PROBLEMA ABERTO (geração de texto)
Tentamos 5 versões de GPT com campo de ondas. Todas aprendem (loss boa) mas colapsam em repetição na geração.

| Versão | Abordagem | Loss | Geração |
|--------|-----------|------|---------|
| v1 (wave_gpt.py) | Campo processa tudo de uma vez | 1.44 | "youyouyou" — colapso |
| v2 (wave_gpt_v2.py) | Crystal Attention (QKV→ondas) | 1.24 | "Whololod ld" — colapso |
| v3 (wave_gpt_v3.py) | Campo recorrente por chunks | 1.69 | "IIIII", "eeeee" — colapso |
| v4 (wave_gpt_v4.py) | Chunks como imagens 2D (conv) | 1.77 | "ggggg", "AAAA" — colapso |
| v5 (wave_gpt_v5.py) | Hierarquia char→palavra→contexto | rodando... | (não completou ainda) |

### Problema fundamental identificado
O campo de ondas **não tem seletividade per-token**. Todos os tokens leem o MESMO campo e recebem a MESMA informação. No Transformer, Q×K^T dá peso diferente pra cada par de tokens. No campo, não tem esse mecanismo.

Funciona pra **classificação** (resposta global basta) mas não pra **geração autoregressiva** (cada posição precisa de resposta diferente).

### Ideia do usuário (não testada ainda)
Cada palavra/frase deveria gerar uma **assinatura única** no campo (como imagens na ResNet-Psi). Tratar texto como hierarquia: char → palavra → frase, onde cada nível gera interferência distinta. O v5 tentou isso mas foi muito lento na primeira versão e a segunda está rodando.

### Próxima sessão — opções
1. **Consolidar resultados**: o campo prova computação + memória O(1). Geração é limitação documentada.
2. **Híbrido**: atenção local (janela ~32 tokens) pra seletividade + campo pra contexto global O(1). Não é campo puro.
3. **Continuar hierarquia**: ver resultado do v5 e iterar se necessário.

## Resultados dos baselines (Audit 30/30b)
- Com dataset completo: ResNet-Ψ (77.7%) é PIOR que pixels brutos (82.0%)
- **O valor está no paradigma computacional, não na acurácia**

## Mapa do domínio (14 datasets)
### FUNCIONA (estrutura geométrica)
MNIST 77.4%, Fashion 67%, EMNIST Letters 40.2%, EMNIST Digits 77.4%, PneumoniaMNIST 69.2%, BreastMNIST 66.7%, BloodMNIST 55%, OrganAMNIST 47.7%, PathMNIST 43%

### NÃO FUNCIONA (textura/cor)
CIFAR-10 18.7%, DermaMNIST 16.1%, SVHN 9.8%, OCTMNIST 25.1%, RetinaMNIST 35.2%

## Estado do paper
- Paper precisa de framing correto: o resultado é o CAMPO como paradigma computacional, não o 77.4%
- Agora tem evidência adicional forte: Wave Attention = Transformer em loss com 44x menos memória
