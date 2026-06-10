# Arpa-150M

Pipeline completo de LLM em PyTorch puro: tokenizer → dados → pré-treino → SFT → chat.

## Modelo

LLaMA-style 150M: 16 camadas, hidden 768, 12 heads (GQA 4 KV), SwiGLU 2048,
RoPE θ=500K, contexto 8192, vocab 64K, embeddings amarrados, **QK-Norm**
(estabilidade estilo Qwen3/Gemma3). Flash Attention (SDPA), BF16, torch.compile.

Otimização: **Muon** (Newton-Schulz, ~2x mais eficiente por FLOP que AdamW)
nas matrizes 2D do miolo (100.7M params) + AdamW nos embeddings/norms (49.2M),
com warmup de momentum 0.85→0.95 e cosine decay compartilhado.

| Preset  | Onde          | Contexto | Batch efetivo | Orçamento |
|---------|---------------|----------|---------------|-----------|
| `a100`  | Colab A100    | 8192     | 524K tok/step | 3.2B tok  |
| `local` | RTX 3050 4GB  | 1024     | 16K tok/step  | 20M (smoke) |
| `tiny`  | CPU           | 128      | —             | teste de pipeline |

## Ordem de execução (Colab A100)

```bash
pip install -q torch transformers datasets tokenizers

# 1. Tokenizer 64K (uma vez; ~30min). Salve no Drive depois.
python arpa/train_tokenizer.py

# 2. Bins de pré-treino (uma vez; streaming do mix real -> ~6.6GB).
#    Salve train_tokens.bin/val_tokens.bin no Drive.
python arpa/prepare_data.py --train-tokens 3.3e9

# 3. Pré-treino (~12-18h de A100 para 3.2B tokens)
python arpa/pretrain.py --config a100
python arpa/pretrain.py --config a100 --resume latest   # apos desconexao

# 4. SFT (depois do pré-treino)
python arpa/sft.py --init checkpoints-arpa150m/best.pt --data "sft/*.jsonl"

# 5. Conversar
python arpa/sample.py --ckpt checkpoints-sft/best.pt --chat
```

Para guardar checkpoints no Drive, aponte `checkpoint_dir` (config.py) para
`/content/drive/MyDrive/arpa150m` ou copie os `.pt` manualmente.

## Smoke test local (antes de gastar Colab)

```bash
python arpa/pretrain.py --config tiny     # valida o pipeline inteiro em CPU
```

## Mix de dados (prepare_data.py)

| Fonte | Peso | Tipo |
|---|---|---|
| Wikipedia PT | 25% | texto |
| Wikipedia EN | 25% | texto |
| codeparrot-clean | 18% | código |
| open-web-math | 15% | reasoning |
| StackMathQA | 5% | reasoning |
| OpenAssistant | 7% | conversa |
| Alpaca PT-BR | 5% | conversa |

Filtros de qualidade por tipo (ratio de letras, repetição de linhas, tamanho).
Documentos `reasoning`/`conversation` recebem prefixo `<|reasoning|>`/`<|conversation|>`.

## Formato SFT

```json
{"text": "<|user|>\nPergunta\n<|model|>\nResposta\n<|end|>"}
```

Loss só nos tokens da resposta (prompt mascarado com -100).

## Modelo de domínio: contabilidade (Arpa-Contábil)

Receita em 3 fases (a arquitetura não muda; o domínio entra pelos dados):

```bash
# Fase 1 — base de português geral (85% do orçamento)
python arpa/pretrain.py --config a100 --max-tokens 8.5e9

# Fase 2 — annealing contábil/fiscal (últimos 1.5B tokens, LR caindo)
#   Corpus: LegalPT_dedup (acórdãos TCU 45% + legislação 30% + web jurídica 25%),
#   filtrado por vocabulário contábil (ICMS, IRPJ, balanço, escrituração...),
#   com 30% de dados gerais pra não esquecer português comum.
python arpa/prepare_domain.py --target-tokens 1.5e9
python arpa/pretrain.py --config a100 --resume latest \
    --train-bin data/contabilidade/train_tokens.bin --max-tokens 10e9

# Fase 3 — SFT com Q&A contábil
#   Seed em sft/contabilidade_seed.jsonl (formato <|user|>/<|model|>/<|end|>).
#   Pra escalar: gere 2-5K exemplos distilando de um modelo grande, variando
#   tema, formato e tom — NUNCA templates repetidos (lição do clean_continue).
python arpa/sft.py --init checkpoints-arpa150m/best.pt \
    --data sft/contabilidade_seed.jsonl sft/contabilidade_distill.jsonl
```

Expectativa honesta para 150M: ótimo em terminologia, classificação de
lançamentos, rascunhos e resumos de documentos fiscais. **Não confiar** em
alíquotas, prazos e citações de norma de cabeça — para isso, acople RAG
(o modelo redige, a base de normas responde).

## Decisões de eficiência

- **Contexto 8192** (não 64K): atenção é quadrática; 64K gastava quase todo o
  FLOP em atenção num modelo de 150M. Contexto longo se estende depois com
  fine-tune curto subindo o RoPE theta.
- **Bins memmap uint16**: resume O(1), sem re-iterar streaming, sem estado de
  dataloader. 3.3B tokens = 6.6GB.
- **PyTorch puro**: sem overhead do HF Trainer/LlamaForCausalLM no loop.
- **Amostragem aleatória de janelas** (nanoGPT-style): determinística por
  (seed, step), resume exato.
