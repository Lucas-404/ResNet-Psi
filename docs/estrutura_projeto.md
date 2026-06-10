# Estrutura Do Projeto

```text
.
├── nano_pt.py                 # treino principal do modelo 30M long-context
├── scripts/
│   ├── data/                  # download, checagem, tokenizer e pre-tokenizacao
│   ├── *.py                   # scripts legados de chat, eval e finetune
├── resnet_psi/                # experimentos ResNet-Psi e WaveGPT
├── sft/                       # dados de SFT e textos de projeto
├── docs/                      # notas e documentos do projeto
├── memory/                    # estado/contexto do projeto
├── data-arpa16k-quality/      # saida gerada por prepare_data.py
├── tokenizer-arpa-16k-clean/  # saida gerada por train_tokenizer_clean.py
├── checkpoints-arpa30m-64k/   # checkpoints do experimento atual
└── logs-arpa30m-64k/          # logs do experimento atual
```

Os wrappers na raiz mantem os comandos antigos funcionando:

```powershell
python download_datasets.py
python check_datasets.py
python train_tokenizer_clean.py --force
python prepare_data.py --max_tokens 300000000 --force
python nano_pt.py
```
