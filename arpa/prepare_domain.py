"""
prepare_domain.py - Gera o bin de dominio CONTABILIDADE para a fase de annealing.

Fonte: eduagarcia/LegalPT_dedup (24M docs juridico-fiscais PT, deduplicado).
Filtra documentos com vocabulario contabil/fiscal/tributario e mistura com
uma fracao de dados gerais (para o modelo nao esquecer portugues comum).

Saida:
    data/contabilidade/train_tokens.bin

Uso (annealing = ultimos ~15% do treino):
    python arpa/prepare_domain.py --target-tokens 1.5e9
    python arpa/pretrain.py --config a100 --resume latest \
        --train-bin data/contabilidade/train_tokens.bin --max-tokens 10e9
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arpa.data import BinWriter
from arpa.prepare_data import clean_text, is_quality_text, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH

# Vocabulario de dominio: um doc precisa de >= MIN_HITS termos DISTINTOS
DOMAIN_TERMS = [
    "contabilidade", "contabil", "contábil", "balanço patrimonial", "balanco patrimonial",
    "demonstração financeira", "demonstracao financeira", "demonstrações contábeis",
    "tributário", "tributario", "tributo", "imposto", "fiscal", "alíquota", "aliquota",
    "icms", "ipi", "irpj", "csll", "cofins", "pis", "iss", "simples nacional",
    "auditoria", "auditor", "escrituração", "escrituracao", "lançamento contábil",
    "débito", "debito", "crédito", "credito", "patrimônio líquido", "patrimonio liquido",
    "ativo circulante", "passivo", "depreciação", "depreciacao", "amortização",
    "receita federal", "nota fiscal", "folha de pagamento", "regime de competência",
    "lucro real", "lucro presumido", "plano de contas", "razão contábil",
    "nbc", "cpc ", "cfc", "sped", "ecd", "ecf", "darf", "dctf",
]
MIN_HITS = 3


def domain_score(text_lower: str) -> int:
    return sum(1 for term in DOMAIN_TERMS if term in text_lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens", type=float, default=1.5e9)
    parser.add_argument("--general-frac", type=float, default=0.3,
                        help="fracao de dados gerais no mix (evita esquecimento)")
    parser.add_argument("--out-dir", default="data/contabilidade")
    parser.add_argument("--tokenizer-dir", default="tokenizer-arpa-64k-clean")
    parser.add_argument("--general-bin", default="data/arpa150m/train_tokens.bin",
                        help="bin geral de onde copiar a fracao de dados gerais")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    import numpy as np
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

    writer = BinWriter(os.path.join(args.out_dir, "train_tokens.bin"))
    target_domain = int(args.target_tokens * (1 - args.general_frac))
    target_general = int(args.target_tokens * args.general_frac)

    # ---- 1. Dominio: LegalPT filtrado por vocabulario contabil ----
    # acordaos_tcu = Tribunal de Contas (auditoria/contabilidade publica) -> peso maior
    # tesemo_v2    = legislacao e normativos federais
    # mlp_pt_legal-mc4 = web juridica generica (filtro pega a parte fiscal/contabil)
    from datasets import interleave_datasets
    subsets = [("acordaos_tcu", 0.45), ("tesemo_v2", 0.30), ("mlp_pt_legal-mc4", 0.25)]
    print(f"[1/2] Streaming LegalPT_dedup {[s for s, _ in subsets]} "
          f"-> alvo {target_domain / 1e9:.2f}B tokens de dominio")
    streams = [load_dataset("eduagarcia/LegalPT_dedup", name, split="train",
                            streaming=True) for name, _ in subsets]
    ds = interleave_datasets(streams, probabilities=[w for _, w in subsets],
                             seed=42, stopping_strategy="all_exhausted")
    seen = kept = 0
    t0 = time.time()
    for sample in ds:
        text = sample.get("text") or ""
        seen += 1
        if len(text) < MIN_TEXT_LENGTH:
            continue
        text = text[:MAX_TEXT_LENGTH]
        if domain_score(text.lower()) < MIN_HITS:
            continue
        text = clean_text(text, "text")
        if len(text) < MIN_TEXT_LENGTH or not is_quality_text(text, "text"):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        writer.write(ids)
        kept += 1
        if kept % 20_000 == 0:
            rate = writer.total / max(1, time.time() - t0)
            print(f"  {writer.total / 1e6:.0f}M tokens | {kept:,}/{seen:,} docs "
                  f"| {rate / 1e3:.0f}K tok/s", flush=True)
        if writer.total >= target_domain:
            break
    print(f"  dominio: {writer.total:,} tokens de {kept:,} docs (aproveitamento {kept}/{seen})")

    # ---- 2. Fracao geral: fatias do bin de pre-treino ----
    print(f"[2/2] Copiando {target_general / 1e9:.2f}B tokens gerais de {args.general_bin}")
    if os.path.exists(args.general_bin):
        general = np.memmap(args.general_bin, dtype=np.uint16, mode="r")
        rng = np.random.default_rng(123)
        chunk = 65_536
        copied = 0
        while copied < target_general:
            start = rng.integers(0, len(general) - chunk)
            writer.write(np.asarray(general[start:start + chunk]))
            copied += chunk
        print(f"  geral: {copied:,} tokens")
    else:
        print(f"  [aviso] {args.general_bin} nao existe — bin ficou 100% dominio")

    writer.close()
    print(f"\nPronto: {writer.total:,} tokens em {args.out_dir}/train_tokens.bin")


if __name__ == "__main__":
    main()
