#!/usr/bin/env python3
"""
prepare_data.py - Pré-tokeniza mix de ouro PT-BR e salva em disco.

Execução única. Depois disso, resume de treino é instantâneo.

Mix (golden data local — lê de ./datasets_raw/):
  40% fineweb-edu        — web educacional filtrado (alta qualidade)
  30% open-web-math      — matemática e raciocínio
  20% Wikipedia PT       — factual, sintaxe limpa
  10% StackMathQA        — perguntas/respostas matemáticas

Saídas:
    ./data/train_tokens.bin   uint16, 2.1B tokens (~4.2 GB)
    ./data/val_tokens.bin     uint16, ~5M tokens   (~10 MB)

Uso:
    python prepare_data.py --max_tokens 2_100_000_000 --force
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk, interleave_datasets

# Mesmos parâmetros do nano_pt.py
TOKENIZER_DIR  = "./tokenizer-arpa-32k"
DATA_DIR       = "./data"
DATASETS_DIR   = "./datasets_raw"
TRAIN_TOKENS   = 2_100_000_000
VAL_TOKENS     =   5_000_000
VAL_SKIP_DOCS  =     200_000   # mesmo skip do evaluate() em nano_pt.py

# Mix de ouro local — lê Arrow de disco (sem streaming, sem timeout)
# (label, pasta_local, peso, campo_texto)
DATASETS_CONFIG = [
    ("fineweb-edu",   "fineweb-edu-sample-10BT", 0.35, "text"),
    ("open-web-math", "open-web-math",            0.25, "text"),
    ("wikipedia-pt",  "wikipedia-20231101.pt",    0.20, "text"),
    ("stackmathqa",   "StackMathQA",              0.15, "Q"),
    ("alpaca-pt-br",  "alpaca-data-pt-br",        0.05, "instruction"),
]

MIN_TEXT_LENGTH = 200
MAX_TEXT_LENGTH = 50_000
SEED = 42

WRITE_CHUNK = 100_000   # tokens por escrita (evita flush excessivo)
LOG_EVERY   =  10_000_000  # log a cada 10M tokens


def is_quality_text(text: str) -> bool:
    """Filtra textos de baixa qualidade."""
    n = len(text)
    if n == 0:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / n < 0.55:   # fineweb/math têm mais números — limiar mais baixo
        return False
    words = text.split()
    if len(words) < 20:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2.5 or avg_word_len > 20.0:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 5:
        top = Counter(lines).most_common(1)[0][1]
        if top / len(lines) > 0.4:
            return False
    return True


def extract_text(sample: dict, text_field: str) -> str:
    """
    Extrai texto de um sample, lidando com formatos diferentes:
    - texto plano: sample["text"]
    - StackMathQA: sample["Q"] (pergunta) + sample["A"] (resposta)
    - conversas: sample["conversation"] / sample["messages"]
    - Alpaca: sample["instruction"] + sample["output"]
    """
    # StackMathQA: Q + A
    if text_field == "Q":
        q = (sample.get("Q") or "").strip()
        a = (sample.get("A") or "").strip()
        parts = [p for p in [q, a] if p]
        return "\n".join(parts)

    # Formato conversa
    for conv_field in ("conversation", "messages"):
        if conv_field in sample and isinstance(sample[conv_field], list):
            parts = []
            for turn in sample[conv_field]:
                role    = turn.get("role", "")
                content = turn.get("content", "").strip()
                if content:
                    prefix = "Humano:" if role == "user" else "Assistente:"
                    parts.append(f"{prefix} {content}")
            return "\n".join(parts)

    # Formato Alpaca (instruction / input / output)
    if "instruction" in sample and "output" in sample:
        instruction = (sample.get("instruction") or "").strip()
        inp         = (sample.get("input")       or "").strip()
        output      = (sample.get("output")      or "").strip()
        parts = [p for p in [instruction, inp, output] if p]
        return "\n".join(parts)

    # Texto plano
    return sample.get(text_field, "") or sample.get("text", "") or ""


def build_interleaved(seed=SEED):
    """
    Carrega datasets Arrow locais e intercala com os pesos definidos.
    """
    streams, weights = [], []

    for label, folder, weight, text_field in DATASETS_CONFIG:
        path = os.path.join(DATASETS_DIR, folder)
        if not os.path.exists(path):
            print(f"  [SKIP] {label} — pasta não encontrada: {path}", flush=True)
            continue
        print(f"  Carregando {label} de {path}...", flush=True)
        try:
            ds = load_from_disk(path)
            # load_from_disk retorna Dataset (não IterableDataset)
            # converte para iterable para o interleave
            streams.append(ds.to_iterable_dataset())
            weights.append(weight)
            print(f"  [OK] {label} ({len(ds):,} exemplos)", flush=True)
        except Exception as e:
            print(f"  [ERRO] {label}: {e} — pulando", flush=True)

    if not streams:
        raise RuntimeError("Nenhum dataset carregado! Rode download_datasets.py primeiro.")

    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    print(f"\n  Mix efetivo: {[f'{w:.0%}' for w in weights]}", flush=True)

    return interleave_datasets(streams, probabilities=weights, seed=seed,
                               stopping_strategy="all_exhausted")


def tokenize_to_file(out_path: str, target_tokens: int, tokenizer,
                     dataset_iter, label: str, text_field: str = "text"):
    """
    Itera dataset_iter, tokeniza, escreve tokens uint16 em out_path.
    Para após target_tokens tokens escritos.
    """
    eos = tokenizer.eos_token_id
    buf = []
    total = 0
    last_log = 0

    with open(out_path, "wb") as f:
        for sample in dataset_iter:
            text = extract_text(sample, text_field)
            if not text or len(text) < MIN_TEXT_LENGTH:
                continue
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            if not is_quality_text(text):
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue

            buf.extend(ids)
            buf.append(eos)

            # Flush em chunks
            if len(buf) >= WRITE_CHUNK:
                chunk = np.array(buf, dtype=np.uint16)
                chunk.tofile(f)
                total += len(buf)
                buf = []

                if total - last_log >= LOG_EVERY:
                    pct = min(100.0, total / target_tokens * 100)
                    print(f"  [{label}] {total//1_000_000}M / {target_tokens//1_000_000}M tokens  ({pct:.1f}%)",
                          flush=True)
                    last_log = total

            if total >= target_tokens:
                break

        # Flush restante
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            total += len(buf)

    size_gb = os.path.getsize(out_path) / (1024 ** 3)
    print(f"  [{label}] Salvo: {total:,} tokens em {out_path}  ({size_gb:.2f} GB)", flush=True)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=TRAIN_TOKENS)
    parser.add_argument("--force", action="store_true", help="Recriar mesmo se já existir")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Carregando tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    print(f"  vocab_size={tokenizer.vocab_size}, eos={tokenizer.eos_token_id}")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    train_path = os.path.join(DATA_DIR, "train_tokens.bin")
    if os.path.exists(train_path) and not args.force:
        n = os.path.getsize(train_path) // 2
        print(f"\n[OK] {train_path} já existe ({n:,} tokens). Use --force para recriar.")
    else:
        print(f"\nTokenizando dataset de TREINO ({args.max_tokens//1_000_000}M tokens)...")
        print("Carregando datasets locais intercalados...")
        dataset = build_interleaved(seed=SEED)
        tokenize_to_file(train_path, args.max_tokens, tokenizer, dataset, "train")

    # -------------------------------------------------------------------------
    # Validation — usa Wikipedia local
    # -------------------------------------------------------------------------
    val_path = os.path.join(DATA_DIR, "val_tokens.bin")
    if os.path.exists(val_path) and not args.force:
        n = os.path.getsize(val_path) // 2
        print(f"\n[OK] {val_path} já existe ({n:,} tokens). Use --force para recriar.")
    else:
        print(f"\nTokenizando dataset de VALIDAÇÃO ({VAL_TOKENS//1_000_000}M tokens)...")
        wiki_path = os.path.join(DATASETS_DIR, "wikipedia-20231101.pt")
        val_ds = load_from_disk(wiki_path).to_iterable_dataset()
        val_ds = val_ds.skip(VAL_SKIP_DOCS)
        tokenize_to_file(val_path, VAL_TOKENS, tokenizer, val_ds, "val")

    print("\nPronto! Agora rode: python nano_pt.py")


if __name__ == "__main__":
    main()
