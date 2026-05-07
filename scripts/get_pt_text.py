#!/usr/bin/env python3
"""
get_pt_text.py - Baixa texto bruto da Wikipedia PT para treino char-level da CLM.

Saída: ./data/pt_wikipedia.txt (~1M chars de português corrido)

Uso:
    python get_pt_text.py
    python get_pt_text.py --chars 2000000
"""

import re
import argparse
from pathlib import Path
from datasets import load_dataset

TARGET_CHARS = 1_000_000
OUT_PATH     = Path("./data/pt_wikipedia.txt")


def clean(text: str) -> str:
    """Remove artefatos do WikiText, deixa só prosa."""
    # Remove headings == Título ==
    text = re.sub(r'={2,}[^=\n]+=+', '', text)
    # Remove linhas com só símbolos/tabelas
    lines = [l for l in text.splitlines()
             if len(l.strip()) > 20 and l.strip()[0].isalpha()]
    text = '\n'.join(lines)
    # Colapsa múltiplas linhas em branco
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chars', type=int, default=TARGET_CHARS)
    parser.add_argument('--out',   type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)

    print(f"Baixando Wikipedia PT (streaming)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.pt",
                      split="train", streaming=True)

    parts = []
    total = 0

    for i, item in enumerate(ds):
        text = clean(item.get("text", ""))
        if len(text) < 400:
            continue
        parts.append(text)
        total += len(text)
        if (i + 1) % 500 == 0:
            pct = min(100.0, total / args.chars * 100)
            print(f"  {i+1:,} artigos | {total:,} chars ({pct:.1f}%)", flush=True)
        if total >= args.chars:
            break

    result = "\n\n".join(parts)[:args.chars]
    out.write_text(result, encoding="utf-8")

    vocab = sorted(set(result))
    print(f"\n[OK] {len(result):,} chars → {out}")
    print(f"     Vocab: {len(vocab)} chars únicos")
    print(f"     Primeiros 200 chars:\n{result[:200]}")


if __name__ == "__main__":
    main()
