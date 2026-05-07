#!/usr/bin/env python3
"""
gather_text.py - Junta arquivos do projeto em um único .txt para treino da CLM.

Coleta: CLAUDE.md, wave_gpt_v12.py, resnet_psi.py, arquivos de memória.

Uso:
    python gather_text.py
    python gather_text.py --out meu_texto.txt
"""

import os
import argparse
from pathlib import Path

ARQUIVOS = [
    # Documentação do projeto
    "CLAUDE.md",
    "revisao_paper.md",
    "codex.md",
    # Código principal
    "wave_gpt_v12.py",
    "resnet_psi.py",
    "nano_pt.py",
]

MEMORIA_DIR = Path(r"C:\Users\lputt\.claude\projects\C--RN-Teste-ResNet-Psi\memory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="project_text.txt")
    parser.add_argument("--min_chars", type=int, default=50,
                        help="Ignora arquivos menores que N chars")
    args = parser.parse_args()

    partes = []
    total  = 0

    def adiciona(path, label):
        nonlocal total
        try:
            texto = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
            if len(texto) < args.min_chars:
                return
            partes.append(f"\n\n# === {label} ===\n\n{texto}")
            total += len(texto)
            print(f"  [OK] {label:40s} {len(texto):>8,} chars")
        except Exception as e:
            print(f"  [--] {label:40s} {e}")

    print("Coletando arquivos do projeto...\n")

    # Arquivos principais
    for arq in ARQUIVOS:
        adiciona(arq, arq)

    # Arquivos de memória
    if MEMORIA_DIR.exists():
        for md in sorted(MEMORIA_DIR.glob("*.md")):
            adiciona(md, f"memory/{md.name}")

    # Escreve
    texto_final = "\n".join(partes)
    Path(args.out).write_text(texto_final, encoding="utf-8")

    print(f"\nTotal: {total:,} chars → {args.out}")
    print(f"Vocabulário estimado: {len(set(texto_final))} chars únicos")


if __name__ == "__main__":
    main()
