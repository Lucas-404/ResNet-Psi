#!/usr/bin/env python3
"""
make_colab.py - Monta a pasta colab_upload/ com tudo que vai pro Colab.

Sempre rode antes de subir pro Drive — a pasta e gerada do zero a partir
dos arquivos canonicos do repo, entao nunca fica desatualizada.

Uso:
    python make_colab.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "colab_upload"

ITEMS = [
    "arpa",                 # pacote completo (modelo, treino, dados, sft, chat)
    "sft",                  # seeds de SFT (contabilidade + conversa)
    "requirements.txt",
    "colab_setup.txt",      # runbook celula por celula
]

EXCLUDE = {"__pycache__"}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    for item in ITEMS:
        src = ROOT / item
        dst = OUT / item
        if src.is_dir():
            shutil.copytree(src, dst,
                            ignore=shutil.ignore_patterns(*EXCLUDE, "*.pyc"))
        else:
            shutil.copy2(src, dst)

    files = sorted(p.relative_to(OUT) for p in OUT.rglob("*") if p.is_file())
    total_kb = sum((OUT / f).stat().st_size for f in files) / 1024
    print(f"colab_upload/ pronta: {len(files)} arquivos, {total_kb:.0f} KB")
    for f in files:
        print(f"  {f}")
    print("\nSobe a pasta inteira pro Drive (ou /content) e segue o colab_setup.txt.")


if __name__ == "__main__":
    main()
