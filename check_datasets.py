#!/usr/bin/env python3
"""
check_datasets.py — Verifica se todos os datasets do mix estão acessíveis.

Testa cada dataset do DATASETS_CONFIG: carrega, puxa 1 sample, valida o campo
de texto. Roda em ~30s. Execute antes de prepare_data.py para não desperdiçar horas.

Uso:
    python check_datasets.py
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Mesmo DATASETS_CONFIG do prepare_data.py / nano_pt.py
# Atualizar aqui se mudar lá.
DATASETS_CONFIG = [
    ("wikimedia/wikipedia",          "20231101.pt", 0.35, "text"),
    ("TucanoBR/GigaVerbo",           None,          0.30, "text"),
    ("codeparrot/codeparrot-clean",  None,          0.20, "content"),
    ("dominguesm/alpaca-data-pt-br", None,          0.10, "instruction"),
    ("allenai/c4",                   "pt",          0.05, "text"),
]

# Datasets extras usados em validação (nano_pt.py avalia no Wikipedia PT)
VAL_DATASETS = [
    ("wikimedia/wikipedia", "20231101.pt", "text"),
]

def check_one(ds_name, ds_config, field):
    """
    Tenta carregar o dataset em streaming e puxa 1 sample.
    Retorna (ok, info_str, sample_preview).
    """
    try:
        from datasets import load_dataset
        t0 = time.time()
        ds = load_dataset(ds_name, ds_config, split="train", streaming=True)
        sample = next(iter(ds))
        elapsed = time.time() - t0

        # Verifica campo de texto
        text = sample.get(field, "")

        # Aira-Dataset pode ser conversa
        if not text:
            for conv_field in ("conversation", "messages"):
                if conv_field in sample and isinstance(sample[conv_field], list):
                    turns = sample[conv_field]
                    text  = " ".join(
                        t.get("content", "") for t in turns if isinstance(t, dict)
                    )
                    break

        if not text:
            available = list(sample.keys())
            return False, f"campo '{field}' vazio. Campos disponíveis: {available}", ""

        preview = text[:80].replace("\n", " ")
        return True, f"{elapsed:.1f}s | {len(text)} chars", preview

    except Exception as e:
        return False, str(e)[:120], ""


def main():
    print("=" * 62)
    print("  check_datasets.py — Verificação do mix de dados")
    print("=" * 62)

    all_ok   = True
    results  = []

    print("\nDatasets de TREINO:")
    for ds_name, ds_config, weight, field in DATASETS_CONFIG:
        label = f"{ds_name}" + (f" ({ds_config})" if ds_config else "")
        print(f"  Testando {label} ...", end=" ", flush=True)
        ok, info, preview = check_one(ds_name, ds_config, field)
        status = "OK" if ok else "ERRO"
        print(f"[{status}] {info}")
        if ok and preview:
            print(f"         preview: \"{preview}\"")
        if not ok:
            all_ok = False
        results.append((label, ok, info))

    print("\nDatasets de VALIDAÇÃO:")
    for ds_name, ds_config, field in VAL_DATASETS:
        label = f"{ds_name}" + (f" ({ds_config})" if ds_config else "")
        # Já testado acima se estiver no mix de treino
        already = any(r[0] == label for r in results)
        if already:
            print(f"  {label} — já testado acima")
            continue
        print(f"  Testando {label} ...", end=" ", flush=True)
        ok, info, preview = check_one(ds_name, ds_config, field)
        status = "OK" if ok else "ERRO"
        print(f"[{status}] {info}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 62)
    if all_ok:
        print("  RESULTADO: Todos os datasets OK — pode rodar prepare_data.py")
    else:
        print("  RESULTADO: Alguns datasets falharam — corrija antes de prosseguir")
        print()
        print("  Datasets com erro:")
        for label, ok, info in results:
            if not ok:
                print(f"    - {label}: {info}")
    print("=" * 62)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
