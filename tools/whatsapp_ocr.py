"""
whatsapp_ocr.py - Prints de WhatsApp -> SFT contabil anonimizado, 100% local.

Pipeline (nada sai da maquina):
    pasta de imagens -> EasyOCR (pt, GPU) -> agrupa em mensagens ->
    classifica remetente pela posicao do balao -> anonimiza (tools.anonimizar)
    -> jsonl no formato <|user|>/<|model|>/<|end|>

Heuristica de remetente: assume print do celular DA CONTADORA.
    balao a ESQUERDA  = cliente  -> <|user|>
    balao a DIREITA   = contadora -> <|model|>
(Ajuste com --split se a divisao sair errada; confira sempre no --audit.)

Uso:
    # 1. Auditar: ve o que o OCR leu e o que foi mascarado (NAO grava SFT)
    python tools/whatsapp_ocr.py prints/ --audit --names "Maria Souza"

    # 2. Gerar o SFT anonimizado
    python tools/whatsapp_ocr.py prints/ --out sft/contabilidade_whatsapp.jsonl \
        --names "Maria Souza"
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.anonimizar import Scrubber

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def load_reader(gpu=True):
    import easyocr
    # verbose=False: evita a barra de progresso com char unicode que quebra
    # no console cp1252 do Windows durante o download dos modelos.
    return easyocr.Reader(["pt"], gpu=gpu, verbose=False)


def transcribe(reader, path, split=0.5):
    """OCR -> lista de (lado, texto) por linha, ordenada de cima pra baixo.

    lado: 'user' (cliente, esquerda) ou 'model' (contadora, direita).
    """
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w = img.width
    results = reader.readtext(np.array(img))  # [(bbox, texto, conf), ...]
    lines = []
    for bbox, text, conf in results:
        if conf < 0.3 or not text.strip():
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        center_x = sum(xs) / len(xs)
        top_y = min(ys)
        lado = "model" if center_x > w * split else "user"
        lines.append((top_y, lado, text.strip()))
    lines.sort(key=lambda r: r[0])
    return [(lado, text) for _, lado, text in lines]


def to_turns(lines):
    """Funde linhas consecutivas do mesmo lado em um turno."""
    turns = []
    for lado, text in lines:
        if turns and turns[-1][0] == lado:
            turns[-1][1].append(text)
        else:
            turns.append([lado, [text]])
    return [(lado, " ".join(parts)) for lado, parts in turns]


def turns_to_sft(turns):
    """Turnos alternados -> string SFT <|user|>/<|model|> ... <|end|>."""
    tag = {"user": "<|user|>", "model": "<|model|>"}
    body = "\n".join(f"{tag[lado]}\n{txt}" for lado, txt in turns)
    return body + "\n<|end|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", help="pasta com os prints")
    ap.add_argument("--out", default=None, help="arquivo jsonl de saida (SFT)")
    ap.add_argument("--audit", action="store_true",
                    help="mostra transcricao + deteccoes, NAO grava SFT")
    ap.add_argument("--names", default="", help="nomes conhecidos (virgula): contadora, clientes")
    ap.add_argument("--split", type=float, default=0.5,
                    help="limiar horizontal esquerda/direita (0-1)")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    paths = []
    for ext in IMG_EXT:
        paths.extend(glob.glob(os.path.join(args.pasta, ext)))
    paths.sort()
    if not paths:
        sys.exit(f"Nenhuma imagem em {args.pasta}/ ({', '.join(IMG_EXT)})")
    print(f"{len(paths)} imagens | carregando EasyOCR...", flush=True)

    reader = load_reader(gpu=not args.no_gpu)
    names = [n.strip() for n in args.names.split(",") if n.strip()]

    out_f = open(args.out, "w", encoding="utf-8") if args.out and not args.audit else None
    total_rep = {}
    gravados = 0
    for path in paths:
        lines = transcribe(reader, path, split=args.split)
        turns = to_turns(lines)
        # anonimiza turno a turno com scrubber novo por conversa (pseudonimo local)
        scr = Scrubber(extra_names=names)
        turns_limpos = [(lado, scr.scrub(txt)[0]) for lado, txt in turns]
        for k, v in scr.report.items():
            total_rep[k] = total_rep.get(k, 0) + v

        if args.audit:
            print(f"\n===== {os.path.basename(path)} =====")
            for lado, txt in turns_limpos:
                print(f"  [{ 'CLIENTE' if lado=='user' else 'CONTADORA'}] {txt}")
        elif out_f:
            sft = turns_to_sft(turns_limpos)
            out_f.write(json.dumps({"text": sft}, ensure_ascii=False) + "\n")
            gravados += 1

    if out_f:
        out_f.close()
    print("\n=== deteccoes de PII (total) ===")
    for k, v in sorted(total_rep.items()):
        print(f"  {k}: {v}")
    if args.audit:
        print("\n[auditoria] nada gravado. Confira a divisao CLIENTE/CONTADORA e "
              "as linhas CREDENCIAL_LINHA antes de gerar o SFT.")
    elif out_f:
        print(f"\nGravados {gravados} exemplos em {args.out}")
        print("Lembrete: apague os prints originais depois de conferir.")


if __name__ == "__main__":
    main()
