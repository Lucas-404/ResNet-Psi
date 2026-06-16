"""
chat.py - Conversa interativa com um modelo Arpa (base ou SFT), na tua GPU.

Uso:
    python tools/chat.py                                  # usa checkpoints-sft/best.pt
    python tools/chat.py --ckpt checkpoints-arpa150m/best.pt   # o base (so completa)
    python tools/chat.py --temp 0.8 --max-tokens 200

Comandos durante a conversa:
    /reset        limpa o historico
    /temp 0.5     muda a temperatura (criatividade)
    /sair         encerra
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arpa.sample import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints-sft/best.pt")
    ap.add_argument("--tokenizer-dir", default="tokenizer-arpa-64k-clean")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint nao encontrado: {args.ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = load_model(args.ckpt, device)
    ctx = model.cfg.context_length
    stop = [i for i in (tok.convert_tokens_to_ids("<|end|>"),
                        tok.convert_tokens_to_ids("<|end_of_text|>"))
            if i is not None and i >= 0]

    print(f"\n  Arpa chat | {os.path.basename(args.ckpt)} | "
          f"{model.num_params() / 1e6:.0f}M | {device}")
    print("  comandos: /reset  /temp 0.X  /sair\n")

    temp = args.temp
    history = ""
    while True:
        try:
            user = input("você> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user:
            continue
        if user == "/sair":
            break
        if user == "/reset":
            history = ""
            print("  (histórico limpo)\n")
            continue
        if user.startswith("/temp"):
            try:
                temp = float(user.split()[1])
                print(f"  (temperatura = {temp})\n")
            except (IndexError, ValueError):
                print("  uso: /temp 0.7\n")
            continue

        history += f"<|user|>\n{user}\n<|model|>\n"
        ids = tok.encode(history, add_special_tokens=False)[-(ctx - args.max_tokens):]
        ids_t = torch.tensor([ids], device=device)
        out = model.generate(ids_t, max_new_tokens=args.max_tokens,
                             temperature=temp, top_p=args.top_p, stop_ids=stop)
        reply = tok.decode([t for t in out[0, ids_t.size(1):].tolist()
                            if t not in stop]).strip()
        print(f"arpa> {reply}\n")
        history += f"{reply}\n<|end|>\n"


if __name__ == "__main__":
    main()
