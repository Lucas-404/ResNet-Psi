"""
sample.py - Geracao de texto e chat com um checkpoint Arpa.

Uso:
    python arpa/sample.py --ckpt checkpoints-arpa150m/best.pt --prompt "O Brasil e"
    python arpa/sample.py --ckpt checkpoints-sft/best.pt --chat
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arpa.config import ModelConfig
from arpa.model import Arpa


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**state["config"]["model"])
    model = Arpa(mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer-dir", default="tokenizer-arpa-64k-clean")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = load_model(args.ckpt, device)
    print(f"Modelo: {model.num_params() / 1e6:.1f}M params | {device}")

    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    stop_ids = [i for i in (end_id, eos_id) if i is not None and i >= 0]

    if args.chat:
        print("Chat (Ctrl+C para sair)\n")
        history = ""
        while True:
            try:
                user = input("Voce: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not user:
                continue
            history += f"<|user|>\n{user}\n<|model|>\n"
            ids = torch.tensor([tokenizer.encode(history, add_special_tokens=False)],
                               device=device)
            out = model.generate(ids, max_new_tokens=args.max_tokens,
                                 temperature=args.temperature, top_p=args.top_p,
                                 stop_ids=stop_ids)
            reply_ids = out[0, ids.size(1):].tolist()
            reply = tokenizer.decode([t for t in reply_ids if t not in stop_ids]).strip()
            print(f"Arpa: {reply}\n")
            history += f"{reply}\n<|end|>\n"
    else:
        prompt = args.prompt or "O Brasil e"
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)],
                           device=device)
        out = model.generate(ids, max_new_tokens=args.max_tokens,
                             temperature=args.temperature, top_p=args.top_p,
                             stop_ids=stop_ids)
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
