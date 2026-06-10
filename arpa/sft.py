"""
sft.py - Supervised fine-tuning no formato <|user|>/<|model|>/<|end|>.

Loss mascarado: o modelo so aprende os tokens das respostas (entre
<|model|> e <|end|>, inclusive o <|end|>). Prompt e padding ficam em -100.

Dados: jsonl com {"text": "<|user|>\\n...\\n<|model|>\\n...\\n<|end|>"}
Multi-turno funciona: cada span <|model|>...<|end|> recebe loss.

Uso:
    python arpa/sft.py --init checkpoints-arpa150m/best.pt --data sft/*.jsonl
    python arpa/sft.py --init best.pt --data sft/chat.jsonl --epochs 3 --lr 2e-5
"""

import argparse
import json
import math
import os
import random
import sys
import time
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from arpa.config import ModelConfig
from arpa.model import Arpa


def load_examples(patterns):
    texts = []
    for pattern in patterns:
        for path in sorted(glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("text"):
                        texts.append(obj["text"])
    return texts


def tokenize_masked(text, tokenizer, model_id, end_id, max_len):
    """Retorna (ids, labels): labels=-100 fora dos spans de resposta."""
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
    labels = [-100] * len(ids)
    in_answer = False
    for i, tok in enumerate(ids):
        if tok == model_id:
            in_answer = True
            continue  # o proprio <|model|> nao recebe loss
        if in_answer:
            labels[i] = ids[i]
        if tok == end_id:
            in_answer = False
    return ids, labels


def make_batch(examples, pad_id, device):
    max_len = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(examples):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(labels)] = torch.tensor(labels)
    # y deslocado: prever o proximo token
    return x[:, :-1].to(device), y[:, 1:].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True, help="checkpoint do pre-treino (.pt)")
    parser.add_argument("--data", nargs="+", required=True, help="jsonl(s) de SFT")
    parser.add_argument("--tokenizer-dir", default="tokenizer-arpa-64k-clean")
    parser.add_argument("--out-dir", default="checkpoints-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model_id = tokenizer.convert_tokens_to_ids("<|model|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    pad_id = tokenizer.pad_token_id
    assert model_id >= 0 and end_id >= 0, "tokenizer sem <|model|>/<|end|>"

    # Modelo do checkpoint de pre-treino
    state = torch.load(args.init, map_location=device, weights_only=False)
    mcfg = ModelConfig(**state["config"]["model"])
    model = Arpa(mcfg).to(device)
    model.load_state_dict(state["model"])
    print(f"Modelo carregado: {model.num_params() / 1e6:.1f}M params "
          f"(step pre-treino {state.get('step')})")

    texts = load_examples(args.data)
    print(f"{len(texts)} exemplos de {args.data}")
    examples = [tokenize_masked(t, tokenizer, model_id, end_id, args.max_len)
                for t in texts]
    examples = [e for e in examples if any(l != -100 for l in e[1])]
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val_set, train_set = examples[:n_val], examples[n_val:]
    print(f"train={len(train_set)} val={len(val_set)}")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 \
        else __import__("contextlib").nullcontext

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.0,
                                  fused=(device == "cuda"))
    steps_per_epoch = math.ceil(len(train_set) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 * args.lr + 0.45 * args.lr * (1 + math.cos(math.pi * min(p, 1.0)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for i in range(0, len(val_set), args.batch_size):
            x, y = make_batch(val_set[i:i + args.batch_size], pad_id, device)
            with autocast():
                losses.append(model(x, y).item())
        model.train()
        return sum(losses) / len(losses)

    best_val = float("inf")
    step = 0
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train_set)
        t0 = time.time()
        for i in range(0, len(train_set), args.batch_size):
            lr = lr_at(step)
            for g in optimizer.param_groups:
                g["lr"] = lr
            x, y = make_batch(train_set[i:i + args.batch_size], pad_id, device)
            with autocast():
                loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch + 1} step {step}/{total_steps} | "
                      f"loss {loss.item():.4f} | lr {lr:.2e} | "
                      f"{(time.time() - t0):.0f}s", flush=True)

        val_loss = evaluate()
        print(f"[epoch {epoch + 1}] val_loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            path = os.path.join(args.out_dir, "best.pt")
            torch.save({"model": model.state_dict(), "config": state["config"],
                        "sft_epoch": epoch + 1, "val_loss": val_loss}, path)
            print(f"  [best] salvo em {path}")

    torch.save({"model": model.state_dict(), "config": state["config"],
                "sft_epoch": args.epochs, "val_loss": best_val},
               os.path.join(args.out_dir, "final.pt"))
    print(f"SFT completo. best val_loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
