#!/usr/bin/env python3
"""
finetune.py - SFT (Supervised Fine-Tuning) do modelo Arpa-30M.

Treina o modelo para seguir instrucoes/conversa em portugues usando
um dataset local de distilacao ou streaming do HuggingFace.

Loss calculado APENAS nas respostas (nao nas instrucoes).

Uso:
    python finetune.py                                      # treina com distill.json
    python finetune.py --epochs 1                           # so 1 epoch
    python finetune.py --data distill.json                  # dataset local
    python finetune.py --hf dominguesm/alpaca-data-pt-br    # streaming HuggingFace
    python finetune.py --hf dominguesm/alpaca-data-pt-br --hf_max 5000  # limita exemplos
    python finetune.py --base checkpoints-sft/best          # checkpoint inicial
    python finetune.py --out checkpoints-distill            # saida
"""

import os
import sys
import json
import time
import math
import shutil
import argparse
from pathlib import Path
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# Configuracoes
# ==============================================================================

TOKENIZER_DIR   = "./tokenizer-arpa-32k"
BASE_MODEL_DIR  = "./checkpoints-sft/best"
DATA_PATH       = "./distill.json"
CKPT_DIR        = "./checkpoints-distill"
BEST_DIR        = "./checkpoints-distill/best"

MODEL_CONFIG = dict(
    vocab_size=32_000,
    hidden_size=384,
    intermediate_size=1024,
    num_hidden_layers=11,
    num_attention_heads=8,
    num_key_value_heads=4,
    max_position_embeddings=1024,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    hidden_act="silu",
    attention_dropout=0.0,
    use_cache=False,
    tie_word_embeddings=True,
)

# Hiperparametros SFT
EPOCHS        = 3
BATCH_SIZE    = 8
MAX_SEQ_LEN   = 512       # instrucoes + respostas raramente passam disso
LR            = 2e-5      # bem mais baixo que pre-training (3e-4)
WARMUP_RATIO  = 0.03
GRAD_CLIP     = 1.0
SAVE_EVERY    = 500       # steps
EVAL_EVERY    = 500

# Template Alpaca
PROMPT_TEMPLATE = """### Instrucao:
{instruction}

### Entrada:
{input}

### Resposta:
"""

PROMPT_TEMPLATE_NO_INPUT = """### Instrucao:
{instruction}

### Resposta:
"""


# ==============================================================================
# Dataset
# ==============================================================================

def load_local_records(data_path, categories=None):
    """Carrega JSON/JSONL local e normaliza para instruction/input/output."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset local nao encontrado: {path}")

    if path.suffix.lower() == ".jsonl":
        raw = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    print(f"  Linha {line_no} ignorada: nao e objeto JSON")
                    continue
                raw.append(item)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Dataset .json precisa conter uma lista de objetos")

    records = []
    skipped = 0
    skipped_category = 0
    wanted_categories = set(categories or [])
    for item in raw:
        if not isinstance(item, dict):
            skipped += 1
            continue

        category = str(item.get("category", "")).strip()
        if wanted_categories and category not in wanted_categories:
            skipped_category += 1
            continue

        instruction = (
            item.get("instruction")
            or item.get("question")
            or item.get("prompt")
            or ""
        )
        inp = item.get("input", "")
        output = (
            item.get("output")
            or item.get("answer")
            or item.get("response")
            or ""
        )

        messages = item.get("messages")
        if messages and not (instruction and output):
            user_parts = []
            assistant_parts = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                if role == "user":
                    user_parts.append(content)
                elif role == "assistant":
                    assistant_parts.append(content)
            if not instruction and user_parts:
                instruction = "\n".join(user_parts)
            if not output and assistant_parts:
                output = assistant_parts[-1]

        instruction = str(instruction).strip()
        inp = str(inp).strip()
        output = str(output).strip()

        if not instruction or not output:
            skipped += 1
            continue

        records.append({
            "instruction": instruction,
            "input": inp,
            "output": output,
            "category": category,
        })

    if not records:
        raise ValueError(f"Nenhum exemplo valido encontrado em {path}")

    print(
        f"  Registros validos: {len(records)} | "
        f"Ignorados: {skipped} | Fora das categorias: {skipped_category}"
    )
    return records


def load_hf_records(hf_path, max_examples=None):
    """Carrega um dataset do HuggingFace em streaming (sem salvar em disco)."""
    from datasets import load_dataset

    print(f"  Streaming HuggingFace: {hf_path}")
    ds = load_dataset(hf_path, split="train", streaming=True)

    records = []
    skipped = 0
    for item in ds:
        instruction = (item.get("instruction") or item.get("question") or item.get("prompt") or "").strip()
        inp         = (item.get("input")  or "").strip()
        output      = (item.get("output") or item.get("answer") or item.get("response") or "").strip()

        # Suporte a formato messages (OpenAssistant, etc.)
        if not (instruction and output):
            messages = item.get("messages") or item.get("conversation") or []
            user_parts, asst_parts = [], []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role    = str(msg.get("role", msg.get("from", ""))).lower()
                content = str(msg.get("content", msg.get("value", ""))).strip()
                if not content:
                    continue
                if role in ("user", "human"):
                    user_parts.append(content)
                elif role in ("assistant", "gpt"):
                    asst_parts.append(content)
            if user_parts and not instruction:
                instruction = user_parts[0]
            if asst_parts and not output:
                output = asst_parts[-1]

        instruction = instruction.strip()
        output      = output.strip()

        if not instruction or not output:
            skipped += 1
            continue

        records.append({"instruction": instruction, "input": inp, "output": output, "category": ""})

        if max_examples and len(records) >= max_examples:
            break

    print(f"  Registros validos: {len(records)} | Ignorados: {skipped}")
    return records


def load_hf_multi(hf_paths, hf_max=None):
    """Carrega e mescla múltiplos datasets HuggingFace. hf_max aplica-se a cada um."""
    import random
    all_records = []
    for path in hf_paths:
        all_records.extend(load_hf_records(path, hf_max))
    random.shuffle(all_records)
    print(f"  Total mesclado: {len(all_records)} exemplos de {len(hf_paths)} datasets")
    return all_records


def split_records(records, split, categories=None):
    """Split deterministico 95/5, com validacao minima em datasets pequenos."""
    if categories:
        wanted_categories = set(categories)
        records = [x for x in records if x.get("category") in wanted_categories]

    rng = np.random.default_rng(42)
    order = rng.permutation(len(records)).tolist()
    records = [records[i] for i in order]

    if len(records) < 2:
        return records

    cut = int(len(records) * 0.95)
    cut = max(1, min(len(records) - 1, cut))
    if split == "train":
        return records[:cut]
    return records[cut:]


class SFTDataset(Dataset):
    """
    Carrega o dataset Alpaca PT e tokeniza cada exemplo.
    O loss e mascarado para zero nas instrucoes — o modelo aprende
    apenas a prever as respostas.
    """

    def __init__(self, tokenizer, data_path=None, split="train", max_len=512,
                 categories=None, hf_path=None, hf_max=None):
        self.tokenizer = tokenizer
        self.max_len   = max_len

        if hf_path:
            # Streaming HuggingFace — carrega uma vez, divide aqui
            cache_key = (tuple(hf_path) if isinstance(hf_path, list) else hf_path, hf_max)
            if not hasattr(SFTDataset, "_hf_cache") or SFTDataset._hf_cache_key != cache_key:
                if isinstance(hf_path, list) and len(hf_path) > 1:
                    SFTDataset._hf_cache = load_hf_multi(hf_path, hf_max)
                else:
                    path = hf_path[0] if isinstance(hf_path, list) else hf_path
                    SFTDataset._hf_cache = load_hf_records(path, hf_max)
                SFTDataset._hf_cache_key = cache_key
            records = split_records(SFTDataset._hf_cache, split)
        else:
            print(f"Carregando dataset local ({split}): {data_path}")
            if categories:
                print(f"  Categorias: {', '.join(categories)}")
            records = split_records(load_local_records(data_path, categories), split)

        print(f"  {len(records)} exemplos ({split})")
        self.examples = self._tokenize_all(records)

    def _tokenize_all(self, ds):
        examples = []
        skipped  = 0

        for item in ds:
            instruction = item.get("instruction", "").strip()
            inp         = item.get("input", "").strip()
            output      = item.get("output", "").strip()

            if not instruction or not output:
                skipped += 1
                continue

            # Monta o prompt
            if inp:
                prompt = PROMPT_TEMPLATE.format(instruction=instruction, input=inp)
            else:
                prompt = PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction)

            # Tokeniza separado para saber onde começa a resposta
            prompt_ids   = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(output, add_special_tokens=False)

            # BOS no inicio, <|stop|> no fim da resposta (sinaliza fim de turno)
            # EOS fica reservado para fim de documento no pre-treino
            bos      = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id else []
            stop_id  = self.tokenizer.convert_tokens_to_ids("<|stop|>")
            stop     = [stop_id] if stop_id is not None else [self.tokenizer.eos_token_id]

            input_ids = bos + prompt_ids + response_ids + stop

            if len(input_ids) > self.max_len:
                # Trunca pela resposta (preserva instrucao)
                max_resp = self.max_len - len(bos) - len(prompt_ids) - len(stop)
                if max_resp < 10:
                    skipped += 1
                    continue
                response_ids = response_ids[:max_resp]
                input_ids    = bos + prompt_ids + response_ids + stop

            # Labels: -100 na instrucao (ignorado no loss), ids na resposta + stop
            n_prompt = len(bos) + len(prompt_ids)
            labels   = [-100] * n_prompt + response_ids + stop

            assert len(input_ids) == len(labels)

            examples.append({
                "input_ids": input_ids,
                "labels":    labels,
            })

        print(f"  Tokenizados: {len(examples)} | Ignorados: {skipped}")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels":    torch.tensor(ex["labels"],    dtype=torch.long),
        }


def collate_fn(batch):
    """Padding dinamico ate o maior exemplo do batch."""
    max_len    = max(x["input_ids"].shape[0] for x in batch)
    pad_id     = 0  # token de padding

    input_ids  = torch.full((len(batch), max_len), pad_id,  dtype=torch.long)
    labels     = torch.full((len(batch), max_len), -100,    dtype=torch.long)
    attn_mask  = torch.zeros((len(batch), max_len),          dtype=torch.long)

    for i, ex in enumerate(batch):
        n = ex["input_ids"].shape[0]
        input_ids[i, :n] = ex["input_ids"]
        labels[i,    :n] = ex["labels"]
        attn_mask[i, :n] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


# ==============================================================================
# Carregamento do modelo
# ==============================================================================

def load_base_model(base_dir, tokenizer, device):
    config = LlamaConfig(
        **MODEL_CONFIG,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config)
    model = model.from_pretrained(str(base_dir), config=config)
    model = model.to(device)

    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Modelo carregado: {n:.1f}M params | device={device}")
    return model


def save_checkpoint(model, tokenizer, step, val_loss, is_best=False):
    ckpt_path = Path(CKPT_DIR) / f"step_{step}"
    model.save_pretrained(str(ckpt_path))
    info = {
        "step":      step,
        "val_loss":  f"{val_loss:.4f}",
        "timestamp": datetime.now().isoformat(),
    }
    (ckpt_path / "info.json").write_text(json.dumps(info, indent=2))

    if is_best:
        if Path(BEST_DIR).exists():
            shutil.rmtree(BEST_DIR)
        shutil.copytree(str(ckpt_path), BEST_DIR)
        print(f"  >> [NOVO MELHOR] Val loss={val_loss:.4f} -> {BEST_DIR}")

    # Mantem so os 2 checkpoints mais recentes
    ckpts = sorted(
        [d for d in Path(CKPT_DIR).iterdir()
         if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    while len(ckpts) > 2:
        shutil.rmtree(ckpts.pop(0))


# ==============================================================================
# Avaliacao
# ==============================================================================

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tok  = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        # loss medio por token valido
        n_tok = (labels != -100).sum().item()
        if n_tok > 0:
            total_loss += out.loss.item() * n_tok
            total_tok  += n_tok

    model.train()
    return total_loss / max(total_tok, 1)


# ==============================================================================
# Geracao de teste
# ==============================================================================

@torch.no_grad()
def test_generation(model, tokenizer, device):
    model.eval()
    prompts = [
        "Oi, tudo bem?",
        "Nao era isso que eu quis dizer. Tenta de novo de um jeito mais simples.",
        "Pode ser mais curto e mais natural?",
    ]
    print("\n--- Geracao de Teste ---")
    for p in prompts:
        prompt_text = PROMPT_TEMPLATE_NO_INPUT.format(instruction=p)
        ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        out = model.generate(
            ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0][ids.shape[-1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"  [{p}]\n  -> {response[:200]}\n")
    model.train()
    print("--- Fim Geracao ---\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    global CKPT_DIR, BEST_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    type=str, default=BASE_MODEL_DIR)
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--data",    type=str, default=DATA_PATH)
    parser.add_argument("--out",     type=str, default=CKPT_DIR)
    parser.add_argument("--hf",      type=str, nargs="+", default=None,
                        help="Datasets HuggingFace em streaming (um ou mais), ex: --hf dominguesm/alpaca-data-pt-br nicholasKluge/Aira-Dataset")
    parser.add_argument("--hf_max", type=int, default=None,
                        help="Limite de exemplos do HuggingFace (None = todos)")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Filtra categorias do JSON local, ex: conversa correcao_redirecionamento")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--lr",      type=float, default=LR)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--device",  type=str, default="auto")
    args = parser.parse_args()

    CKPT_DIR = args.out
    BEST_DIR = str(Path(args.out) / "best")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    Path(CKPT_DIR).mkdir(exist_ok=True)

    # Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    # Garante que <|stop|> existe — modelos novos tem, antigos podem nao ter
    if tokenizer.convert_tokens_to_ids("<|stop|>") == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|stop|>"]})
        print("  [!] <|stop|> adicionado ao tokenizer (nao estava no vocab)")

    # Dataset
    print("Carregando datasets...")
    train_ds = SFTDataset(
        tokenizer, args.data, split="train",
        max_len=args.max_len, categories=args.categories,
        hf_path=args.hf, hf_max=args.hf_max,
    )
    val_ds   = SFTDataset(
        tokenizer, args.data, split="val",
        max_len=args.max_len, categories=args.categories,
        hf_path=args.hf, hf_max=args.hf_max,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Modelo
    base_dir = Path(args.resume) if args.resume else Path(args.base)
    model    = load_base_model(base_dir, tokenizer, device)
    model.train()

    # Otimizador
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # AMP
    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")
    global_step   = 0
    t0            = time.time()

    print(f"\nIniciando SFT: {args.epochs} epochs | {total_steps} steps | lr={args.lr}")
    print(f"Train: {len(train_ds)} ex | Val: {len(val_ds)} ex | Batch: {args.batch}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % 10 == 0:
                elapsed  = time.time() - t0
                lr_now   = scheduler.get_last_lr()[0]
                pct      = global_step / total_steps * 100
                print(
                    f"Step {global_step} ({pct:.1f}%) | "
                    f"loss={loss.item():.4f} ppl={math.exp(loss.item()):.1f} | "
                    f"lr={lr_now:.2e} | {elapsed:.0f}s",
                    flush=True,
                )

            if global_step % EVAL_EVERY == 0:
                print("Avaliando...")
                val_loss = evaluate(model, val_loader, device)
                val_ppl  = math.exp(val_loss)
                is_best  = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                print(f"  Val loss={val_loss:.4f} ppl={val_ppl:.1f}")
                save_checkpoint(model, tokenizer, global_step, val_loss, is_best)
                test_generation(model, tokenizer, device)

        # Salva ao fim de cada epoch
        print(f"Fim epoch {epoch} — avaliando...")
        val_loss = evaluate(model, val_loader, device)
        val_ppl  = math.exp(val_loss)
        is_best  = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        print(f"  Val loss={val_loss:.4f} ppl={val_ppl:.1f}")
        save_checkpoint(model, tokenizer, global_step, val_loss, is_best)
        test_generation(model, tokenizer, device)

    # Salva modelo final
    final_path = Path(CKPT_DIR) / "final"
    model.save_pretrained(str(final_path))
    print(f"\nSFT concluido! Melhor val loss: {best_val_loss:.4f}")
    print(f"Melhor modelo em: {BEST_DIR}")


if __name__ == "__main__":
    main()
