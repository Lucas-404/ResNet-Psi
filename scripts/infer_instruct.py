#!/usr/bin/env python3
"""
infer_instruct.py - inferencia batch do modelo instruct.

Exemplos:
    python infer_instruct.py
    python infer_instruct.py --question "Explique a fotossintese"
    python infer_instruct.py --question "O que e IA?" --question "Qual e a capital do Brasil?"
    python infer_instruct.py --questions perguntas.txt --output respostas.jsonl

Arquivo .txt:
    uma pergunta por linha

Arquivo .jsonl:
    {"question": "..."}
    {"instruction": "...", "input": "..."}
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


TOKENIZER_DIR = "./tokenizer-arpa-32k"
SFT_DIR = "./checkpoints-sft/best"

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
    use_cache=True,
    tie_word_embeddings=True,
)

INSTRUCT_TEMPLATE = """### Instrucao:
{instruction}

### Resposta:
"""

INSTRUCT_TEMPLATE_WITH_INPUT = """### Instrucao:
{instruction}

### Entrada:
{input}

### Resposta:
"""


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(model_path: Path, tokenizer_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer nao encontrado: {tokenizer_path}")

    print(f"Carregando tokenizer: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    config = LlamaConfig(
        **MODEL_CONFIG,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
    )

    print(f"Carregando modelo: {model_path}")
    model = LlamaForCausalLM(config)
    model = model.from_pretrained(str(model_path), config=config)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Modelo: {n_params:.1f}M params | device={device}\n")
    return model, tokenizer


def format_prompt(question: str, input_text: str = "") -> str:
    question = question.strip()
    input_text = input_text.strip()
    if input_text:
        return INSTRUCT_TEMPLATE_WITH_INPUT.format(instruction=question, input=input_text)
    return INSTRUCT_TEMPLATE.format(instruction=question)


def clean_answer(text: str) -> str:
    text = text.strip()

    for marker in (
        "\n### Instrucao:",
        "\n### Entrada:",
        "\n### Resposta:",
        "\nInstruction:",
        "\nInput:",
        "\nResponse:",
    ):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    return text


@torch.no_grad()
def generate_answer(model, tokenizer, device: torch.device, question: str, input_text: str, args) -> str:
    prompt = format_prompt(question, input_text)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    do_sample = args.temp > 0
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_tokens,
        do_sample=do_sample,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        kwargs.update(temperature=args.temp, top_p=args.top_p, top_k=args.top_k)

    output = model.generate(**kwargs)
    new_tokens = output[0][input_ids.shape[-1]:]
    return clean_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))


def load_questions(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de perguntas nao encontrado: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Linha {line_no} nao e um objeto JSON")
                question = obj.get("question") or obj.get("instruction") or obj.get("prompt")
                if not question:
                    raise ValueError(f"Linha {line_no} sem question/instruction/prompt")
                items.append({
                    "question": str(question).strip(),
                    "input": str(obj.get("input", "")).strip(),
                })
        return items

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Arquivo .json precisa conter uma lista")
        items = []
        for idx, obj in enumerate(data, 1):
            if isinstance(obj, str):
                items.append({"question": obj.strip(), "input": ""})
                continue
            if not isinstance(obj, dict):
                raise ValueError(f"Item {idx} nao e string nem objeto")
            question = obj.get("question") or obj.get("instruction") or obj.get("prompt")
            if not question:
                raise ValueError(f"Item {idx} sem question/instruction/prompt")
            items.append({
                "question": str(question).strip(),
                "input": str(obj.get("input", "")).strip(),
            })
        return items

    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append({"question": line, "input": ""})
    return items


def build_questions(args) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    if args.questions:
        items.extend(load_questions(Path(args.questions)))

    for question in args.question or []:
        question = question.strip()
        if question:
            items.append({"question": question, "input": ""})

    if args.limit is not None:
        items = items[:args.limit]

    return items


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_answer(output_path: Path | None, question: str, input_text: str, answer: str, args) -> None:
    if not output_path:
        return
    append_jsonl(output_path, {
        "question": question,
        "input": input_text,
        "answer": answer,
        "model": args.model,
        "created_at": datetime.now().isoformat(),
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    })


def run_one(model, tokenizer, device: torch.device, item: Dict[str, str], args, idx: int, total: int | None) -> str:
    question = item["question"]
    input_text = item.get("input", "")

    print("=" * 78)
    if total is None:
        print("Pergunta:")
    else:
        print(f"[{idx}/{total}] Pergunta:")
    print(question)
    if input_text:
        print("\nEntrada:")
        print(input_text)

    answer = generate_answer(model, tokenizer, device, question, input_text, args)

    print("\nResposta:")
    print(answer if answer else "[vazia]")
    print()
    return answer


def interactive_loop(model, tokenizer, device: torch.device, output_path: Path | None, args) -> None:
    print("Modo interativo. Digite uma pergunta por vez.")
    print("Comandos: /sair para encerrar, /multi para colar pergunta com varias linhas.\n")

    while True:
        try:
            question = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrado.")
            return

        if not question:
            continue
        if question.lower() in {"/sair", "/quit", "q"}:
            print("Encerrado.")
            return
        if question == "/multi":
            print("Cole a pergunta. Termine com uma linha contendo apenas ponto: .")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == ".":
                    break
                lines.append(line)
            question = "\n".join(lines).strip()
            if not question:
                continue

        item = {"question": question, "input": ""}
        answer = run_one(model, tokenizer, device, item, args, idx=1, total=None)
        save_answer(output_path, question, "", answer, args)


def main() -> None:
    configure_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=SFT_DIR,
                        help="Checkpoint instruct (default: checkpoints-sft/best)")
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_DIR)
    parser.add_argument("--questions", type=str, default=None,
                        help="Arquivo .txt, .json ou .jsonl com perguntas")
    parser.add_argument("--question", action="append", default=None,
                        help="Pergunta direta. Pode repetir varias vezes.")
    parser.add_argument("--output", type=str, default=None,
                        help="Opcional: salva respostas em JSONL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    args = parser.parse_args()

    set_seed(args.seed)
    questions = build_questions(args)
    device = resolve_device(args.device)
    model, tokenizer = load_model(Path(args.model), Path(args.tokenizer), device)

    output_path = Path(args.output) if args.output else None
    if output_path:
        print(f"Saida JSONL: {output_path}\n")

    if not questions:
        interactive_loop(model, tokenizer, device, output_path, args)
        return

    for idx, item in enumerate(questions, 1):
        answer = run_one(model, tokenizer, device, item, args, idx=idx, total=len(questions))
        save_answer(output_path, item["question"], item.get("input", ""), answer, args)


if __name__ == "__main__":
    main()
