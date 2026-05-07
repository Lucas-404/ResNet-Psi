#!/usr/bin/env python3
"""
gerar_chat.py - gera um arquivo de conversa a partir de varias perguntas.

Uso simples:
    python gerar_chat.py

Por padrao:
    entrada: perguntas.txt
    saida:   chat_gerado.md

Formato de perguntas.txt:
    uma pergunta por linha
    linhas vazias e linhas iniciadas com # sao ignoradas
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from infer_instruct import (
    SFT_DIR,
    TOKENIZER_DIR,
    generate_answer,
    load_model,
    resolve_device,
    set_seed,
)


DEFAULT_INPUT = "perguntas.txt"
DEFAULT_OUTPUT = "chat_gerado.md"


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de perguntas nao encontrado: {path}\n"
            "Crie um perguntas.txt com uma pergunta por linha, ou passe --input outro_arquivo.txt"
        )

    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(line)

    if not questions:
        raise ValueError(f"Nenhuma pergunta encontrada em {path}")

    return questions


def write_header(path: Path, args, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Chat Gerado\n\n")
        f.write(f"- Modelo: `{args.model}`\n")
        f.write(f"- Perguntas: `{args.input}`\n")
        f.write(f"- Total: {total}\n")
        f.write(f"- Criado em: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Temperatura: {args.temp}\n")
        f.write(f"- Max tokens: {args.max_tokens}\n\n")


def append_turn(path: Path, idx: int, question: str, answer: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"## Turno {idx}\n\n")
        f.write("**Usuario:**\n\n")
        f.write(question.strip() + "\n\n")
        f.write("**Modelo:**\n\n")
        f.write((answer.strip() or "[vazia]") + "\n\n")


def append_jsonl(path: Path, idx: int, question: str, answer: str, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "turn": idx,
        "question": question,
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
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    configure_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Arquivo com uma pergunta por linha")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Arquivo de chat gerado em Markdown")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Opcional: tambem salva pares pergunta/resposta em JSONL")
    parser.add_argument("--model", type=str, default=SFT_DIR)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    args = parser.parse_args()

    set_seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl) if args.jsonl else None

    questions = load_questions(input_path)
    device = resolve_device(args.device)
    model, tokenizer = load_model(Path(args.model), Path(args.tokenizer), device)

    write_header(output_path, args, len(questions))
    if jsonl_path and jsonl_path.exists():
        jsonl_path.unlink()

    print(f"Perguntas: {len(questions)}")
    print(f"Gerando chat em: {output_path}")
    if jsonl_path:
        print(f"Gerando JSONL em: {jsonl_path}")
    print()

    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] {question}")
        answer = generate_answer(model, tokenizer, device, question, "", args)
        append_turn(output_path, idx, question, answer)
        if jsonl_path:
            append_jsonl(jsonl_path, idx, question, answer, args)
        print((answer[:160] + "...") if len(answer) > 160 else answer)
        print()

    print(f"Pronto: {output_path}")


if __name__ == "__main__":
    main()
