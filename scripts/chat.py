#!/usr/bin/env python3
"""
chat.py - Inferencia interativa com o modelo Arpa-30M.

Uso:
    python chat.py                          # modelo SFT (instruct)
    python chat.py --base                   # modelo base (completacao de texto)
    python chat.py --model checkpoints-sft/best
    python chat.py --prompt "Explique a fotossintese"
    python chat.py --temp 0.7 --top_p 0.9
"""

import sys
import argparse
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# Configuracoes
# ==============================================================================

TOKENIZER_DIR = "./tokenizer-arpa-32k"
SFT_DIR       = "./checkpoints-sft/best"
BASE_DIR      = "./checkpoints-arpa100m/best"

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


# ==============================================================================
# Carregamento
# ==============================================================================

def load_model(ckpt_path: Path, device: torch.device):
    print(f"Carregando: {ckpt_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    config = LlamaConfig(
        **MODEL_CONFIG,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config)
    model = model.from_pretrained(str(ckpt_path), config=config)
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Modelo: {n:.1f}M params | device={device}")
    return model, tokenizer


# ==============================================================================
# Geracao
# ==============================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    device: torch.device = torch.device("cpu"),
) -> str:
    # Para em <|stop|> (fim de resposta SFT) ou EOS (fim de documento)
    stop_id = tokenizer.convert_tokens_to_ids("<|stop|>")
    stop_ids = [tokenizer.eos_token_id]
    if stop_id is not None and stop_id != tokenizer.unk_token_id:
        stop_ids.append(stop_id)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
    )
    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ==============================================================================
# Loop interativo
# ==============================================================================

def chat_loop(model, tokenizer, device, args, instruct_mode: bool):
    mode_label = "Instruct" if instruct_mode else "Base (completacao de texto)"
    print("\n" + "=" * 58)
    print(f"  Arpa-30M — {mode_label}")
    print("=" * 58)
    print(f"  temp={args.temp}  top_p={args.top_p}  max={args.max_tokens}")
    print("  Comandos: /temp X  /top_p X  /max X  /sair")
    if instruct_mode:
        print("  Digite sua pergunta ou instrucao diretamente.")
    print("=" * 58 + "\n")

    temp    = args.temp
    top_p   = args.top_p
    max_tok = args.max_tokens

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte logo.")
            break

        if not user_input:
            continue

        if user_input == "/sair":
            print("Ate logo.")
            break
        if user_input.startswith("/temp "):
            temp = float(user_input.split()[1])
            print(f"  temperatura -> {temp}")
            continue
        if user_input.startswith("/top_p "):
            top_p = float(user_input.split()[1])
            print(f"  top_p -> {top_p}")
            continue
        if user_input.startswith("/max "):
            max_tok = int(user_input.split()[1])
            print(f"  max_tokens -> {max_tok}")
            continue

        # Monta o prompt
        if instruct_mode:
            prompt = INSTRUCT_TEMPLATE.format(instruction=user_input)
        else:
            prompt = user_input

        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_tok,
            temperature=temp,
            top_p=top_p,
            device=device,
        )

        # <|stop|> já interrompe a geração — nenhum corte manual necessário

        print(f"\n{response}\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str,   default=None,
                        help="Caminho do checkpoint (default: SFT best)")
    parser.add_argument("--base",       action="store_true",
                        help="Usar modelo base em vez do SFT")
    parser.add_argument("--instruct",   action="store_true",
                        help="Forcar modo instruct independente do nome do checkpoint")
    parser.add_argument("--prompt",     type=str,   default=None,
                        help="Geracao direta sem loop")
    parser.add_argument("--temp",       type=float, default=0.7)
    parser.add_argument("--top_p",      type=float, default=0.9)
    parser.add_argument("--top_k",      type=int,   default=50)
    parser.add_argument("--max_tokens", type=int,   default=300)
    parser.add_argument("--device",     type=str,   default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    # Resolve qual checkpoint usar
    if args.model:
        ckpt_path = Path(args.model)
    elif args.base:
        ckpt_path = Path(BASE_DIR)
    else:
        ckpt_path = Path(SFT_DIR) if Path(SFT_DIR).exists() else Path(BASE_DIR)

    instruct_mode = args.instruct or (not args.base and "sft" in str(ckpt_path).lower())

    model, tokenizer = load_model(ckpt_path, device)

    if args.prompt:
        prompt = INSTRUCT_TEMPLATE.format(instruction=args.prompt) if instruct_mode else args.prompt
        out = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
        # <|stop|> já interrompe a geração no generate()
        print(out)
        return

    chat_loop(model, tokenizer, device, args, instruct_mode)


if __name__ == "__main__":
    main()
