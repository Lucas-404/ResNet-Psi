"""
train_tokenizer.py - Treina o tokenizer BPE byte-level 64K limpo (PT/EN/codigo).

Amostra ~2GB de texto do mesmo mix do pre-treino e treina BPE com os
tokens especiais do chat (<|user|>, <|model|>, <|end|>).

Uso:
    python arpa/train_tokenizer.py
    python arpa/train_tokenizer.py --vocab-size 64000 --target-bytes 2e9
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|pad|>",
    "<|user|>",
    "<|model|>",
    "<|end|>",
    "<|reasoning|>",
    "<|conversation|>",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument("--target-bytes", type=float, default=2e9)
    parser.add_argument("--output-dir", default="tokenizer-arpa-64k-clean")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.output_dir, "tokenizer.json")) and not args.force:
        sys.exit(f"Tokenizer ja existe em {args.output_dir}. Use --force.")

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from arpa.prepare_data import build_stream, clean_text, is_quality_text, MIN_TEXT_LENGTH

    # 1. Amostra corpus para arquivo temporario
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt",
                                      delete=False, dir=".")
    print(f"[1/3] Amostrando {args.target_bytes / 1e9:.1f}GB de texto -> {tmp.name}")
    written = 0
    stream = build_stream(seed=7)
    for sample in stream:
        text, kind = sample["text"], sample["kind"]
        if not text or len(text) < MIN_TEXT_LENGTH:
            continue
        text = clean_text(text[:50_000], kind)
        if len(text) < MIN_TEXT_LENGTH or not is_quality_text(text, kind):
            continue
        tmp.write(text + "\n")
        written += len(text.encode("utf-8", errors="ignore"))
        if written >= args.target_bytes:
            break
    tmp.close()
    print(f"  {written / 1e9:.2f}GB escritos")

    # 2. Treina BPE byte-level
    print(f"[2/3] Treinando BPE vocab={args.vocab_size}...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([tmp.name], trainer)
    os.unlink(tmp.name)

    # 3. Salva como tokenizer HF
    print(f"[3/3] Salvando em {args.output_dir}/")
    from transformers import PreTrainedTokenizerFast
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|pad|>",
        model_max_length=8192,
        additional_special_tokens=SPECIAL_TOKENS[3:],
    )
    fast.save_pretrained(args.output_dir)

    # Teste rapido
    for frase in ["O Brasil e o maior pais da America do Sul.",
                  "def soma(a, b):\n    return a + b",
                  "<|user|>\nOi!\n<|model|>\nOla!<|end|>"]:
        ids = fast.encode(frase)
        print(f"  {len(ids):>3} tokens | {frase[:50]!r}")
    print("Pronto.")


if __name__ == "__main__":
    main()
