#!/usr/bin/env python3
"""
train_tokenizer.py - Treinar tokenizer BPE do zero para PT-BR

Coleta ~500MB de texto portugues brasileiro via streaming de
Wikipedia PT + C4 PT, treina um tokenizer BPE de 32K tokens e salva
em formato compativel com LlamaTokenizerFast.

Uso:
    python train_tokenizer.py
"""

import os
import sys
import time
import tempfile
import shutil

# ==============================================================================
# Configuracoes
# ==============================================================================

VOCAB_SIZE = 32_000
OUTPUT_DIR = "./tokenizer-arpa-32k"
TARGET_BYTES = 3 * 1024 * 1024 * 1024  # 3GB de texto — base sólida para merges BPE
MIN_TEXT_LENGTH = 100  # caracteres minimos por texto

# Mix espelha o DATASETS_CONFIG do nano_pt.py:
# PT base para vocabulário nativo + código para merges técnicos
DATASETS = [
    # (nome, config, peso, campo_texto)
    ("wikimedia/wikipedia", "20231101.pt", 0.40, "text"),    # PT limpo, factual
    ("TucanoBR/GigaVerbo",  None,          0.35, "text"),    # diversidade PT-BR
    ("codeparrot/codeparrot-clean", None,    0.25, "content"), # código — merges técnicos
]

# Tokens especiais (estilo LLaMA 3)
SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|pad|>",
    "<|stop|>",       # fim de resposta no SFT — distinto do EOS de documento
]

BOS_TOKEN  = "<|begin_of_text|>"
EOS_TOKEN  = "<|end_of_text|>"
PAD_TOKEN  = "<|pad|>"
STOP_TOKEN = "<|stop|>"


def collect_texts(tmp_file_path: str) -> int:
    """
    Stream textos PT-BR de multiplos datasets e salva em arquivo temporario.
    Retorna o numero de textos coletados.
    """
    from datasets import load_dataset, interleave_datasets

    print(f"[1/3] Coletando ~{TARGET_BYTES // (1024*1024)}MB de texto PT-BR...")
    print(f"      Salvando em: {tmp_file_path}")

    # Carregar cada dataset
    streams = []
    weights = []
    for ds_name, ds_config, weight, text_field in DATASETS:
        try:
            ds = load_dataset(ds_name, ds_config, split="train", streaming=True)
            streams.append(ds)
            weights.append(weight)
            print(f"      [OK] {ds_name} ({ds_config}) peso={weight}")
        except Exception as e:
            print(f"      [ERRO] {ds_name}: {e}")

    if not streams:
        print("ERRO: Nenhum dataset carregado!")
        sys.exit(1)

    # Normalizar pesos e intercalar
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    dataset = interleave_datasets(streams, probabilities=weights, seed=42)

    total_bytes = 0
    num_texts = 0
    start_time = time.time()

    with open(tmp_file_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            # Suporta campo "text" (Wikipedia/GigaVerbo) e "content" (código)
            text = sample.get("text") or sample.get("content", "")

            if not text or len(text) < MIN_TEXT_LENGTH:
                continue

            f.write(text + "\n")
            total_bytes += len(text.encode("utf-8"))
            num_texts += 1

            if num_texts % 10_000 == 0:
                elapsed = time.time() - start_time
                mb_collected = total_bytes / (1024 * 1024)
                speed = mb_collected / elapsed if elapsed > 0 else 0
                pct = (total_bytes / TARGET_BYTES) * 100
                print(
                    f"      {num_texts:>8} textos | "
                    f"{mb_collected:>7.1f}MB / {TARGET_BYTES // (1024*1024)}MB "
                    f"({pct:.1f}%) | "
                    f"{speed:.1f} MB/s"
                )

            if total_bytes >= TARGET_BYTES:
                break

    elapsed = time.time() - start_time
    mb_final = total_bytes / (1024 * 1024)
    print(f"      Concluido: {num_texts} textos, {mb_final:.1f}MB em {elapsed:.0f}s")
    return num_texts


def train_tokenizer(tmp_file_path: str):
    """
    Treina tokenizer BPE usando a biblioteca tokenizers (backend Rust).
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

    print(f"\n[2/3] Treinando tokenizer BPE (vocab_size={VOCAB_SIZE})...")

    # Criar tokenizer BPE
    tokenizer = Tokenizer(models.BPE(unk_token=None))

    # Normalizer: NFC — normaliza acentos PT sem perder informação
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenizer: Sequence de duas regras
    #   1. Digits — separa dígitos individuais: "2024" → ["2","0","2","4"]
    #      Isso faz o modelo aprender aritmética e datas corretamente,
    #      e evita tokens esquisitos como "2024" que nunca generalizam.
    #   2. ByteLevel — converte tudo para bytes (como GPT-2/LLaMA),
    #      garantindo cobertura total do unicode sem UNK tokens.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    # Decoder: ByteLevel — inverte o ByteLevel do pre-tokenizer
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,           # merge só se aparece pelo menos 2x
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Treinar
    start_time = time.time()
    tokenizer.train(files=[tmp_file_path], trainer=trainer)
    elapsed = time.time() - start_time

    print(f"      Tokenizer treinado em {elapsed:.0f}s")
    print(f"      Vocabulario final: {tokenizer.get_vocab_size()} tokens")

    return tokenizer


def save_as_llama_tokenizer(tokenizer):
    """
    Converte o tokenizer treinado para formato LlamaTokenizerFast e salva.
    """
    from transformers import PreTrainedTokenizerFast

    print(f"\n[3/3] Salvando tokenizer em {OUTPUT_DIR}/ ...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Salvar tokenizer.json intermediario
    tmp_tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer.save(tmp_tokenizer_path)

    # Carregar como PreTrainedTokenizerFast (compativel com LLaMA)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tmp_tokenizer_path,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        model_max_length=1024,
        clean_up_tokenization_spaces=False,
    )

    # Salvar
    fast_tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"      Arquivos salvos em {OUTPUT_DIR}/")

    return fast_tokenizer


def test_tokenizer(tokenizer):
    """Testa o tokenizer com exemplos em portugues."""
    print("\n" + "=" * 60)
    print("Teste do Tokenizer")
    print("=" * 60)

    test_texts = [
        "Olá, como você está?",
        "O Brasil é o maior país da América do Sul.",
        "A inteligência artificial está transformando o mundo.",
        "Programação em Python é muito divertida!",
        "O gato sentou no telhado e observou a lua cheia.",
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        token_strs = tokenizer.convert_ids_to_tokens(tokens)

        print(f"\nTexto:    '{text}'")
        print(f"Tokens:   {len(tokens)} ids")
        print(f"Token strs: {token_strs[:15]}{'...' if len(token_strs) > 15 else ''}")
        print(f"Decoded:  '{decoded}'")

    # Testar tokens especiais
    print(f"\n--- Tokens Especiais ---")
    print(f"BOS token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"Vocab size: {tokenizer.vocab_size}")


def main():
    print("=" * 60)
    print("Arpa-160M: Treinamento de Tokenizer BPE PT-BR")
    print(f"Vocab size: {VOCAB_SIZE} | Target: {TARGET_BYTES // (1024*1024)}MB")
    print("=" * 60)

    # Verificar se ja existe
    if os.path.exists(os.path.join(OUTPUT_DIR, "tokenizer.json")):
        print(f"\nTokenizer ja existe em {OUTPUT_DIR}/")
        resp = input("Deseja re-treinar? (s/N): ").strip().lower()
        if resp != "s":
            print("Usando tokenizer existente.")
            from transformers import PreTrainedTokenizerFast
            tok = PreTrainedTokenizerFast.from_pretrained(OUTPUT_DIR)
            test_tokenizer(tok)
            return

    # Criar arquivo temporario para textos
    tmp_dir = tempfile.mkdtemp(prefix="arpa_tokenizer_")
    tmp_file = os.path.join(tmp_dir, "corpus_ptbr.txt")

    try:
        # 1. Coletar textos
        num_texts = collect_texts(tmp_file)

        if num_texts < 1000:
            print(f"ERRO: Poucos textos coletados ({num_texts}). Verifique conexao.")
            sys.exit(1)

        # 2. Treinar tokenizer
        raw_tokenizer = train_tokenizer(tmp_file)

        # 3. Salvar como LlamaTokenizerFast
        fast_tokenizer = save_as_llama_tokenizer(raw_tokenizer)

        # 4. Testar
        test_tokenizer(fast_tokenizer)

        print("\n" + "=" * 60)
        print("Tokenizer treinado e salvo com sucesso!")
        print(f"Diretorio: {os.path.abspath(OUTPUT_DIR)}")
        print("=" * 60)

    finally:
        # Limpar arquivo temporario
        print(f"\nLimpando arquivos temporarios ({tmp_dir})...")
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()