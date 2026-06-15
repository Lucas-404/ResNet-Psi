"""
prepare_data.py - Gera os bins de pre-treino a partir do mix de datasets reais.

Mix (streaming HuggingFace):
    25% Wikipedia PT          25% Wikipedia EN
    18% codigo Python limpo   15% open-web-math
     5% StackMathQA            7% OpenAssistant     5% Alpaca PT-BR

Saida:
    data/arpa150m/train_tokens.bin   (uint16, ~3.3B tokens p/ orcamento 3.2B)
    data/arpa150m/val_tokens.bin     (uint16, 5M tokens)

Uso:
    python arpa/prepare_data.py                          # alvo padrao 3.3B
    python arpa/prepare_data.py --train-tokens 100e6     # smoke test
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Estabilidade de download da HF: desliga o backend Xet (origem dos 408/timeout
# vistos no CDN) e aumenta o timeout. setdefault: respeita override do usuario.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from arpa.data import BinWriter

# (hf_name, hf_config, peso, campo, kind)
DATASETS_CONFIG = [
    ("wikimedia/wikipedia", "20231101.pt", 0.25, "text", "text"),
    ("wikimedia/wikipedia", "20231101.en", 0.25, "text", "text"),
    ("codeparrot/codeparrot-clean", None, 0.18, "content", "code"),
    ("open-web-math/open-web-math", None, 0.15, "text", "reasoning"),
    ("math-ai/StackMathQA", None, 0.05, "Q", "reasoning"),
    ("OpenAssistant/oasst1", None, 0.07, "text", "conversation"),
    ("dominguesm/alpaca-data-pt-br", None, 0.05, "instruction", "conversation"),
]

MIN_TEXT_LENGTH = 200
MAX_TEXT_LENGTH = 50_000
REASONING_TOKEN = "<|reasoning|>"
CONVERSATION_TOKEN = "<|conversation|>"

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
RE_WIKI_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")
RE_HTML_TAG = re.compile(r"<[^>]{1,80}>")
RE_MULTI_NL = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

WIKI_STOP_SECTIONS = ("Ver também", "Ligações externas", "Referências",
                      "Bibliografia", "References", "External links", "See also")


def clean_text(text: str, kind: str) -> str:
    if kind == "code":
        return text.strip()
    # Corta seções de rodape da Wikipedia
    for marker in WIKI_STOP_SECTIONS:
        pos = text.find(marker)
        if pos > 200:
            text = text[:pos]
    text = RE_MD_LINK.sub(r"\1", text)
    text = RE_WIKI_LINK.sub(r"\1", text)
    for _ in range(3):  # templates aninhados {{...{{...}}...}}: ate 3 niveis
        new = RE_WIKI_TEMPLATE.sub(" ", text)
        if new == text:
            break
        text = new
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NL.sub("\n\n", text)
    return text.strip()


def is_quality_text(text: str, kind: str) -> bool:
    from collections import Counter
    n = len(text)
    if n == 0:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    words = text.split()

    if kind == "code":
        if len(lines) < 8:
            return False
        markers = ("def ", "class ", "import ", "from ", "return ", "if ",
                   "for ", "while ", "try:", "except ")
        hits = sum(1 for l in lines if l.lstrip().startswith(markers))
        if hits < 2:
            return False
        printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
        if printable / n < 0.98:
            return False
        top = Counter(lines).most_common(1)[0][1]
        return top / len(lines) <= 0.3

    if kind == "reasoning":
        if len(words) < 20:
            return False
        alpha = sum(1 for c in text if c.isalpha())
        mathc = sum(1 for c in text if c.isdigit() or c in "+-=*/^_()[]{}<>")
        if (alpha + mathc) / n < 0.45:
            return False
    elif kind == "conversation":
        if len(words) < 12:
            return False
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / n < 0.45:
            return False
    else:  # texto generico (wiki)
        # markup residual (inclusive marcadores de fechamento orfaos) = descarta
        if any(m in text for m in ("{{", "}}", "[[", "]]", "<ref")):
            return False
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / n < 0.70:
            return False
        digits = sum(1 for c in text if c.isdigit())
        if digits / n > 0.10:
            return False
        if len(words) < 50:
            return False
        avg_w = sum(len(w) for w in words) / len(words)
        if avg_w < 3.0 or avg_w > 15.0:
            return False
        short = sum(1 for w in words if len(w) <= 2)
        if short / len(words) > 0.25:
            return False

    if len(lines) >= 5:
        top = Counter(lines).most_common(1)[0][1]
        if top / len(lines) > 0.4:
            return False
    return True


def normalize_sample(sample: dict, field: str) -> str:
    if field == "Q":
        q = (sample.get("Q") or "").strip()
        a = (sample.get("A") or "").strip()
        return "\n".join(p for p in (q, a) if p)
    if "instruction" in sample and "output" in sample:
        parts = [(sample.get(k) or "").strip() for k in ("instruction", "input", "output")]
        return "\n".join(p for p in parts if p)
    return sample.get(field) or sample.get("text") or sample.get("content") or ""


def build_stream(seed: int):
    from datasets import load_dataset, interleave_datasets
    streams, weights, kinds = [], [], []
    for name, config, weight, field, kind in DATASETS_CONFIG:
        try:
            ds = load_dataset(name, config, split="train", streaming=True)
            ds = ds.map(
                lambda s, field=field, kind=kind: {
                    "text": normalize_sample(s, field), "kind": kind},
                remove_columns=list(ds.features) if ds.features else None,
            )
            streams.append(ds)
            weights.append(weight)
            print(f"  [OK] {name} ({config}) peso={weight}")
        except Exception as e:
            print(f"  [ERRO] {name}: {e}")
    if not streams:
        raise RuntimeError("Nenhum dataset carregado")
    total = sum(weights)
    return interleave_datasets(
        streams,
        probabilities=[w / total for w in weights],
        seed=seed,
        stopping_strategy="all_exhausted",
    )


def load_progress(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_progress(path, seen, kept, train_total, val_total):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"seen": seen, "kept": kept,
                   "train_total": train_total, "val_total": val_total}, f)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=float, default=3.3e9)
    parser.add_argument("--val-tokens", type=float, default=5e6)
    parser.add_argument("--out-dir", default="data/arpa150m")
    parser.add_argument("--tokenizer-dir", default="tokenizer-arpa-64k-clean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="ignora progresso salvo e recomeca do zero")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    assert eos_id is not None and eos_id >= 0

    target_train = int(args.train_tokens)
    target_val = int(args.val_tokens)
    progress_path = os.path.join(args.out_dir, "progress.json")

    # Auto-resume: se ha progresso salvo e os bins existem, retoma de onde parou
    prog = None if args.force else load_progress(progress_path)
    if prog:
        skip_docs = prog["seen"]
        kept = prog["kept"]
        train_w = BinWriter(os.path.join(args.out_dir, "train_tokens.bin"),
                            resume_tokens=prog["train_total"])
        val_w = BinWriter(os.path.join(args.out_dir, "val_tokens.bin"),
                          resume_tokens=prog["val_total"])
        print(f"Retomando: {skip_docs:,} docs ja vistos | "
              f"train={train_w.total:,} val={val_w.total:,} tokens")
    else:
        skip_docs = 0
        kept = 0
        train_w = BinWriter(os.path.join(args.out_dir, "train_tokens.bin"))
        val_w = BinWriter(os.path.join(args.out_dir, "val_tokens.bin"))

    print("Carregando streams...")

    seen = skip_docs
    t0 = time.time()
    tokens_at_start = train_w.total
    next_report = ((train_w.total // 50_000_000) + 1) * 50_000_000

    def checkpoint():
        train_w.sync()
        val_w.sync()
        save_progress(progress_path, seen, kept, train_w.total, val_w.total)

    # Loop externo auto-curavel: a HF deixa cair conexao de vez em quando
    # (408/timeout no CDN). Em vez de morrer, faz checkpoint, reconstroi o
    # stream e retoma de onde parou. So aborta apos muitas falhas seguidas.
    finished = False
    fails = 0
    while not finished:
        try:
            stream = build_stream(args.seed)
            if seen:
                print(f"Avancando o stream {seen:,} docs (re-streaming, sem retokenizar)...",
                      flush=True)
                stream = stream.skip(seen)

            for sample in stream:
                text, kind = sample["text"], sample["kind"]
                seen += 1
                if not text or len(text) < MIN_TEXT_LENGTH:
                    continue
                text = clean_text(text[:MAX_TEXT_LENGTH], kind)
                if len(text) < MIN_TEXT_LENGTH or not is_quality_text(text, kind):
                    continue
                if kind == "reasoning":
                    text = f"{REASONING_TOKEN}\n{text}"
                elif kind == "conversation":
                    text = f"{CONVERSATION_TOKEN}\n{text}"

                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(eos_id)
                kept += 1

                # 1 a cada 500 docs vai pro val ate fechar o alvo
                if val_w.total < target_val and kept % 500 == 0:
                    val_w.write(ids)
                else:
                    train_w.write(ids)

                if train_w.total >= next_report:
                    rate = (train_w.total - tokens_at_start) / (time.time() - t0)
                    eta_h = (target_train - train_w.total) / max(rate, 1) / 3600
                    print(f"  {train_w.total / 1e9:.2f}B tokens | {kept:,}/{seen:,} docs "
                          f"| {rate / 1e6:.1f}M tok/s | ETA {eta_h:.1f}h", flush=True)
                    checkpoint()  # resume perde no maximo 50M tokens
                    next_report += 50_000_000
                    fails = 0     # progresso real zera o contador de falhas

                if train_w.total >= target_train and val_w.total >= target_val:
                    finished = True
                    break
            else:
                finished = True  # stream esgotou antes do alvo

        except KeyboardInterrupt:
            checkpoint()
            train_w.close(); val_w.close()
            print("\nInterrompido — progresso salvo. Rode de novo para retomar.")
            return
        except Exception as e:
            fails += 1
            checkpoint()
            wait = min(120, 10 * fails)
            print(f"  [rede] {type(e).__name__}: {str(e)[:160]}")
            print(f"  Checkpoint em {train_w.total / 1e9:.2f}B tokens. "
                  f"Retomando em {wait}s (falha {fails}/30)...", flush=True)
            if fails > 30:
                train_w.close(); val_w.close()
                sys.exit("Muitas falhas seguidas. Rode de novo mais tarde (retoma sozinho).")
            time.sleep(wait)

    train_w.close()
    val_w.close()
    if os.path.exists(progress_path):
        os.remove(progress_path)  # concluido: nao ha mais o que retomar
    with open(os.path.join(args.out_dir, "DONE"), "w", encoding="utf-8") as f:
        f.write(f"train={train_w.total} val={val_w.total}\n")  # marcador p/ o notebook
    print(f"\nPronto: train={train_w.total:,} tokens | val={val_w.total:,} tokens "
          f"| aproveitamento {kept}/{seen} docs")
    print(f"Arquivos em {args.out_dir}/")


if __name__ == "__main__":
    main()
