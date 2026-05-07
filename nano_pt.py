#!/usr/bin/env python3
"""
nano_pt.py - Pre-treino LLaMA 800M para H100/A100

Modelo: LLaMA-style 800M parametros, contexto 8192 tokens
Precisao: BF16 nativo (A100/H100 Ampere+)
Datasets (streaming): Wikipedia PT+EN, dialogos OpenAssistant,
                      codigo Python/JS (The Stack), C4 PT
Tokenizer: BPE 32K customizado PT-BR (./tokenizer-arpa-32k/)

Uso:
    python nano_pt.py                       # Treinar do zero
    python nano_pt.py --resume step_5000    # Resumir de checkpoint
    python nano_pt.py --resume latest       # Ultimo checkpoint
"""

import os
import sys
import gc
import time
import json
import math
import random
import signal
import shutil
import argparse
import logging
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Optional, Iterator

# ==============================================================================
# Deteccao de hardware (ANTES de importar torch)
# ==============================================================================
# Detecta AMD ROCm vs NVIDIA CUDA checando arquivos de dispositivo e variaveis
# de ambiente que existem antes do torch ser carregado.

_IS_ROCM = (
    os.path.exists("/dev/kfd")                       # dispositivo AMD ROCm
    or os.environ.get("ROCM_VERSION") is not None    # variavel de ambiente ROCm
    or os.environ.get("HSA_VERSION") is not None
)

if _IS_ROCM:
    # AMD ROCm: configuracoes especificas para gfx1032 (RX 6600)
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = (
        "garbage_collection_threshold:0.6,max_split_size_mb:256,expandable_segments:True"
    )
else:
    # NVIDIA CUDA: gerenciamento de memoria moderno
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset, interleave_datasets

# ==============================================================================
# Perfil de hardware (apos import do torch)
# ==============================================================================

def detect_hardware_profile():
    """
    Detecta o hardware disponivel e retorna um perfil com as configuracoes
    otimas para aquele dispositivo.

    Retorna dict com:
      is_rocm      : bool  - True se AMD ROCm, False se NVIDIA CUDA
      compute_dtype: dtype - bfloat16 (Ampere+) ou float16 (AMD/Pascal)
      use_scaler   : bool  - GradScaler necessario apenas com float16
      fused_adam   : bool  - AdamW fused disponivel apenas em CUDA
      num_workers  : int   - workers do DataLoader
      pin_memory   : bool  - pin_memory do DataLoader
    """
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

    if is_rocm:
        return dict(
            is_rocm=True,
            compute_dtype=torch.float16,   # ROCm: float16 + GradScaler
            use_scaler=True,
            fused_adam=False,              # fused nao suportado no ROCm
            num_workers=0,                 # multiprocessing instavel no ROCm
            pin_memory=False,
        )

    # NVIDIA: checar capacidade para BF16 (Ampere = compute capability 8.x+)
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    has_bf16 = capability[0] >= 8  # RTX 3080/3050/4090 etc.

    # 3050 Laptop: bf16 suportado mas throughput baixo. FORCE_FP16=1 força fp16.
    force_fp16 = os.environ.get("FORCE_FP16", "0") == "1"
    use_bf16 = has_bf16 and not force_fp16

    return dict(
        is_rocm=False,
        compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        use_scaler=not use_bf16,           # BF16 nao precisa de scaler
        fused_adam=True,                   # fused AdamW disponivel no CUDA
        num_workers=2,
        pin_memory=True,
    )

HW = detect_hardware_profile()

# ==============================================================================
# Configuracoes
# ==============================================================================

RUN_NAME = "arpa-800m-pretrain"
CHECKPOINT_DIR = "./checkpoints-arpa800m"
TOKENIZER_DIR = "./tokenizer-arpa-32k"
LOG_DIR = "./logs-arpa800m"

# Arquitetura LLaMA ~800M
# hidden=2048, layers=24, heads=16, kv=8, intermediate=5504
# Params: embed(2048*32000) + 24*(attn+mlp+norms) + lm_head (tied) ≈ 800M
MODEL_CONFIG = dict(
    vocab_size=32_000,
    hidden_size=2048,
    intermediate_size=5504,      # SwiGLU: 2/3 * 4 * 2048 arredondado p/ multiplo de 64
    num_hidden_layers=24,
    num_attention_heads=16,      # head_dim = 128
    num_key_value_heads=8,       # GQA 2:1 (bom trade-off memoria/qualidade)
    max_position_embeddings=8192,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,         # RoPE theta alto para contexto longo (LLaMA 3 style)
    rope_scaling={"rope_type": "llama3", "factor": 8.0,
                  "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 4096},
    hidden_act="silu",
    attention_dropout=0.0,
    use_cache=False,
    tie_word_embeddings=True,
)

# Hiperparametros de treino — Chinchilla para 800M params
# 800M * 20 = 16B tokens minimo; usamos 20B para margem
# H100 80GB: batch_size=4, block_size=8192 → 32K tokens/step
# grad_accum=32 → effective batch = 1M tokens/step
BATCH_SIZE = 4                   # seqs por step (ajustar se OOM: 2 com grad_ckpt)
GRAD_ACCUMULATION = 32           # effective batch = 4*32*8192 = 1.05M tokens
BLOCK_SIZE = 8192                # contexto
MAX_TOKENS = 20_000_000_000      # 20B tokens (Chinchilla 800M * 25)
LEARNING_RATE = 3e-4             # pico LR (escala com sqrt do batch vs 100M)
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
BETAS = (0.9, 0.95)
MIN_LR_RATIO = 0.1               # LR final = 10% do pico
EVAL_EVERY = 500
EVAL_TOKENS = 131_072            # 16 blocos de 8192 tokens

# Checkpointing
SAVE_EVERY = 500
KEEP_LAST_N = 3
LOG_EVERY = 10
GENERATE_EVERY = 2000
CLEAR_MEMORY_EVERY = 1000

# Mix multilingual: PT-BR foco, EN generalização, codigo raciocínio
DATASETS_CONFIG = [
    # (nome, config/subset, peso, campo_texto)
    ("wikimedia/wikipedia",          "20231101.pt", 0.25, "text"),    # factual PT
    ("wikimedia/wikipedia",          "20231101.en", 0.20, "text"),    # factual EN (generalização)
    ("TucanoBR/GigaVerbo",           None,          0.20, "text"),    # diversidade PT-BR
    ("OpenAssistant/oasst1",         None,          0.10, "text"),    # dialogos instrução
    ("bigcode/the-stack-dedup",      "data/python", 0.10, "content"), # codigo Python
    ("bigcode/the-stack-dedup",      "data/javascript", 0.05, "content"), # codigo JS
    ("allenai/c4",                   "pt",          0.05, "text"),    # web PT diverso
    ("allenai/c4",                   "en",          0.05, "text"),    # web EN diverso
]

# Filtros de texto
MIN_TEXT_LENGTH = 200            # minimo real: textos muito curtos sao lixo
MAX_TEXT_LENGTH = 50_000         # caracteres maximos

# Pre-tokenized binary files (gerados por prepare_data.py)
# Quando existem, resume e instantaneo — sem re-iterar 475M tokens.
TRAIN_BIN = "./data/train_tokens.bin"
VAL_BIN   = "./data/val_tokens.bin"


# ==============================================================================
# BinaryTokenDataset — dataset de tokens pre-tokenizados (O(1) seek)
# ==============================================================================

class BinaryTokenDataset(torch.utils.data.IterableDataset):
    """
    Le tokens uint16 de um arquivo binario pre-tokenizado via memmap.
    Resume e instantaneo: basta pular N*2 bytes no arquivo.

    Formato: tokens uint16 concatenados sem padding, gerado por prepare_data.py.
    """

    def __init__(self, bin_path: str, block_size: int, skip_tokens: int = 0):
        self.bin_path   = bin_path
        self.block_size = block_size
        self.skip_tokens = skip_tokens
        # Valida arquivo
        n = os.path.getsize(bin_path) // 2  # uint16 = 2 bytes
        if n < block_size:
            raise ValueError(f"Arquivo muito pequeno: {n} tokens < block_size {block_size}")
        self.n_tokens = n

    def __iter__(self):
        data = np.memmap(self.bin_path, dtype=np.uint16, mode='r')
        # Resume: pula tokens ja processados (O(1) — sem loop)
        start = min(self.skip_tokens, self.n_tokens - self.block_size)
        idx = start

        while idx + self.block_size <= self.n_tokens:
            block = torch.from_numpy(
                data[idx : idx + self.block_size].astype(np.int64)
            )
            idx += self.block_size
            yield {"input_ids": block, "labels": block.clone()}


# ==============================================================================
# Utilidades
# ==============================================================================

def clear_memory():
    """Libera memoria GPU sem synchronize (overhead menor)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory():
    """Retorna uso de VRAM em MB."""
    if not torch.cuda.is_available():
        return 0, 0
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return allocated, reserved


def count_parameters(model):
    """Conta parametros totais e treinaveis."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_num(n):
    """Formata numero grande: 163M, 1.2B, etc."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def set_seed(seed=42):
    """Seta seed para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Logger
# ==============================================================================

class TrainingLogger:
    """Logger unificado: console + arquivo + TensorBoard."""

    def __init__(self, run_name: str, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)

        # File logger
        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(ch)

        # File handler
        log_file = os.path.join(log_dir, f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(fh)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard", run_name))

        self.info(f"Log: {log_file}")

    def info(self, msg: str):
        self.logger.info(msg)

    def scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


# ==============================================================================
# Token Packing Dataset
# ==============================================================================

class PackedPretrainDataset(IterableDataset):
    """
    Dataset que faz token packing: concatena textos e corta em blocos
    de tamanho fixo (BLOCK_SIZE). Sem padding, 100% eficiencia de tokens.

    Cada bloco retorna input_ids e labels identicos (causal LM).
    """

    def __init__(
        self,
        tokenizer,
        block_size: int = 1024,
        seed: int = 42,
        skip_tokens: int = 0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seed = seed
        self.skip_tokens = skip_tokens
        self.eos_token_id = tokenizer.eos_token_id

    def _load_datasets(self):
        """Carrega e intercala datasets via streaming."""
        streams = []
        weights = []

        for ds_name, ds_config, weight, text_field in DATASETS_CONFIG:
            try:
                ds = load_dataset(
                    ds_name,
                    ds_config,
                    split="train",
                    streaming=True,
                )

                streams.append(ds)
                weights.append(weight)
                print(f"  [OK] {ds_name} ({ds_config}) peso={weight}")
            except Exception as e:
                print(f"  [ERRO] {ds_name}: {e}")

        if not streams:
            raise RuntimeError("Nenhum dataset carregado!")

        # Normalizar pesos
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        return interleave_datasets(
            streams,
            probabilities=weights,
            seed=self.seed,
            stopping_strategy="first_exhausted",
        )

    @staticmethod
    def _is_quality_text(text: str) -> bool:
        """
        Filtros de qualidade para pre-treino limpo.
        Retorna False para textos que devem ser rejeitados.
        """
        n = len(text)

        # 1. Ratio de letras: rejeita HTML, codigo, tabelas, spam de simbolos
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / n < 0.65:
            return False

        # 2. Ratio de digitos: rejeita listas de numeros, tabelas financeiras
        digits = sum(1 for c in text if c.isdigit())
        if digits / n > 0.15:
            return False

        # 3. Contagem de palavras: textos muito curtos nao ensinam nada
        words = text.split()
        if len(words) < 40:
            return False

        # 4. Comprimento medio de palavra: palavras normais tem 3-15 chars
        #    Links, hashes, codigos quebram essa media
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3.0 or avg_word_len > 15.0:
            return False

        # 5. Repeticao de linhas: spam, listas repetitivas, boilerplate
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 5:
            top_line_count = Counter(lines).most_common(1)[0][1]
            if top_line_count / len(lines) > 0.3:
                return False

        return True

    def _text_iterator(self):
        """Itera sobre textos filtrados dos datasets."""
        dataset = self._load_datasets()

        for sample in dataset:
            text = sample.get("text", "")

            if not text or len(text) < MIN_TEXT_LENGTH:
                continue

            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]

            if not self._is_quality_text(text):
                continue

            yield text

    def __iter__(self) -> Iterator[dict]:
        """
        Token packing: concatena tokens com EOS entre textos,
        e corta em blocos de block_size.
        """
        buffer = []
        tokens_emitted = 0

        for text in self._text_iterator():
            # Tokenizar texto
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if not token_ids:
                continue

            # Adicionar tokens + EOS ao buffer
            buffer.extend(token_ids)
            buffer.append(self.eos_token_id)

            # Emitir blocos completos
            while len(buffer) >= self.block_size:
                block = buffer[:self.block_size]
                buffer = buffer[self.block_size:]

                # Pular tokens ja processados antes do resume
                if tokens_emitted < self.skip_tokens:
                    tokens_emitted += self.block_size
                    continue

                tokens_emitted += self.block_size
                input_ids = torch.tensor(block, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }


# ==============================================================================
# Checkpoint Management
# ==============================================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    total_tokens: int,
    loss: float,
    samples_seen: int,
    logger: TrainingLogger,
):
    """Salva checkpoint completo para resume."""
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"Salvando checkpoint em {ckpt_dir}...")

    # Modelo
    unwrapped = model
    if hasattr(model, "module"):
        unwrapped = model.module
    if hasattr(model, "_orig_mod"):
        unwrapped = model._orig_mod

    unwrapped.save_pretrained(ckpt_dir)

    # Training state
    train_state = {
        "step": step,
        "total_tokens": total_tokens,
        "loss": loss,
        "samples_seen": samples_seen,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": torch.random.get_rng_state(),
        "timestamp": datetime.now().isoformat(),
    }
    if torch.cuda.is_available():
        train_state["cuda_rng_state"] = torch.cuda.get_rng_state()

    torch.save(train_state, os.path.join(ckpt_dir, "training_state.pt"))

    # Info JSON
    ppl = math.exp(min(loss, 20))
    info = {
        "step": step,
        "total_tokens": total_tokens,
        "loss": f"{loss:.4f}",
        "perplexity": f"{ppl:.2f}",
        "samples_seen": samples_seen,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(ckpt_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Checkpoint salvo: step={step}, loss={loss:.4f}, ppl={ppl:.2f}")

    # Limpar checkpoints antigos
    cleanup_checkpoints(logger)


def cleanup_checkpoints(logger: TrainingLogger):
    """Mantem apenas os ultimos KEEP_LAST_N checkpoints."""
    ckpt_base = Path(CHECKPOINT_DIR)
    if not ckpt_base.exists():
        return

    ckpt_dirs = sorted(
        [d for d in ckpt_base.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if len(ckpt_dirs) <= KEEP_LAST_N:
        return

    to_remove = ckpt_dirs[:-KEEP_LAST_N]
    for d in to_remove:
        logger.info(f"Removendo checkpoint antigo: {d.name}")
        shutil.rmtree(d, ignore_errors=True)


def save_best_checkpoint(
    model,
    step: int,
    total_tokens: int,
    val_loss: float,
    logger: TrainingLogger,
):
    """
    Salva o melhor modelo (por val loss) em checkpoints-arpa3/best/.
    Nunca e deletado pela rotacao normal — sempre representa o melhor
    checkpoint visto durante todo o treino.
    """
    best_dir = os.path.join(CHECKPOINT_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)

    unwrapped = model
    if hasattr(model, "module"):
        unwrapped = model.module
    if hasattr(model, "_orig_mod"):
        unwrapped = model._orig_mod

    unwrapped.save_pretrained(best_dir)

    val_ppl = math.exp(min(val_loss, 20))
    info = {
        "step": step,
        "total_tokens": total_tokens,
        "val_loss": f"{val_loss:.4f}",
        "val_perplexity": f"{val_ppl:.2f}",
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(best_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info(
        f"  >> [NOVO MELHOR] Val loss={val_loss:.4f} ppl={val_ppl:.1f} "
        f"salvo em {CHECKPOINT_DIR}/best/"
    )


def load_checkpoint(
    ckpt_name: str,
    model,
    optimizer,
    scheduler,
    logger: TrainingLogger,
):
    """Carrega checkpoint para resume."""
    ckpt_dir = os.path.join(CHECKPOINT_DIR, ckpt_name)

    if not os.path.exists(ckpt_dir):
        logger.info(f"Checkpoint nao encontrado: {ckpt_dir}")
        return 0, 0, 0

    logger.info(f"Carregando checkpoint de {ckpt_dir}...")

    # Carregar pesos do modelo (sempre que existirem)
    from safetensors.torch import load_file
    model_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict, strict=False)
    else:
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)

    # Training state: se nao existir (ex: checkpoint best/), usa info.json e reseta optimizer/scheduler
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    if not os.path.exists(state_path):
        info_path = os.path.join(ckpt_dir, "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            total_tokens = int(info.get("total_tokens", 0))
            logger.info(
                f"training_state.pt ausente (best/): pesos carregados, "
                f"optimizer/scheduler reinicializados. total_tokens={format_num(total_tokens)}"
            )
            # start_step=0 para reinicializar scheduler com warmup no LR novo
            return 0, total_tokens, 0
        logger.info("training_state.pt nao encontrado, iniciando do zero")
        return 0, 0, 0

    state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    # RNG states
    if "rng_state" in state:
        torch.random.set_rng_state(state["rng_state"])
    if "cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])

    step = state["step"]
    total_tokens = state.get("total_tokens", 0)
    samples_seen = state.get("samples_seen", 0)

    logger.info(
        f"Checkpoint carregado: step={step}, "
        f"tokens={format_num(total_tokens)}, loss={state.get('loss', '?')}"
    )

    return step, total_tokens, samples_seen


@torch.no_grad()
def evaluate(model, tokenizer, device, block_size, num_tokens=50_000):
    """
    Calcula validation loss em dados separados (Wikipedia PT).
    Usa val_tokens.bin quando disponivel (rapido); fallback para streaming.
    """
    model.eval()
    total_loss  = 0.0
    total_steps = 0
    tokens_seen = 0

    if os.path.exists(VAL_BIN):
        # Caminho rapido: tokens pre-tokenizados
        data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
        idx  = 0
        while idx + block_size <= len(data) and tokens_seen < num_tokens:
            block = torch.from_numpy(
                data[idx : idx + block_size].astype(np.int64)
            ).unsqueeze(0).to(device)
            idx += block_size
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=block, labels=block)
            total_loss  += outputs.loss.item()
            total_steps += 1
            tokens_seen += block_size
    else:
        # Fallback: streaming Wikipedia PT (mais lento, pula 200K docs)
        from datasets import load_dataset
        val_ds = load_dataset(
            "wikimedia/wikipedia", "20231101.pt",
            split="train", streaming=True,
        )
        eos_id = tokenizer.eos_token_id
        buffer = []
        skip   = 200_000
        for sample in val_ds:
            if skip > 0:
                skip -= 1
                continue
            text = sample.get("text", "")
            if len(text) < 100:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)
            buffer.append(eos_id)
            while len(buffer) >= block_size:
                block = buffer[:block_size]
                buffer = buffer[block_size:]
                input_ids = torch.tensor([block], dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=input_ids)
                total_loss  += outputs.loss.item()
                total_steps += 1
                tokens_seen += block_size
                if tokens_seen >= num_tokens:
                    break
            if tokens_seen >= num_tokens:
                break

    model.train()
    return total_loss / total_steps if total_steps > 0 else float("inf")


def find_latest_checkpoint() -> Optional[str]:
    """Encontra o checkpoint mais recente."""
    ckpt_base = Path(CHECKPOINT_DIR)
    if not ckpt_base.exists():
        return None

    ckpt_dirs = sorted(
        [d for d in ckpt_base.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if ckpt_dirs:
        return ckpt_dirs[-1].name
    return None


# ==============================================================================
# Geracao de texto (teste)
# ==============================================================================

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="O Brasil", max_new_tokens=100):
    """Gera texto de teste."""
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    model.train()
    return text


# ==============================================================================
# Main Training
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Arpa-160M Pre-training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Nome do checkpoint para resume (ex: step_5000). "
                             "Use 'latest' para o mais recente.")
    args, _ = parser.parse_known_args()  # parse_known_args ignora args do Jupyter/Colab

    # --------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------
    set_seed(42)

    logger = TrainingLogger(RUN_NAME, LOG_DIR)
    logger.info("=" * 60)
    logger.info("Arpa-160M: Pre-treino do Zero")
    logger.info("=" * 60)

    # Hardware
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA nao disponivel!")

    # Verifica BF16 nativo (requer Ampere: A100, H100, RTX 30xx+)
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        logger.info(f"AVISO: GPU compute capability {capability[0]}.{capability[1]} < 8.0 "
                    f"— BF16 pode nao ter suporte nativo. Use A100/H100.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB) | CC {capability[0]}.{capability[1]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.version.cuda}")

    # --------------------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------------------
    logger.info(f"\nCarregando tokenizer de {TOKENIZER_DIR}...")

    if not os.path.exists(TOKENIZER_DIR):
        logger.info(f"ERRO: Tokenizer nao encontrado em {TOKENIZER_DIR}")
        logger.info("Execute primeiro: python train_tokenizer.py")
        raise RuntimeError(f"Tokenizer nao encontrado em {TOKENIZER_DIR}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    logger.info(f"Tokenizer: vocab_size={tokenizer.vocab_size}, "
                f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, "
                f"pad={tokenizer.pad_token_id}")

    # --------------------------------------------------------------------------
    # Modelo LLaMA do zero
    # --------------------------------------------------------------------------
    logger.info("\nCriando modelo LLaMA...")

    # Tenta FA2; cai para sdpa se flash-attn nao estiver instalado
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2: disponivel")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("Flash Attention 2: NAO instalada — usando sdpa. "
                    "Instale com: pip install flash-attn --no-build-isolation")

    config = LlamaConfig(
        **MODEL_CONFIG,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation=attn_impl,
    )

    # Cria em BF16 direto — evita custo de conversao float32->bf16 depois
    model = LlamaForCausalLM(config).to(torch.bfloat16)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Parametros: {format_num(total_params)} total, "
                f"{format_num(trainable_params)} treinaveis")
    logger.info(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
                f"ctx={BLOCK_SIZE}")

    # Gradient checkpointing: recomendado para 800M + 8192 ctx em <80GB
    use_grad_ckpt = os.environ.get("GRAD_CKPT", "1") == "1"
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: ativado (GRAD_CKPT=1)")
    else:
        logger.info("Gradient checkpointing: DESATIVADO (set GRAD_CKPT=1 se OOM)")

    model.to(device)

    # TF32 nas matmuls — gratis em Ampere. torch.compile foi movido para DEPOIS do resume
    # porque compilar antes prefixa state_dict com _orig_mod. e load_state_dict(strict=False)
    # falha silenciosamente (loss volta a ~10.5 = random init).
    torch.set_float32_matmul_precision("high")

    alloc, reserved = get_gpu_memory()
    logger.info(f"VRAM apos modelo: {alloc:.0f}MB alocado, {reserved:.0f}MB reservado")

    # --------------------------------------------------------------------------
    # Optimizer, Scheduler, Scaler
    # --------------------------------------------------------------------------

    # Separar parametros com e sem weight decay
    no_decay = ["bias", "layernorm", "layer_norm", "rmsnorm", "rms_norm"]
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=1e-8,
        fused=True,   # fused AdamW: mais rapido e menos VRAM em CUDA
    )

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    total_optimizer_steps = MAX_TOKENS // (tokens_per_step * GRAD_ACCUMULATION)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS // GRAD_ACCUMULATION,
        num_training_steps=total_optimizer_steps,
    )

    logger.info(f"Hardware: NVIDIA CUDA | dtype=bfloat16 | fused_adam=True")
    logger.info(f"Optimizer: AdamW (lr={LEARNING_RATE}, betas={BETAS}, wd={WEIGHT_DECAY})")
    logger.info(f"Scheduler: cosine (warmup={WARMUP_STEPS} steps, {total_optimizer_steps} optimizer steps totais)")
    logger.info(f"Orcamento: {format_num(MAX_TOKENS)} tokens (Chinchilla para {format_num(sum(p.numel() for p in model.parameters()))} params)")
    logger.info(f"Batch: {BATCH_SIZE} x {GRAD_ACCUMULATION} accum = "
                f"{BATCH_SIZE * GRAD_ACCUMULATION} seqs = "
                f"{format_num(BATCH_SIZE * GRAD_ACCUMULATION * BLOCK_SIZE)} tokens efetivos/step")

    # --------------------------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------------------------
    start_step = 0
    total_tokens = 0
    samples_seen = 0

    if args.resume:
        ckpt_name = args.resume
        if ckpt_name == "latest":
            ckpt_name = find_latest_checkpoint()
            if ckpt_name is None:
                logger.info("Nenhum checkpoint encontrado para resume.")
            else:
                logger.info(f"Ultimo checkpoint: {ckpt_name}")

        if ckpt_name:
            start_step, total_tokens, samples_seen = load_checkpoint(
                ckpt_name, model, optimizer, scheduler, logger
            )

    # torch.compile APOS load_checkpoint: compilar antes prefixa state_dict com _orig_mod.
    # e load_state_dict(strict=False) falha silenciosamente (loss volta a ~10.5 = random init).
    if not use_grad_ckpt:
        try:
            model = torch.compile(model, mode="max-autotune")
            logger.info("torch.compile: ativado (mode=max-autotune)")
        except Exception as e:
            logger.info(f"torch.compile: falhou ({e}), seguindo sem")
    else:
        logger.info("torch.compile: desativado (gradient checkpointing incompativel)")

    # max_steps calculado DEPOIS do resume: baseado em tokens restantes, nao tokens totais.
    # Garante que sempre treinamos exatamente MAX_TOKENS tokens independente do batch_size
    # ou de quantos tokens ja foram processados em runs anteriores.
    tokens_remaining = MAX_TOKENS - total_tokens
    max_steps = start_step + (tokens_remaining // tokens_per_step)
    logger.info(f"Tokens ja processados: {format_num(total_tokens)} | Restantes: {format_num(tokens_remaining)}")
    logger.info(f"Steps restantes: {max_steps - start_step} (ate step {max_steps})")

    # --------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------
    if os.path.exists(TRAIN_BIN):
        # Caminho rapido: tokens pre-tokenizados — resume e O(1)
        logger.info(f"\nDataset binario encontrado: {TRAIN_BIN}")
        n_total = os.path.getsize(TRAIN_BIN) // 2
        logger.info(f"  {n_total:,} tokens disponíveis | pulando {total_tokens:,} (resume O(1))")
        dataset = BinaryTokenDataset(
            bin_path=TRAIN_BIN,
            block_size=BLOCK_SIZE,
            skip_tokens=total_tokens,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,   # memmap nao precisa de workers
            pin_memory=True,
        )
    else:
        # Fallback: streaming HuggingFace (lento no resume)
        logger.info("\nCarregando datasets (streaming)...")
        dataset = PackedPretrainDataset(
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE,
            seed=42,
            skip_tokens=total_tokens,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,   # interleave_datasets nao e thread-safe com workers>0
            pin_memory=True,
        )

    data_iter = iter(dataloader)

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO TREINO")
    logger.info(f"Meta: {format_num(MAX_TOKENS)} tokens ({total_tokens/MAX_TOKENS*100:.1f}% ja concluido)")
    logger.info(f"Block size: {BLOCK_SIZE} tokens")
    logger.info("=" * 60 + "\n")

    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    log_loss_count = 0
    best_loss = float("inf")
    best_val_loss = float("inf")
    last_grad_norm = 0.0
    window_tokens = 0
    train_start = time.time()
    step_start = time.time()

    # Signal handler para salvar ao interromper
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        logger.info("\n[!] Interrupcao detectada, salvando checkpoint...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Prompts de teste para geracao
    test_prompts = [
        "O Brasil é",
        "A inteligência artificial",
        "No ano de 2024",
        "A ciência moderna",
    ]

    for step in range(start_step, max_steps):
        if interrupted:
            break

        if total_tokens >= MAX_TOKENS:
            logger.info(f"Limite de tokens atingido: {format_num(total_tokens)} >= {format_num(MAX_TOKENS)}")
            break

        # Obter batch
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.info("Dataset esgotado, reiniciando iterator...")
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Contar tokens reais (sem padding)
        batch_tokens = input_ids.numel()
        total_tokens += batch_tokens
        window_tokens += batch_tokens
        samples_seen += input_ids.size(0)

        # Forward pass com mixed precision
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUMULATION

        # Backward — BF16 nao precisa de GradScaler
        loss.backward()

        running_loss += outputs.loss.item()
        log_loss_count += 1

        # Optimizer step a cada GRAD_ACCUMULATION
        if (step + 1) % GRAD_ACCUMULATION == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            last_grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = running_loss / log_loss_count if log_loss_count > 0 else 0
            ppl = math.exp(min(avg_loss, 20))
            lr = scheduler.get_last_lr()[0]

            elapsed = time.time() - step_start
            tokens_per_sec = window_tokens / elapsed if elapsed > 0 else 0

            alloc, _ = get_gpu_memory()
            pct = total_tokens / MAX_TOKENS * 100

            logger.info(
                f"Step {step + 1:>6} ({pct:.1f}%) | "
                f"loss={avg_loss:.4f} ppl={ppl:.1f} | "
                f"lr={lr:.2e} | "
                f"+{format_num(window_tokens)} / {format_num(total_tokens)} ({tokens_per_sec:.0f} tok/s) | "
                f"VRAM={alloc:.0f}MB"
            )

            # TensorBoard
            logger.scalar("Loss/train", avg_loss, step + 1)
            logger.scalar("Perplexity", ppl, step + 1)
            logger.scalar("Learning_Rate", lr, step + 1)
            logger.scalar("Tokens_Total", total_tokens, step + 1)
            logger.scalar("Tokens_per_sec", tokens_per_sec, step + 1)
            logger.scalar("VRAM_MB", alloc, step + 1)
            logger.scalar("Gradient_Norm", last_grad_norm, step + 1)

            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss

            window_tokens = 0

            running_loss = 0.0
            log_loss_count = 0
            step_start = time.time()

        # Limpar memoria
        if (step + 1) % CLEAR_MEMORY_EVERY == 0:
            clear_memory()

        # Validacao
        if (step + 1) % EVAL_EVERY == 0:
            logger.info("Avaliando no validation set...")
            val_loss = evaluate(model, tokenizer, device, BLOCK_SIZE, EVAL_TOKENS)
            val_ppl = math.exp(min(val_loss, 20))
            logger.info(f"  >> Val loss={val_loss:.4f}, Val ppl={val_ppl:.1f}")
            logger.scalar("Loss/val", val_loss, step + 1)
            logger.scalar("Perplexity/val", val_ppl, step + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint(model, step + 1, total_tokens, val_loss, logger)
            clear_memory()

        # Checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            current_loss = outputs.loss.item()
            save_checkpoint(
                model, optimizer, scheduler,
                step + 1, total_tokens, current_loss, samples_seen, logger,
            )

        # Geracao de teste
        if (step + 1) % GENERATE_EVERY == 0:
            logger.info("\n--- Geracao de Teste ---")
            for prompt in test_prompts:
                try:
                    # Habilitar cache temporariamente para generate
                    model.config.use_cache = True
                    text = generate_sample(model, tokenizer, device, prompt=prompt)
                    model.config.use_cache = False
                    model.train()
                    logger.info(f"  [{prompt}] -> {text[:200]}")
                except Exception as e:
                    logger.info(f"  [{prompt}] -> ERRO: {e}")
                    model.config.use_cache = False
                    model.train()
            logger.info("--- Fim Geracao ---\n")
            clear_memory()

    # --------------------------------------------------------------------------
    # Finalizacao
    # --------------------------------------------------------------------------
    total_time = time.time() - train_start
    hours = total_time / 3600

    logger.info("\n" + "=" * 60)
    logger.info("TREINO FINALIZADO")
    logger.info("=" * 60)
    logger.info(f"Steps completados: {step + 1 if 'step' in locals() else start_step}")
    logger.info(f"Tokens processados: {format_num(total_tokens)}")
    logger.info(f"Melhor train loss: {best_loss:.4f} (ppl={math.exp(min(best_loss, 20)):.1f})")
    if best_val_loss < float("inf"):
        logger.info(f"Melhor val loss:   {best_val_loss:.4f} (ppl={math.exp(min(best_val_loss, 20)):.1f})")
    logger.info(f"Tempo total: {hours:.1f}h")

    # Salvar checkpoint final
    final_step = step + 1 if 'step' in locals() else start_step
    if final_step > start_step:
        save_checkpoint(
            model, optimizer, scheduler,
            final_step, total_tokens,
            outputs.loss.item() if 'outputs' in locals() else best_loss,
            samples_seen, logger,
        )

    # Salvar modelo final separado
    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    unwrapped = model
    if hasattr(model, "module"):
        unwrapped = model.module
    if hasattr(model, "_orig_mod"):
        unwrapped = model._orig_mod
    unwrapped.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Modelo final salvo em {final_dir}")

    alloc, reserved = get_gpu_memory()
    logger.info(f"VRAM final: {alloc:.0f}MB alocado, {reserved:.0f}MB reservado")

    logger.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
