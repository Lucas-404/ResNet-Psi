"""
config.py - Configuracao do modelo e do treino Arpa-150M.

Presets:
    a100   - pre-treino serio no Colab A100 40GB
    local  - smoke test na RTX 3050 4GB (mesmo modelo, batch minimo)
    tiny   - modelo minusculo pra testar o pipeline em CPU
"""

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 64_000
    hidden_size: int = 768
    num_layers: int = 16
    num_heads: int = 12          # head_dim = 64
    num_kv_heads: int = 4        # GQA 3:1
    intermediate_size: int = 2048
    context_length: int = 8192
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class TrainConfig:
    # Dados (gerados por arpa/prepare_data.py)
    train_bin: str = "data/arpa150m/train_tokens.bin"
    val_bin: str = "data/arpa150m/val_tokens.bin"
    tokenizer_dir: str = "tokenizer-arpa-64k-clean"
    checkpoint_dir: str = "checkpoints-arpa150m"

    # Orcamento: 10B tokens (~66x params, overtraining estilo SmolLM).
    # Chinchilla (20x = 3B) e o MINIMO; a loss continua caindo bem ate ~100x.
    # Com 3.3B tokens unicos no bin, 10B = ~3 passadas (ok ate 4; depois degrada).
    max_tokens: int = 10_000_000_000

    # Batch: 4 x 16 x 8192 = 524K tokens/step
    micro_batch: int = 4
    grad_accum: int = 16

    # Otimizacao: Muon nas matrizes 2D do miolo, AdamW em embeddings/norms
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    lr: float = 6e-4          # AdamW (embeddings/norms)
    min_lr_ratio: float = 0.1
    warmup_steps: int = 300
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Cadencia
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 24
    save_every: int = 500
    keep_last: int = 3
    sample_every: int = 1000

    seed: int = 42
    compile: bool = True


def get_configs(preset: str):
    """Retorna (ModelConfig, TrainConfig) para o preset pedido."""
    model = ModelConfig()
    train = TrainConfig()

    if preset == "a100":
        pass  # defaults ja sao o preset A100

    elif preset == "local":
        # RTX 3050 4GB: contexto curto, batch minimo, sem compile (Windows)
        model.context_length = 1024
        train.micro_batch = 2
        train.grad_accum = 8
        train.max_tokens = 20_000_000
        train.compile = False
        train.eval_every = 100
        train.save_every = 200

    elif preset == "tiny":
        # Teste de pipeline em CPU: ~1M params
        model.vocab_size = 512
        model.hidden_size = 64
        model.num_layers = 2
        model.num_heads = 4
        model.num_kv_heads = 2
        model.intermediate_size = 128
        model.context_length = 128
        train.micro_batch = 2
        train.grad_accum = 2
        train.max_tokens = 40_000
        train.compile = False
        train.warmup_steps = 5
        train.log_every = 5
        train.eval_every = 20
        train.eval_batches = 2
        train.save_every = 30
        train.sample_every = 0
        train.train_bin = "data/tiny/train_tokens.bin"
        train.val_bin = "data/tiny/val_tokens.bin"
        train.checkpoint_dir = "checkpoints-tiny"

    else:
        raise ValueError(f"Preset desconhecido: {preset} (use a100 | local | tiny)")

    return model, train


def config_to_dict(model: ModelConfig, train: TrainConfig) -> dict:
    return {"model": asdict(model), "train": asdict(train)}
