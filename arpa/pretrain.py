"""
pretrain.py - Pre-treino do Arpa-150M.

Uso (na raiz do repo):
    python arpa/pretrain.py --config a100
    python arpa/pretrain.py --config a100 --resume latest
    python arpa/pretrain.py --config tiny          # smoke test do pipeline

Eficiencia:
    - BF16 autocast + TF32 nas matmuls
    - torch.compile (Linux/CUDA)
    - AdamW fused, weight decay so em tensores 2D
    - bins memmap: resume O(1), sem dataloader, sem re-iterar dataset
"""

import argparse
import json
import math
import os
import sys
import time
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arpa.config import get_configs, config_to_dict
from arpa.data import TokenBin
from arpa.model import Arpa

SAMPLE_PROMPTS = ["O Brasil e um pais", "A inteligencia artificial", "def fibonacci(n):"]


def lr_factor_at(step, total_steps, cfg):
    """Fator multiplicativo [min_lr_ratio, 1.0] aplicado ao base_lr de cada grupo."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    lo = cfg.min_lr_ratio
    return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * min(progress, 1.0)))


def configure_optimizers(model, cfg, fused: bool):
    """Muon nas matrizes 2D do miolo; AdamW em embeddings e norms.

    Retorna lista de otimizadores. Cada param group guarda base_lr para
    o schedule aplicar um fator multiplicativo comum.
    """
    from arpa.muon import Muon

    embed_params = list(model.embed.parameters())
    embed_ids = {id(p) for p in embed_params}
    muon_params, adamw_other = [], []
    for p in model.parameters():
        if not p.requires_grad or id(p) in embed_ids:
            continue
        (muon_params if p.ndim == 2 else adamw_other).append(p)

    muon = Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum)
    adamw = torch.optim.AdamW(
        [
            {"params": embed_params, "weight_decay": cfg.weight_decay},
            {"params": adamw_other, "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=fused,
    )
    for opt in (muon, adamw):
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in embed_params + adamw_other)
    print(f"Otimizadores: Muon {n_muon / 1e6:.1f}M params | AdamW {n_adamw / 1e6:.1f}M params")
    return [muon, adamw]


def save_checkpoint(path, raw_model, optimizers, step, val_loss, mcfg, tcfg):
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "step": step,
        "val_loss": val_loss,
        "config": config_to_dict(mcfg, tcfg),
    }, tmp)
    os.replace(tmp, path)


def rotate_checkpoints(ckpt_dir, keep_last):
    ckpts = sorted(glob(os.path.join(ckpt_dir, "step_*.pt")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    for old in ckpts[:-keep_last]:
        os.remove(old)


@torch.no_grad()
def evaluate(model, val_bin, tcfg, mcfg, device, autocast_ctx):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(tcfg.eval_batches):
        x, y = val_bin.get_batch(tcfg.micro_batch, mcfg.context_length, device, gen)
        with autocast_ctx():
            losses.append(model(x, y).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(raw_model, tokenizer, device, log):
    if tokenizer is None:
        return
    for prompt in SAMPLE_PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        out = raw_model.generate(ids, max_new_tokens=60, temperature=0.8, top_p=0.9)
        text = tokenizer.decode(out[0].tolist()).replace("\n", " ")
        log(f"  > {text[:200]}")
    raw_model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="a100", choices=["a100", "local", "tiny"])
    parser.add_argument("--resume", default=None, help="latest | step_N | caminho.pt")
    parser.add_argument("--max-tokens", type=float, default=None,
                        help="sobrescreve o orcamento de tokens (ex: 10e9)")
    parser.add_argument("--train-bin", default=None,
                        help="sobrescreve o bin de treino (ex: fase de annealing)")
    args = parser.parse_args()

    mcfg, tcfg = get_configs(args.config)
    if args.max_tokens:
        tcfg.max_tokens = int(args.max_tokens)
    if args.train_bin:
        tcfg.train_bin = args.train_bin
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)

    log_path = os.path.join(tcfg.checkpoint_dir, "train_log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)

    def log_metrics(d):
        log_f.write(json.dumps(d) + "\n")
        log_f.flush()

    torch.manual_seed(tcfg.seed)
    torch.set_float32_matmul_precision("high")

    # Tokenizer so para amostras de texto durante o treino (opcional)
    tokenizer = None
    if os.path.isdir(tcfg.tokenizer_dir):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tcfg.tokenizer_dir)
            assert tokenizer.vocab_size <= mcfg.vocab_size, \
                f"Tokenizer {tokenizer.vocab_size} > modelo {mcfg.vocab_size}"
        except Exception as e:
            log(f"[aviso] tokenizer indisponivel ({e}); sem amostras de texto")

    train_bin = TokenBin(tcfg.train_bin)
    val_bin = TokenBin(tcfg.val_bin)
    log(f"Dados: train={len(train_bin):,} tokens | val={len(val_bin):,} tokens")

    model = Arpa(mcfg).to(device)
    raw_model = model
    n_params = model.num_params()
    log(f"Modelo: {n_params / 1e6:.1f}M params | ctx={mcfg.context_length} | device={device}")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    elif device == "cuda":
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.float16)
        log("[aviso] GPU sem BF16 — usando FP16")
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext

    optimizers = configure_optimizers(model, tcfg, fused=(device == "cuda"))
    muon_opt = optimizers[0]

    tokens_per_step = tcfg.micro_batch * tcfg.grad_accum * mcfg.context_length
    total_steps = tcfg.max_tokens // tokens_per_step
    log(f"Batch efetivo: {tokens_per_step:,} tokens/step | {total_steps:,} steps "
        f"| orcamento {tcfg.max_tokens / 1e9:.2f}B tokens")

    # Resume ANTES do compile (state_dict sem prefixo _orig_mod)
    start_step, best_val = 0, float("inf")
    if args.resume:
        if args.resume == "latest":
            ckpts = sorted(glob(os.path.join(tcfg.checkpoint_dir, "step_*.pt")),
                           key=lambda p: int(p.split("_")[-1].split(".")[0]))
            if not ckpts:
                sys.exit(f"Nenhum checkpoint em {tcfg.checkpoint_dir}")
            ckpt_path = ckpts[-1]
        elif os.path.exists(args.resume):
            ckpt_path = args.resume
        else:
            ckpt_path = os.path.join(tcfg.checkpoint_dir, f"{args.resume}.pt")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        for opt, sd in zip(optimizers, state["optimizers"]):
            opt.load_state_dict(sd)
        start_step = state["step"]
        best_val = state.get("val_loss") or float("inf")
        log(f"Resumido de {ckpt_path} (step {start_step})")

    if tcfg.compile and device == "cuda" and sys.platform != "win32":
        log("Compilando modelo (torch.compile)...")
        model = torch.compile(model)

    # GPU vector fp16/fp32: GradScaler so se fp16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

    model.train()
    t0 = time.time()
    tokens_seen_window = 0
    loss_window = []

    try:
        for step in range(start_step, total_steps):
            factor = lr_factor_at(step, total_steps, tcfg)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * factor
            lr = tcfg.lr * factor
            # Momentum warmup do Muon (speedrun): 0.85 -> alvo nos primeiros steps
            warm = min(1.0, (step + 1) / max(1, tcfg.warmup_steps))
            for g in muon_opt.param_groups:
                g["momentum"] = 0.85 + warm * (tcfg.muon_momentum - 0.85)

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro in range(tcfg.grad_accum):
                # Gerador deterministico por (step, micro): resume exato
                gen = torch.Generator().manual_seed(
                    tcfg.seed + step * tcfg.grad_accum + micro)
                x, y = train_bin.get_batch(tcfg.micro_batch, mcfg.context_length,
                                           device, gen)
                with autocast_ctx():
                    loss = model(x, y) / tcfg.grad_accum
                scaler.scale(loss).backward()
                step_loss += loss.item()

            for opt in optimizers:
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()

            tokens_seen_window += tokens_per_step
            loss_window.append(step_loss)

            if (step + 1) % tcfg.log_every == 0:
                dt = time.time() - t0
                tps = tokens_seen_window / dt
                avg_loss = sum(loss_window) / len(loss_window)
                mem = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0
                log(f"step {step + 1:>6}/{total_steps} | loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | gnorm {grad_norm:.2f} | {tps / 1e3:.0f}K tok/s | "
                    f"mem {mem:.1f}GB")
                log_metrics({"step": step + 1, "loss": avg_loss, "lr": lr,
                             "tok_s": tps, "grad_norm": float(grad_norm)})
                t0, tokens_seen_window, loss_window = time.time(), 0, []

            if tcfg.eval_every and (step + 1) % tcfg.eval_every == 0:
                val_loss = evaluate(model, val_bin, tcfg, mcfg, device, autocast_ctx)
                log(f"  [eval] val_loss {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.1f}")
                log_metrics({"step": step + 1, "val_loss": val_loss})
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(os.path.join(tcfg.checkpoint_dir, "best.pt"),
                                    raw_model, optimizers, step + 1, val_loss, mcfg, tcfg)
                    log(f"  [best] salvo (val_loss {val_loss:.4f})")
                t0, tokens_seen_window, loss_window = time.time(), 0, []

            if tcfg.save_every and (step + 1) % tcfg.save_every == 0:
                path = os.path.join(tcfg.checkpoint_dir, f"step_{step + 1}.pt")
                save_checkpoint(path, raw_model, optimizers, step + 1, None, mcfg, tcfg)
                rotate_checkpoints(tcfg.checkpoint_dir, tcfg.keep_last)
                log(f"  [ckpt] {path}")
                t0, tokens_seen_window, loss_window = time.time(), 0, []

            if tcfg.sample_every and (step + 1) % tcfg.sample_every == 0:
                sample_text(raw_model, tokenizer, device, log)
                t0, tokens_seen_window, loss_window = time.time(), 0, []

    except KeyboardInterrupt:
        log("\nInterrompido — salvando checkpoint de emergencia...")
        save_checkpoint(os.path.join(tcfg.checkpoint_dir, "interrupt.pt"),
                        raw_model, optimizers, step, None, mcfg, tcfg)
        return

    save_checkpoint(os.path.join(tcfg.checkpoint_dir, "final.pt"),
                    raw_model, optimizers, total_steps, best_val, mcfg, tcfg)
    log(f"Treino completo. best val_loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
