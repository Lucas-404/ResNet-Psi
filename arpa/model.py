"""
model.py - Transformer LLaMA-style em PyTorch puro, otimizado para treino.

- RMSNorm pre-norm
- RoPE (rotary embeddings, estilo GPT-NeoX)
- SwiGLU MLP
- GQA (grouped-query attention)
- Flash Attention via F.scaled_dot_product_attention
- Embeddings de entrada/saida amarrados (tied)
- KV-cache para geracao

Sem dependencia de HuggingFace no caminho do treino.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from arpa.config import ModelConfig


# ------------------------------------------------------------------ RoPE

def precompute_rope(head_dim: int, max_seq: int, theta: float):
    """cos/sin (max_seq, head_dim/2) em float32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd). cos/sin: (T, hd/2)."""
    cos = torch.cat((cos, cos), dim=-1)[None, None, :, :]
    sin = torch.cat((sin, sin), dim=-1)[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, KV, T, hd) -> (B, KV*n_rep, T, hd) sem copia desnecessaria."""
    if n_rep == 1:
        return x
    b, kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, kv, n_rep, t, hd).reshape(b, kv * n_rep, t, hd)


# ------------------------------------------------------------------ Camadas

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.n_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        # QK-Norm (Qwen3/Gemma3/OLMo2): evita saturacao do softmax, permite LR maior
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.norm_eps)

    def forward(self, x, cos, sin, cache: Optional[list] = None, layer_idx: int = 0):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            past = cache[layer_idx]
            if past is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            cache[layer_idx] = (k, v)

        k = repeat_kv(k, self.n_heads // self.n_kv)
        v = repeat_kv(v, self.n_heads // self.n_kv)

        # Treino/prefill: causal. Decode com cache (T=1): atende tudo.
        is_causal = cache is None or q.size(2) > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer_idx)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ------------------------------------------------------------------ Modelo

class Arpa(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual scaling (GPT-2 style): projecoes de saida menores
        scale = 0.02 / math.sqrt(2 * cfg.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            return n  # lm_head compartilha o tensor do embed, ja nao conta dobrado
        return n

    def forward(self, idx, targets=None, cache=None, start_pos: int = 0):
        T = idx.size(1)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1),
                ignore_index=-100,
            )
            return loss

        # Inferencia: so o ultimo token precisa de logits
        logits = self.lm_head(x[:, -1:, :])
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_p: float = 0.9, stop_ids=None):
        """Geracao autoregressiva com KV-cache. idx: (1, T)."""
        self.eval()
        stop_ids = set(stop_ids or [])
        cache = [None] * self.cfg.num_layers

        # Prefill do prompt inteiro
        idx = idx[:, -self.cfg.context_length:]
        logits = self(idx, cache=cache, start_pos=0)
        pos = idx.size(1)

        out = idx
        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits_last.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits_last / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = probs.sort(descending=True)
                    cum = sorted_probs.cumsum(dim=-1)
                    mask = cum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    pick = torch.multinomial(sorted_probs, 1)
                    next_id = sorted_idx.gather(-1, pick)
                else:
                    next_id = torch.multinomial(probs, 1)

            out = torch.cat([out, next_id], dim=1)
            if next_id.item() in stop_ids or pos >= self.cfg.context_length - 1:
                break
            logits = self(next_id, cache=cache, start_pos=pos)
            pos += 1

        return out
