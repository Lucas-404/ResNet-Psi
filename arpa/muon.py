"""
muon.py - Otimizador Muon (Momentum Orthogonalized by Newton-Schulz).

Origem: modded-nanogpt speedrun (Keller Jordan, 2024-2025). ~2x mais eficiente
por FLOP que AdamW para matrizes 2D ocultas. Uso correto:

    Muon  -> matrizes 2D do miolo (atencao, MLP)
    AdamW -> embeddings, lm_head, norms (tudo que nao e matriz oculta)

A atualizacao: SGD com momentum Nesterov, seguido de ortogonalizacao
aproximada da matriz de update via 5 iteracoes de Newton-Schulz em bf16.
"""

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Aproxima a ortogonalizacao de G (substitui SVD, roda em bf16)."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon e so para matrizes 2D"
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Escala pela razao de aspecto (matrizes nao-quadradas)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u.to(p.dtype), alpha=-lr * scale)
        return loss
