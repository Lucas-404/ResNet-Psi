"""
Wave Transformer — Transformer com Attention por Campo de Ondas

Substitui Q×K×V por propagação de ondas num campo fixo.

Transformer normal:
  tokens → Q,K,V → softmax(QK^T/√d) × V → saída
  Memória: O(N²) pela matriz QK^T

Wave Transformer:
  tokens → emitem ondas num campo fixo → ondas propagam e interferem → leitura do campo
  Memória: O(F²) onde F = tamanho do campo (fixo, independente de N tokens)

O campo é diferenciável (Laplaciano = convolução 3×3, poucos steps).
Treina com backprop normal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# WAVE ATTENTION — O coração do novo sistema
# ══════════════════════════════════════════════════════════════════════════════
#
# Em vez de Q×K^T (N×N), cada token emite uma onda num campo 2D fixo.
# As ondas propagam (Laplaciano), interferem, e o resultado é lido de volta.
#
# Tokens próximos no espaço de embedding emitem ondas parecidas → interferem
# construtivamente → campo amplifica a relação.
# Tokens distantes → interferência destrutiva → campo ignora.
#
# É attention por física em vez de álgebra linear.

class WaveAttention(nn.Module):
    def __init__(self, embed_dim=64, field_size=16, n_steps=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_size = field_size
        self.n_steps = n_steps

        # Projeção: token → amplitude de onda no campo
        # Cada token emite um padrão no campo F×F
        self.to_wave = nn.Linear(embed_dim, field_size * field_size)

        # Parâmetros físicos treináveis
        self.c2 = nn.Parameter(torch.tensor(0.3))       # velocidade de onda
        self.gamma = nn.Parameter(torch.tensor(0.06))    # amortecimento
        self.alpha = nn.Parameter(torch.tensor(0.04))    # não-linearidade

        # Kernel do Laplaciano (fixo, não treinável)
        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

        # Leitura: campo F×F → vetor de embed_dim
        self.from_field = nn.Linear(field_size * field_size, embed_dim)

        # Projeção de saída
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def propagate(self, field, velocity, source, active):
        """Um step de propagação de onda — diferenciável."""
        dt = 0.1

        # Injetar fonte
        if active:
            field = field + source * dt

        # Laplaciano via convolução (diferenciável)
        B_N, H, W = field.shape
        f_pad = F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular')
        lap = F.conv2d(f_pad, self.lap_kernel).squeeze(1)

        # Equação de onda: propagação + amortecimento + não-linearidade
        c2 = torch.clamp(self.c2, 0.01, 1.0)
        gamma = torch.clamp(self.gamma, 0.01, 0.5)
        alpha = torch.clamp(self.alpha, 0.0, 0.2)

        acc = c2 * lap - gamma * velocity + alpha * torch.tanh(field) * field
        velocity = velocity + acc * dt
        field = field + velocity * dt

        return field, velocity

    def forward(self, x):
        """
        x: (B, N, D) — B batches, N tokens, D dimensão
        retorna: (B, N, D) — mesma shape, mas com informação misturada pelo campo
        """
        B, N, D = x.shape
        FS = self.field_size

        # Cada token gera um padrão de onda no campo
        # (B, N, D) → (B, N, F*F) → (B, N, F, F)
        wave_patterns = self.to_wave(x).view(B, N, FS, FS)

        # Somar todas as ondas dos tokens no campo (superposição)
        # (B, N, F, F) → (B, F, F)
        source = wave_patterns.sum(dim=1)

        # Inicializar campo
        field = torch.zeros(B, FS, FS, device=x.device, dtype=x.dtype)
        velocity = torch.zeros_like(field)

        # Propagar — poucos steps, tudo diferenciável
        for s in range(self.n_steps):
            active = s < self.n_steps // 2  # metade com estímulo, metade livre
            field, velocity = self.propagate(field, velocity, source, active)

        # O campo agora tem o resultado da interferência de todos os tokens.
        # Cada token "lê" o campo de volta — como se perguntasse
        # "o que o campo acumulou que é relevante pra mim?"

        # Modular a leitura pelo padrão original de cada token
        # (B, N, F, F) × (B, 1, F, F) → (B, N, F, F)
        field_expanded = field.unsqueeze(1).expand_as(wave_patterns)
        token_reads = wave_patterns * field_expanded  # cada token lê sua "região"

        # (B, N, F, F) → (B, N, F*F) → (B, N, D)
        token_reads = token_reads.view(B, N, FS * FS)
        out = self.from_field(token_reads)

        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# WAVE TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class WaveTransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, field_size=16, n_steps=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WaveAttention(embed_dim, field_size, n_steps)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# PATCH EMBEDDING (mesmo do Transformer original)
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size

        x = x.unfold(1, p, p).unfold(2, p, p)
        x = x.contiguous().view(B, -1, p * p)
        x = self.proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        return x


# ══════════════════════════════════════════════════════════════════════════════
# WAVE VISION TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class WaveVisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, n_classes=10,
                 embed_dim=64, depth=4, field_size=16, n_steps=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.blocks = nn.Sequential(*[
            WaveTransformerBlock(embed_dim, field_size, n_steps)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)      # (B, 17, 64)
        x = self.blocks(x)            # (B, 17, 64) — wave attention entre patches
        x = self.norm(x)
        cls_token = x[:, 0]           # (B, 64)
        return self.head(cls_token)    # (B, 10)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER ORIGINAL (pra comparação lado a lado)
# ══════════════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, n_classes=10,
                 embed_dim=64, depth=4, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# TREINO E COMPARAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from torchvision import datasets, transforms

    print(f"Dispositivo: {DEVICE}")

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256, shuffle=False)

    def treinar(model, nome, epochs=10):
        model = model.to(DEVICE)
        n_params = model.count_params()
        print(f"\n{'='*60}")
        print(f"{nome}: {n_params:,} params")
        print(f"{'='*60}")

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            t0 = time.time()

            for imgs, labels in train_loader:
                imgs = imgs.squeeze(1).to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(imgs)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(imgs)

            acc_train = 100 * correct / total
            avg_loss = total_loss / total
            dt = time.time() - t0

            # Teste
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.squeeze(1).to(DEVICE)
                    labels = labels.to(DEVICE)
                    logits = model(imgs)
                    correct_test += (logits.argmax(1) == labels).sum().item()
                    total_test += len(imgs)

            acc_test = 100 * correct_test / total_test
            print(f"  Epoca {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}  "
                  f"train={acc_train:.1f}%  test={acc_test:.1f}%  ({dt:.0f}s)")

        return acc_test, n_params

    # ── Treinar os dois ──
    EPOCHS = 10

    print("\n" + "#"*60)
    print("# COMPARACAO: Transformer vs Wave Transformer")
    print("#"*60)

    # Transformer original
    vit = VisionTransformer(
        img_size=28, patch_size=7, n_classes=10,
        embed_dim=64, depth=4, n_heads=4
    )
    acc_vit, params_vit = treinar(vit, "Vision Transformer (QKV)", EPOCHS)

    # Wave Transformer
    wave_vit = WaveVisionTransformer(
        img_size=28, patch_size=7, n_classes=10,
        embed_dim=64, depth=4, field_size=16, n_steps=8
    )
    acc_wave, params_wave = treinar(wave_vit, "Wave Transformer (campo de ondas)", EPOCHS)

    # ── Resumo ──
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"  {'Modelo':<35s}  {'Params':>10s}  {'Test':>7s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*7}")
    print(f"  {'Transformer (QKV attention)':<35s}  {params_vit:>10,}  {acc_vit:>6.1f}%")
    print(f"  {'Wave Transformer (campo ondas)':<35s}  {params_wave:>10,}  {acc_wave:>6.1f}%")
    print(f"  {'-'*35}  {'-'*10}  {'-'*7}")

    # Memória do attention
    N = 17  # tokens (16 patches + cls)
    mem_qkv = N * N * 4  # QK^T matrix, 4 heads, float
    mem_wave = 16 * 16    # campo fixo
    print(f"\n  Memoria do attention por amostra:")
    print(f"    Transformer: {N}x{N} = {N*N} (escala com N²)")
    print(f"    Wave:        {16}x{16} = {16*16} (fixo, independente de N)")
    print(f"{'='*60}")
