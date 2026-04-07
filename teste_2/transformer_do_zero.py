"""
Transformer do Zero — Classificação de Imagens

Cada peça explicada. Sem biblioteca pronta.
Mesma tarefa que a ResNet-Ψ: MNIST 28x28 → 10 classes.

FUNDAMENTO:
A ideia central do Transformer é UMA coisa: attention.
Attention = "pra entender essa parte, quais outras partes importam?"

No MNIST: pra entender o pixel (14,14), quais outros pixels
são relevantes? O attention aprende isso durante o treino.

Pipeline:
  Imagem 28x28
  → cortar em patches 7x7 (16 patches de 49 pixels)
  → cada patch vira um vetor (embedding)
  → attention entre patches (quais patches importam pra quais)
  → classificar

A ResNet-Ψ faz isso com física. O Transformer faz com álgebra linear + treino.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# PEÇA 1: PATCH EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
#
# Corta a imagem em pedaços (patches) e transforma cada pedaço num vetor.
# Imagem 28x28 com patches 7x7 = 16 patches, cada um com 49 pixels.
# Cada patch é projetado num vetor de dimensão D (ex: 64).
#
# É como olhar pra imagem em pedaços em vez de pixel por pixel.

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 16 patches

        # Projeção linear: cada patch (49 pixels) → vetor de embed_dim
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

        # Token de classe: um vetor extra que vai "absorver" informação
        # de todos os patches via attention. É ele que classifica no final.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Posição: sem isso o Transformer não sabe ONDE cada patch está.
        # São vetores aprendidos, um por posição (16 patches + 1 cls token = 17).
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        # x: (B, 28, 28)
        B = x.shape[0]
        p = self.patch_size

        # Cortar em patches: (B, 28, 28) → (B, 16, 49)
        # Reshape pra grid de patches
        x = x.unfold(1, p, p).unfold(2, p, p)  # (B, 4, 4, 7, 7)
        x = x.contiguous().view(B, -1, p * p)    # (B, 16, 49)

        # Projetar cada patch: (B, 16, 49) → (B, 16, 64)
        x = self.proj(x)

        # Adicionar cls_token: (B, 17, 64)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Adicionar posição: cada patch sabe onde está
        x = x + self.pos_embed

        return x  # (B, 17, 64)


# ══════════════════════════════════════════════════════════════════════════════
# PEÇA 2: SELF-ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
#
# O coração do Transformer. Pra cada patch, calcula:
#   "quais outros patches são relevantes pra mim?"
#
# Como funciona:
#   1. Cada patch gera 3 vetores: Query (Q), Key (K), Value (V)
#      - Q = "o que eu estou procurando?"
#      - K = "o que eu tenho pra oferecer?"
#      - V = "qual informação eu carrego?"
#
#   2. Attention = softmax(Q · K^T / √d) · V
#      - Q · K^T = "quão relevante é cada patch pra mim?"
#      - softmax = normaliza pra somar 1 (probabilidades)
#      - × V = "pego a informação dos patches mais relevantes"
#
# Resultado: cada patch agora contém informação de todos os outros,
# ponderada por relevância. Patches distantes podem se comunicar diretamente.

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads  # 64/4 = 16 por cabeça

        # Projeções Q, K, V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape  # (B, 17, 64)

        # Gerar Q, K, V: (B, 17, 192) → 3 × (B, 17, 64)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: Q · K^T / √d
        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)

        # Aplicar attention nos values
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, 64)

        return self.proj_out(out)


# ══════════════════════════════════════════════════════════════════════════════
# PEÇA 3: TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════
#
# Um bloco completo:
#   1. Layer Norm → Self-Attention → Residual
#   2. Layer Norm → MLP (feedforward) → Residual
#
# Layer Norm = normaliza os valores (estabiliza o treino)
# Residual = soma a entrada com a saída (permite gradiente fluir fácil)
# MLP = duas camadas lineares com GELU (expande e comprime)

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
        # Attention + residual
        x = x + self.attn(self.norm1(x))
        # MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# PEÇA 4: VISION TRANSFORMER (ViT) COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
#
# Junta tudo:
#   Imagem → Patches → Embedding → N blocos Transformer → cls_token → Classificação
#
# O cls_token passa por todos os blocos de attention, absorvendo informação
# de todos os patches. No final, ele é o "resumo" da imagem inteira.
# Uma camada linear no cls_token dá a classe.

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
        # x: (B, 28, 28)
        x = self.patch_embed(x)      # (B, 17, 64)
        x = self.blocks(x)            # (B, 17, 64) — attention entre patches
        x = self.norm(x)
        cls_token = x[:, 0]           # (B, 64) — pega só o cls_token
        return self.head(cls_token)    # (B, 10)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# TREINO + TESTE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from torchvision import datasets, transforms

    print(f"Dispositivo: {DEVICE}")

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256, shuffle=False)

    # ── Modelo ──
    model = VisionTransformer(
        img_size=28, patch_size=7, n_classes=10,
        embed_dim=64, depth=4, n_heads=4
    ).to(DEVICE)

    n_params = model.count_params()
    print(f"\nVision Transformer: {n_params:,} parâmetros")
    print(f"  embed_dim=64, depth=4, heads=4, patches=7x7")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # ── Treino ──
    EPOCHS = 10
    print(f"\nTreinando por {EPOCHS} épocas...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.squeeze(1).to(DEVICE)  # (B, 28, 28)
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
        print(f"  Época {epoch+1:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"train={acc_train:.1f}%  test={acc_test:.1f}%  ({dt:.0f}s)")

    # ── Resumo ──
    print(f"\n{'='*60}")
    print(f"COMPARAÇÃO")
    print(f"{'='*60}")
    print(f"  Vision Transformer ({n_params:,} params, {EPOCHS} épocas): {acc_test:.1f}%")
    print(f"  ResNet-Ψ (0 params, 0 treino):                        77.4%")
    print(f"{'='*60}")
    print(f"\n  O Transformer PRECISA de treino pra funcionar.")
    print(f"  Sem treino (pesos aleatórios) = 10% = chance.")
    print(f"  A ResNet-Ψ faz 77.4% sem treinar nada.")
