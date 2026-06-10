#!/usr/bin/env python3
"""
wave_gpt_v12.py — Crystal Attention LM

Hipótese central:
  N tokens → campo Ψ (reservoir físico) → K cristais (K fixo, ~32)
  Atenção KxK em vez de NxN → memória constante com contexto arbitrário

Campo: física pura, sem parâmetros treinados.
Treina: crystal_proj + atenção + head.

Uso:
  python wave_gpt_v12.py
  python wave_gpt_v12.py --text arquivo.txt
  python wave_gpt_v12.py --text arquivo.txt --steps 5000
"""

import os, sys, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Física do campo Ψ
# ==============================================================================
C2         = 0.3
GAMMA      = 0.06
ALPHA      = 0.04
BETA       = 0.005
DT         = 0.05
WAVE_STEPS = 1     # steps de física por token — 1 é suficiente, 3x mais rápido

# ==============================================================================
# Arquitetura
# ==============================================================================
FIELD_H  = 16
FIELD_W  = 16
K        = 16      # cristais extraídos por snapshot do campo
CRYS_DIM = 3       # features por cristal: (pos_x, pos_y, amplitude)
HIDDEN   = 64      # dimensão da atenção
N_HEADS  = 4       # cabeças de atenção
N_LAYERS = 2       # camadas de atenção

# ==============================================================================
# Treino
# ==============================================================================
SEQ_LEN   = 32
BATCH     = 64     # batch maior → melhor uso da GPU com textos pequenos
LR        = 3e-4
LOG_EVERY = 100


# ==============================================================================
# Campo Ψ — reservoir puro, sem gradiente
# ==============================================================================

def _laplacian(f):
    """Laplaciano 2D via torch.roll (bordas periódicas)."""
    return (torch.roll(f, -1, -2) + torch.roll(f,  1, -2) +
            torch.roll(f, -1, -1) + torch.roll(f,  1, -1) - 4 * f)


def _parity_mask(field):
    """Paridade 8-bit vetorizada (XOR-fold, sem loop Python)."""
    q = (field * 127).to(torch.int32) & 0xFF
    q = q ^ (q >> 4)
    q = q ^ (q >> 2)
    q = q ^ (q >> 1)
    return ((q & 1) == 0).float()


def _wave_step(field, vel, stim, lam):
    """Um passo da equação de onda + atenção Energy (elementwise, sem gate).
    lam: tensor 0-d (evita especialização do torch.compile)."""
    acc = (C2 * _laplacian(field)
           - GAMMA * vel
           - ALPHA * torch.tanh(field)
           - BETA  * field ** 3
           + stim)

    # Energy attention (validada em 20newsgroups 2026-04-18)
    Q = vel.abs() * DT
    K = field.abs()
    R = torch.exp(-0.1 * (Q - K).abs())
    V = field * _parity_mask(field)
    acc = acc + lam * R * V

    vel   = vel   + DT * acc
    field = field + DT * vel
    return field, vel

# Compilar o step interno elimina overhead Python nas iterações internas
try:
    _wave_step = torch.compile(_wave_step)
except Exception:
    pass  # fallback silencioso se compile não disponível


def _extract_crystals(field, H, W):
    """Extrai K cristais top-K por amplitude do campo."""
    B    = field.shape[0]
    flat = field.view(B, -1)
    _, idx    = flat.abs().topk(K, dim=-1)
    amplitude = flat.gather(1, idx)
    pos_x     = (idx // W).float() / H
    pos_y     = (idx  % W).float() / W
    return torch.stack([pos_x, pos_y, amplitude], dim=-1)   # (B, K, 3)


@torch.no_grad()
def step_field(token_id, inject_patterns, field, vel, lam):
    H, W = field.shape[-2], field.shape[-1]
    stim = inject_patterns[token_id].view(-1, H, W)
    for _ in range(WAVE_STEPS):
        field, vel = _wave_step(field, vel, stim, lam)
    return _extract_crystals(field, H, W), field, vel


@torch.no_grad()
def process_field(input_ids, inject_patterns, lam):
    """
    Processa sequência de tokens no campo Ψ.
    Cada token injeta um padrão fixo no campo e a física propaga.
    Extrai K cristais por snapshot (top-K por amplitude).

    input_ids      : (B, T)  — índices dos chars
    inject_patterns: (V, H*W) — padrão fixo por char (buffer do modelo)

    retorna: crystal_seq (B, T, K, CRYS_DIM)
    """
    B, T = input_ids.shape
    H, W = FIELD_H, FIELD_W
    dev  = input_ids.device

    field = torch.zeros(B, H, W, device=dev)
    vel   = torch.zeros(B, H, W, device=dev)

    snapshots = []

    for t in range(T):
        stim = inject_patterns[input_ids[:, t]].view(B, H, W)
        for _ in range(WAVE_STEPS):
            field, vel = _wave_step(field, vel, stim, lam)
        snapshots.append(_extract_crystals(field, H, W))

    return torch.stack(snapshots, dim=1)                # (B, T, K, 3)


# ==============================================================================
# Modelo
# ==============================================================================

class CrystalBlock(nn.Module):
    """Atenção KxK + FF sobre os K cristais de um snapshot."""

    def __init__(self):
        super().__init__()
        self.attn  = nn.MultiheadAttention(HIDDEN, N_HEADS, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * 2),
            nn.GELU(),
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        # x: (BT, K, HIDDEN) — K cristais como sequência pequena
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class CrystalLM(nn.Module):
    """
    Crystal Attention Language Model (v12)

    Fluxo:
      input_ids (B, T)
        → campo Ψ [sem grad] → crystal_seq (B, T, K, 3)
        → crystal_proj        → (B*T, K, HIDDEN)
        → N_LAYERS x attn KxK → (B*T, K, HIDDEN)
        → mean over K          → (B, T, HIDDEN)
        → head                 → (B, T, vocab_size)

    Matriz de atenção: KxK = 32x32 (fixa).
    Transformer equivalente: SEQ_LENxSEQ_LEN = 128x128.
    """

    def __init__(self, vocab_size, lam=0.0):
        super().__init__()
        self.vocab_size = vocab_size

        # Padrões de injeção fixos — um padrão único por caractere.
        # Determinísticos (seed=42), não treináveis.
        g = torch.Generator().manual_seed(42)
        self.register_buffer(
            'inject_patterns',
            torch.randn(vocab_size, FIELD_H * FIELD_W, generator=g) * 0.15
        )
        # λ da atenção Energy (tensor pra não ser especializado por torch.compile)
        self.register_buffer('energy_lam', torch.tensor(float(lam), dtype=torch.float32))

        # Partes treináveis
        self.embed        = nn.Embedding(vocab_size, HIDDEN)   # identidade do char
        self.crystal_proj = nn.Linear(CRYS_DIM, HIDDEN)        # contexto do campo
        self.blocks       = nn.ModuleList([CrystalBlock() for _ in range(N_LAYERS)])
        self.norm         = nn.LayerNorm(HIDDEN)
        self.recurrent    = nn.Linear(HIDDEN, HIDDEN)          # estado temporal (fino)
        self.head         = nn.Linear(HIDDEN, vocab_size, bias=False)

    def forward(self, input_ids):
        """
        input_ids: (B, T) → logits (B, T, vocab_size)

        Fluxo:
          1. Campo Ψ: todos T tokens → crystal_seq (sequential, no grad)
          2. Atenção KxK: todos T steps em paralelo (rápido, B*T batch)
          3. Estado recorrente: loop fino sobre T (barato, só HIDDEN-dim)
        """
        B, T   = input_ids.shape
        device = input_ids.device

        # 1. Campo — sem gradiente, processa tudo de uma vez
        crystal_seq = process_field(input_ids, self.inject_patterns, self.energy_lam)
        char_emb    = self.embed(input_ids)                           # (B, T, HIDDEN)

        # 2. Atenção KxK sobre todos os steps em paralelo (rápido)
        BT = B * T
        x  = self.crystal_proj(crystal_seq.view(BT, K, CRYS_DIM))    # (BT, K, HIDDEN)
        x  = x + char_emb.view(BT, 1, HIDDEN)                        # + identidade
        for block in self.blocks:
            x = block(x)                                              # (BT, K, HIDDEN)
        pooled = self.norm(x.mean(dim=1)).view(B, T, HIDDEN)          # (B, T, HIDDEN)

        # 3. Estado recorrente fino — loop sobre T, mas só opera em HIDDEN-dim
        state      = torch.zeros(B, HIDDEN, device=device)
        all_logits = []
        for t in range(T):
            state = torch.tanh(self.recurrent(pooled[:, t] + state))  # (B, HIDDEN)
            all_logits.append(self.head(state))                        # (B, vocab_size)

        return torch.stack(all_logits, dim=1)                          # (B, T, vocab_size)

    def loss(self, input_ids):
        logits  = self(input_ids[:, :-1])        # (B, T-1, V)
        targets = input_ids[:, 1:].reshape(-1)   # (B*(T-1),)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets)

    @torch.no_grad()
    def step(self, token_id, field, vel, rstate):
        """
        Processa UM token incrementalmente — O(1) por step.
        O campo já carrega a assinatura de todos os tokens anteriores.
        Não precisa reprocessar o contexto todo.

        token_id : (B,)       — índice do char atual
        field    : (B, H, W)  — estado do campo (acumula história)
        vel      : (B, H, W)  — velocidade do campo
        rstate   : (B, HIDDEN) — estado recorrente da atenção

        retorna: logits (B, V), new_field, new_vel, new_rstate
        """
        # Atualiza campo com o novo token (a história está no campo)
        crystal_feats, field, vel = step_field(
            token_id, self.inject_patterns, field, vel, self.energy_lam
        )                                                    # (B, K, 3)

        # Atenção KxK sobre cristais atuais
        x = self.crystal_proj(crystal_feats)                 # (B, K, HIDDEN)
        x = x + self.embed(token_id).unsqueeze(1)
        for block in self.blocks:
            x = block(x)

        # Estado recorrente
        pooled = self.norm(x.mean(dim=1))                    # (B, HIDDEN)
        rstate = torch.tanh(self.recurrent(pooled + rstate))

        return self.head(rstate), field, vel, rstate

    @torch.no_grad()
    def generate(self, stoi, itos, seed='O campo', n=200, temp=0.8):
        """
        Geração incremental — O(n).
        Cada token processa só ele mesmo; o campo carrega a história.
        """
        self.eval()
        device = next(self.parameters()).device
        H, W   = FIELD_H, FIELD_W

        # Estados iniciais
        field  = torch.zeros(1, H, W, device=device)
        vel    = torch.zeros(1, H, W, device=device)
        rstate = torch.zeros(1, HIDDEN, device=device)

        result = list(seed)

        # Processa seed — cada char atualiza o campo acumulativamente
        logits = None
        for c in seed:
            tid    = torch.tensor([stoi.get(c, 0)], device=device)
            logits, field, vel, rstate = self.step(tid, field, vel, rstate)

        # Gera n tokens — campo já tem a assinatura da seed
        for _ in range(n):
            probs  = F.softmax(logits[0] / temp, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            result.append(itos[nxt.item()])
            logits, field, vel, rstate = self.step(nxt, field, vel, rstate)

        self.train()
        return ''.join(result)


# ==============================================================================
# Dataset
# ==============================================================================

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_len=SEQ_LEN, split='train', val_ratio=0.1):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

        # Split: primeiros (1-val_ratio)% treino, últimos val_ratio% validação
        cut    = int(len(text) * (1 - val_ratio))
        text   = text[:cut] if split == 'train' else text[cut:]

        stride = seq_len + 1
        data   = [self.stoi[c] for c in text]
        self.chunks = [
            torch.tensor(data[i : i + stride], dtype=torch.long)
            for i in range(0, len(data) - stride, stride)
        ]

    def __len__(self):  return len(self.chunks)
    def __getitem__(self, i): return self.chunks[i]


# ==============================================================================
# Texto de exemplo (fallback)
# ==============================================================================

SAMPLE_PT = """
A linguagem humana carrega estrutura em múltiplas escalas temporais.
Letras formam sílabas. Sílabas formam palavras. Palavras formam frases.
O campo de ondas recebe perturbações e cristaliza padrões estáveis.
Cada cristal preserva a informação que o campo considerou relevante.
A atenção opera sobre os cristais, não sobre a sequência inteira.
A matriz de atenção é pequena e constante, independente do contexto.
O campo Ψ comprime N tokens em K cristais onde K é fixo pela física.
Regiões de alta amplitude e baixa variância cristalizam primeiro.
Cristais competem por sobrevivência através do mecanismo de HP.
Cristais que ressoam com o campo ganham vida. Os outros morrem.
O resultado é uma representação esparsa e estável do input.
""" * 400


# ==============================================================================
# Treino
# ==============================================================================

def main():
    global SEQ_LEN, BATCH, WAVE_STEPS
    # Garante UTF-8 no stdout (necessário no Windows com terminais cp1252)
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',    type=str, default=None,
                        help='Arquivo .txt para treino')
    parser.add_argument('--steps',   type=int, default=2000,
                        help='Steps de treino (default: 2000)')
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN,
                        help=f'Contexto por sequência (default: {SEQ_LEN}). '
                             'Atenção permanece KxK independente deste valor.')
    parser.add_argument('--device',  type=str, default='auto')
    parser.add_argument('--save',    type=str, default=None,
                        help='Salva checkpoint ao final (ex: clm_ckpt.pt)')
    parser.add_argument('--load',    type=str, default=None,
                        help='Carrega checkpoint antes de treinar')
    parser.add_argument('--lam',     type=float, default=0.0,
                        help='λ da atenção Energy (0=desligada; teste 0.05)')
    parser.add_argument('--seed',    type=int, default=42,
                        help='Seed para reprodutibilidade')
    parser.add_argument('--wave_steps', type=int, default=WAVE_STEPS,
                        help=f'Steps de física por token (default: {WAVE_STEPS})')
    args = parser.parse_args()
    WAVE_STEPS = args.wave_steps

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Aplica seq_len dinamicamente (permite testar sem editar constantes)
    if args.seq_len != SEQ_LEN:
        SEQ_LEN = args.seq_len
        # Ajusta batch para manter uso de memória similar
        BATCH = max(8, 64 * 32 // SEQ_LEN)
        print(f"[cfg] SEQ_LEN={SEQ_LEN}  BATCH={BATCH} (ajustado)")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Texto
    if args.text and os.path.exists(args.text):
        text = open(args.text, encoding='utf-8').read()
        print(f"Texto: {args.text}  ({len(text):,} chars)")
    else:
        text = SAMPLE_PT
        print(f"Texto embutido  ({len(text):,} chars)")

    # Dataset — 90% treino, 10% validação
    ds      = CharDataset(text, split='train')
    ds_val  = CharDataset(text, split='val')
    loader  = torch.utils.data.DataLoader(ds,     batch_size=BATCH, shuffle=True,  drop_last=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=BATCH, shuffle=False, drop_last=True)

    # Modelo
    model   = CrystalLM(ds.vocab_size, lam=args.lam).to(device)
    print(f"[cfg] energy_lam = {args.lam}  (atenção Energy {'ATIVA' if args.lam > 0 else 'OFF'})")

    # Carrega checkpoint se pedido
    if args.load and os.path.exists(args.load):
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f"[ckpt] Carregado: {args.load}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fixed = sum(b.numel() for b in model.buffers())
    print(f"  Treino: {len(ds)} seqs | Validação: {len(ds_val)} seqs")

    attn_seq  = K          # tamanho da sequência na atenção (KxK, fixo)
    attn_std  = SEQ_LEN    # tamanho que o transformer usaria (NxN, escala com contexto)

    print(f"\n{'='*55}")
    print(f"  CrystalLM v12 — Crystal Attention")
    print(f"{'='*55}")
    print(f"  Device  : {device}")
    print(f"  Vocab   : {ds.vocab_size} chars")
    print(f"  Sequências : {len(ds)}")
    print(f"  Params treináveis : {n_train:,}")
    print(f"  Params fixos (campo): {n_fixed:,}")
    print(f"")
    print(f"  Campo     : {FIELD_H}x{FIELD_W}  |  K={K} cristais")
    print(f"  Atenção   : {attn_seq}x{attn_seq} + estado recorrente HIDDEN={HIDDEN}  (transformer usaria {attn_std}x{attn_std})")
    print(f"  Redução   : {attn_std**2 / (2*attn_seq)**2:.0f}x menor")
    print(f"{'='*55}\n")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    losses = []
    step   = 0
    t0     = time.time()

    print(f"Treinando por {args.steps} steps...\n")

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = batch.to(device)

            opt.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            step += 1

            if step % LOG_EVERY == 0:
                avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY

                # Validation loss
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        val_losses.append(model.loss(vb.to(device)).item())
                        if len(val_losses) >= 20:  # ~20 batches é suficiente
                            break
                val_avg = sum(val_losses) / len(val_losses)
                model.train()

                gap = val_avg - avg
                flag = "OK generaliza" if gap < 0.3 else ("ALERTA overfitting" if gap > 1.0 else "vigiando")
                print(f"step {step:5d} | train {avg:.4f} | val {val_avg:.4f} | gap {gap:+.4f} | {flag} | {time.time()-t0:.1f}s")
                sample = model.generate(ds.stoi, ds.itos, n=120, temp=0.8)
                print(f"  > {sample}\n")

    final_loss = sum(losses[-100:]) / min(len(losses), 100)
    print(f"\n{'='*55}")
    print(f"  Loss final: {final_loss:.4f}")
    print(f"  (aleatório seria: {math.log(ds.vocab_size):.4f})")
    print(f"{'='*55}")
    print("\nGeração final (temp=0.7):")
    print(model.generate(ds.stoi, ds.itos, n=400, temp=0.7))

    # Salva checkpoint
    if args.save:
        ckpt = {
            'model':      model.state_dict(),
            'vocab_size': ds.vocab_size,
            'stoi':       ds.stoi,
            'itos':       ds.itos,
            'step':       step,
            'loss':       final_loss,
            'seq_len':    SEQ_LEN,
        }
        torch.save(ckpt, args.save)
        print(f"\n[ckpt] Salvo: {args.save}")


if __name__ == '__main__':
    main()
