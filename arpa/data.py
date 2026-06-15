"""
data.py - Leitura dos bins pre-tokenizados (uint16, memmap).

O treino le exclusivamente bins gerados por arpa/prepare_data.py.
Amostragem aleatoria de janelas (estilo nanoGPT): resume deterministico,
zero estado de dataloader, throughput maximo.
"""

import os

import numpy as np
import torch


class TokenBin:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Bin nao encontrado: {path}\n"
                f"Gere com: python arpa/prepare_data.py"
            )
        self.path = path
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) < 2:
            raise ValueError(f"Bin vazio: {path}")

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size: int, block_size: int, device: str,
                  generator: torch.Generator):
        """Janelas aleatorias (x, y) com y = x deslocado em 1."""
        max_start = len(self.data) - block_size - 1
        ix = torch.randint(max_start, (batch_size,), generator=generator)
        xs = np.stack([self.data[i:i + block_size] for i in ix.tolist()])
        ys = np.stack([self.data[i + 1:i + 1 + block_size] for i in ix.tolist()])
        x = torch.from_numpy(xs.astype(np.int64))
        y = torch.from_numpy(ys.astype(np.int64))
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


class BinWriter:
    """Escreve tokens uint16 em streaming, com flush por buffer.

    resume_tokens > 0 retoma um bin existente: trunca para o ultimo
    checkpoint (descarta qualquer cauda parcial) e abre em append.
    """

    def __init__(self, path: str, buffer_tokens: int = 8_000_000,
                 resume_tokens: int = 0):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        if resume_tokens > 0 and os.path.exists(path):
            with open(path, "r+b") as fp:
                fp.truncate(resume_tokens * 2)  # uint16 = 2 bytes/token
            self.f = open(path, "ab")
            self.total = resume_tokens
        else:
            self.f = open(path, "wb")
            self.total = 0
        self.buffer = np.empty(buffer_tokens, dtype=np.uint16)
        self.fill = 0

    def write(self, token_ids):
        n = len(token_ids)
        if self.fill + n > len(self.buffer):
            self.flush()
        if n > len(self.buffer):  # documento gigante: escreve direto
            np.asarray(token_ids, dtype=np.uint16).tofile(self.f)
        else:
            self.buffer[self.fill:self.fill + n] = token_ids
            self.fill += n
        self.total += n

    def flush(self):
        if self.fill:
            self.buffer[:self.fill].tofile(self.f)
            self.fill = 0

    def sync(self):
        """Garante que tudo esta no disco (apos flush). self.total == bytes/2."""
        self.flush()
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.flush()
        self.f.close()
