import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _positive(raw: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    return F.softplus(raw) + floor


def _make_laplacian() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)


def _soft_clip(x: torch.Tensor, limit: float) -> torch.Tensor:
    return limit * torch.tanh(x / limit)


@dataclass
class PsiTrainConfig:
    field_size: int = 32
    steps: int = 16
    stim_steps: int = 8
    hidden_channels: int = 24
    num_classes: int = 10


class InputProjector(nn.Module):
    def __init__(self, field_size: int, hidden_channels: int):
        super().__init__()
        self.field_size = field_size
        self.input_mix = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Entrada invalida com shape {tuple(x.shape)}")

        x = F.interpolate(
            x,
            size=(self.field_size, self.field_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.input_mix(x).squeeze(1)


class PsiFieldCell(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.register_buffer("laplace_kernel", _make_laplacian())

        self.log_dt = nn.Parameter(torch.tensor(math.log(math.exp(0.08) - 1.0)))
        self.log_c2 = nn.Parameter(torch.tensor(math.log(math.exp(0.25) - 1.0)))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(math.exp(0.08) - 1.0)))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(math.exp(0.10) - 1.0)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(math.exp(0.02) - 1.0)))
        self.log_mem_decay = nn.Parameter(torch.tensor(math.log(math.exp(0.05) - 1.0)))
        self.log_mem_gain = nn.Parameter(torch.tensor(math.log(math.exp(0.15) - 1.0)))
        self.log_drive_gain = nn.Parameter(torch.tensor(math.log(math.exp(0.40) - 1.0)))

        self.field_to_memory = nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.memory_mixer = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.memory_to_field = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)
        self.field_gate = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

    def forward(
        self,
        field: torch.Tensor,
        velocity: torch.Tensor,
        memory: torch.Tensor,
        drive: torch.Tensor,
        inject_drive: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        field_4d = field.unsqueeze(1)
        lap = F.conv2d(F.pad(field_4d, (1, 1, 1, 1), mode="circular"), self.laplace_kernel).squeeze(1)

        dt = _positive(self.log_dt)
        c2 = _positive(self.log_c2)
        gamma = _positive(self.log_gamma)
        alpha = _positive(self.log_alpha)
        beta = _positive(self.log_beta)
        mem_decay = _positive(self.log_mem_decay)
        mem_gain = _positive(self.log_mem_gain)
        drive_gain = _positive(self.log_drive_gain)

        mem_feat = self.memory_mixer(memory)
        mem_back = self.memory_to_field(mem_feat).squeeze(1)
        gate = torch.sigmoid(self.field_gate(mem_feat)).squeeze(1)

        nonlinear = alpha * torch.tanh(field) - beta * field.pow(3)
        drive_scale = 1.0 if inject_drive else 0.15
        drive_term = drive_scale * drive_gain * drive
        acc = c2 * lap - gamma * velocity + nonlinear + gate * mem_back + drive_term

        velocity = velocity + dt * acc
        field = field + dt * velocity

        memory_input = torch.cat([field.unsqueeze(1), drive.unsqueeze(1)], dim=1)
        memory_update = torch.tanh(self.field_to_memory(memory_input))
        memory = torch.exp(-mem_decay * dt) * memory + torch.tanh(mem_gain) * memory_update

        field = _soft_clip(field, 3.0)
        velocity = _soft_clip(velocity, 2.0)
        memory = _soft_clip(memory, 3.0)
        return field, velocity, memory


class PsiClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_channels: int):
        super().__init__()
        in_channels = hidden_channels + 3
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden_channels * 4, num_classes)

    def forward(
        self,
        drive: torch.Tensor,
        field: torch.Tensor,
        velocity: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([drive.unsqueeze(1), field.unsqueeze(1), velocity.unsqueeze(1), memory], dim=1)
        x = self.head(x).flatten(1)
        return self.fc(x)


class ResNetPsiTrainable(nn.Module):
    def __init__(self, config: PsiTrainConfig | None = None):
        super().__init__()
        self.config = config or PsiTrainConfig()
        self.projector = InputProjector(self.config.field_size, self.config.hidden_channels)
        self.cell = PsiFieldCell(self.config.hidden_channels)
        self.classifier = PsiClassifier(self.config.num_classes, self.config.hidden_channels)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.config.field_size
        ch = self.config.hidden_channels
        field = torch.zeros(batch_size, size, size, device=device)
        velocity = torch.zeros_like(field)
        memory = torch.zeros(batch_size, ch, size, size, device=device)
        return field, velocity, memory

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(next(self.parameters()).device)
        drive = self.projector(x)
        field, velocity, memory = self.init_state(len(x), drive.device)

        for step in range(self.config.steps):
            field, velocity, memory = self.cell(
                field,
                velocity,
                memory,
                drive,
                inject_drive=step < self.config.stim_steps,
            )

        return drive, field, velocity, memory

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drive, field, velocity, memory = self.encode(x)
        return self.classifier(drive, field, velocity, memory)


def train_epoch(model: nn.Module, loader, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += len(x)

    return total_loss / max(total_seen, 1), 100.0 * total_correct / max(total_seen, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * len(x)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += len(x)

    return total_loss / max(total_seen, 1), 100.0 * total_correct / max(total_seen, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino da ResNet-Psi v2 em arquivo unico")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field-size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--stim-steps", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", type=str, default="resnet_psi_trainable_single.pt")
    return parser.parse_args()


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)

    val_size = min(args.val_size, len(train_ds) - 1)
    train_size = len(train_ds) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_split, val_split = random_split(train_ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = PsiTrainConfig(
        field_size=args.field_size,
        steps=args.steps,
        stim_steps=args.stim_steps,
        hidden_channels=args.hidden_channels,
        num_classes=10,
    )

    train_loader, val_loader, test_loader = make_loaders(args)
    model = ResNetPsiTrainable(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_state = None
    save_path = Path(args.save_path)

    print(f"device={DEVICE}")
    print(
        f"config field_size={args.field_size} steps={args.steps} "
        f"stim_steps={args.stim_steps} hidden_channels={args.hidden_channels}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} lr={lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(args),
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }
            torch.save(best_state, save_path)

    if best_state is None:
        raise RuntimeError("Treino nao gerou checkpoint valido.")

    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, DEVICE)

    print(f"best_val_acc={best_val_acc:.2f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}")
    print(f"checkpoint={save_path.resolve()}")


if __name__ == "__main__":
    main()
