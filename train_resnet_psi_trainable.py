import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from resnet_psi_trainable import (
    DEVICE,
    PsiTrainConfig,
    ResNetPsiTrainable,
    evaluate,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino da ResNet-Psi v2 treinavel")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--stim-steps", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="checkpoints/resnet_psi_trainable.pt")
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
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
    os.environ.setdefault("TMPDIR", str(Path.cwd() / "tmp"))
    main()
