"""
train.py – Train (or fine-tune) the EfficientNet-B2 FER model on your data.

Usage:
    python train.py                        # defaults (30 epochs, lr=1e-4)
    python train.py --epochs 50 --lr 3e-4  # custom
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloader
from model import build_model

CHECKPOINT_DIR = Path("checkpoints")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run evaluation. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    # ── Data ───────────────────────────────────────────────────────
    train_loader, train_dataset = get_dataloader(args.data, split="train", batch_size=args.batch_size)
    val_loader, val_dataset = get_dataloader(args.data, split="validation", batch_size=args.batch_size)

    num_classes = len(train_dataset.classes)
    print(f"▶ Classes ({num_classes}): {train_dataset.classes}")

    # ── Model ──────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, pretrained_backbone=True).to(device)

    # ★ FREEZE the backbone — only train the classifier head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"▶ Trainable params: {trainable:,} / {total_params:,} ({trainable/total_params:.1%})")

    # Optionally load a previous checkpoint to continue training
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"▶ Resumed from {args.resume}")

    # ── Optimiser & scheduler ──────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2,          # ★ stronger weight decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # ★ UNFREEZE backbone after epoch 5 for fine-tuning
        if epoch == 6:
            print("  ★ Unfreezing backbone for fine-tuning (lower lr)")
            for param in model.parameters():
                param.requires_grad = True
            # Use a lower lr for the backbone
            optimizer = AdamW([
                {"params": model.classifier.parameters(), "lr": args.lr},
                {"params": [p for n, p in model.named_parameters()
                            if "classifier" not in n and p.requires_grad],
                 "lr": args.lr * 0.1},   # 10x lower for backbone
            ], weight_decay=1e-2)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 5)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{args.epochs}  │  "
            f"Train loss {train_loss:.4f}  acc {train_acc:.4f}  │  "
            f"Val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model ({val_acc:.4f}) → {ckpt_path}")
        else:
            patience_counter += 1

        # ★ Early stopping: stop if val acc doesn't improve for 7 epochs
        if patience_counter >= 7:
            print(f"\n⏹ Early stopping at epoch {epoch} (no improvement for 7 epochs)")
            break

    # Save the final model as well
    torch.save(model.state_dict(), CHECKPOINT_DIR / "final_model.pt")
    print(f"\n✅ Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FER model")
    parser.add_argument("--data", type=str, default="data", help="Path to data/ folder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    main(parser.parse_args())