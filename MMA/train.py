"""
train.py – Training MMA FER with freeze/unfreeze fine-tuning

Recommended for low-resolution images (48x48 source images).

Examples:
    python train.py
    python train.py --use_class_weights
    python train.py --backbone mobilenetv3_small_100 --img_size 64 --hidden_dim 128
"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset import get_dataloader
from model import build_model

CHECKPOINT_DIR = Path("checkpoints")
FREEZE_BATCH_SIZE = 16

def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def count_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def compute_class_weights(dataset, device: torch.device) -> torch.Tensor:
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(dataset.classes))
    total = len(targets)
    weights = total / (len(dataset.classes) * class_counts.astype(np.float64))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("▶ Class weights:")
    for cls, w, c in zip(dataset.classes, weights, class_counts):
        print(f"    {cls:12s}: count={c:6d}  weight={w:.4f}")
    return weights


def make_checkpoint(
    model: nn.Module,
    backbone: str,
    classes: list[str],
    best_val_acc: float | None = None,
) -> dict:
    ckpt = {
        "arch": backbone,
        "num_classes": len(classes),
        "classes": classes,
        "state_dict": model.state_dict(),
    }
    if best_val_acc is not None:
        ckpt["best_val_acc"] = best_val_acc
    return ckpt


def load_final_best_acc(final_path: Path) -> float:
    if final_path.exists():
        try:
            ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
            acc = ckpt.get("best_val_acc", 0.0)
            print(f"▶ Existing final model found: val_acc = {acc:.4f}")
            return acc
        except Exception:
            pass
    return 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    model.eval()
    total = correct = 0
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    mixup_alpha: float=0.0,
):
    model.train()
    total = correct = 0
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if mixup_alpha > 0.0:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def run_phase(
    phase_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    best_val_acc: float,
    classes: list[str],
    backbone: str,
    best_ckpt_path: Path,
    mixup_alpha: float=0.0
):
    patience_counter = 0

    trainable, total = count_params(model)
    print(f"\n{'=' * 60}")
    print(f"Phase: {phase_name}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=mixup_alpha)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{phase_name}] Epoch {epoch:>3}/{epochs}  │  "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f}  │  "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}  │  "
            f"LR {lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(make_checkpoint(model, backbone, classes, best_val_acc), best_ckpt_path)
            print(f"  ✓ New best ({val_acc:.4f}) → {best_ckpt_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  ⏹ Early stopping ({patience} epochs without improvement)")
            break

    return best_val_acc


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"▶ Run timestamp: {timestamp}")

    # For freeze/head-only phase
    train_loader_freeze, train_ds = get_dataloader(
        args.data,
        split="train",
        batch_size=FREEZE_BATCH_SIZE,
        augmentation=args.augmentation,
        img_size=args.img_size,
    )
    val_loader_freeze, val_ds = get_dataloader(
        args.data,
        split="validation",
        batch_size=FREEZE_BATCH_SIZE,
        augmentation=args.augmentation,
        img_size=args.img_size,
    )

    # For fine-tune phase (custom batch size)
    train_loader_ft, _ = get_dataloader(
        args.data,
        split="train",
        batch_size=args.batch_size,
        augmentation=args.augmentation,
        img_size=args.img_size,
    )
    val_loader_ft, _ = get_dataloader(
        args.data,
        split="validation",
        batch_size=args.batch_size,
        augmentation=args.augmentation,
        img_size=args.img_size,
    )

    num_classes = len(train_ds.classes)
    classes = train_ds.classes
    print(f"▶ Classes ({num_classes}): {classes}")
    print(f"▶ Train images: {len(train_ds)}")
    print(f"▶ Val images:   {len(val_ds)}")

    model = build_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        input_size=args.img_size,
    ).to(device)

    ghost_keys = [k for k in model.state_dict().keys() if k.startswith("logits.")]
    assert not ghost_keys, f"Unexpected 'logits' keys in state_dict: {ghost_keys}"

    if args.use_class_weights:
        class_weights = compute_class_weights(train_ds, device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
        print("▶ Loss: CrossEntropyLoss with class weights")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("▶ Loss: CrossEntropyLoss without class weights")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_ckpt_path = CHECKPOINT_DIR / f"mma_best_{timestamp}.pt"
    final_path = CHECKPOINT_DIR / "mma_final_model.pt"

    global_best_acc = load_final_best_acc(final_path)
    run_best_acc = 0.0

    model.freeze_backbone()
    optimizer_head = AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler_head = CosineAnnealingLR(optimizer_head, T_max=args.warmup_epochs, eta_min=1e-6)

    run_best_acc = run_phase(
        phase_name="HEAD-ONLY",
        model=model,
        train_loader=train_loader_freeze,
        val_loader=val_loader_freeze,
        criterion=criterion,
        optimizer=optimizer_head,
        scheduler=scheduler_head,
        device=device,
        epochs=args.warmup_epochs,
        patience=args.patience,
        best_val_acc=run_best_acc,
        classes=classes,
        backbone=args.backbone,
        best_ckpt_path=best_ckpt_path,
        mixup_alpha=args.mixup_alpha
    )

    model.unfreeze_backbone()
    optimizer_ft = AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.head.parameters(),     "lr": args.lr * 0.5},
    ], weight_decay=1e-2)
    scheduler_ft = CosineAnnealingLR(optimizer_ft, T_max=args.ft_epochs, eta_min=1e-7)

    run_best_acc = run_phase(
        phase_name="FINE-TUNE",
        model=model,
        train_loader=train_loader_ft,
        val_loader=val_loader_ft,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=scheduler_ft,
        device=device,
        epochs=args.ft_epochs,
        patience=args.patience,
        best_val_acc=run_best_acc,
        classes=classes,
        backbone=args.backbone,
        best_ckpt_path=best_ckpt_path,
        mixup_alpha=args.mixup_alpha
    )

    print(f"\n{'─' * 60}")
    print(f"This run best val acc:    {run_best_acc:.4f}")
    print(f"Previous global best acc: {global_best_acc:.4f}")

    if run_best_acc > global_best_acc:
        best_state = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        final_ckpt = make_checkpoint(model, args.backbone, classes, run_best_acc)
        final_ckpt["state_dict"] = best_state["state_dict"]
        torch.save(final_ckpt, final_path)
        print(f"🏆 NEW GLOBAL BEST! Saved → {final_path}")
    else:
        print("ℹ️  Final model NOT updated (previous run was better)")

    print("\n✅ Training complete.")
    print(f"   Run best model  → {best_ckpt_path}")
    print(f"   Final model     → {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MMA FER")
    parser.add_argument("--data", type=str, default="data_mma")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--ft_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=13)
    parser.add_argument("--augmentation", type=str, default="light", choices=["none", "light", "medium", "heavy"])
    parser.add_argument("--label_smoothing", type=float, default=0.00)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Mixup alpha parameter (default: 0.0, disables Mixup)')
    main(parser.parse_args())