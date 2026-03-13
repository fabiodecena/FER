"""
evaluate_best.py – Evaluation on validation sets
"""


import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataset import get_dataloader
from model import build_model


DATASET_CONFIGS = {
    "kdef": {
        "data_dir": "KDEF/data_split",
        "checkpoint_dir": "KDEF/checkpoints",
        "checkpoint_glob": "kdef_final_model.pt",
        "output_prefix": "KDEF/kdef_validation",
        "title": "KDEF Validation",
    },
    "merged": {
        "data_dir": "Merged/data_merged",
        "checkpoint_dir": "Merged/checkpoints",
        "checkpoint_glob": "*.pt",
        "output_prefix": "Merged/merged_validation",
        "title": "Merged Validation",
    },
    "mma": {
        "data_dir": "MMA/data_mma",
        "checkpoint_dir": "MMA/checkpoints",
        "checkpoint_glob": "*.pt",
        "output_prefix": "MMA/mma_validation",
        "title": "MMA Validation",
    },
}

def find_latest_checkpoint(checkpoint_dir: Path, pattern: str) -> Path:
    candidates = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
    return candidates[0]


def infer_hidden_dim_from_state_dict(state_dict: dict) -> int:
    if "head.4.weight" in state_dict:
        return state_dict["head.1.weight"].shape[0]
    return 0


def main(args: argparse.Namespace):
    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    data_dir = Path(cfg["data_dir"])
    checkpoint_path = (
        Path(cfg["checkpoint_dir"] + "/" + args.checkpoint)
        if args.checkpoint is not None
        else find_latest_checkpoint(Path(cfg["checkpoint_dir"]), cfg["checkpoint_glob"])
    )

    print(f"▶ Dataset: {args.dataset}")
    print(f"▶ Data dir: {data_dir}")
    print(f"▶ Checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    hidden_dim = infer_hidden_dim_from_state_dict(ckpt["state_dict"])
    model = build_model(
        num_classes=ckpt["num_classes"],
        backbone=ckpt["arch"],
        pretrained=False,
        dropout=0.0,
        hidden_dim=hidden_dim,
        input_size=args.img_size,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    classes = ckpt["classes"]
    print(f"▶ Model: {ckpt['arch']}")
    print(f"▶ Classes: {classes}")
    print(f"▶ Hidden dim inferred from checkpoint: {hidden_dim}")

    val_loader, _ = get_dataloader(
        data_dir,
        split="validation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        augmentation="none",
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report_text = classification_report(all_labels, all_preds, target_names=classes, digits=4)

    print(f"\n{'=' * 60}")
    print(f"Classification Report ({args.dataset.upper()} Validation)")
    print(f"{'=' * 60}")
    print(report_text)

    output_prefix = Path(cfg["output_prefix"])
    report_path = output_prefix.with_name(output_prefix.name + "_metrics.txt")
    cm_image_path = output_prefix.with_name(output_prefix.name + "_confusion_matrix.png")
    cm_csv_path = output_prefix.with_name(output_prefix.name + "_confusion_matrix.csv")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Model: {ckpt['arch']}\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"Hidden dim: {hidden_dim}\n\n")
        f.write(report_text)

    print(f"✅ Saved metrics report to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",", header=",".join(classes), comments="")
    print(f"✅ Saved confusion matrix values to {cm_csv_path}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"{cfg['title']} - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(cm_image_path, dpi=150)
    print(f"✅ Saved confusion matrix image to {cm_image_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate best FER checkpoint on validation set")
    parser.add_argument("--dataset", type=str, required=True, choices=["kdef", "merged", "mma"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    main(parser.parse_args())