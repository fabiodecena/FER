"""
evaluate_best.py – Evaluation sul validation set (Merged)
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset import get_dataloader
from model import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("checkpoints/merged_best_20260305_162825.pt", map_location=device, weights_only=True)
    model = build_model(
        num_classes=ckpt["num_classes"],
        backbone=ckpt["arch"],
        pretrained=False,
        dropout=0.0,
        big_head=False
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    classes = ckpt["classes"]
    print(f"▶ Model: {ckpt['arch']}, {ckpt['num_classes']} classes")

    val_loader, _ = get_dataloader("data_merged", split="validation", batch_size=32, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n{'='*60}")
    print("Classification Report (Validation)")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Merged Validation - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix_val.png", dpi=150)
    plt.show()
    print("\n✅ Saved confusion_matrix_val.png")


if __name__ == "__main__":
    main()