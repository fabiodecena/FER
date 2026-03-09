"""
evaluate_best.py – Evaluation del modello migliore su validation set
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from KDEF.dataset import get_dataloader
from KDEF.model import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Carica modello ────────────────────────────────────────────
    ckpt = torch.load("checkpoints/kdef_best_model.pt", map_location=device, weights_only=True)
    model = build_model(
        num_classes=ckpt["num_classes"],
        backbone=ckpt["arch"],
        pretrained=False,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    classes = ckpt["classes"]
    print(f"▶ Loaded model: {ckpt['arch']}, {ckpt['num_classes']} classes")

    # ── Evaluation (num_workers=0 per evitare problemi su Windows) ─
    val_loader, _ = get_dataloader("data_split", split="validation", batch_size=32, num_workers=0)

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

    # ── Classification report ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Classification Report")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("KDEF Validation - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("\n✅ Saved confusion_matrix.png")


if __name__ == "__main__":
    main()