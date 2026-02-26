"""
model.py – EfficientNet-B2 backbone for Facial Expression Recognition.

Uses a pretrained EfficientNet-B2 from `timm` (ImageNet weights) and replaces
the classifier head with one that outputs 8 emotion classes (AffectNet mapping):
    0: Anger, 1: Disgust, 2: Fear,
    3: Happiness, 4: Neutral, 5: Sadness, 6: Surprise

If you later download the HSEmotion .pt checkpoint, you can load it with
`load_hsemotion_weights()`.
"""

import torch
import torch.nn as nn
import timm

# AffectNet-7 emotion labels (same order used by HSEmotion)
EMOTION_LABELS = [
    "Anger", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]


def build_model(num_classes: int = 8, pretrained_backbone: bool = True) -> nn.Module:
    """
    Build an EfficientNet-B2 with a custom classification head.

    Args:
        num_classes: Number of emotion categories.
        pretrained_backbone: Use ImageNet-pretrained weights for the backbone.

    Returns:
        A `torch.nn.Module` ready for training or inference.
    """
    model = timm.create_model(
        "tf_efficientnet_b2",
        pretrained=pretrained_backbone,
        num_classes=num_classes,
    )
    return model


def load_hsemotion_weights(
    model: nn.Module,
    checkpoint_path: str,
    my_device: torch.device | str = "cpu",
) -> nn.Module:
    """
    Load weights from an HSEmotion / custom checkpoint.

    Args:
        model: The model returned by `build_model()`.
        checkpoint_path: Path to a `.pt` or `.pth` file.
        my_device: Target device.

    Returns:
        The model with loaded weights, in eval mode.
    """
    state = torch.load(checkpoint_path, map_location=my_device, weights_only=True)
    # Some checkpoints wrap the dict in a "state_dict" key
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(my_device).eval()
    return model


# ── Quick sanity check ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using my_device: {device}")

    net = build_model().to(device)
    dummy = torch.randn(1, 3, 260, 260, device=device)
    logits = net(dummy)
    print(f"Output shape: {logits.shape}")          # → [1, 8]
    print(f"Predicted class: {EMOTION_LABELS[logits.argmax(dim=1).item()]}")