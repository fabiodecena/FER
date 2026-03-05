"""
model.py – KDEF Facial Emotion Recognition
Backbone: timm (ConvNeXt-Tiny default, configurabile)
"""

from __future__ import annotations

import torch.nn as nn
import timm

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FERModel(nn.Module):
    """
    Wrapper pulito: backbone timm + dropout + linear head.
    Nessun monkey-patching, nessun logits fantasma.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Crea backbone senza head (num_classes=0 → restituisce solo features)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features  # dimensione embedding (es. 768 per convnext_tiny)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)          # (B, num_features)
        return self.head(features)            # (B, num_classes)

    def freeze_backbone(self):
        """Congela il backbone, allena solo la head."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Scongela il backbone per fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(
    num_classes: int = 7,
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> FERModel:
    return FERModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )