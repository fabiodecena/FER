"""
model.py – Facial Emotion Recognition
Backbone: timm (ConvNeXt-Tiny default, configurable)
Head: Multi-layer with BatchNorm + Dropout (for larger datasets)
"""

import torch.nn as nn
import timm

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FERModel(nn.Module):
    """
    Wrapper: timm backbone + multi-layer head.
    Small head for small datasets (KDEF), big head for large datasets (merged).
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.3,
        big_head: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features  # 768 for convnext_tiny

        if big_head:
            # ~460k params — for datasets > 10k images
            self.head = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout * 0.5),
                nn.Linear(128, num_classes),
            )
        else:
            # ~5k params — for small datasets (KDEF)
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes),
            )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(
    num_classes: int = 7,
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    dropout: float = 0.3,
    big_head: bool = False
) -> FERModel:
    return FERModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        big_head=big_head
    )