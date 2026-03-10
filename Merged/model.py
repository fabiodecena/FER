"""
model.py – Facial Emotion Recognition
Backbone: timm (ConvNeXt-Tiny default, configurable)
Head: Multi-layer with BatchNorm + Dropout (for larger datasets)
"""

import torch.nn as nn
import timm

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FERModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 0,            # <-- add this!
        input_size: int = 96,           # <-- optional: add this!
    ):
        super().__init__()

        self.dropout = dropout
        self.input_size = input_size
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features

        if hidden_dim and hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Identity(),  # head.0
                nn.Linear(num_features, hidden_dim),  # head.1
                nn.ReLU(inplace=True),  # head.2
                nn.Dropout(p=dropout),  # head.3
                nn.Linear(hidden_dim, num_classes),  # head.4
            )
        else:
            self.head = nn.Linear(num_features, num_classes)

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
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
    hidden_dim: int = 0,
) -> FERModel:
    return FERModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim
    )