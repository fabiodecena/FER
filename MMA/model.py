"""
model.py – FER model for MMA low-resolution dataset

Designed for smaller backbones and lower-resolution inputs.
Recommended defaults:
- backbone: mobilenetv3_small_100
- input size: 96
"""

import torch
import torch.nn as nn
import timm

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FERModel(nn.Module):
    """
    Lightweight FER wrapper:
    timm backbone without classifier + small custom head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        dropout: float = 0.2,
        hidden_dim: int = 0,
        image_size: int = 96,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        num_features = self._infer_feature_dim(image_size)

        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes),
            )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)

            if hasattr(self.backbone, "forward_head"):
                features = self.backbone.forward_head(features, pre_logits=True)
            elif features.ndim > 2:
                features = torch.flatten(features, 1)

            if features.ndim > 2:
                features = torch.flatten(features, 1)

            return features

        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features

    def _infer_feature_dim(self, input_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = self._extract_features(dummy)
        return features.shape[1]

    def forward(self, x):
        features = self._extract_features(x)
        return self.head(features)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(
    num_classes: int = 7,
    backbone: str = "mobilenetv3_small_100",
    pretrained: bool = True,
    dropout: float = 0.2,
    hidden_dim: int = 0,
    input_size: int = 96,
) -> FERModel:
    return FERModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim,
        image_size=input_size,
    )