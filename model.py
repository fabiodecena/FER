"""
    model.py

    Defines MMA FER model architecture:
    - Lightweight wrapper around a timm backbone for Facial Expression Recognition (FER)
    - Customizable head for classification
    - Utilities for feature extraction, freezing/unfreezing, and model instantiation

    Main entities:
        - FERModel: Main model class for FER
        - build_model: Factory for simplified model creation

    Labels:
        EMOTION_LABELS: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
"""
import torch
import torch.nn as nn
import timm

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FERModel(nn.Module):
    """
        MMA FER model wrapper around a timm backbone, with a custom classification head.

        Features:
            - Removes original classifier from backbone, replacing with a custom head
            - Optionally includes a hidden layer before output
            - Supports backbone feature freezing and unfreezing
            - Automatic input feature dimension inference

        Args:
            num_classes (int): Number of output classes (default 7).
            backbone (str): Backbone model from timm library (default 'convnext_tiny').
            pretrained (bool): If True, initializes backbone with pretrained weights.
            dropout (float): Dropout probability for head layers.
            hidden_dim (int): If >0, uses intermediate hidden layer of this dimension before output.
            image_size (int): Input image size for feature dimension inference.

        Attributes:
            backbone (nn.Module): Feature extractor from timm.
            head (nn.Module): Classification head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "convnext_tiny",
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
        """
            Extracts features from backbone, handling forward_features and flattening.

            Args: x (torch.Tensor): Input image batch (B, C, H, W).

            Returns: torch.Tensor: Feature tensor (B, feature_dim)
        """
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
        """
            Infers feature dimension of backbone output using a dummy input.

             Args: input_size (int): Input image size.

            Returns: int: Number of backbone features.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = self._extract_features(dummy)
        return features.shape[1]

    def forward(self, x):
        """
            Runs input through backbone and head classification layers.

            Args: x (torch.Tensor): Input batch (B, C, H, W).

            Returns: torch.Tensor: Output logits (B, num_classes)
        """
        features = self._extract_features(x)
        return self.head(features)

    def freeze_backbone(self):
        """
            Sets backbone parameters as non-trainable (freezes feature extractor).
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """
            Sets backbone parameters as trainable (unfreezes feature extractor).
        """
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(
    num_classes: int = 7,
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    dropout: float = 0.2,
    hidden_dim: int = 0,
    input_size: int = 96,
) -> FERModel:
    """
        Factory function to build FERModel with provided arguments.

        Args:
            num_classes (int): Number of output classes.
            backbone (str): Backbone model name.
            pretrained (bool): Whether to use pretrained weights.
            dropout (float): Dropout probability.
            hidden_dim (int): Size of hidden layer (if >0).
            input_size (int): Input image size.

        Returns: FERModel: Instantiated FER model.
    """
    return FERModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim,
        image_size=input_size,
    )