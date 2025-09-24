# ================================== Imports ================================== #
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel

# =============================== Constants =================================== #
DINOV3_MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


# ================================ Dataclasses ================================= #
@dataclass
class ModelConfig:
    """Configuration for building a DINOv3-based multi-label classifier.

    Args:
        num_labels: Number of target labels for multi-label classification.
        dropout: Dropout probability before the classifier.
        freeze_backbone: If True, freeze DINOv3 backbone parameters.
        pooled: If True, use pooled output; otherwise use CLS token.
    """

    num_labels: int
    dropout: float = 0.1
    freeze_backbone: bool = True
    pooled: bool = True


# ================================ Model Module ================================ #
class DinoV3ForMultiLabel(nn.Module):
    """Wraps a pretrained DINOv3 model with a multi-label head.

    Uses the pooled representation (or CLS token) and a linear classifier that
    outputs logits for each label. Loss is BCEWithLogits for multi-label tasks.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
        self.backbone = AutoModel.from_pretrained(DINOV3_MODEL_ID)

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(p=self.cfg.dropout)
        self.classifier = nn.Linear(hidden_size, self.cfg.num_labels)

        if self.cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through backbone and classification head.

        Args:
            pixel_values: Preprocessed images, shape (B, 3, H, W).
            labels: Optional multi-hot labels, shape (B, num_labels).

        Returns:
            A dict with keys: logits, loss (if labels provided).
        """
        outputs = self.backbone(pixel_values=pixel_values)
        if self.cfg.pooled and hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output  # (B, hidden)
        else:
            # CLS token is at index 0 of sequence
            features = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(self.dropout(features))
        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            result["loss"] = loss
        return result


def get_processor() -> AutoImageProcessor:
    """Returns an image processor aligned with the backbone."""
    return AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
