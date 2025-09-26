# ================================== Imports ================================== #
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from .modeling import DINOV3_MODEL_ID

# =============================== Dataclasses ================================= #
@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing.

    Args:
        dataset_id: Hugging Face dataset identifier (e.g., "lucaantonucci/llmbook-multilabel-images").
        split_train: Split name for training.
        split_val: Split name for validation.
        image_column: Column name containing images or paths.
        label_column: Column name containing multi-label lists or multi-hot vectors.
        num_labels: Total number of labels in the task.
        batch_size: Per-device batch size.
        num_workers: Dataloader workers for CPU preprocessing.
    """

    dataset_id: str
    split_train: str
    split_val: Optional[str]
    image_column: str
    label_column: str
    num_labels: int
    batch_size: int = 16
    num_workers: int = 4


# ================================ Dataset ==================================== #
class MultiLabelImageDataset(Dataset):
    """Wraps an HF dataset dict to provide model-ready tensors."""

    def __init__(
        self,
        hf_ds,
        processor: AutoImageProcessor,
        image_column: str,
        label_column: str,
        num_labels: int,
    ) -> None:
        self.hf_ds = hf_ds
        self.processor = processor
        self.image_column = image_column
        self.label_column = label_column
        self.num_labels = num_labels

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.hf_ds)

    def __getitem__(self, idx: int):  # type: ignore[override]
        example = self.hf_ds[idx]
        image = example[self.image_column]
        labels = example[self.label_column]

        # Normalize labels to a fixed-length multi-hot tensor
        if isinstance(labels, list):
            # Labels can be class indices; convert to multi-hot
            target = torch.zeros(self.num_labels, dtype=torch.float32)
            for label_idx in labels:
                if 0 <= int(label_idx) < self.num_labels:
                    target[int(label_idx)] = 1.0
        elif isinstance(labels, dict):
            # Dict of class_name->0/1; take values and pad/clip
            values = list(labels.values())
            target = torch.tensor(values, dtype=torch.float32)
            if target.numel() != self.num_labels:
                target = torch.nn.functional.pad(
                    target, (0, max(0, self.num_labels - target.numel()))
                )[: self.num_labels]
        else:
            # Already a vector
            target = torch.tensor(labels, dtype=torch.float32)
            if target.numel() != self.num_labels:
                target = torch.nn.functional.pad(
                    target, (0, max(0, self.num_labels - target.numel()))
                )[: self.num_labels]

        return {"image": image, "labels": target}


class DinoCollator:
    """Collate function that applies the DINOv3 processor on a batch of images."""

    def __init__(self, processor: AutoImageProcessor) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict]):
        images = [item["image"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        pixel_values = self.processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]
        return {"pixel_values": pixel_values, "labels": labels}


# ================================ Loader API ================================= #

def create_dataloaders(cfg: DataConfig) -> tuple[DataLoader, Optional[DataLoader]]:
    """Creates PyTorch dataloaders for train/val from an HF dataset.

    Expects the dataset to expose an image column (PIL.Image or path) and a
    label column (list[int], dict[str,int], or multi-hot array-like).
    """
    ds_train = load_dataset(cfg.dataset_id, split=cfg.split_train)
    ds_val = None
    if cfg.split_val:
        ds_val = load_dataset(cfg.dataset_id, split=cfg.split_val)

    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
    train_dataset = MultiLabelImageDataset(
        ds_train, processor, cfg.image_column, cfg.label_column, cfg.num_labels
    )
    val_dataset = (
        MultiLabelImageDataset(
            ds_val, processor, cfg.image_column, cfg.label_column, cfg.num_labels
        )
        if ds_val is not None
        else None
    )

    collator = DinoCollator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        if val_dataset is not None
        else None
    )

    return train_loader, val_loader
