from pathlib import Path

import pytest
import torch

from src.e2e_ml.modeling import DinoV3ForMultiLabel, ModelConfig
from src.e2e_ml.data import DataConfig, create_dataloaders


@pytest.mark.slow
@pytest.mark.parametrize("pooled", [True, False])
def test_model_forward_no_labels(pooled: bool):
    cfg = ModelConfig(num_labels=5, dropout=0.0, freeze_backbone=True, pooled=pooled)
    model = DinoV3ForMultiLabel(cfg)

    # Create dummy pixel_values in expected shape (1, 3, 224, 224)
    x = torch.randn(1, 3, 224, 224)
    out = model(pixel_values=x)
    assert "logits" in out
    assert out["logits"].shape == (1, 5)


def test_dataloaders_smoke():
    # Small public dataset with images
    data_cfg = DataConfig(
        dataset_id="ashraq/fashion-product-images-small",
        split_train="train",
        split_val=None,
        image_column="image",
        label_column="label",
        num_labels=27,
        batch_size=2,
        num_workers=0,
    )

    train_loader, val_loader = create_dataloaders(data_cfg)
    assert val_loader is None

    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"pixel_values", "labels"}
    assert batch["pixel_values"].shape[0] == 2
    assert batch["labels"].shape == (2, 27)
