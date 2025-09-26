# ================================== Imports ================================== #
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlflow
import torch
import typer
from accelerate import Accelerator
from loguru import logger
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .data import DataConfig, create_dataloaders
from .modeling import DinoV3ForMultiLabel, ModelConfig

console = Console()
app = typer.Typer(add_completion=False, help="Train DINOv3 multi-label model")


# =============================== Train Routine =============================== #

def _evaluate(model: DinoV3ForMultiLabel, loader: Optional[DataLoader]) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            out = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = out["loss"].detach()
            total_loss += loss.item() * batch["labels"].size(0)
            total_count += batch["labels"].size(0)
    model.train()
    return total_loss / max(1, total_count)


@app.command()
def train(
    dataset_id: str = typer.Option(
        "ashraq/fashion-product-images-small",
        help="HF dataset id (must include images and labels)",
    ),
    image_column: str = typer.Option(
        "image", help="Column name for images (PIL.Image or path)"
    ),
    label_column: str = typer.Option(
        "label", help="Column name for labels (list[int] or multi-hot)"
    ),
    num_labels: int = typer.Option(27, help="Number of target labels"),
    split_train: str = typer.Option("train", help="Train split name"),
    split_val: Optional[str] = typer.Option("validation", help="Val split name"),
    output_dir: Path = typer.Option(Path("./artifacts"), help="Artifacts dir"),
    epochs: int = typer.Option(1, min=1),
    lr: float = typer.Option(5e-4),
    batch_size: int = typer.Option(16),
    num_workers: int = typer.Option(4),
    freeze_backbone: bool = typer.Option(True),
    pooled: bool = typer.Option(True),
    log_dir: Path = typer.Option(Path("./runs")),
) -> None:
    """Train a multi-label classifier on top of DINOv3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    writer = SummaryWriter(log_dir=str(log_dir))

    data_cfg = DataConfig(
        dataset_id=dataset_id,
        split_train=split_train,
        split_val=split_val,
        image_column=image_column,
        label_column=label_column,
        num_labels=num_labels,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader, val_loader = create_dataloaders(data_cfg)

    model_cfg = ModelConfig(
        num_labels=num_labels, dropout=0.1, freeze_backbone=freeze_backbone, pooled=pooled
    )
    model = DinoV3ForMultiLabel(model_cfg)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    total_steps = epochs * math.ceil(len(train_loader.dataset) / batch_size)

    with mlflow.start_run():
        mlflow.log_param("dataset_id", dataset_id)
        mlflow.log_param("num_labels", num_labels)
        mlflow.log_param("lr", lr)
        mlflow.log_param("freeze_backbone", freeze_backbone)
        mlflow.log_param("pooled", pooled)
        mlflow.log_param("epochs", epochs)

        global_step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in progress:
                outputs = model(
                    pixel_values=batch["pixel_values"], labels=batch["labels"]
                )
                loss = outputs["loss"]
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.is_main_process:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)
                global_step += 1

            val_loss = _evaluate(model, val_loader)
            if accelerator.is_main_process:
                writer.add_scalar("val/loss", val_loss, epoch)
                mlflow.log_metric("val_loss", float(val_loss), step=epoch)

        if accelerator.is_main_process:
            save_path = output_dir / "model.pt"
            accelerator.unwrap_model(model).cpu()
            torch.save({"model_state": accelerator.unwrap_model(model).state_dict()}, save_path)
            mlflow.log_artifact(str(save_path))
            console.print(f"[green]Saved model to {save_path}[/green]")

    writer.close()
