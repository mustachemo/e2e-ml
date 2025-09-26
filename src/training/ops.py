# ================================== Imports ================================== #
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from dagster import In, Out, OpExecutionContext, op
from datasets import DatasetDict, load_dataset
from transformers import AutoImageProcessor, AutoModel

from src.e2e_ml.modeling import DINOV3_MODEL_ID


# =============================== Dataclasses ================================= #
@dataclass
class DagDataConfig:
    dataset_id: str
    split_train: str = "train"
    split_val: Optional[str] = "validation"
    image_column: str = "image"
    label_column: str = "label"


# =================================== Ops ==================================== #
@op(out=Out(DatasetDict))
def load_hf_dataset(context: OpExecutionContext, cfg: DagDataConfig) -> DatasetDict:
    context.log.info(f"Loading dataset {cfg.dataset_id}")
    ds_train = load_dataset(cfg.dataset_id, split=cfg.split_train)
    if cfg.split_val:
        ds_val = load_dataset(cfg.dataset_id, split=cfg.split_val)
        return DatasetDict({"train": ds_train, "validation": ds_val})
    return DatasetDict({"train": ds_train})


@op(out=Out(AutoImageProcessor))
def load_processor(context: OpExecutionContext) -> AutoImageProcessor:
    context.log.info(f"Loading processor for {DINOV3_MODEL_ID}")
    return AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)


@op(out=Out(AutoModel))
def load_backbone(context: OpExecutionContext) -> AutoModel:
    context.log.info(f"Loading model {DINOV3_MODEL_ID}")
    model = AutoModel.from_pretrained(DINOV3_MODEL_ID)
    model.eval()
    return model


@op(ins={"ds": In(DatasetDict), "processor": In(AutoImageProcessor)}, out=Out(dict))
def preprocess_batch(context: OpExecutionContext, ds: DatasetDict, processor: AutoImageProcessor) -> dict:
    sample = ds["train"][0]
    pixel_values = processor(images=sample["image"], return_tensors="pt")[
        "pixel_values"
    ]
    context.log.info(f"Preprocessed image tensor shape: {tuple(pixel_values.shape)}")
    return {"pixel_values": pixel_values}


@op(ins={"model": In(AutoModel), "batch": In(dict)}, out=Out(dict))
def run_inference(context: OpExecutionContext, model: AutoModel, batch: dict) -> dict:
    with torch.inference_mode():
        outputs = model(**batch)
    pool = outputs.pooler_output if hasattr(outputs, "pooler_output") else None
    context.log.info(
        f"last_hidden_state: {tuple(outputs.last_hidden_state.shape)}; pool: {None if pool is None else tuple(pool.shape)}"
    )
    return {"last_hidden_state_shape": tuple(outputs.last_hidden_state.shape)}
