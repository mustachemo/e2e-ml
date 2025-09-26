from __future__ import annotations

from dagster import Definitions, JobDefinition, job

from .ops import DagDataConfig, load_backbone, load_hf_dataset, load_processor, preprocess_batch, run_inference


@job
def dinov3_test_job() -> None:
    ds = load_hf_dataset(DagDataConfig(dataset_id="ashraq/fashion-product-images-small"))
    proc = load_processor()
    model = load_backbone()
    batch = preprocess_batch(ds, proc)
    run_inference(model, batch)


def get_defs() -> Definitions:
    return Definitions(jobs=[dinov3_test_job])
