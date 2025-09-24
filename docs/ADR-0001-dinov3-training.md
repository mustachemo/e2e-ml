## ADR-0001: DINOv3 multi-label training scaffold

Status: Accepted
Date: 2025-09-24

### Context
We need a minimal, production-ready scaffold to fine-tune a DINOv3 backbone on
multi-label image datasets from Hugging Face Datasets. We prefer modern tooling
(uv, Typer, MLflow, Accelerate) and a clean architecture.

### Decision
- Use `facebook/dinov3-vit7b16-pretrain-lvd1689m` as the backbone, loaded via
  `transformers.AutoModel` with `AutoImageProcessor` for preprocessing.
- Add a lightweight linear head over pooled features (or CLS token) and train
  with BCEWithLogitsLoss for multi-label objectives.
- Data pipeline uses HF Datasets; a collator applies the image processor at
  batch-time for simplicity.
- Training loop uses `accelerate` for device handling and `mlflow` for logging.
- CLI is built with Typer to expose dataset and training knobs.

### Alternatives Considered
- Using `timm` models instead of HF Transformers: broader zoo, but DINOv3
  integration and image processors are better aligned in Transformers.
- End-to-end Trainer API: possible, but a light custom loop is clearer and
  easier to adapt for multi-label metrics.

### Consequences
- Simplicity and clarity, fast to iterate. For large-scale training, we may
  extend to gradient accumulation, LR schedulers, mixed precision, and metrics.

### References
- DINOv3 Transformers docs: [link](https://huggingface.co/docs/transformers/main/en/model_doc/dinov3)
