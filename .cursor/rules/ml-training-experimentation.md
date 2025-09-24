# ML Training & Experimentation Rules

## Distributed Training Architecture

All model training must leverage Ray for distributed computing and MLflow for comprehensive experiment tracking. This ensures scalability, reproducibility, and systematic experimentation.

### Ray Distributed Training Standards

#### Training Cluster Configuration
```python
# configs/ray_config.py
"""Ray cluster configuration for distributed training."""

import ray
from ray import train
from ray.train import ScalingConfig
from typing import Dict, Any

def get_ray_training_config(num_workers: int = 2, use_gpu: bool = True) -> Dict[str, Any]:
    """Returns Ray training configuration optimized for ML workloads."""
    return {
        "scaling_config": ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={"CPU": 2, "GPU": 1 if use_gpu else 0},
            placement_strategy="SPREAD"  # Distribute across nodes
        ),
        "run_config": train.RunConfig(
            name="ml_training_experiment",
            storage_path="s3://ml-artifacts/ray-results",
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_accuracy",
                checkpoint_score_order="max"
            ),
            failure_config=train.FailureConfig(
                max_failures=3,
                fail_fast=False
            )
        )
    }
```

#### Distributed Training Framework
```python
# src/training/distributed_trainer.py
"""Distributed training framework using Ray Train and PyTorch."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer, get_device
import mlflow
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    model_name: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True

class DistributedTrainer:
    """Handles distributed model training with Ray and MLflow integration."""

    def __init__(self, config: TrainingConfig, model_class: type, dataset_class: type):
        self.config = config
        self.model_class = model_class
        self.dataset_class = dataset_class

    def train_func(self, config_dict: Dict[str, Any]) -> None:
        """Main training function executed on each Ray worker."""
        # Initialize MLflow tracking
        mlflow.set_tracking_uri("http://mlflow:5000")

        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params(config_dict)

            # Setup distributed training
            device = get_device()
            torch.cuda.set_device(device)

            # Initialize model
            model = self._create_model(config_dict).to(device)
            model = torch.nn.parallel.DistributedDataParallel(model)

            # Setup data loaders
            train_loader, val_loader = self._create_data_loaders(config_dict)

            # Setup optimizer and scheduler
            optimizer = self._create_optimizer(model, config_dict)
            scheduler = self._create_scheduler(optimizer, config_dict)

            # Mixed precision training
            scaler = torch.cuda.amp.GradScaler() if config_dict['mixed_precision'] else None

            # Training loop
            for epoch in range(config_dict['num_epochs']):
                train_metrics = self._train_epoch(
                    model, train_loader, optimizer, scaler, device
                )
                val_metrics = self._validate_epoch(model, val_loader, device)

                if scheduler:
                    scheduler.step()

                # Log metrics to MLflow
                epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
                mlflow.log_metrics(epoch_metrics, step=epoch)

                # Report to Ray Train
                train.report(epoch_metrics)

                # Save checkpoint
                if epoch % 5 == 0:
                    self._save_checkpoint(model, optimizer, epoch, val_metrics['val_accuracy'])

    def _create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Creates and initializes the model."""
        return self.model_class(**config.get('model_params', {}))

    def _create_data_loaders(self, config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
        """Creates distributed data loaders."""
        train_dataset = self.dataset_class(split='train', **config.get('dataset_params', {}))
        val_dataset = self.dataset_class(split='val', **config.get('dataset_params', {}))

        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader

    def _train_epoch(self, model: nn.Module, dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler],
                    device: torch.device) -> Dict[str, float]:
        """Executes one training epoch."""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = nn.CrossEntropyLoss()(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        return {
            "train_loss": total_loss / len(dataloader),
            "train_accuracy": correct_predictions / total_samples
        }
```

### MLflow Experiment Management

#### Experiment Organization Structure
```python
# src/experiment_management.py
"""MLflow experiment management with organized tracking."""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import json

class ExperimentManager:
    """Manages MLflow experiments with consistent organization."""

    def __init__(self, tracking_uri: str = "http://mlflow:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def create_experiment_hierarchy(self, project_name: str,
                                  model_type: str, dataset_version: str) -> str:
        """Creates hierarchical experiment organization."""
        experiment_name = f"{project_name}/{model_type}/dataset_v{dataset_version}"

        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags={
                    "project": project_name,
                    "model_type": model_type,
                    "dataset_version": dataset_version,
                    "framework": "pytorch",
                    "distributed": "ray"
                }
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id

        return experiment_id

    def log_comprehensive_metrics(self, metrics: Dict[str, Any],
                                artifacts_dir: Path, model: torch.nn.Module,
                                step: Optional[int] = None) -> None:
        """Logs comprehensive experiment information."""
        # Log scalar metrics
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value, step=step)

        # Log model architecture
        model_summary = self._get_model_summary(model)
        mlflow.log_text(model_summary, "model_architecture.txt")

        # Log artifacts
        if artifacts_dir.exists():
            mlflow.log_artifacts(str(artifacts_dir))

        # Log model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=f"{metrics.get('model_name', 'default')}_model"
        )
```

### Hyperparameter Optimization

#### Ray Tune Integration
```python
# src/hyperparameter_optimization.py
"""Hyperparameter optimization using Ray Tune with MLflow tracking."""

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import mlflow
from typing import Dict, Any, Callable

class HyperparameterOptimizer:
    """Orchestrates hyperparameter optimization with Ray Tune."""

    def __init__(self, experiment_name: str, num_samples: int = 20):
        self.experiment_name = experiment_name
        self.num_samples = num_samples

    def optimize(self, train_func: Callable, search_space: Dict[str, Any],
                metric: str = "val_accuracy", mode: str = "max") -> Dict[str, Any]:
        """Runs hyperparameter optimization with early stopping."""

        # Configure search algorithm
        search_alg = OptunaSearch(
            metric=metric,
            mode=mode,
            points_to_evaluate=[self._get_baseline_config()]
        )

        # Configure scheduler for early stopping
        scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=50,  # Maximum epochs
            grace_period=10,  # Minimum epochs before stopping
            reduction_factor=3
        )

        # Configure MLflow callback
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri="http://mlflow:5000",
            experiment_name=self.experiment_name,
            save_artifact=True
        )

        # Run optimization
        analysis = tune.run(
            train_func,
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            callbacks=[mlflow_callback],
            resources_per_trial={"cpu": 2, "gpu": 1},
            local_dir="./ray_results"
        )

        return analysis.best_config

    def _get_baseline_config(self) -> Dict[str, Any]:
        """Returns baseline configuration for comparison."""
        return {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "optimizer": "adamw"
        }

# Search space definition
SEARCH_SPACE = {
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "optimizer": tune.choice(["adam", "adamw", "sgd"]),
    "model_params": {
        "dropout_rate": tune.uniform(0.1, 0.5),
        "hidden_dims": tune.choice([256, 512, 1024]),
        "num_layers": tune.choice([3, 4, 5, 6])
    }
}
```

### Model Versioning & Registry

#### Model Registry Management
```python
# src/model_registry.py
"""Model registry management with automated versioning and promotion."""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class ModelStage(Enum):
    """Model lifecycle stages."""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

@dataclass
class ModelValidationResult:
    """Results from model validation checks."""
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_latency_ms: float
    model_size_mb: float
    passed_validation: bool
    validation_notes: str

class ModelRegistry:
    """Manages model versioning, validation, and lifecycle."""

    def __init__(self, tracking_uri: str = "http://mlflow:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str,
                      model_path: str = "model") -> ModelVersion:
        """Registers a model from an MLflow run."""
        model_uri = f"runs:/{run_id}/{model_path}"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={
                "framework": "pytorch",
                "distributed_training": "ray",
                "data_version": "v1.0"
            }
        )

        # Initialize in staging
        self.client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=ModelStage.STAGING.value
        )

        return model_version

    def validate_model(self, model_name: str, version: str,
                      validation_dataset_path: str) -> ModelValidationResult:
        """Performs comprehensive model validation."""

        # Load model for validation
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pytorch.load_model(model_uri)

        # Run validation suite
        validation_metrics = self._run_validation_suite(model, validation_dataset_path)

        # Performance benchmarking
        performance_metrics = self._benchmark_model_performance(model)

        # Combine results
        validation_result = ModelValidationResult(
            model_name=model_name,
            version=version,
            **validation_metrics,
            **performance_metrics,
            passed_validation=self._check_validation_thresholds(validation_metrics),
            validation_notes=self._generate_validation_notes(validation_metrics)
        )

        # Log validation results
        self._log_validation_results(model_name, version, validation_result)

        return validation_result

    def promote_model(self, model_name: str, version: str,
                     target_stage: ModelStage, validation_result: ModelValidationResult) -> bool:
        """Promotes model to target stage after validation."""

        if not validation_result.passed_validation and target_stage == ModelStage.PRODUCTION:
            raise ValueError("Model failed validation - cannot promote to production")

        # Transition model stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage.value,
            archive_existing_versions=True if target_stage == ModelStage.PRODUCTION else False
        )

        # Update model description with validation info
        self.client.update_model_version(
            name=model_name,
            version=version,
            description=f"Promoted to {target_stage.value}. Validation: {validation_result.validation_notes}"
        )

        return True
```

### Training Configuration Management

#### Hydra Configuration Schema
```yaml
# configs/training_config.yaml
defaults:
  - model: resnet50
  - dataset: image_classification
  - optimizer: adamw
  - scheduler: cosine
  - _self_

# Experiment metadata
experiment:
  name: "image_classification_v1"
  project: "ml_lifecycle_demo"
  description: "Distributed training experiment with Ray and MLflow"
  tags:
    - "computer_vision"
    - "classification"
    - "distributed"

# Training configuration
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  weight_decay: 1e-4
  gradient_clip_norm: 1.0
  mixed_precision: true
  warmup_epochs: 5

  # Early stopping
  early_stopping:
    patience: 10
    min_delta: 0.001
    monitor: "val_accuracy"

# Distributed training
distributed:
  num_workers: 2
  use_gpu: true
  resources_per_worker:
    CPU: 2
    GPU: 1
  placement_strategy: "SPREAD"

# Data configuration
data:
  dataset_path: "/data/processed"
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# Checkpointing
checkpointing:
  save_freq: 5
  keep_top_k: 3
  monitor: "val_accuracy"
  mode: "max"

# Hyperparameter optimization
hyperopt:
  enabled: false
  num_trials: 20
  search_space:
    learning_rate:
      type: "loguniform"
      low: 1e-5
      high: 1e-1
    batch_size:
      type: "choice"
      choices: [16, 32, 64, 128]
    weight_decay:
      type: "loguniform"
      low: 1e-6
      high: 1e-2

# Model validation thresholds
validation:
  min_accuracy: 0.85
  max_inference_latency_ms: 100
  max_model_size_mb: 500
  min_precision: 0.80
  min_recall: 0.80
```

This comprehensive training framework ensures scalable, reproducible, and well-tracked ML experiments using distributed computing and systematic experiment management.
