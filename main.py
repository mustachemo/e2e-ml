#!/usr/bin/env python3
"""Main entry point for the E2E ML workflow.

This script uses Hydra configuration management to orchestrate the complete
machine learning pipeline including data loading, model training, and evaluation.
"""

import os
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from src.data import create_data_module
from src.models import create_model, create_trainer
from src.utils import MLflowManager, setup_mlflow_logging

# =============================== Constants ================================== #
# ! AMD Ryzen AI Max+ 395 (gfx1151) has MIOpen BatchNorm compilation issues
# ! Force CPU until official ROCm support is available or workaround is confirmed
# DEVICE = "cpu"  # Set to "cuda" to test GPU (may crash on BatchNorm operations)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================== Utilities ================================= #
def create_summary_table(config: DictConfig, metrics: dict[str, Any]) -> Table:
    """Create a summary table for display.

    Args:
        config: Hydra configuration.
        metrics: Dictionary of metrics.

    Returns:
        Rich Table object.
    """
    table = Table(title="[bold blue]E2E ML Workflow Summary[/bold blue]")
    table.add_column("Component", style="cyan", width=30)
    table.add_column("Value", style="white")

    # * Configuration summary
    table.add_section()
    table.add_row("Dataset", config.data.dataset_name)
    table.add_row("Model", "SimpleCNN")
    table.add_row("Epochs", str(config.training.epochs))
    table.add_row("Batch Size", str(config.data.batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Device", DEVICE)

    # * Metrics summary
    if metrics:
        table.add_section()
        table.add_row(
            "Best Val Accuracy", f"{metrics.get('best_val_accuracy', 0):.2f}%"
        )
        table.add_row("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2f}%")
        table.add_row("Test Loss", f"{metrics.get('test_loss', 0):.4f}")

    return table


def train_pipeline(config: DictConfig) -> dict[str, Any]:
    """Execute the training pipeline.

    Args:
        config: Hydra configuration.

    Returns:
        Dictionary containing training results and metrics.
    """
    logger.info("Starting training pipeline...")

    # * Set up MLflow
    mlflow_manager = MLflowManager(config.mlflow)
    mlflow_manager.create_artifacts_directories()

    with mlflow_manager.start_run():
        # * Log configuration
        mlflow_manager.log_config(config)

        # * Log Hydra configurations
        mlflow_manager.log_hydra_configs(config)

        # * Create data module
        data_module = create_data_module(config.data)
        data_module.prepare_data()
        data_module.setup()

        # * Create model
        model = create_model(config.model)
        mlflow_manager.log_parameters({
            "model_input_channels": config.model.input_channels,
            "model_num_classes": config.model.num_classes,
            "model_dropout_rate": config.model.dropout_rate,
            "model_pool_type": config.model.pool_type,
        })

        # * Create trainer with model directory
        # ! Temporarily disable struct mode to add model_dir
        OmegaConf.set_struct(config.training, False)
        config.training.model_dir = config.paths.model_dir
        OmegaConf.set_struct(config.training, True)

        trainer = create_trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            test_loader=data_module.test_dataloader(),
            cfg=config.training,
            device=DEVICE,
        )

        # * Log training parameters
        mlflow_manager.log_parameters({
            "training_epochs": config.training.epochs,
            "training_learning_rate": config.training.learning_rate,
            "data_batch_size": config.data.batch_size,
            "data_num_workers": config.data.num_workers,
        })

        # * Train the model
        history = trainer.train(
            epochs=config.training.epochs,
            log_every_n_steps=getattr(config.training, "log_every_n_steps", 50),
        )

        # * Evaluate the model
        test_metrics = trainer.evaluate()

        # * Log model
        mlflow_manager.log_model(
            model=model,
            model_name="pytorch_model",
            registered_model_name=config.mlflow.model_registry.model_name,
        )

        # * Register model if enabled
        if config.mlflow.model_registry.enabled:
            mlflow_manager.register_model(
                model_name=config.mlflow.model_registry.model_name,
                stage=config.mlflow.model_registry.stage,
            )

        # * Log artifacts
        if config.mlflow.log_artifacts:
            mlflow_manager.log_artifacts(config.paths.model_dir, "models")
            mlflow_manager.log_artifacts(config.paths.log_dir, "logs")

        # * Prepare results
        results = {
            "best_val_accuracy": trainer.best_val_accuracy,
            "test_accuracy": test_metrics["test_accuracy"],
            "test_loss": test_metrics["test_loss"],
            "history": history,
        }

        logger.info("Training pipeline completed successfully")
        return results


def eval_pipeline(config: DictConfig) -> dict[str, Any]:
    """Execute the evaluation pipeline.

    Args:
        config: Hydra configuration.

    Returns:
        Dictionary containing evaluation results.
    """
    logger.info("Starting evaluation pipeline...")

    # * Set up MLflow
    mlflow_manager = MLflowManager(config.mlflow)

    with mlflow_manager.start_run():
        # * Log configuration
        mlflow_manager.log_config(config)

        # * Create data module
        data_module = create_data_module(config.data)
        data_module.prepare_data()
        data_module.setup()

        # * Create model
        model = create_model(config.model)

        # * Load best model checkpoint
        best_model_path = Path(config.paths.model_dir) / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.warning(f"Best model checkpoint not found at {best_model_path}")
            return {"error": "Best model checkpoint not found"}

        # * Create trainer for evaluation with model directory
        # ! Temporarily disable struct mode to add model_dir
        OmegaConf.set_struct(config.training, False)
        config.training.model_dir = config.paths.model_dir
        OmegaConf.set_struct(config.training, True)

        trainer = create_trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            test_loader=data_module.test_dataloader(),
            cfg=config.training,
            device=DEVICE,
        )

        # * Evaluate the model
        test_metrics = trainer.evaluate()

        # * Log metrics
        mlflow_manager.log_metrics(test_metrics)

        logger.info("Evaluation pipeline completed successfully")
        return test_metrics


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function orchestrated by Hydra.

    Args:
        config: Hydra configuration object.
    """
    # * Set up logging
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # * Remove default handler
    logger.remove()

    # * Add file handler
    logger.add(
        config.log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # * Add console handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # * Create timestamped output directory structure
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    timestamped_output_dir = Path(config.paths.output_dir) / timestamp
    timestamped_output_dir.mkdir(parents=True, exist_ok=True)

    # * Update paths to use timestamped directory for run-specific artifacts
    # ! mlruns_dir stays at the shared location from config
    config.paths.model_dir = str(timestamped_output_dir / "models")
    config.paths.log_dir = str(timestamped_output_dir / "logs")
    # * Store run output dir for display (not in config struct)
    run_output_dir = str(timestamped_output_dir)

    # * Create subdirectories for run-specific artifacts
    (timestamped_output_dir / "models").mkdir(exist_ok=True)
    (timestamped_output_dir / "logs").mkdir(exist_ok=True)
    (timestamped_output_dir / "plots").mkdir(exist_ok=True)

    # * Ensure shared mlruns directory exists (for all runs to use)
    Path(config.paths.mlruns_dir).mkdir(parents=True, exist_ok=True)

    # * Set up MLflow
    setup_mlflow_logging(config.mlflow)

    # * Log Docker environment info
    if os.getenv("DOCKER_CONTAINER"):
        logger.info("Running in Docker container")
        logger.info(f"MLflow tracking URI: {config.mlflow.tracking_uri}")

    # * Create console for output
    console = Console()

    # * Log device information
    logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device count: {torch.cuda.device_count()}")
        logger.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")

        # * Log ROCm-specific information for AMD GPUs
        if hasattr(torch.version, "hip") and torch.version.hip:
            logger.info(f"ROCm HIP version: {torch.version.hip}")
            # ? Try to get GFX architecture from environment or device properties
            gfx_arch = os.getenv("HSA_OVERRIDE_GFX_VERSION", "native")
            logger.info(f"HSA_OVERRIDE_GFX_VERSION: {gfx_arch}")

        logger.info(f"Training device: {DEVICE} (GPU)")
    else:
        logger.info(f"Training device: {DEVICE} (CPU)")

    logger.info(f"Starting {config.app_name} in {config.mode} mode...")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    try:
        if config.mode == "train":
            results = train_pipeline(config)
        elif config.mode == "eval":
            results = eval_pipeline(config)
        elif config.mode == "full":
            # * Run both training and evaluation
            train_results = train_pipeline(config)
            eval_results = eval_pipeline(config)
            results = {**train_results, **eval_results}
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        # * Display summary
        summary_table = create_summary_table(config, results)
        console.print(summary_table)

        # * Display output locations and MLflow UI instructions
        mlflow_uri = config.mlflow.tracking_uri
        console.print()
        console.print(f"[bold green]📁 Run artifacts: {run_output_dir}[/bold green]")
        console.print(
            f"[bold green]📊 MLflow tracking: {config.paths.mlruns_dir}[/bold green]"
        )

        if mlflow_uri.startswith("file:"):
            console.print()
            console.print("[bold blue]🔍 To view MLflow UI, run:[/bold blue]")
            console.print(f"   [cyan]mlflow ui --backend-store-uri {mlflow_uri}[/cyan]")
            console.print("   [cyan]Then open: http://localhost:5000[/cyan]")
        else:
            console.print()
            console.print("[bold blue]🔍 MLflow UI available at:[/bold blue]")
            console.print("   [cyan]http://localhost:5000[/cyan]")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.opt(exception=True).error(f"Pipeline failed: {e}")
        console.print(f"[bold red]Pipeline failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
