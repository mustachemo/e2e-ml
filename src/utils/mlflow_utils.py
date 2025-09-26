"""MLflow utilities for experiment tracking and model management."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pytorch
import torch
from loguru import logger
from omegaconf import DictConfig


class MLflowManager:
    """Manager class for MLflow experiment tracking and model logging.

    Handles experiment creation, run management, parameter logging,
    metric tracking, and model artifact storage.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize MLflow manager with configuration.

        Args:
            cfg: MLflow configuration object.
        """
        self.cfg = cfg
        self.experiment_name = cfg.experiment_name
        self.run_name = cfg.run_name
        self.tracking_uri = cfg.tracking_uri

        # * Set up MLflow tracking
        mlflow.set_tracking_uri(self.tracking_uri)

        # * Create or get experiment
        self.experiment_id = self._setup_experiment()

        logger.info(f"Initialized MLflow manager for experiment: {self.experiment_name}")

    def _setup_experiment(self) -> str:
        """Set up MLflow experiment.

        Returns:
            Experiment ID.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")

            return experiment_id
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run.

        Args:
            run_name: Optional custom run name.

        Returns:
            Active MLflow run object.
        """
        run_name = run_name or self.run_name
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        )
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log.
        """
        if not self.cfg.log_params:
            return

        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number for the metrics.
        """
        if not self.cfg.log_metrics:
            return

        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str = "pytorch_model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log PyTorch model to MLflow.

        Args:
            model: PyTorch model to log.
            model_name: Name for the model artifact.
            registered_model_name: Optional name for model registry.
        """
        if not self.cfg.log_models:
            return

        try:
            # * Create input example for model signature inference (convert to numpy)
            input_example = torch.randn(1, 3, 32, 32).numpy()  # * CIFAR-10 input shape
            
            # * Log model as artifact
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=model_name,  # * Use 'name' instead of deprecated 'artifact_path'
                registered_model_name=registered_model_name,
                input_example=input_example,  # * Add input example for signature inference
            )
            logger.info(f"Logged model '{model_name}' to MLflow")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_artifacts(self, artifacts_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log directory of artifacts to MLflow.

        Args:
            artifacts_dir: Directory containing artifacts to log.
            artifact_path: Optional path within the run to store artifacts.
        """
        if not self.cfg.log_artifacts:
            return

        try:
            mlflow.log_artifacts(artifacts_dir, artifact_path)
            logger.info(f"Logged artifacts from {artifacts_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")

    def log_figure(self, figure, artifact_name: str) -> None:
        """Log matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure object.
            artifact_name: Name for the figure artifact.
        """
        if not self.cfg.log_artifacts:
            return

        try:
            mlflow.log_figure(figure, artifact_name)
            logger.info(f"Logged figure '{artifact_name}' to MLflow")
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    def log_text(self, text: str, artifact_name: str) -> None:
        """Log text content to MLflow.

        Args:
            text: Text content to log.
            artifact_name: Name for the text artifact.
        """
        if not self.cfg.log_artifacts:
            return

        try:
            mlflow.log_text(text, artifact_name)
            logger.info(f"Logged text '{artifact_name}' to MLflow")
        except Exception as e:
            logger.error(f"Failed to log text: {e}")

    def log_config(self, config: DictConfig, config_name: str = "config.yaml") -> None:
        """Log Hydra configuration to MLflow.

        Args:
            config: Hydra configuration object.
            config_name: Name for the config artifact.
        """
        if not self.cfg.log_artifacts:
            return

        try:
            # * Convert config to YAML string
            from omegaconf import OmegaConf
            config_yaml = OmegaConf.to_yaml(config)

            mlflow.log_text(config_yaml, config_name)
            logger.info(f"Logged configuration '{config_name}' to MLflow")
        except Exception as e:
            logger.error(f"Failed to log configuration: {e}")

    def register_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        stage: str = "None",
    ) -> None:
        """Register model in MLflow Model Registry.

        Args:
            model_name: Name of the model to register.
            model_version: Optional specific version to register.
            stage: Stage to assign to the model (None, Staging, Production, Archived).
        """
        if not self.cfg.model_registry.enabled:
            return

        try:
            # * Get the current run
            current_run = mlflow.active_run()
            if current_run is None:
                logger.warning("No active run found, cannot register model")
                return

            # * Get model URI from current run
            model_uri = f"runs:/{current_run.info.run_id}/pytorch_model"

            # * Register model
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )

            # * Set stage if specified
            if stage != "None":
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model.version,
                    stage=stage,
                )
                logger.info(f"Registered model '{model_name}' version {registered_model.version} in stage '{stage}'")
            else:
                logger.info(f"Registered model '{model_name}' version {registered_model.version}")

        except Exception as e:
            logger.error(f"Failed to register model: {e}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")

    def get_run_info(self) -> Dict[str, Any]:
        """Get information about the current run.

        Returns:
            Dictionary containing run information.
        """
        try:
            current_run = mlflow.active_run()
            if current_run is None:
                return {}

            return {
                "run_id": current_run.info.run_id,
                "experiment_id": current_run.info.experiment_id,
                "run_name": current_run.data.tags.get("mlflow.runName", ""),
                "status": current_run.info.status,
                "start_time": current_run.info.start_time,
                "end_time": current_run.info.end_time,
            }
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            return {}

    def create_artifacts_directories(self) -> None:
        """Create directories for storing artifacts."""
        try:
            # * Create artifact directories
            for artifact_type in ["plots", "logs", "models"]:
                artifact_dir = Path(self.cfg.artifacts[f"{artifact_type}_dir"])
                artifact_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created artifact directory: {artifact_dir}")
        except Exception as e:
            logger.error(f"Failed to create artifact directories: {e}")


def setup_mlflow_logging(cfg: DictConfig) -> None:
    """Set up MLflow logging configuration.

    Args:
        cfg: MLflow configuration object.
    """
    # * Set environment variables for MLflow
    os.environ["MLFLOW_TRACKING_URI"] = cfg.tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.experiment_name

    logger.info(f"MLflow tracking URI: {cfg.tracking_uri}")
    logger.info(f"MLflow experiment: {cfg.experiment_name}")
