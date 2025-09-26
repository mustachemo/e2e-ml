"""PyTorch training utilities with MLflow integration."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Early stopping utility to prevent overfitting.

    Monitors a metric and stops training if it doesn't improve for a
    specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            monitor: Metric to monitor.
            mode: 'min' for loss metrics, 'max' for accuracy metrics.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early.

        Args:
            metrics: Dictionary of current metrics.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return False

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")

        return self.early_stop

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current score is an improvement.

        Args:
            current: Current metric value.
            best: Best metric value so far.

        Returns:
            True if current is an improvement, False otherwise.
        """
        if self.mode == "min":
            return current < best - self.min_delta
        else:  # mode == "max"
            return current > best + self.min_delta


class PyTorchTrainer:
    """PyTorch trainer with MLflow integration and comprehensive logging.

    Handles training, validation, checkpointing, and experiment tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_dir: str = "./models",
        early_stopping: Optional[EarlyStopping] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader.
            optimizer: Optimizer for training.
            criterion: Loss function.
            scheduler: Learning rate scheduler.
            device: Device to run training on.
            model_dir: Directory to save model checkpoints.
            early_stopping: Early stopping configuration.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.model_dir = Path(model_dir)
        self.early_stopping = early_stopping

        # * Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # * Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []

        logger.info(f"Initialized trainer on device: {self.device}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # * Zero gradients
            self.optimizer.zero_grad()

            # * Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # * Backward pass
            loss.backward()
            self.optimizer.step()

            # * Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # * Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.0 * correct / total:.2f}%",
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch.

        Returns:
            Tuple of (average_loss, accuracy) for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, epochs: int, log_every_n_steps: int = 50) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs.

        Args:
            epochs: Number of epochs to train.
            log_every_n_steps: Log metrics every N steps.

        Returns:
            Dictionary containing training history.
        """
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # * Train for one epoch
            train_loss, train_acc = self.train_epoch()

            # * Validate
            val_loss, val_acc = self.validate_epoch()

            # * Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # * Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # * Log metrics to MLflow
            current_lr = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
                "epoch": epoch + 1,
            }, step=epoch)

            # * Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"Time: {epoch_time:.2f}s"
            )

            # * Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.save_checkpoint(is_best=True)

            # * Check early stopping
            if self.early_stopping is not None:
                metrics = {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
                if self.early_stopping(metrics):
                    logger.info("Early stopping triggered, stopping training")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set.

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating model on test set...")
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        test_loss = total_loss / len(self.test_loader)
        test_accuracy = 100.0 * correct / total

        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # * Log test metrics to MLflow
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        })

        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # * Save latest checkpoint
        checkpoint_path = self.model_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # * Save best checkpoint
        if is_best:
            best_path = self.model_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_accuracies = checkpoint["train_accuracies"]
        self.val_accuracies = checkpoint["val_accuracies"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: DictConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> PyTorchTrainer:
    """Create a PyTorchTrainer from configuration.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        cfg: Training configuration object.
        device: Device to run training on.

    Returns:
        Configured PyTorchTrainer instance.
    """
    # * Create optimizer
    optimizer_class = getattr(torch.optim, cfg.optimizer._target_.split('.')[-1])
    optimizer = optimizer_class(
        model.parameters(),
        **{k: v for k, v in cfg.optimizer.items() if k != "_target_"}
    )

    # * Create loss function
    loss_class = getattr(torch.nn, cfg.loss_function._target_.split('.')[-1])
    criterion = loss_class(**{
        k: v for k, v in cfg.loss_function.items() if k != "_target_"
    })

    # * Create scheduler
    scheduler = None
    if "scheduler" in cfg and cfg.scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.scheduler._target_.split('.')[-1])
        scheduler = scheduler_class(
            optimizer,
            **{k: v for k, v in cfg.scheduler.items() if k != "_target_"}
        )

    # * Create early stopping
    early_stopping = None
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            monitor=cfg.early_stopping.monitor,
            mode=cfg.early_stopping.mode,
        )

    return PyTorchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        model_dir=cfg.get("model_dir", "./models"),
        early_stopping=early_stopping,
    )
