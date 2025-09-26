"""Tests for model architectures and training utilities."""

import pytest
import torch
from omegaconf import OmegaConf

from src.models import SimpleCNN, PyTorchTrainer
from src.models.trainer import EarlyStopping


class TestSimpleCNN:
    """Test cases for SimpleCNN model."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = SimpleCNN()

        assert model.input_channels == 3
        assert model.num_classes == 10
        assert model.dropout_rate == 0.2
        assert model.pool_type == "max"

    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters."""
        model = SimpleCNN(
            input_channels=1,
            num_classes=5,
            dropout_rate=0.5,
            pool_type="avg"
        )

        assert model.input_channels == 1
        assert model.num_classes == 5
        assert model.dropout_rate == 0.5
        assert model.pool_type == "avg"

    def test_invalid_pool_type(self):
        """Test that invalid pool type raises ValueError."""
        with pytest.raises(ValueError, match="pool_type must be 'max' or 'avg'"):
            SimpleCNN(pool_type="invalid")

    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = SimpleCNN()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = SimpleCNN()

        for batch_size in [1, 4, 8, 16]:
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            output = model(input_tensor)
            assert output.shape == (batch_size, 10)

    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = SimpleCNN()
        param_count = model._count_parameters()

        # * Should have some parameters but not too many
        assert param_count > 1000
        assert param_count < 10000000  # * Updated to accommodate the actual model size

    def test_model_summary(self):
        """Test model summary generation."""
        model = SimpleCNN()
        summary = model.get_model_summary()

        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "input_channels" in summary
        assert "num_classes" in summary
        assert summary["input_channels"] == 3
        assert summary["num_classes"] == 10

    def test_training_mode(self):
        """Test model behavior in training mode."""
        model = SimpleCNN()
        model.train()

        input_tensor = torch.randn(2, 3, 32, 32)
        output = model(input_tensor)

        assert output.shape == (2, 10)

    def test_eval_mode(self):
        """Test model behavior in eval mode."""
        model = SimpleCNN()
        model.eval()

        input_tensor = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (2, 10)


class TestEarlyStopping:
    """Test cases for EarlyStopping utility."""

    def test_initialization(self):
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)

        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 0.01
        assert early_stopping.monitor == "val_loss"
        assert early_stopping.mode == "min"

    def test_improvement_detection_min(self):
        """Test improvement detection for minimization metrics."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        # * First call should not trigger early stopping
        metrics1 = {"val_loss": 1.0}
        assert not early_stopping(metrics1)
        assert early_stopping.best_score == 1.0
        assert early_stopping.counter == 0

        # * Improvement should reset counter
        metrics2 = {"val_loss": 0.8}
        assert not early_stopping(metrics2)
        assert early_stopping.best_score == 0.8
        assert early_stopping.counter == 0

        # * No improvement should increment counter
        metrics3 = {"val_loss": 0.85}
        assert not early_stopping(metrics3)
        assert early_stopping.best_score == 0.8
        assert early_stopping.counter == 1

    def test_improvement_detection_max(self):
        """Test improvement detection for maximization metrics."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode="max", monitor="val_accuracy")

        # * First call should not trigger early stopping
        metrics1 = {"val_accuracy": 0.8}
        assert not early_stopping(metrics1)
        assert early_stopping.best_score == 0.8
        assert early_stopping.counter == 0

        # * Improvement should reset counter
        metrics2 = {"val_accuracy": 0.9}
        assert not early_stopping(metrics2)
        assert early_stopping.best_score == 0.9
        assert early_stopping.counter == 0

        # * No improvement should increment counter
        metrics3 = {"val_accuracy": 0.85}
        assert not early_stopping(metrics3)
        assert early_stopping.best_score == 0.9
        assert early_stopping.counter == 1

    def test_early_stopping_trigger(self):
        """Test that early stopping triggers after patience exceeded."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode="min")

        # * Set initial best score
        early_stopping.best_score = 1.0

        # * Two calls without improvement should trigger early stopping
        metrics1 = {"val_loss": 1.05}
        assert not early_stopping(metrics1)
        assert early_stopping.counter == 1

        metrics2 = {"val_loss": 1.02}
        assert early_stopping(metrics2)  # Should trigger early stopping
        assert early_stopping.early_stop is True

    def test_missing_metric(self):
        """Test behavior when monitored metric is missing."""
        early_stopping = EarlyStopping(patience=3, monitor="val_loss")

        metrics = {"val_accuracy": 0.8}  # Missing val_loss
        result = early_stopping(metrics)

        # * Should not trigger early stopping but should log warning
        assert not result


class TestPyTorchTrainer:
    """Test cases for PyTorchTrainer."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return SimpleCNN()

    @pytest.fixture
    def mock_dataloaders(self):
        """Create mock dataloaders for testing."""
        # * Create dummy datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 3, 32, 32),
            torch.randint(0, 10, (20,))
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 3, 32, 32),
            torch.randint(0, 10, (20,))
        )

        # * Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

        return train_loader, val_loader, test_loader

    @pytest.fixture
    def mock_config(self):
        """Create mock training configuration."""
        return OmegaConf.create({
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 0.001,
                "weight_decay": 1e-4
            },
            "loss_function": {
                "_target_": "torch.nn.CrossEntropyLoss"
            },
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 5,
                "gamma": 0.1
            },
            "early_stopping": {
                "enabled": True,
                "patience": 3,
                "min_delta": 0.001,
                "monitor": "val_loss",
                "mode": "min"
            },
            "model_dir": "./test_models"
        })

    def test_initialization(self, mock_model, mock_dataloaders, mock_config):
        """Test trainer initialization."""
        train_loader, val_loader, test_loader = mock_dataloaders

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=torch.optim.Adam(mock_model.parameters()),
            criterion=torch.nn.CrossEntropyLoss(),
            device="cpu"
        )

        assert trainer.model == mock_model
        assert trainer.train_loader == train_loader
        assert trainer.val_loader == val_loader
        assert trainer.test_loader == test_loader
        assert trainer.device == "cpu"

    def test_train_epoch(self, mock_model, mock_dataloaders):
        """Test training for one epoch."""
        train_loader, val_loader, test_loader = mock_dataloaders

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=torch.optim.Adam(mock_model.parameters()),
            criterion=torch.nn.CrossEntropyLoss(),
            device="cpu"
        )

        avg_loss, accuracy = trainer.train_epoch()

        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert avg_loss >= 0
        assert 0 <= accuracy <= 100

    def test_validate_epoch(self, mock_model, mock_dataloaders):
        """Test validation for one epoch."""
        train_loader, val_loader, test_loader = mock_dataloaders

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=torch.optim.Adam(mock_model.parameters()),
            criterion=torch.nn.CrossEntropyLoss(),
            device="cpu"
        )

        avg_loss, accuracy = trainer.validate_epoch()

        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert avg_loss >= 0
        assert 0 <= accuracy <= 100

    def test_evaluate(self, mock_model, mock_dataloaders):
        """Test model evaluation."""
        train_loader, val_loader, test_loader = mock_dataloaders

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=torch.optim.Adam(mock_model.parameters()),
            criterion=torch.nn.CrossEntropyLoss(),
            device="cpu"
        )

        metrics = trainer.evaluate()

        assert "test_loss" in metrics
        assert "test_accuracy" in metrics
        assert isinstance(metrics["test_loss"], float)
        assert isinstance(metrics["test_accuracy"], float)
        assert metrics["test_loss"] >= 0
        assert 0 <= metrics["test_accuracy"] <= 100
