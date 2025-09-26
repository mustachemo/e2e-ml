"""Tests for main application module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from omegaconf import OmegaConf

from src.main import create_summary_table, setup_logging


class TestMainUtilities:
    """Test cases for main module utilities."""

    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test.log"

            # * Should not raise an exception
            setup_logging(str(log_file))

            # * Log file should be created
            assert log_file.exists()

    def test_create_summary_table(self):
        """Test summary table creation."""
        config = OmegaConf.create({
            "data": {"dataset_name": "CIFAR10", "batch_size": 32},
            "model": {"input_channels": 3, "num_classes": 10},
            "training": {"epochs": 10, "learning_rate": 0.001},
        })

        metrics = {
            "best_val_accuracy": 85.5,
            "test_accuracy": 84.2,
            "test_loss": 0.5234,
        }

        table = create_summary_table(config, metrics)

        # * Table should be created successfully
        assert table is not None
        assert len(table.columns) == 2
        assert table.title == "[bold blue]E2E ML Workflow Summary[/bold blue]"

    def test_create_summary_table_no_metrics(self):
        """Test summary table creation without metrics."""
        config = OmegaConf.create({
            "data": {"dataset_name": "CIFAR10", "batch_size": 32},
            "model": {"input_channels": 3, "num_classes": 10},
            "training": {"epochs": 10, "learning_rate": 0.001},
        })

        table = create_summary_table(config, {})

        # * Table should be created successfully even without metrics
        assert table is not None
        assert len(table.columns) == 2
