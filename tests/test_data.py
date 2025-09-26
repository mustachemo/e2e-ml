"""Tests for data loading and preprocessing modules."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from src.data import CIFAR10DataModule


class TestCIFAR10DataModule:
    """Test cases for CIFAR10DataModule."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return OmegaConf.create({
            "dataset_name": "CIFAR10",
            "data_dir": "./data",
            "download": False,  # Don't download during tests
            "transform": {
                "train": [
                    {"_target_": "torchvision.transforms.ToTensor"},
                    {"_target_": "torchvision.transforms.Normalize",
                     "mean": [0.4914, 0.4822, 0.4465],
                     "std": [0.2023, 0.1994, 0.2010]}
                ],
                "val": [
                    {"_target_": "torchvision.transforms.ToTensor"},
                    {"_target_": "torchvision.transforms.Normalize",
                     "mean": [0.4914, 0.4822, 0.4465],
                     "std": [0.2023, 0.1994, 0.2010]}
                ]
            },
            "batch_size": 32,
            "num_workers": 0,  # Use 0 for testing
            "pin_memory": False,
            "shuffle_train": True,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
        })

    def test_initialization(self, mock_config):
        """Test data module initialization."""
        data_module = CIFAR10DataModule(**mock_config)

        assert data_module.dataset_name == "CIFAR10"
        assert data_module.batch_size == 32
        assert data_module.train_split == 0.8
        assert data_module.val_split == 0.1
        assert data_module.test_split == 0.1

    def test_invalid_split_fractions(self):
        """Test that invalid split fractions raise ValueError."""
        with pytest.raises(ValueError, match="Split fractions must sum to 1.0"):
            CIFAR10DataModule(
                dataset_name="CIFAR10",
                data_dir="./data",
                train_split=0.5,
                val_split=0.3,
                test_split=0.1,  # Sums to 0.9, not 1.0
            )

    def test_get_class_names(self, mock_config):
        """Test getting class names."""
        data_module = CIFAR10DataModule(**mock_config)
        class_names = data_module.get_class_names()

        assert len(class_names) == 10
        assert "airplane" in class_names
        assert "automobile" in class_names

    def test_get_num_classes(self, mock_config):
        """Test getting number of classes."""
        data_module = CIFAR10DataModule(**mock_config)
        num_classes = data_module.get_num_classes()

        assert num_classes == 10

    @patch('torchvision.datasets.CIFAR10')
    def test_prepare_data(self, mock_cifar10, mock_config, temp_dir):
        """Test data preparation."""
        # * Mock the CIFAR10 dataset
        mock_cifar10.return_value = None

        data_module = CIFAR10DataModule(
            **{**mock_config, "data_dir": str(temp_dir)}
        )

        # * Should not raise an exception
        data_module.prepare_data()

        # * Verify CIFAR10 was called twice (train and test)
        assert mock_cifar10.call_count == 2

    @patch('torchvision.datasets.CIFAR10')
    def test_setup(self, mock_cifar10, mock_config, temp_dir):
        """Test dataset setup with train/val/test splits."""
        # * Create mock datasets
        mock_train_dataset = type('MockDataset', (), {
            '__len__': lambda self: 100,
            '__getitem__': lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item())
        })()

        mock_test_dataset = type('MockDataset', (), {
            '__len__': lambda self: 20,
            '__getitem__': lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item())
        })()

        # * Configure mock to return different datasets for train=True/False
        def mock_cifar10_side_effect(root, train, download, transform):
            if train:
                return mock_train_dataset
            else:
                return mock_test_dataset

        mock_cifar10.side_effect = mock_cifar10_side_effect

        data_module = CIFAR10DataModule(
            **{**mock_config, "data_dir": str(temp_dir)}
        )
        data_module.setup()

        # * Check that datasets were created
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None

        # * Check split sizes (100 total, 90 train, 10 val, 20 test)
        assert len(data_module.train_dataset) == 90
        assert len(data_module.val_dataset) == 10
        assert len(data_module.test_dataset) == 20

    @patch('torchvision.datasets.CIFAR10')
    def test_dataloaders(self, mock_cifar10, mock_config, temp_dir):
        """Test DataLoader creation."""
        # * Create mock datasets
        mock_dataset = type('MockDataset', (), {
            '__len__': lambda self: 100,
            '__getitem__': lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item())
        })()

        mock_cifar10.return_value = mock_dataset

        data_module = CIFAR10DataModule(
            **{**mock_config, "data_dir": str(temp_dir)}
        )
        data_module.setup()

        # * Test train dataloader
        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert train_loader.batch_size == 32
        # * DataLoader doesn't expose shuffle as an attribute, check the sampler instead
        assert hasattr(train_loader, 'sampler')

        # * Test val dataloader
        val_loader = data_module.val_dataloader()
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert val_loader.batch_size == 32
        # * DataLoader doesn't expose shuffle as an attribute, check the sampler instead
        assert hasattr(val_loader, 'sampler')

        # * Test test dataloader
        test_loader = data_module.test_dataloader()
        assert isinstance(test_loader, torch.utils.data.DataLoader)
        assert test_loader.batch_size == 32
        # * DataLoader doesn't expose shuffle as an attribute, check the sampler instead
        assert hasattr(test_loader, 'sampler')

    def test_dataloader_without_setup(self, mock_config):
        """Test that dataloaders raise error if setup not called."""
        data_module = CIFAR10DataModule(**mock_config)

        with pytest.raises(RuntimeError, match="Must call setup"):
            data_module.train_dataloader()

        with pytest.raises(RuntimeError, match="Must call setup"):
            data_module.val_dataloader()

        with pytest.raises(RuntimeError, match="Must call setup"):
            data_module.test_dataloader()
