"""CIFAR-10 dataset module with PyTorch DataLoader integration."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# =============================== Constants ================================== #
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class CIFAR10DataModule:
    """Data module for CIFAR-10 dataset with train/val/test splits.

    This class handles downloading, preprocessing, and creating DataLoaders
    for the CIFAR-10 dataset with proper train/validation/test splits.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        download: bool = True,
        transform: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        shuffle_train: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> None:
        """Initialize the CIFAR-10 data module.

        Args:
            dataset_name: Name of the dataset (for logging).
            data_dir: Directory to store the dataset.
            download: Whether to download the dataset if not present.
            transform: Dictionary containing transform configurations for train/val.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            shuffle_train: Whether to shuffle training data.
            train_split: Fraction of data to use for training.
            val_split: Fraction of data to use for validation.
            test_split: Fraction of data to use for testing.

        Raises:
            ValueError: If train_split + val_split + test_split != 1.0.
        """
        if not (abs(train_split + val_split + test_split - 1.0) < 1e-6):
            raise ValueError(
                f"Split fractions must sum to 1.0, got "
                f"train={train_split}, val={val_split}, test={test_split}"
            )

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # * Create transforms from configuration
        self.transform = self._create_transforms(transform)

        # * Initialize dataset attributes
        self.train_dataset: Optional[datasets.CIFAR10] = None
        self.val_dataset: Optional[datasets.CIFAR10] = None
        self.test_dataset: Optional[datasets.CIFAR10] = None

        logger.info(f"Initialized {dataset_name} data module")

    def _create_transforms(
        self, transform_config: Optional[Dict[str, Any]]
    ) -> Dict[str, transforms.Compose]:
        """Create transform compositions from configuration.

        Args:
            transform_config: Dictionary containing transform configurations.

        Returns:
            Dictionary with 'train' and 'val' transform compositions.
        """
        if transform_config is None:
            # * Default transforms if none provided
            transform_config = {
                "train": [
                    {"_target_": "torchvision.transforms.ToTensor"},
                    {
                        "_target_": "torchvision.transforms.Normalize",
                        "mean": [0.4914, 0.4822, 0.4465],
                        "std": [0.2023, 0.1994, 0.2010],
                    },
                ],
                "val": [
                    {"_target_": "torchvision.transforms.ToTensor"},
                    {
                        "_target_": "torchvision.transforms.Normalize",
                        "mean": [0.4914, 0.4822, 0.4465],
                        "std": [0.2023, 0.1994, 0.2010],
                    },
                ],
            }

        transforms_dict = {}
        for split in ["train", "val"]:
            if split in transform_config:
                transform_list = []
                for transform_spec in transform_config[split]:
                    # * Extract the target class and parameters
                    target = transform_spec["_target_"]
                    # * Create a copy without the _target_ key
                    params = {
                        k: v for k, v in transform_spec.items() if k != "_target_"
                    }
                    transform_class = self._get_transform_class(target)
                    transform_list.append(transform_class(**params))

                transforms_dict[split] = transforms.Compose(transform_list)
            else:
                # * Fallback to identity transform
                transforms_dict[split] = transforms.Compose([transforms.ToTensor()])

        return transforms_dict

    def _get_transform_class(self, target: str) -> type:
        """Get transform class from target string.

        Args:
            target: String like 'torchvision.transforms.RandomCrop'.

        Returns:
            The transform class.

        Raises:
            ImportError: If the transform class cannot be imported.
        """
        try:
            module_path, class_name = target.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import transform class {target}: {e}")

    def prepare_data(self) -> None:
        """Download and prepare the dataset."""
        logger.info(f"Preparing {self.dataset_name} dataset...")

        # * Download training data
        datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=self.download,
            transform=transforms.ToTensor(),  # * Use basic transform for download
        )

        # * Download test data
        datasets.CIFAR10(
            root=str(self.data_dir),
            train=False,
            download=self.download,
            transform=transforms.ToTensor(),  # * Use basic transform for download
        )

        logger.info(f"Dataset prepared in {self.data_dir}")

    def setup(self) -> None:
        """Set up train/val/test datasets with proper splits."""
        logger.info("Setting up train/val/test splits...")

        # * Load full training dataset
        full_train_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=False,
            transform=self.transform["train"],
        )

        # * Load test dataset
        self.test_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=False,
            download=False,
            transform=self.transform["val"],  # * Use val transforms for test
        )

        # * Calculate split sizes
        total_size = len(full_train_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        # * Remaining samples go to train if there's any rounding error
        remaining_size = total_size - train_size - val_size
        train_size += remaining_size

        # * Create splits
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # * For reproducibility
        )

        logger.info(
            f"Dataset splits - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            DataLoader for training data.

        Raises:
            RuntimeError: If setup() hasn't been called yet.
        """
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before creating dataloaders")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # * Drop last incomplete batch for consistent training
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            DataLoader for validation data.

        Raises:
            RuntimeError: If setup() hasn't been called yet.
        """
        if self.val_dataset is None:
            raise RuntimeError("Must call setup() before creating dataloaders")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader.

        Returns:
            DataLoader for test data.

        Raises:
            RuntimeError: If setup() hasn't been called yet.
        """
        if self.test_dataset is None:
            raise RuntimeError("Must call setup() before creating dataloaders")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_class_names(self) -> list[str]:
        """Get list of class names.

        Returns:
            List of CIFAR-10 class names.
        """
        return CIFAR10_CLASSES.copy()

    def get_num_classes(self) -> int:
        """Get number of classes.

        Returns:
            Number of classes in the dataset.
        """
        return len(CIFAR10_CLASSES)


def create_data_module(cfg: DictConfig) -> CIFAR10DataModule:
    """Create a CIFAR10DataModule from configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Configured CIFAR10DataModule instance.
    """
    return CIFAR10DataModule(
        dataset_name=cfg.dataset_name,
        data_dir=cfg.data_dir,
        download=cfg.download,
        transform=cfg.transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle_train=cfg.shuffle_train,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
    )
