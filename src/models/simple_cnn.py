"""Simple CNN architecture for CIFAR-10 classification."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for CIFAR-10 classification.

    This model consists of multiple convolutional layers followed by
    fully connected layers with dropout for regularization.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        conv_layers: Optional[List[dict]] = None,
        pool_size: int = 2,
        pool_type: str = "max",
        dense_layers: Optional[List[int]] = None,
    ) -> None:
        """Initialize the SimpleCNN model.

        Args:
            input_channels: Number of input channels (3 for RGB images).
            num_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.
            conv_layers: List of dictionaries defining convolutional layers.
            pool_size: Size of pooling kernel.
            pool_type: Type of pooling ('max' or 'avg').
            dense_layers: List of hidden units for dense layers.

        Raises:
            ValueError: If pool_type is not 'max' or 'avg'.
        """
        super().__init__()

        if pool_type not in ["max", "avg"]:
            raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type}")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.pool_type = pool_type

        # * Default convolutional layers if none provided
        if conv_layers is None:
            conv_layers = [
                {"out_channels": 32, "kernel_size": 3, "padding": 1},
                {"out_channels": 64, "kernel_size": 3, "padding": 1},
                {"out_channels": 128, "kernel_size": 3, "padding": 1},
            ]

        # * Default dense layers if none provided
        if dense_layers is None:
            dense_layers = [512, 256]

        # * Build convolutional layers
        self.conv_layers = self._build_conv_layers(conv_layers)

        # * Calculate the size after convolutions and pooling
        # * CIFAR-10 images are 32x32, we'll calculate the final size
        self._calculate_conv_output_size()

        # * Build dense layers
        self.dense_layers = self._build_dense_layers(dense_layers)

        # * Final classification layer
        self.classifier = nn.Linear(dense_layers[-1], num_classes)

        # * Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        logger.info(f"Initialized SimpleCNN with {self._count_parameters()} parameters")

    def _build_conv_layers(self, conv_configs: List[dict]) -> nn.ModuleList:
        """Build convolutional layers from configuration.

        Args:
            conv_configs: List of dictionaries defining conv layers.

        Returns:
            ModuleList containing the convolutional layers.
        """
        layers = nn.ModuleList()
        in_channels = self.input_channels

        for i, config in enumerate(conv_configs):
            out_channels = config["out_channels"]
            kernel_size = config["kernel_size"]
            padding = config["padding"]

            # * Add convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            # * Add batch normalization
            layers.append(nn.BatchNorm2d(out_channels))

            # * Add ReLU activation
            layers.append(nn.ReLU(inplace=True))

            # * Add pooling layer (except for the last conv layer)
            if i < len(conv_configs) - 1:
                if self.pool_type == "max":
                    layers.append(nn.MaxPool2d(self.pool_size))
                else:
                    layers.append(nn.AvgPool2d(self.pool_size))

            in_channels = out_channels

        return layers

    def _build_dense_layers(self, dense_configs: List[int]) -> nn.ModuleList:
        """Build dense layers from configuration.

        Args:
            dense_configs: List of hidden units for dense layers.

        Returns:
            ModuleList containing the dense layers.
        """
        layers = nn.ModuleList()

        # * First dense layer connects from conv output
        layers.append(nn.Linear(self.conv_output_size, dense_configs[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(self.dropout_rate))

        # * Additional dense layers
        for i in range(1, len(dense_configs)):
            layers.append(nn.Linear(dense_configs[i-1], dense_configs[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout_rate))

        return layers

    def _calculate_conv_output_size(self) -> None:
        """Calculate the output size after convolutional layers.

        This is needed to determine the input size for the first dense layer.
        """
        # * Create a dummy input to calculate the output size
        dummy_input = torch.zeros(1, self.input_channels, 32, 32)

        # * Forward pass through conv layers only
        x = dummy_input
        for layer in self.conv_layers:
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                x = layer(x)
            elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
                x = layer(x)

        # * Flatten to get the size
        self.conv_output_size = x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # * Forward through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # * Flatten for dense layers
        x = x.view(x.size(0), -1)

        # * Forward through dense layers
        for layer in self.dense_layers:
            x = layer(x)

        # * Final classification
        x = self.classifier(x)

        return x

    def _count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> dict:
        """Get a summary of the model architecture.

        Returns:
            Dictionary containing model summary information.
        """
        total_params = self._count_parameters()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "pool_type": self.pool_type,
        }


def create_model(cfg) -> SimpleCNN:
    """Create a SimpleCNN model from configuration.

    Args:
        cfg: Model configuration object.

    Returns:
        Configured SimpleCNN instance.
    """
    return SimpleCNN(
        input_channels=cfg.input_channels,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
        conv_layers=cfg.conv_layers,
        pool_size=cfg.pool_size,
        pool_type=cfg.pool_type,
        dense_layers=cfg.dense_layers,
    )
