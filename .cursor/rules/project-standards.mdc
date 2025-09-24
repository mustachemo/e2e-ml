# Project Standards & Best Practices

## Code Quality & Consistency

All code in this ML lifecycle project must adhere to strict quality standards ensuring maintainability, reproducibility, and collaboration efficiency.

### Python Code Standards

#### Type Hinting Requirements
```python
# All functions must have comprehensive type hints
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import pandas as pd
import torch

def process_dataset(
    data_path: Path,
    target_column: str,
    preprocessing_config: Dict[str, Any],
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Processes dataset with specified configuration.

    Args:
        data_path: Path to the dataset file
        target_column: Name of the target column for prediction
        preprocessing_config: Configuration dictionary for preprocessing steps
        split_ratios: Train/validation/test split ratios

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames

    Raises:
        FileNotFoundError: If data_path does not exist
        ValueError: If split_ratios don't sum to 1.0
    """
    pass
```

#### Error Handling Patterns
```python
# Specific exception handling with context
def load_model_safely(model_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Loads PyTorch model with comprehensive error handling."""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = torch.load(model_path, map_location=device)
        model.eval()

        logger.info(f"Successfully loaded model from {model_path}")
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except RuntimeError as e:
        logger.error(f"Failed to load model due to PyTorch error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e
```

#### Configuration Management
```python
# Use dataclasses for configuration structures
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Nested configurations
    optimizer_config: Dict[str, Any] = field(default_factory=lambda: {
        "type": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-8
    })

    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {
        "type": "cosine",
        "warmup_epochs": 5,
        "min_lr": 1e-6
    })

    def validate(self) -> None:
        """Validates configuration parameters."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
```

### Documentation Standards

#### Module Documentation
```python
"""Module for distributed model training using Ray and MLflow.

This module provides comprehensive training capabilities including:
- Distributed training across multiple GPUs/nodes
- Automatic experiment tracking with MLflow
- Hyperparameter optimization with Ray Tune
- Model checkpointing and recovery

Example:
    Basic training workflow:

    ```python
    from src.training.distributed_trainer import DistributedTrainer

    trainer = DistributedTrainer(config)
    results = trainer.train(model, dataset)
    ```

Attributes:
    DEFAULT_CONFIG (Dict[str, Any]): Default training configuration
    SUPPORTED_OPTIMIZERS (List[str]): List of supported optimizer types
"""

# Module-level constants
DEFAULT_CONFIG = {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3
}

SUPPORTED_OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop"]
```

#### Class Documentation
```python
class MLModelRegistry:
    """Manages model versioning and lifecycle in MLflow registry.

    This class provides comprehensive model management including registration,
    validation, promotion between stages, and automated deployment workflows.

    Attributes:
        client: MLflow tracking client for registry operations
        validation_threshold: Minimum accuracy for production promotion

    Example:
        ```python
        registry = MLModelRegistry()
        model_version = registry.register_model(run_id, "my_model")
        registry.promote_to_production(model_version, validation_results)
        ```
    """

    def __init__(self, tracking_uri: str, validation_threshold: float = 0.9):
        """Initializes the model registry manager.

        Args:
            tracking_uri: MLflow tracking server URI
            validation_threshold: Minimum accuracy required for production

        Raises:
            ConnectionError: If cannot connect to MLflow server
        """
        pass
```

### Testing Standards

#### Test Structure Requirements
```python
# tests/unit/test_data_processing.py
"""Unit tests for data processing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.processing import DataProcessor, ProcessingConfig


class TestDataProcessor:
    """Test suite for DataProcessor class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Provides sample dataset for testing."""
        return pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })

    @pytest.fixture
    def processing_config(self) -> ProcessingConfig:
        """Provides standard processing configuration."""
        return ProcessingConfig(
            normalize_features=True,
            handle_missing='drop',
            feature_selection_method='variance'
        )

    def test_data_loading_success(self, tmp_path: Path, sample_data: pd.DataFrame):
        """Tests successful data loading from CSV file.

        Arrange:
            - Create temporary CSV file with sample data
            - Initialize DataProcessor

        Act:
            - Load data using processor

        Assert:
            - Data is loaded correctly
            - Shape matches expected dimensions
        """
        # Arrange
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        processor = DataProcessor()

        # Act
        loaded_data = processor.load_data(csv_path)

        # Assert
        assert loaded_data.shape == sample_data.shape
        pd.testing.assert_frame_equal(loaded_data, sample_data)

    def test_data_loading_file_not_found(self):
        """Tests error handling for missing data file."""
        processor = DataProcessor()

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            processor.load_data(Path("nonexistent.csv"))

    @patch('src.data.processing.pd.read_csv')
    def test_data_loading_with_mock(self, mock_read_csv: Mock, sample_data: pd.DataFrame):
        """Tests data loading with mocked pandas read_csv."""
        # Arrange
        mock_read_csv.return_value = sample_data
        processor = DataProcessor()

        # Act
        result = processor.load_data(Path("mock_data.csv"))

        # Assert
        mock_read_csv.assert_called_once_with(Path("mock_data.csv"))
        pd.testing.assert_frame_equal(result, sample_data)
```

#### Integration Test Patterns
```python
# tests/integration/test_training_pipeline.py
"""Integration tests for complete training pipeline."""

import pytest
import mlflow
from pathlib import Path
import tempfile
import shutil

from src.training.pipeline import TrainingPipeline
from src.data.dataset import ImageDataset
from src.models.classifier import ImageClassifier


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for end-to-end training workflow."""

    @pytest.fixture(scope="class")
    def temp_workspace(self) -> Path:
        """Creates temporary workspace for integration tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture(scope="class")
    def mlflow_experiment(self):
        """Sets up MLflow experiment for testing."""
        experiment_name = "test_integration_experiment"
        mlflow.set_experiment(experiment_name)
        yield experiment_name
        # Cleanup handled by MLflow server reset

    def test_complete_training_workflow(self, temp_workspace: Path, mlflow_experiment: str):
        """Tests complete training workflow from data to model registry.

        This integration test verifies:
        - Data loading and preprocessing
        - Model training with MLflow logging
        - Model evaluation and metrics
        - Model registration in MLflow
        """
        # Arrange
        config = self._create_test_config(temp_workspace)
        pipeline = TrainingPipeline(config)

        # Create minimal test dataset
        self._create_test_dataset(temp_workspace)

        # Act
        with mlflow.start_run():
            results = pipeline.run()

        # Assert
        assert results['final_accuracy'] > 0.5  # Reasonable threshold for test data
        assert results['model_path'].exists()
        assert results['mlflow_run_id'] is not None

        # Verify MLflow artifacts
        run = mlflow.get_run(results['mlflow_run_id'])
        assert 'model' in run.data.tags
        assert run.data.metrics['val_accuracy'] > 0
```

### Git Workflow Standards

#### Commit Message Format
```
<type>(<scope>): <description>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(training): add distributed training with Ray

- Implement Ray Tune integration for hyperparameter optimization
- Add MLflow logging for distributed training metrics
- Support multi-GPU training across nodes

Closes #123
```

#### Branch Naming Convention
- `feature/description-of-feature`
- `bugfix/description-of-fix`
- `hotfix/critical-issue-fix`
- `experiment/ml-experiment-name`
- `data/dataset-version-update`

### Performance Standards

#### Memory Management
```python
# Efficient data loading for large datasets
def load_large_dataset_efficiently(data_path: Path, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """Loads large datasets in chunks to manage memory usage."""
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        yield chunk

# Proper cleanup of GPU memory
def cleanup_gpu_memory():
    """Cleans up GPU memory after training/inference."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

#### Optimization Guidelines
```python
# Use generators for data processing pipelines
def process_images_efficiently(image_paths: List[Path]) -> Iterator[torch.Tensor]:
    """Processes images efficiently using generators."""
    for path in image_paths:
        image = load_and_preprocess_image(path)
        yield image

# Vectorized operations over loops
def calculate_metrics_vectorized(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculates metrics using vectorized operations."""
    accuracy = np.mean(predictions == targets)
    precision = np.sum((predictions == 1) & (targets == 1)) / np.sum(predictions == 1)
    recall = np.sum((predictions == 1) & (targets == 1)) / np.sum(targets == 1)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall)
    }
```

### Security Standards

#### Secrets Management
```python
# Never hardcode credentials
import os
from pathlib import Path

def get_mlflow_credentials() -> tuple[str, str]:
    """Retrieves MLflow credentials from environment or secrets file."""
    # Priority: Environment variables -> Secrets file -> Error
    username = os.getenv('MLFLOW_USERNAME')
    password = os.getenv('MLFLOW_PASSWORD')

    if not username or not password:
        secrets_file = Path('.secrets/mlflow_credentials.json')
        if secrets_file.exists():
            import json
            with secrets_file.open() as f:
                creds = json.load(f)
                username = creds.get('username')
                password = creds.get('password')

    if not username or not password:
        raise ValueError("MLflow credentials not found in environment or secrets file")

    return username, password
```

#### Input Validation
```python
def validate_model_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and sanitizes model input data."""
    # Type validation
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a dictionary")

    # Required fields validation
    required_fields = ['image', 'metadata']
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")

    # Data sanitization
    sanitized_data = {}

    # Image validation
    image_data = input_data['image']
    if not isinstance(image_data, (list, np.ndarray)):
        raise ValueError("Image data must be list or numpy array")

    sanitized_data['image'] = np.array(image_data, dtype=np.float32)

    # Metadata validation
    metadata = input_data['metadata']
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")

    sanitized_data['metadata'] = {
        k: str(v) for k, v in metadata.items()
        if isinstance(k, str) and len(str(v)) < 1000
    }

    return sanitized_data
```

These standards ensure consistent, maintainable, and secure code throughout the ML lifecycle project.
