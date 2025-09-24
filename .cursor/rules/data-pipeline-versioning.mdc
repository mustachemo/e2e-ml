# Data Pipeline & Versioning Rules

## Data Version Control (DVC) Integration

All data management must follow DVC best practices for reproducibility, versioning, and pipeline orchestration. This ensures data lineage tracking and enables collaborative data science workflows.

### Repository Structure

```
data/
├── raw/                    # Original, immutable source data
│   ├── images/
│   ├── labels/
│   └── metadata/
├── interim/                # Intermediate processed data
│   ├── resized/
│   ├── augmented/
│   └── features/
├── processed/              # Final, analysis-ready data
│   ├── train/
│   ├── validation/
│   └── test/
└── external/               # External reference data
    ├── pretrained_models/
    └── benchmarks/
```

### DVC Configuration Standards

#### Remote Storage Configuration
```yaml
# .dvc/config
[core]
    remote = minio
    autostage = true
    analytics = false

[remote "minio"]
    url = s3://ml-data-bucket
    endpointurl = http://minio:9000
    access_key_id = minioadmin
    secret_access_key = minioadmin
    ssl_verify = false
```

#### Data Pipeline Definition
- **Pipeline stages**: Each data transformation must be a separate DVC stage
- **Dependency tracking**: All input files, scripts, and parameters must be declared
- **Output versioning**: All outputs must be tracked by DVC
- **Reproducibility**: Pipelines must be deterministic and reproducible

### Data Processing Pipeline Architecture

#### Stage 1: Data Ingestion
```python
# scripts/ingest_data.py
"""Data ingestion from various sources with validation and metadata extraction."""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from loguru import logger

class DataIngestionPipeline:
    """Handles raw data ingestion with validation and metadata extraction."""

    def __init__(self, source_path: Path, target_path: Path):
        self.source_path = source_path
        self.target_path = target_path

    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Validates basic data quality metrics."""
        validations = {
            "no_missing_required_fields": not data[['image_path', 'label']].isnull().any().any(),
            "valid_file_paths": all(Path(p).suffix.lower() in ['.jpg', '.png', '.jpeg']
                                  for p in data['image_path']),
            "label_consistency": data['label'].dtype in ['object', 'category'],
            "sufficient_samples": len(data) >= 100
        }

        failed_checks = [check for check, passed in validations.items() if not passed]
        if failed_checks:
            logger.warning(f"Data quality issues detected: {failed_checks}")

        return validations
```

#### Stage 2: Data Preprocessing
```python
# scripts/preprocess_data.py
"""Data preprocessing with augmentation and feature extraction."""

import torch
from torchvision import transforms
from PIL import Image
import albumentations as A
from typing import Dict, Any

class ImagePreprocessor:
    """Handles image preprocessing with configurable transformations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> Dict[str, Any]:
        """Builds transformation pipelines for different data splits."""
        base_transforms = A.Compose([
            A.Resize(self.config['image_size'], self.config['image_size']),
            A.Normalize(mean=self.config['normalize_mean'],
                       std=self.config['normalize_std']),
        ])

        train_transforms = A.Compose([
            A.Resize(self.config['image_size'], self.config['image_size']),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=self.config['normalize_mean'],
                       std=self.config['normalize_std']),
        ])

        return {
            'train': train_transforms,
            'val': base_transforms,
            'test': base_transforms
        }
```

#### Stage 3: Data Splitting
```python
# scripts/split_data.py
"""Stratified data splitting with cross-validation support."""

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from typing import Tuple, List

class DataSplitter:
    """Handles stratified data splitting with reproducible random seeds."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def create_stratified_splits(self,
                               data: pd.DataFrame,
                               target_column: str,
                               test_size: float = 0.2,
                               val_size: float = 0.1) -> Tuple[pd.DataFrame, ...]:
        """Creates stratified train/val/test splits maintaining class distribution."""

        # First split: separate test set
        train_val, test = train_test_split(
            data,
            test_size=test_size,
            stratify=data[target_column],
            random_state=self.random_state
        )

        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val[target_column],
            random_state=self.random_state
        )

        return train, val, test
```

### DVC Pipeline Configuration

#### dvc.yaml Structure
```yaml
stages:
  data_ingestion:
    cmd: python scripts/ingest_data.py
    deps:
    - scripts/ingest_data.py
    - configs/data_config.yaml
    params:
    - data_ingestion.source_path
    - data_ingestion.batch_size
    outs:
    - data/raw/

  data_preprocessing:
    cmd: python scripts/preprocess_data.py
    deps:
    - scripts/preprocess_data.py
    - data/raw/
    - configs/preprocessing_config.yaml
    params:
    - preprocessing.image_size
    - preprocessing.augmentation_config
    outs:
    - data/interim/

  data_splitting:
    cmd: python scripts/split_data.py
    deps:
    - scripts/split_data.py
    - data/interim/
    params:
    - splitting.test_size
    - splitting.val_size
    - splitting.random_state
    outs:
    - data/processed/train/
    - data/processed/val/
    - data/processed/test/
    metrics:
    - metrics/data_splits.json

  feature_extraction:
    cmd: python scripts/extract_features.py
    deps:
    - scripts/extract_features.py
    - data/processed/
    - models/feature_extractor.pkl
    params:
    - feature_extraction.method
    - feature_extraction.dimensions
    outs:
    - features/train_features.h5
    - features/val_features.h5
    - features/test_features.h5
```

### Data Validation Framework

#### Data Quality Checks
```python
# src/data_validation.py
"""Comprehensive data validation framework for ML pipelines."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

@dataclass
class ValidationResult:
    """Results from data validation checks."""
    check_name: str
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    details: Optional[Dict[str, Any]] = None

class DataValidator:
    """Comprehensive data validation for ML pipelines."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[ValidationResult] = []

    def validate_dataset(self, data: pd.DataFrame, split_name: str) -> List[ValidationResult]:
        """Runs all validation checks on a dataset split."""
        self.results = []

        # Basic structure validation
        self._check_required_columns(data)
        self._check_data_types(data)
        self._check_missing_values(data)

        # Distribution validation
        self._check_class_distribution(data, split_name)
        self._check_feature_distributions(data)

        # Image-specific validation
        if 'image_path' in data.columns:
            self._check_image_files(data)
            self._check_image_properties(data)

        return self.results

    def _check_class_distribution(self, data: pd.DataFrame, split_name: str) -> None:
        """Validates class distribution meets requirements."""
        if 'label' not in data.columns:
            return

        class_counts = data['label'].value_counts()
        min_samples_per_class = self.config.get('min_samples_per_class', 10)

        insufficient_classes = class_counts[class_counts < min_samples_per_class]

        if len(insufficient_classes) > 0:
            self.results.append(ValidationResult(
                check_name="class_distribution",
                passed=False,
                message=f"Classes with insufficient samples in {split_name}: {insufficient_classes.to_dict()}",
                severity="warning",
                details={"insufficient_classes": insufficient_classes.to_dict()}
            ))
```

### Data Lineage Tracking

#### Metadata Management
```python
# src/data_lineage.py
"""Data lineage tracking for ML pipeline transparency."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

@dataclass
class DataLineageRecord:
    """Records data transformation lineage."""
    stage_name: str
    input_paths: List[str]
    output_paths: List[str]
    script_path: str
    parameters: Dict[str, Any]
    timestamp: datetime
    git_commit: Optional[str] = None
    data_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts record to dictionary for JSON serialization."""
        record_dict = asdict(self)
        record_dict['timestamp'] = self.timestamp.isoformat()
        return record_dict

class DataLineageTracker:
    """Tracks data transformations throughout the ML pipeline."""

    def __init__(self, lineage_file: Path):
        self.lineage_file = lineage_file
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)

    def record_transformation(self, record: DataLineageRecord) -> None:
        """Records a data transformation step."""
        lineage_data = self._load_existing_lineage()
        lineage_data.append(record.to_dict())

        with self.lineage_file.open('w') as f:
            json.dump(lineage_data, f, indent=2)

    def get_data_lineage(self, output_path: str) -> List[Dict[str, Any]]:
        """Retrieves lineage for a specific output path."""
        lineage_data = self._load_existing_lineage()
        return [
            record for record in lineage_data
            if output_path in record.get('output_paths', [])
        ]
```

### Pipeline Monitoring

#### Data Drift Detection
```python
# src/data_monitoring.py
"""Data drift detection and monitoring for production ML systems."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Any
from dataclasses import dataclass

@dataclass
class DriftDetectionResult:
    """Results from data drift detection."""
    feature_name: str
    drift_detected: bool
    drift_score: float
    p_value: float
    method: str
    threshold: float

class DataDriftDetector:
    """Detects statistical drift between reference and current datasets."""

    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        self.reference_data = reference_data
        self.significance_level = significance_level

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, DriftDetectionResult]:
        """Detects drift across all comparable features."""
        results = {}

        for column in self.reference_data.columns:
            if column in current_data.columns:
                if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                    result = self._kolmogorov_smirnov_test(column, current_data)
                else:
                    result = self._chi_square_test(column, current_data)

                results[column] = result

        return results

    def _kolmogorov_smirnov_test(self, column: str, current_data: pd.DataFrame) -> DriftDetectionResult:
        """Performs Kolmogorov-Smirnov test for numerical features."""
        ref_values = self.reference_data[column].dropna()
        cur_values = current_data[column].dropna()

        statistic, p_value = stats.ks_2samp(ref_values, cur_values)

        return DriftDetectionResult(
            feature_name=column,
            drift_detected=p_value < self.significance_level,
            drift_score=statistic,
            p_value=p_value,
            method="kolmogorov_smirnov",
            threshold=self.significance_level
        )
```

### Configuration Management

#### Data Configuration Schema
```yaml
# configs/data_config.yaml
data_ingestion:
  source_path: "data/external/"
  target_path: "data/raw/"
  batch_size: 1000
  validation_rules:
    required_columns: ["image_path", "label"]
    max_missing_ratio: 0.05

preprocessing:
  image_size: 224
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  augmentation_config:
    horizontal_flip: 0.5
    rotation_limit: 15
    brightness_contrast: 0.3

splitting:
  test_size: 0.2
  val_size: 0.1
  random_state: 42
  stratify: true

feature_extraction:
  method: "resnet50"
  dimensions: 2048
  batch_size: 32
  use_pretrained: true

data_validation:
  min_samples_per_class: 10
  max_class_imbalance_ratio: 10.0
  required_image_formats: [".jpg", ".png", ".jpeg"]
  min_image_size: [64, 64]
  max_image_size: [4096, 4096]

monitoring:
  drift_detection:
    significance_level: 0.05
    reference_window_days: 30
    monitoring_frequency_hours: 24
```

This data pipeline framework ensures reproducible, versioned, and monitored data workflows essential for production ML systems.
