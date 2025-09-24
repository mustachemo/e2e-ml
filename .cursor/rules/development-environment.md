# Development Environment & Tooling Standards

## Unified Development Container

All development must occur within a standardized container environment that includes all necessary ML tools, ensuring consistency across different development machines and seamless integration with the containerized infrastructure.

### Development Container Specification

#### Base Dockerfile
```dockerfile
# Dockerfile.dev
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv sync --frozen --no-dev

# Install development dependencies
RUN uv add --dev \
    jupyter \
    jupyterlab \
    ipykernel \
    notebook \
    pytest \
    pytest-cov \
    pytest-mock \
    pre-commit \
    black \
    isort \
    flake8 \
    mypy \
    rich \
    typer

# Install ML platform tools
RUN uv add \
    mlflow \
    "ray[default,serve,tune]" \
    "dvc[s3]" \
    torch \
    torchvision \
    transformers \
    scikit-learn \
    pandas \
    numpy \
    pillow \
    opencv-python-headless \
    albumentations \
    optuna \
    hydra-core \
    omegaconf \
    loguru \
    prometheus-client \
    opentracing \
    jaeger-client

# Setup Jupyter kernel
RUN python -m ipykernel install --user --name ml-env --display-name "ML Environment"

# Configure Git (will be overridden by volume mounts in development)
RUN git config --global user.name "ML Developer" && \
    git config --global user.email "dev@ml-platform.local"

# Setup pre-commit hooks
COPY .pre-commit-config.yaml .
RUN pre-commit install

# Expose ports for development services
EXPOSE 8888 8000 8265 6006

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
```

#### Development Docker Compose Override
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  ml-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    container_name: ml-dev-environment
    volumes:
      - .:/workspace
      - ~/.ssh:/root/.ssh:ro
      - ~/.gitconfig:/root/.gitconfig:ro
      - ml-dev-cache:/root/.cache
      - ml-dev-conda:/opt/conda/pkgs
    ports:
      - "8888:8888"  # Jupyter Lab
      - "8000:8000"  # Ray Dashboard
      - "8265:8265"  # Ray Serve
      - "6006:6006"  # TensorBoard
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - RAY_HEAD_SERVICE_HOST=ray-head
      - DVC_CACHE_DIR=/workspace/.dvc/cache
    networks:
      - ml-backend
      - ml-frontend
    depends_on:
      - mlflow
      - minio
      - postgres
    runtime: nvidia  # Enable GPU access
    shm_size: 2gb   # Increase shared memory for data loading

volumes:
  ml-dev-cache:
  ml-dev-conda:
```

### Project Structure Standards

#### Recommended Directory Layout
```
ml-lifecycle-project/
├── .cursor/                    # Cursor IDE configuration
│   └── rules/                  # Development rules and guidelines
├── .dvc/                       # DVC configuration and cache
├── .github/                    # GitHub workflows and templates
│   └── workflows/
├── configs/                    # Hydra configuration files
│   ├── data/
│   ├── model/
│   ├── training/
│   └── config.yaml
├── data/                       # Data storage (DVC tracked)
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/                       # Documentation
│   ├── api/
│   ├── architecture/
│   └── guides/
├── models/                     # Saved models (DVC tracked)
├── notebooks/                  # Jupyter notebooks
│   ├── exploratory/
│   ├── experiments/
│   └── analysis/
├── scripts/                    # Automation scripts
│   ├── data/
│   ├── training/
│   └── deployment/
├── src/                        # Source code
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── serving/
│   └── monitoring/
├── tests/                      # Test files
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .env.example               # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── docker-compose.dev.yml
├── Dockerfile.dev
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

### Development Workflow Configuration

#### Pre-commit Hooks Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.3
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
        args: [--strict, --ignore-missing-imports]

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/unit/, -v, --tb=short]

      - id: dvc-status
        name: dvc-status
        entry: dvc status
        language: system
        pass_filenames: false
        always_run: true
```

#### Makefile for Development Automation
```makefile
# Makefile
.PHONY: help install dev-setup test lint format clean build deploy

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies with uv"
	@echo "  dev-setup   - Setup development environment"
	@echo "  test        - Run all tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean temporary files"
	@echo "  build       - Build Docker containers"
	@echo "  deploy      - Deploy services"

# Installation and setup
install:
	uv sync

dev-setup: install
	pre-commit install
	dvc pull
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

# Code quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Development services
dev-up:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

dev-down:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

dev-logs:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Data operations
data-pull:
	dvc pull

data-push:
	dvc push

data-status:
	dvc status

# ML operations
train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

serve:
	python scripts/serve_model.py

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/

# Docker operations
build:
	docker-compose build

deploy-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

deploy-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Monitoring
logs:
	docker-compose logs -f

status:
	docker-compose ps

metrics:
	curl -s http://localhost:9090/api/v1/query?query=up | jq '.'
```

### IDE Configuration

#### VS Code / Cursor Settings
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "/opt/conda/bin/python",
    "python.formatting.provider": "none",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        },
        "editor.rulers": [88]
    },
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "files.watcherExclude": {
        "**/.dvc/**": true,
        "**/data/**": true,
        "**/models/**": true
    },
    "git.ignoredRepositories": [
        ".dvc"
    ]
}
```

#### VS Code Extensions Recommendations
```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "charliermarsh.ruff",
        "ms-vscode.vscode-json",
        "ms-azuretools.vscode-docker",
        "iterative.dvc",
        "ms-python.mypy-type-checker",
        "github.copilot",
        "github.copilot-chat",
        "ms-vscode-remote.remote-containers"
    ]
}
```

### Environment Management

#### Environment Variables Configuration
```bash
# .env.example
# Copy to .env and update values for local development

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# Database Configuration
POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=localhost:9000

# Ray Configuration
RAY_HEAD_SERVICE_HOST=localhost
RAY_DASHBOARD_HOST=0.0.0.0

# Development Configuration
PYTHONPATH=/workspace/src
JUPYTER_TOKEN=
JUPYTER_LAB_PORT=8888

# DVC Configuration
DVC_CACHE_DIR=/workspace/.dvc/cache

# Monitoring Configuration
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

#### Development Scripts
```python
# scripts/dev_setup.py
"""Development environment setup script."""

import subprocess
import sys
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd: List[str], cwd: Path = None) -> bool:
    """Runs a shell command and returns success status."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {' '.join(cmd)}: {e.stderr}")
        return False

def setup_development_environment():
    """Sets up the complete development environment."""
    workspace = Path.cwd()

    steps = [
        # Copy environment file
        (["cp", ".env.example", ".env"], "Environment configuration"),

        # Install dependencies
        (["uv", "sync"], "Python dependencies"),

        # Setup pre-commit hooks
        (["pre-commit", "install"], "Pre-commit hooks"),

        # Initialize DVC
        (["dvc", "init", "--no-scm"], "DVC initialization"),

        # Build development containers
        (["docker-compose", "-f", "docker-compose.yml", "-f", "docker-compose.dev.yml", "build"], "Docker containers"),

        # Start development services
        (["docker-compose", "-f", "docker-compose.yml", "-f", "docker-compose.dev.yml", "up", "-d"], "Development services"),

        # Wait for services to be ready
        (["sleep", "30"], "Service initialization"),

        # Run initial tests
        (["pytest", "tests/unit/", "-v"], "Unit tests"),
    ]

    for cmd, description in steps:
        logger.info(f"Setting up: {description}")
        if not run_command(cmd, workspace):
            logger.error(f"Failed to setup: {description}")
            sys.exit(1)

    logger.info("🎉 Development environment setup completed!")
    logger.info("Access Jupyter Lab at: http://localhost:8888")
    logger.info("Access MLflow UI at: http://localhost:5000")
    logger.info("Access Ray Dashboard at: http://localhost:8265")

if __name__ == "__main__":
    setup_development_environment()
```

### Development Utilities

#### Jupyter Notebook Templates
```python
# templates/experiment_notebook_template.py
"""Template for ML experiment notebooks."""

# ================================== Setup ================================== #
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path.cwd().parent / "src"))

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# ML imports
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# MLflow tracking
import mlflow
import mlflow.pytorch
mlflow.set_tracking_uri("http://mlflow:5000")

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ================================== Configuration ================================== #
EXPERIMENT_NAME = "experiment_template"
MODEL_NAME = "baseline_model"
DATA_VERSION = "v1.0"

# Start MLflow run
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"{MODEL_NAME}_{DATA_VERSION}"):
    mlflow.log_param("data_version", DATA_VERSION)
    mlflow.log_param("model_type", MODEL_NAME)

    # ================================== Data Loading ================================== #
    logger.info("Loading data...")

    # ================================== Exploratory Analysis ================================== #
    logger.info("Performing exploratory analysis...")

    # ================================== Model Training ================================== #
    logger.info("Training model...")

    # ================================== Evaluation ================================== #
    logger.info("Evaluating model...")

    # ================================== Results & Artifacts ================================== #
    logger.info("Saving results...")
```

This development environment ensures consistent, reproducible, and efficient ML development workflows with comprehensive tooling integration.
