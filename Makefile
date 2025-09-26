.PHONY: install test build clean run-train run-eval lint format type-check

# =============================== Development ================================= #
install:
	@echo "Installing dependencies..."
	uv pip install -e ".[dev]"

# =============================== Testing ===================================== #
test:
	@echo "Running tests..."
	pytest

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term-missing

# =============================== Code Quality =============================== #
lint:
	@echo "Running linter..."
	uv lint

format:
	@echo "Formatting code..."
	uv format

type-check:
	@echo "Running type checker..."
	mypy src/

# =============================== ML Workflow ================================ #
run-train:
	@echo "Running training pipeline..."
	python main.py mode=train

run-eval:
	@echo "Running evaluation pipeline..."
	python main.py mode=eval

run-full:
	@echo "Running full pipeline (train + eval)..."
	python main.py mode=full

run-example:
	@echo "Running example with default config..."
	python main.py

# =============================== Cleanup ==================================== #
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

# =============================== Docker ==================================== #
docker-build:
	@echo "Building Docker images..."
	docker build -t e2e-ml-pipeline:latest .
	docker build -f Dockerfile.mlflow -t e2e-ml-mlflow:latest .

docker-run:
	@echo "Running ML pipeline with Docker Compose..."
	docker compose up --build

docker-run-mlflow:
	@echo "Running only MLflow UI..."
	docker compose up mlflow-ui

docker-stop:
	@echo "Stopping Docker services..."
	docker compose down
	@echo "Killing any processes using port 5000..."
	@lsof -ti:5000 | xargs -r kill -9 2>/dev/null || true

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker compose down
	docker rmi e2e-ml-pipeline:latest e2e-ml-mlflow:latest 2>/dev/null || true
	docker volume prune -f
	docker network prune -f

docker-logs:
	@echo "Showing Docker logs..."
	docker compose logs -f

docker-shell:
	@echo "Accessing ML pipeline container shell..."
	docker exec -it e2e-ml-pipeline bash

docker-status:
	@echo "Docker service status:"
	docker compose ps

# =============================== Help ======================================= #
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage report"
	@echo "  lint        - Run linter"
	@echo "  format      - Format code"
	@echo "  type-check  - Run type checker"
	@echo "  run-train   - Run training pipeline"
	@echo "  run-eval    - Run evaluation pipeline"
	@echo "  run-full    - Run full pipeline (train + eval)"
	@echo "  run-example - Run example with default config"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-run  - Run ML pipeline with Docker Compose"
	@echo "  docker-run-mlflow - Run only MLflow UI"
	@echo "  docker-stop - Stop Docker services"
	@echo "  docker-clean - Clean up Docker resources"
	@echo "  docker-logs - Show Docker logs"
	@echo "  docker-shell - Access container shell"
	@echo "  docker-status - Show Docker service status"
	@echo "  clean       - Clean up temporary files"
	@echo "  help        - Show this help message"
