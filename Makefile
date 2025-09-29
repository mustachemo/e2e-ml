.PHONY: install test build clean run-train run-eval lint format type-check
# =============================== Docker ===================================== #
docker-build:
	docker build -t e2e-ml-pipeline:latest -f docker/Dockerfile .
	docker build -f docker/Dockerfile.mlflow -t e2e-ml-mlflow:latest .

docker-run:
	docker compose up --build

docker-run-mlflow:
	docker compose up mlflow-ui

docker-stop:
	docker compose down
	@echo "Killing any processes using port 5000..."
	@lsof -ti:5000 | xargs -r kill -9 2>/dev/null || true

docker-clean:
	docker compose down
	docker rmi e2e-ml-pipeline:latest e2e-ml-mlflow:latest 2>/dev/null || true
	docker volume prune -f
	docker network prune -f

docker-logs:
	docker compose logs -f

docker-shell:
	docker exec -it e2e-ml-pipeline bash

docker-status:
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
