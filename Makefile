PYTHON := python
PACKAGE := e2e_ml

.PHONY: install format lint test build clean dagster-build dagster-up dagster-down

install:
	uv pip install -e ".[dev]"

format:
	uv format

lint:
	uv lint

test:
	pytest -q --maxfail=1 --disable-warnings

build:
	docker build -t e2e-ml:latest .

dagster-build:
	docker build -f Dockerfile.dagster -t e2e-ml-dagster:latest .

dagster-up:
	docker run --rm -it -p 3000:3000 --name e2e-ml-dagster e2e-ml-dagster:latest

dagster-down:
	-docker stop e2e-ml-dagster

clean:
	rm -rf .pytest_cache __pycache__ build dist *.egg-info logs
