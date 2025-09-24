# syntax=docker/dockerfile:1.7-labs

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system -e ".[dev]"

# --- Runtime stage ---
FROM python:3.11-slim AS runtime
ENV PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1

RUN useradd -m -u 10001 appuser
WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local /usr/local

# Copy minimal runtime files
COPY src ./src
COPY pyproject.toml ./

USER appuser

ENTRYPOINT ["python", "-m", "src.e2e_ml.cli"]
