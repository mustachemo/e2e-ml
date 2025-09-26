# ==============================================================================
# Multi-stage Dockerfile for E2E ML Pipeline
# ==============================================================================

# =============================== Base Stage ================================= #
FROM python:3.11-slim as base

# * Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# * Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# * Install uv
RUN pip install uv

# =============================== Dependencies Stage ========================= #
FROM base as deps

# * Set working directory
WORKDIR /app

# * Copy dependency files
COPY pyproject.toml ./

# * Install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# =============================== Production Stage =========================== #
FROM base as production

# * Set working directory
WORKDIR /app

# * Copy virtual environment from deps stage
COPY --from=deps /app/.venv /app/.venv

# * Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# * Copy application code
COPY . .

# * Create output directory
RUN mkdir -p /app/output

# * Set default command
CMD ["python", "main.py"]
