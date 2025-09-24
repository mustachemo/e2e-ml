# Infrastructure & Containerization Rules

## Docker Compose Architecture

All services must be defined in a comprehensive `docker-compose.yml` that enables the complete ML lifecycle simulation. The compose file should support both development and production-like configurations.

### Network Configuration

```yaml
# Required networks for service isolation and communication
networks:
  ml-backend:
    driver: bridge
    name: ml-backend
  ml-frontend:
    driver: bridge
    name: ml-frontend
  monitoring:
    driver: bridge
    name: monitoring
```

### Service Categories

#### 1. Storage & Database Services

**MinIO (S3-Compatible Object Storage)**
- **Image**: `minio/minio:latest`
- **Purpose**: Central artifact store for datasets, models, and experiment outputs
- **Configuration Requirements**:
  - Persistent volumes for data and configuration
  - Environment variables for access keys and console access
  - Health checks for service readiness
  - Exposed ports: 9000 (API), 9001 (Console)

**PostgreSQL (MLflow Backend Store)**
- **Image**: `postgres:15-alpine`
- **Purpose**: Metadata storage for MLflow experiments and model registry
- **Configuration Requirements**:
  - Persistent volume for database data
  - Environment variables for database credentials
  - Health checks using pg_isready
  - Custom initialization scripts for MLflow schema

#### 2. ML Platform Services

**MLflow Tracking Server**
- **Base Image**: `python:3.11-slim`
- **Purpose**: Centralized experiment tracking and model registry
- **Build Requirements**:
  - Install MLflow with PostgreSQL and S3 support
  - Configure backend store and artifact store URLs
  - Expose port 5000 for web UI and API
  - Health checks for service availability

**Ray Head Node**
- **Image**: `rayproject/ray-ml:2.8.0-py311`
- **Purpose**: Distributed computing coordinator for training and serving
- **Configuration Requirements**:
  - GPU support configuration (runtime: nvidia)
  - Shared memory configuration for efficient data sharing
  - Dashboard port exposure (8265)
  - Worker node discovery configuration

**Ray Worker Nodes**
- **Image**: `rayproject/ray-ml:2.8.0-py311`
- **Purpose**: Distributed computing workers for parallel processing
- **Configuration Requirements**:
  - Dynamic scaling capabilities (compose profiles)
  - GPU resource allocation per worker
  - Connection to Ray head node
  - Resource limits based on available hardware

#### 3. Monitoring & Observability

**Prometheus**
- **Image**: `prom/prometheus:latest`
- **Purpose**: Metrics collection and time-series storage
- **Configuration Requirements**:
  - Custom prometheus.yml configuration
  - Service discovery for all ML services
  - Persistent storage for metrics data
  - Alert manager integration

**Grafana**
- **Image**: `grafana/grafana:latest`
- **Purpose**: Visualization dashboards for system and ML metrics
- **Configuration Requirements**:
  - Persistent storage for dashboard configurations
  - Pre-configured data sources (Prometheus)
  - Custom dashboards for ML lifecycle monitoring
  - Environment variables for admin credentials

**Jaeger**
- **Image**: `jaegertracing/all-in-one:latest`
- **Purpose**: Distributed tracing for request flow analysis
- **Configuration Requirements**:
  - In-memory storage for development
  - Trace collection endpoints
  - Web UI exposure for trace visualization

#### 4. Infrastructure Services

**NGINX (Load Balancer & API Gateway)**
- **Image**: `nginx:alpine`
- **Purpose**: Request routing and load balancing for inference services
- **Configuration Requirements**:
  - Custom nginx.conf for Ray Serve routing
  - Health check endpoints
  - SSL termination capabilities
  - Rate limiting configuration

### Development Environment Container

**Custom ML Development Image**
- **Base Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
- **Purpose**: Unified development environment with all ML tools
- **Required Packages**:
  - Core ML: `torch`, `torchvision`, `transformers`, `scikit-learn`
  - Data Processing: `pandas`, `numpy`, `pillow`, `opencv-python`
  - ML Platform: `mlflow`, `ray[default]`, `dvc[s3]`
  - Development: `jupyter`, `ipykernel`, `rich`, `typer`
  - Testing: `pytest`, `pytest-cov`, `pytest-mock`
  - Tooling: `uv`, `pre-commit`

### Container Build Standards

#### Multi-Stage Builds
All custom containers must use multi-stage builds for optimization:
```dockerfile
# Builder stage
FROM python:3.11-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim as runtime
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
WORKDIR /app
COPY . .
CMD ["python", "app.py"]
```

#### Security Standards
- **Non-root user**: All containers must run as non-root users
- **Minimal base images**: Use alpine or slim variants when possible
- **Security scanning**: Implement container vulnerability scanning
- **Secrets management**: Use Docker secrets or environment files for sensitive data

#### Resource Management
- **Memory limits**: Define appropriate memory limits for each service
- **CPU limits**: Set CPU limits to prevent resource contention
- **GPU allocation**: Proper GPU resource sharing for ML workloads
- **Health checks**: All services must implement health check endpoints

### Environment Configuration

#### Development Profile
```yaml
# docker-compose.override.yml for development
services:
  ray-worker:
    scale: 1  # Single worker for development
  mlflow:
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
    ports:
      - "5000:5000"  # Expose for direct access
```

#### Production Profile
```yaml
# docker-compose.prod.yml for production simulation
services:
  ray-worker:
    scale: 3  # Multiple workers for production
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Volume Management

#### Persistent Volumes
- **Database data**: PostgreSQL data persistence
- **Object storage**: MinIO data and configuration persistence
- **Model artifacts**: Shared volume for model storage
- **Logs**: Centralized log storage across services

#### Shared Volumes
- **Dataset cache**: Shared dataset storage for training processes
- **Model registry**: Shared model artifact storage
- **Configuration**: Shared configuration files across services

### Service Discovery & Communication

#### Internal DNS
- Use Docker's built-in DNS for service-to-service communication
- Consistent service naming across environments
- Environment-specific configuration through DNS

#### Port Management
- **Internal ports**: Use standard ports for internal communication
- **External ports**: Expose only necessary ports for development
- **Port conflicts**: Avoid conflicts with commonly used ports

### Deployment Commands

#### Development Setup
```bash
# Full development stack
docker-compose up -d

# Individual service development
docker-compose up -d postgres minio mlflow

# Scale Ray workers
docker-compose up -d --scale ray-worker=3
```

#### Production Simulation
```bash
# Production-like deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Health check validation
docker-compose ps --format="table {{.Name}}\t{{.Status}}\t{{.Ports}}"
```

### Monitoring & Logging

#### Container Metrics
- All containers must expose metrics endpoints
- CPU, memory, and disk usage monitoring
- Custom application metrics for ML workloads

#### Log Aggregation
- Structured logging using JSON format
- Centralized log collection via Docker logging drivers
- Log rotation and retention policies

#### Health Monitoring
- Comprehensive health checks for all services
- Dependency health validation
- Automatic restart policies for failed services

This infrastructure setup provides a realistic simulation of cloud-native ML operations while maintaining development efficiency and operational simplicity.
