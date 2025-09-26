# 🐳 Docker Setup for E2E ML Pipeline

This document describes how to run the E2E ML Pipeline using Docker and Docker Compose.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git

## 🚀 Quick Start

### 1. Build Docker Images

```bash
# Build all images
./scripts/docker-build.sh

# Or build manually
docker build -t e2e-ml-pipeline:latest .
docker build -f Dockerfile.mlflow -t e2e-ml-mlflow:latest .
```

### 2. Run the Pipeline

```bash
# Run with default settings (1 epoch, batch size 32)
./scripts/docker-run.sh

# Run with custom parameters
./scripts/docker-run.sh --mode train --epochs 5 --batch-size 64

# Run only MLflow UI
./scripts/docker-run.sh --mode mlflow-only
```

### 3. Access Services

- **MLflow UI**: http://localhost:5000
- **Jupyter Lab**: http://localhost:8888 (if enabled)

## 🏗️ Architecture

The Docker setup consists of three main services:

### 1. ML Pipeline (`ml-pipeline`)
- **Image**: `e2e-ml-pipeline:latest`
- **Purpose**: Runs the main ML training/evaluation pipeline
- **Configuration**: Uses `configs/docker.yaml`
- **Dependencies**: Depends on `mlflow-ui` service

### 2. MLflow UI (`mlflow-ui`)
- **Image**: `e2e-ml-mlflow:latest`
- **Purpose**: Provides MLflow tracking UI and model registry
- **Port**: 5000
- **Health Check**: Built-in health monitoring

### 3. Jupyter Lab (`jupyter`)
- **Image**: `e2e-ml-pipeline:latest` (same as ML pipeline)
- **Purpose**: Interactive development and analysis
- **Port**: 8888
- **Dependencies**: Depends on `mlflow-ui` service

## 📁 Volume Mounts

The following directories are mounted as volumes:

- `./output` → `/app/output` (ML pipeline outputs)
- `./data` → `/app/data` (Dataset storage)
- `./configs` → `/app/configs` (Configuration files)
- `./notebooks` → `/app/notebooks` (Jupyter notebooks)

## 🔧 Configuration

### Docker-Specific Configuration

The pipeline uses `configs/docker.yaml` which:
- Sets MLflow tracking URI to `http://mlflow-ui:5000`
- Uses Docker-appropriate paths (`/app/output`, `/app/data`)
- Enables all MLflow logging features

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server URL
- `DOCKER_CONTAINER`: Set to "true" when running in Docker
- `JUPYTER_ENABLE_LAB`: Enable Jupyter Lab interface

## 🐛 Debugging

### View MLflow Logs

```bash
# View MLflow UI logs
docker logs e2e-ml-mlflow-ui

# Follow logs in real-time
docker logs -f e2e-ml-mlflow-ui
```

### View Pipeline Logs

```bash
# View ML pipeline logs
docker logs e2e-ml-pipeline

# Follow logs in real-time
docker logs -f e2e-ml-pipeline
```

### Access Container Shell

```bash
# Access ML pipeline container
docker exec -it e2e-ml-pipeline bash

# Access MLflow container
docker exec -it e2e-ml-mlflow-ui bash
```

### Check MLflow Health

```bash
# Check MLflow UI health
curl http://localhost:5000/health

# Check MLflow API
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

## 🧹 Cleanup

```bash
# Clean up all Docker resources
./scripts/docker-clean.sh

# Or clean up manually
docker-compose down
docker rmi e2e-ml-pipeline:latest e2e-ml-mlflow:latest
docker volume prune -f
docker network prune -f
```

## 🔄 Development Workflow

### 1. Make Code Changes
Edit your Python code in the local directory.

### 2. Rebuild and Run
```bash
# Rebuild and run with changes
docker-compose up --build

# Or use the script
./scripts/docker-run.sh --mode train --epochs 1
```

### 3. View Results
- Check MLflow UI at http://localhost:5000
- View outputs in `./output/` directory

## 📊 Monitoring

### Service Status
```bash
# Check running containers
docker ps

# Check service health
docker-compose ps
```

### Resource Usage
```bash
# Check resource usage
docker stats

# Check specific container
docker stats e2e-ml-pipeline e2e-ml-mlflow-ui
```

## 🚨 Troubleshooting

### Common Issues

1. **MLflow UI not accessible**
   - Check if port 5000 is available
   - Verify MLflow container is running: `docker ps`
   - Check MLflow logs: `docker logs e2e-ml-mlflow-ui`

2. **Pipeline fails to connect to MLflow**
   - Ensure MLflow UI is running first
   - Check network connectivity: `docker network ls`
   - Verify MLflow tracking URI in logs

3. **Permission issues with volumes**
   - Check file permissions in mounted directories
   - Ensure Docker has access to local directories

4. **Out of memory errors**
   - Reduce batch size in configuration
   - Increase Docker memory limits
   - Use smaller model or dataset

### Debug Commands

```bash
# Check Docker network
docker network inspect e2e-ml_ml-network

# Check container environment
docker exec e2e-ml-pipeline env | grep MLFLOW

# Test MLflow connectivity from pipeline container
docker exec e2e-ml-pipeline curl -f http://mlflow-ui:5000/health
```

## 📈 Performance Tips

1. **Use multi-stage builds** for smaller images
2. **Mount volumes** for data persistence
3. **Use health checks** for service monitoring
4. **Set resource limits** for production use
5. **Use .dockerignore** to exclude unnecessary files

## 🔒 Security Considerations

1. **Don't expose unnecessary ports**
2. **Use non-root users** in containers
3. **Scan images** for vulnerabilities
4. **Use secrets management** for sensitive data
5. **Regularly update** base images
