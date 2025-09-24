# Model Serving, Monitoring & Observability Rules

## Ray Serve Model Deployment

All model serving must use Ray Serve for scalable, production-ready inference with comprehensive monitoring and observability. This ensures automatic scaling, A/B testing capabilities, and detailed performance tracking.

### Model Serving Architecture

#### Ray Serve Deployment Configuration
```python
# src/serving/deployment_manager.py
"""Ray Serve deployment management with MLflow model integration."""

import ray
from ray import serve
from ray.serve import DeploymentHandle
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
import time
from prometheus_client import Counter, Histogram, Gauge
import logging

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['model_name', 'version'])
REQUEST_DURATION = Histogram('inference_duration_seconds', 'Inference duration', ['model_name', 'version'])
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model loading duration')
ACTIVE_MODELS = Gauge('active_models_count', 'Number of active model deployments')

@dataclass
class ModelDeploymentConfig:
    """Configuration for model deployment."""
    model_name: str
    model_version: str
    num_replicas: int = 2
    max_concurrent_queries: int = 100
    ray_actor_options: Dict[str, Any] = None
    health_check_period_s: float = 10.0
    graceful_shutdown_timeout_s: float = 20.0

@serve.deployment(
    name="ml_model_predictor",
    num_replicas=2,
    max_concurrent_queries=100,
    ray_actor_options={"num_gpus": 1}
)
class MLModelPredictor:
    """Ray Serve deployment for ML model inference."""

    def __init__(self, model_name: str, model_version: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.model_version = model_version
        self.model_config = model_config
        self.model = None
        self.device = None
        self.load_model()

        # Setup logging
        self.logger = logging.getLogger(f"ModelPredictor-{model_name}-{model_version}")

    def load_model(self) -> None:
        """Loads model from MLflow registry with performance tracking."""
        start_time = time.time()

        try:
            # Load model from MLflow
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self.model = mlflow.pytorch.load_model(model_uri)

            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            # Warm up model
            self._warmup_model()

            load_time = time.time() - start_time
            MODEL_LOAD_TIME.observe(load_time)
            ACTIVE_MODELS.inc()

            self.logger.info(f"Model {self.model_name}:{self.model_version} loaded in {load_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _warmup_model(self) -> None:
        """Performs model warmup with dummy data."""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles inference requests with monitoring."""
        start_time = time.time()
        REQUEST_COUNT.labels(model_name=self.model_name, version=self.model_version).inc()

        try:
            # Extract and validate input data
            input_data = self._preprocess_input(request)

            # Run inference
            with torch.no_grad():
                predictions = self.model(input_data)
                probabilities = torch.softmax(predictions, dim=1)

            # Post-process results
            results = self._postprocess_output(probabilities)

            # Track performance metrics
            duration = time.time() - start_time
            REQUEST_DURATION.labels(model_name=self.model_name, version=self.model_version).observe(duration)

            return {
                "predictions": results,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "inference_time_ms": duration * 1000,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return {
                "error": str(e),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "timestamp": time.time()
            }

    def _preprocess_input(self, request: Dict[str, Any]) -> torch.Tensor:
        """Preprocesses input data for model inference."""
        # Implement input preprocessing based on model requirements
        image_data = np.array(request["image"])

        # Convert to tensor and normalize
        tensor_data = torch.from_numpy(image_data).float()
        if len(tensor_data.shape) == 3:
            tensor_data = tensor_data.unsqueeze(0)  # Add batch dimension

        return tensor_data.to(self.device)

    def _postprocess_output(self, predictions: torch.Tensor) -> List[Dict[str, Any]]:
        """Postprocesses model output for response."""
        results = []

        for pred in predictions:
            top_k_values, top_k_indices = torch.topk(pred, k=5)

            results.append({
                "class_probabilities": [
                    {"class_id": idx.item(), "probability": val.item()}
                    for idx, val in zip(top_k_indices, top_k_values)
                ],
                "predicted_class": top_k_indices[0].item(),
                "confidence": top_k_values[0].item()
            })

        return results

class ModelDeploymentManager:
    """Manages model deployments and A/B testing."""

    def __init__(self):
        self.active_deployments: Dict[str, DeploymentHandle] = {}
        self.logger = logging.getLogger("ModelDeploymentManager")

    def deploy_model(self, config: ModelDeploymentConfig) -> str:
        """Deploys a model with the specified configuration."""
        deployment_name = f"{config.model_name}_v{config.model_version}"

        # Create deployment
        deployment = MLModelPredictor.options(
            name=deployment_name,
            num_replicas=config.num_replicas,
            max_concurrent_queries=config.max_concurrent_queries,
            ray_actor_options=config.ray_actor_options or {"num_gpus": 1}
        ).bind(config.model_name, config.model_version, {})

        # Deploy to Ray Serve
        serve.run(deployment, name=deployment_name, route_prefix=f"/predict/{deployment_name}")

        # Store deployment handle
        self.active_deployments[deployment_name] = deployment

        self.logger.info(f"Deployed model: {deployment_name}")
        return deployment_name

    def setup_ab_testing(self, model_configs: List[ModelDeploymentConfig],
                        traffic_split: Dict[str, float]) -> str:
        """Sets up A/B testing between multiple model versions."""

        # Deploy all models
        deployments = {}
        for config in model_configs:
            deployment_name = self.deploy_model(config)
            deployments[deployment_name] = self.active_deployments[deployment_name]

        # Create traffic splitting logic
        @serve.deployment(name="ab_test_router")
        class ABTestRouter:
            def __init__(self, deployments: Dict[str, DeploymentHandle], splits: Dict[str, float]):
                self.deployments = deployments
                self.splits = splits
                self.total_requests = 0

            async def __call__(self, request: Dict[str, Any]):
                self.total_requests += 1

                # Simple round-robin based on traffic split
                cumulative_prob = 0
                random_val = (self.total_requests % 100) / 100

                for deployment_name, prob in self.splits.items():
                    cumulative_prob += prob
                    if random_val <= cumulative_prob:
                        return await self.deployments[deployment_name].remote(request)

                # Fallback to first deployment
                first_deployment = list(self.deployments.values())[0]
                return await first_deployment.remote(request)

        router = ABTestRouter.bind(deployments, traffic_split)
        serve.run(router, name="ab_test_router", route_prefix="/predict")

        return "ab_test_router"
```

### API Gateway & Load Balancing

#### NGINX Configuration for Model Serving
```nginx
# configs/nginx.conf
upstream ray_serve_backend {
    least_conn;
    server ray-head:8000 max_fails=3 fail_timeout=30s;
    server ray-worker-1:8000 max_fails=3 fail_timeout=30s backup;
    server ray-worker-2:8000 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 80;
    server_name api.ml-platform.local;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }

    # Model inference endpoints
    location /predict/ {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://ray_serve_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;

        # Enable request/response logging
        access_log /var/log/nginx/api_access.log combined;
        error_log /var/log/nginx/api_error.log;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://ray_serve_backend/metrics;
    }

    # Block common attack patterns
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
}
```

### Monitoring & Observability Stack

#### Prometheus Configuration
```yaml
# configs/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Ray Serve metrics
  - job_name: 'ray-serve'
    static_configs:
      - targets: ['ray-head:8000', 'ray-worker-1:8000', 'ray-worker-2:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # MLflow metrics
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  # NGINX metrics
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']

  # PostgreSQL metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # MinIO metrics
  - job_name: 'minio'
    static_configs:
      - targets: ['minio:9000']
    metrics_path: '/minio/v2/metrics/cluster'

  # Custom application metrics
  - job_name: 'ml-app-metrics'
    static_configs:
      - targets: ['ml-app:8080']
    metrics_path: '/metrics'
```

#### Alert Rules Configuration
```yaml
# configs/alert_rules.yml
groups:
  - name: ml_serving_alerts
    rules:
      # High error rate
      - alert: HighInferenceErrorRate
        expr: rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High inference error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for model {{ $labels.model_name }}"

      # High latency
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 1.0
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile latency is {{ $value }}s for model {{ $labels.model_name }}"

      # Model deployment down
      - alert: ModelDeploymentDown
        expr: up{job="ray-serve"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model serving deployment is down"
          description: "Ray Serve instance {{ $labels.instance }} is down"

      # High memory usage
      - alert: HighMemoryUsage
        expr: (ray_node_memory_used / ray_node_memory_total) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on Ray node"
          description: "Memory usage is {{ $value | humanizePercentage }} on node {{ $labels.node_id }}"
```

#### Grafana Dashboard Configuration
```python
# src/monitoring/dashboard_generator.py
"""Automated Grafana dashboard generation for ML serving monitoring."""

import json
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class DashboardPanel:
    """Configuration for a Grafana dashboard panel."""
    title: str
    panel_type: str
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]
    options: Dict[str, Any] = None

class MLServingDashboard:
    """Generates comprehensive ML serving monitoring dashboards."""

    def __init__(self, dashboard_name: str = "ML Model Serving"):
        self.dashboard_name = dashboard_name
        self.panels = []

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generates complete Grafana dashboard configuration."""

        dashboard = {
            "dashboard": {
                "id": None,
                "title": self.dashboard_name,
                "tags": ["ml", "serving", "monitoring"],
                "timezone": "browser",
                "panels": [
                    self._create_overview_panel(),
                    self._create_request_rate_panel(),
                    self._create_latency_panel(),
                    self._create_error_rate_panel(),
                    self._create_resource_usage_panel(),
                    self._create_model_performance_panel(),
                    self._create_ab_testing_panel()
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }

        return dashboard

    def _create_overview_panel(self) -> Dict[str, Any]:
        """Creates overview statistics panel."""
        return {
            "id": 1,
            "title": "Service Overview",
            "type": "stat",
            "targets": [
                {
                    "expr": "sum(rate(inference_requests_total[5m]))",
                    "legendFormat": "Requests/sec"
                },
                {
                    "expr": "count(up{job='ray-serve'} == 1)",
                    "legendFormat": "Active Deployments"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))",
                    "legendFormat": "P95 Latency"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        }

    def _create_request_rate_panel(self) -> Dict[str, Any]:
        """Creates request rate monitoring panel."""
        return {
            "id": 2,
            "title": "Request Rate by Model",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum(rate(inference_requests_total[5m])) by (model_name)",
                    "legendFormat": "{{ model_name }}"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "yAxes": [
                {"label": "Requests/sec", "min": 0},
                {"show": False}
            ]
        }

    def _create_latency_panel(self) -> Dict[str, Any]:
        """Creates latency monitoring panel."""
        return {
            "id": 3,
            "title": "Inference Latency Distribution",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(inference_duration_seconds_bucket[5m]))",
                    "legendFormat": "P50"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))",
                    "legendFormat": "P95"
                },
                {
                    "expr": "histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))",
                    "legendFormat": "P99"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "yAxes": [
                {"label": "Seconds", "min": 0},
                {"show": False}
            ]
        }
```

### Distributed Tracing with Jaeger

#### Request Tracing Implementation
```python
# src/monitoring/tracing.py
"""Distributed tracing implementation for ML serving pipeline."""

import opentracing
from jaeger_client import Config
import time
from typing import Dict, Any, Optional
from functools import wraps
import logging

class TracingManager:
    """Manages distributed tracing across ML serving components."""

    def __init__(self, service_name: str = "ml-serving"):
        self.service_name = service_name
        self.tracer = self._initialize_tracer()

    def _initialize_tracer(self) -> opentracing.Tracer:
        """Initializes Jaeger tracer with configuration."""
        config = Config(
            config={
                'sampler': {
                    'type': 'const',
                    'param': 1,
                },
                'logging': True,
                'reporter_batch_size': 1,
            },
            service_name=self.service_name,
            validate=True,
        )

        return config.initialize_tracer()

    def trace_inference_request(self, operation_name: str = "model_inference"):
        """Decorator for tracing inference requests."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_span(operation_name) as span:
                    span.set_tag("component", "ml-model")
                    span.set_tag("span.kind", "server")

                    start_time = time.time()

                    try:
                        result = await func(*args, **kwargs)
                        span.set_tag("success", True)
                        span.set_tag("model_name", result.get("model_name", "unknown"))
                        return result
                    except Exception as e:
                        span.set_tag("error", True)
                        span.set_tag("error.message", str(e))
                        span.log_kv({"event": "error", "error.object": e})
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_tag("inference_duration_ms", duration * 1000)

            return wrapper
        return decorator

# Apply tracing to model serving
tracing_manager = TracingManager("ml-model-serving")

@tracing_manager.trace_inference_request("preprocess_input")
async def traced_preprocess(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Traced preprocessing function."""
    # Preprocessing logic here
    pass

@tracing_manager.trace_inference_request("model_inference")
async def traced_inference(model, input_data: torch.Tensor) -> torch.Tensor:
    """Traced model inference function."""
    # Inference logic here
    pass
```

This comprehensive serving and monitoring framework ensures production-ready model deployment with full observability, automatic scaling, and robust error handling.
