# OpenMythos — Deployment Guide

## Quick Start (Local Dev)

```bash
git clone https://github.com/openmythos/openmythos && cd openmythos
pip install -e ".[serve]"
export OPENMYTHOS_API_KEYS="dev-key"
export CORS_ORIGINS="http://localhost:3000"
python -m open_mythos.serving.entrypoint --device cpu
# API at http://localhost:8000
```

## Docker (Single Node)

```bash
cp .env.example .env           # fill in API_KEYS, MODEL_PATH
docker-compose up -d
curl -H "X-API-Key: $KEY" http://localhost:8000/health/ready
```

## Kubernetes (Helm)

### Prerequisites
- Kubernetes 1.28+
- NVIDIA GPU Operator installed
- cert-manager (for TLS)
- Prometheus Operator (for ServiceMonitor)

### Deploy

```bash
# 1. Build & push image
docker build -t ghcr.io/org/openmythos:v2.0.0 .
docker push ghcr.io/org/openmythos:v2.0.0

# 2. Create model PVC and populate it
kubectl apply -f k8s/model-pvc.yaml
kubectl cp checkpoints/100m openmythos-init:/models/100m

# 3. Install chart
helm upgrade --install openmythos helm/openmythos \
  --set image.tag=v2.0.0 \
  --set secrets.apiKeys="key1,key2,key3" \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=api.example.com \
  -n openmythos --create-namespace

# 4. Verify
kubectl rollout status deployment/openmythos -n openmythos
kubectl port-forward svc/openmythos 8000:8000 -n openmythos &
curl -H "X-API-Key: key1" http://localhost:8000/health/ready
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENMYTHOS_API_KEYS` | No (dev mode) | "" | Comma-separated valid API keys |
| `CORS_ORIGINS` | Yes (prod) | "http://localhost:3000" | Allowed CORS origins |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | "" | Jaeger/Tempo OTLP endpoint |
| `OPENMYTHOS_RATE_LIMIT` | No | "60/minute" | Rate limit per IP |
| `MODEL_PATH` | No | "" | Path to native checkpoint dir |
| `VLLM_MODEL_PATH` | No | "" | Path to HF-format model for vLLM |

## vLLM Serving

1. Export model: `make export-hf CHECKPOINT=checkpoints/100m OUTPUT=models/100m-hf`
2. Set `VLLM_MODEL_PATH=models/100m-hf` in `.env`
3. Restart server: `docker-compose restart server`

## Monitoring

- Grafana: http://localhost:3001 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger traces: http://localhost:16686
- Alerts: configured in `monitoring/alerts.yml`
