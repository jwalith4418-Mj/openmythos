# Runbook: Scaling the Serving Tier

## Horizontal Pod Autoscaler

HPA is enabled by default (`autoscaling.enabled: true`). It scales on CPU utilization.
For GPU-based scaling, enable KEDA with the custom metric:

```yaml
autoscaling:
  customMetric:
    enabled: true
    metricName: openmythos_active_requests
    targetValue: "10"
```

## Manual Scale

```bash
kubectl scale deployment openmythos --replicas=4 -n openmythos
```

## Multi-GPU Single Node

Set `tensor_parallel_size` in the Helm values or via env:

```bash
TENSOR_PARALLEL_SIZE=4 python -m open_mythos.serving.entrypoint
```

Requires vLLM engine (`VLLM_MODEL_PATH` set).
