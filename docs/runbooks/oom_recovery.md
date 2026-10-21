# Runbook: Out-of-Memory Recovery

## Symptoms
- Training crashes with `torch.cuda.OutOfMemoryError`
- Server returns 500 on large requests
- Pod OOMKilled in Kubernetes

## Training OOM

The trainer (v2) automatically recovers from a single OOM by clearing the CUDA
cache and skipping the offending batch. If OOM persists:

1. Reduce `micro_batch_size` in `TrainerConfig`
2. Reduce `seq_len` or enable curriculum (`curriculum=True`)
3. Enable gradient checkpointing: `model.enable_gradient_checkpointing()`
4. Switch to ZeRO-3: `use_deepspeed=True, ds_zero_stage=3`

## Serving OOM

The KV cache evicts old entries when `max_cache_len` is reached. If serving
still OOMs under high concurrency:

1. Reduce `max_cache_len` in `MythosConfig`
2. Add `resources.limits.memory` to Helm `values.yaml`
3. Enable vLLM with paged attention (automatically manages KV memory)

## Kubernetes OOMKilled

```bash
kubectl describe pod <pod> -n openmythos    # check limits
kubectl top pod -n openmythos               # check live usage
```

Increase `resources.limits.memory` in `values.yaml` and redeploy.
