# Changelog

## v2.0.0 — Production Remediation Release

### Critical Bug Fixes
- **FIX**: Streaming SSE (`stream=True`) was a `pass` statement — now fully implemented
- **FIX**: `temperature=0.0` greedy decoding silently used `1e-5` — now uses true argmax
- **FIX**: `mythos_7b()` passed `qk_nope_head_dim` (wrong kwarg) — fixed to `qk_nope_head_dim`
- **FIX**: Expert Parallelism group was set to DP group — now has dedicated EP groups
- **FIX**: ACT early exit with KV cache was disabled — now works in all modes
- **FIX**: CORS wildcard `allow_origins=["*"]` — now reads from `CORS_ORIGINS` env var
- **FIX**: `FineWebEduDataset` silently fell back to random tokens — now raises `RuntimeError`
- **FIX**: Token counting used whitespace split — now uses `tokenizer.encode()`

### New Features
- **Infrastructure**: Dockerfile, docker-compose, Helm chart, HPA, PDB
- **CI/CD**: GitHub Actions (lint, test, security scan, release, Docker build)
- **Security**: API key authentication, rate limiting (slowapi), audit logging
- **Observability**: prometheus_client metrics, Grafana dashboard, OpenTelemetry, Jaeger
- **Serving**: SGLang engine, vLLM engine with HF export, dynamic batching
- **Training**: MFU/TFLOP/s tracking, GPU util via pynvml, DeepSpeed branch active, SIGTERM checkpoint
- **Evaluation**: TruthfulQA MC2, HumanEval pass@k, BBH via lm-eval, HellaSwag fallback, MMLU all 57 subjects
- **Data Platform**: MinHash deduplication, quality filtering, PII scrubbing, DVC config, lineage tracking
- **Reliability**: OOM recovery in trainer, chaos tests, checkpoint recovery tests
- **Developer Experience**: Makefile, pre-commit hooks, deployment guide, runbooks

### Removed
- `training/3b_fine_web_edu.py` — superseded by `training/pretrain_3b.py`
- `tests/small_benchmark.py`, `tests/bench_vs_transformer.py` — informal, replaced by `eval/benchmarks.py`
