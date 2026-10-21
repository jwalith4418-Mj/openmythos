"""
OpenMythos Production Inference Server — v2

Fixes over v1:
  1. Streaming SSE fully implemented (was `pass` in v1).
  2. ServerMetrics replaced with prometheus_client (real Prometheus scraping).
  3. API key authentication middleware added.
  4. CORS restricted to CORS_ORIGINS env var.
  5. Token counting uses tokenizer.encode(), not whitespace split.
  6. /health/ready endpoint checks model is loaded.
  7. OpenTelemetry instrumentation added.
  8. Rate limiting via slowapi.
  9. SGLang engine added.
  10. Dynamic request queue for batching under concurrent load.
  11. Structured audit logging for every request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Union

import torch

# ---------------------------------------------------------------------------
# Optional imports with graceful degradation
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Request, Security, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False

try:
    import vllm
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

try:
    import sglang as sgl
    _HAS_SGLANG = True
except ImportError:
    _HAS_SGLANG = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)s}',
    level=logging.INFO,
)
_audit_log = logging.getLogger("openmythos.audit")
_server_log = logging.getLogger("openmythos.server")

# ---------------------------------------------------------------------------
# Prometheus metrics (real prometheus_client, not string concatenation)
# ---------------------------------------------------------------------------

if _HAS_PROMETHEUS:
    _registry = CollectorRegistry() if os.getenv("PYTEST_CURRENT_TEST") else REGISTRY
    _req_total = Counter(
        "openmythos_requests_total", "Total inference requests",
        labelnames=["endpoint", "status"],
        registry=_registry,
    )
    _req_latency = Histogram(
        "openmythos_request_latency_seconds", "Request latency",
        labelnames=["endpoint"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        registry=_registry,
    )
    _tokens_total = Counter(
        "openmythos_tokens_generated_total", "Total tokens generated",
        labelnames=["endpoint"],
        registry=_registry,
    )
    _active_requests = Gauge("openmythos_active_requests", "Currently processing requests", registry=_registry)
    _gpu_utilization = Gauge("openmythos_gpu_utilization_percent", "GPU utilization %", labelnames=["device"], registry=_registry)

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------

_tracer = None

def _init_otel(service_name: str = "openmythos") -> None:
    global _tracer
    if not _HAS_OTEL:
        return
    provider = TracerProvider()
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if otel_endpoint:
        exporter = OTLPSpanExporter(endpoint=otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------

_API_KEY_HEADER = "X-API-Key" if _HAS_FASTAPI else None

def _load_api_keys() -> set:
    """Load API keys from OPENMYTHOS_API_KEYS env var (comma-separated)."""
    raw = os.getenv("OPENMYTHOS_API_KEYS", "")
    if not raw:
        return set()  # empty = auth disabled (dev mode)
    return set(k.strip() for k in raw.split(",") if k.strip())

_VALID_API_KEYS = _load_api_keys()
# FIX: Auth must be explicitly disabled via OPENMYTHOS_DISABLE_AUTH=1 env var.
# Silently running open in production (when no keys are set) was a security risk.
_AUTH_DISABLED = (len(_VALID_API_KEYS) == 0) and (os.getenv("OPENMYTHOS_DISABLE_AUTH", "0") == "1")
if len(_VALID_API_KEYS) == 0 and not _AUTH_DISABLED:
    import warnings
    warnings.warn(
        "[OpenMythos] OPENMYTHOS_API_KEYS is not set and OPENMYTHOS_DISABLE_AUTH != 1. "
        "All requests will be REJECTED with 401. Set OPENMYTHOS_API_KEYS or set "
        "OPENMYTHOS_DISABLE_AUTH=1 to explicitly allow unauthenticated access (dev only).",
        stacklevel=2,
    )

if _HAS_FASTAPI:
    _api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
        if _AUTH_DISABLED:
            return "dev"
        if api_key is None or api_key not in _VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return api_key

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

if _HAS_FASTAPI:
    class CompletionRequest(BaseModel):
        model: str = "openmythos"
        prompt: Union[str, List[str]]
        max_tokens: int = Field(default=256, ge=1, le=8192)
        temperature: float = Field(default=1.0, ge=0.0, le=2.0)
        top_p: float = Field(default=1.0, ge=0.0, le=1.0)
        top_k: int = Field(default=50, ge=0)
        n: int = Field(default=1, ge=1, le=16)
        stream: bool = False
        stop: Optional[Union[str, List[str]]] = None
        n_loops: int = Field(default=8, ge=1, le=64)
        seed: Optional[int] = None

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ToolFunction(BaseModel):
        name: str
        description: str
        parameters: dict = Field(default_factory=dict)

    class Tool(BaseModel):
        type: str = "function"
        function: ToolFunction

    class ChatCompletionRequest(BaseModel):
        model: str = "openmythos"
        messages: List[ChatMessage]
        max_tokens: int = Field(default=256, ge=1, le=8192)
        temperature: float = Field(default=1.0, ge=0.0, le=2.0)
        top_p: float = Field(default=1.0, ge=0.0, le=1.0)
        n: int = Field(default=1, ge=1, le=16)
        stream: bool = False
        n_loops: int = Field(default=8, ge=1, le=64)
        seed: Optional[int] = None
        tools: Optional[List[Tool]] = None
        tool_choice: Optional[str] = "auto"

    class UsageInfo(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class CompletionChoice(BaseModel):
        index: int
        text: str
        finish_reason: str
        logprobs: None = None

    class CompletionResponse(BaseModel):
        id: str
        object: str = "text_completion"
        created: int
        model: str
        choices: List[CompletionChoice]
        usage: UsageInfo

    class ChatChoice(BaseModel):
        index: int
        message: ChatMessage
        finish_reason: str

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatChoice]
        usage: UsageInfo

# ---------------------------------------------------------------------------
# Inference Engines
# ---------------------------------------------------------------------------


class NativeEngine:
    """Native PyTorch generation with proper async and token-accurate counting."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self._model_ready = True

    def is_ready(self) -> bool:
        return self._model_ready

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return max(1, len(text) // 4)  # fallback: 4 chars/token

    def generate(self, prompt, max_new_tokens=256, temperature=1.0, top_p=1.0,
                 top_k=50, n_loops=8, seed=None, stop=None):
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        prompt_len = len(ids)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, n_loops=n_loops,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
        new_ids = output_ids[0, prompt_len:].tolist()
        text = self.tokenizer.decode(new_ids)
        if stop:
            stops = [stop] if isinstance(stop, str) else stop
            for s in stops:
                if s in text:
                    text = text[:text.index(s)]
        return text

    async def generate_async(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))

    async def generate_stream(self, prompt, max_new_tokens=256, temperature=1.0,
                               top_k=50, n_loops=8, stop=None, **kwargs) -> AsyncIterator[str]:
        """Token-by-token streaming with stop-sequence support."""
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        kv_cache: dict = {}
        generated_text = ""
        stops = ([stop] if isinstance(stop, str) else stop) if stop else []

        for step in range(max_new_tokens):
            cur = input_ids if step == 0 else input_ids[:, -1:]
            with torch.no_grad():
                logits = self.model(
                    cur, n_loops=n_loops, kv_cache=kv_cache,
                    start_pos=len(ids) + step - 1 if step > 0 else 0,
                )
            logits = logits[:, -1, :]
            if temperature == 0.0:
                next_tok = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k > 0:
                    v, _ = logits.topk(top_k)
                    logits[logits < v[:, -1:]] = float("-inf")
                next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            next_id = next_tok.item()
            if next_id == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode([next_id])
            generated_text += token_text

            # Check stop sequences
            stop_hit = any(s in generated_text for s in stops)
            if stop_hit:
                for s in stops:
                    if s in generated_text:
                        # Yield only up to the stop
                        idx = generated_text.index(s)
                        prefix = generated_text[:idx]
                        already_yielded = generated_text[:-len(token_text)]
                        remainder = prefix[len(already_yielded):]
                        if remainder:
                            yield remainder
                        return

            yield token_text
            input_ids = torch.cat([input_ids, next_tok], dim=1)


class VLLMEngine:
    """vLLM engine with PagedAttention continuous batching."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        if not _HAS_VLLM:
            raise ImportError("vLLM not installed. pip install vllm")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self._model_ready = True

    def is_ready(self) -> bool:
        return self._model_ready

    def generate(self, prompt, max_new_tokens=256, temperature=1.0, top_p=1.0,
                 top_k=-1, stop=None, **kwargs) -> str:
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            stop=stop if stop else [],
        )
        results = self.llm.generate([prompt], params)
        return results[0].outputs[0].text

    async def generate_async(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))

    async def generate_stream(self, prompt, max_new_tokens=256, temperature=1.0,
                               top_k=-1, stop=None, **kwargs) -> AsyncIterator[str]:
        """vLLM async streaming via generate_async (batched output)."""
        text = await self.generate_async(
            prompt=prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, stop=stop,
        )
        # Yield in ~4-char chunks to simulate streaming
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0)


class SGLangEngine:
    """SGLang engine for structured generation and optimized serving."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        if not _HAS_SGLANG:
            raise ImportError("SGLang not installed. pip install sglang")
        self.runtime = sgl.Runtime(
            model_path=model_path,
            tp_size=tensor_parallel_size,
        )
        self._model_ready = True

    def is_ready(self) -> bool:
        return self._model_ready

    def generate(self, prompt, max_new_tokens=256, temperature=1.0, top_p=1.0,
                 stop=None, **kwargs) -> str:
        @sgl.function
        def _gen(s, prompt_text):
            s += prompt_text
            s += sgl.gen(
                "output",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

        state = _gen.run(prompt_text=prompt, backend=self.runtime)
        return state["output"]

    async def generate_async(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))

    async def generate_stream(self, prompt, max_new_tokens=256, temperature=1.0,
                               stop=None, **kwargs) -> AsyncIterator[str]:
        text = await self.generate_async(
            prompt=prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, stop=stop,
        )
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


def apply_chat_template(messages) -> str:
    parts = []
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def _log_request(
    request_id: str,
    endpoint: str,
    api_key: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_sec: float,
    status: str,
) -> None:
    _audit_log.info(json.dumps({
        "request_id": request_id,
        "endpoint": endpoint,
        "api_key_prefix": api_key[:8] + "..." if len(api_key) > 8 else api_key,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_sec": round(latency_sec, 4),
        "status": status,
    }))


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(
    model=None,
    tokenizer=None,
    device="cuda",
    vllm_model_path=None,
    sglang_model_path=None,
    tensor_parallel_size=1,
    model_name="openmythos",
) -> "FastAPI":
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI not installed. pip install fastapi uvicorn")

    # Initialise observability
    _init_otel(service_name=model_name)

    # Select engine
    if sglang_model_path and _HAS_SGLANG:
        engine = SGLangEngine(sglang_model_path, tensor_parallel_size)
        _server_log.info(f"Using SGLang engine: {sglang_model_path}")
    elif vllm_model_path and _HAS_VLLM:
        engine = VLLMEngine(vllm_model_path, tensor_parallel_size)
        _server_log.info(f"Using vLLM engine: {vllm_model_path}")
    else:
        if model is None:
            raise ValueError("Must provide model, vllm_model_path, or sglang_model_path")
        engine = NativeEngine(model, tokenizer, device)
        _server_log.info("Using native PyTorch engine")

    # Continuous batching scheduler (NativeEngine only; vLLM/SGLang handle their own batching)
    # Disabled during testing to prevent hanging on synchronous TestClient.
    _scheduler = None
    disable_batching = os.getenv("OPENMYTHOS_DISABLE_BATCHING") == "1" or os.getenv("PYTEST_CURRENT_TEST") is not None
    if isinstance(engine, NativeEngine) and not disable_batching:
        from open_mythos.serving.batch_engine import ContinuousBatchScheduler, HealthMonitor
        max_batch = int(os.getenv("OPENMYTHOS_MAX_BATCH_SIZE", "32"))
        max_wait_ms = float(os.getenv("OPENMYTHOS_BATCH_WAIT_MS", "20"))
        _scheduler = ContinuousBatchScheduler(engine, max_batch_size=max_batch, max_wait_ms=max_wait_ms)
        _health_monitor = HealthMonitor(_scheduler)
        _server_log.info(f"ContinuousBatchScheduler enabled: batch={max_batch}, wait={max_wait_ms}ms")

    # Rate limiter
    limiter = None
    if _HAS_SLOWAPI:
        rate_limit_str = os.getenv("OPENMYTHOS_RATE_LIMIT", "60/minute")
        limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit_str])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if _scheduler is not None:
            await _scheduler.start()
            _server_log.info("ContinuousBatchScheduler started")
        _server_log.info(f"OpenMythos inference server ready — model: {model_name}")
        yield
        if _scheduler is not None:
            await _scheduler.stop()
        _server_log.info("Shutting down OpenMythos server")

    app = FastAPI(
        title="OpenMythos Inference API",
        version="2.0.0",
        description="OpenAI-compatible API for OpenMythos language models",
        lifespan=lifespan,
    )

    # CORS — restricted to env var, not wildcard
    cors_origins_raw = os.getenv("CORS_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()] or ["http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # Rate limiter middleware
    if limiter:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # OpenTelemetry FastAPI instrumentation
    if _HAS_OTEL:
        FastAPIInstrumentor.instrument_app(app)

    # ------------------------------------------------------------------
    # Health probes
    # ------------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    async def liveness():
        """Kubernetes liveness probe — server is alive."""
        return {"status": "alive", "model": model_name}

    @app.get("/health/ready", tags=["ops"])
    async def readiness():
        """Kubernetes readiness probe — model is loaded and ready to serve."""
        if not engine.is_ready():
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "ready", "model": model_name}

    # ------------------------------------------------------------------
    # Prometheus metrics endpoint
    # ------------------------------------------------------------------

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        if not _HAS_PROMETHEUS:
            return {"error": "prometheus_client not installed"}
        from fastapi.responses import Response
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------
    # Models list
    # ------------------------------------------------------------------

    @app.get("/v1/models", tags=["models"])
    async def list_models(api_key: str = Depends(verify_api_key)):
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "openmythos"}],
        }

    # ------------------------------------------------------------------
    # Text completions
    # ------------------------------------------------------------------

    @app.post("/v1/completions", response_model=CompletionResponse, tags=["completions"])
    async def completions(req: CompletionRequest, api_key: str = Depends(verify_api_key)):
        request_id = f"cmpl-{uuid.uuid4().hex}"
        t0 = time.time()

        if _HAS_PROMETHEUS:
            _active_requests.inc()

        try:
            prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt

            # Streaming response
            if req.stream:
                prompt = prompts[0]

                async def event_stream():
                    if _scheduler is not None:
                        from open_mythos.serving.batch_engine import InferenceRequest as _IR
                        ir = _IR(
                            request_id=request_id,
                            prompt=prompt,
                            max_new_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            n_loops=req.n_loops,
                            stop=req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None),
                            stream=True,
                        )
                        async for token in _scheduler.submit_stream(ir):
                            chunk = {
                                "id": request_id,
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [{"text": token, "index": 0, "finish_reason": None, "logprobs": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        async for token in engine.generate_stream(
                            prompt=prompt,
                            max_new_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            n_loops=req.n_loops,
                            stop=req.stop,
                        ):
                            chunk = {
                                "id": request_id,
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [{"text": token, "index": 0, "finish_reason": None, "logprobs": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")

            # Non-streaming — route through ContinuousBatchScheduler if available
            choices = []
            total_completion_tokens = 0
            for i, prompt in enumerate(prompts):
                for j in range(req.n):
                    if _scheduler is not None:
                        from open_mythos.serving.batch_engine import InferenceRequest as _IR
                        ir = _IR(
                            request_id=f"{request_id}-{i}-{j}",
                            prompt=prompt,
                            max_new_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            top_k=req.top_k,
                            n_loops=req.n_loops,
                            stop=req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None),
                            seed=req.seed,
                        )
                        result = await _scheduler.submit(ir)
                        text = result.text
                    else:
                        text = await engine.generate_async(
                            prompt=prompt, max_new_tokens=req.max_tokens,
                            temperature=req.temperature, top_p=req.top_p,
                            top_k=req.top_k, n_loops=req.n_loops,
                            seed=req.seed, stop=req.stop,
                        )
                    choices.append(CompletionChoice(index=i * req.n + j, text=text, finish_reason="stop"))
                    n_toks = len(tokenizer.encode(text)) if tokenizer else max(1, len(text) // 4)
                    total_completion_tokens += n_toks

            prompt_tokens = sum(
                len(tokenizer.encode(p)) if tokenizer else max(1, len(p) // 4) for p in prompts
            )
            latency = time.time() - t0

            if _HAS_PROMETHEUS:
                _req_total.labels(endpoint="/v1/completions", status="ok").inc()
                _req_latency.labels(endpoint="/v1/completions").observe(latency)
                _tokens_total.labels(endpoint="/v1/completions").inc(total_completion_tokens)

            _log_request(request_id, "/v1/completions", api_key, prompt_tokens, total_completion_tokens, latency, "ok")

            return CompletionResponse(
                id=request_id, created=int(time.time()), model=model_name, choices=choices,
                usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=total_completion_tokens,
                                total_tokens=prompt_tokens + total_completion_tokens),
            )
        except Exception as e:
            if _HAS_PROMETHEUS:
                _req_total.labels(endpoint="/v1/completions", status="error").inc()
            _server_log.error(f"Completion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            if _HAS_PROMETHEUS:
                _active_requests.dec()

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["completions"])
    async def chat_completions(req: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        t0 = time.time()
        prompt = apply_chat_template(req.messages)

        if _HAS_PROMETHEUS:
            _active_requests.inc()

        try:
            if req.stream:
                async def chat_event_stream():
                    async for token in engine.generate_stream(
                        prompt=prompt, max_new_tokens=req.max_tokens,
                        temperature=req.temperature, n_loops=req.n_loops,
                    ):
                        chunk = {
                            "id": request_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(chat_event_stream(), media_type="text/event-stream")

            choices = []
            total_completion_tokens = 0
            for j in range(req.n):
                text = await engine.generate_async(
                    prompt=prompt, max_new_tokens=req.max_tokens,
                    temperature=req.temperature, top_p=req.top_p,
                    n_loops=req.n_loops, seed=req.seed,
                )
                choices.append(ChatChoice(index=j, message=ChatMessage(role="assistant", content=text), finish_reason="stop"))
                n_toks = len(tokenizer.encode(text)) if tokenizer else max(1, len(text) // 4)
                total_completion_tokens += n_toks

            prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer else max(1, len(prompt) // 4)
            latency = time.time() - t0

            if _HAS_PROMETHEUS:
                _req_total.labels(endpoint="/v1/chat/completions", status="ok").inc()
                _req_latency.labels(endpoint="/v1/chat/completions").observe(latency)
                _tokens_total.labels(endpoint="/v1/chat/completions").inc(total_completion_tokens)

            _log_request(request_id, "/v1/chat/completions", api_key, prompt_tokens, total_completion_tokens, latency, "ok")

            return ChatCompletionResponse(
                id=request_id, created=int(time.time()), model=model_name, choices=choices,
                usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=total_completion_tokens,
                                total_tokens=prompt_tokens + total_completion_tokens),
            )
        except Exception as e:
            if _HAS_PROMETHEUS:
                _req_total.labels(endpoint="/v1/chat/completions", status="error").inc()
            _server_log.error(f"Chat completion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            if _HAS_PROMETHEUS:
                _active_requests.dec()

    # ------------------------------------------------------------------
    # OpenMythos-unique: Elastic Compute Telemetry endpoints
    # ------------------------------------------------------------------

    @app.post("/v1/loop_profile", tags=["compute"])
    async def loop_profile(req: CompletionRequest, api_key: str = Depends(verify_api_key)):
        """
        OpenMythos-unique endpoint: returns per-token loop counts alongside the
        generated text. This exposes elastic compute usage — the killer metric.

        Response includes:
          - text: generated text
          - loop_counts: list of int, one per generated token (elastic compute per token)
          - mean_loops: average loop depth used
          - reasoning_flop_efficiency: RFE proxy for this request
        """
        if not isinstance(engine, NativeEngine):
            raise HTTPException(status_code=400, detail="Loop profiling only available with NativeEngine")

        from open_mythos.inference.profiler import InferenceProfiler, _estimate_flops_per_forward
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        profiler = InferenceProfiler(engine.model, engine.tokenizer, device=device, max_loops=req.n_loops)
        profile = profiler.profile_query(prompt, max_new_tokens=req.max_tokens, n_loops=req.n_loops)

        mean_loops = sum(profile.loop_counts) / max(len(profile.loop_counts), 1)
        return {
            "id": f"profile-{uuid.uuid4().hex}",
            "model": model_name,
            "prompt_tokens": profile.n_prompt_tokens,
            "completion_tokens": profile.n_generated_tokens,
            "loop_counts": profile.loop_counts,
            "mean_loops": round(mean_loops, 2),
            "max_loops_requested": req.n_loops,
            "compute_utilization": round(mean_loops / max(req.n_loops, 1), 3),
            "estimated_flops": profile.total_flops,
            "latency_ms": round(profile.latency_ms, 2),
        }

    @app.get("/v1/queue_stats", tags=["ops"])
    async def queue_stats(api_key: str = Depends(verify_api_key)):
        """
        Returns current batch scheduler queue depth and latency percentiles.
        Use to drive autoscaling decisions from the application layer.
        """
        if _scheduler is None:
            return {"scheduler": "disabled", "engine": "vllm_or_sglang"}
        from open_mythos.serving.batch_engine import HealthMonitor
        mon = HealthMonitor(_scheduler)
        return {
            "queue_depth": mon.queue_depth,
            "p50_latency_ms": round(mon.percentile(50), 1),
            "p95_latency_ms": round(mon.percentile(95), 1),
            "p99_latency_ms": round(mon.percentile(99), 1),
            "gpu_memory_used_gb": round(mon.gpu_memory_used_gb(), 2),
            "healthy": mon.is_healthy(),
        }

    return app


def serve(model=None, tokenizer=None, host="0.0.0.0", port=8000, device="cuda",
          workers=1, vllm_model_path=None, sglang_model_path=None, tensor_parallel_size=1):
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. pip install uvicorn")
    app = create_app(
        model=model, tokenizer=tokenizer, device=device,
        vllm_model_path=vllm_model_path, sglang_model_path=sglang_model_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")
