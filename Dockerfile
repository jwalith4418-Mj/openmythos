# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.11

# ─── Build stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ARG BUILD_SHA=unknown

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python3-pip python3-venv git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /build

# Install Python deps from locked requirements first (better layer caching)
COPY requirements.lock /build/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.lock

# Copy and install the package
COPY . /build
RUN pip install --no-cache-dir -e ".[serve]" --no-deps

# ─── Runtime stage ──────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION
ARG BUILD_SHA

LABEL org.opencontainers.image.title="OpenMythos Inference Server"
LABEL org.opencontainers.image.version="${BUILD_SHA}"
LABEL org.opencontainers.image.description="Recurrent-Depth Transformer serving API"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python3-pip curl tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Non-root user for runtime security
RUN groupadd -g 1000 openmythos && useradd -u 1000 -g openmythos -m -s /bin/bash openmythos

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION} /usr/local/lib/python${PYTHON_VERSION}
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

WORKDIR /app
RUN chown -R openmythos:openmythos /app

USER openmythos

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BUILD_SHA=${BUILD_SHA} \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as PID 1 for proper signal handling (SIGTERM → graceful shutdown)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "open_mythos.serving.entrypoint"]
