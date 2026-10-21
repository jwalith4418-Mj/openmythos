# syntax=docker/dockerfile:1.7

# Supply the Python version directly instead of relying on Ubuntu's apt packages.
FROM python:3.11-slim AS builder
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY open_mythos ./open_mythos

# Install serving dependencies only. GPU users can supply a CUDA wheel index.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir=/wheels torch --index-url=${PYTORCH_INDEX_URL} && \
    pip wheel --wheel-dir=/wheels --find-links=/wheels ".[serve]"

FROM python:3.11-slim AS runtime
ARG BUILD_SHA=unknown
LABEL org.opencontainers.image.title="OpenMythos Inference Server" \
    org.opencontainers.image.version="${BUILD_SHA}" \
    org.opencontainers.image.source="https://github.com/jwalith4418-Mj/openmythos"

RUN apt-get update && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 openmythos \
    && useradd -u 1000 -g openmythos -m -s /bin/bash openmythos

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels "open-mythos[serve]" \
    && rm -rf /wheels

WORKDIR /app
RUN chown openmythos:openmythos /app
USER openmythos
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BUILD_SHA=${BUILD_SHA} \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "open_mythos.serving.entrypoint"]
