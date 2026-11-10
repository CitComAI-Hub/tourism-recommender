# Web app: static export, served by the API container under BASE_PATH
FROM node:22-slim AS web
WORKDIR /web
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
ARG BASE_PATH=/tourism
ENV NEXT_PUBLIC_BASE_PATH=${BASE_PATH} NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM python:3.12-slim
# git: ds-consumer-auth is installed from its GitHub tag
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[dataspace]"
COPY configs/base.yaml configs/base.yaml
# Catalog and its embeddings. The behaviour model is mounted at BEHAVIOUR_MODEL_PATH
# (without it the service ranks by Wikipedia popularity).
COPY data/processed/poi.jsonld data/processed/embeddings.jsonl data/processed/
COPY --from=web /web/out /app/web

ENV PMV_CONFIG=configs/base.yaml WEB_DIR=/app/web PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN useradd --uid 10001 --create-home app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"]
# One process: the catalog (a few hundred POIs) lives in its memory.
CMD ["uvicorn", "pmv_turismo.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
