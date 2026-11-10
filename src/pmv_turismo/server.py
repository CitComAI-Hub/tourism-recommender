"""Production entry point: the API under /api and the web app (static export) at /.

One container, one origin: the gateway forwards /<base path>/* here with the prefix
stripped, the app (built with NEXT_PUBLIC_BASE_PATH) asks for <base path>/api/*.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pmv_turismo.api import app as api

WEB_DIR = Path(os.environ.get("WEB_DIR", "frontend/out"))

app = FastAPI(title="Tourism recommender", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/api", api)


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Liveness for Kubernetes: the process answers (the catalog may still be loading)."""
    return {"status": "ok"}


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:  # API-only (e.g. local development with `pnpm dev`)
    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": "tourism recommender", "api": "/api/docs"}
