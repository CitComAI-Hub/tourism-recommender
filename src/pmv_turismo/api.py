"""HTTP API of the tourism recommender.

Stateless: the client keeps the user's likes, dislikes and position and sends them
with each request, so the server stores nothing about users. Behind the data space
connector the user is already authenticated by APISIX.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from pmv_turismo.behaviour import load_model, model_path
from pmv_turismo.common import poi_coordinates, processed_dir
from pmv_turismo.config import Config
from pmv_turismo.data_source import load_pois, refresh_seconds, source_kind
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.recommendation import Recommender

CONFIG_PATH = Path(os.environ.get("PMV_CONFIG", "configs/base.yaml"))
logger = logging.getLogger(__name__)
MAX_IDS = 200


def _config() -> dict[str, Any]:
    return Config.load(CONFIG_PATH).data


class _State:
    """Recommender over the current catalog.

    Rebuilt when a local file changes (catalog, embeddings, behaviour model) and,
    when the POIs come from a broker, every POI_REFRESH_SECONDS. If a broker read
    fails, the previous catalog keeps being served.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._recommender: Recommender | None = None

    def get(self) -> Recommender:
        config = _config()
        paths = [processed_dir(config) / "poi.jsonld", embeddings_path(config), model_path(config)]
        key = tuple(p.stat().st_mtime if p.exists() else None for p in paths)
        if source_kind() != "file":
            key = key[1:] + (int(time.time() // max(refresh_seconds(), 1)),)
        with self._lock:
            if self._recommender is None or key != self._key:
                try:
                    pois = load_pois(config)
                    index = load_embedding_index(paths[1])
                except Exception as exc:
                    if self._recommender is not None:
                        logger.warning("Catalog refresh failed, serving the previous one: %s", exc)
                        self._key = key
                        return self._recommender
                    raise HTTPException(status_code=503, detail=f"Catalog not ready: {exc}") from exc
                rec = Recommender(pois, index, load_model(config), config)
                if len(rec.ids) < len(pois):
                    logger.warning("%d POIs without embeddings left out", len(pois) - len(rec.ids))
                self._recommender, self._key = rec, key
            return self._recommender


STATE = _State()


app = FastAPI(title="Tourism recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("PMV_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Location(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Feedback(BaseModel):
    liked: list[str] = Field(default_factory=list, max_length=MAX_IDS)
    disliked: list[str] = Field(default_factory=list, max_length=MAX_IDS)
    # Kinds of place the user prefers: they go first, the rest still appear
    categories: list[str] = Field(default_factory=list, max_length=50)


class CardsRequest(Feedback):
    seen: list[str] = Field(default_factory=list, max_length=MAX_IDS)
    limit: int = Field(default=6, ge=1, le=50)


class RecommendationRequest(Feedback):
    # Places already visited, oldest first: excluded, and the last one brings up
    # where visitors usually go next.
    visited: list[str] = Field(default_factory=list, max_length=MAX_IDS)
    location: Location | None = None
    top_n: int = Field(default=30, ge=1, le=200)
    at: datetime | None = None

    @field_validator("at")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("`at` must include a timezone")
        return v


def _csv(raw: str | None) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()] if raw else []


@app.get("/health")
def health() -> dict[str, Any]:
    rec = STATE.get()
    return {"status": "ok", "pois": len(rec.ids), "source": source_kind(), "behaviourModel": rec.model_meta}


@app.get("/pois")
def list_pois(
    category: str | None = Query(default=None, description="Comma-separated categories"),
    limit: int | None = Query(default=None, ge=1, le=2000),
    min_lat: float | None = None,
    min_lon: float | None = None,
    max_lat: float | None = None,
    max_lon: float | None = None,
) -> list[dict[str, Any]]:
    rec = STATE.get()
    wanted = set(_csv(category))
    bbox = None if None in (min_lat, min_lon, max_lat, max_lon) else (min_lat, min_lon, max_lat, max_lon)
    out = []
    for poi in rec.pois:
        if wanted and not wanted & {str(c).lower() for c in poi.get("category") or []}:
            continue
        if bbox:
            lon, lat = poi_coordinates(poi) or (None, None)
            if lat is None or not (bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]):
                continue
        out.append(poi)
    return out[:limit] if limit else out


@app.get("/pois/{poi_id}")
def get_poi(poi_id: str) -> dict[str, Any]:
    rec = STATE.get()
    if poi_id not in rec.position:
        raise HTTPException(status_code=404, detail="POI not found")
    return rec.pois[rec.position[poi_id]]


@app.post("/onboarding/cards", summary="Next onboarding cards: the most popular places not shown yet")
def onboarding_cards(payload: CardsRequest) -> list[dict[str, Any]]:
    rec = STATE.get()
    return rec.onboarding_cards(payload.limit, seen=payload.seen + payload.liked + payload.disliked,
                                categories=payload.categories)


@app.post("/recommendations")
def recommendations(payload: RecommendationRequest) -> dict[str, Any]:
    rec = STATE.get()
    results = rec.recommend(
        liked=payload.liked, disliked=payload.disliked,
        location=payload.location.model_dump() if payload.location else None,
        categories=payload.categories, top_n=payload.top_n, at=payload.at, visited=payload.visited,
    )
    return {"results": results, "behaviourModel": rec.model_meta}


@app.get("/search", summary="Search places by words in their name and description")
def search(
    q: str = Query(min_length=2, max_length=300),
    limit: int = Query(default=20, ge=1, le=100),
    category: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return STATE.get().text_search(q, limit=limit, categories=_csv(category))
