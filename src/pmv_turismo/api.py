"""HTTP API of the tourism recommender.

Stateless: the client keeps the user's likes, dislikes and position and sends them
with each request, so the server stores nothing about users. Behind the data space
connector the user is already authenticated by APISIX.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from pmv_turismo.behaviour import load_model, model_path
from pmv_turismo.common import load_poi_list, poi_coordinates, processed_dir
from pmv_turismo.config import Config
from pmv_turismo.embeddings import embeddings_path, encode_query, load_embedding_index
from pmv_turismo.recommendation import Recommender

CONFIG_PATH = Path(os.environ.get("PMV_CONFIG", "configs/base.yaml"))
MAX_IDS = 200


def _config() -> Dict[str, Any]:
    return Config.load(CONFIG_PATH).data


class _State:
    """Recommender built from the files on disk, rebuilt when any of them changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: Optional[tuple] = None
        self._recommender: Optional[Recommender] = None

    def get(self) -> Recommender:
        config = _config()
        paths = [processed_dir(config) / "poi.jsonld", embeddings_path(config), model_path(config)]
        key = tuple(p.stat().st_mtime if p.exists() else None for p in paths)
        with self._lock:
            if self._recommender is None or key != self._key:
                try:
                    pois = load_poi_list(paths[0])
                    index = load_embedding_index(paths[1])
                except FileNotFoundError as exc:
                    raise HTTPException(status_code=503, detail=f"Catalog not ready: {exc}") from exc
                self._recommender = Recommender(pois, index, load_model(config), config)
                self._key = key
            return self._recommender


STATE = _State()


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Loading the text model takes seconds; warm it up so the first search is fast.
    def warm_up() -> None:
        try:
            encode_query(_config(), "warm up")
        except Exception as exc:
            print(f"Query model warm-up failed: {exc}")

    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title="Tourism recommender", lifespan=_lifespan)
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
    liked: List[str] = Field(default_factory=list, max_length=MAX_IDS)
    disliked: List[str] = Field(default_factory=list, max_length=MAX_IDS)
    categories: List[str] = Field(default_factory=list, max_length=50)


class CardsRequest(Feedback):
    seen: List[str] = Field(default_factory=list, max_length=MAX_IDS)
    limit: int = Field(default=6, ge=1, le=50)


class RecommendationRequest(Feedback):
    location: Optional[Location] = None
    top_n: int = Field(default=30, ge=1, le=200)
    at: Optional[datetime] = None

    @field_validator("at")
    @classmethod
    def _aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and v.tzinfo is None:
            raise ValueError("`at` must include a timezone")
        return v


def _csv(raw: Optional[str]) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()] if raw else []


@app.get("/health")
def health() -> Dict[str, Any]:
    rec = STATE.get()
    return {"status": "ok", "pois": len(rec.ids), "behaviourModel": rec.model_meta}


@app.get("/pois")
def list_pois(
    category: Optional[str] = Query(default=None, description="Comma-separated categories"),
    limit: Optional[int] = Query(default=None, ge=1, le=2000),
    min_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
) -> List[Dict[str, Any]]:
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
def get_poi(poi_id: str) -> Dict[str, Any]:
    rec = STATE.get()
    if poi_id not in rec.position:
        raise HTTPException(status_code=404, detail="POI not found")
    return rec.pois[rec.position[poi_id]]


@app.post("/onboarding/cards", summary="Next onboarding cards: the most popular places not shown yet")
def onboarding_cards(payload: CardsRequest) -> List[Dict[str, Any]]:
    return STATE.get().onboarding_cards(payload.limit, seen=payload.seen + payload.liked + payload.disliked,
                                        categories=payload.categories)


@app.post("/recommendations")
def recommendations(payload: RecommendationRequest) -> Dict[str, Any]:
    rec = STATE.get()
    results = rec.recommend(
        liked=payload.liked, disliked=payload.disliked,
        location=payload.location.model_dump() if payload.location else None,
        categories=payload.categories, top_n=payload.top_n, at=payload.at,
    )
    return {"results": results, "behaviourModel": rec.model_meta}


@app.get("/search", summary="Free-text search over the POI descriptions")
def search(
    q: str = Query(min_length=2, max_length=300),
    limit: int = Query(default=20, ge=1, le=100),
    category: Optional[str] = Query(default=None),
) -> List[Dict[str, Any]]:
    query = encode_query(_config(), q)
    if query is None:
        raise HTTPException(status_code=503, detail="Text search needs the embeddings extra installed")
    return STATE.get().search(query, limit=limit, categories=_csv(category))
