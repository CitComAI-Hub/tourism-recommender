from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pmv_turismo.common import load_poi_list, poi_coordinates, processed_dir, write_json_atomic
from pmv_turismo.config import Config
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.ingestion import run_ingestion
from pmv_turismo.planning import plan_itinerary
from pmv_turismo.recommendation import apply_style, recommend, select_active_learning_pois

CONFIG_PATH = Path(os.environ.get("PMV_CONFIG", "configs/base.yaml"))
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = FastAPI(title="PMV IA Turismo API")

# Comma-separated list of allowed origins; "*" is fine for local development
# because the frontend never sends credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("PMV_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serialises read-modify-write cycles on user files (single-process server).
_USER_DATA_LOCK = threading.Lock()


class UserProfile(BaseModel):
    name: str = Field(default="tourist")
    preferred_categories: list[str] = Field(default_factory=list)
    recommendation_style: str = Field(default="explorer")


class UserPreferences(BaseModel):
    user_id: str
    profile: UserProfile


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    profile: UserProfile | None = None
    liked_pois: Optional[List[str]] = None
    disliked_pois: Optional[List[str]] = None
    user_location: Optional[Dict[str, float]] = None
    user_mobility: str = "walking"
    requested_at: Optional[datetime] = None


class RecommendationGenerateRequest(RecommendationRequest):
    top_n: Optional[int] = Field(default=None, ge=1, le=500)
    category: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=1000)


class PlanRequest(BaseModel):
    user_id: Optional[str] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    max_stops_per_day: Optional[int] = Field(default=None, ge=1, le=50)
    max_total_minutes: Optional[int] = Field(default=None, ge=30, le=1440)
    visit_time_minutes: Optional[int] = Field(default=None, ge=5, le=480)
    max_candidates: Optional[int] = Field(default=None, ge=1, le=200)
    start_location: Optional[Dict[str, float]] = None
    start_at: Optional[datetime] = None


class PlannerPreferences(BaseModel):
    max_stops_per_day: int = Field(default=5, ge=1, le=50)
    max_total_minutes: int = Field(default=480, ge=30, le=1440)
    visit_time_minutes: int = Field(default=60, ge=5, le=480)
    start_location: Optional[Dict[str, float]] = None
    use_current_location: bool = True
    mobility: str = "walking"


class OnboardingSelections(BaseModel):
    user_id: str
    profile: UserProfile
    liked_pois: list[str] = Field(default_factory=list)
    disliked_pois: list[str] = Field(default_factory=list)
    start_location: Optional[Dict[str, float]] = None
    use_current_location: bool = True
    planner_preferences: PlannerPreferences


# --------------------------------------------------------------------------- #
# Config, paths and storage helpers
# --------------------------------------------------------------------------- #

def _config() -> Dict[str, Any]:
    """Fresh config per request (cheap) so edits to the YAML apply without restart."""
    return Config.load(CONFIG_PATH).data


def _processed_dir() -> Path:
    return processed_dir(_config())


def _profiles_path() -> Path:
    return _processed_dir() / "user_profiles.json"


def _validate_user_id(user_id: str) -> str:
    if not USER_ID_PATTERN.match(user_id):
        raise HTTPException(status_code=422, detail="Invalid user_id (allowed: letters, digits, '_' and '-')")
    return user_id


def _user_file(user_id: str, filename: str) -> Path:
    """Path of a per-user file. Never creates directories: writes do that."""
    return _processed_dir() / "users" / _validate_user_id(user_id) / filename


def _output_file(user_id: Optional[str], filename: str) -> Path:
    return _user_file(user_id, filename) if user_id else _processed_dir() / filename


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_optional(path: Path) -> Any:
    try:
        return _read_json(path)
    except HTTPException:
        return None


def _read_user_profiles() -> Dict[str, Any]:
    return _read_json_optional(_profiles_path()) or {}


def _update_user_profile(user_id: str, update: Dict[str, Any], replace: bool = False) -> None:
    with _USER_DATA_LOCK:
        profiles = _read_user_profiles()
        current = {} if replace else dict(profiles.get(user_id) or {})
        current.update(update)
        profiles[user_id] = current
        write_json_atomic(_profiles_path(), profiles)


def _load_pois() -> List[Dict[str, Any]]:
    try:
        return load_poi_list(_processed_dir() / "poi.jsonld")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="POI catalog not found; run ingestion first")


def _normalize_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": data.get("name", "tourist"),
        "preferred_categories": data.get("preferred_categories", []),
        "recommendation_style": data.get("recommendation_style", "explorer"),
    }


def _resolve_profile(user_id: Optional[str], profile: Optional[UserProfile]) -> Dict[str, Any]:
    if profile is not None:
        return _normalize_profile(profile.model_dump())
    if user_id:
        stored = _read_user_profiles().get(_validate_user_id(user_id))
        if not stored:
            raise HTTPException(status_code=404, detail="User profile not found")
        return _normalize_profile(stored)
    raise HTTPException(status_code=400, detail="Provide profile or user_id")


def _id_list(user_id: Optional[str], provided: Optional[List[str]], filename: str) -> Optional[List[str]]:
    if provided is not None:
        return [str(x) for x in provided if x]
    if user_id:
        data = _read_json_optional(_user_file(user_id, filename))
        if isinstance(data, list):
            return [str(x) for x in data if x]
    return None


def _load_liked_pois(user_id: Optional[str], provided: Optional[List[str]]) -> List[str]:
    likes = [str(x) for x in (provided or []) if x]
    if not likes and user_id:
        likes = _id_list(user_id, None, "liked_pois.json")
    if not likes:
        raise HTTPException(status_code=422, detail="liked_pois required to generate recommendations")
    return likes


def _load_user_location(user_id: Optional[str]) -> Optional[Dict[str, float]]:
    if not user_id:
        return None
    location_data = _read_json_optional(_user_file(user_id, "start_location.json"))
    if not isinstance(location_data, dict):
        return None
    location = location_data.get("start_location")
    if not isinstance(location, dict) or "lat" not in location or "lon" not in location:
        return None
    return {"lat": float(location["lat"]), "lon": float(location["lon"])}


def _filter_by_category(items: List[Dict[str, Any]], category: Optional[str]) -> List[Dict[str, Any]]:
    categories = [c.strip().lower() for c in category.split(",") if c.strip()] if category else []
    if not categories:
        return items
    return [
        item for item in items
        if any(c in categories for c in (str(x).lower() for x in (item.get("category") or [])))
    ]


def _validate_location(loc: Optional[Dict[str, float]]) -> None:
    if loc is None:
        return
    lat, lon = loc.get("lat"), loc.get("lon")
    if lat is None or lon is None:
        raise HTTPException(status_code=422, detail="start_location must include lat and lon")
    if not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
        raise HTTPException(status_code=422, detail="start_location out of bounds")


def _run_recommendation(payload: RecommendationRequest, top_n: Optional[int] = None) -> Dict[str, Any]:
    if payload.user_id:
        _validate_user_id(payload.user_id)
    profile = _resolve_profile(payload.user_id, payload.profile)
    config = apply_style(_config(), profile.get("recommendation_style"))
    rec_cfg = config.setdefault("recommendation", {})
    rec_cfg["user_profile"] = profile
    if top_n is not None:
        rec_cfg["top_n"] = top_n
    try:
        return recommend(
            config,
            liked_poi_ids=_load_liked_pois(payload.user_id, payload.liked_pois),
            disliked_poi_ids=_id_list(payload.user_id, payload.disliked_pois, "disliked_pois.json") or [],
            user_location=payload.user_location or _load_user_location(payload.user_id),
            user_mobility=payload.user_mobility,
            requested_at=payload.requested_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/embedding/{poi_id}", summary="Return embedding vector for a POI id")
def get_embedding(poi_id: str) -> Dict[str, Any]:
    try:
        index = load_embedding_index(embeddings_path(_config()))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Embeddings file not found; run embeddings step first")
    resolved = index.resolve(poi_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Embedding for POI not found")
    resolved_id, vec = resolved
    return {"id": resolved_id, "embedding": [float(x) for x in vec], "dimension": int(vec.shape[0])}


@app.get("/poi")
def get_poi(
    category: str | None = Query(default=None, description="Comma-separated categories"),
    source: str | None = Query(default=None, description="Source filter"),
    limit: int | None = Query(default=None, ge=1, le=1000),
    min_lat: float | None = None,
    min_lon: float | None = None,
    max_lat: float | None = None,
    max_lon: float | None = None,
) -> Any:
    bbox = None if None in (min_lat, min_lon, max_lat, max_lon) else (min_lat, min_lon, max_lat, max_lon)

    def in_bbox(item: dict) -> bool:
        if bbox is None:
            return True
        coords = poi_coordinates(item)
        return bool(coords) and bbox[0] <= coords[1] <= bbox[2] and bbox[1] <= coords[0] <= bbox[3]

    filtered = [
        item for item in _filter_by_category(_load_pois(), category)
        if (not source or item.get("source") == source) and in_bbox(item)
    ]
    return filtered[:limit] if limit else filtered


@app.get("/poi/sample", summary="Return POIs for the next onboarding round (active learning)")
def get_poi_sample(
    categories: str | None = Query(default=None, description="Comma-separated categories"),
    limit: int = Query(default=12, ge=1, le=200),
    liked_pois: str | None = Query(default=None, description="Comma-separated liked POI ids"),
    disliked_pois: str | None = Query(default=None, description="Comma-separated disliked POI ids"),
    seen_pois: str | None = Query(default=None, description="Comma-separated already shown POI ids"),
) -> Any:
    def parse_csv_ids(raw: str | None) -> List[str]:
        return [x.strip() for x in raw.split(",") if x.strip()] if raw else []

    filtered = _filter_by_category(_load_pois(), categories)
    if not filtered:
        return []

    config = _config()
    rec_cfg = config.get("recommendation", {})
    try:
        index = load_embedding_index(embeddings_path(config))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Embeddings are required to generate active-learning POI samples",
        ) from exc
    return select_active_learning_pois(
        pois=filtered,
        index=index,
        k=min(limit, len(filtered)),
        liked_poi_ids=parse_csv_ids(liked_pois),
        disliked_poi_ids=parse_csv_ids(disliked_pois),
        seen_poi_ids=parse_csv_ids(seen_pois),
        bandwidth=float(rec_cfg.get("active_learning_bandwidth", 0.1)),
        prior_strength=float(rec_cfg.get("active_learning_prior_strength", 2.0)),
        exploration=float(rec_cfg.get("active_learning_exploration", 1.0)),
        popularity_weight=float(rec_cfg.get("active_learning_popularity_weight", 0.15)),
    )


@app.get("/recommendations")
def get_recommendations(
    category: str | None = Query(default=None, description="Comma-separated categories"),
    limit: int | None = Query(default=None, ge=1, le=1000),
    user_id: str | None = Query(default=None, description="If provided, read user's recommendations"),
) -> Any:
    data = _read_json(_output_file(user_id, "recommendations.jsonld"))
    if not isinstance(data, dict) or "results" not in data:
        return data
    results = _filter_by_category(data.get("results", []), category)
    return {"profile": data.get("profile"), "results": results[:limit] if limit else results}


@app.get("/itinerary")
def get_itinerary(
    limit: int | None = Query(default=None, ge=1, le=1000),
    user_id: str | None = Query(default=None, description="If provided, read user's itinerary"),
) -> Any:
    data = _read_json(_output_file(user_id, "itinerary.jsonld"))
    if not isinstance(data, dict) or "stops" not in data:
        return data
    stops = data.get("stops", [])
    return {
        "startLocation": data.get("startLocation"),
        "totalMinutes": data.get("totalMinutes"),
        "totalDistanceKm": data.get("totalDistanceKm"),
        "stops": stops[:limit] if limit else stops,
    }


@app.post("/run/ingestion")
def run_ingestion_endpoint() -> dict:
    run_ingestion(_config())
    return {"status": "ok"}


@app.post("/run/recommendations")
def run_recommendations_endpoint(payload: RecommendationRequest) -> Any:
    output = _run_recommendation(payload)
    write_json_atomic(_output_file(payload.user_id, "recommendations.jsonld"), output)
    return output


@app.post("/recommendations", summary="Generate recommendations for a user or inline profile")
def generate_recommendations(payload: RecommendationGenerateRequest) -> Any:
    output = _run_recommendation(payload, top_n=payload.top_n)
    write_json_atomic(_output_file(payload.user_id, "recommendations.jsonld"), output)
    results = _filter_by_category(output.get("results", []), payload.category)
    output["results"] = results[: payload.limit] if payload.limit else results
    return output


def _enrich_candidates(candidates: List[Dict[str, Any]], user_id: Optional[str]) -> List[Dict[str, Any]]:
    """Complete lightweight candidates (id + location) with catalog data and scores.

    The frontend only sends id/name/location/category; the planner also needs
    duration, opening hours and the recommendation score.
    """
    try:
        catalog = {str(p.get("id")): p for p in _load_pois()}
    except HTTPException:
        catalog = {}
    scores: Dict[str, float] = {}
    recs = _read_json_optional(_output_file(user_id, "recommendations.jsonld"))
    if isinstance(recs, dict):
        scores = {str(r.get("id")): r.get("score") for r in recs.get("results", []) if r.get("id")}

    enriched = []
    for cand in candidates:
        cid = str(cand.get("id") or "")
        merged = {**catalog.get(cid, {}), **cand}
        if "score" not in cand and isinstance(scores.get(cid), (int, float)):
            merged["score"] = scores[cid]
        coords = poi_coordinates(merged)
        try:
            valid = coords is not None and -90 <= float(coords[1]) <= 90 and -180 <= float(coords[0]) <= 180
        except (TypeError, ValueError):
            valid = False
        if valid:
            enriched.append(merged)
    return enriched


@app.post("/itinerary", summary="Generate itinerary for a user or provided candidates")
def generate_itinerary(payload: PlanRequest) -> Any:
    if payload.user_id:
        _validate_user_id(payload.user_id)
    config = _config()
    planning_cfg = config.setdefault("parameters", {}).setdefault("planning", {})
    for key in ("max_stops_per_day", "max_total_minutes", "max_candidates"):
        value = getattr(payload, key)
        if value is not None:
            planning_cfg[key] = value
    if payload.start_location is not None:
        _validate_location(payload.start_location)
        planning_cfg["start_location"] = payload.start_location

    candidates = payload.candidates
    if candidates is None:
        rec_data = _read_json(_output_file(payload.user_id, "recommendations.jsonld"))
        if isinstance(rec_data, dict) and "results" in rec_data:
            candidates = rec_data.get("results", [])
        elif isinstance(rec_data, list):
            candidates = rec_data
        else:
            raise HTTPException(status_code=400, detail="Invalid recommendations format")

    candidates = _enrich_candidates(candidates, payload.user_id)
    if not candidates:
        raise HTTPException(status_code=422, detail="No valid candidates with coordinates")

    try:
        output = plan_itinerary(
            config,
            candidates=candidates,
            visit_time_override=payload.visit_time_minutes,
            start_at=payload.start_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    write_json_atomic(_output_file(payload.user_id, "itinerary.jsonld"), output)
    return output


@app.post("/user/preferences")
def set_user_preferences(payload: UserPreferences) -> Dict[str, str]:
    _validate_user_id(payload.user_id)
    _update_user_profile(payload.user_id, payload.profile.model_dump(), replace=True)
    return {"status": "ok"}


@app.post("/user/onboarding")
def set_user_onboarding(payload: OnboardingSelections) -> Dict[str, str]:
    user_id = _validate_user_id(payload.user_id)
    planner_prefs = payload.planner_preferences.model_dump()
    _update_user_profile(
        user_id,
        {
            **payload.profile.model_dump(),
            "liked_pois": payload.liked_pois,
            "disliked_pois": payload.disliked_pois,
            "start_location": payload.start_location,
            "use_current_location": payload.use_current_location,
            "planner_preferences": planner_prefs,
        },
    )
    with _USER_DATA_LOCK:
        write_json_atomic(_user_file(user_id, "liked_pois.json"), payload.liked_pois)
        write_json_atomic(_user_file(user_id, "disliked_pois.json"), payload.disliked_pois)
        write_json_atomic(_user_file(user_id, "planner_preferences.json"), planner_prefs)
        write_json_atomic(
            _user_file(user_id, "start_location.json"),
            {"start_location": payload.start_location, "use_current_location": payload.use_current_location},
        )
    return {"status": "ok"}


@app.get("/user/preferences")
def get_user_preferences(user_id: str = Query(...)) -> Any:
    profile = _read_user_profiles().get(_validate_user_id(user_id))
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    return {"user_id": user_id, "profile": profile}
