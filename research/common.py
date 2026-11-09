"""Shared helpers for the behaviour-based evaluation (Flickr YFCC100M visits to catalog POIs)."""
from __future__ import annotations

import collections
import datetime as dt
import glob
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from pmv_turismo.common import load_poi_list
from pmv_turismo.config import Config
from pmv_turismo.embeddings import embeddings_path, load_embedding_index

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "yfcc"
BIG = {"park", "garden", "beach", "nature_reserve", "zoo", "attraction", "marketplace"}
VALENCIA = {"lat": (39.40, 39.53), "lon": (-0.44, -0.27)}
TOURIST_MAX_DAYS = 21


def load_config() -> dict:
    return Config.load(REPO / "configs" / "base.yaml").data


def load_catalog(cfg: dict):
    """POIs, their unit embeddings (same order) and the assignment radius in km."""
    pois = load_poi_list(REPO / "data" / "processed" / "poi.jsonld")
    index = load_embedding_index(embeddings_path(cfg))
    emb = index.units[[index.index_of(p["id"]) for p in pois]]
    radius = np.array([0.150 if set(p.get("category", [])) & BIG else 0.040 for p in pois])
    return pois, emb, radius


def city_photos(bbox: dict = VALENCIA, min_accuracy: int = 14) -> list[dict]:
    """Street-level photos inside bbox, cached as one gzip file per city."""
    cache = RAW / f"city_{bbox['lat'][0]}_{bbox['lon'][0]}.jsonl.gz"
    if not cache.exists():
        rows = []
        for path in sorted(glob.glob(str(RAW / "chunks" / "*.jsonl.gz"))):
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    la, lo = float(r["lat"]), float(r["lon"])
                    if bbox["lat"][0] <= la <= bbox["lat"][1] and bbox["lon"][0] <= lo <= bbox["lon"][1]:
                        rows.append(r)
        with gzip.open(cache, "wt") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(cache, "rt") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if isinstance(r.get("accuracy"), int) and r["accuracy"] >= min_accuracy
            and str(r.get("datetaken", "")).startswith("20")]


def pseudonym(uid: str) -> str:
    return hashlib.sha256(("rp-salt:" + uid).encode()).hexdigest()[:12]


def visits(photos: list[dict], pois: list[dict], radius: np.ndarray):
    """user -> {poi index: first datetaken}, and the set of tourist users.

    Each photo goes to the nearest POI within that POI's radius, so a photo
    between two POIs is counted once. A tourist has all photos in the city
    within TOURIST_MAX_DAYS days (De Choudhury et al. 2010 style heuristic).
    """
    lat = np.array([p["location"]["coordinates"][1] for p in pois])
    lon = np.array([p["location"]["coordinates"][0] for p in pois])
    out: dict[str, dict[int, str]] = collections.defaultdict(dict)
    dates: dict[str, list[dt.datetime]] = collections.defaultdict(list)
    for r in photos:
        u = pseudonym(r["uid"])
        try:
            dates[u].append(dt.datetime.fromisoformat(r["datetaken"][:19]))
        except ValueError:
            continue
        fl, fo = float(r["lat"]), float(r["lon"])
        dk = 6371 * np.sqrt((np.radians(lon - fo) * math.cos(math.radians(fl))) ** 2 + np.radians(lat - fl) ** 2)
        ok = dk <= radius
        if ok.any():
            j = int(np.argmin(np.where(ok, dk, 1e9)))
            out[u][j] = min(out[u].get(j, r["datetaken"]), r["datetaken"])
    tourists = {u for u, d in dates.items() if (max(d) - min(d)).days <= TOURIST_MAX_DAYS}
    return dict(out), tourists


def folds(users, k: int = 5, seed: int = 0) -> dict[str, int]:
    uids = sorted(users)
    np.random.default_rng(seed).shuffle(uids)
    return {u: n % k for n, u in enumerate(uids)}


def minmax(x: np.ndarray) -> np.ndarray:
    span = float(np.ptp(x))
    return (x - x.min()) / span if span > 1e-12 else np.full_like(x, 0.5, dtype=float)


def distance_matrix_km(pois: list[dict]) -> np.ndarray:
    lat = np.radians([p["location"]["coordinates"][1] for p in pois])
    lon = np.radians([p["location"]["coordinates"][0] for p in pois])
    return 6371 * np.sqrt(((lon[:, None] - lon[None]) * math.cos(float(np.mean(lat)))) ** 2 + (lat[:, None] - lat[None]) ** 2)
