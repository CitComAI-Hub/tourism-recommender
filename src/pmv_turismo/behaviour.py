"""Behaviour model: what real visitors do, learnt offline from geotagged photos.

Input: photo records {"user", "lat", "lon", "ts"} from any source (Flickr YFCC100M,
Wikimedia Commons...). Each photo is assigned to the nearest catalog POI within that
POI's radius, so a photo between two POIs counts once; a user "visited" a POI if they
have at least one photo there.

Output (one .npz file, aggregates only, no user data):
- `visitors`: distinct visitors per POI (aligned with `ids`).
- `pop_model`: ridge regression from the POI embedding (PCA) + category to
  log(1 + visitors). It predicts the popularity of POIs nobody photographed: the long
  tail and new POIs of the same city (a model of another city does not transfer;
  the Recommender ignores it). Measured on Valencia: Spearman 0.52 with real
  visitors vs 0.26 for Wikipedia pageviews (research/item_cold_start.py).
- `projection`: ridge projection W of the embeddings onto the item factors of the
  co-visit matrix. Similarity in E @ W means "visited by the same people" instead of
  "described with similar words" (hit@10 0.29 vs 0.09, research/loo_eval.py).
- `transitions`: P(next POI | current POI) from the order of each visitor's first
  photo at every POI (needs `ts` in the photos). Where to go after a place is the
  strongest signal after popularity and proximity: +0.05 hit@10 in Valencia,
  +0.08 in Milano (research/protocol.py).
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

AREA_CATEGORIES = {"park", "garden", "beach", "nature_reserve", "zoo", "attraction", "marketplace"}
POINT_RADIUS_KM = 0.040
AREA_RADIUS_KM = 0.150
PCA_COMPONENTS = 16
FACTORS = 16
RIDGE = 1.0


def model_path(config: dict) -> Path:
    import os

    from pmv_turismo.common import processed_dir

    configured = os.environ.get("BEHAVIOUR_MODEL_PATH") or config.get("behaviour", {}).get("model_path")
    return Path(configured or processed_dir(config) / "behaviour_model.npz")


def poi_radius_km(poi: dict[str, Any]) -> float:
    return AREA_RADIUS_KM if set(poi.get("category") or []) & AREA_CATEGORIES else POINT_RADIUS_KM


def assign_visits(photos: Iterable[dict[str, Any]], pois: list[dict[str, Any]],
                  shapes: dict[str, Any] | None = None) -> dict[str, set[int]]:
    """user -> indices of the POIs they photographed (see `first_visits`)."""
    return {u: set(v) for u, v in first_visits(photos, pois, shapes).items()}


def first_visits(photos: Iterable[dict[str, Any]], pois: list[dict[str, Any]],
                 shapes: dict[str, Any] | None = None) -> dict[str, dict[int, str]]:
    """user -> {POI index: time of their first photo there ("" without `ts`)}.

    With `shapes` (see geometry.py) a photo is a visit to every POI whose real
    OSM shape it falls in (or near, for points and lines). Without them, to the nearest
    POI within a fixed radius (150 m for large areas), which mis-assigns photos
    of busy squares to small POIs next to them.
    """
    photos = list(photos)
    if not photos:
        return {}
    lat = np.array([float(p["lat"]) for p in photos])
    lon = np.array([float(p["lon"]) for p in photos])
    users = [str(p["user"]) for p in photos]
    times = [str(p.get("ts") or "")[:19] for p in photos]
    visits: dict[str, dict[int, str]] = {}

    def add(k: int, j: int) -> None:
        seen = visits.setdefault(users[k], {})
        seen[j] = min(seen.get(j, times[k]), times[k])

    if shapes:
        from pmv_turismo.geometry import assign_photos

        for j, photo_idx in enumerate(assign_photos(pois, shapes, lat, lon)):
            for k in photo_idx:
                add(int(k), j)
        return visits
    plat = np.array([p["location"]["coordinates"][1] for p in pois], dtype=float)
    plon = np.array([p["location"]["coordinates"][0] for p in pois], dtype=float)
    radius = np.array([poi_radius_km(p) for p in pois])
    target = np.full(len(photos), -1)
    for k in range(len(photos)):
        dx = np.radians(plon - lon[k]) * math.cos(math.radians(lat[k]))
        dk = 6371.0 * np.sqrt(dx * dx + np.radians(plat - lat[k]) ** 2)
        inside = dk <= radius
        if inside.any():
            target[k] = int(np.argmin(np.where(inside, dk, np.inf)))
    for k, j in enumerate(target):
        if j >= 0:
            add(k, int(j))
    return visits


def _categories(pois: list[dict[str, Any]], vocabulary: list[str]) -> np.ndarray:
    return np.array([[1.0 if c in (p.get("category") or []) else 0.0 for c in vocabulary] for p in pois])


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)


@dataclass
class BehaviourModel:
    ids: list[str]
    visitors: np.ndarray
    emb_mean: np.ndarray
    pca: np.ndarray            # (components, dim)
    vocabulary: list[str]
    pop_mean_x: np.ndarray
    pop_mean_y: float
    pop_weights: np.ndarray
    projection: np.ndarray     # (dim, factors)
    transitions: np.ndarray    # (n, n) P(next | current), aligned with ids
    meta: dict[str, Any]

    def predicted_popularity(self, pois: list[dict[str, Any]], units: np.ndarray) -> np.ndarray:
        """log(1 + visitors) predicted from content, for POIs without visits."""
        x = np.hstack([(units - self.emb_mean) @ self.pca.T, _categories(pois, self.vocabulary)])
        return (x - self.pop_mean_x) @ self.pop_weights + self.pop_mean_y

    def adapted(self, units: np.ndarray) -> np.ndarray:
        z = units @ self.projection
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, ids=np.array(self.ids), visitors=self.visitors, emb_mean=self.emb_mean, pca=self.pca,
            vocabulary=np.array(self.vocabulary), pop_mean_x=self.pop_mean_x, pop_mean_y=np.array(self.pop_mean_y),
            pop_weights=self.pop_weights, projection=self.projection,
            transitions=self.transitions.astype(np.float32), meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: Path) -> BehaviourModel:
        with np.load(path, allow_pickle=False) as d:
            return cls(
                ids=[str(x) for x in d["ids"]], visitors=d["visitors"], emb_mean=d["emb_mean"], pca=d["pca"],
                vocabulary=[str(x) for x in d["vocabulary"]], pop_mean_x=d["pop_mean_x"],
                pop_mean_y=float(d["pop_mean_y"]), pop_weights=d["pop_weights"], projection=d["projection"],
                transitions=d["transitions"].astype(float), meta=json.loads(str(d["meta"])),
            )


def transition_matrix(visits: dict[str, dict[int, str]], n: int) -> np.ndarray:
    """P(next | current) from each user's POIs ordered by their first visit."""
    counts = np.zeros((n, n))
    for seen in visits.values():
        if not all(seen.values()):  # no timestamps: no order
            continue
        seq = sorted(seen, key=lambda j: (seen[j], j))
        for a, b in zip(seq, seq[1:], strict=False):
            counts[a, b] += 1
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)


def train(pois: list[dict[str, Any]], units: np.ndarray, visits: dict[str, dict[int, str]],
          source: str) -> BehaviourModel:
    """Fit the popularity model, the projection and the transitions on one city's
    visits (`first_visits`)."""
    n = len(pois)
    visitors = np.zeros(n)
    co = np.zeros((n, n))
    for seen in visits.values():
        idx = sorted(seen)
        visitors[idx] += 1
        for a in idx:
            co[a, idx] += 1
    np.fill_diagonal(co, 0.0)
    if not visitors.any():
        raise ValueError("No photo falls on any catalog POI; check the photos' area")

    emb_mean = units.mean(axis=0)
    _, _, vt = np.linalg.svd(units - emb_mean, full_matrices=False)
    pca = vt[:PCA_COMPONENTS]
    vocabulary = sorted({c for p in pois for c in p.get("category") or []})
    x = np.hstack([(units - emb_mean) @ pca.T, _categories(pois, vocabulary)])
    y = np.log1p(visitors)
    mean_x, mean_y = x.mean(axis=0), float(y.mean())
    pop_weights = _ridge(x - mean_x, y - mean_y, RIDGE)

    similarity = co / (np.sqrt(np.outer(visitors, visitors)) + 1e-9)
    eigval, eigvec = np.linalg.eigh(similarity)
    top = np.argsort(-eigval)[:FACTORS]
    factors = eigvec[:, top] * np.sqrt(np.clip(eigval[top], 0.0, None))
    seen = visitors > 0
    projection = _ridge(units[seen], factors[seen], RIDGE)

    transitions = transition_matrix(visits, n)
    meta = {"source": source, "users": len(visits), "visited_pois": int(seen.sum()), "pois": n,
            "transitions": int((transitions > 0).sum())}
    logger.info("Behaviour model: %s", meta)
    return BehaviourModel([str(p["id"]) for p in pois], visitors, emb_mean, pca, vocabulary,
                          mean_x, mean_y, pop_weights, projection, transitions, meta)


def load_model(config: dict) -> BehaviourModel | None:
    path = model_path(config)
    if not path.exists():
        return None
    try:
        return BehaviourModel.load(path)
    except Exception as exc:  # a broken artifact must not take the service down
        logger.warning("Ignoring behaviour model %s: %s", path, exc)
        return None


def read_photos(path: Path) -> list[dict[str, Any]]:
    """Photo records from a JSON Lines file (optionally gzip): user, lat, lon[, ts]."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_from_config(config: dict) -> None:
    """CLI step `behaviour`: train on `behaviour.photos_path` and save the model."""
    from pmv_turismo.common import load_poi_list, processed_dir
    from pmv_turismo.embeddings import embeddings_path, load_embedding_index

    cfg = config.get("behaviour", {})
    photos_path = cfg.get("photos_path")
    if not photos_path or not Path(photos_path).exists():
        print(f"behaviour: skipped (photos file not found: {photos_path})")
        return
    pois = load_poi_list(processed_dir(config) / "poi.jsonld")
    index = load_embedding_index(embeddings_path(config))
    keep = [i for i, p in enumerate(pois) if index.index_of(str(p["id"])) is not None]
    pois = [pois[i] for i in keep]
    units = index.units[[index.index_of(str(p["id"])) for p in pois]]
    from pmv_turismo.geometry import load_geometries

    shapes = load_geometries(config)
    if not shapes:
        logger.warning("No POI geometries (run `--step geometry`): falling back to fixed radii")
    visits = first_visits(read_photos(Path(photos_path)), pois, shapes)
    model = train(pois, units, visits, str(cfg.get("source", photos_path)))
    model.save(model_path(config))
    print(f"behaviour: model saved to {model_path(config)} ({model.meta})")
