"""Recommendation: popularity x proximity, personalised with behaviour-adapted semantics.

Every choice here was measured on real tourist visits (Flickr YFCC100M, Valencia;
see research/ and docs/proposal.md):

- Tourists mostly go where others go: popularity learnt from real visits ranks a
  held-out visit in the top 10 for 54% of cases, the previous embedding-only
  ranking for 9-12%.
- Beyond the famous places, the strongest personal signal is proximity to what the
  user liked ("gravity": log popularity - distance / 0.3 km).
- Semantics is what works where there are no visits: the popularity of an unvisited
  POI is predicted from its description (BehaviourModel.predicted_popularity), and
  similarity is computed on embeddings adapted to co-visits.
- Onboarding shows the most popular places first: real tourists recognise and like
  them, so the first likes arrive quickly (recall 0.52 vs 0.23 for the previous
  exploration-first bandit).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np

from pmv_turismo.behaviour import BehaviourModel
from pmv_turismo.common import LOCAL_TZ, now_local, opening_hours_cover_duration
from pmv_turismo.embeddings import EmbeddingIndex

logger = logging.getLogger(__name__)

WALKING_KMH = 4.5
# Share of the catalog the behaviour model must have been trained on to be used.
MIN_MODEL_OVERLAP = 0.5
EARTH_KM = 6371.0


def _minmax(x: np.ndarray) -> np.ndarray:
    span = float(np.ptp(x)) if x.size else 0.0
    return (x - x.min()) / span if span > 1e-12 else np.full_like(x, 0.5, dtype=float)


def _fold(text: str) -> str:
    """Lowercase without accents, for matching typed text."""
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def _rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(len(x) - 1, 1)


class Recommender:
    """Ranks one city's catalog. Stateless: callers pass the user's feedback."""

    def __init__(
        self,
        pois: list[dict[str, Any]],
        index: EmbeddingIndex,
        model: BehaviourModel | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        rec = (config or {}).get("recommendation", {})
        self.distance_scale_km = float(rec.get("distance_scale_km", 0.3))
        self.similarity_weight = float(rec.get("similarity_weight", 0.3))
        self.dislike_weight = float(rec.get("dislike_weight", 0.1))
        self.top_n = int(rec.get("top_n", 30))
        self.check_hours = bool(rec.get("filter_opening_hours", True))

        self.pois = [p for p in pois if p.get("id") and index.index_of(str(p["id"])) is not None]
        self.ids = [str(p["id"]) for p in self.pois]
        self.position = {pid: i for i, pid in enumerate(self.ids)}
        self.units = index.units[[index.index_of(pid) for pid in self.ids]] if self.ids else np.zeros((0, 1))
        coords = np.array([p["location"]["coordinates"] for p in self.pois], dtype=float).reshape(-1, 2)
        self.lat, self.lon = np.radians(coords[:, 1]), np.radians(coords[:, 0])
        if model is not None and self.ids:
            known = len(set(model.ids) & set(self.ids)) / len(self.ids)
            if known < MIN_MODEL_OVERLAP:
                # A model of another city does not transfer (Milano test: hit@10 0.25
                # with Valencia's model vs 0.30 without, 0.66 with its own photos).
                logger.warning("Behaviour model covers %.0f%% of this catalog: ignored", 100 * known)
                model = None
        self.popularity, self.popularity_source = self._popularity(model)
        self.adapted = model.adapted(self.units) if model is not None and self.ids else self.units
        self.model_meta = model.meta if model is not None else None

    # ------------------------------------------------------------------ signals

    def _popularity(self, model: BehaviourModel | None) -> tuple[np.ndarray, list[str]]:
        """log(1 + visitors) per POI and where each value comes from.

        Visited POIs use their real count; the rest the count predicted from their
        content. Without a behaviour model, Wikipedia pageviews (`relevance` is the
        inverse popularity percentile) are put on a comparable 0-3 scale.
        """
        n = len(self.ids)
        if model is None or n == 0:
            wiki = np.array([1.0 - float(p["relevance"]) if isinstance(p.get("relevance"), (int, float)) else 0.5
                             for p in self.pois])
            return 3.0 * _rank01(wiki), ["wikipedia"] * n
        predicted = np.clip(model.predicted_popularity(self.pois, self.units), 0.0, None)
        visitors = dict(zip(model.ids, model.visitors, strict=False))
        pop, source = predicted.copy(), ["predicted"] * n
        for i, pid in enumerate(self.ids):
            if visitors.get(pid, 0) > 0:
                pop[i], source[i] = float(np.log1p(visitors[pid])), "visits"
        return pop, source

    def _distances_km(self, lat: float, lon: float) -> np.ndarray:
        la, lo = np.radians(lat), np.radians(lon)
        dx = (self.lon - lo) * np.cos((self.lat + la) / 2)
        return EARTH_KM * np.sqrt(dx * dx + (self.lat - la) ** 2)

    def _distances_from(self, i: int) -> np.ndarray:
        return self._distances_km(float(np.degrees(self.lat[i])), float(np.degrees(self.lon[i])))

    def _indices(self, ids: Sequence[str] | None) -> list[int]:
        return [self.position[str(x)] for x in ids or [] if str(x) in self.position]

    def _in_categories(self, categories: Sequence[str] | None) -> np.ndarray:
        wanted = {str(c).lower() for c in categories or [] if c}
        if not wanted:
            return np.ones(len(self.ids), dtype=bool)
        return np.array([bool(wanted & {str(c).lower() for c in p.get("category") or []}) for p in self.pois])

    # ------------------------------------------------------------------ API

    def onboarding_cards(self, k: int, seen: Sequence[str] | None = None,
                         categories: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """The k most popular POIs not shown yet, optionally within some categories."""
        mask = self._in_categories(categories)
        mask[self._indices(seen)] = False
        order = [i for i in np.argsort(-self.popularity, kind="stable") if mask[i]]
        return [self.pois[i] for i in order[:k]]

    def recommend(
        self,
        liked: Sequence[str] | None = None,
        disliked: Sequence[str] | None = None,
        location: dict[str, float] | None = None,
        categories: Sequence[str] | None = None,
        top_n: int | None = None,
        at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        n = len(self.ids)
        if n == 0:
            return []
        liked_idx, disliked_idx = self._indices(liked), self._indices(disliked)

        # Proximity: distance to the nearest liked POI or to the user's position.
        anchors = [self._distances_from(i) for i in liked_idx]
        if location and "lat" in location and "lon" in location:
            anchors.append(self._distances_km(float(location["lat"]), float(location["lon"])))
        nearest_km = np.min(anchors, axis=0) if anchors else np.zeros(n)
        score = _minmax(self.popularity - nearest_km / self.distance_scale_km)

        similarity = np.zeros(n)
        if liked_idx:
            similarity = (self.adapted @ self.adapted[liked_idx].T).mean(axis=1)
            score = score + self.similarity_weight * _minmax(similarity)
        if disliked_idx:
            score = score - self.dislike_weight * _minmax((self.adapted @ self.adapted[disliked_idx].T).max(axis=1))

        candidates = self._in_categories(categories)
        candidates[liked_idx + disliked_idx] = False
        if self.check_hours:
            start = (at or now_local()).astimezone(LOCAL_TZ).replace(second=0, microsecond=0)
            for i in np.flatnonzero(candidates):
                minutes = float(self.pois[i].get("duration", 60) or 60) + nearest_km[i] / WALKING_KMH * 60.0
                if not opening_hours_cover_duration(self.pois[i].get("openingHours"), start, minutes):
                    candidates[i] = False

        order = [i for i in np.argsort(-score, kind="stable") if candidates[i]][: top_n or self.top_n]
        top_popular = set(np.argsort(-self.popularity)[: max(10, n // 20)].tolist())
        similar_cut = float(np.quantile(similarity, 0.9)) if liked_idx else None
        results = []
        for rank, i in enumerate(order, start=1):
            item = dict(self.pois[i])
            item.update({
                "rank": rank,
                "score": round(float(score[i]), 4),
                "popularity": round(float(self.popularity[i]), 3),
                "popularitySource": self.popularity_source[i],
                "distanceKm": round(float(nearest_km[i]), 3) if anchors else None,
                "reasons": self._reasons(i, liked_idx, nearest_km, bool(anchors), similar_cut, top_popular),
            })
            results.append(item)
        return results

    def _reasons(self, i: int, liked_idx: list[int], nearest_km: np.ndarray, has_anchor: bool,
                 similar_cut: float | None, top_popular: set) -> list[dict[str, Any]]:
        """Why a POI is in the list, for the explanation shown to the user."""
        reasons: list[dict[str, Any]] = []
        if i in top_popular:
            reasons.append({"type": "popular"})
        if has_anchor and nearest_km[i] <= 0.5:
            near = [j for j in liked_idx if self._distances_from(j)[i] <= 0.5]
            reasons.append({"type": "near", "km": round(float(nearest_km[i]), 2),
                            "poi": self.pois[near[0]].get("name") if near else None})
        if liked_idx and similar_cut is not None:
            j = max(liked_idx, key=lambda j: float(self.adapted[i] @ self.adapted[j]))
            if float((self.adapted[i] @ self.adapted[liked_idx].T).mean()) >= similar_cut:
                reasons.append({"type": "similar", "poi": self.pois[j].get("name")})
        return reasons

    def search(self, query_unit: np.ndarray | None, limit: int = 20,
               categories: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Free-text search over the descriptions (plain embeddings, not the adapted ones)."""
        if query_unit is None or not self.ids:
            return []
        score = self.units @ query_unit
        mask = self._in_categories(categories)
        order = [i for i in np.argsort(-score) if mask[i]][:limit]
        return [{**self.pois[i], "rank": r, "score": round(float(score[i]), 4)} for r, i in enumerate(order, start=1)]

    def text_search(self, query: str, limit: int = 20,
                    categories: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Word search over name and description, when the text model is not installed.

        Each query word (3+ letters, accents ignored) found in the name counts 3,
        in the description 1; ties go to the more popular POI.
        """
        words = [w for w in re.findall(r"\w+", _fold(query)) if len(w) >= 3]
        if not words or not self.ids:
            return []
        mask = self._in_categories(categories)
        scored = []
        for i, poi in enumerate(self.pois):
            if not mask[i]:
                continue
            name, desc = _fold(str(poi.get("name") or "")), _fold(str(poi.get("description") or ""))
            score = sum(3 * (w in name) + (w in desc) for w in words)
            if score:
                scored.append((score, float(self.popularity[i]), i))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [{**self.pois[i], "rank": r, "score": float(sc)} for r, (sc, _, i) in enumerate(scored[:limit], start=1)]
