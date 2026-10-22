from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pmv_turismo.common import (
    LOCAL_TZ,
    haversine_km,
    load_poi_list,
    now_local,
    opening_hours_cover_duration,
    poi_coordinates,
    processed_dir,
    write_json_atomic,
)
from pmv_turismo.embeddings import EmbeddingIndex, embeddings_path, encode_query, load_embedding_index

MOBILITY_SPEED_KMH = {"walking": 4.5, "transit": 20.0, "car": 35.0}
DEFAULT_STYLE = "explorer"


# --------------------------------------------------------------------------- #
# Recommendation styles
# --------------------------------------------------------------------------- #

def available_styles(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return config.get("recommendation", {}).get("styles", {}) or {}


def apply_style(config: Dict[str, Any], style: Optional[str]) -> Dict[str, Any]:
    """Return a copy of config with the weights of `style` applied."""
    styles = available_styles(config)
    styled = copy.deepcopy(config)
    weights = styles.get(str(style or DEFAULT_STYLE)) or styles.get(DEFAULT_STYLE) or {}
    styled.setdefault("recommendation", {}).update(weights)
    return styled


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #

def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def _minmax(x: np.ndarray) -> np.ndarray:
    span = float(np.max(x) - np.min(x)) if x.size else 0.0
    if span <= 1e-9:
        return np.full_like(x, 0.5)
    return (x - np.min(x)) / span


def _kmeans_centroids(
    x: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 50,
) -> np.ndarray:
    """K-Means with K-Means++ initialisation (no sklearn dependency)."""
    n_samples = x.shape[0]
    if n_samples == 0:
        return x
    if n_clusters <= 1:
        return np.mean(x, axis=0, keepdims=True)

    n_clusters = min(n_clusters, n_samples)
    rng = np.random.default_rng(random_state)

    centroid_indices = [int(rng.integers(0, n_samples))]
    for _ in range(1, n_clusters):
        selected = x[np.asarray(centroid_indices)]
        sq_dist = np.min(np.sum((x[:, None, :] - selected[None, :, :]) ** 2, axis=2), axis=1)
        sq_dist[centroid_indices] = 0.0
        total = float(np.sum(sq_dist))
        if total <= 0.0:
            available = [i for i in range(n_samples) if i not in centroid_indices]
            centroid_indices.append(int(rng.choice(available)))
            continue
        centroid_indices.append(int(rng.choice(n_samples, p=sq_dist / total)))

    centroids = x[np.asarray(centroid_indices)].copy()
    for _ in range(max_iter):
        distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(n_clusters):
            members = x[labels == cluster_id]
            if len(members) == 0:
                new_centroids[cluster_id] = x[int(np.argmax(np.min(distances, axis=1)))]
            else:
                new_centroids[cluster_id] = np.mean(members, axis=0)
        converged = np.allclose(new_centroids, centroids, atol=1e-6)
        centroids = new_centroids
        if converged:
            break
    return centroids


def build_interest_vectors(liked_vectors: np.ndarray, max_interests: int) -> np.ndarray:
    """Build several unit interest vectors to avoid average-embedding collapse."""
    if liked_vectors.ndim != 2 or liked_vectors.shape[0] == 0:
        return liked_vectors
    units = _unit_rows(liked_vectors)
    k = min(max_interests, units.shape[0])
    if k <= 1:
        return _unit_rows(np.mean(units, axis=0, keepdims=True))
    return _unit_rows(_kmeans_centroids(units, n_clusters=k))


def semantic_scores(
    cand_units: np.ndarray,
    interest_units: np.ndarray,
    dislike_units: Optional[np.ndarray],
    dislike_weight: float,
) -> np.ndarray:
    """Attraction to the closest interest minus repulsion from the closest dislike.

    Raw cosines are combined first and min-max scaled afterwards. Scaling each
    term to [0, 1] and clipping the difference (the previous approach) zeroed
    out about half of the catalog after a single dislike, because POI
    descriptions share a high baseline similarity.
    """
    raw = np.max(cand_units @ interest_units.T, axis=1)
    if dislike_units is not None and dislike_units.size:
        raw = raw - dislike_weight * np.max(cand_units @ dislike_units.T, axis=1)
    return _minmax(raw)


def greedy_diverse_selection(
    units: np.ndarray,
    quality: np.ndarray,
    k: int,
    theta: float,
) -> List[int]:
    """Maximal Marginal Relevance: theta * quality - (1 - theta) * max similarity."""
    n = units.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    sim = units @ units.T
    max_sim = np.full(n, -np.inf)
    available = np.ones(n, dtype=bool)
    selected: List[int] = []
    for _ in range(k):
        redundancy = np.where(np.isfinite(max_sim), max_sim, 0.0)
        mmr = theta * quality - (1.0 - theta) * redundancy
        mmr[~available] = -np.inf
        best = int(np.argmax(mmr))
        selected.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, sim[best])
    return selected


# --------------------------------------------------------------------------- #
# Onboarding (active learning)
# --------------------------------------------------------------------------- #

def _popularity(pois: List[Dict[str, Any]]) -> np.ndarray:
    """Popularity in [0, 1] from `relevance` (inverse popularity); 0.5 if missing."""
    return np.asarray(
        [1.0 - float(r) if isinstance(r := p.get("relevance"), (int, float)) else 0.5 for p in pois],
        dtype=np.float32,
    )


def preference_posterior(
    cand_units: np.ndarray,
    labelled_units: np.ndarray,
    labels: np.ndarray,
    bandwidth: float,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of P(like) per candidate from similarity-weighted feedback.

    Each answered POI adds a soft count to a Beta(prior/2, prior/2) on every
    candidate, weighted by exp((cos - 1) / bandwidth): near neighbours inherit
    the label, distant regions stay at 0.5 with high uncertainty.
    """
    w = np.exp((cand_units @ labelled_units.T - 1.0) / bandwidth)
    a = 0.5 * prior_strength + w @ labels
    b = 0.5 * prior_strength + w @ (1.0 - labels)
    n = a + b
    return a / n, np.sqrt(a * b / (n * n * (n + 1.0)))


def select_active_learning_pois(
    pois: List[Dict[str, Any]],
    index: EmbeddingIndex,
    k: int,
    liked_poi_ids: Optional[List[str]] = None,
    disliked_poi_ids: Optional[List[str]] = None,
    seen_poi_ids: Optional[List[str]] = None,
    bandwidth: float = 0.1,
    prior_strength: float = 2.0,
    exploration: float = 1.0,
    popularity_weight: float = 0.15,
    interest_unit: Optional[np.ndarray] = None,
    interest_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Select onboarding POIs by rounds.

    Round 1 (no feedback): representative (close to the rest of the pool) and
    well-known POIs, spread out with MMR so each answer covers a region.
    Later rounds: optimistic (UCB) ranking on a Bayesian estimate of P(like):
    liked regions are refined, unexplored ones probed and disliked ones
    dropped. A free-text interest (`interest_unit`, from `encode_query`) adds
    its min-max scaled similarity with weight `interest_weight` in every round.
    Ranking on affinity alone (the previous approach) kept showing
    neighbours of the first like, did worse than random in simulation and
    left users who disliked round 1 without any like.
    """
    if k <= 0 or not pois:
        return []

    seen = {str(x) for x in (seen_poi_ids or []) if x}
    candidates: List[Dict[str, Any]] = []
    cand_idx: List[int] = []
    for poi in pois:
        poi_id = str(poi.get("id") or "")
        if not poi_id or poi_id in seen:
            continue
        idx = index.index_of(poi_id)
        if idx is None:
            continue
        candidates.append(poi)
        cand_idx.append(idx)
    if not candidates:
        return []

    cand_units = index.units[np.asarray(cand_idx)]
    popularity = _popularity(candidates)
    # Query-document cosines live on a different scale than document pairs,
    # so the interest only contributes through its min-max scaled ranking.
    interest = _minmax(cand_units @ interest_unit) if interest_unit is not None else None
    liked_idx = index.indices_of([str(x) for x in (liked_poi_ids or []) if x])
    disliked_idx = index.indices_of([str(x) for x in (disliked_poi_ids or []) if x])

    if not liked_idx and not disliked_idx:
        representativeness = _minmax(np.mean(cand_units @ cand_units.T, axis=1))
        quality = 0.5 * representativeness + 0.5 * popularity
        if interest is not None:
            quality = (1.0 - interest_weight) * quality + interest_weight * interest
        picks = greedy_diverse_selection(cand_units, quality, k, theta=0.5)
        return [candidates[i] for i in picks]

    labelled = index.units[np.asarray(liked_idx + disliked_idx)]
    labels = np.asarray([1.0] * len(liked_idx) + [0.0] * len(disliked_idx), dtype=np.float32)
    mean, std = preference_posterior(cand_units, labelled, labels, bandwidth, prior_strength)
    score = mean + exploration * std + popularity_weight * popularity
    if interest is not None:
        score = score + interest_weight * interest
    picks = greedy_diverse_selection(cand_units, _minmax(score), k, theta=0.6)
    return [candidates[i] for i in sorted(picks, key=lambda i: -score[i])]


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def recommend(
    config: dict,
    poi_data: Optional[List[Dict[str, Any]]] = None,
    liked_poi_ids: Optional[List[str]] = None,
    disliked_poi_ids: Optional[List[str]] = None,
    user_location: Optional[Dict[str, float]] = None,
    user_mobility: str = "walking",
    requested_at: Optional[datetime] = None,
    include_liked: bool = True,
) -> Dict[str, Any]:
    """Hybrid recommendation (semantic + popularity + distance) with MMR.

    Pure function: it never writes to disk. Use `write_recommendations` to
    persist the result where the caller decides.
    """
    poi_items = poi_data if poi_data is not None else load_poi_list(processed_dir(config) / "poi.jsonld")

    rec_cfg = config.get("recommendation", {})
    top_n = int(rec_cfg.get("top_n", 30))
    profile = rec_cfg.get("user_profile", {})
    alpha = float(rec_cfg.get("alpha", 0.5))  # semantic weight
    beta = float(rec_cfg.get("beta", 0.3))  # quality weight
    gamma = float(rec_cfg.get("gamma", 0.2))  # geo weight
    preferred = {str(c).lower() for c in (profile or {}).get("preferred_categories") or [] if c}
    interests_text = str((profile or {}).get("interests") or "").strip()
    interest_text_weight = float(rec_cfg.get("interest_text_weight", 0.5)) if interests_text else 0.0
    # Bonus for the categories picked in onboarding; ignored without them.
    delta = float(rec_cfg.get("category_weight", 0.0)) if preferred else 0.0
    weight_sum = alpha + beta + gamma + delta
    if weight_sum > 0:
        alpha, beta, gamma, delta = (w / weight_sum for w in (alpha, beta, gamma, delta))
    lambda_geo = float(rec_cfg.get("lambda_geo", 0.5))  # distance sensitivity
    theta_mmr = float(rec_cfg.get("theta_mmr", 0.7))
    hidden_gems_bias = float(rec_cfg.get("hidden_gems_bias", 0.0))
    multi_interest_enabled = bool(rec_cfg.get("multi_interest_enabled", True))
    multi_interest_k = max(1, int(rec_cfg.get("multi_interest_k", 3)))
    dislike_weight = float(rec_cfg.get("dislike_weight", 0.5))
    check_hours = bool(rec_cfg.get("filter_opening_hours", True))

    liked_ids = [str(x) for x in (liked_poi_ids or rec_cfg.get("liked_pois") or []) if x]
    disliked_ids = [str(x) for x in (disliked_poi_ids or rec_cfg.get("disliked_pois") or []) if x]
    if not liked_ids and not preferred and not interests_text:
        raise ValueError("liked_poi_ids, preferred categories or interests are required to compute recommendations")
    liked_set, disliked_set = set(liked_ids), set(disliked_ids)
    visit_start = (requested_at or now_local()).astimezone(LOCAL_TZ).replace(second=0, microsecond=0)

    try:
        index = load_embedding_index(embeddings_path(config))
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc

    liked_idx = index.indices_of(liked_ids)
    if not liked_idx and not preferred and not interests_text:
        raise ValueError("No embeddings found for liked POIs; generate embeddings and retry")
    # Without likes (e.g. every onboarding card was rejected) the ranking falls
    # back to interests, preferred categories, popularity and distance, minus dislikes.
    interest_units: Optional[np.ndarray] = None
    if liked_idx:
        liked_units = index.units[np.asarray(liked_idx)]
        if multi_interest_enabled:
            interest_units = build_interest_vectors(liked_units, multi_interest_k)
        else:
            interest_units = _unit_rows(np.mean(liked_units, axis=0, keepdims=True))
    disliked_idx = index.indices_of(disliked_ids)
    dislike_units = index.units[np.asarray(disliked_idx)] if disliked_idx else None

    start_cfg = config.get("parameters", {}).get("planning", {}).get("start_location", {})
    loc = user_location or start_cfg or {}
    has_location = "lat" in loc and "lon" in loc
    speed_kmh = MOBILITY_SPEED_KMH.get(user_mobility, MOBILITY_SPEED_KMH["walking"])

    records: List[Dict[str, Any]] = []
    emb_rows: List[int] = []
    relevances: List[float] = []
    distances: List[float] = []
    in_preferred: List[float] = []
    for poi in poi_items:
        pid = str(poi.get("id") or "")
        if not pid or pid in disliked_set:
            continue
        is_liked = pid in liked_set
        if is_liked and not include_liked:
            continue
        emb_idx = index.index_of(pid)
        if emb_idx is None:
            continue

        d_km = 0.0
        coords = poi_coordinates(poi)
        if has_location and coords:
            d_km = haversine_km(float(coords[1]), float(coords[0]), float(loc["lat"]), float(loc["lon"]))
        if check_hours and not is_liked:
            # The POI must stay open for the trip there plus the whole visit.
            duration = float(poi.get("duration", 60) or 60)
            transit_minutes = d_km / speed_kmh * 60.0
            if not opening_hours_cover_duration(poi.get("openingHours"), visit_start, duration + transit_minutes):
                continue

        relevance = poi.get("relevance")
        records.append(poi)
        emb_rows.append(emb_idx)
        relevances.append(max(0.0, min(1.0, float(relevance))) if isinstance(relevance, (int, float)) else 0.5)
        distances.append(d_km)
        in_preferred.append(float(any(str(c).lower() in preferred for c in poi.get("category") or [])))

    output: Dict[str, Any] = {
        "profile": profile,
        "liked_pois": liked_ids,
        "disliked_pois": disliked_ids,
        "semantic_interest_count": int(interest_units.shape[0]) if interest_units is not None else 0,
        "results": [],
    }
    if not records:
        return output

    cand_units = index.units[np.asarray(emb_rows)]
    if interest_units is not None:
        s_sem = semantic_scores(cand_units, interest_units, dislike_units, dislike_weight)
    elif dislike_units is not None:
        s_sem = _minmax(-np.max(cand_units @ dislike_units.T, axis=1))
    else:
        s_sem = np.full(len(records), 0.5, dtype=np.float32)
    query_unit = encode_query(config, interests_text) if interest_text_weight > 0 else None
    if query_unit is not None:
        s_query = _minmax(cand_units @ query_unit)
        s_sem = (1.0 - interest_text_weight) * s_sem + interest_text_weight * s_query

    # `relevance` is an inverse popularity signal (high = less popular).
    # hidden_gems_bias blends between favouring popular (0) and hidden (1) POIs.
    rel = np.asarray(relevances, dtype=np.float32)
    s_qual = hidden_gems_bias * rel + (1.0 - hidden_gems_bias) * (1.0 - rel)

    geo = np.asarray(distances, dtype=np.float32)
    s_geo = 1.0 / (1.0 + lambda_geo * geo) if has_location else np.ones_like(s_sem)

    s_cat = np.asarray(in_preferred, dtype=np.float32)
    global_score = alpha * s_sem + beta * s_qual + gamma * s_geo + delta * s_cat
    order = greedy_diverse_selection(cand_units, global_score, top_n, theta_mmr)

    results: List[Dict[str, Any]] = []
    for rank, i in enumerate(order):
        enriched = dict(records[i])
        enriched["score"] = float(global_score[i])
        enriched["mmr_rank"] = rank + 1
        enriched["score_semantic"] = float(s_sem[i])
        enriched["score_quality"] = float(s_qual[i])
        enriched["score_geo"] = float(s_geo[i])
        enriched["score_category"] = float(s_cat[i])
        enriched["is_liked"] = str(records[i].get("id")) in liked_set
        # Exposed for the frontend, which reads `relevance` as "0 is best".
        enriched["relevance"] = float(1.0 - np.clip(global_score[i], 0.0, 1.0))
        results.append(enriched)

    # Liked POIs first, then MMR order (which already balances score and diversity).
    results.sort(key=lambda poi: (not poi["is_liked"], poi["mmr_rank"]))
    output["results"] = results
    return output


def write_recommendations(output: Dict[str, Any], path: Path) -> None:
    write_json_atomic(path, output)
