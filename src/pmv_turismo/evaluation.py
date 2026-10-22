"""Offline evaluation: leave-one-out over liked POIs.

For every profile with at least two likes, each like is held out in turn, the
recommender is run with the remaining likes and we check where the held-out
POI lands. Methods compared:

- the configured recommendation styles (explorer, conservative, ...);
- `semantic_mean`: cosine to the average liked vector, no popularity/geo/MMR;
- `semantic_multi`: same, but with multi-interest vectors (ablation);
- `popularity` and `random` baselines.

`simulate_onboarding` replays the onboarding rounds with simulated users whose
taste is hidden (a set of categories or 1-3 regions of the embedding space)
and swipes with 10% noise, then scores the recommendations built from the
collected likes/dislikes (precision@k on unseen POIs) against random rounds.

Profiles are the stored user profiles (if any) plus deterministic synthetic
profiles built from catalog categories. Synthetic likes are category-driven
and POI embeddings include the category text, so absolute numbers favour
semantic methods; use them to compare variants, not as product metrics.
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from pmv_turismo.common import haversine_km, load_poi_list, poi_coordinates, processed_dir, write_json_atomic
from pmv_turismo.embeddings import EmbeddingIndex, embeddings_path, load_embedding_index
from pmv_turismo.recommendation import apply_style, available_styles, recommend, select_active_learning_pois


def synthetic_profiles(pois: List[Dict[str, Any]], n_users: int, seed: int) -> Dict[str, Dict[str, Any]]:
    """Users with 3 likes in a main category, 1 in a secondary one and 2 dislikes."""
    by_category: Dict[str, List[str]] = defaultdict(list)
    for poi in pois:
        cats = poi.get("category") or []
        if cats and poi.get("id"):
            by_category[str(cats[0])].append(str(poi["id"]))
    main_categories = sorted(c for c, ids in by_category.items() if len(ids) >= 3)
    if len(main_categories) < 3:
        return {}

    rng = random.Random(seed)
    profiles: Dict[str, Dict[str, Any]] = {}
    for i in range(n_users):
        main = main_categories[i % len(main_categories)]
        secondary, disliked = rng.sample([c for c in main_categories if c != main], 2)
        profiles[f"synthetic_{i:03d}"] = {
            "liked_pois": rng.sample(by_category[main], 3) + rng.sample(by_category[secondary], 1),
            "disliked_pois": rng.sample(by_category[disliked], 2),
        }
    return profiles


def _real_profiles(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {uid: p for uid, p in data.items() if len([x for x in p.get("liked_pois", []) if x]) >= 2}


def _ild(ids: List[str], index: EmbeddingIndex) -> float:
    rows = index.indices_of(ids)
    if len(rows) < 2:
        return 0.0
    units = index.units[np.asarray(rows)]
    sim = units @ units.T
    n = len(rows)
    return float(1.0 - (sim.sum() - n) / (n * (n - 1)))


def _hidden_tastes(
    pois: List[Dict[str, Any]], index: EmbeddingIndex, n_users: int, seed: int
) -> List[Dict[str, Any]]:
    """Users alternate between category tastes and embedding-region tastes."""
    rng = random.Random(seed)
    ids = [str(p["id"]) for p in pois if index.index_of(str(p.get("id") or "")) is not None]
    category = {str(p["id"]): str((p.get("category") or ["unknown"])[0]) for p in pois}
    units = index.units[np.asarray(index.indices_of(ids))]
    counts: Dict[str, int] = defaultdict(int)
    for pid in ids:
        counts[category[pid]] += 1
    big = sorted(c for c, n in counts.items() if n >= 4)
    users: List[Dict[str, Any]] = []
    for i in range(n_users):
        if i % 2 == 0 and len(big) >= 2:
            tastes = set(rng.sample(big, rng.choice([1, 2])))
            truth = {pid for pid in ids if category[pid] in tastes}
            others = [c for c in big if c not in tastes]
            selected = sorted(tastes | set(rng.sample(others, min(len(others), rng.choice([0, 1, 2])))))
        else:
            anchors = rng.sample(range(len(ids)), min(len(ids), rng.choice([1, 2, 3])))
            sim = np.max(units @ units[anchors].T, axis=1)
            truth = {ids[j] for j in np.flatnonzero(sim >= np.quantile(sim, 0.8))}
            # Half of them pick the categories of their taste, the rest skip the filter.
            selected = sorted({category[pid] for pid in truth}) if rng.random() < 0.5 else []
        users.append({"truth": truth, "categories": selected})
    return users


def simulate_onboarding(
    config: dict,
    pois: List[Dict[str, Any]],
    index: EmbeddingIndex,
    rounds: List[int],
    n_users: int,
    seed: int,
    k: int,
    noise: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """Replay the onboarding rounds with simulated users and score the outcome."""
    rec_cfg = config.get("recommendation", {})
    al_kwargs = {
        "bandwidth": float(rec_cfg.get("active_learning_bandwidth", 0.1)),
        "prior_strength": float(rec_cfg.get("active_learning_prior_strength", 2.0)),
        "exploration": float(rec_cfg.get("active_learning_exploration", 1.0)),
        "popularity_weight": float(rec_cfg.get("active_learning_popularity_weight", 0.15)),
    }
    style_cfg = apply_style(config, rec_cfg.get("style"))
    users = _hidden_tastes(pois, index, n_users, seed)

    def active(cands: List[Dict[str, Any]], size: int, liked, disliked, seen, rng) -> List[Dict[str, Any]]:
        return select_active_learning_pois(cands, index, size, liked, disliked, seen, **al_kwargs)

    def random_rounds(cands: List[Dict[str, Any]], size: int, liked, disliked, seen, rng) -> List[Dict[str, Any]]:
        unseen = [p for p in cands if str(p["id"]) not in set(seen)]
        return rng.sample(unseen, min(size, len(unseen)))

    report: Dict[str, Dict[str, float]] = {}
    for name, selector in (("active_learning", active), ("random", random_rounds)):
        rng = random.Random(seed)
        precision, likes, no_likes = [], [], 0
        for user in users:
            truth, cats = user["truth"], set(user["categories"])
            cands = [p for p in pois if not cats or str((p.get("category") or [""])[0]) in cats]
            liked: List[str] = []
            disliked: List[str] = []
            seen: List[str] = []
            for size in rounds:
                for poi in selector(cands, size, liked, disliked, seen, rng):
                    pid = str(poi["id"])
                    seen.append(pid)
                    (liked if (pid in truth) != (rng.random() < noise) else disliked).append(pid)
            likes.append(len(liked))
            if not liked:
                no_likes += 1
                precision.append(0.0)
                continue
            result = recommend(style_cfg, poi_data=pois, liked_poi_ids=liked, disliked_poi_ids=disliked,
                               include_liked=False)
            ranked = [str(r["id"]) for r in result["results"] if str(r["id"]) not in set(seen)][:k]
            precision.append(float(np.mean([pid in truth for pid in ranked])) if ranked else 0.0)
        report[name] = {
            f"precision@{k}": round(float(np.mean(precision)), 4),
            "likes": round(float(np.mean(likes)), 3),
            "no_likes": round(no_likes / max(1, len(users)), 4),
        }
    return report


def run_evaluation(config: dict) -> Dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    k = int(eval_cfg.get("k", 10))
    seed = int(eval_cfg.get("seed", 42))
    n_synthetic = int(eval_cfg.get("synthetic_users", 40))
    onboarding_rounds = [int(x) for x in eval_cfg.get("onboarding_rounds", [6, 4, 4])]
    out_dir = processed_dir(config)

    pois = load_poi_list(out_dir / "poi.jsonld")
    poi_by_id = {str(p.get("id")): p for p in pois}
    index = load_embedding_index(embeddings_path(config))

    profiles = {**_real_profiles(out_dir / "user_profiles.json"), **synthetic_profiles(pois, n_synthetic, seed)}
    if not profiles:
        raise ValueError("No profiles to evaluate (need user_profiles.json or enough catalog categories)")

    # Accuracy must not depend on the clock: disable opening-hours filtering.
    base = copy.deepcopy(config)
    rec_cfg = base.setdefault("recommendation", {})
    rec_cfg.update({"top_n": k, "filter_opening_hours": False})
    default_start = base.get("parameters", {}).get("planning", {}).get("start_location")

    def model(cfg: Dict[str, Any]) -> Callable[..., List[str]]:
        def run(liked: List[str], disliked: List[str], location: Any) -> List[str]:
            result = recommend(
                cfg, poi_data=pois, liked_poi_ids=liked, disliked_poi_ids=disliked,
                user_location=location, include_liked=False,
            )
            return [str(item.get("id")) for item in result["results"]]
        return run

    def semantic_only(multi: bool) -> Dict[str, Any]:
        cfg = copy.deepcopy(base)
        cfg["recommendation"].update({
            "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "theta_mmr": 1.0,
            "dislike_weight": 0.0, "multi_interest_enabled": multi,
        })
        return cfg

    rng = random.Random(seed)

    def popularity(liked: List[str], disliked: List[str], location: Any) -> List[str]:
        excluded = set(liked) | set(disliked)
        ranked = sorted(
            (p for p in pois if str(p.get("id")) not in excluded),
            key=lambda p: float(p.get("relevance", 0.5)),
        )
        return [str(p.get("id")) for p in ranked[:k]]

    def random_baseline(liked: List[str], disliked: List[str], location: Any) -> List[str]:
        excluded = set(liked) | set(disliked)
        ids = [str(p.get("id")) for p in pois if str(p.get("id")) not in excluded]
        return rng.sample(ids, min(k, len(ids)))

    methods: Dict[str, Callable[..., List[str]]] = {
        f"style:{name}": model(apply_style(base, name)) for name in available_styles(base)
    }
    methods.update({
        "semantic_mean": model(semantic_only(multi=False)),
        "semantic_multi": model(semantic_only(multi=True)),
        "popularity": popularity,
        "random": random_baseline,
    })

    per_method: Dict[str, Dict[str, List[float]]] = {m: defaultdict(list) for m in methods}
    recommended_items: Dict[str, set] = {m: set() for m in methods}
    for profile in profiles.values():
        liked = [str(x) for x in profile.get("liked_pois", []) if x]
        disliked = [str(x) for x in profile.get("disliked_pois", []) if x]
        location = profile.get("start_location") or default_start
        for held_out in liked:
            train = [x for x in liked if x != held_out]
            for name, run in methods.items():
                ids = run(train, disliked, location)
                rank = ids.index(held_out) if held_out in ids else None
                metrics = per_method[name]
                metrics["hit"].append(1.0 if rank is not None else 0.0)
                metrics["ndcg"].append(1.0 / np.log2(rank + 2) if rank is not None else 0.0)
                metrics["ild"].append(_ild(ids, index))
                metrics["novelty"].append(float(np.mean([
                    float(poi_by_id[i].get("relevance", 0.5)) for i in ids if i in poi_by_id
                ] or [0.0])))
                metrics["dislike_leak"].append(1.0 if set(ids) & set(disliked) else 0.0)
                if location and "lat" in location:
                    dists = [
                        haversine_km(c[1], c[0], float(location["lat"]), float(location["lon"]))
                        for i in ids if i in poi_by_id and (c := poi_coordinates(poi_by_id[i]))
                    ]
                    if dists:
                        metrics["distance_km"].append(float(np.mean(dists)))
                recommended_items[name].update(ids)

    summary = {
        name: {
            f"hit@{k}": round(float(np.mean(m["hit"])), 4),
            f"ndcg@{k}": round(float(np.mean(m["ndcg"])), 4),
            "ild": round(float(np.mean(m["ild"])), 4),
            "novelty": round(float(np.mean(m["novelty"])), 4),
            "coverage": round(len(recommended_items[name]) / max(1, len(pois)), 4),
            "dislike_leak": round(float(np.mean(m["dislike_leak"])), 4),
            "distance_km": round(float(np.mean(m["distance_km"])), 3) if m["distance_km"] else None,
        }
        for name, m in per_method.items()
    }
    report = {
        "k": k,
        "seed": seed,
        "profiles": len(profiles),
        "synthetic_profiles": sum(1 for uid in profiles if uid.startswith("synthetic_")),
        "evaluations": len(per_method[next(iter(per_method))]["hit"]),
        "metrics": summary,
        "onboarding_rounds": onboarding_rounds,
        "onboarding": simulate_onboarding(
            base, pois, index, onboarding_rounds, n_synthetic, seed, k
        ),
    }
    write_json_atomic(out_dir / "evaluation.json", report)
    (out_dir / "evaluation_report.txt").write_text(format_report(report), encoding="utf-8")
    return report


def format_report(report: Dict[str, Any]) -> str:
    k = report["k"]
    columns = [f"hit@{k}", f"ndcg@{k}", "ild", "novelty", "coverage", "dislike_leak", "distance_km"]
    lines = [
        f"Leave-one-out evaluation: {report['evaluations']} held-out likes, "
        f"{report['profiles']} profiles ({report['synthetic_profiles']} synthetic), k={k}",
        "",
        f"{'method':<22}" + "".join(f"{c:>14}" for c in columns),
    ]
    for name, metrics in report["metrics"].items():
        cells = ["-" if metrics[c] is None else f"{metrics[c]:.3f}" for c in columns]
        lines.append(f"{name:<22}" + "".join(f"{c:>14}" for c in cells))
    if report.get("onboarding"):
        onboarding_columns = [f"precision@{k}", "likes", "no_likes"]
        lines += [
            "",
            f"Onboarding simulation: rounds {report['onboarding_rounds']}, 10% swipe noise",
            "",
            f"{'method':<22}" + "".join(f"{c:>14}" for c in onboarding_columns),
        ]
        for name, metrics in report["onboarding"].items():
            lines.append(f"{name:<22}" + "".join(f"{metrics[c]:>14.3f}" for c in onboarding_columns))
    return "\n".join(lines) + "\n"
