"""Experiment 3 - user cold start: simulated onboarding with real tourists.

Each evaluated tourist answers the onboarding cards (rounds from the config) with an
oracle: like if they really visited the POI, dislike otherwise. After onboarding the
system ranks the catalog; metric below. Protocol: the last two visits of each tourist are hidden and never shown as
cards (so every strategy is scored on the same POIs); the oracle likes a card only if it
is among the other visits. Metric = how many hidden visits reach the top 10. Training signals (popularity, adapted embeddings) come from other folds.
"""
import collections
import copy
import math

import numpy as np

from common import city_photos, distance_matrix_km, folds, load_catalog, load_config, minmax, visits
from loo_eval import adapted_embeddings, train_stats
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.recommendation import apply_style, recommend, select_active_learning_pois

K = 10


def main():
    cfg = load_config()
    pois, emb, radius = load_catalog(cfg)
    n = len(pois)
    ids = [p["id"] for p in pois]
    pos = {i: j for j, i in enumerate(ids)}
    index = load_embedding_index(embeddings_path(cfg))
    vis, tourists = visits(city_photos(), pois, radius)
    evaluated = sorted(u for u in tourists if len(vis.get(u, {})) >= 3)
    fold = folds(vis.keys())
    D = distance_matrix_km(pois)
    stats = {f: train_stats(vis, fold, f, n) for f in range(5)}
    Z = {f: adapted_embeddings(emb, *stats[f]) for f in range(5)}
    rounds = [int(x) for x in cfg.get("evaluation", {}).get("onboarding_rounds", [6, 4, 4])]
    rec = cfg["recommendation"]

    def pois_with_visit_pop(f):
        """Catalog copy whose `relevance` (inverse popularity) comes from training visits."""
        pop = stats[f][0]
        pct = np.argsort(np.argsort(pop)) / max(n - 1, 1)
        out = []
        for p, q in zip(pois, pct):
            p = dict(p)
            p["relevance"] = float(1 - q)
            out.append(p)
        return out

    def onboard(u, catalog, cards, hidden):
        liked, disliked, seen = [], [], []
        blocked = [ids[j] for j in hidden]
        for k in rounds:
            if cards == "popular":
                f = fold[u]
                order = [j for j in np.argsort(-stats[f][0]) if ids[j] not in seen and j not in hidden][:k]
                shown = [ids[j] for j in order]
            else:
                shown = [p["id"] for p in select_active_learning_pois(
                    catalog, index, k, liked_poi_ids=liked, disliked_poi_ids=disliked, seen_poi_ids=seen + blocked,
                    bandwidth=float(rec.get("active_learning_bandwidth", 0.1)),
                    prior_strength=float(rec.get("active_learning_prior_strength", 2.0)),
                    exploration=float(rec.get("active_learning_exploration", 1.0)),
                    popularity_weight=float(rec.get("active_learning_popularity_weight", 0.15)))]
            for pid in shown:
                seen.append(pid)
                (liked if pos[pid] in vis[u] and pos[pid] not in hidden else disliked).append(pid)
        return liked, disliked, seen

    base = copy.deepcopy(cfg)
    base["recommendation"].update({"top_n": 60, "filter_opening_hours": False})
    base.setdefault("parameters", {}).setdefault("planning", {})["start_location"] = {}
    explorer = apply_style(base, "explorer")
    explorer["recommendation"]["gamma"] = 0.0

    def rank_current(u, liked, disliked, seen):
        if not liked:
            return [j for j in np.argsort(-np.array([1 - (p.get("relevance") or 0.5) for p in pois])) if ids[j] not in seen]
        out = recommend(explorer, poi_data=pois, liked_poi_ids=liked, disliked_poi_ids=disliked, include_liked=False)
        return [pos[x["id"]] for x in out["results"] if x["id"] not in seen]

    def rank_new(u, liked, disliked, seen, semantic=True):
        f = fold[u]
        s = np.log1p(stats[f][0])
        if liked:
            L = [pos[x] for x in liked]
            s = minmax(s - D[:, L].min(axis=1) / 0.3)
            if semantic:
                s = s + 0.3 * minmax((Z[f] @ Z[f][L].T).mean(axis=1))
        if disliked and semantic:
            Dl = [pos[x] for x in disliked]
            s = s - 0.1 * minmax((Z[f] @ Z[f][Dl].T).max(axis=1))
        return [j for j in np.argsort(-s, kind="stable") if ids[j] not in seen]

    strategies = {
        "no onboarding: popular (visits)": None,
        "current: bandit cards -> current explorer": ("bandit_wiki", rank_current),
        "popular cards -> gravity": ("popular", lambda *a: rank_new(*a, semantic=False)),
        "popular cards -> gravity + adapted semantic": ("popular", rank_new),
        "bandit cards (visit pop) -> gravity": ("bandit_visits", lambda *a: rank_new(*a, semantic=False)),
        "bandit cards (visit pop) -> gravity + adapted semantic": ("bandit_visits", rank_new),
    }
    res = collections.defaultdict(list)
    likes = collections.defaultdict(list)
    for u in evaluated:
        f = fold[u]
        hidden = set(sorted(vis[u], key=vis[u].get)[-2:])
        cat_visits = pois_with_visit_pop(f)
        for name, strat in strategies.items():
            if strat is None:
                liked, disliked, seen = [], [], []
                order = list(np.argsort(-stats[f][0]))
            else:
                cards, ranker = strat
                catalog = {"bandit_wiki": pois, "bandit_visits": cat_visits}.get(cards, pois)
                liked, disliked, seen = onboard(u, catalog, "popular" if cards == "popular" else "bandit", hidden)
                order = ranker(u, liked, disliked, seen)
            likes[name].append(len(liked))
            res[name].append(len(hidden & set(order[:K])) / len(hidden))
    print(f"evaluated tourists {len(evaluated)}; onboarding rounds {rounds} ({sum(rounds)} cards)")
    print(f"{'strategy':52s} hidden@10  users  mean likes in onboarding")
    for name in strategies:
        r = res[name]
        print(f"{name:52s} {np.mean(r):.3f}      {len(r):4d}   {np.mean(likes[name]):.2f}")


if __name__ == "__main__":
    main()
