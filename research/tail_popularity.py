"""Task 3 - popularity of POIs without visits (item cold start), with the production code.

The catalog is split into 5 folds of POIs. For each fold, every visit to its POIs is
removed from training (BehaviourModel `unknown`), so the service has to predict their
popularity from content, as it does for POIs nobody photographed. Two measures:

1. Spearman between the predicted and the real log(1 + visitors) of the masked POIs
   (all of them, and only the visited ones).
2. hit@10 of the tourists' held-out visits that fall on a masked POI (the same
   leave-one-out cases as protocol.py; service settings: tau 0.3 km, next visit 1.0).

Feature sets of the popularity model: embedding + category (current), + Wikipedia
(pageview percentile, has article), + structured (links, heritage inventory, image,
generated description, description length), and Wikipedia / structured without the
embedding. Reference: the popularity the service uses without a behaviour model
(Wikipedia percentile), and the real counts (upper bound, nothing masked).

Usage: PYTHONPATH=research python research/tail_popularity.py <config> <lat0> <lat1> <lon0> <lon1>
"""
from __future__ import annotations

import sys

import numpy as np

from protocol import K, City
from pmv_turismo.behaviour import first_visits, train
from pmv_turismo.recommendation import Recommender

FOLDS = 5
FEATURE_SETS = {
    "embedding + category (current)": ("embedding", "category"),
    "+ wikipedia": ("embedding", "category", "wikipedia"),
    "+ wikipedia + structured": ("embedding", "category", "wikipedia", "structured"),
    "category + wikipedia + structured": ("category", "wikipedia", "structured"),
    "category + wikipedia": ("category", "wikipedia"),
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)  # noqa: E731
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if len(a) > 2 else float("nan")


def calibrate(pred: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Quantile mapping: the i-th smallest prediction takes the value of the same
    quantile of `reference` (the real log counts of the POIs the model did see)."""
    q = (np.argsort(np.argsort(pred, kind="stable"), kind="stable") + 0.5) / len(pred)
    return np.quantile(reference, q)


def hits(city: City, rec: Recommender, cases: list[tuple]) -> np.ndarray:
    rec.distance_scale_km, rec.similarity_weight, rec.next_weight = 0.3, 0.0, 1.0
    out = np.zeros(len(cases))
    for n, (_, liked, held, _, prev) in enumerate(cases):
        top = rec.recommend(liked=liked, visited=[city.ids[prev]], top_n=K)
        out[n] = city.ids[held] in {x["id"] for x in top}
    return out


def main() -> None:
    cfg_name, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    city = City(cfg_name, {"lat": (la0, la1), "lon": (lo0, lo1)})
    n = len(city.pois)
    visits = first_visits(city.photos, city.pois, city.shapes)
    true_pop = np.zeros(n)
    for v in visits.values():
        true_pop[list(v)] += 1
    true_pop = np.log1p(true_pop)
    cases = city.cases(list(city.visits))
    rng = np.random.default_rng(0)
    fold = rng.permutation(n) % FOLDS
    print(f"{cfg_name}: POIs {n}, visited {int((true_pop > 0).sum())}, cases {len(cases)}")

    full = Recommender(city.pois, city.index, train(city.pois, city.units, visits, "all"), city.cfg)
    wiki_pop = Recommender(city.pois, city.index, None, city.cfg).popularity

    pred = {name: np.zeros(n) for name in FEATURE_SETS}
    case_hits = {name: [] for name in FEATURE_SETS}
    case_hits["real counts, all users (optimistic bound)"] = []
    case_hits["Wikipedia percentile for masked POIs"] = []
    case_hits["+ wikipedia, calibrated"] = []
    for f in range(FOLDS):
        masked = np.flatnonzero(fold == f)
        fold_cases = [c for c in cases if fold[c[2]] == f]
        case_hits["real counts, all users (optimistic bound)"].append(hits(city, full, fold_cases))
        for name, feats in FEATURE_SETS.items():
            model = train(city.pois, city.units, visits, "masked", features=feats, unknown=masked)
            rec = Recommender(city.pois, city.index, model, city.cfg)
            pred[name][masked] = rec.popularity[masked]
            case_hits[name].append(hits(city, rec, fold_cases))
            if name == "+ wikipedia":
                seen = np.setdiff1d(np.arange(n), masked)
                rec.popularity = rec.popularity.copy()
                rec.popularity[masked] = calibrate(rec.popularity[masked], rec.popularity[seen])
                case_hits["+ wikipedia, calibrated"].append(hits(city, rec, fold_cases))
            if name == "embedding + category (current)":
                # same model, but the masked POIs take the no-model popularity instead
                # (Wikipedia percentile on a 0-3 scale)
                rec.popularity = rec.popularity.copy()
                rec.popularity[masked] = wiki_pop[masked]
                case_hits["Wikipedia percentile for masked POIs"].append(hits(city, rec, fold_cases))

    visited = true_pop > 0
    print(f"\n{'popularity of masked POIs':40s} spearman all  visited only")
    print(f"{'Wikipedia percentile (no model)':40s} {spearman(wiki_pop, true_pop):.3f}         "
          f"{spearman(wiki_pop[visited], true_pop[visited]):.3f}")
    for name, p in pred.items():
        print(f"{name:40s} {spearman(p, true_pop):.3f}         {spearman(p[visited], true_pop[visited]):.3f}")

    tail = set(np.argsort(-true_pop, kind="stable")[:20].tolist())
    ordered = [c for f in range(FOLDS) for c in cases if fold[c[2]] == f]
    is_tail = np.array([c[2] not in tail for c in ordered])
    base = np.concatenate(case_hits["embedding + category (current)"])
    print(f"\n{'hit@10, held-out visit on a masked POI':40s} all ({len(ordered)})  outside top 20 ({int(is_tail.sum())})"
          "  vs current: diff, McNemar p")
    from protocol import paired
    for name, h in case_hits.items():
        h = np.concatenate(h)
        t = paired(h, base)
        print(f"{name:40s} {h.mean():.3f}       {h[is_tail].mean():.3f}"
              f"                {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['mcnemar_p']:.2g}")


if __name__ == "__main__":
    main()
