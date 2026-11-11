"""Ideas C and I: one ranking for every city instead of weights tuned per city.

Today the distance weight is chosen per city (Valencia 1-1.5, Milano 2-4), which
is tuning on each city's Flickr data. Part of the difference is only the scale
of the popularity (log visitor counts grow with the number of photos). Here:

I  fixed formula chosen on the OTHER city (leave-city-out): score =
   z(popularity) + next_w * log(1 + 100 P(next | last)) - dist_w * log(km + 0.05),
   with popularity z-scored within the city so the weights mean the same in
   every city;
C  learnt linear ranker (pairwise logistic regression, held-out visit vs 30
   sampled candidates) on the same signals plus the context ones of
   research/context.py (hour, weekend, trip length lifts; Wikipedia; distance
   from the last visit), trained on the OTHER city.

Both against the per-city formula with weights chosen on the city's own
validation fold (`own validation`, = research/context.py base). Users split,
5 folds; the other city's model and cases use all its users.

Usage: PYTHONPATH=research python research/one_config.py
"""
from __future__ import annotations

import collections
import datetime as dt
import itertools
import math

import numpy as np
from context import GRID_DIST, GRID_NEXT, Data, cases, fold_name, hour_bucket, run, top10, trip_bucket, weekend
from external_check import CITIES
from protocol import City, paired
from sklearn.linear_model import LogisticRegression

GRID_Z_DIST = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
GRID_Z_NEXT = (0.0, 0.1, 0.25, 0.5)
FEATURES = ("z popularity", "next", "-log distance", "-log distance from last", "hour lift", "weekend lift",
            "trip lift", "wikipedia")


def prepare(cfg, bbox) -> City:
    city = City(cfg, bbox)
    lat = np.radians([p["location"]["coordinates"][1] for p in city.pois])
    lon = np.radians([p["location"]["coordinates"][0] for p in city.pois])
    city.dist = 6371 * np.sqrt(((lon[:, None] - lon[None]) * np.cos((lat[:, None] + lat[None]) / 2)) ** 2
                               + (lat[:, None] - lat[None]) ** 2)
    city.names = [fold_name(str(p.get("name") or p["id"])) for p in city.pois]
    city.trip_days = {u: (dt.date.fromisoformat(max(p["ts"] for p in ps)[:10])
                          - dt.date.fromisoformat(min(p["ts"] for p in ps)[:10])).days
                      for u, ps in city.by_user.items()}
    return city


def feats(city: City, d: Data, c: dict) -> np.ndarray:
    near = city.dist[sorted(set(c["liked"]))].min(axis=0)
    last = c["before"][-1]
    z = (d.pop - d.pop.mean()) / (d.pop.std() + 1e-9)
    return np.stack([z, np.log1p(100 * d.t1[last]), -np.log(near + 0.05), -np.log(city.dist[last] + 0.05),
                     d.lift_hour[hour_bucket(c["now"])], d.lift_weekend[weekend(c["now"])],
                     d.lift_trip[trip_bucket(c["trip"])], d.wiki / (d.pop.std() + 1e-9)], axis=1)


def z_formula(city, d, cs, w) -> np.ndarray:
    return np.array([c["held"] in top10(city, feats(city, d, c)[:, :3] @ np.array([1.0, w[1], w[0]]), c)
                     for c in cs], dtype=float)


def learnt(city, d, cs, coef) -> np.ndarray:
    return np.array([c["held"] in top10(city, feats(city, d, c) @ coef, c) for c in cs], dtype=float)


def out_of_fold(city: City) -> list[tuple[Data, list[dict]]]:
    """(model of the other folds, cases of this fold) x 5: no held-out visit is in
    the popularity or the transitions its features are computed with."""
    users = sorted(city.visits)
    order = list(users)
    np.random.default_rng(1).shuffle(order)
    fold = {u: n % 5 for n, u in enumerate(order)}
    return [(Data(city, [p for u in users if fold[u] != f for p in city.by_user[u]], city.trip_days),
             cases(city, [u for u in users if fold[u] == f])) for f in range(5)]


def fit_ranker(city, parts, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for d, c in ((d, c) for d, cs in parts for c in cs):
        f = feats(city, d, c)
        blocked = set(c["liked"]) | set(c["before"]) | {c["held"]}
        pool = [j for j in range(len(city.pois)) if j not in blocked]
        for j in rng.choice(pool, size=min(30, len(pool)), replace=False):
            diff = f[c["held"]] - f[j]
            xs += [diff, -diff]
            ys += [1, 0]
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(np.array(xs), np.array(ys))
    return m.coef_[0]


def main():
    cities = {name: prepare(*spec) for name, spec in CITIES.items()}
    full = {}
    for name, city in cities.items():
        parts = out_of_fold(city)
        w = max(itertools.product(GRID_Z_DIST, GRID_Z_NEXT),
                key=lambda w: np.concatenate([z_formula(city, d, cs, w) for d, cs in parts]).mean())
        full[name] = {"z_weights": w, "coef": fit_ranker(city, parts)}
        print(f"{name}: z-formula weights, all its cases out of fold (dist, next) = {w}; ranker coef "
              + ", ".join(f"{k} {v:+.2f}" for k, v in zip(FEATURES, full[name]["coef"], strict=True)), flush=True)

    for name, city in cities.items():
        other = next(o for o in cities if o != name)
        users = sorted(city.visits)
        order = list(users)
        np.random.default_rng(0).shuffle(order)
        fold = {u: n % 5 for n, u in enumerate(order)}
        rows = collections.defaultdict(list)
        for f in range(5):
            tr = [u for u in users if fold[u] != f]
            val = {u for u in tr if fold[u] == (f + 1) % 5}
            dv = Data(city, [p for u in tr if u not in val for p in city.by_user[u]], city.trip_days)
            vc = cases(city, val)
            base = max(({"dist": a, "next": b} for a, b in itertools.product(GRID_DIST, GRID_NEXT)),
                       key=lambda prm: run(city, dv, vc, prm).mean())
            zw = max(itertools.product(GRID_Z_DIST, GRID_Z_NEXT), key=lambda w: z_formula(city, dv, vc, w).mean())
            d = Data(city, [p for u in tr for p in city.by_user[u]], city.trip_days)
            tc = cases(city, [u for u in users if fold[u] == f])
            rows["own validation (production formula)"].append(run(city, d, tc, base))
            rows["z-formula, own validation"].append(z_formula(city, d, tc, zw))
            rows[f"I: z-formula, weights from {other}"].append(z_formula(city, d, tc, full[other]["z_weights"]))
            rows[f"C: learnt ranker from {other}"].append(learnt(city, d, tc, full[other]["coef"]))
            c3 = full[other]["coef"].copy()
            c3[3:] = 0.0
            rows[f"C: learnt ranker from {other}, 3 signals"].append(learnt(city, d, tc, c3))
        rows = {k: np.concatenate(v) for k, v in rows.items()}
        ref = rows["own validation (production formula)"]
        n = len(ref)
        print(f"\n== {name}: {n} cases (95% CI ~ +-{1.96 * math.sqrt(0.25 / n):.3f})")
        for k, h in rows.items():
            t = paired(h, ref)
            print(f"   {k:44s} {h.mean():.3f}  {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['mcnemar_p']:.2g}")


if __name__ == "__main__":
    main()
