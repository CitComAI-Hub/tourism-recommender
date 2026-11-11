"""Evaluation protocol of the paper, for any city, with the production code.

1. Hyperparameters never see the test data: the distance scale and the similarity
   weight (and the season weight) are chosen on a validation split inside the
   training data, then the test split is scored once.
2. Two splits:
   - users: 5 folds by user (cold-start users, the service's main case);
   - time: train on photos before CUTOFF, test on tourists after it (deployment:
     the model is from the past, the users from the future).
3. Paired significance against plain popularity: bootstrap CI of the hit@10
   difference and an exact McNemar test.
4. Context: popularity of the season of the visit (the date is known when the user
   asks), blended with the overall one; where visitors go next after the POI visited
   just before (production code: Recommender.recommend(visited=...)).
5. Besides hit@10: hit@10 of held-out visits outside the 20 most visited POIs (long
   tail), coverage (share of the catalog in some top 10) and diversity of the top 10
   (mean pairwise cosine distance of the description embeddings; distinct categories).

Leave-one-out: each of the last two visits of every tourist with >= 3 visited POIs
is held out; the rest (up to 10) are the likes.

Usage: PYTHONPATH=research python research/protocol.py <config> <lat0> <lat1> <lon0> <lon1> [cutoff]
e.g.   PYTHONPATH=research python research/protocol.py base 39.40 39.53 -0.44 -0.27 2012-01-01
"""
from __future__ import annotations

import collections
import copy
import datetime as dt
import itertools
import json
import math
import sys
from typing import Dict, List, Sequence

import numpy as np

from common import city_photos, load_config, pseudonym
from pmv_turismo.behaviour import first_visits, train
from pmv_turismo.common import load_poi_list, processed_dir
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.geometry import assign_photos, load_geometries
from pmv_turismo.recommendation import Recommender

K = 10
# distance penalty: ("linear", km scale) or ("log", weight of log(km + 50 m))
GRID_TAU = (("linear", 0.15), ("linear", 0.3), ("linear", 0.6), ("linear", 1.0),
            ("log", 0.5), ("log", 1.0), ("log", 1.5), ("log", 2.0), ("log", 3.0), ("log", 4.0))
# Similarity to the likes was removed from the service (research/traits.py): kept at 0.
GRID_SIM = (0.0,)
GRID_SEASON = (0.0, 0.25, 0.5, 0.75)
GRID_NEXT = (0.0, 0.25, 0.5, 1.0)
GRID_CAL = (0.0, 0.25, 0.5, 0.75, 1.0)
# Category diversity (Recommender.diversity_weight): the largest weight whose
# validation hit@10 is within NON_INFERIORITY of the best one.
GRID_DIV = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
NON_INFERIORITY = 0.01
TOURIST_MAX_DAYS = 21
HEAD = 20


def season(ts: str) -> int:
    return (int(ts[5:7]) % 12) // 3  # 0 DJF, 1 MAM, 2 JJA, 3 SON


class City:
    def __init__(self, cfg_name: str, bbox: Dict[str, tuple]):
        self.cfg = load_config(cfg_name)
        self.cfg.setdefault("recommendation", {}).update({"filter_opening_hours": False, "top_n": K})
        pois = load_poi_list(processed_dir(self.cfg) / "poi.jsonld")
        self.index = load_embedding_index(embeddings_path(self.cfg))
        self.pois = [p for p in pois if self.index.index_of(p["id"]) is not None]
        self.ids = [p["id"] for p in self.pois]
        self.units = self.index.units[[self.index.index_of(i) for i in self.ids]]
        self.shapes = load_geometries(self.cfg)
        self.photos = [{"user": pseudonym(r["uid"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
                        "ts": r["datetaken"][:19]} for r in city_photos(bbox)]
        lat = np.array([p["lat"] for p in self.photos])
        lon = np.array([p["lon"] for p in self.photos])
        # (user, poi) -> first visit time
        self.visits: Dict[str, Dict[int, str]] = collections.defaultdict(dict)
        for j, idx in enumerate(assign_photos(self.pois, self.shapes, lat, lon)):
            for k in idx:
                u, ts = self.photos[k]["user"], self.photos[k]["ts"]
                self.visits[u][j] = min(self.visits[u].get(j, ts), ts)
        dates = collections.defaultdict(list)
        for p in self.photos:
            dates[p["user"]].append(p["ts"][:10])
        self.tourists = {u for u, d in dates.items()
                         if (dt.date.fromisoformat(max(d)) - dt.date.fromisoformat(min(d))).days <= TOURIST_MAX_DAYS}
        self.by_user = collections.defaultdict(list)
        for p in self.photos:
            self.by_user[p["user"]].append(p)

    def cases(self, users: Sequence[str]) -> List[tuple]:
        """(user, liked ids, held-out index, date of the held-out visit, POI visited just before)."""
        out = []
        for u in sorted(users):
            v = self.visits.get(u, {})
            if u not in self.tourists or len(v) < 3:
                continue
            seq = sorted(v, key=v.get)
            for held in seq[-2:]:
                prev = seq[seq.index(held) - 1]
                out.append((u, [self.ids[x] for x in seq if x != held][-10:], held, v[held], prev))
        return out

    def model(self, photos: List[dict]):
        visits = first_visits(photos, self.pois, self.shapes)
        seasonal = np.zeros((4, len(self.pois)))
        lat = np.array([p["lat"] for p in photos])
        lon = np.array([p["lon"] for p in photos])
        seen = collections.defaultdict(set)
        for j, idx in enumerate(assign_photos(self.pois, self.shapes, lat, lon)):
            for k in idx:
                seen[(photos[k]["user"], j, season(photos[k]["ts"]))].add(1)
        for (_, j, s) in seen:
            seasonal[s, j] += 1
        return train(self.pois, self.units, visits, "eval"), np.log1p(seasonal)


class Run(np.ndarray):
    """Hits of one method (a float array, so means and pairing work as before) plus
    the top 10 of every case, for coverage and diversity."""
    tops: List[List[int]]


def evaluate(city: City, rec: Recommender, cases: List[tuple], tau: tuple, sim: float,
             seasonal: np.ndarray | None = None, season_w: float = 0.0, pop_only: bool = False,
             next_w: float = 0.0, cal: float = 0.0, div: float = 0.0) -> Run:
    r = copy.copy(rec)
    r.distance_form = tau[0]
    if tau[0] == "log":
        r.distance_log_weight = tau[1]
    else:
        r.distance_scale_km = tau[1]
    r.next_weight, r.diversity_weight = next_w, div
    r.set_predicted_calibration(cal)
    hits, tops = np.zeros(len(cases)), []
    base_pop = r.popularity
    for n, (_, liked, held, ts, prev) in enumerate(cases):
        r.popularity = base_pop
        if seasonal is not None and season_w:
            r.popularity = (1 - season_w) * base_pop + season_w * seasonal[season(ts)]
        if pop_only:
            liked_set = {rec.position[x] for x in liked}
            top = [i for i in np.argsort(-r.popularity, kind="stable") if i not in liked_set][:K]
        else:
            visited = [city.ids[prev]] if next_w else None
            top = [r.position[x["id"]] for x in r.recommend(liked=liked, visited=visited, top_n=K)]
        hits[n] = held in top
        tops.append(top)
    out = hits.view(Run)
    out.tops = tops
    return out


def extra_metrics(city: City, rec: Recommender, cases: List[tuple], run: Run) -> dict:
    head = set(np.argsort(-rec.popularity, kind="stable")[:HEAD].tolist())
    tail = np.array([held not in head for (_, _, held, _, _) in cases])
    seen = {i for top in run.tops for i in top}
    dist, cats = [], []
    for top in run.tops:
        u = city.units[top]
        sim = u @ u.T
        m = len(top)
        dist.append(1.0 - (sim.sum() - np.trace(sim)) / max(m * (m - 1), 1))
        cats.append(len({c for i in top for c in city.pois[i].get("category") or []}))
    return {"hit_tail": float(np.asarray(run)[tail].mean()) if tail.any() else float("nan"),
            "tail_cases": int(tail.sum()), "coverage": len(seen) / len(city.pois),
            "diversity": float(np.mean(dist)), "categories": float(np.mean(cats))}


def choose(city, rec, val_cases, seasonal=None):
    best = (-1.0, (GRID_TAU[0], GRID_SIM[0]))  # defaults when there are no validation cases
    for tau, sim in itertools.product(GRID_TAU, GRID_SIM):
        h = evaluate(city, rec, val_cases, tau, sim).mean() if val_cases else -1.0
        best = max(best, (h, (tau, sim)), key=lambda t: t[0])
    tau, sim = best[1]
    sw = 0.0
    if seasonal is not None:
        sw = max(GRID_SEASON, key=lambda w: evaluate(city, rec, val_cases, tau, sim, seasonal, w).mean())
    return tau, sim, sw


def weighted_extra(folds: List[dict]) -> dict:
    """Pool the per-fold extra metrics (tail hit weighted by tail cases; coverage of
    the union is not recoverable, so the mean per fold)."""
    tail = sum(f["tail_cases"] for f in folds)
    hit_tail = (sum(f["hit_tail"] * f["tail_cases"] for f in folds if f["tail_cases"]) / tail) if tail else float("nan")
    return {"hit_tail": hit_tail, "tail_cases": tail,
            **{k: float(np.mean([f[k] for f in folds])) for k in ("coverage", "diversity", "categories")}}


def paired(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(10000, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "mcnemar_p": p}


def run_split(city: City, train_users, test_cases, val_users, val_cases_fn, label: str, photo_filter=None):
    photos = [p for u in train_users for p in city.by_user[u] if photo_filter is None or photo_filter(p)]
    val_train = [p for u in train_users if u not in val_users for p in city.by_user[u]
                 if photo_filter is None or photo_filter(p)]
    m_val, s_val = city.model(val_train)
    rec_val = Recommender(city.pois, city.index, m_val, city.cfg)
    val_cases = val_cases_fn()
    tau, sim, sw = choose(city, rec_val, val_cases, s_val)
    nw = max(GRID_NEXT, key=lambda w: evaluate(city, rec_val, val_cases, tau, sim, next_w=w).mean())
    cw = max(GRID_CAL, key=lambda w: evaluate(city, rec_val, val_cases, tau, sim, next_w=nw, cal=w).mean())
    div_hits = {w: evaluate(city, rec_val, val_cases, tau, sim, next_w=nw, div=w).mean() for w in GRID_DIV}
    dw = max(w for w in GRID_DIV if div_hits[w] >= max(div_hits.values()) - NON_INFERIORITY)
    lin = max((t for t in GRID_TAU if t[0] == "linear"),
              key=lambda t: evaluate(city, rec_val, val_cases, t, sim, next_w=nw).mean())
    model, seasonal = city.model(photos)
    rec = Recommender(city.pois, city.index, model, city.cfg)
    wiki = Recommender(city.pois, city.index, None, city.cfg)
    rows = {
        "popularity": evaluate(city, rec, test_cases, tau, sim, pop_only=True),
        "Wikipedia popularity + proximity": evaluate(city, wiki, test_cases, tau, 0.0),
        "gravity (no similarity)": evaluate(city, rec, test_cases, tau, 0.0),
        "service": evaluate(city, rec, test_cases, tau, sim),
        "service + season": evaluate(city, rec, test_cases, tau, sim, seasonal, sw),
        "service + next POI, linear distance": evaluate(city, rec, test_cases, lin, sim, next_w=nw),
        "service + next POI": evaluate(city, rec, test_cases, tau, sim, next_w=nw),
        "+ calibrated predicted popularity": evaluate(city, rec, test_cases, tau, sim, next_w=nw, cal=cw),
        "service + next POI + diversity": evaluate(city, rec, test_cases, tau, sim, next_w=nw, div=dw),
    }
    extra = {k: extra_metrics(city, rec, test_cases, h) for k, h in rows.items()}
    return {"split": label, "chosen": {"distance": f"{tau[0]} {tau[1]}", "similarity_weight": sim, "season_weight": sw, "next_weight": nw,
                       "calibration": cw, "diversity_weight": dw},
            "rows": rows, "extra": extra}


def main():
    cfg_name, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    cutoff = sys.argv[6] if len(sys.argv) > 6 else "2012-01-01"
    city = City(cfg_name, {"lat": (la0, la1), "lon": (lo0, lo1)})
    users = sorted(city.visits)
    rng = np.random.default_rng(0)
    order = list(users)
    rng.shuffle(order)
    fold = {u: n % 5 for n, u in enumerate(order)}
    print(f"{cfg_name}: POIs {len(city.pois)}, photos {len(city.photos)}, users with visits {len(users)}, "
          f"tourists {len(city.tourists)}, LOO cases {len(city.cases(users))}")

    results = []
    # users split: outer fold f is the test; inner validation = fold (f+1) % 5
    per_fold = collections.defaultdict(list)
    for f in range(5):
        test_users = [u for u in users if fold[u] == f]
        v = (f + 1) % 5
        train_users = [u for u in users if fold[u] != f]
        val_users = {u for u in train_users if fold[u] == v}
        res = run_split(city, train_users, city.cases(test_users), val_users,
                        lambda val_users=val_users: city.cases(val_users), f"users fold {f}")
        per_fold["chosen"].append(res["chosen"])
        per_fold["extra"].append(res["extra"])
        for k, h in res["rows"].items():
            per_fold[k].append(h)
    extra = {k: weighted_extra([e[k] for e in per_fold["extra"]]) for k in per_fold["extra"][0]}
    results.append(("users (5 folds)", {k: np.concatenate([np.asarray(x) for x in v])
                                        for k, v in per_fold.items() if k not in ("chosen", "extra")},
                    per_fold["chosen"], extra))

    # time split: train before the cutoff, test tourists whose visits are all after it
    before = [u for u in users if max(p["ts"] for p in city.by_user[u]) < cutoff]
    after = [u for u in users if min(p["ts"] for p in city.by_user[u]) >= cutoff]
    times = sorted(min(p["ts"] for p in city.by_user[u]) for u in before)
    val_start = times[int(0.8 * len(times))] if times else cutoff
    val_users = {u for u in before if min(p["ts"] for p in city.by_user[u]) >= val_start}
    res = run_split(city, before, city.cases(after), val_users, lambda: city.cases(val_users), "time")
    results.append((f"time (train < {cutoff}, {len(before)} users; test {len(after)} users)",
                    {k: np.asarray(v) for k, v in res["rows"].items()}, [res["chosen"]], res["extra"]))

    report = {"city": cfg_name, "splits": {}}
    for label, rows, chosen, extra in results:
        n = len(rows["service"])
        print(f"\n== split: {label}; cases {n} (95% CI ~ +-{1.96 * math.sqrt(0.25 / max(n, 1)):.3f})")
        print(f"   chosen on validation: {chosen}")
        print(f"   {'':34s} hit@10  tail({extra['service']['tail_cases']})  coverage  diversity  categories")
        for k, h in rows.items():
            e = extra[k]
            print(f"   {k:34s} {h.mean():.3f}   {e['hit_tail']:.3f}      {e['coverage']:.3f}     "
                  f"{e['diversity']:.3f}      {e['categories']:.2f}")
        for a in ("service", "service + season"):
            t = paired(rows[a], rows["popularity"])
            print(f"   {a} vs popularity: diff {t['diff']:+.3f} CI95 [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] McNemar p={t['mcnemar_p']:.2g}")
        t = paired(rows["service + season"], rows["service"])
        print(f"   season vs service: diff {t['diff']:+.3f} CI95 [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] McNemar p={t['mcnemar_p']:.2g}")
        tc = paired(rows["+ calibrated predicted popularity"], rows["service + next POI"])
        print(f"   calibration vs next POI: diff {tc['diff']:+.3f} CI95 [{tc['ci95'][0]:+.3f}, {tc['ci95'][1]:+.3f}] "
              f"McNemar p={tc['mcnemar_p']:.2g}")
        td = paired(rows["service + next POI + diversity"], rows["service + next POI"])
        print(f"   diversity vs next POI: diff {td['diff']:+.3f} CI95 [{td['ci95'][0]:+.3f}, {td['ci95'][1]:+.3f}] "
              f"McNemar p={td['mcnemar_p']:.2g}")
        tl = paired(rows["service + next POI"], rows["service + next POI, linear distance"])
        print(f"   chosen distance vs linear: diff {tl['diff']:+.3f} CI95 [{tl['ci95'][0]:+.3f}, {tl['ci95'][1]:+.3f}] "
              f"McNemar p={tl['mcnemar_p']:.2g}")
        tn = paired(rows["service + next POI"], rows["service"])
        print(f"   next POI vs service: diff {tn['diff']:+.3f} CI95 [{tn['ci95'][0]:+.3f}, {tn['ci95'][1]:+.3f}] McNemar p={tn['mcnemar_p']:.2g}")
        report["splits"][label] = {"cases": n, "chosen": chosen, "hit@10": {k: float(h.mean()) for k, h in rows.items()},
                                   "extra": extra,
                                   "service_vs_popularity": paired(rows["service"], rows["popularity"]),
                                   "season_vs_service": t,
                                   "next_vs_service": paired(rows["service + next POI"], rows["service"]),
                                   "calibration_vs_next": tc,
                                   "diversity_vs_next": td,
                                   "distance_vs_linear": tl}
    out = processed_dir(city.cfg) / "protocol_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
