"""Context experiments: richer next-POI, time of day, trip length (plan v2, ideas A, B, D, H).

Same protocol as research/protocol.py (validation-chosen weights; users split in
5 folds and time split; paired bootstrap + McNemar), but scored with a fast,
vectorised copy of the production formula so many variants fit in one run:

    score = popularity + next_w * log(1 + 100 * P(next | context)) - dist_w * log(km + 0.05)
            [+ variant terms]

`base` reproduces Recommender.recommend (log distance, next POI from the last
visit, liked/visited excluded, one entry per name). Variants, each chosen on
validation on top of base, then all together (greedy, in the order below):

A1 order 2: P(next | the two last visits), backed off to order 1 (kappa).
A2 decayed history: P(next | each earlier visit), weights gamma^k.
A3 category smoothing: P(next | last) shrunk towards P(category of next | category
   of last) x share of the POI within its category (kappa), so unseen pairs get
   a plausible value.
A4 distance from the last visit (the user is there now), on top of the nearest
   anchor.
B1 hour of the day (bucket of the time of the last visit = "now"): log lift of the
   POI's share of visits in that bucket. B2 weekend vs weekday, same form.
D1 trip length (days between the user's first and last photo, which a user can
   say at onboarding): popularity multiplier per bucket (short trips go to the
   famous places?). D2 lift of the POI among users of the same bucket.
H  Wikipedia blended into the popularity (z-score x beta): closer to the official
   visitor counts (research/external_blend.py); here only its cost in hit@10, at
   fixed beta, on top of the combination (not chosen on Flickr visits).

Usage: PYTHONPATH=research python research/context.py <config> <lat0> <lat1> <lon0> <lon1> [cutoff]
"""
from __future__ import annotations

import collections
import datetime as dt
import itertools
import math
import sys
import unicodedata

import numpy as np
from protocol import HEAD, City, paired

from pmv_turismo.behaviour import first_visits, train
from pmv_turismo.recommendation import Recommender

K = 10
GRID_DIST = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
GRID_NEXT = (0.0, 0.25, 0.5, 1.0, 1.5)
HOURS = ((6, 11), (11, 15), (15, 19), (19, 30))  # morning, midday, afternoon, evening/night
TRIP = ((0, 0), (1, 2), (3, 10 ** 6))  # days between first and last photo


def hour_bucket(ts: str) -> int:
    h = int(ts[11:13]) if len(ts) >= 13 else 12
    h = h + 24 if h < 6 else h
    return next(b for b, (a, z) in enumerate(HOURS) if a <= h < z)


def weekend(ts: str) -> int:
    return int(dt.date.fromisoformat(ts[:10]).weekday() >= 5)


def trip_bucket(days: int) -> int:
    return next(b for b, (a, z) in enumerate(TRIP) if a <= days <= z)


def fold_name(s: str) -> str:
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode().strip()


class Data:
    """Everything learnt from one training set of photos."""

    def __init__(self, city: City, photos: list[dict], trip_days: dict[str, int]):
        n = len(city.pois)
        visits = first_visits(photos, city.pois, city.shapes)
        model = train(city.pois, city.units, visits, "context")
        self.rec = Recommender(city.pois, city.index, model, city.cfg)
        self.pop = self.rec.popularity.copy()
        self.t1 = self.rec.transitions
        counts1 = np.zeros((n, n))
        self.c2: dict[tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
        hours, wkend, trips = np.zeros((len(HOURS), n)), np.zeros((2, n)), np.zeros((len(TRIP), n))
        for u, seen in visits.items():
            seq = sorted(seen, key=lambda j: (seen[j], j))
            for a, b in zip(seq, seq[1:], strict=False):
                counts1[a, b] += 1
            for a, b, c in zip(seq, seq[1:], seq[2:], strict=False):
                self.c2[(a, b)][c] += 1
            for j, ts in seen.items():
                if ts:
                    hours[hour_bucket(ts), j] += 1
                    wkend[weekend(ts), j] += 1
            if u in trip_days:
                trips[trip_bucket(trip_days[u]), list(seen)] += 1
        self.lift_hour = self._lift(hours)
        self.lift_weekend = self._lift(wkend, prior=wkend.sum(axis=1))
        self.lift_trip = self._lift(trips, prior=trips.sum(axis=1))
        # category smoothing target: P(cat b | cat a) * share of b within its category
        cat = [(p.get("category") or ["?"])[0] for p in city.pois]
        cats = sorted(set(cat))
        ci = np.array([cats.index(c) for c in cat])
        cc = np.zeros((len(cats), len(cats)))
        for a, b in zip(*np.nonzero(counts1), strict=False):
            cc[ci[a], ci[b]] += counts1[a, b]
        pcat = (cc + 0.1) / (cc + 0.1).sum(axis=1, keepdims=True)
        w = np.exp(self.pop)
        share = w / np.array([w[ci == ci[j]].sum() for j in range(n)])
        self.q = pcat[ci][:, ci] * share[None, :]
        self.counts1 = counts1
        wiki = np.array([1.0 - float(p["relevance"]) if isinstance(p.get("relevance"), (int, float)) else 0.0
                         for p in city.pois])
        # H: Wikipedia on the scale of the popularity (debias towards official counts)
        self.wiki = (wiki - wiki.mean()) / (wiki.std() + 1e-9) * self.pop.std()
        self.out1 = counts1.sum(axis=1)

    @staticmethod
    def _lift(m: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        """log((n_bj + a p_b) / (n_j + a) / p_b): how much more often POI j is visited
        in bucket b than the average POI (0 without visits). Smoothed with a = 3."""
        p = (prior if prior is not None else m.sum(axis=1)) + 1e-9
        p = p / p.sum()
        nj = m.sum(axis=0)
        return np.log((m + 3 * p[:, None]) / (nj[None, :] + 3) / p[:, None])


def cases(city: City, users) -> list[dict]:
    out = []
    for u in sorted(users):
        v = city.visits.get(u, {})
        if u not in city.tourists or len(v) < 3:
            continue
        seq = sorted(v, key=v.get)
        for held in seq[-2:]:
            pos = seq.index(held)
            before = seq[:pos]
            out.append({"liked": [x for x in seq if x != held][-10:], "held": held, "before": before,
                        "now": v[before[-1]], "trip": city.trip_days[u]})
    return out


def score_case(city: City, d: Data, c: dict, prm: dict) -> np.ndarray:
    anchors = sorted(set(c["liked"]))
    near = city.dist[anchors].min(axis=0)
    pop = d.pop * prm.get("trip_mult", {}).get(trip_bucket(c["trip"]), 1.0)
    last = c["before"][-1]
    p = d.t1[last]
    if prm.get("a3"):
        k = prm["a3"]
        p = (d.counts1[last] + k * d.q[last]) / (d.out1[last] + k)
    if prm.get("a1") and len(c["before"]) >= 2:
        cnt = d.c2.get((c["before"][-2], last))
        if cnt:
            n2 = sum(cnt.values())
            p2 = np.zeros_like(p)
            for j, x in cnt.items():
                p2[j] = x / n2
            lam = n2 / (n2 + prm["a1"])
            p = lam * p2 + (1 - lam) * p
    if prm.get("a2"):
        g = prm["a2"]
        hist = c["before"][::-1]
        ws = np.array([g ** k for k in range(len(hist))])
        p = (ws[:, None] * d.t1[hist]).sum(axis=0) / ws.sum() if prm.get("a2_replace") else \
            p + sum(g ** k * d.t1[h] for k, h in enumerate(hist[1:], start=1))
    s = pop + prm["next"] * np.log1p(100 * p) - prm["dist"] * np.log(near + 0.05)
    if prm.get("a4"):
        s = s - prm["a4"] * np.log(city.dist[last] + 0.05)
    if prm.get("b1"):
        s = s + prm["b1"] * d.lift_hour[hour_bucket(c["now"])]
    if prm.get("b2"):
        s = s + prm["b2"] * d.lift_weekend[weekend(c["now"])]
    if prm.get("h"):
        s = s + prm["h"] * d.wiki
    if prm.get("d2"):
        s = s + prm["d2"] * d.lift_trip[trip_bucket(c["trip"])]
    return s


def top10(city: City, s: np.ndarray, c: dict) -> list[int]:
    blocked = set(c["liked"]) | set(c["before"])
    names = {city.names[j] for j in blocked}
    out = []
    for j in np.argsort(-s, kind="stable"):
        if j in blocked or city.names[j] in names:
            continue
        names.add(city.names[j])
        out.append(int(j))
        if len(out) == K:
            break
    return out


def run(city: City, d: Data, cs: list[dict], prm: dict) -> np.ndarray:
    return np.array([c["held"] in top10(city, score_case(city, d, c, prm), c) for c in cs], dtype=float)


VARIANTS = {
    "A1 order-2 next": [{"a1": k} for k in (1, 3, 10)],
    "A2 decayed history": [{"a2": g} for g in (0.2, 0.4, 0.6)] + [{"a2": g, "a2_replace": 1} for g in (0.3, 0.6)],
    "A3 category smoothing": [{"a3": k} for k in (0.5, 2, 5)],
    "A4 distance from last": [{"a4": w} for w in (0.25, 0.5, 1.0)],
    "B1 hour of day": [{"b1": w} for w in (0.25, 0.5, 1.0)],
    "B2 weekend": [{"b2": w} for w in (0.25, 0.5, 1.0)],
    "D1 trip length x popularity": [{"trip_mult": dict(zip(range(3), m, strict=True))}
                                    for m in itertools.product((0.75, 1.0, 1.25, 1.5), repeat=3)],
    "D2 trip length lift": [{"d2": w} for w in (0.25, 0.5, 1.0)],
}


def choose(city, d, val, base):
    """Validation choice: the variant setting with the best hit@10, or none if no
    setting beats base."""
    base_hit = run(city, d, val, base).mean() if val else 0.0
    chosen = {}
    for name, grid in VARIANTS.items():
        best, best_hit = None, base_hit
        for g in grid:
            h = run(city, d, val, {**base, **g}).mean()
            if h > best_hit + 1e-9:
                best, best_hit = g, h
        chosen[name] = best
    # all together: add the chosen ones greedily, keep each only if validation improves
    combo, combo_hit = dict(base), base_hit
    for name, g in chosen.items():
        if g is None:
            continue
        h = run(city, d, val, {**combo, **g}).mean()
        if h > combo_hit + 1e-9:
            combo, combo_hit = {**combo, **g}, h
    return chosen, combo


def split(city, train_users, val_users, test_cases):
    photos = [p for u in train_users for p in city.by_user[u]]
    val_photos = [p for u in train_users if u not in val_users for p in city.by_user[u]]
    dv = Data(city, val_photos, city.trip_days)
    val = cases(city, val_users)
    base = max(({"dist": a, "next": b} for a, b in itertools.product(GRID_DIST, GRID_NEXT)),
               key=lambda prm: run(city, dv, val, prm).mean())
    chosen, combo = choose(city, dv, val, base)
    d = Data(city, photos, city.trip_days)
    rows = {"popularity": np.array([c["held"] in top10(city, d.pop.copy(), c) for c in test_cases], dtype=float),
            "base (production formula)": run(city, d, test_cases, base),
            "base without next POI": run(city, d, test_cases, {**base, "next": 0.0})}
    for name, g in chosen.items():
        rows[name] = run(city, d, test_cases, {**base, **g}) if g is not None else rows["base (production formula)"]
    rows["all chosen together"] = run(city, d, test_cases, combo)
    # H is not chosen on Flickr visits (it moves away from them on purpose): its cost
    for h in (0.25, 0.5):
        rows[f"H Wikipedia blend {h} (fixed)"] = run(city, d, test_cases, {**combo, "h": h})
    head = set(np.argsort(-d.pop)[:HEAD].tolist())
    tail = np.array([c["held"] not in head for c in test_cases])
    return rows, tail, {"base": base, "chosen": chosen, "combo": combo}


def main():
    cfg, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    cutoff = sys.argv[6] if len(sys.argv) > 6 else "2012-01-01"
    city = City(cfg, {"lat": (la0, la1), "lon": (lo0, lo1)})
    lat = np.radians([p["location"]["coordinates"][1] for p in city.pois])
    lon = np.radians([p["location"]["coordinates"][0] for p in city.pois])
    city.dist = 6371 * np.sqrt(((lon[:, None] - lon[None]) * np.cos((lat[:, None] + lat[None]) / 2)) ** 2
                               + (lat[:, None] - lat[None]) ** 2)
    city.names = [fold_name(str(p.get("name") or p["id"])) for p in city.pois]
    city.trip_days = {u: (dt.date.fromisoformat(max(p["ts"] for p in ps)[:10])
                          - dt.date.fromisoformat(min(p["ts"] for p in ps)[:10])).days
                      for u, ps in city.by_user.items()}
    users = sorted(city.visits)
    order = list(users)
    np.random.default_rng(0).shuffle(order)
    fold = {u: n % 5 for n, u in enumerate(order)}
    trips = collections.Counter(trip_bucket(city.trip_days[u]) for u in users if u in city.tourists)
    print(f"{cfg}: POIs {len(city.pois)}, tourists {len(city.tourists)}, trip buckets {dict(sorted(trips.items()))}")

    results = []
    per = collections.defaultdict(list)
    tails, choices = [], []
    for f in range(5):
        test = [u for u in users if fold[u] == f]
        tr = [u for u in users if fold[u] != f]
        val = {u for u in tr if fold[u] == (f + 1) % 5}
        rows, tail, ch = split(city, tr, val, cases(city, test))
        for k, v in rows.items():
            per[k].append(v)
        tails.append(tail)
        choices.append(ch)
    results.append(("users (5 folds)", {k: np.concatenate(v) for k, v in per.items()}, np.concatenate(tails), choices))
    before = [u for u in users if max(p["ts"] for p in city.by_user[u]) < cutoff]
    after = [u for u in users if min(p["ts"] for p in city.by_user[u]) >= cutoff]
    times = sorted(min(p["ts"] for p in city.by_user[u]) for u in before)
    vs = times[int(0.8 * len(times))]
    val = {u for u in before if min(p["ts"] for p in city.by_user[u]) >= vs}
    rows, tail, ch = split(city, before, val, cases(city, after))
    results.append((f"time (< {cutoff})", rows, tail, [ch]))

    for label, rows, tail, choices in results:
        n = len(rows["popularity"])
        print(f"\n== {label}: {n} cases, tail {int(tail.sum())} (95% CI ~ +-{1.96 * math.sqrt(0.25 / n):.3f})")
        for ch in choices:
            print(f"   chosen: base {ch['base']}; " + "; ".join(f"{k.split()[0]}={v}" for k, v in ch["chosen"].items() if v)
                  + f"; combo {ch['combo']}")
        b = rows["base (production formula)"]
        print(f"   {'':32s} hit@10  tail   vs base: diff  CI95             McNemar p")
        for k, h in rows.items():
            t = paired(h, b)
            print(f"   {k:32s} {h.mean():.3f}  {h[tail].mean():.3f}  {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  {t['mcnemar_p']:.2g}")


if __name__ == "__main__":
    main()
