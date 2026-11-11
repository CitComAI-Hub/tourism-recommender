"""Idea E: which onboarding cards get the most useful answers?

Same simulation as research/traits.py (tourists with >= 3 visits; the last two
held out; earlier visits = what the user would like; position = first photo;
the oracle likes a card that is a known visit and gives no answer otherwise).
Likes are used as distance anchors (production). 14 cards, strategies:

- popular: the 14 most popular places (production);
- near position: the 14 best by the production score from the user's position
  (the onboarding asks for the position before the cards);
- adaptive (batch b): b cards at a time; each batch is the top of the production
  score from the position and the likes so far, without the cards already shown;
- adaptive spread: like adaptive, but the cards of one batch are at least 300 m
  apart (looks for the user in more areas).

Usage: PYTHONPATH=research python research/cards.py <config> <lat0> <lat1> <lon0> <lon1>
"""
from __future__ import annotations

import collections
import math
import sys

import numpy as np
from protocol import City, paired
from traits import CARDS, GRID_DIST, Data, rank

from pmv_turismo.recommendation import LOG_DISTANCE_OFFSET_KM, Recommender


def score(d: Data, rec: Recommender, u: str, anchors: list[int], dist_w: float, position: bool = True) -> np.ndarray:
    dists = [d.D[i] for i in anchors]
    if position:
        dists.append(rec._distances_km(*d.first[u]))
    near = np.min(dists, axis=0) if dists else np.zeros(d.n)
    return rec.popularity - dist_w * np.log(near + LOG_DISTANCE_OFFSET_KM)


def show(d: Data, rec: Recommender, case, kind: str, dist_w: float) -> tuple[list[int], list[int]]:
    u, known, _ = case
    known = set(known)
    if kind == "popular":
        shown = list(np.argsort(-rec.popularity, kind="stable")[:CARDS])
        return shown, [i for i in shown if i in known]
    if kind == "near position":
        s = score(d, rec, u, [], dist_w)
        shown = list(np.argsort(-s, kind="stable")[:CARDS])
        return shown, [i for i in shown if i in known]
    batch = int(kind.split()[-1])
    spread = "spread" in kind
    shown: list[int] = []
    liked: list[int] = []
    while len(shown) < CARDS:
        s = score(d, rec, u, liked, dist_w)
        new: list[int] = []
        for i in np.argsort(-s, kind="stable"):
            if i in shown or i in new:
                continue
            if spread and new and d.D[i, new].min() < 0.3:
                continue  # within a batch, cards in different areas
            new.append(int(i))
            if len(new) == min(batch, CARDS - len(shown)):
                break
        shown += new
        liked += [i for i in new if i in known]
    return shown, liked


KINDS = ("popular", "near position", "adaptive 1", "adaptive 2", "adaptive 7", "adaptive spread 2")


def main():
    cfg_name, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    city = City(cfg_name, {"lat": (la0, la1), "lon": (lo0, lo1)})
    d = Data(city)
    users = sorted(city.visits)
    order = list(users)
    np.random.default_rng(0).shuffle(order)
    fold = {u: n % 5 for n, u in enumerate(order)}
    full = Recommender(city.pois, city.index, city.model(city.photos)[0], city.cfg)
    d.D = np.array([full._distances_from(i) for i in range(d.n)])
    rows = collections.defaultdict(list)
    likes = collections.defaultdict(list)
    for f in range(5):
        v = (f + 1) % 5
        val_train = [p for u in users if fold[u] not in (f, v) for p in city.by_user[u]]
        test_train = [p for u in users if fold[u] != f for p in city.by_user[u]]
        rec_val = Recommender(city.pois, city.index, city.model(val_train)[0], city.cfg)
        val_cases = d.cases([u for u in users if fold[u] == v])
        dist_w = max(GRID_DIST, key=lambda w: np.mean([c[2] in rank(d, rec_val, c, dist_w=w) for c in val_cases]))
        rec = Recommender(city.pois, city.index, city.model(test_train)[0], city.cfg)
        cases = d.cases([u for u in users if fold[u] == f])
        rows["position only"].append(np.array([c[2] in rank(d, rec, c, dist_w=dist_w) for c in cases], dtype=float))
        rows["oracle: every known visit as like"].append(
            np.array([c[2] in rank(d, rec, c, anchors=c[1], dist_w=dist_w) for c in cases], dtype=float))
        for kind in KINDS:
            h = []
            for c in cases:
                _, liked = show(d, rec, c, kind, dist_w)
                likes[kind].append(len(liked))
                h.append(c[2] in rank(d, rec, c, anchors=liked, dist_w=dist_w))
            rows[kind].append(np.array(h, dtype=float))
        print(f"   fold {f}: dist_w {dist_w}", flush=True)
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    n = len(rows["popular"])
    print(f"\n== {cfg_name}: onboarding cards, {n} cases (95% CI ~ +-{1.96 * math.sqrt(0.25 / n):.3f})")
    print(f"   {'':36s} hit@10  likes/14   vs popular cards")
    for k, h in rows.items():
        t = paired(h, rows["popular"])
        lk = f"{np.mean(likes[k]):.2f}" if k in likes else "  - "
        print(f"   {k:36s} {h.mean():.3f}   {lk}      {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['mcnemar_p']:.2g}")


if __name__ == "__main__":
    main()
