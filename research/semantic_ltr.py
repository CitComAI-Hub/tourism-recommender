"""Does semantics add to the ranking when a learnt ranker can use it as it likes?

The protocol only tried "gravity + w * mean similarity to the likes" and the
validation picked w = 0. Here a conditional logit (softmax over the candidates of
each case) learns the weights of several signals at once, so semantics can enter
in other forms: max similarity (one liked POI is enough), similarity weighted by
how coherent the user's likes are (a user who only photographs churches vs one who
photographs everything), share of likes in the same category, and the popularity
predicted from the description (a content prior also for visited POIs).

Features of a candidate j for a case (likes L, previous visit p):
  base:      log popularity, distance to the nearest like (km and log), log(1 + 100 P(j | p))
  raw:       mean / max cosine of the description embeddings to L, mean x coherence(L)
  adapted:   mean / max cosine of the behaviour-adapted embeddings to L
  category:  share of L with a category of j
  content:   popularity predicted from the description (BehaviourModel.predicted_popularity)

No leakage: training cases of the ranker come from cross-fitting inside the training
users (the behaviour model of each inner fold never saw that fold's users); the test
users of the outer fold are scored once, with a model trained on all the training users.

Usage: [SCENARIO=onsite|likes|remote] PYTHONPATH=research python research/semantic_ltr.py <config> <lat0> <lat1> <lon0> <lon1>
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
from scipy.optimize import minimize

from protocol import HEAD, K, City, paired
from pmv_turismo.behaviour import first_visits, train
from pmv_turismo.recommendation import Recommender

FOLDS, INNER = 5, 4
L2 = 1.0
# SCENARIO=remote: the user likes places from home (planning), so there is no
# location anchor: distance and the previous visit are not features.
# SCENARIO=likes: from home too, but with the distance to the liked POIs (the
# production gravity uses it without any user location); no previous visit.
SCENARIO = os.environ.get("SCENARIO", "onsite")
REMOTE = SCENARIO == "remote"
BASE = {"remote": ["pop"], "likes": ["pop", "dist", "logdist"]}.get(SCENARIO, ["pop", "dist", "logdist", "next"])
GROUPS = {
    "base": BASE,
    "raw": ["sim_mean", "sim_max", "sim_coh"],
    "adapted": ["ad_mean", "ad_max"],
    "category": ["cat"],
    "content": ["content_pop"],
}
VARIANTS = {
    "learnt gravity (base)": ["base"],
    "+ raw semantics": ["base", "raw"],
    "+ adapted semantics": ["base", "adapted"],
    "+ category": ["base", "category"],
    "+ content popularity": ["base", "content"],
    "+ all semantics": ["base", "raw", "adapted", "category", "content"],
}
SEMANTIC = "+ all semantics"


class Features:
    """Per-case candidate features for one behaviour model."""

    def __init__(self, city: City, rec: Recommender, model) -> None:
        self.city, self.rec = city, rec
        self.content = np.clip(model.predicted_popularity(rec.pois, rec.units), 0.0, None)
        self.cats = [set(p.get("category") or []) for p in rec.pois]
        lat, lon = rec.lat, rec.lon
        dx = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2)
        self.dist = 6371.0 * np.sqrt(dx ** 2 + (lat[:, None] - lat[None, :]) ** 2)
        self.names = [f for g in GROUPS.values() for f in g]

    def case(self, liked: list[str], prev: int) -> tuple[np.ndarray, np.ndarray]:
        rec = self.rec
        li = [rec.position[x] for x in liked]
        anchors = list(dict.fromkeys(li + [prev]))
        d = self.dist[li if SCENARIO == "likes" else anchors].min(axis=0)
        u, a = rec.units, rec.adapted
        s = u @ u[li].T
        sa = a @ a[li].T
        coh = float((s[li].sum() - len(li)) / max(len(li) * (len(li) - 1), 1)) if len(li) > 1 else 0.0
        cat = np.array([sum(bool(c & self.cats[j]) for j in li) / len(li) for c in self.cats])
        cols = {
            "pop": rec.popularity, "dist": d, "logdist": np.log(d + 0.05),
            "next": np.log1p(100 * rec.transitions[prev]),
            "sim_mean": s.mean(axis=1), "sim_max": s.max(axis=1), "sim_coh": s.mean(axis=1) * coh,
            "ad_mean": sa.mean(axis=1), "ad_max": sa.max(axis=1),
            "cat": cat, "content_pop": self.content,
        }
        x = np.column_stack([cols[n] for n in self.names])
        mask = np.ones(len(rec.ids), dtype=bool)
        mask[anchors] = False
        return x, mask


def build(city: City, users: list[str], model_users: list[str]):
    """Feature tensors of `users`' cases with a model trained on `model_users`."""
    photos = [p for u in model_users for p in city.by_user[u]]
    model = train(city.pois, city.units, first_visits(photos, city.pois, city.shapes), "ltr")
    rec = Recommender(city.pois, city.index, model, city.cfg)
    feats = Features(city, rec, model)
    out = []
    for (_, liked, held, _, prev) in city.cases(users):
        x, mask = feats.case(liked, prev)
        out.append((x, mask, held))
    return out, rec


def fit(cases, cols: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allx = np.vstack([x[m][:, cols] for x, m, _ in cases])
    mu, sd = allx.mean(axis=0), allx.std(axis=0) + 1e-9

    def loss(w):
        f, g = 0.0, np.zeros_like(w)
        for x, m, held in cases:
            z = (x[:, cols] - mu) / sd
            # the held-out POI is never a like nor the previous visit, so m[held] holds
            sc, zm = z[m] @ w, z[m]
            mx = sc.max()
            e = np.exp(sc - mx)
            p = e / e.sum()
            f += -(z[held] @ w - mx) + np.log(e.sum())
            g -= z[held] - p @ zm
        f += L2 * w @ w
        g += 2 * L2 * w
        return f, g

    w = minimize(loss, np.zeros(len(cols)), jac=True, method="L-BFGS-B").x
    return w, mu, sd


def hits(cases, cols, w, mu, sd):
    out, tops = np.zeros(len(cases)), []
    for n, (x, m, held) in enumerate(cases):
        sc = np.where(m, ((x[:, cols] - mu) / sd) @ w, -np.inf)
        top = np.argsort(-sc, kind="stable")[:K]
        out[n] = held in top
        tops.append(top)
    return out, tops


def hand_service(cases, rec: Recommender, names: list[str]):
    """The production formula (tau 0.3 km, next weight 1), from the same features."""
    ip = names.index("pop")
    out = np.zeros(len(cases))
    for n, (x, m, held) in enumerate(cases):
        if REMOTE:
            sc = x[:, ip]
        elif SCENARIO == "likes":
            sc = x[:, ip] - x[:, names.index("dist")] / rec.distance_scale_km
        else:
            sc = x[:, ip] + x[:, names.index("next")] - x[:, names.index("dist")] / rec.distance_scale_km
        sc = np.where(m, sc, -np.inf)
        out[n] = held in np.argsort(-sc, kind="stable")[:K]
    return out


def main() -> None:
    cfg_name, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    city = City(cfg_name, {"lat": (la0, la1), "lon": (lo0, lo1)})
    users = sorted(city.visits)
    rng = np.random.default_rng(0)
    order = list(users)
    rng.shuffle(order)
    fold = {u: n % FOLDS for n, u in enumerate(order)}
    names = [f for g in GROUPS.values() for f in g]
    print(f"{cfg_name}: POIs {len(city.pois)}, LOO cases {len(city.cases(users))}")

    res = collections.defaultdict(list)
    weights = collections.defaultdict(list)
    tail_flags, coh_flags = [], []
    for f in range(FOLDS):
        train_users = [u for u in users if fold[u] != f]
        test_users = [u for u in users if fold[u] == f]
        inner = {u: n % INNER for n, u in enumerate(train_users)}
        train_cases = []
        for g in range(INNER):
            part, _ = build(city, [u for u in train_users if inner[u] == g],
                            [u for u in train_users if inner[u] != g])
            train_cases += part
        test_cases, rec = build(city, test_users, train_users)
        head = set(np.argsort(-rec.popularity, kind="stable")[:HEAD].tolist())
        tail_flags += [held not in head for _, _, held in test_cases]
        isc = names.index("sim_coh"), names.index("sim_mean")
        for x, m, held in test_cases:
            # coherence of the likes = sim_coh / sim_mean (same for every candidate)
            coh_flags.append(float(x[0, isc[0]] / x[0, isc[1]]) if abs(x[0, isc[1]]) > 1e-9 else 0.0)
        res["production formula"].append(hand_service(test_cases, rec, names))
        for v, groups in VARIANTS.items():
            cols = [names.index(n) for g in groups for n in GROUPS[g]]
            w, mu, sd = fit(train_cases, cols)
            h, _ = hits(test_cases, cols, w, mu, sd)
            res[v].append(h)
            weights[v].append(dict(zip([names[c] for c in cols], np.round(w, 3))))
        print(f"  fold {f}: train cases {len(train_cases)}, test {len(test_cases)}", flush=True)

    tail = np.array(tail_flags)
    coh = np.array(coh_flags)
    hi = coh >= np.quantile(coh, 2 / 3)
    rows = {k: np.concatenate(v) for k, v in res.items()}
    base = rows["learnt gravity (base)"]
    print(f"\n{'hit@10 (users split)':28s} all ({len(base)})  tail ({int(tail.sum())})  coherent likes ({int(hi.sum())})"
          "   vs learnt gravity")
    for k, h in rows.items():
        t = paired(h, base)
        print(f"{k:28s} {h.mean():.3f}       {h[tail].mean():.3f}       {h[hi].mean():.3f}"
              f"               {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['mcnemar_p']:.2g}")
    for name in ("tail", "coherent"):
        sel = tail if name == "tail" else hi
        t = paired(rows[SEMANTIC][sel], base[sel])
        print(f"  all semantics vs base on {name}: {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] "
              f"p={t['mcnemar_p']:.2g}")
    print("\nweights (standardised), mean over folds:")
    for v, ws in weights.items():
        keys = ws[0].keys()
        print(f"  {v}: " + ", ".join(f"{k} {np.mean([w[k] for w in ws]):+.2f}" for k in keys))


if __name__ == "__main__":
    main()
