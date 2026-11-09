"""Experiment 4 - transfer to a city with no visit data.

Everything learnt comes from Valencia: the ridge model that predicts a POI's
popularity from its embedding + category, and the behaviour-adapted projection W.
In the target city only its own catalog (SDM PointOfInterest + Wikipedia pageviews)
is used; its Flickr visits are held out for evaluation only (an "oracle" row shows
what local visit data would give, as an upper reference).

Usage: PYTHONPATH=research python research/transfer_eval.py transfer_milano 45.39 45.54 9.04 9.28
"""
import collections
import copy
import math
import sys

import numpy as np

from common import city_photos, distance_matrix_km, folds, load_catalog, load_config, minmax, visits
from loo_eval import adapted_embeddings, train_stats
from pmv_turismo.recommendation import apply_style, recommend

K = 10


def categories(pois, vocab):
    return np.array([[1.0 if c in p.get("category", []) else 0.0 for c in vocab] for p in pois])


def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(x)).astype(float)
    return float(np.corrcoef(r(a), r(b))[0, 1])


def main():
    cfg_name, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    # --- source city: Valencia ---
    src_cfg = load_config()
    s_pois, s_emb, s_rad = load_catalog(src_cfg)
    s_vis, _ = visits(city_photos(), s_pois, s_rad)
    s_pop = np.zeros(len(s_pois))
    for v in s_vis.values():
        s_pop[list(v)] += 1
    vocab = sorted({c for p in s_pois for c in p.get("category", [])})
    mean = s_emb.mean(0)
    _, _, Vt = np.linalg.svd(s_emb - mean, full_matrices=False)
    feats = lambda e, pois: np.hstack([(e - mean) @ Vt[:16].T, categories(pois, vocab)])
    Xs, y = feats(s_emb, s_pois), np.log1p(s_pop)
    mu, ym = Xs.mean(0), y.mean()
    w_pop = np.linalg.solve((Xs - mu).T @ (Xs - mu) + np.eye(Xs.shape[1]), (Xs - mu).T @ (y - ym))
    s_co = train_stats(s_vis, {}, -1, len(s_pois))[1]
    # projection learnt in Valencia, applied to the target embeddings
    norm = np.sqrt(np.outer(s_pop, s_pop)) + 1e-9
    wv, V = np.linalg.eigh(s_co / norm)
    top = np.argsort(-wv)[:16]
    F = V[:, top] * np.sqrt(np.clip(wv[top], 0, None))
    seen = s_pop > 0
    W = np.linalg.solve(s_emb[seen].T @ s_emb[seen] + np.eye(s_emb.shape[1]), s_emb[seen].T @ F[seen])

    # --- target city ---
    cfg = load_config(cfg_name)
    pois, emb, rad = load_catalog(cfg)
    n = len(pois)
    ids = [p["id"] for p in pois]
    pos = {i: j for j, i in enumerate(ids)}
    vis, tourists = visits(city_photos({"lat": (la0, la1), "lon": (lo0, lo1)}), pois, rad)
    counts = np.zeros(n)
    for v in vis.values():
        counts[list(v)] += 1
    sem_pop = (feats(emb, pois) - mu) @ w_pop + ym
    wiki = np.array([1 - p["relevance"] if isinstance(p.get("relevance"), (int, float)) else 0.5 for p in pois])
    Z = emb @ W
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    D = distance_matrix_km(pois)
    print(f"target: {cfg_name}, POIs {n}, visited {int((counts > 0).sum())}, photo users {len(vis)}")
    print(f"popularity prediction (Spearman vs real visitors): Wikipedia {spearman(wiki, counts):.3f}, "
          f"semantic model from Valencia {spearman(sem_pop, counts):.3f}, "
          f"both (rank mean) {spearman(minmax(np.argsort(np.argsort(wiki))) + minmax(np.argsort(np.argsort(sem_pop))), counts):.3f}")

    evaluated = sorted(u for u in tourists if len(vis.get(u, {})) >= 3)
    fold = folds(vis.keys())
    oracle = {f: np.log1p(train_stats(vis, fold, f, n)[0]) for f in range(5)}
    near = lambda liked: -D[:, liked].min(axis=1)
    rk = lambda x: minmax(np.argsort(np.argsort(x)).astype(float))
    base = copy.deepcopy(cfg)
    base["recommendation"].update({"top_n": 60, "filter_opening_hours": False})
    base.setdefault("parameters", {}).setdefault("planning", {})["start_location"] = {}
    explorer = apply_style(base, "explorer")
    explorer["recommendation"]["gamma"] = 0.0

    def scored(fn):
        def run(liked, f):
            s = fn(liked, f).astype(float)
            s[liked] = -1e9
            return list(np.argsort(-s, kind="stable")[:60])
        return run

    methods = {
        "random": scored(lambda l, f: np.random.default_rng(len(l) * 7919 + l[0]).random(n)),
        "current system: explorer": lambda l, f: [pos[x["id"]] for x in recommend(
            explorer, poi_data=pois, liked_poi_ids=[ids[j] for j in l], include_liked=False)["results"]],
        "Wikipedia popularity": scored(lambda l, f: wiki),
        "semantic popularity (from Valencia)": scored(lambda l, f: sem_pop),
        "near liked POIs": scored(lambda l, f: near(l)),
        "adapted semantic (W from Valencia)": scored(lambda l, f: (Z @ Z[l].T).mean(axis=1)),
        "gravity, Wikipedia pop": scored(lambda l, f: 3 * rk(wiki) + near(l) / 0.3),
        "gravity, semantic pop": scored(lambda l, f: 3 * rk(sem_pop) + near(l) / 0.3),
        "gravity, Wikipedia + semantic pop": scored(lambda l, f: 1.5 * rk(wiki) + 1.5 * rk(sem_pop) + near(l) / 0.3),
        "gravity, W+S pop, + 0.3 adapted sem": scored(lambda l, f: minmax(1.5 * rk(wiki) + 1.5 * rk(sem_pop) + near(l) / 0.3)
                                                      + 0.3 * minmax((Z @ Z[l].T).mean(axis=1))),
        "ORACLE gravity, local visit pop": scored(lambda l, f: oracle[f] + near(l) / 0.3),
    }
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for u in evaluated:
        f = fold[u]
        seq = sorted(vis[u], key=vis[u].get)
        head = set(np.argsort(-counts)[:20])
        for held in seq[-2:]:
            liked = [x for x in seq if x != held][-10:]
            for name, m in methods.items():
                rank = m(liked, f)
                r = rank.index(held) + 1 if held in rank else None
                hit = 1.0 if r and r <= K else 0.0
                res[name]["hit"].append(hit)
                res[name]["mrr"].append(1 / r if r else 0.0)
                if held not in head:
                    res[name]["tail"].append(hit)
    cases = len(res["random"]["hit"])
    print(f"evaluated tourists {len(evaluated)}, cases {cases} (hit@10 95% CI ~ +-{1.96 * math.sqrt(0.25 / max(cases, 1)):.3f}), "
          f"held-out outside top-20: {len(res['random']['tail'])}")
    print(f"{'method':40s} hit@10  mrr    hit@10 non-top20")
    for name, r in res.items():
        print(f"{name:40s} {np.mean(r['hit']):.3f}   {np.mean(r['mrr']):.3f}  {np.mean(r['tail']):.3f}")


if __name__ == "__main__":
    main()
