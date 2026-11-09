"""Experiment 2 - leave-one-out on real visits, including behaviour-adapted semantics.

For every evaluated user (tourists with >= 3 catalog POIs) the last two visits are
held out in turn; the other visits (up to 10) are the "likes". Users are split in
5 folds: popularity, co-visits and the learned projection only use the other folds
(all users there, tourists and locals).

"adapted semantics" = ridge projection W of the POI embeddings onto item factors of
the training co-visit matrix (cosine, top-k eigenvectors). Z = E W exists for every
POI, also those nobody visited, which is what plain collaborative filtering lacks.
"""
import collections
import copy
import math

import numpy as np

from common import city_photos, distance_matrix_km, folds, load_catalog, load_config, minmax, visits
from pmv_turismo.recommendation import apply_style, available_styles, recommend

K = 10
FACTORS, LAMBDA = 16, 1.0


def train_stats(vis, fold, f, n):
    pop = np.zeros(n)
    co = np.zeros((n, n))
    for u, v in vis.items():
        if fold.get(u) == f:
            continue
        js = list(v)
        pop[js] += 1
        for a in js:
            co[a, js] += 1
    np.fill_diagonal(co, 0)
    return pop, co


def adapted_embeddings(emb, pop, co, k=FACTORS, lam=LAMBDA):
    norm = np.sqrt(np.outer(pop, pop)) + 1e-9
    S = co / norm
    w, V = np.linalg.eigh(S)
    top = np.argsort(-w)[:k]
    F = V[:, top] * np.sqrt(np.clip(w[top], 0, None))
    seen = pop > 0
    A = emb[seen]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ F[seen])
    Z = emb @ W
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def main():
    cfg = load_config()
    pois, emb, radius = load_catalog(cfg)
    n = len(pois)
    ids = [p["id"] for p in pois]
    pos = {i: j for j, i in enumerate(ids)}
    vis, tourists = visits(city_photos(), pois, radius)
    evaluated = {u for u in tourists if len(vis.get(u, {})) >= 3}
    fold = folds(vis.keys())
    D = distance_matrix_km(pois)
    wiki = np.array([1 - p["relevance"] if isinstance(p.get("relevance"), (int, float)) else 0.5 for p in pois])
    stats = {f: train_stats(vis, fold, f, n) for f in range(5)}
    Z = {f: adapted_embeddings(emb, *stats[f]) for f in range(5)}
    # "City without visit data": each POI's popularity predicted from its embedding and
    # category by a ridge fit on the OTHER POIs' training counts (5-fold over POIs too).
    cats = sorted({c for p in pois for c in p.get("category", [])})
    C = np.array([[1.0 if c in p.get("category", []) else 0.0 for c in cats] for p in pois])
    U, sv, _ = np.linalg.svd(emb - emb.mean(0), full_matrices=False)
    Xs = np.hstack([U[:, :16] * sv[:16], C])
    item_fold = np.arange(n) % 5
    np.random.default_rng(1).shuffle(item_fold)
    def predicted_pop(f):
        y = np.log1p(stats[f][0]); out = np.zeros(n)
        for g in range(5):
            tr, te = item_fold != g, item_fold == g
            mu, ym = Xs[tr].mean(0), y[tr].mean(); A = Xs[tr] - mu
            w = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ (y[tr] - ym))
            out[te] = (Xs[te] - mu) @ w + ym
        return out
    PP = {f: predicted_pop(f) for f in range(5)}

    base = copy.deepcopy(cfg)
    base["recommendation"].update({"top_n": 60, "filter_opening_hours": False})
    base.setdefault("parameters", {}).setdefault("planning", {})["start_location"] = {}

    def system(style):
        c = apply_style(base, style)
        c["recommendation"]["gamma"] = 0.0
        return lambda liked, f: [pos[x["id"]] for x in recommend(
            c, poi_data=pois, liked_poi_ids=[ids[j] for j in liked], include_liked=False)["results"]]

    def scored(fn):
        def run(liked, f):
            s = fn(liked, f).astype(float)
            s[liked] = -1e9
            return list(np.argsort(-s, kind="stable")[:60])
        return run

    lp = lambda f: np.log1p(stats[f][0])
    near = lambda liked: -D[:, liked].min(axis=1)
    cf = lambda liked, f: (stats[f][1][liked] / np.sqrt(np.outer(stats[f][0][liked] + 1, stats[f][0] + 1))).sum(0)
    raw_sem = lambda liked: (emb @ emb[liked].T).max(axis=1)
    ad_sem = lambda liked, f: (Z[f] @ Z[f][liked].T).mean(axis=1)
    methods = {
        "random": scored(lambda liked, f: np.random.default_rng(len(liked) * 7919 + liked[0]).random(n)),
        "popular (Wikipedia pageviews)": scored(lambda liked, f: wiki),
        "popular (visits, train users)": scored(lambda liked, f: lp(f)),
        "near liked POIs": scored(lambda liked, f: near(liked)),
        "item-item CF": scored(cf),
        "semantic, raw embeddings": scored(lambda liked, f: raw_sem(liked)),
        "semantic, adapted to behaviour": scored(ad_sem),
        "gravity: log(pop) - d/0.3km": scored(lambda liked, f: lp(f) + near(liked) / 0.3),
        "NO VISIT DATA: gravity w/ Wikipedia pop": scored(lambda liked, f: 3 * wiki + near(liked) / 0.3),
        "NO VISIT DATA: gravity w/ semantic pop": scored(lambda liked, f: PP[f] + near(liked) / 0.3),
        "NO VISIT DATA: semantic pop only": scored(lambda liked, f: PP[f]),
        "gravity + 0.3 raw semantic": scored(lambda liked, f: minmax(lp(f) + near(liked) / 0.3) + 0.3 * minmax(raw_sem(liked))),
        "gravity + 0.3 adapted semantic": scored(lambda liked, f: minmax(lp(f) + near(liked) / 0.3) + 0.3 * minmax(ad_sem(liked, f))),
    }
    for s in available_styles(cfg):
        methods[f"current system: {s}"] = system(s)

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for u in sorted(evaluated):
        f = fold[u]
        seq = sorted(vis[u], key=vis[u].get)
        head = set(np.argsort(-stats[f][0])[:20])
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
                if stats[f][0][held] <= 2:
                    res[name]["rare"].append(hit)
                res[name]["cov"].extend(rank[:K])
    cases = len(res["random"]["hit"])
    print(f"evaluated tourists {len(evaluated)}, cases {cases} (hit@10 95% CI ~ +-{1.96 * math.sqrt(0.25 / cases):.3f}); "
          f"held-out outside top-20: {len(res['random']['tail'])}; held-out with <=2 training visitors: {len(res['random']['rare'])}")
    print(f"{'method':34s} hit@10  mrr    hit@10 non-top20  hit@10 rare  coverage")
    for name, r in res.items():
        print(f"{name:34s} {np.mean(r['hit']):.3f}   {np.mean(r['mrr']):.3f}  {np.mean(r['tail']):.3f}             "
              f"{np.mean(r['rare']) if r['rare'] else float('nan'):.3f}        {len(set(r['cov'])) / n:.2f}")


if __name__ == "__main__":
    main()
