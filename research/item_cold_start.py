"""Experiment 1 - item cold start: does a POI's embedding predict how many people visit it?

Leave-one-POI-out ridge regression of log(1 + distinct Flickr visitors) on the
embedding (PCA) and the category, compared with the current popularity signal
(Wikipedia pageviews). Spearman correlation, overall and split by whether the
description comes from Wikipedia or from a template.
"""
import numpy as np

from common import load_catalog, load_config, city_photos, visits


def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(x)).astype(float)
    return float(np.corrcoef(r(a), r(b))[0, 1])


def loo_ridge(X, y, lam=1.0):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, ym = X[tr].mean(0), y[tr].mean()
        A = X[tr] - mu
        w = np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ (y[tr] - ym))
        pred[i] = (X[i] - mu) @ w + ym
    return pred


def main():
    cfg = load_config()
    pois, emb, radius = load_catalog(cfg)
    vis, tourists = visits(city_photos(), pois, radius)
    for label, users in (("all users", vis.keys()), ("tourists", tourists)):
        counts = np.zeros(len(pois))
        for u in users:
            for j in vis.get(u, {}):
                counts[j] += 1
        y = np.log1p(counts)
        cats = sorted({c for p in pois for c in p.get("category", [])})
        C = np.array([[1.0 if c in p.get("category", []) else 0.0 for c in cats] for p in pois])
        wiki = np.array([1 - p["relevance"] if isinstance(p.get("relevance"), (int, float)) else np.nan for p in pois])
        wiki = np.where(np.isnan(wiki), np.nanmedian(wiki), wiki)
        gen = np.array([bool(p.get("descriptionGenerated")) for p in pois])
        U, s, _ = np.linalg.svd(emb - emb.mean(0), full_matrices=False)
        print(f"\n== {label}: POIs {len(pois)}, visited {int((counts > 0).sum())}")
        print(f"{'predictor':42s} all    wikipedia-desc  template-desc")
        def show(name, pred):
            print(f"{name:42s} {spearman(pred, y):.3f}  {spearman(pred[~gen], y[~gen]):.3f}           {spearman(pred[gen], y[gen]):.3f}")
        show("category", loo_ridge(C, y))
        show("Wikipedia pageviews (current)", wiki)
        for k in (8, 16, 32):
            show(f"embedding PCA{k} + category", loo_ridge(np.hstack([U[:, :k] * s[:k], C]), y))
        show("pageviews + embedding PCA16 + category", loo_ridge(np.hstack([U[:, :16] * s[:16], C, wiki[:, None]]), y))


if __name__ == "__main__":
    main()
