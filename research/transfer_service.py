"""Transfer test with the production code: a behaviour model trained in Valencia,
applied unchanged to another CitCom TEF city, whose photos are only used to evaluate.

Rows:
- no model: Wikipedia popularity + plain embeddings (what a new city gets today)
- Valencia model: popularity predicted from the content of the target POIs and the
  projection learnt in Valencia (no data of the target city at all)
- local model (reference): trained on the target city's own photos, other folds

Usage: PYTHONPATH=research python research/transfer_service.py transfer_milano 45.39 45.54 9.04 9.28
"""
import collections
import sys

import numpy as np

from common import city_photos, folds, load_config, pseudonym
from pmv_turismo.behaviour import first_visits, read_photos, train
from pmv_turismo.common import load_poi_list, processed_dir
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.geometry import assign_photos, load_geometries
from pmv_turismo.recommendation import Recommender

K = 10


def catalog(cfg):
    pois = load_poi_list(processed_dir(cfg) / "poi.jsonld")
    index = load_embedding_index(embeddings_path(cfg))
    pois = [p for p in pois if index.index_of(p["id"]) is not None]
    return pois, index, index.units[[index.index_of(p["id"]) for p in pois]]


def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(x)).astype(float)
    return float(np.corrcoef(r(a), r(b))[0, 1])


def main():
    name, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    src = load_config()
    src["recommendation"]["filter_opening_hours"] = False
    s_pois, _, s_units = catalog(src)
    from common import RAW
    s_photos = read_photos(RAW / "valencia_photos.jsonl.gz")
    valencia = train(s_pois, s_units, first_visits(s_photos, s_pois, load_geometries(src)), "Valencia")

    cfg = load_config(name)
    cfg.setdefault("recommendation", {}).update({"filter_opening_hours": False, "top_n": K})
    for key in ("distance_scale_km", "similarity_weight", "dislike_weight"):
        cfg["recommendation"][key] = src["recommendation"][key]
    pois, index, units = catalog(cfg)
    shapes = load_geometries(cfg)
    raw = city_photos({"lat": (la0, la1), "lon": (lo0, lo1)})
    photos = [{"user": pseudonym(r["uid"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "ts": r["datetaken"]} for r in raw]
    lat = np.array([p["lat"] for p in photos]); lon = np.array([p["lon"] for p in photos])
    vis = collections.defaultdict(dict)
    for j, idx in enumerate(assign_photos(pois, shapes, lat, lon)):
        for k in idx:
            u, ts = photos[k]["user"], photos[k]["ts"]
            vis[u][j] = min(vis[u].get(j, ts), ts)
    days = collections.defaultdict(list)
    for p in photos:
        days[p["user"]].append(p["ts"][:10])
    import datetime as dt
    tourist = {u for u, d in days.items()
               if (dt.date.fromisoformat(max(d)) - dt.date.fromisoformat(min(d))).days <= 21}
    counts = np.zeros(len(pois))
    for v in vis.values():
        counts[list(v)] += 1
    print(f"{name}: POIs {len(pois)}, shapes {len(shapes)}, photos {len(photos)}, users with visits {len(vis)}, "
          f"POIs visited {int((counts > 0).sum())}")

    no_model = Recommender(pois, index, None, cfg)
    transferred = Recommender(pois, index, valencia, cfg)
    print(f"popularity vs real visitors (Spearman): Wikipedia {spearman(no_model.popularity, counts):.3f}, "
          f"predicted by the Valencia model {spearman(transferred.popularity, counts):.3f}")

    import copy
    rank01 = lambda x: np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
    combo = copy.copy(transferred)
    combo.popularity = 1.5 * rank01(no_model.popularity) + 1.5 * rank01(transferred.popularity)
    combo_raw = copy.copy(combo)
    combo_raw.adapted = no_model.adapted
    pop_only = copy.copy(transferred)
    pop_only.adapted = no_model.adapted
    print(f"  Spearman of Wikipedia + predicted (rank mean): {spearman(combo.popularity, counts):.3f}")
    fold = folds(vis.keys())
    by_user = collections.defaultdict(list)
    for p in photos:
        by_user[p["user"]].append(p)
    local = {}
    for f in range(5):
        tr = [p for u, ps in by_user.items() if fold.get(u) != f for p in ps]
        local[f] = Recommender(pois, index, train(pois, units, first_visits(tr, pois, shapes), "local"), cfg)
    ids = [p["id"] for p in pois]
    hits = collections.defaultdict(list)
    for u in sorted(t for t in tourist if len(vis.get(t, {})) >= 3):
        seq = sorted(vis[u], key=vis[u].get)
        for held in seq[-2:]:
            liked = [ids[x] for x in seq if x != held][-10:]
            rows = (("no model (Wikipedia)", no_model), ("Valencia model, transferred", transferred),
                    ("Valencia predicted pop, raw embeddings", pop_only),
                    ("Wikipedia + predicted pop, adapted", combo),
                    ("Wikipedia + predicted pop, raw emb.", combo_raw),
                    ("local model (reference)", local[fold[u]]))
            for label, rec in rows:
                hits[label].append(ids[held] in {r["id"] for r in rec.recommend(liked=liked, top_n=K)})
            rng = np.random.default_rng(len(hits["random"]))
            pool = [i for i in range(len(ids)) if ids[i] not in liked]
            hits["random"].append(ids.index(ids[held]) in set(rng.choice(pool, K, replace=False)))
    n = len(hits["random"])
    print(f"leave-one-out on tourists: cases {n} (95% CI ~ +-{1.96 * (0.25 / max(n, 1)) ** 0.5:.3f})")
    for label, h in hits.items():
        print(f"  {label:32s} hit@10 {np.mean(h):.3f}")


if __name__ == "__main__":
    main()
