"""Check that the production code (behaviour.train + Recommender) reproduces the
research numbers: same tourists, folds and leave-one-out as loo_eval.py.
Also exports the Valencia photos (pseudonymised) used to train the shipped model."""
import collections
import gzip
import json

import numpy as np

from common import REPO, RAW, city_photos, folds, load_catalog, load_config, pseudonym, visits
from pmv_turismo.behaviour import assign_visits, train
from pmv_turismo.geometry import assign_photos, load_geometries
from pmv_turismo.recommendation import Recommender
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.common import load_poi_list, processed_dir


def main():
    cfg = load_config()
    cfg["recommendation"]["filter_opening_hours"] = False
    pois = load_poi_list(processed_dir(cfg) / "poi.jsonld")
    index = load_embedding_index(embeddings_path(cfg))
    _, emb, radius = load_catalog(cfg)
    photos = [{"user": pseudonym(r["uid"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "ts": r["datetaken"]}
              for r in city_photos()]
    out = RAW / "valencia_photos.jsonl.gz"
    with gzip.open(out, "wt") as f:
        for p in photos:
            f.write(json.dumps(p) + "\n")
    print(f"exported {len(photos)} photos to {out.relative_to(REPO)}")

    shapes = {} if "--radius" in __import__("sys").argv else load_geometries(cfg)
    print("assignment:", "real OSM shapes" if shapes else "fixed radii")
    # first visit time per (user, POI), with the same assignment the service trains on
    lat = np.array([p["lat"] for p in photos]); lon = np.array([p["lon"] for p in photos])
    vis = collections.defaultdict(dict)
    if shapes:
        for j, idx in enumerate(assign_photos(pois, shapes, lat, lon)):
            for k in idx:
                u, ts = photos[k]["user"], photos[k]["ts"]
                vis[u][j] = min(vis[u].get(j, ts), ts)
        _, tourists = visits(city_photos(), pois, radius)
    else:
        vis, tourists = visits(city_photos(), pois, radius)
    fold = folds(vis.keys())
    by_user = collections.defaultdict(list)
    for p in photos:
        by_user[p["user"]].append(p)
    units = index.units[[index.index_of(p["id"]) for p in pois]]
    recs = {}
    for f in range(5):
        train_photos = [p for u, ps in by_user.items() if fold.get(u) != f for p in ps]
        model = train(pois, units, assign_visits(train_photos, pois, shapes), "fold")
        recs[f] = Recommender(pois, index, model, cfg)
    rec_nomodel = Recommender(pois, index, None, cfg)
    ids = [p["id"] for p in pois]
    hits = collections.defaultdict(list)
    for u in sorted(t for t in tourists if len(vis.get(t, {})) >= 3):
        seq = sorted(vis[u], key=vis[u].get)
        for held in seq[-2:]:
            liked = [ids[x] for x in seq if x != held][-10:]
            for name, rec in (("service with behaviour model", recs[fold[u]]), ("service without model (Wikipedia)", rec_nomodel)):
                top = [r["id"] for r in rec.recommend(liked=liked, top_n=10)]
                hits[name].append(ids[held] in top)
            # references under the same protocol
            liked_idx = {recs[fold[u]].position[x] for x in liked}
            pop = recs[fold[u]].popularity
            order = [i for i in np.argsort(-pop, kind="stable") if i not in liked_idx][:10]
            hits["popularity only (same model)"].append(recs[fold[u]].position[ids[held]] in order)
            rng = np.random.default_rng(len(hits["random"]))
            pool = [i for i in range(len(ids)) if i not in liked_idx]
            hits["random"].append(recs[fold[u]].position[ids[held]] in set(rng.choice(pool, 10, replace=False)))
    for name, h in hits.items():
        print(f"{name:36s} hit@10 {np.mean(h):.3f}  (cases {len(h)})")


if __name__ == "__main__":
    main()
