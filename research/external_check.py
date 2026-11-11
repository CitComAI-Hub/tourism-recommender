"""External check of the learnt popularity against official visitor counts.

The popularity the service ranks with is learnt from Flickr photos (2004-2014).
Is it the popularity of the places, or of what Flickr users photograph? Official
yearly visitor counts of museums and monuments (research/external_visitors.csv:
Ajuntament de València yearbook 2025 = year 2024; Comune di Milano open data
2004-2017, CC BY) are an independent measure of the same thing.

Spearman correlation with log(visitors per year) over the matched POIs, with a
bootstrap 95% CI, for:
- Flickr visitors (all users / tourists only), what the service uses;
- Wikipedia popularity (1 - relevance), the fallback without a behaviour model;
- popularity predicted from content with the POI masked (as if nobody had
  photographed it: 5 folds over the matched POIs);
- the service's popularity (Flickr where visited, predicted otherwise).
And the photo bias by kind of place: median log(official / Flickr) per category
(museums are photographed inside less than towers and markets?).

Usage: PYTHONPATH=research python research/external_check.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from protocol import City

from pmv_turismo.behaviour import first_visits, train
from pmv_turismo.recommendation import Recommender

HERE = Path(__file__).resolve().parent
CITIES = {"valencia": ("base", {"lat": (39.40, 39.53), "lon": (-0.44, -0.27)}),
          "milano": ("transfer_milano", {"lat": (45.39, 45.54), "lon": (9.04, 9.28)})}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    for r, x in ((ra, a), (rb, b)):
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        if np.ptp(a[i]) > 0 and np.ptp(b[i]) > 0:
            vals.append(spearman(a[i], b[i]))
    return spearman(a, b), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def rows(city: str) -> list[dict]:
    with open(HERE / "external_visitors.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["city"] == city and "excluded" not in r["note"]]


def main() -> None:
    for name, (cfg, bbox) in CITIES.items():
        c = City(cfg, bbox)
        ext = rows(name)
        idx, y, foreign = [], [], []
        for r in ext:
            hit = [j for j, pid in enumerate(c.ids) if pid.rsplit(":", 1)[-1].split("-")[-1] == r["poi_osm"]]
            if not hit:
                print(f"  not in catalog: {r['name']}")
                continue
            idx.append(hit[0])
            y.append(math.log(float(r["visitors_per_year"]) + 1))
            foreign.append(math.log(float(r["foreign_visitors"]) + 1) if r["foreign_visitors"] else np.nan)
        idx, y, foreign = np.array(idx), np.array(y), np.array(foreign)

        photos = [p for u in c.by_user for p in c.by_user[u]]
        visits = first_visits(photos, c.pois, c.shapes)
        model = train(c.pois, c.units, visits, "external")
        rec = Recommender(c.pois, c.index, model, c.cfg)
        flickr = np.log1p(model.visitors)
        tour = np.zeros(len(c.pois))
        for u, v in visits.items():
            if u in c.tourists:
                tour[list(v)] += 1
        wiki = np.array([1.0 - float(p["relevance"]) if isinstance(p.get("relevance"), (int, float)) else 0.0
                         for p in c.pois])
        masked = np.zeros(len(idx))
        order = np.random.default_rng(0).permutation(len(idx))
        for f in range(5):
            part = order[f::5]
            m = train(c.pois, c.units, visits, "masked", unknown=idx[part].tolist())
            masked[part] = m.predicted_popularity(c.pois, c.units)[idx[part]]

        print(f"\n== {name}: {len(idx)} places with official counts; Spearman with log(visitors/year) [95% CI]")
        for label, x in (("Flickr visitors, all users", flickr[idx]), ("Flickr visitors, tourists", np.log1p(tour[idx])),
                         ("service popularity", rec.popularity[idx]), ("Wikipedia popularity", wiki[idx]),
                         ("predicted from content (masked)", masked)):
            s, lo, hi = boot(x, y)
            print(f"   {label:34s} {s:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        ok = ~np.isnan(foreign)
        if ok.sum() >= 8:
            s, lo, hi = boot(np.log1p(tour[idx][ok]), foreign[ok])
            s2, lo2, hi2 = boot(flickr[idx][ok], foreign[ok])
            print(f"   foreign visitors ({ok.sum()} places): tourists {s:+.2f} [{lo:+.2f}, {hi:+.2f}], "
                  f"all users {s2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]")
        print("   photo bias by kind: log(official) - log(Flickr visitors), residual of a linear fit")
        a, b = np.polyfit(flickr[idx], y, 1)
        res = y - (a * flickr[idx] + b)
        kinds: dict[str, list[float]] = {}
        for j, rr in zip(idx, res, strict=True):
            kinds.setdefault((c.pois[j].get("category") or ["?"])[0], []).append(float(rr))
        for k, v in sorted(kinds.items(), key=lambda t: -len(t[1])):
            print(f"     {k:14s} n={len(v):2d} median residual {np.median(v):+.2f}")
        print("   largest disagreements (residual):")
        for j, rr in sorted(zip(idx, res, strict=True), key=lambda t: -abs(t[1]))[:6]:
            print(f"     {c.pois[j].get('name')[:40]:40s} Flickr {int(model.visitors[j]):4d}  residual {rr:+.2f}")


if __name__ == "__main__":
    main()
