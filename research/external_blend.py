"""Idea H: does blending Wikipedia into the Flickr popularity move it closer to the
official visitor counts (research/external_visitors.csv)? Spearman of
z(log Flickr visitors) + beta * z(Wikipedia popularity) on the matched places.
The cost in hit@10 is measured by research/context.py (variant H)."""
import math

import numpy as np
from external_check import CITIES, boot, rows
from protocol import City

from pmv_turismo.behaviour import first_visits, train


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


for name, (cfg, bbox) in CITIES.items():
    c = City(cfg, bbox)
    idx, y = [], []
    for r in rows(name):
        hit = [j for j, pid in enumerate(c.ids) if pid.rsplit(":", 1)[-1].split("-")[-1] == r["poi_osm"]]
        if hit:
            idx.append(hit[0])
            y.append(math.log(float(r["visitors_per_year"]) + 1))
    idx, y = np.array(idx), np.array(y)
    visits = first_visits([p for ps in c.by_user.values() for p in ps], c.pois, c.shapes)
    model = train(c.pois, c.units, visits, "blend")
    fl = np.log1p(model.visitors)
    wiki = np.array([1.0 - float(p["relevance"]) if isinstance(p.get("relevance"), (int, float)) else 0.0 for p in c.pois])
    print(f"== {name} ({len(idx)} places)")
    for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
        s, lo, hi = boot(z(fl)[idx] + beta * z(wiki)[idx], y)
        print(f"   beta {beta:4.2f}: Spearman {s:+.2f} [{lo:+.2f}, {hi:+.2f}]")
