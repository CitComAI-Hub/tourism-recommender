"""One pass over the European YFCC rows: cache the photos of several cities
(same cache files as common.city_photos) and print how many each one has.
Usage: PYTHONPATH=research python research/yfcc_cities.py cities.json
cities.json: {"name": {"lat": [lat0, lat1], "lon": [lon0, lon1]}, ...}"""
import glob
import gzip
import json
import sys

from common import RAW

cities = json.load(open(sys.argv[1]))
out = {n: [] for n in cities}
for path in sorted(glob.glob(str(RAW / "chunks" / "*.jsonl.gz"))):
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            la, lo = float(r["lat"]), float(r["lon"])
            for n, b in cities.items():
                if b["lat"][0] <= la <= b["lat"][1] and b["lon"][0] <= lo <= b["lon"][1]:
                    out[n].append(r)
for n, b in cities.items():
    cache = RAW / f"city_{b['lat'][0]}_{b['lon'][0]}.jsonl.gz"
    with gzip.open(cache, "wt") as f:
        for r in out[n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    users = {r["uid"] for r in out[n]}
    street = sum(1 for r in out[n] if isinstance(r.get("accuracy"), int) and r["accuracy"] >= 14)
    print(f"{n:12s} photos {len(out[n]):7d}  street-level {street:7d}  users {len(users):6d}")
