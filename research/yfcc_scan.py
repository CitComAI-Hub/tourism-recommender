"""Stream the YFCC100M metadata (65 GB SQLite on AWS Open Data, no geo index) by HTTP
ranges and keep the geotagged rows inside Europe, so any city can be cut out later
without another pass. Each 64 MiB chunk is parsed page by page in memory.

Rows whose coordinates live in far-away overflow pages (~3%, photos with very long
descriptions) are skipped. Output: data/raw/yfcc/chunks/NNNNN.jsonl.gz (git-ignored).
Usage: python research/yfcc_scan.py [first_chunk last_chunk]   (env W = workers)
"""
import gzip, json, os, struct, sys, time, urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

URL = "https://multimedia-commons.s3.amazonaws.com/tools/etc/yfcc100m_dataset.sql"
SIZE = 65644027904
PS = U = 1024                      # page size; no reserved bytes in this file
X = U - 35                         # max local payload of a table leaf cell
M = ((U - 12) * 32 // 255) - 23    # min local payload
CHUNK = 64 * 1024 * 1024
LAT, LON = (34.0, 72.0), (-25.0, 45.0)   # Europe
# column index -> output key (photoid is the rowid)
KEEP = {1: "uid", 3: "datetaken", 6: "title", 8: "usertags", 10: "lon", 11: "lat", 12: "accuracy"}
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "yfcc", "chunks")


def get(a, b):
    for t in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            time.sleep(5 * (t + 1))
    raise RuntimeError(f"range {a}-{b} failed")


def varint(b, i):
    v = 0
    for k in range(9):
        c = b[i + k]
        if k == 8:
            return (v << 8) | c, i + 9
        v = (v << 7) | (c & 0x7F)
        if c < 0x80:
            return v, i + k + 1


def ssize(t):
    if t < 12:
        return (0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0)[t]
    return (t - 12) // 2 if t % 2 == 0 else (t - 13) // 2


def parse_chunk(ci):
    path = os.path.join(OUT, f"{ci:05d}.jsonl.gz")
    if os.path.exists(path):
        return ci, -1, 0
    a = ci * CHUNK
    data = get(a, min(SIZE, a + CHUNK) - 1)
    rows, skipped = [], 0
    for p in range(len(data) // PS):
        pg = data[p * PS:(p + 1) * PS]
        hdr = 100 if a == 0 and p == 0 else 0
        if pg[hdr] != 0x0D:            # table leaf pages only
            continue
        for c in range(struct.unpack(">H", pg[hdr + 3:hdr + 5])[0]):
            try:
                cp = struct.unpack(">H", pg[hdr + 8 + 2 * c:hdr + 10 + 2 * c])[0]
                P, i = varint(pg, cp)
                rowid, i = varint(pg, i)
                if P <= X:
                    payload = pg[i:i + P]
                else:
                    K = M + (P - M) % (U - 4)
                    payload = pg[i:i + (K if K <= X else M)]
                hs, j = varint(payload, 0)
                types = []
                while j < hs:
                    t, j = varint(payload, j)
                    types.append(t)
                offs, pos = [], hs
                for t in types[:13]:
                    offs.append(pos)
                    pos += ssize(t)
                if pos > len(payload):
                    skipped += 1
                    continue
                try:
                    lon = float(payload[offs[10]:offs[10] + ssize(types[10])])
                    lat = float(payload[offs[11]:offs[11] + ssize(types[11])])
                except ValueError:
                    continue
                if not (LAT[0] <= lat <= LAT[1] and LON[0] <= lon <= LON[1]):
                    continue
                rec = {"photoid": rowid}
                for k, key in KEEP.items():
                    t = types[k]
                    raw = bytes(payload[offs[k]:offs[k] + ssize(t)])
                    rec[key] = int.from_bytes(raw, "big", signed=True) if 1 <= t <= 6 else raw.decode("utf-8", "replace")
                rows.append(rec)
            except Exception:
                continue
    with gzip.open(path + ".tmp", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    return ci, len(rows), skipped


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n = (SIZE + CHUNK - 1) // CHUNK
    chunks = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(n)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(int(os.environ.get("W", "24"))) as ex:
        for fut in as_completed([ex.submit(parse_chunk, c) for c in chunks]):
            ci, rows, skipped = fut.result()
            done += 1
            print(f"{time.time() - t0:7.0f}s chunk {ci} rows {rows} skipped {skipped} ({done}/{len(chunks)})", flush=True)
