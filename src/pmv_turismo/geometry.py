"""Real OSM geometry of the catalog POIs, to tell which POI a photo shows.

A fixed radius around the POI's point fails both ways: 150 m around a small
"attraction" swallows the photos of the square next to it, and no radius fits a
park 8 km long. Here each POI is matched by its shape:

- polygons (closed ways, multipolygon relations): photo inside, or within
  BUFFER_KM of the edge;
- lines (bridges, walls...): within POINT_RADIUS_KM of the line;
- points (nodes) and POIs without geometry: within POINT_RADIUS_KM.

A photo counts as a visit to every POI it matches: someone photographing the
fountain in a square visited both the square and the fountain, and a visitor of
a museum inside the City of Arts and Sciences visited both. Picking a single
"most specific" POI instead gave every square's photos to the statue in it.

Geometries come from Overpass (`out geom`) by OSM id and are stored next to the
catalog in `poi_geometry.json` (not part of the SDM entities).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests

logger = logging.getLogger(__name__)

POINT_RADIUS_KM = 0.040
BUFFER_KM = 0.015
EARTH_KM = 6371.0


def geometry_path(config: dict) -> Path:
    from pmv_turismo.common import processed_dir

    return processed_dir(config) / "poi_geometry.json"


def osm_ref(poi_id: str) -> Optional[tuple[str, int]]:
    """`urn:ngsi-ld:PointOfInterest:way:123` -> ("way", 123)."""
    parts = str(poi_id).split(":")
    if len(parts) >= 2 and parts[-2] in ("node", "way", "relation") and parts[-1].isdigit():
        return parts[-2], int(parts[-1])
    return None


def fetch_geometries(pois: Sequence[Dict[str, Any]], overpass_urls: Sequence[str], timeout: int = 180) -> Dict[str, Any]:
    """{poi id: {"kind": "polygon"|"line", "rings": [[[lon, lat], ...], ...]}} for ways and relations."""
    refs: Dict[tuple[str, int], str] = {}
    for p in pois:
        ref = osm_ref(p.get("id", ""))
        if ref and ref[0] != "node":
            refs[ref] = str(p["id"])
    if not refs:
        return {}
    ways = ",".join(str(i) for kind, i in refs if kind == "way")
    rels = ",".join(str(i) for kind, i in refs if kind == "relation")
    query = f"[out:json][timeout:{timeout}];(" + (f"way(id:{ways});" if ways else "") + (
        f"relation(id:{rels});" if rels else "") + ");out geom;"
    last_error: Optional[Exception] = None
    for url in overpass_urls:
        try:
            resp = requests.post(url, data={"data": query}, timeout=timeout + 30,
                                 headers={"User-Agent": "pmv-turismo (geometry)"})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break
        except Exception as exc:  # try the next mirror
            last_error = exc
            logger.warning("Overpass %s failed: %s", url, exc)
    else:
        raise RuntimeError(f"No Overpass endpoint answered: {last_error}")

    out: Dict[str, Any] = {}
    for el in elements:
        pid = refs.get((el.get("type"), el.get("id")))
        if not pid:
            continue
        if el["type"] == "way":
            ring = [[g["lon"], g["lat"]] for g in el.get("geometry") or []]
            if len(ring) >= 2:
                closed = len(ring) >= 4 and ring[0] == ring[-1]
                out[pid] = {"kind": "polygon" if closed else "line", "rings": [ring]}
        else:
            rings = [[[g["lon"], g["lat"]] for g in m.get("geometry") or []]
                     for m in el.get("members") or [] if m.get("type") == "way" and m.get("role") in ("outer", "")]
            rings = [r for r in rings if len(r) >= 2]
            if rings:
                kind = "polygon" if el.get("tags", {}).get("type") in ("multipolygon", "boundary") else "line"
                out[pid] = {"kind": kind, "rings": rings}
    logger.info("Geometries: %d of %d ways/relations", len(out), len(refs))
    return out


def save_geometries(config: dict) -> None:
    """CLI step `geometry`: fetch and store the shapes of the current catalog."""
    from pmv_turismo.common import load_poi_list, processed_dir

    pois = load_poi_list(processed_dir(config) / "poi.jsonld")
    urls = config.get("ingestion", {}).get("overpass_urls") or ["https://overpass-api.de/api/interpreter"]
    shapes = fetch_geometries(pois, urls)
    geometry_path(config).write_text(json.dumps(shapes), encoding="utf-8")
    print(f"geometry: {len(shapes)} shapes saved to {geometry_path(config)}")


def load_geometries(config: dict) -> Dict[str, Any]:
    path = geometry_path(config)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class _Projector:
    """Equirectangular projection to km around a reference latitude (city scale)."""

    def __init__(self, lat0: float) -> None:
        self.kx = math.radians(1) * EARTH_KM * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * EARTH_KM

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(lon) * self.kx, np.asarray(lat) * self.ky


def _segments(rings: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = np.concatenate([r[:-1] for r in rings])
    b = np.concatenate([r[1:] for r in rings])
    return a, b


def _distance_to_segments(px: np.ndarray, py: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest segment (points x segments, chunked)."""
    out = np.full(px.shape, np.inf)
    d = b - a
    length2 = np.maximum((d ** 2).sum(axis=1), 1e-12)
    for s in range(0, len(a), 512):
        ax, ay = a[s:s + 512, 0], a[s:s + 512, 1]
        dx, dy, l2 = d[s:s + 512, 0], d[s:s + 512, 1], length2[s:s + 512]
        t = np.clip(((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / l2, 0.0, 1.0)
        cx, cy = ax + t * dx, ay + t * dy
        out = np.minimum(out, np.sqrt((px[:, None] - cx) ** 2 + (py[:, None] - cy) ** 2).min(axis=1))
    return out


def _inside(px: np.ndarray, py: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Even-odd rule over all ring segments (works with unassembled multipolygon members)."""
    inside = np.zeros(px.shape, dtype=bool)
    for s in range(0, len(a), 512):
        ax, ay, bx, by = a[s:s + 512, 0], a[s:s + 512, 1], b[s:s + 512, 0], b[s:s + 512, 1]
        crosses = (ay > py[:, None]) != (by > py[:, None])
        x_at = ax + (py[:, None] - ay) * (bx - ax) / np.where(by - ay == 0, 1e-12, by - ay)
        inside ^= np.logical_xor.reduce(crosses & (px[:, None] < x_at), axis=1)
    return inside


def assign_photos(pois: Sequence[Dict[str, Any]], shapes: Dict[str, Any],
                  lat: np.ndarray, lon: np.ndarray) -> List[np.ndarray]:
    """For each POI, the indices of the photos that show it."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if not len(pois) or not len(lat):
        return [np.zeros(0, dtype=int) for _ in pois]
    proj = _Projector(float(np.mean([p["location"]["coordinates"][1] for p in pois])))
    px, py = proj(lon, lat)
    out: List[np.ndarray] = []
    for poi in pois:
        shape = shapes.get(str(poi.get("id")))
        if not shape:
            cx, cy = proj(np.array([poi["location"]["coordinates"][0]]), np.array([poi["location"]["coordinates"][1]]))
            out.append(np.flatnonzero((px - cx[0]) ** 2 + (py - cy[0]) ** 2 <= POINT_RADIUS_KM ** 2))
            continue
        rings = [np.column_stack(proj(np.array(r)[:, 0], np.array(r)[:, 1])) for r in shape["rings"]]
        allpts = np.concatenate(rings)
        pad = POINT_RADIUS_KM
        idx = np.flatnonzero((px >= allpts[:, 0].min() - pad) & (px <= allpts[:, 0].max() + pad)
                             & (py >= allpts[:, 1].min() - pad) & (py <= allpts[:, 1].max() + pad))
        if not idx.size:
            out.append(idx)
            continue
        a, b = _segments(rings)
        dist = _distance_to_segments(px[idx], py[idx], a, b)
        if shape["kind"] == "polygon":
            match = _inside(px[idx], py[idx], a, b) | (dist <= BUFFER_KM)
        else:
            match = dist <= POINT_RADIUS_KM
        out.append(idx[match])
    return out
