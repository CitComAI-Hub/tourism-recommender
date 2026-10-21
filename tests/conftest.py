from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

# Three semantic groups (museums / parks / churches) plus a lone viewpoint.
POIS = [
    ("m1", "museum", [1.0, 0.1, 0.0, 0.0], 0.1, (39.470, -0.376)),
    ("m2", "museum", [0.9, 0.2, 0.0, 0.0], 0.5, (39.471, -0.377)),
    ("m3", "museum", [0.95, 0.0, 0.1, 0.0], 0.9, (39.475, -0.380)),
    ("p1", "park", [0.0, 1.0, 0.1, 0.0], 0.2, (39.480, -0.370)),
    ("p2", "park", [0.1, 0.9, 0.0, 0.0], 0.6, (39.481, -0.371)),
    ("p3", "park", [0.0, 0.95, 0.0, 0.1], 0.8, (39.490, -0.360)),
    ("c1", "place_of_worship", [0.0, 0.1, 1.0, 0.0], 0.3, (39.474, -0.375)),
    ("c2", "place_of_worship", [0.1, 0.0, 0.9, 0.0], 0.5, (39.476, -0.378)),
    ("c3", "place_of_worship", [0.0, 0.0, 0.95, 0.1], 0.7, (39.478, -0.379)),
    ("v1", "viewpoint", [0.0, 0.0, 0.0, 1.0], 0.4, (39.460, -0.390)),
]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    pois = [
        {
            "id": f"urn:ngsi-ld:PointOfInterest:node:{pid}",
            "name": pid,
            "category": [cat],
            "relevance": rel,
            "duration": 30,
            "location": {"type": "Point", "coordinates": [lon, lat]},
        }
        for pid, cat, _, rel, (lat, lon) in POIS
    ]
    (processed / "poi.jsonld").write_text(json.dumps(pois), encoding="utf-8")
    with (processed / "embeddings.jsonl").open("w", encoding="utf-8") as f:
        for pid, _, vec, _, _ in POIS:
            unit = (np.asarray(vec) / np.linalg.norm(vec)).tolist()
            f.write(json.dumps({"id": f"urn:ngsi-ld:PointOfInterest:node:{pid}", "embedding": unit}) + "\n")

    config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config["data"] = {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(processed)}
    config["embeddings"]["output_path"] = str(processed / "embeddings.jsonl")
    config["parameters"]["planning"]["routing"] = {"provider": "osrm", "profile": "foot"}
    config["recommendation"]["filter_opening_hours"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return tmp_path


@pytest.fixture
def config(workspace: Path) -> dict:
    return yaml.safe_load((workspace / "config.yaml").read_text(encoding="utf-8"))


def poi_id(short: str) -> str:
    return f"urn:ngsi-ld:PointOfInterest:node:{short}"
