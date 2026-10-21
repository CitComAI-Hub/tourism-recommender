from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Madrid")
POI_URN_PREFIX = "urn:ngsi-ld:PointOfInterest:"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def poi_coordinates(item: Dict[str, Any]) -> Optional[List[float]]:
    """Return [lon, lat] for a POI or a wrapped {"poi": {...}} record."""
    coords = (item.get("location") or {}).get("coordinates") or (
        (item.get("poi") or {}).get("location") or {}
    ).get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    return coords


def processed_dir(config: Dict[str, Any]) -> Path:
    return Path(config.get("data", {}).get("processed_dir", "data/processed"))


def load_poi_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed POI file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("poi.jsonld must be a list of entities")
    return data


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON through a temp file + rename so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


_OSM_DAY_INDEX = {"Mo": 0, "Tu": 1, "We": 2, "Th": 3, "Fr": 4, "Sa": 5, "Su": 6}


def opening_hours_cover_duration(
    opening_hours: Any,
    visit_start: datetime,
    duration_minutes: float,
) -> bool:
    """Check common OSM opening-hours expressions for the complete visit.

    Unknown or unparseable expressions are treated as open.
    """

    if isinstance(opening_hours, dict):
        opening_hours = opening_hours.get("value")
    if not isinstance(opening_hours, str) or not opening_hours.strip():
        return True
    if opening_hours.strip() == "24/7":
        return True

    required_start = visit_start.hour * 60 + visit_start.minute
    required_end = required_start + max(0.0, duration_minutes)
    day = visit_start.weekday()
    matched_day = False

    for clause in opening_hours.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        match = re.match(r"^(Mo|Tu|We|Th|Fr|Sa|Su)(?:-(Mo|Tu|We|Th|Fr|Sa|Su))?\s+(.+)$", clause)
        if not match:
            continue
        start_day = _OSM_DAY_INDEX[match.group(1)]
        end_day = _OSM_DAY_INDEX[match.group(2) or match.group(1)]
        days = {(start_day + offset) % 7 for offset in range((end_day - start_day) % 7 + 1)}
        if day not in days:
            continue
        matched_day = True
        if "off" in match.group(3).lower() or "closed" in match.group(3).lower():
            continue
        for interval in re.findall(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", match.group(3)):
            start = int(interval[0]) * 60 + int(interval[1])
            end = int(interval[2]) * 60 + int(interval[3])
            if end == 0:
                end = 24 * 60
            if start <= required_start and required_end <= end:
                return True
    return not matched_day
