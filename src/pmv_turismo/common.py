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
_DAY_RULE = re.compile(r"^((?:Mo|Tu|We|Th|Fr|Sa|Su|PH|SH)(?:\s*[-,]\s*(?:Mo|Tu|We|Th|Fr|Sa|Su|PH|SH))*)\s+(.+)$")


def _rule_days(spec: str) -> set[int]:
    """Weekdays of an OSM day selector such as "Mo-Fr", "Tu,Th" or "Sa,Su,PH"."""
    days: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        start, _, end = part.partition("-")
        if start not in _OSM_DAY_INDEX or (end and end not in _OSM_DAY_INDEX):
            continue  # public/school holidays (PH, SH) are not weekdays
        first, last = _OSM_DAY_INDEX[start], _OSM_DAY_INDEX[end or start]
        days.update((first + offset) % 7 for offset in range((last - first) % 7 + 1))
    return days


def opening_hours_cover_duration(
    opening_hours: Any,
    visit_start: datetime,
    duration_minutes: float,
) -> bool:
    """Check common OSM opening-hours expressions for the complete visit.

    Weekday rules ("Tu-Su 10:00-19:00", "Sa,Su 10:00-14:00, 16:00-20:00") are
    understood; as in OSM, a weekday not mentioned by any rule is closed.
    Expressions with no understandable weekday rule (month ranges, sunrise...)
    are treated as open rather than hiding the POI.
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
    understood = False

    for clause in opening_hours.split(";"):
        match = _DAY_RULE.match(clause.strip())
        if not match:
            continue
        days = _rule_days(match.group(1))
        if not days:
            continue
        understood = True
        if day not in days:
            continue
        times = match.group(2).lower()
        if "off" in times or "closed" in times:
            continue
        for interval in re.findall(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", times):
            start = int(interval[0]) * 60 + int(interval[1])
            end = int(interval[2]) * 60 + int(interval[3])
            if end == 0:
                end = 24 * 60
            if start <= required_start and required_end <= end:
                return True
    return not understood
