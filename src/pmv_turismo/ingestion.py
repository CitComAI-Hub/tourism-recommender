from __future__ import annotations

import json
import math
import unicodedata
import calendar
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

from urllib.parse import quote

import requests

# Simple in-process caches to avoid repeated external calls during one run
_commons_cache: Dict[str, str | None] = {}
_wikidata_entity_cache: Dict[str, Dict[str, Any]] = {}
_pageviews_cache: Dict[Tuple[str, str, str, str], int] = {}
_gplaces_hours_cache: Dict[str, str | None] = {}


def _fetch_wikidata_entity(entity_id: str, timeout_seconds: int = 10) -> Dict[str, Any] | None:
    if not entity_id or not entity_id.startswith("Q"):
        return None
    if entity_id in _wikidata_entity_cache:
        return _wikidata_entity_cache[entity_id]
    try:
        resp = requests.get(
            f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json",
            timeout=timeout_seconds,
            headers={"User-Agent": "rp-turismo-ingestion/0.1"},
        )
        resp.raise_for_status()
        data = resp.json()
        ent = data.get("entities", {}).get(entity_id)
        _wikidata_entity_cache[entity_id] = ent or {}
        return ent or {}
    except Exception:
        _wikidata_entity_cache[entity_id] = {}
        return None


def _extract_wikidata_filename(ent: Dict[str, Any] | None) -> str | None:
    if not ent:
        return None
    claims = ent.get("claims", {}) or {}

    def first_value(prop: str) -> str | None:
        vals = claims.get(prop) or []
        if not vals:
            return None
        try:
            value = vals[0]["mainsnak"]["datavalue"]["value"]
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return value.get("title") or value.get("text")
            return None
        except Exception:
            return None

    return first_value("P18") or first_value("P154")


def _fetch_wikidata_image(entity_id: str, width: int = 1200, timeout_seconds: int = 10) -> str | None:
    ent = _fetch_wikidata_entity(entity_id, timeout_seconds=timeout_seconds)
    filename = _extract_wikidata_filename(ent)
    if not filename:
        return None

    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width={width}"


def _fetch_wikidata_description(entity_id: str, lang: str = "es", timeout_seconds: int = 10) -> str | None:
    ent = _fetch_wikidata_entity(entity_id, timeout_seconds=timeout_seconds)
    if not ent:
        return None
    desc = (ent.get("descriptions", {}) or {}).get(lang, {})
    text = desc.get("value") if isinstance(desc, dict) else None
    if text:
        return text
    return None


def _build_overpass_query(
    area_name: str,
    tag_filters: List[Dict[str, Any]],
    area_admin_level: str | None,
    area_boundary: str | None,
    bbox: List[float] | None,
    area_relation_id: int | None,
) -> str:
    parts: List[str] = []
    if not tag_filters:
        tag_filters = [{"key": "tourism", "values": ["*"]}]

    def add_filters(scope: str) -> None:
        for f in tag_filters:
            key = f.get("key")
            values = f.get("values") or ["*"]
            for value in values:
                if value == "*":
                    parts.append(f"{scope}[\"{key}\"]")
                else:
                    parts.append(f"{scope}[\"{key}\"=\"{value}\"]")
    if bbox:
        bbox_str = ",".join(str(v) for v in bbox)
        add_filters("node")
        add_filters("way")
        add_filters("relation")
        inner = "".join([f"{p}({bbox_str});" for p in parts])
        return (
            f"[out:json][timeout:60];"
            f"({inner});"
            "out center tags;"
        )

    add_filters("node")
    add_filters("way")
    add_filters("relation")
    inner = "".join([f"{p}(area.searchArea);" for p in parts])

    if area_relation_id:
        area_id = 3600000000 + int(area_relation_id)
        return (
            f"[out:json][timeout:60];"
            f"area({area_id})->.searchArea;"
            f"({inner});"
            "out center tags;"
        )

    area_filters = [f"[\"name\"=\"{area_name}\"]"]
    if area_boundary:
        area_filters.append(f"[\"boundary\"=\"{area_boundary}\"]")
    if area_admin_level:
        area_filters.append(f"[\"admin_level\"=\"{area_admin_level}\"]")
    area_filter_str = "".join(area_filters)
    return (
        f"[out:json][timeout:60];"
        f"area{area_filter_str}->.searchArea;"
        f"({inner});"
        "out center tags;"
    )


# Tag keys that define a POI category, by priority. Generic values such as
# `tourism=yes` carry no category, so the first specific value wins and the
# key itself (e.g. "historic") is only used as a fallback.
_CATEGORY_KEYS = ("tourism", "amenity", "leisure", "historic", "memorial", "man_made", "natural", "building")
_CATEGORY_FALLBACK_KEYS = ("historic", "memorial", "man_made", "natural", "leisure", "tourism")

# Estimated visit length per category (minutes). Replaced by real data when available.
_VISIT_MINUTES_BY_CATEGORY = {
    "museum": 90, "gallery": 60, "arts_centre": 60, "zoo": 180, "aquarium": 150, "theme_park": 240,
    "park": 45, "garden": 30, "nature_reserve": 90, "picnic_site": 45, "beach": 120,
    "attraction": 45, "marketplace": 45, "place_of_worship": 30, "cathedral": 45,
    "viewpoint": 15, "artwork": 10, "memorial": 10, "information": 10, "historic": 20,
}
DEFAULT_VISIT_MINUTES = 30


def _poi_category(tags: Dict[str, Any]) -> str | None:
    for key in _CATEGORY_KEYS:
        value = tags.get(key)
        if value and value not in ("yes", "no"):
            return str(value)
    for key in _CATEGORY_FALLBACK_KEYS:
        if tags.get(key) == "yes":
            return key
    return None


def estimate_visit_minutes(category: Any) -> int:
    return _VISIT_MINUTES_BY_CATEGORY.get(str(category or "").lower(), DEFAULT_VISIT_MINUTES)


def _extract_pois(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pois: List[Dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue

        poi = {
            "id": f"{el.get('type')}:{el.get('id')}",
            "name": name,
            "lat": lat,
            "lon": lon,
            "category": _poi_category(tags),
            "description": tags.get("description") or tags.get("short_description"),
            "address": tags.get("addr:full") or tags.get("addr:street"),
            "website": tags.get("website"),
            "phone": tags.get("phone"),
            "opening_hours": tags.get("opening_hours"),
            "image": tags.get("image"),
            "wikidata": tags.get("wikidata"),
            "wikipedia": tags.get("wikipedia"),
            "source": "osm",
        }
        pois.append(poi)
    return pois


def _normalize_name(text: str) -> str:
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    for ch in [",", ".", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "'", '"']:
        text = text.replace(ch, " ")
    text = " ".join(text.split())
    return text


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_name(a), _normalize_name(b)).ratio()


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _google_time_str(raw: str | None) -> str | None:
    if not raw or len(raw) != 4:
        return None
    return f"{raw[:2]}:{raw[2:]}"


def _google_periods_to_opening_hours(periods: List[Dict[str, Any]] | None) -> str | None:
    if not periods:
        return None

    # Google uses 0=Sunday..6=Saturday; we expose in Mo-Su order
    day_map = {0: "Su", 1: "Mo", 2: "Tu", 3: "We", 4: "Th", 5: "Fr", 6: "Sa"}
    ordered_days = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    slots: Dict[str, Tuple[str, str] | None] = {d: None for d in ordered_days}

    for period in periods:
        open_data = period.get("open", {}) if isinstance(period, dict) else {}
        close_data = period.get("close", {}) if isinstance(period, dict) else {}
        day = open_data.get("day")
        start = _google_time_str(open_data.get("time"))
        end = _google_time_str(close_data.get("time"))
        if day is None or not start or not end:
            continue
        day_code = day_map.get(day)
        if not day_code:
            continue
        # Skip overnight ranges for now
        if start >= end:
            continue
        slots[day_code] = (start, end)

    def flush(group_days: List[str], slot: Tuple[str, str], acc: List[str]) -> None:
        if not group_days:
            return
        prefix = group_days[0] if len(group_days) == 1 else f"{group_days[0]}-{group_days[-1]}"
        acc.append(f"{prefix} {slot[0]}-{slot[1]}")

    segments: List[str] = []
    current_days: List[str] = []
    current_slot: Tuple[str, str] | None = None

    for day in ordered_days:
        slot = slots.get(day)
        if slot and slot == current_slot:
            current_days.append(day)
            continue
        if current_slot:
            flush(current_days, current_slot, segments)
        current_slot = slot
        current_days = [day] if slot else []

    if current_slot:
        flush(current_days, current_slot, segments)

    return "; ".join(segments) if segments else None


def _fetch_google_place_opening_hours(
    name: str,
    lat: float,
    lon: float,
    api_key: str,
    timeout_seconds: int = 10,
    radius_m: int = 250,
    min_name_similarity: float = 0.75,
) -> str | None:
    if not api_key or not name:
        return None
    cache_key = f"{name}::{lat},{lon}"
    if cache_key in _gplaces_hours_cache:
        return _gplaces_hours_cache[cache_key]

    params = {
        "input": name,
        "inputtype": "textquery",
        "fields": "place_id,name,geometry,opening_hours",
        "locationbias": f"point:{lat},{lon}",
        "key": api_key,
    }
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
            params=params,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        _gplaces_hours_cache[cache_key] = None
        return None

    if data.get("status") not in {"OK", "ZERO_RESULTS"}:
        _gplaces_hours_cache[cache_key] = None
        return None

    candidates = data.get("candidates", []) if isinstance(data, dict) else []
    best: Dict[str, Any] | None = None
    best_sim = 0.0
    for cand in candidates:
        cand_name = cand.get("name") or ""
        cand_loc = (cand.get("geometry", {}).get("location") or {}) if isinstance(cand, dict) else {}
        cand_lat = cand_loc.get("lat")
        cand_lon = cand_loc.get("lng")
        dist_ok = cand_lat is not None and cand_lon is not None
        dist = _haversine_m(lat, lon, float(cand_lat), float(cand_lon)) if dist_ok else None
        sim = _name_similarity(name, cand_name) if cand_name else 0.0
        if dist is not None and dist > radius_m:
            continue
        if sim < min_name_similarity:
            continue
        if sim > best_sim:
            best = cand
            best_sim = sim

    if not best:
        _gplaces_hours_cache[cache_key] = None
        return None

    periods = best.get("opening_hours", {}).get("periods") if isinstance(best.get("opening_hours"), dict) else None
    hours = _google_periods_to_opening_hours(periods)

    if not hours:
        place_id = best.get("place_id")
        if place_id:
            try:
                details_resp = requests.get(
                    "https://maps.googleapis.com/maps/api/place/details/json",
                    params={
                        "place_id": place_id,
                        "fields": "opening_hours,name,geometry",
                        "key": api_key,
                    },
                    timeout=timeout_seconds,
                )
                details_resp.raise_for_status()
                details_data = details_resp.json()
                if details_data.get("status") == "OK":
                    details_result = details_data.get("result", {})
                    periods = details_result.get("opening_hours", {}).get("periods")
                    hours = _google_periods_to_opening_hours(periods)
            except Exception:
                hours = None

    _gplaces_hours_cache[cache_key] = hours
    return hours


def _merge_records(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(primary)
    for key, value in secondary.items():
        if not merged.get(key) and value:
            merged[key] = value
    if primary.get("source") and secondary.get("source"):
        sources = set(str(primary.get("source")).split("+")) | set(
            str(secondary.get("source")).split("+")
        )
        merged["source"] = "+".join(sorted(sources))
    return merged


def _synthetic_opening_hours(category: Any) -> str:
    """Return a conservative default schedule when a POI has no source hours."""

    category_name = str(category or "").lower()
    if category_name in {"park", "garden", "nature_reserve", "artwork"}:
        return "Mo-Su 00:00-24:00"
    if category_name in {"place_of_worship"}:
        return "Mo-Su 08:00-20:00"
    if category_name in {"museum", "aquarium", "zoo", "arts_centre", "attraction", "information"}:
        return "Tu-Su 10:00-19:00; Mo off"
    if category_name in {"marketplace"}:
        return "Mo-Su 09:00-21:00"
    return "Mo-Su 09:00-20:00"


def _deduplicate_pois(
    pois: List[Dict[str, Any]],
    max_distance_m: float,
    min_name_similarity: float,
    prefer_sources: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def source_rank(poi: Dict[str, Any]) -> int:
        source = str(poi.get("source") or "")
        for idx, name in enumerate(prefer_sources):
            if name in source:
                return idx
        return len(prefer_sources)

    remaining = sorted(pois, key=source_rank)
    result: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    while remaining:
        current = remaining.pop(0)
        if current.get("name") is None or current.get("lat") is None or current.get("lon") is None:
            continue
        merged = current
        to_remove: List[int] = []
        for idx, candidate in enumerate(remaining):
            if candidate.get("name") is None or candidate.get("lat") is None or candidate.get("lon") is None:
                continue
            dist = _haversine_m(
                float(merged["lat"]),
                float(merged["lon"]),
                float(candidate["lat"]),
                float(candidate["lon"]),
            )
            if dist > max_distance_m:
                continue
            sim = _name_similarity(str(merged["name"]), str(candidate["name"]))
            if sim >= min_name_similarity:
                matches.append(
                    {
                        "name_a": merged.get("name"),
                        "source_a": merged.get("source"),
                        "id_a": merged.get("id"),
                        "lat_a": merged.get("lat"),
                        "lon_a": merged.get("lon"),
                        "name_b": candidate.get("name"),
                        "source_b": candidate.get("source"),
                        "id_b": candidate.get("id"),
                        "lat_b": candidate.get("lat"),
                        "lon_b": candidate.get("lon"),
                        "distance_m": round(dist, 2),
                        "name_similarity": round(sim, 3),
                    }
                )
                merged = _merge_records(merged, candidate)
                to_remove.append(idx)
        for offset, idx in enumerate(to_remove):
            remaining.pop(idx - offset)
        result.append(merged)
    return result, matches


def _fetch_commons_image(
    name: str,
    city_hint: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    timeout_seconds: int = 10,
    radius_m: int = 800,
) -> str | None:
    if not name:
        return None
    key = f"{name}::{city_hint or ''}::{lat},{lon}"
    if key in _commons_cache:
        return _commons_cache[key]

    headers = {"User-Agent": "rp-turismo-ingestion/0.1"}

    # First try geosearch (File namespace) for nearby geotagged media
    if lat is not None and lon is not None:
        geo_params = {
            "action": "query",
            "generator": "geosearch",
            "ggscoord": f"{lat}|{lon}",
            "ggsradius": radius_m,
            "ggsnamespace": 6,
            "ggslimit": 1,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 1200,
            "format": "json",
        }
        try:
            resp = requests.get("https://commons.wikimedia.org/w/api.php", params=geo_params, headers=headers, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages") or {}
            first = next(iter(pages.values()), None)
            info = (first or {}).get("imageinfo", [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url:
                _commons_cache[key] = url
                return url
        except Exception:
            pass

    # Fallback: search by name + city hint in File namespace
    query = name if not city_hint else f"{name} {city_hint}"
    search_params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 1,
        "gsrnamespace": 6,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1200,
        "format": "json",
    }

    try:
        resp = requests.get("https://commons.wikimedia.org/w/api.php", params=search_params, headers=headers, timeout=timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages") or {}
        first = next(iter(pages.values()), None)
        info = (first or {}).get("imageinfo", [{}])[0]
        url = info.get("thumburl") or info.get("url")
        _commons_cache[key] = url
        return url
    except Exception:
        _commons_cache[key] = None
        return None


def _fetch_wikipedia_extract(title: str, lang: str = "es", chars: int = 900, timeout_seconds: int = 10) -> str | None:
    if not title:
        return None
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "exchars": chars,
        "format": "json",
        "redirects": 1,
        "titles": title,
    }
    try:
        resp = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=params,
            headers={"User-Agent": "rp-turismo-ingestion/0.1"},
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        first = next(iter(pages.values()), {})
        extract = first.get("extract")
        if extract:
            return str(extract).strip()
        return None
    except Exception:
        return None


def _fetch_overpass_data(
    overpass_urls: List[str], query: str, timeout_seconds: int
) -> Dict[str, Any]:
    last_error: Exception | None = None
    errors: List[str] = []
    data = None
    headers = {
        "User-Agent": "rp-turismo-ingestion/0.1",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    for url in overpass_urls:
        try:
            response = requests.post(url, data={"data": query}, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException as exc:
            last_error = exc
            errors.append(f"{url}: {exc}")
            continue

    if data is None:
        error_details = "\n".join(errors) if errors else "No errors captured"
        raise RuntimeError(f"All Overpass endpoints failed\n{error_details}") from last_error
    return data


def _pageview_window(months: int = 12) -> Tuple[str, str]:
    today = date.today()
    last_full_month = (today.replace(day=1) - timedelta(days=1))
    end_year, end_month = last_full_month.year, last_full_month.month
    end_day = calendar.monthrange(end_year, end_month)[1]

    start_month_index = end_month - (months - 1)
    start_year = end_year
    while start_month_index <= 0:
        start_month_index += 12
        start_year -= 1
    start_date = date(start_year, start_month_index, 1)

    start_str = f"{start_date.year}{start_date.month:02d}01"
    end_str = f"{end_year}{end_month:02d}{end_day:02d}"
    return start_str, end_str


def _fetch_pageviews(title: str, project: str, start: str, end: str, timeout_seconds: int = 10) -> int:
    if not title or not project:
        return 0
    key = (title, project, start, end)
    if key in _pageviews_cache:
        return _pageviews_cache[key]

    article = quote(title.replace(" ", "_"))
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/{article}/monthly/{start}/{end}"
    try:
        resp = requests.get(url, headers={"User-Agent": "rp-turismo-ingestion/0.1"}, timeout=timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", []) if isinstance(data, dict) else []
        total = 0
        for item in items:
            try:
                total += int(item.get("views", 0))
            except Exception:
                continue
        _pageviews_cache[key] = total
        return total
    except Exception:
        _pageviews_cache[key] = 0
        return 0


def _wikipedia_sitelinks(ent: Dict[str, Any] | None) -> List[Tuple[str, str]]:
    if not ent:
        return []
    sitelinks = ent.get("sitelinks", {}) or {}
    titles: List[Tuple[str, str]] = []
    for key, val in sitelinks.items():
        if not key.endswith("wiki"):
            continue
        if key.endswith("wikivoyage"):
            continue
        title = val.get("title") if isinstance(val, dict) else None
        if not title:
            continue
        lang = key[:-4]
        project = f"{lang}.wikipedia.org"
        titles.append((title, project))
    return titles


def run_ingestion(config: dict) -> None:
    """Ingestion pipeline for OSM Valencia monuments."""
    ingestion_cfg = config.get("ingestion", {})
    data_cfg = config.get("data", {})

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = int(ingestion_cfg.get("request_timeout_seconds", 120))
    fetch_images = bool(ingestion_cfg.get("fetch_images", True))
    require_description = bool(ingestion_cfg.get("require_description", False))
    city_hint = ingestion_cfg.get("image_city_hint", "Valencia")
    pois: List[Dict[str, Any]] = []
    pageview_start, pageview_end = _pageview_window(months=12)

    if ingestion_cfg.get("osm", {}).get("enabled", True):
        overpass_urls = ingestion_cfg.get("overpass_urls") or [
            ingestion_cfg.get("overpass_url", "https://overpass-api.de/api/interpreter")
        ]
        area_name = ingestion_cfg.get("area_name", "Valencia")
        tag_filters = ingestion_cfg.get("tag_filters", [])
        area_admin_level = ingestion_cfg.get("area_admin_level")
        area_boundary = ingestion_cfg.get("area_boundary")
        bbox = ingestion_cfg.get("bbox")
        area_relation_id = ingestion_cfg.get("area_relation_id")
        query = _build_overpass_query(area_name, tag_filters, area_admin_level, area_boundary, bbox, area_relation_id)
        data = _fetch_overpass_data(overpass_urls, query, timeout_seconds)
        raw_path = raw_dir / "poi_raw_osm.json"
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        pois.extend(_extract_pois(data.get("elements", [])))

    dedup_cfg = ingestion_cfg.get("dedup", {})
    if dedup_cfg.get("enabled", False):
        max_distance_m = float(dedup_cfg.get("max_distance_m", 80))
        min_name_similarity = float(dedup_cfg.get("min_name_similarity", 0.82))
        prefer_sources = dedup_cfg.get("prefer_sources", ["osm"])
        pois, matches = _deduplicate_pois(pois, max_distance_m, min_name_similarity, prefer_sources)
        report_path = dedup_cfg.get("report_path", str(processed_dir / "dedup_matches.csv"))
        report_headers = [
            "name_a",
            "source_a",
            "id_a",
            "lat_a",
            "lon_a",
            "name_b",
            "source_b",
            "id_b",
            "lat_b",
            "lon_b",
            "distance_m",
            "name_similarity",
        ]
        report_lines = [",".join(report_headers)]
        for m in matches:
            row = ["" if m.get(h) is None else str(m.get(h)).replace(",", " ") for h in report_headers]
            report_lines.append(",".join(row))
        Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")

    gplaces_cfg = ingestion_cfg.get("google_places", {})
    if gplaces_cfg.get("enabled") and gplaces_cfg.get("api_key"):
        radius_m = int(gplaces_cfg.get("radius_m", 250))
        min_name_similarity = float(gplaces_cfg.get("min_name_similarity", 0.75))
        for poi in pois:
            if poi.get("opening_hours"):
                continue
            name = poi.get("name")
            lat = poi.get("lat")
            lon = poi.get("lon")
            if name is None or lat is None or lon is None:
                continue
            hours = _fetch_google_place_opening_hours(
                str(name),
                float(lat),
                float(lon),
                api_key=str(gplaces_cfg.get("api_key")),
                timeout_seconds=timeout_seconds,
                radius_m=radius_m,
                min_name_similarity=min_name_similarity,
            )
            if hours:
                poi["opening_hours"] = hours
                if poi.get("source"):
                    poi["source"] = f"{poi['source']}+gplaces"
                else:
                    poi["source"] = "gplaces"

    processed_path = processed_dir / "poi.jsonld"
    entities: List[Dict[str, Any]] = []
    popularity_entries: List[Tuple[int, str, float]] = []
    for poi in pois:
        name = poi.get("name")
        lat = poi.get("lat")
        lon = poi.get("lon")
        if name is None or lat is None or lon is None:
            continue
        visit_minutes = estimate_visit_minutes(poi.get("category"))
        image_url = poi.get("image")
        description = poi.get("description")
        wikidata_id = poi.get("wikidata")
        entity_title = None
        wikipedia_lang = "es"

        # Prefer an explicit Wikipedia tag from the source (format "lang:Title")
        wikipedia_tag = poi.get("wikipedia")
        if isinstance(wikipedia_tag, str) and wikipedia_tag:
            if ":" in wikipedia_tag:
                lang_candidate, title_candidate = wikipedia_tag.split(":", 1)
                if title_candidate:
                    wikipedia_lang = lang_candidate or "es"
                    entity_title = title_candidate
            else:
                entity_title = wikipedia_tag

        ent = _fetch_wikidata_entity(wikidata_id, timeout_seconds=timeout_seconds) if wikidata_id else None
        pageview_candidates: List[Tuple[str, str]] = []
        if ent:
            if not image_url and fetch_images:
                filename = _extract_wikidata_filename(ent)
                if filename:
                    image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width=1200"
            sitelinks = ent.get("sitelinks", {}) or {}
            es_link = sitelinks.get("eswiki") or sitelinks.get("eswikivoyage")
            en_link = sitelinks.get("enwiki")
            if not entity_title:
                entity_title = (es_link or en_link or {}).get("title")
                if es_link:
                    wikipedia_lang = "es"
                elif en_link:
                    wikipedia_lang = "en"
            pageview_candidates.extend(_wikipedia_sitelinks(ent))

        if entity_title:
            pageview_candidates.append((entity_title, f"{wikipedia_lang}.wikipedia.org"))

        total_views = 0
        seen_candidates = set()
        for title, project in pageview_candidates:
            key = (title, project)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            total_views += _fetch_pageviews(title, project, pageview_start, pageview_end, timeout_seconds=timeout_seconds)

        popularity_log = math.log10(total_views + 1)

        if not description and entity_title:
            description = _fetch_wikipedia_extract(
                entity_title,
                lang=wikipedia_lang or "es",
                chars=900,
                timeout_seconds=timeout_seconds,
            )

        if require_description and not description:
            continue

        if not image_url and fetch_images:
            image_url = _fetch_commons_image(
                str(name),
                city_hint=city_hint,
                lat=float(lat) if lat is not None else None,
                lon=float(lon) if lon is not None else None,
                timeout_seconds=timeout_seconds,
            )
        entity = {
            "id": f"urn:ngsi-ld:PointOfInterest:{poi.get('id')}",
            "type": "PointOfInterest",
            "name": str(name),
            "location": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            },
            "category": [str(poi.get("category"))] if poi.get("category") else [],
            "duration": visit_minutes,
            "durationEstimated": True,
            "image": image_url,
            "description": description,
            "address": {
                "streetAddress": poi.get("address")
            } if poi.get("address") else None,
            "source": "https://www.openstreetmap.org" if poi.get("source") == "osm" else "https://valencia.opendatasoft.com",
            "refSeeAlso": [v for v in [poi.get("website"), poi.get("wikidata"), poi.get("wikipedia")] if v],
            "contactPoint": {
                "telephone": poi.get("phone")
            } if poi.get("phone") else None,
            "openingHours": {
                "value": poi.get("opening_hours") or _synthetic_opening_hours(poi.get("category")),
                "type": "Text",
                "synthetic": not bool(poi.get("opening_hours")),
            },
            "@context": [
                "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
                "https://raw.githubusercontent.com/smart-data-models/dataModel.PointOfInterest/master/context.jsonld",
            ],
        }
        entity = {k: v for k, v in entity.items() if v not in (None, [], "")}
        entities.append(entity)

        cat_key = None
        if isinstance(entity.get("category"), list) and entity.get("category"):
            cat_key = str(entity.get("category")[0])
        elif entity.get("category"):
            cat_key = str(entity.get("category"))
        popularity_entries.append((len(entities) - 1, cat_key or "unknown", popularity_log))

    def percentile_ranks(values: List[float]) -> Dict[float, float]:
        if not values:
            return {}
        if len(values) == 1:
            return {values[0]: 0.5}
        sorted_vals = sorted(values)
        ranks: Dict[float, float] = {}
        n = len(sorted_vals)
        for v in set(sorted_vals):
            positions = [i for i, val in enumerate(sorted_vals) if val == v]
            mean_pos = sum(positions) / len(positions)
            ranks[v] = mean_pos / (n - 1)
        return ranks

    per_category: Dict[str, List[Tuple[int, float]]] = {}
    for idx, cat, log_score in popularity_entries:
        per_category.setdefault(cat, []).append((idx, log_score))

    for cat, items in per_category.items():
        ranks = percentile_ranks([v for _, v in items])
        for idx, log_score in items:
            p = ranks.get(log_score, 0.0)
            score = max(0.0, min(1.0, 0.05 + 0.9 * p))
            relevance = 1.0 - score
            entities[idx]["relevance"] = relevance
    processed_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
