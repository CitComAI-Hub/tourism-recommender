from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlsplit, urlunsplit

import requests

from .descriptions import build_description
from .heritage import match_heritage, names_match, parse_inventory

logger = logging.getLogger(__name__)

# Wikimedia asks for a descriptive User-Agent; set PMV_HTTP_CONTACT (URL or
# e-mail) to include contact information as their policy recommends.
USER_AGENT = "pmv-turismo-ingestion/0.2" + (
    f" ({os.environ['PMV_HTTP_CONTACT']})" if os.environ.get("PMV_HTTP_CONTACT") else ""
)

class FetchError(RuntimeError):
    """An external source could not be reached after retries.

    Distinct from "the source has no data": a POI that fails to enrich must
    not be dropped silently as if it had no description.
    """


_HOST_MIN_INTERVAL_S = 0.25  # at most ~4 requests/second per host...
_HOST_MIN_INTERVAL_OVERRIDES = {"wikimedia.org": 0.05}  # ...the pageviews REST API allows more
_host_lock = threading.Lock()
_host_next_slot: dict[str, float] = {}
_session = requests.Session()


def _wait_for_host(host: str) -> None:
    with _host_lock:
        now = time.monotonic()
        slot = max(now, _host_next_slot.get(host, 0.0))
        _host_next_slot[host] = slot + _HOST_MIN_INTERVAL_OVERRIDES.get(host, _HOST_MIN_INTERVAL_S)
    if slot > now:
        time.sleep(slot - now)


def _http_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout_seconds: int = 20,
    retries: int = 5,
    not_found_ok: bool = False,
    as_text: bool = False,
) -> Any:
    """GET with per-host throttling and retries on 429/5xx (honours Retry-After).

    Returns the parsed JSON (or the body as text with `as_text`), None for 404
    when `not_found_ok`; raises FetchError otherwise.
    """
    host = urlparse(url).netloc
    delay = 2.0
    last_error = ""
    for attempt in range(retries):
        _wait_for_host(host)
        try:
            resp = _session.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if resp.status_code == 404 and not_found_ok:
                return None
            if resp.status_code < 400:
                if as_text:
                    resp.encoding = resp.encoding if "charset" in resp.headers.get("Content-Type", "") else "utf-8"
                    return resp.text
                return resp.json()
            last_error = f"HTTP {resp.status_code}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        if attempt < retries - 1:
            logger.warning("%s (%s), retrying in %.0fs", host, last_error, min(delay, 60.0))
            time.sleep(min(delay, 60.0))
            delay *= 2
    raise FetchError(f"{host}: {last_error}")


# In-process caches (only successful responses are cached)
_wikidata_entity_cache: dict[str, dict[str, Any]] = {}
_gplaces_hours_cache: dict[str, str | None] = {}

# Wikipedia editions whose pageviews are summed as the popularity signal.
DEFAULT_PAGEVIEW_LANGUAGES = ["es", "ca", "en", "fr", "de", "it", "pt", "nl", "ru", "pl", "ja", "zh"]


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _fetch_wikidata_entities(qids: list[str], timeout_seconds: int = 20) -> dict[str, dict[str, Any]]:
    """Sitelinks and claims for many items, 50 per request."""
    wanted = sorted({q for q in qids if q and q.startswith("Q")} - set(_wikidata_entity_cache))
    for chunk in _chunks(wanted, 50):
        data = _http_get_json(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbgetentities", "ids": "|".join(chunk), "props": "sitelinks|claims|labels",
                    "languages": "es|ca|en|fr|nl|de|it|pl|da|fi|lb", "format": "json"},
            timeout_seconds=timeout_seconds,
        )
        entities = (data or {}).get("entities") or {}
        for qid in chunk:
            ent = entities.get(qid) or {}
            _wikidata_entity_cache[qid] = {} if "missing" in ent else ent
    return {q: _wikidata_entity_cache.get(q) or {} for q in qids if q}


def _mw_pages_by_title(
    api_url: str, titles: list[str], params: dict[str, Any], batch_size: int, timeout_seconds: int
) -> dict[str, dict[str, Any]]:
    """Run a MediaWiki `action=query` over many titles in batches.

    Returns {requested title: page}, following title normalisation, redirects
    and `continue` pagination (props such as extracts merge across pages).
    """
    result: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(sorted(set(titles)), batch_size):
        base = {**params, "action": "query", "format": "json", "formatversion": 2,
                "redirects": 1, "titles": "|".join(chunk)}
        pages: dict[str, dict[str, Any]] = {}
        normalized: dict[str, str] = {}
        redirects: dict[str, str] = {}
        cont: dict[str, Any] = {}
        while True:
            data = _http_get_json(api_url, params={**base, **cont}, timeout_seconds=timeout_seconds) or {}
            query = data.get("query") or {}
            normalized.update({n["from"]: n["to"] for n in query.get("normalized", [])})
            redirects.update({r["from"]: r["to"] for r in query.get("redirects", [])})
            for page in query.get("pages", []):
                merged = pages.setdefault(page.get("title", ""), {})
                for key, value in page.items():
                    if isinstance(value, dict) and isinstance(merged.get(key), dict):
                        merged[key].update(value)
                    elif value is not None and key not in merged:
                        merged[key] = value
            if "continue" not in data:
                break
            cont = data["continue"]
        for title in chunk:
            final = normalized.get(title, title)
            final = redirects.get(final, final)
            if final in pages and "missing" not in pages[final]:
                result[title] = pages[final]
    return result


def _fetch_wikipedia_extracts(
    lang: str, titles: list[str], max_chars: int = 900, timeout_seconds: int = 20
) -> dict[str, str]:
    """Plain-text lead sections, 20 articles per request, cut at a full sentence."""
    pages = _mw_pages_by_title(
        f"https://{lang}.wikipedia.org/w/api.php",
        titles,
        {"prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max"},
        batch_size=20,
        timeout_seconds=timeout_seconds,
    )
    return {
        title: _shorten_to_sentence(page["extract"], max_chars)
        for title, page in pages.items()
        if page.get("extract")
    }


def _fetch_pageviews(lang: str, titles: list[str], days: int = 60, timeout_seconds: int = 20) -> dict[str, int]:
    """Total pageviews over the last `days` days, 50 articles per request."""
    pages = _mw_pages_by_title(
        f"https://{lang}.wikipedia.org/w/api.php",
        titles,
        {"prop": "pageviews", "pvipmetric": "pageviews", "pvipdays": days},
        batch_size=50,
        timeout_seconds=timeout_seconds,
    )
    return {
        title: sum(int(v or 0) for v in (page.get("pageviews") or {}).values())
        for title, page in pages.items()
    }


def _extract_wikidata_filename(ent: dict[str, Any] | None) -> str | None:
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

    # P18 = image. P154 (logo) is not used: a logo is not a picture of the place.
    return first_value("P18")


def _build_overpass_query(
    area_name: str,
    tag_filters: list[dict[str, Any]],
    area_admin_level: str | None,
    area_boundary: str | None,
    bbox: list[float] | None,
    area_relation_id: int | None,
) -> str:
    parts: list[str] = []
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


# Tag keys that define a POI subcategory, by priority. Generic values such as
# `tourism=yes` carry no category, so the first specific value wins and the
# key itself (e.g. "historic") is only used as a fallback.
_CATEGORY_KEYS = ("tourism", "amenity", "leisure", "historic", "memorial", "man_made", "natural", "building")
_CATEGORY_FALLBACK_KEYS = ("historic", "memorial", "man_made", "natural", "leisure", "tourism")

# OSM subcategory -> canonical category shared with the frontend
# (frontend/lib/poi-categories.ts). Anything not listed (castles, palaces,
# bridges, gates, towers, memorials...) is cultural heritage: "historic".
_CANONICAL_CATEGORY = {
    "sculpture_collection": "artwork",
    "museum": "museum", "gallery": "museum",
    "arts_centre": "arts_centre",
    "park": "park", "picnic_site": "park",
    "garden": "garden",
    "nature_reserve": "nature_reserve", "peak": "nature_reserve", "waterfall": "nature_reserve",
    "beach": "beach",
    "attraction": "attraction", "theme_park": "attraction", "viewpoint": "attraction",
    "artwork": "artwork",
    "place_of_worship": "place_of_worship", "cathedral": "place_of_worship", "church": "place_of_worship",
    "chapel": "place_of_worship", "monastery": "place_of_worship", "convent": "place_of_worship",
    "marketplace": "marketplace",
    "information": "information",
    "aquarium": "aquarium",
    "zoo": "zoo",
}
DEFAULT_CANONICAL_CATEGORY = "historic"

# Estimated visit length (minutes) by subcategory, then canonical category.
_VISIT_MINUTES = {
    "sculpture_collection": 60,
    "museum": 90, "gallery": 60, "arts_centre": 60, "zoo": 180, "aquarium": 150, "theme_park": 240,
    "park": 45, "garden": 30, "nature_reserve": 90, "picnic_site": 45, "beach": 120,
    "attraction": 45, "marketplace": 45, "place_of_worship": 30, "cathedral": 45, "monastery": 45,
    "viewpoint": 15, "artwork": 10, "memorial": 10, "monument": 10, "information": 10,
    "bridge": 10, "city_gate": 30, "castle": 45, "manor": 30, "tower": 30, "building": 20,
    "archaeological_site": 30, "train_station": 15, "historic": 20,
}
DEFAULT_VISIT_MINUTES = 30

# Outdoor places that can be seen at any time.
# `tourism=information` worth keeping; boards, maps and guideposts are not destinations.
_USEFUL_INFORMATION = {"office", "visitor_centre"}

# Description tags in order of preference (Spanish > English > untagged > Valencian)
# and the language each one is in (None: not stated).
DEFAULT_DESCRIPTION_LANGUAGES = ("es", "en", "ca")


def _description_keys(languages: Sequence[str]) -> list[tuple[str, str | None]]:
    """OSM description tags in order of preference, with their language: the
    preferred languages first, then the untagged one, then any other language."""
    keys: list[tuple[str, str | None]] = [(f"description:{lang}", lang) for lang in languages[:2]]
    keys.append(("description", None))
    keys += [(f"description:{lang}", lang) for lang in languages[2:]]
    keys.append(("short_description", None))
    return keys
MIN_USEFUL_DESCRIPTION_CHARS = 80


def _poi_category(tags: dict[str, Any]) -> str | None:
    for key in _CATEGORY_KEYS:
        value = tags.get(key)
        if value and value not in ("yes", "no"):
            return str(value)
    for key in _CATEGORY_FALLBACK_KEYS:
        if tags.get(key) == "yes":
            return key
    return None


def canonical_category(subcategory: Any) -> str | None:
    if not subcategory:
        return None
    return _CANONICAL_CATEGORY.get(str(subcategory).lower(), DEFAULT_CANONICAL_CATEGORY)


def estimate_visit_minutes(subcategory: Any) -> int:
    key = str(subcategory or "").lower()
    return _VISIT_MINUTES.get(key) or _VISIT_MINUTES.get(canonical_category(key) or "", DEFAULT_VISIT_MINUTES)


def _commons_file_url(filename: str, width: int = 1200) -> str:
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width={width}"


def _osm_image(tags: dict[str, Any]) -> str | None:
    """Image from OSM tags, only when it points to Wikimedia Commons.

    Other hosts are usually hotlinked thumbnails (e.g. Google) with unclear
    licensing and short-lived URLs.
    """
    for key in ("wikimedia_commons", "image"):
        value = str(tags.get(key) or "").strip()
        if value.startswith("File:"):
            return _commons_file_url(value[len("File:"):])
    image = str(tags.get("image") or "").strip()
    if image.startswith("https://") and urlparse(image).netloc.endswith("wikimedia.org"):
        return image
    return None


def _osm_address(tags: dict[str, Any]) -> dict[str, str]:
    """schema.org PostalAddress fields from OSM `addr:*` tags (may be empty)."""
    address: dict[str, str] = {}
    if tags.get("addr:full"):
        address["streetAddress"] = str(tags["addr:full"])
    elif tags.get("addr:street"):
        number = tags.get("addr:housenumber")
        address["streetAddress"] = f"{tags['addr:street']}, {number}" if number else str(tags["addr:street"])
    for tag, field in (("addr:postcode", "postalCode"), ("addr:city", "addressLocality")):
        if tags.get(tag):
            address[field] = str(tags[tag])
    return address


def _web_url(value: Any) -> str | None:
    """Normalise an OSM website tag to an absolute http(s) URL, or None."""
    url = str(value or "").strip().split(";")[0].strip()
    if not url or " " in url:
        return None
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parts = urlsplit(url)
    if not parts.hostname:
        return None
    # RFC 3986 URIs are ASCII: punycode the host, percent-encode the rest.
    try:
        host = parts.hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    netloc = f"{host}:{parts.port}" if parts.port else host
    safe = "/%:@!$&'()*+,;=-._~?"
    return urlunsplit((parts.scheme, netloc, quote(parts.path, safe=safe), quote(parts.query, safe=safe), ""))


def _wikipedia_url(tag: Any) -> str | None:
    """OSM `wikipedia=lang:Title` -> https://lang.wikipedia.org/wiki/Title."""
    lang, sep, title = str(tag or "").partition(":")
    if not sep or not title or not lang.isalpha():
        return None
    return f"https://{lang}.wikipedia.org/wiki/{quote(title.strip().replace(' ', '_'))}"


def _osm_object_url(poi_id: str) -> str:
    """`node:123` -> https://www.openstreetmap.org/node/123 (the source object)."""
    osm_type, _, osm_id = str(poi_id).partition(":")
    return f"https://www.openstreetmap.org/{osm_type}/{osm_id}"


def _extract_pois(elements: list[dict[str, Any]],
                  languages: Sequence[str] = DEFAULT_DESCRIPTION_LANGUAGES) -> list[dict[str, Any]]:
    keys = _description_keys(languages)
    pois: list[dict[str, Any]] = []
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
            "osm_type": el.get("type"),
            "name": name,
            "lat": lat,
            "lon": lon,
            "category": _poi_category(tags),
            "information": tags.get("information"),
            "description": next((clean_text(tags[k]) for k, _ in keys if tags.get(k)), None),
            "description_language": next((lang for k, lang in keys if tags.get(k)), None),
            "address": _osm_address(tags),
            "website": _web_url(tags.get("website") or tags.get("contact:website")),
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "opening_hours": tags.get("opening_hours"),
            "image": _osm_image(tags),
            "wikidata": tags.get("wikidata"),
            "wikipedia": tags.get("wikipedia"),
            "source": "osm",
            "tags": dict(tags),
        }
        pois.append(poi)
    return pois


def _is_low_value_information(poi: dict[str, Any]) -> bool:
    return poi.get("category") == "information" and poi.get("information") not in _USEFUL_INFORMATION


def _is_falla(poi: dict[str, Any]) -> bool:
    """Falla monuments stand only during the Fallas (March) and are burnt on
    the 19th; OSM maps where each commission plants one. Not a year-round POI."""
    return (poi.get("tags") or {}).get("artwork_type") == "falla"


# OSM `wikidata`/`wikipedia` tags sometimes point to a namesake or to the
# wrong place (the artwork "Marqués del Túria" links the noble title, the
# bullring links Alicante's), which would import an unrelated description,
# image and popularity. An item is trusted if it is located near the POI;
# large areas (a beach, the Albufera) have their point kilometres away.
MAX_WIKIDATA_DISTANCE_M = 1500.0
MAX_WIKIDATA_DISTANCE_AREA_M = 10000.0
_AREA_CATEGORIES = {"beach", "nature_reserve", "park", "garden"}
# Without coordinates, monuments named after someone usually link the person.
_NAMESAKE_CATEGORIES = {"artwork", "memorial", "monument"}
_WIKIDATA_HUMAN = "Q5"


def _wikidata_coordinates(ent: dict[str, Any] | None) -> tuple[float, float] | None:
    for claim in ((ent or {}).get("claims") or {}).get("P625") or []:
        value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}
        if "latitude" in value and "longitude" in value:
            return float(value["latitude"]), float(value["longitude"])
    return None


def _wikidata_matches_poi(poi: dict[str, Any], ent: dict[str, Any] | None) -> bool:
    """Whether the linked item is this place (see MAX_WIKIDATA_DISTANCE_M).

    Items without coordinates are accepted only when their label names the
    POI, they are not a person, and the POI is not a monument to someone.
    """
    coords = _wikidata_coordinates(ent)
    subcategory = str(poi.get("category") or "")
    if coords is not None:
        is_area = canonical_category(subcategory) in _AREA_CATEGORIES
        limit = MAX_WIKIDATA_DISTANCE_AREA_M if is_area else MAX_WIKIDATA_DISTANCE_M
        return _haversine_m(float(poi["lat"]), float(poi["lon"]), *coords) <= limit
    if subcategory in _NAMESAKE_CATEGORIES or canonical_category(subcategory) in _NAMESAKE_CATEGORIES:
        return False
    classes = {
        ((c.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
        for c in ((ent or {}).get("claims") or {}).get("P31") or []
    }
    if _WIKIDATA_HUMAN in classes:
        return False
    labels = [v.get("value", "") for v in ((ent or {}).get("labels") or {}).values()]
    return any(label and names_match(str(poi.get("name") or ""), label) for label in labels)


def _record_quality_rank(poi: dict[str, Any]) -> tuple[int, int, int]:
    """Lower is better: the best record of a duplicate group becomes the primary
    (keeps id, name, category and position); the rest only fill missing fields."""
    return (
        1 if _is_low_value_information(poi) else 0,
        0 if poi.get("wikidata") else 1,
        0 if poi.get("osm_type") in ("way", "relation") else 1,
    )


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


def _google_periods_to_opening_hours(periods: list[dict[str, Any]] | None) -> str | None:
    if not periods:
        return None

    # Google uses 0=Sunday..6=Saturday; we expose in Mo-Su order
    day_map = {0: "Su", 1: "Mo", 2: "Tu", 3: "We", 4: "Th", 5: "Fr", 6: "Sa"}
    ordered_days = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    slots: dict[str, tuple[str, str] | None] = {d: None for d in ordered_days}

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

    def flush(group_days: list[str], slot: tuple[str, str], acc: list[str]) -> None:
        if not group_days:
            return
        prefix = group_days[0] if len(group_days) == 1 else f"{group_days[0]}-{group_days[-1]}"
        acc.append(f"{prefix} {slot[0]}-{slot[1]}")

    segments: list[str] = []
    current_days: list[str] = []
    current_slot: tuple[str, str] | None = None

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
    best: dict[str, Any] | None = None
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


def _merge_records(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key, value in secondary.items():
        if not merged.get(key) and value:
            merged[key] = value
    if primary.get("tags") or secondary.get("tags"):
        merged["tags"] = {**(secondary.get("tags") or {}), **(primary.get("tags") or {})}
    if primary.get("source") and secondary.get("source"):
        sources = set(str(primary.get("source")).split("+")) | set(
            str(secondary.get("source")).split("+")
        )
        merged["source"] = "+".join(sorted(sources))
    return merged


def _deduplicate_pois(
    pois: list[dict[str, Any]],
    max_distance_m: float,
    min_name_similarity: float,
    prefer_sources: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def source_rank(poi: dict[str, Any]) -> int:
        source = str(poi.get("source") or "")
        for idx, name in enumerate(prefer_sources):
            if name in source:
                return idx
        return len(prefer_sources)

    remaining = sorted(pois, key=lambda poi: (source_rank(poi), *_record_quality_rank(poi)))
    result: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    while remaining:
        current = remaining.pop(0)
        if current.get("name") is None or current.get("lat") is None or current.get("lon") is None:
            continue
        merged = current
        to_remove: list[int] = []
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


_NAME_STOPWORDS = {
    "de", "del", "dels", "la", "les", "el", "els", "los", "las", "lo", "y", "i", "a", "al", "en",
    "the", "of", "and", "valencia", "file", "jpg", "jpeg", "png", "img", "dsc",
}


_INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")


def clean_text(text: Any) -> str:
    """Remove invisible characters (Wikipedia leaves zero-width spaces where
    references were) and collapse whitespace, including before punctuation."""
    text = _INVISIBLE_CHARS.sub("", str(text or ""))
    text = " ".join(text.split())
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def _shorten_to_sentence(text: str, max_chars: int = 900) -> str:
    """Clean the text and cut at the last full sentence within max_chars."""
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > 0 else cut.rstrip() + "…"


def _fetch_overpass_data(
    overpass_urls: list[str], query: str, timeout_seconds: int
) -> dict[str, Any]:
    last_error: Exception | None = None
    errors: list[str] = []
    data = None
    headers = {
        "User-Agent": USER_AGENT,
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


def _wikipedia_sitelinks(ent: dict[str, Any] | None, languages: list[str]) -> list[tuple[str, str]]:
    """(lang, title) of the item's Wikipedia articles in the given languages."""
    sitelinks = (ent or {}).get("sitelinks", {}) or {}
    return [
        (lang, sitelinks[f"{lang}wiki"]["title"])
        for lang in languages
        if (sitelinks.get(f"{lang}wiki") or {}).get("title")
    ]


def _choose_article(poi: dict[str, Any], ent: dict[str, Any] | None,
                    languages: Sequence[str] = DEFAULT_DESCRIPTION_LANGUAGES) -> tuple[str, str] | None:
    """Wikipedia article (lang, title) for descriptions, in the configured order of
    languages (Valencia: Spanish > English > Valencian, the interface's languages
    first). The OSM `wikipedia` tag is used only when Wikidata has none of them."""
    sitelinks = (ent or {}).get("sitelinks", {}) or {}
    for lang in languages:
        if (sitelinks.get(f"{lang}wiki") or {}).get("title"):
            return lang, sitelinks[f"{lang}wiki"]["title"]
    tag = poi.get("wikipedia")
    if isinstance(tag, str) and tag:
        lang, sep, title = tag.partition(":")
        if sep and title:
            return lang or "es", title
        return "es", tag
    return None


# Kinds of place worth listing even when nothing more is known about them.
_DESTINATION_CATEGORIES = {"museum", "arts_centre", "zoo", "aquarium"}
_HERITAGE_REPORT_FIELDS = ["poi_id", "poi_name", "igpcv", "heritage_name", "designation", "category", "distance_m"]
_DEDUP_REPORT_FIELDS = [
    "name_a", "source_a", "id_a", "lat_a", "lon_a",
    "name_b", "source_b", "id_b", "lat_b", "lon_b",
    "distance_m", "name_similarity",
]
MIN_CATEGORY_SIZE = 5
DEFAULT_LOCALITY = "València"
DEFAULT_COUNTRY = "ES"


def _aggregate_artworks(
    pois: list[dict[str, Any]], collections: list[dict[str, Any]], overpass_urls: list[str], timeout: int,
) -> tuple[list[dict[str, Any]], int]:
    """Replace the artworks inside an area (e.g. a university campus with an
    open-air sculpture collection) by one POI for the whole collection.

    Dozens of individual sculptures flood the catalog with places almost no
    tourist visits (UPV campus: 65 POIs, median 1 Flickr visitor), while the
    collection as a whole is worth a visit. The new POI takes the area's OSM
    relation as identity (so its real shape is used for visits), the given
    Wikidata item for enrichment, and a generated description listing artists.
    """
    from pmv_turismo.geometry import fetch_geometries, points_in_shape

    removed = 0
    for col in collections:
        rid = int(col["area_relation"])
        shapes = fetch_geometries([{"id": f"urn:ngsi-ld:PointOfInterest:relation:{rid}"}], overpass_urls, timeout)
        shape = next(iter(shapes.values()), None)
        if not shape:
            raise FetchError(f"No geometry for OSM relation {rid}")
        artworks = [p for p in pois if p.get("category") == "artwork"]
        inside = points_in_shape(shape, [float(p["lat"]) for p in artworks], [float(p["lon"]) for p in artworks])
        members = [p for p, ok in zip(artworks, inside, strict=False) if ok]
        if len(members) < int(col.get("min_artworks", 5)):
            continue
        ids = {id(p) for p in members}
        pois = [p for p in pois if id(p) not in ids]
        removed += len(members)
        artists = list(dict.fromkeys(
            str((p.get("tags") or {}).get("artist_name")).strip() for p in members
            if (p.get("tags") or {}).get("artist_name")))
        text = f"Colección de {len(members)} esculturas y obras de arte al aire libre en el {col['area_name']}"
        if artists:
            text += ", con obras de " + ", ".join(artists[:6]) + (" y otros" if len(artists) > 6 else "")
        pois.append({
            "id": f"relation:{rid}", "osm_type": "relation", "name": col["name"],
            "lat": sum(float(p["lat"]) for p in members) / len(members),
            "lon": sum(float(p["lon"]) for p in members) / len(members),
            "category": "sculpture_collection", "description": text + ".",
            "description_generated": True, "description_language": "es",
            "address": {}, "website": None, "phone": None, "opening_hours": None, "image": None,
            "wikidata": col.get("wikidata"), "wikipedia": None, "source": "osm",
            "tags": {"aggregates": ";".join(str(p["id"]) for p in members)},
        })
        logger.info("%s: %d artworks aggregated into one POI", col["name"], len(members))
    return pois, removed


def run_ingestion(config: dict) -> dict[str, Any]:
    """OSM -> dedup -> enrichment (Wikidata/Wikipedia/Commons) -> NGSI-LD catalog.

    Writes data/processed/poi.jsonld and ingestion_report.json (counts, drop
    reasons and data-quality figures) and returns the report.
    """
    ingestion_cfg = config.get("ingestion", {})
    data_cfg = config.get("data", {})

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = int(ingestion_cfg.get("request_timeout_seconds", 120))
    http_timeout = min(timeout_seconds, 20)
    fetch_images = bool(ingestion_cfg.get("fetch_images", True))
    require_description = bool(ingestion_cfg.get("require_description", False))
    workers = max(1, int(ingestion_cfg.get("workers", 4)))
    languages = tuple(ingestion_cfg.get("description_languages") or DEFAULT_DESCRIPTION_LANGUAGES)
    pois: list[dict[str, Any]] = []
    report: dict[str, Any] = {"dropped": Counter()}

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
        elements = data.get("elements", [])
        pois.extend(_extract_pois(elements, languages))
        report["osm_elements"] = len(elements)
        report["osm_named"] = len(pois)

    dedup_cfg = ingestion_cfg.get("dedup", {})
    if dedup_cfg.get("enabled", False):
        max_distance_m = float(dedup_cfg.get("max_distance_m", 80))
        min_name_similarity = float(dedup_cfg.get("min_name_similarity", 0.82))
        prefer_sources = dedup_cfg.get("prefer_sources", ["osm"])
        pois, matches = _deduplicate_pois(pois, max_distance_m, min_name_similarity, prefer_sources)
        report_path = dedup_cfg.get("report_path", str(processed_dir / "dedup_matches.csv"))
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_DEDUP_REPORT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(matches)
        report["deduplicated"] = len(matches)

    before = len(pois)
    pois = [poi for poi in pois if not _is_low_value_information(poi)]
    report["dropped"]["information_board"] = before - len(pois)
    before = len(pois)
    pois = [poi for poi in pois if not _is_falla(poi)]
    report["dropped"]["falla"] = before - len(pois)
    if ingestion_cfg.get("artwork_collections"):
        pois, aggregated = _aggregate_artworks(
            pois, ingestion_cfg["artwork_collections"],
            ingestion_cfg.get("overpass_urls") or ["https://overpass-api.de/api/interpreter"], timeout_seconds)
        report["dropped"]["aggregated_into_collection"] = aggregated

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

    failures: list[str] = []

    def guarded(label: str, fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except FetchError as exc:
            failures.append(f"{label}: {exc}")
            return None

    logger.info("OSM: %d named POIs after dedup and filtering", len(pois))

    # 0. Protected heritage (GVA inventory): an official quality signal.
    heritage_cfg = ingestion_cfg.get("heritage", {})
    if heritage_cfg.get("enabled", False):
        municipality = str(heritage_cfg.get("municipality", "VALÈNCIA"))
        entries: list[dict[str, Any]] = []
        for designation, url in (heritage_cfg.get("sources") or {}).items():
            text = guarded(f"heritage:{designation}", _http_get_json, str(url), None, timeout_seconds, 5, False, True)
            if text:
                entries.extend(parse_inventory(text, str(designation), municipality))
        # Artworks stand next to protected buildings but are not them.
        candidates = [p for p in pois if canonical_category(p.get("category")) != "artwork"]
        rows = match_heritage(candidates, entries, _haversine_m, float(heritage_cfg.get("max_distance_m", 60)))
        report["heritage"] = {"inventory_entries": len(entries), "matched_pois": len(rows)}
        heritage_report = processed_dir / "heritage_matches.csv"
        with open(heritage_report, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_HERITAGE_REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Heritage: %d inventory entries, %d POIs matched", len(entries), len(rows))

    # 1. Wikidata items (batched).
    qids = [str(p["wikidata"]) for p in pois if p.get("wikidata")]
    wikidata = guarded("wikidata", _fetch_wikidata_entities, qids, http_timeout) or {}
    entity_of = {id(p): wikidata.get(str(p.get("wikidata"))) for p in pois}
    rejected = 0
    for poi in pois:
        qid = poi.get("wikidata")
        if qid and str(qid) in wikidata and not _wikidata_matches_poi(poi, entity_of[id(poi)]):
            logger.info("Wikidata %s is not located at %s: link ignored", qid, poi.get("name"))
            poi["wikidata"] = poi["wikipedia"] = None
            entity_of[id(poi)] = None
            rejected += 1
    report["wikidata_links_ignored"] = rejected
    article_of = {id(p): _choose_article(p, entity_of[id(p)], languages) for p in pois}
    logger.info("Wikidata: %d items", len(wikidata))

    # 2. Descriptions: Wikipedia lead when the OSM text is too short (batched per language).
    wanted: dict[str, list[str]] = {}
    for poi in pois:
        article = article_of[id(poi)]
        if article and len(poi.get("description") or "") < MIN_USEFUL_DESCRIPTION_CHARS:
            wanted.setdefault(article[0], []).append(article[1])
    extracts = {
        lang: guarded(f"extracts:{lang}", _fetch_wikipedia_extracts, lang, titles, 900, http_timeout) or {}
        for lang, titles in wanted.items()
    }
    # 2b. Without any source text, a description from the structured data
    # (OSM tags, heritage inventory) if it has enough facts.
    min_facts = int(ingestion_cfg.get("generated_description_min_facts", 1))
    kept: list[dict[str, Any]] = []
    for poi in pois:
        article = article_of[id(poi)]
        extract = extracts.get(article[0], {}).get(article[1]) if article else None
        if extract and len(extract) > len(poi.get("description") or ""):
            poi["description"] = extract
            poi["description_from_wikipedia"] = True
            poi["description_language"] = article[0]
        if not poi.get("description"):
            text, facts = build_description(poi.get("tags") or {}, str(poi.get("category") or ""), poi.get("heritage"))
            if facts >= min_facts or canonical_category(poi.get("category")) in _DESTINATION_CATEGORIES:
                poi["description"] = text
                poi["description_generated"] = True
                poi["description_language"] = "es"
        if require_description and not poi.get("description"):
            report["dropped"]["no_description"] += 1
            continue
        kept.append(poi)
    pois = kept
    logger.info("Descriptions: %d extracts fetched, %d POIs kept", sum(map(len, extracts.values())), len(pois))

    # 3. Popularity: pageviews summed over the configured Wikipedia editions (batched).
    pageview_languages = [str(x) for x in ingestion_cfg.get("pageview_languages", DEFAULT_PAGEVIEW_LANGUAGES)]
    pageview_days = int(ingestion_cfg.get("pageview_days", 60))
    articles_of: dict[int, set] = {}
    by_lang: dict[str, list[str]] = {}
    for poi in pois:
        articles = set(_wikipedia_sitelinks(entity_of[id(poi)], pageview_languages))
        if article_of[id(poi)]:
            articles.add(article_of[id(poi)])
        articles_of[id(poi)] = articles
        for lang, title in articles:
            by_lang.setdefault(lang, []).append(title)
    views_by_lang = {
        lang: guarded(f"pageviews:{lang}", _fetch_pageviews, lang, titles, pageview_days, http_timeout) or {}
        for lang, titles in by_lang.items()
    }
    logger.info("Pageviews: %d articles in %d editions", sum(map(len, by_lang.values())), len(by_lang))
    popularity = [
        math.log10(1 + sum(views_by_lang.get(lang, {}).get(title, 0) for lang, title in articles_of[id(poi)]))
        for poi in pois
    ]

    # 4. Images: OSM (Commons only) > Wikidata P18. No search by name: in the
    # manual audit it returned a photo of something else 4 times out of 5
    # ("Lluís Vives" -> the Lluís Vives school). No image beats a wrong one.
    def find_image(poi: dict[str, Any]) -> str | None:
        if poi.get("image") or not fetch_images:
            return poi.get("image")
        filename = _extract_wikidata_filename(entity_of[id(poi)])
        return _commons_file_url(filename) if filename else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = list(pool.map(find_image, pois))

    locality = str(ingestion_cfg.get("address_locality", DEFAULT_LOCALITY))
    country = str(ingestion_cfg.get("address_country", DEFAULT_COUNTRY))
    entities = [_build_entity(poi, image, locality, country) for poi, image in zip(pois, images, strict=False)]
    logger.info("Images: %d/%d POIs", sum(1 for i in images if i), len(pois))
    _assign_relevance(entities, popularity)
    report["pois"] = len(entities)
    report["dropped"] = dict(report["dropped"])
    report["fetch_errors"] = sorted(failures)[:50]
    report["quality"] = _quality_summary(entities)
    (processed_dir / "ingestion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # A partial catalog (sources rate-limited or down) must not replace a good one.
    if failures and not ingestion_cfg.get("allow_partial", False):
        raise RuntimeError(
            f"{len(failures)} enrichment requests failed (see ingestion_report.json); "
            "poi.jsonld was not updated. Retry later or set ingestion.allow_partial: true."
        )
    (processed_dir / "poi.jsonld").write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def data_provider(wikidata: bool, wikipedia: bool, commons_image: bool, heritage: bool) -> str:
    """Only the sources whose content the entity actually carries (attribution)."""
    parts = ["OpenStreetMap contributors"]
    if wikidata:
        parts.append("Wikidata")
    if wikipedia:
        parts.append("Wikipedia (CC BY-SA)")
    if commons_image:
        parts.append("Wikimedia Commons")
    if heritage:
        parts.append("Generalitat Valenciana (Inventario General del Patrimonio Cultural Valenciano)")
    return ", ".join(parts)


def _build_entity(
    poi: dict[str, Any],
    image_url: str | None,
    locality: str = DEFAULT_LOCALITY,
    country: str = DEFAULT_COUNTRY,
) -> dict[str, Any]:
    """Smart Data Models PointOfInterest entity (NGSI-LD, keyValues format).

    Standard attributes follow dataModel.PointOfInterest; `subcategory`,
    `duration` and `durationEstimated` are project extensions (see README).
    `openingHours` only when the source has them: an invented schedule would
    hide open places from the recommendations and mislead whoever reads it.
    """
    subcategory = poi.get("category")
    category = canonical_category(subcategory)
    heritage = poi.get("heritage") or {}
    address = {"addressLocality": locality, "addressCountry": country, **(poi.get("address") or {})}
    see_also = [
        poi.get("website"),
        f"https://www.wikidata.org/wiki/{poi['wikidata']}" if poi.get("wikidata") else None,
        _wikipedia_url(poi.get("wikipedia")),
    ]
    entity = {
        "id": f"urn:ngsi-ld:PointOfInterest:{poi.get('id')}",
        "type": "PointOfInterest",
        "name": str(poi["name"]),
        "description": poi.get("description"),
        "category": [category] if category else [],
        "location": {"type": "Point", "coordinates": [float(poi["lon"]), float(poi["lat"])]},
        "address": address,
        "image": image_url,
        "source": _osm_object_url(str(poi.get("id"))),
        "dataProvider": data_provider(
            wikidata=bool(poi.get("wikidata")),
            wikipedia=bool(poi.get("description_from_wikipedia")),
            commons_image=bool(image_url and "wikimedia.org" in image_url),
            heritage=bool(heritage),
        ),
        "refSeeAlso": list(dict.fromkeys(url for url in see_also if url)),
        "contactPoint": {"telephone": poi.get("phone")} if poi.get("phone") else None,
        "openingHours": poi.get("opening_hours") or None,
        "subcategory": subcategory,
        "duration": estimate_visit_minutes(subcategory),
        "durationEstimated": True,
        "descriptionGenerated": True if poi.get("description_generated") else None,
        "descriptionLanguage": poi.get("description_language"),
        "heritageDesignation": heritage.get("designation"),
        "heritageInventoryId": heritage.get("id"),
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
            "https://raw.githubusercontent.com/smart-data-models/dataModel.PointOfInterest/master/context.jsonld",
        ],
    }
    return {k: v for k, v in entity.items() if v not in (None, [], "")}


def _percentile_ranks(values: list[float]) -> list[float]:
    """Mean-rank percentile in [0, 1] for each value (ties share the rank)."""
    if len(values) == 1:
        return [0.5]
    order = sorted(values)
    positions: dict[float, list[int]] = {}
    for i, v in enumerate(order):
        positions.setdefault(v, []).append(i)
    return [sum(positions[v]) / len(positions[v]) / (len(values) - 1) for v in values]


def _assign_relevance(entities: list[dict[str, Any]], popularity: list[float]) -> None:
    """`relevance` = inverse popularity percentile within the canonical category.

    0.05 = most popular of its category, 0.95 = least known. Categories with
    fewer than MIN_CATEGORY_SIZE POIs are ranked against the whole catalog,
    otherwise a lone POI would always get the uninformative 0.5.
    """
    by_category: dict[str, list[int]] = {}
    for idx, entity in enumerate(entities):
        by_category.setdefault((entity.get("category") or ["unknown"])[0], []).append(idx)
    global_ranks = _percentile_ranks(popularity) if popularity else []
    for members in by_category.values():
        if len(members) >= MIN_CATEGORY_SIZE:
            ranks = _percentile_ranks([popularity[i] for i in members])
        else:
            ranks = [global_ranks[i] for i in members]
        for idx, p in zip(members, ranks, strict=False):
            entities[idx]["relevance"] = round(1.0 - (0.05 + 0.9 * p), 4)


def _quality_summary(entities: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(entities))
    descriptions = [len(e.get("description") or "") for e in entities]
    return {
        "categories": dict(Counter((e.get("category") or ["none"])[0] for e in entities).most_common()),
        "with_image_pct": round(100 * sum(1 for e in entities if e.get("image")) / n, 1),
        "with_street_address_pct": round(
            100 * sum(1 for e in entities if (e.get("address") or {}).get("streetAddress")) / n, 1
        ),
        "opening_hours_pct": round(100 * sum(1 for e in entities if e.get("openingHours")) / n, 1),
        "description_chars_median": sorted(descriptions)[len(descriptions) // 2] if descriptions else 0,
        "short_descriptions": sum(1 for d in descriptions if d < MIN_USEFUL_DESCRIPTION_CHARS),
        "generated_descriptions": sum(1 for e in entities if e.get("descriptionGenerated")),
        "protected_heritage": dict(Counter(e["heritageDesignation"] for e in entities if e.get("heritageDesignation"))),
    }
