from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from pmv_turismo.common import (
    LOCAL_TZ,
    now_local,
    opening_hours_cover_duration,
    poi_coordinates,
    write_json_atomic,
)


def _ors_matrix(
    endpoint: str,
    api_key: str,
    profile: str,
    locations: List[List[float]],
) -> Dict[str, Any]:
    url = f"{endpoint}/v2/matrix/{profile}"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    payload = {"locations": locations, "metrics": ["distance", "duration"], "units": "m"}
    backoff = 2
    for attempt in range(3):
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 429 and attempt < 2:
            time.sleep(backoff)
            backoff *= 2
            continue
        response.raise_for_status()
        return response.json()
    return {}


def _osrm_table(
    endpoint: str,
    profile: str,
    locations: List[List[float]],
) -> Dict[str, Any]:
    # OSRM table service; profile examples: foot, car, bike
    coord_str = ";".join([f"{lon},{lat}" for lon, lat in locations])
    url = f"{endpoint}/table/v1/{profile}/{coord_str}?annotations=duration,distance"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def _osrm_route(
    endpoint: str,
    profile: str,
    coordinates: List[List[float]],
) -> Dict[str, Any]:
    coord_str = ";".join([f"{lon},{lat}" for lon, lat in coordinates])
    url = f"{endpoint}/route/v1/{profile}/{coord_str}?overview=full&geometries=polyline6&steps=true"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def _routing_settings(planning_cfg: Dict[str, Any]) -> Dict[str, str]:
    routing_cfg = planning_cfg.get("routing", {})
    provider = routing_cfg.get("provider", "osrm")
    profile = routing_cfg.get("profile", "foot")
    if provider == "openrouteservice":
        api_key = routing_cfg.get("api_key") or ""
        if not api_key:
            raise ValueError("OpenRouteService requires ORS_API_KEY (environment or .env)")
        return {
            "provider": provider,
            "endpoint": routing_cfg.get("endpoint") or "https://api.openrouteservice.org",
            "profile": profile or "foot-walking",
            "api_key": api_key,
        }
    profile = {
        "foot-walking": "foot", "walking": "foot", "pedestrian": "foot",
        "driving": "car", "cycling": "bike",
    }.get(profile, profile)
    return {
        "provider": "osrm",
        "endpoint": routing_cfg.get("endpoint") or "https://router.project-osrm.org",
        "profile": profile,
        "api_key": "",
    }


def _start_datetime(planning_cfg: Dict[str, Any], start_at: Optional[datetime]) -> datetime:
    if start_at is not None:
        return start_at.astimezone(LOCAL_TZ)
    now = now_local().replace(second=0, microsecond=0)
    minutes = planning_cfg.get("start_time_minutes")
    if not isinstance(minutes, (int, float)):
        raw = planning_cfg.get("start_time")
        try:
            h, m = [int(x) for x in str(raw).split(":", 1)]
            minutes = h * 60 + m
        except (TypeError, ValueError):
            return now
    return now.replace(hour=0, minute=0) + timedelta(minutes=int(minutes))


def _field(candidate: Dict[str, Any], key: str) -> Any:
    value = candidate.get(key)
    if value is None:
        value = (candidate.get("poi") or {}).get(key)
    return value


def plan_itinerary(
    config: dict,
    candidates: List[Dict[str, Any]],
    visit_time_override: Optional[int] = None,
    start_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Score-aware greedy itinerary with time, stop and opening-hour limits.

    At each step the next stop maximises `score / (1 + travel / penalty)`:
    with equal scores this is nearest-neighbour, otherwise a slightly longer
    walk is accepted for a clearly better POI. Stops that would not fit in the
    time budget, or that are closed at arrival, are skipped.

    `visit_time_override` (the user's chosen time per visit) replaces POI
    durations; without it each POI uses its own `duration` and falls back to
    `visit_time_minutes`.
    """
    planning_cfg = config.get("parameters", {}).get("planning", {})
    max_stops = int(planning_cfg.get("max_stops_per_day", 6))
    default_visit = int(planning_cfg.get("visit_time_minutes", 60))
    max_total = int(planning_cfg.get("max_total_minutes", 420))
    max_candidates = int(planning_cfg.get("max_candidates", 20))
    travel_penalty = max(1.0, float(planning_cfg.get("travel_penalty_minutes", 15)))
    respect_hours = bool(planning_cfg.get("respect_opening_hours", True))

    start = planning_cfg.get("start_location", {})
    if "lat" not in start or "lon" not in start:
        raise ValueError("start_location must include lat and lon")
    start_lat = float(start["lat"])
    start_lon = float(start["lon"])
    if not (-90 <= start_lat <= 90 and -180 <= start_lon <= 180):
        raise ValueError("start_location out of bounds")

    routing = _routing_settings(planning_cfg)

    remaining = [c for c in candidates if poi_coordinates(c)][:max_candidates]
    locations = [[start_lon, start_lat]] + [poi_coordinates(c)[:2] for c in remaining]

    try:
        if routing["provider"] == "openrouteservice":
            matrix = _ors_matrix(routing["endpoint"], routing["api_key"], routing["profile"], locations)
        else:
            matrix = _osrm_table(routing["endpoint"], routing["profile"], locations)
    except Exception as exc:
        raise ValueError(f"Routing matrix request failed ({routing['provider']}): {exc}") from exc
    distances = matrix.get("distances") or []
    durations = matrix.get("durations") or []

    start_dt = _start_datetime(planning_cfg, start_at)
    itinerary: List[Dict[str, Any]] = []
    visited_locations = [locations[0]]
    current_idx = 0
    total_minutes = 0
    total_distance_km = 0.0
    remaining_indices = list(range(1, len(locations)))

    while remaining_indices and len(itinerary) < max_stops and distances and durations:
        best = None
        for idx in remaining_indices:
            dist = distances[current_idx][idx]
            dur = durations[current_idx][idx]
            if dist is None or dur is None:
                continue
            poi = remaining[idx - 1]
            travel_minutes = int(round(dur / 60))
            visit_minutes = int(visit_time_override or _field(poi, "duration") or default_visit)
            if total_minutes + travel_minutes + visit_minutes > max_total:
                continue
            arrival = start_dt + timedelta(minutes=total_minutes + travel_minutes)
            if respect_hours and not opening_hours_cover_duration(
                _field(poi, "openingHours"), arrival, visit_minutes
            ):
                continue
            score = _field(poi, "score")
            score = float(score) if isinstance(score, (int, float)) else 1.0
            utility = max(score, 1e-3) / (1.0 + travel_minutes / travel_penalty)
            key = (utility, -dist)
            if best is None or key > best[0]:
                best = (key, idx, dist, travel_minutes, visit_minutes)

        if best is None:
            break

        _, idx, dist, travel_minutes, visit_minutes = best
        poi = remaining[idx - 1]
        arrival_dt = start_dt + timedelta(minutes=total_minutes + travel_minutes)
        departure_dt = arrival_dt + timedelta(minutes=visit_minutes)
        total_minutes += travel_minutes + visit_minutes
        total_distance_km += dist / 1000.0

        flattened = {key: _field(poi, key) for key in ("id", "name", "location", "category")}
        itinerary.append(
            {
                "poi": poi,
                "travelMinutes": travel_minutes,
                "visitMinutes": visit_minutes,
                "duration_minutes": visit_minutes,
                "arrival_time": arrival_dt.strftime("%H:%M"),
                "departure_time": departure_dt.strftime("%H:%M"),
                "order": len(itinerary) + 1,
                **{k: v for k, v in flattened.items() if v is not None},
            }
        )
        visited_locations.append(locations[idx])
        current_idx = idx
        remaining_indices.remove(idx)

    output: Dict[str, Any] = {
        "startLocation": start,
        "startTime": start_dt.strftime("%H:%M"),
        "totalMinutes": total_minutes,
        "total_duration_minutes": total_minutes,
        "totalDistanceKm": round(total_distance_km, 2),
        "total_distance_km": round(total_distance_km, 2),
        "stops": itinerary,
    }

    # Route geometry for the final stop order (OSRM only), for the frontend map.
    if routing["provider"] == "osrm" and len(visited_locations) > 1:
        try:
            route = (_osrm_route(routing["endpoint"], routing["profile"], visited_locations).get("routes") or [{}])[0]
            output["polyline"] = route.get("geometry")
            legs = [
                {"distance_m": leg.get("distance"), "duration_s": leg.get("duration"), "summary": leg.get("summary")}
                for leg in route.get("legs") or []
            ]
            if legs:
                output["legs"] = legs
        except Exception:
            pass

    return output


def write_itinerary(output: Dict[str, Any], path: Path) -> None:
    write_json_atomic(path, output)
