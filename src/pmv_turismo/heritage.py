"""Protected heritage from the Inventario General del Patrimonio Cultural Valenciano.

The Generalitat Valenciana publishes the inventory's immovable property as two
CSV files (BIC = Bien de Interés Cultural, BRL = Bien de Relevancia Local) on
dadesobertes.gva.es under CC BY 4.0. They carry the official name, the
category and UTM coordinates, but no description.

The catalog uses them as an official quality signal: an OSM POI matched to an
inventory entry is protected heritage, which is a reason to keep it even when
no source has a description for it.
"""

from __future__ import annotations

import csv
import io
import math
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple

# ETRS89 / UTM zone 30N (EPSG:25830). GRS80 and WGS84 differ by < 1 mm here.
_A = 6378137.0
_F = 1 / 298.257222101
_K0 = 0.9996
_ZONE = 30

DESIGNATION_LABEL = {
    "BIC": "Bien de Interés Cultural",
    "BRL": "Bien de Relevancia Local",
}

# Inventory categories that are places a visitor can go to. Movable items
# (bells, coats of arms inside buildings), museum collections and archives are
# not destinations.
VISITABLE_CATEGORIES = {
    "Monumento",
    "Jardín histórico",
    "Conjunto histórico",
    "Sitio histórico",
    "Zona arqueológica",
    "Monumento de interés local",
    "Espacio etnológico de interés local",
    "Sitio histórico de interés local",
    "Jardín histórico de interés local",
    "Espacio de protección arqueológica",
}


# Inventory names are in Spanish (sometimes Valencian), OSM names mostly in
# Valencian: map the frequent words to one form before comparing.
_WORD_FORMS = {
    "sant": "san", "santo": "san", "sants": "santos", "santes": "santas",
    "palau": "palacio", "monestir": "monasterio", "convent": "monasterio", "convento": "monasterio",
    "pont": "puente", "jardi": "jardin", "jardins": "jardin", "jardines": "jardin",
    "mercat": "mercado", "llotja": "lonja", "creu": "cruz", "joan": "juan", "joans": "juanes",
    "josep": "jose", "miquel": "miguel", "vicent": "vicente", "llorenc": "lorenzo", "marti": "martin",
    "agusti": "agustin", "caterina": "catalina", "nicolau": "nicolas", "esteve": "esteban",
    "jaume": "santiago", "jaime": "santiago", "pere": "pedro", "pau": "pablo", "benet": "benito",
    "lluis": "luis", "bartomeu": "bartolome", "roc": "roque", "reis": "reyes", "capella": "capilla",
    "mare": "virgen", "senyora": "virgen", "senora": "virgen", "deu": "", "nostra": "", "nuestra": "",
    "reial": "real", "torres": "torre", "banys": "banos", "almirall": "almirante",
    "moli": "molino", "refugi": "refugio", "drassanes": "atarazanas", "grau": "grao",
    "museu": "museo", "collegi": "colegio", "col": "", "legi": "colegio",
    "comtes": "condes", "trinitat": "trinidad", "bous": "toros", "placa": "plaza", "retaule": "retablo",
}
# Words that say what kind of building it is but not which one.
_GENERIC_WORDS = {
    "iglesia", "esglesia", "parroquia", "parroquial", "templo", "edificio", "antiguo", "antic",
    "valencia", "de", "del", "dels", "la", "les", "el", "els", "los", "las", "lo", "y", "i",
    "a", "al", "en", "o", "d", "l", "e",
}
# Inventory entries that are objects or features of a place (a tiled
# altarpiece on a façade, a shelter under a square) and squares: they only
# match an OSM name that says the same thing.
_KIND_WORDS = {"retablo", "panells", "refugio", "fumeral", "escudo", "campana", "plaza", "sitges", "senia"}
_WORD_SIMILARITY = 0.75


def _normalize(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    text = "".join(c if c.isalnum() else " " for c in text)
    return " ".join(text.split())


def _words(text: str) -> List[str]:
    words = [_WORD_FORMS.get(w, w) for w in _normalize(text).split()]
    return [w for w in words if w and w not in _GENERIC_WORDS]


def _similar_word(word: str, others: Iterable[str]) -> bool:
    return any(word == o or SequenceMatcher(None, word, o).ratio() >= _WORD_SIMILARITY for o in others)


def names_match(osm_name: str, heritage_name: str) -> bool:
    """Whether an OSM name and an inventory name denote the same place.

    Every distinctive word of the OSM name must appear, allowing spelling
    variants, in the usually longer official name: "Catedral de València" ~
    "Santa Iglesia Catedral Basílica Metropolitana de Santa María". Overall
    string similarity is not used: "Jardí de la Generalitat" and "Palau de la
    Generalitat" are close strings but different places. A kind word
    (altarpiece, shelter, square) on either side must be on both.
    """
    osm_words, heritage_words = _words(osm_name), _words(heritage_name)
    if not osm_words:
        return False
    if (set(osm_words) ^ set(heritage_words)) & _KIND_WORDS:
        return False
    return all(_similar_word(w, heritage_words) for w in osm_words)


def utm_to_latlon(easting: float, northing: float, zone: int = _ZONE) -> Tuple[float, float]:
    """Inverse transverse Mercator (northern hemisphere), accurate to centimetres."""
    e2 = _F * (2 - _F)
    ep2 = e2 / (1 - e2)
    x = easting - 500000.0
    m = northing / _K0
    mu = m / (_A * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    sin1, cos1, tan1 = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    n1 = _A / math.sqrt(1 - e2 * sin1**2)
    t1 = tan1**2
    c1 = ep2 * cos1**2
    r1 = _A * (1 - e2) / (1 - e2 * sin1**2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * tan1 / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720
    )
    lon = (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120
    ) / cos1
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat), math.degrees(lon0 + lon)


def _coordinates(row: Dict[str, str]) -> Tuple[float, float] | None:
    """(lat, lon) of a row. The BIC file swaps the `utmeste`/`utmnorte`
    columns, so the axis is told apart by magnitude: in this zone eastings
    have 6 digits and northings 7."""
    try:
        a, b = float(row["utmeste"]), float(row["utmnorte"])
    except (KeyError, TypeError, ValueError):
        return None
    easting, northing = (a, b) if a < b else (b, a)
    if not (100_000 <= easting <= 900_000 and 3_000_000 <= northing <= 6_000_000):
        return None
    return utm_to_latlon(easting, northing)


def parse_inventory(text: str, designation: str, municipality: str) -> List[Dict[str, Any]]:
    """Visitable entries of one inventory CSV (`;`-separated) in a municipality."""
    entries: List[Dict[str, Any]] = []
    wanted = municipality.strip().upper()
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿")), delimiter=";"):
        if (row.get("municipio") or "").strip().upper() != wanted:
            continue
        category = (row.get("categoria") or "").strip()
        if category not in VISITABLE_CATEGORIES:
            continue
        coords = _coordinates(row)
        if coords is None:
            continue
        entries.append({
            "id": (row.get("igpcv") or "").strip(),
            "name": " ".join((row.get("denominacion") or "").split()),
            "category": category,
            "designation": designation,
            "lat": coords[0],
            "lon": coords[1],
        })
    return entries


def match_heritage(
    pois: Iterable[Dict[str, Any]],
    entries: List[Dict[str, Any]],
    distance_fn: Any,
    max_distance_m: float = 60.0,
) -> List[Dict[str, Any]]:
    """Attach the closest inventory entry with a similar name to each POI.

    Sets `poi["heritage"]` and returns one report row per match. Position alone
    is not enough: a protected church and the square in front of it are a few
    metres apart. BIC wins over BRL when both match.
    """
    rows: List[Dict[str, Any]] = []
    for poi in pois:
        best: Tuple[int, float, Dict[str, Any]] | None = None
        for entry in entries:
            dist = distance_fn(float(poi["lat"]), float(poi["lon"]), entry["lat"], entry["lon"])
            if dist > max_distance_m:
                continue
            if not names_match(str(poi["name"]), entry["name"]):
                continue
            key = (0 if entry["designation"] == "BIC" else 1, dist)
            if best is None or key < best[:2]:
                best = (*key, entry)
        if best is None:
            continue
        entry = best[2]
        poi["heritage"] = entry
        rows.append({
            "poi_id": poi.get("id"),
            "poi_name": poi.get("name"),
            "igpcv": entry["id"],
            "heritage_name": entry["name"],
            "designation": entry["designation"],
            "category": entry["category"],
            "distance_m": round(best[1], 1),
        })
    return rows
