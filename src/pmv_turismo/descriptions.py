"""Descriptions built from structured data, for POIs that no source describes.

Only facts present in the sources are written (OSM tags, the heritage
inventory), in Spanish, with fixed templates: nothing is inferred or
paraphrased. Values without a known translation are left out rather than
shown in English. The entity is flagged with `descriptionGenerated: true`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .heritage import DESIGNATION_LABEL

# What the place is, by OSM subcategory (artworks use `artwork_type` first).
_KIND = {
    "artwork": "Obra de arte pública",
    "sculpture": "Escultura", "statue": "Estatua", "bust": "Busto", "mural": "Mural",
    "installation": "Instalación artística", "graffiti": "Grafiti", "relief": "Relieve",
    "mosaic": "Mosaico", "fountain": "Fuente monumental",
    "place_of_worship": "Templo", "church": "Iglesia", "chapel": "Capilla", "cathedral": "Catedral",
    "monastery": "Monasterio", "convent": "Convento",
    "memorial": "Monumento conmemorativo", "monument": "Monumento", "plaque": "Placa conmemorativa",
    "stele": "Estela", "obelisk": "Obelisco",
    "ruins": "Restos históricos", "archaeological_site": "Yacimiento arqueológico",
    "castle": "Castillo", "manor": "Casa señorial", "palace": "Palacio", "tower": "Torre",
    "city_gate": "Puerta de la muralla", "bridge": "Puente", "building": "Edificio histórico",
    "bunker": "Refugio antiaéreo", "wayside_cross": "Cruz de término", "wayside_shrine": "Retablo devocional",
    "water_well": "Pozo histórico", "windmill": "Molino", "lighthouse": "Faro",
    "museum": "Museo", "gallery": "Galería de arte", "arts_centre": "Centro de arte",
    "park": "Parque", "garden": "Jardín", "beach": "Playa", "viewpoint": "Mirador",
    "attraction": "Lugar de interés", "marketplace": "Mercado",
    "nature_reserve": "Espacio natural",
}
_MATERIAL = {
    "bronze": "bronce", "steel": "acero", "metal": "metal", "stone": "piedra", "iron": "hierro",
    "marble": "mármol", "glass": "vidrio", "concrete": "hormigón", "wood": "madera",
    "brick": "ladrillo", "sandstone": "arenisca", "ceramic": "cerámica", "granite": "granito",
    "limestone": "piedra caliza", "plastic": "plástico", "aluminium": "aluminio",
}
_CIVILIZATION = {
    "islamic_iberia": "andalusí", "roman": "romana", "medieval_christianity": "cristiana medieval",
    "iberian": "ibérica", "visigothic": "visigoda",
}
_DENOMINATION = {
    "catholic": "católico", "roman_catholic": "católico", "orthodox": "ortodoxo",
    "baptist": "bautista", "evangelical": "evangélico", "protestant": "protestante",
}
_RELIGION = {"muslim": "Mezquita", "jewish": "Sinagoga", "buddhist": "Templo budista", "hindu": "Templo hindú"}

_MAX_INSCRIPTION_CHARS = 200


def _year(value: Any) -> str | None:
    match = re.search(r"\b(1\d{3}|20\d{2})\b", str(value or ""))
    return match.group(1) if match else None


def _kind(tags: Dict[str, Any], subcategory: str) -> str:
    if subcategory == "place_of_worship" and tags.get("religion") in _RELIGION:
        return _RELIGION[str(tags["religion"])]
    for value in (tags.get("artwork_type"), tags.get("memorial"), tags.get("building"), subcategory):
        if value in _KIND:
            return _KIND[str(value)]
    return "Lugar de interés"


def build_description(
    tags: Dict[str, Any], subcategory: str, heritage: Dict[str, Any] | None = None
) -> Tuple[str, int]:
    """(description, number of facts beyond the kind of place).

    The kind alone ("Escultura.") says nothing a category does not; the
    caller decides how many facts make a description worth keeping.
    """
    facts = 0
    head = _kind(tags, subcategory)
    material = _MATERIAL.get(str(tags.get("material") or "").split(";")[0].strip())
    if material and head not in ("Templo", "Parque", "Jardín", "Playa"):
        head += f" de {material}"
        facts += 1
    if tags.get("artist_name"):
        head += f", obra de {str(tags['artist_name']).strip()}"
        facts += 1
    denomination = _DENOMINATION.get(str(tags.get("denomination") or ""))
    if denomination and head == "Templo":
        head += f" {denomination}"
    year = _year(tags.get("start_date"))
    if year:
        head += f" ({year})"
        facts += 1
    sentences: List[str] = [head + "."]

    if tags.get("architect") and not tags.get("artist_name"):
        # Often a studio rather than a person: no "the architect X".
        sentences.append(f"Proyecto arquitectónico: {str(tags['architect']).strip()}.")
        facts += 1
    civilization = _CIVILIZATION.get(str(tags.get("historic:civilization") or ""))
    if civilization:
        sentences.append(f"De época {civilization}.")
        facts += 1
    inscription = " ".join(str(tags.get("inscription") or "").split())
    if inscription:
        if len(inscription) > _MAX_INSCRIPTION_CHARS:
            inscription = inscription[:_MAX_INSCRIPTION_CHARS].rsplit(" ", 1)[0] + "…"
        sentences.append(f"Inscripción: «{inscription}».")
        facts += 1
    if heritage:
        label = DESIGNATION_LABEL.get(heritage.get("designation", ""), heritage.get("designation", ""))
        category = str(heritage.get("category") or "").lower()
        sentences.append(
            f"Protegido como {label}"
            + (f" ({category})" if category else "")
            + " en el Inventario General del Patrimonio Cultural Valenciano"
            + (f", donde figura como «{heritage['name']}»" if heritage.get("name") else "")
            + "."
        )
        facts += 1
    return " ".join(sentences), facts
