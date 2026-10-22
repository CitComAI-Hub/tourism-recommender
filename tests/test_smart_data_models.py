"""The catalog must validate against the official Smart Data Models schema."""
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from pmv_turismo.ingestion import _build_entity, _extract_pois, _web_url

SCHEMAS = Path(__file__).parent / "schemas"
CATALOG = Path("data/processed/poi.jsonld")
COMMON_URL = "https://smart-data-models.github.io/data-models/common-schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    common = json.loads((SCHEMAS / "common-schema.json").read_text(encoding="utf-8"))
    schema = json.loads((SCHEMAS / "PointOfInterest.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource(COMMON_URL, Resource.from_contents(common, default_specification=DRAFT7))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def _errors(validator, entity):
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(entity)]


def test_built_entity_is_valid(validator):
    [poi] = _extract_pois([{
        "type": "way", "id": 9281502, "center": {"lat": 39.4792, "lon": -0.3758},
        "tags": {
            "name": "Torres dels Serrans", "historic": "city_gate", "wikidata": "Q2455550",
            "wikipedia": "es:Torres de Serranos", "website": "www.example.org",
            "addr:street": "Plaça dels Furs", "addr:housenumber": "1", "addr:postcode": "46003",
            "opening_hours": "Tu-Su 10:00-19:00; Mo off",
        },
    }])
    poi["description"] = "Puerta de la antigua muralla."
    entity = _build_entity(poi, "https://commons.wikimedia.org/wiki/Special:FilePath/Torres.jpg")
    assert _errors(validator, entity) == []
    assert entity["source"] == "https://www.openstreetmap.org/way/9281502"
    assert entity["address"] == {
        "addressLocality": "València", "addressCountry": "ES",
        "streetAddress": "Plaça dels Furs, 1", "postalCode": "46003",
    }
    assert entity["refSeeAlso"] == [
        "https://www.example.org",
        "https://www.wikidata.org/wiki/Q2455550",
        "https://es.wikipedia.org/wiki/Torres_de_Serranos",
    ]
    assert entity["openingHours"] == "Tu-Su 10:00-19:00; Mo off" and not entity["openingHoursEstimated"]


def test_website_urls_are_ascii_uris():
    assert _web_url("http://www.plaçadebous.es/") == "http://www.xn--plaadebous-q6a.es/"
    assert _web_url("www.example.org/a b") is None
    assert _web_url("https://example.org/ruta ñ") is None
    assert _web_url("https://example.org/ruta-ñ") == "https://example.org/ruta-%C3%B1"


@pytest.mark.skipif(not CATALOG.exists(), reason="catalog not generated")
def test_catalog_is_valid(validator):
    pois = json.loads(CATALOG.read_text(encoding="utf-8"))
    invalid = {p.get("name"): errs for p in pois if (errs := _errors(validator, p))}
    assert not invalid, f"{len(invalid)} invalid POIs, e.g. {next(iter(invalid.items()))}"
