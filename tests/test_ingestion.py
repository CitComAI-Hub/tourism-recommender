import pytest

from pmv_turismo import ingestion
from pmv_turismo.ingestion import (
    _assign_relevance,
    _deduplicate_pois,
    _extract_pois,
    _is_low_value_information,
    _osm_address,
    _osm_image,
    _shorten_to_sentence,
    _synthetic_opening_hours,
    canonical_category,
    estimate_visit_minutes,
)


def _element(el_type, el_id, lat, lon, **tags):
    return {"type": el_type, "id": el_id, "lat": lat, "lon": lon, "tags": tags}


def test_categories_are_canonical_and_keep_osm_detail():
    pois = _extract_pois([
        _element("node", 1, 39.47, -0.37, name="Sirena", tourism="yes", historic="yes"),
        _element("way", 2, 39.47, -0.37, name="Pont", man_made="bridge"),
        _element("node", 3, 39.47, -0.37, name="Galeria", tourism="gallery"),
    ])
    assert [p["category"] for p in pois] == ["historic", "bridge", "gallery"]
    assert [canonical_category(p["category"]) for p in pois] == ["historic", "historic", "museum"]
    assert estimate_visit_minutes("bridge") == 10
    assert estimate_visit_minutes("castle") == 45
    assert estimate_visit_minutes("unknown_thing") == 20  # canonical "historic"
    assert _synthetic_opening_hours("bridge") == "24/7"
    assert _synthetic_opening_hours("gallery").startswith("Tu-Su")


def test_dedup_keeps_the_monument_not_the_information_board():
    board = _element("node", 1, 39.4790, -0.3757, name="Torres de Serrans", tourism="information", information="board")
    gate = _element("way", 2, 39.4792, -0.3758, name="Torres dels Serrans", historic="city_gate", wikidata="Q2455550")
    pois, matches = _deduplicate_pois(_extract_pois([board, gate]), 80, 0.82, ["osm"])
    assert len(pois) == 1 and len(matches) == 1
    assert pois[0]["id"] == "way:2" and pois[0]["category"] == "city_gate"


def test_information_boards_are_low_value_but_offices_are_kept():
    board, office = _extract_pois([
        _element("node", 1, 39.4, -0.3, name="Panel", tourism="information", information="board"),
        _element("node", 2, 39.4, -0.3, name="Tourist Info", tourism="information", information="office"),
    ])
    assert _is_low_value_information(board)
    assert not _is_low_value_information(office)


def test_osm_images_only_from_wikimedia():
    assert _osm_image({"wikimedia_commons": "File:Pont de Fusta.jpg"}).startswith(
        "https://commons.wikimedia.org/wiki/Special:FilePath/Pont%20de%20Fusta.jpg"
    )
    assert _osm_image({"image": "https://upload.wikimedia.org/a.jpg"}) == "https://upload.wikimedia.org/a.jpg"
    assert _osm_image({"image": "https://encrypted-tbn0.gstatic.com/images?q=x"}) is None
    assert _osm_image({"wikimedia_commons": "Category:Valencia"}) is None


def test_article_language_priority():
    ent = {"sitelinks": {"cawiki": {"title": "Llotja"}, "enwiki": {"title": "Lonja"}}}
    assert ingestion._choose_article({}, ent) == ("en", "Lonja")
    ent["sitelinks"]["eswiki"] = {"title": "Lonja de la Seda"}
    assert ingestion._choose_article({"wikipedia": "ca:Llotja"}, ent) == ("es", "Lonja de la Seda")
    assert ingestion._choose_article({"wikipedia": "ca:Llotja"}, None) == ("ca", "Llotja")


def test_description_language_follows_the_tag():
    tags = {"description:ca": "Text", "tourism": "artwork"}
    (poi,) = _extract_pois([_element("node", 1, 39.47, -0.37, name="A", **tags)])
    assert poi["description_language"] == "ca"
    (poi,) = _extract_pois([_element("node", 1, 39.47, -0.37, name="A", description="x",
                                     **{"description:es": "Texto", "tourism": "artwork"})])
    assert (poi["description"], poi["description_language"]) == ("Texto", "es")


def test_artworks_in_an_area_become_one_collection(monkeypatch):
    square = [[-0.35, 39.47], [-0.34, 39.47], [-0.34, 39.48], [-0.35, 39.48], [-0.35, 39.47]]
    import pmv_turismo.geometry as geometry
    monkeypatch.setattr(geometry, "fetch_geometries", lambda *a, **k: {"x": {"kind": "polygon", "rings": [square]}})
    inside = [{"id": f"node:{i}", "name": f"S{i}", "category": "artwork", "lat": 39.475, "lon": -0.345,
               "tags": {"artist_name": f"Artist {i % 3}"}} for i in range(6)]
    outside = {"id": "node:99", "name": "Out", "category": "artwork", "lat": 39.40, "lon": -0.30, "tags": {}}
    church = {"id": "way:5", "name": "Ermita", "category": "place_of_worship", "lat": 39.475, "lon": -0.345, "tags": {}}
    pois, removed = ingestion._aggregate_artworks(
        inside + [outside, church],
        [{"area_relation": 7, "wikidata": "Q1", "name": "Campus", "area_name": "campus X"}], [], 10)
    assert removed == 6
    assert {p["name"] for p in pois} == {"Out", "Ermita", "Campus"}
    col = next(p for p in pois if p["name"] == "Campus")
    assert col["id"] == "relation:7" and col["wikidata"] == "Q1" and col["description_generated"]
    assert "6 esculturas" in col["description"] and "Artist 0" in col["description"]


class _Response:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_http_retries_rate_limits(monkeypatch):
    responses = iter([_Response(429, headers={"Retry-After": "1"}), _Response(200, {"ok": True})])
    monkeypatch.setattr(ingestion._session, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)
    assert ingestion._http_get_json("https://example.org/api") == {"ok": True}


def test_http_failure_is_an_error_not_missing_data(monkeypatch):
    monkeypatch.setattr(ingestion._session, "get", lambda *a, **k: _Response(429))
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)
    with pytest.raises(ingestion.FetchError):
        ingestion._http_get_json("https://example.org/api")
    monkeypatch.setattr(ingestion._session, "get", lambda *a, **k: _Response(404))
    assert ingestion._http_get_json("https://example.org/api", not_found_ok=True) is None


def test_partial_enrichment_does_not_overwrite_catalog(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "poi.jsonld").write_text("[\"previous\"]", encoding="utf-8")
    elements = [_element("node", 1, 39.47, -0.37, name="Museu", tourism="museum", wikidata="Q1")]
    monkeypatch.setattr(ingestion, "_fetch_overpass_data", lambda *a, **k: {"elements": elements})

    def rate_limited(*a, **k):
        raise ingestion.FetchError("www.wikidata.org: HTTP 429")

    monkeypatch.setattr(ingestion, "_http_get_json", rate_limited)
    ingestion._wikidata_entity_cache.clear()
    config = {
        "data": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(processed)},
        "ingestion": {"dedup": {"enabled": False}},
    }
    with pytest.raises(RuntimeError, match="not updated"):
        ingestion.run_ingestion(config)
    assert (processed / "poi.jsonld").read_text(encoding="utf-8") == "[\"previous\"]"
    assert "fetch_error" in (processed / "ingestion_report.json").read_text(encoding="utf-8")


def test_wikipedia_extract_is_cut_at_a_sentence():
    text = "Primera frase. Segunda frase bastante más larga que no cabe."
    assert _shorten_to_sentence(text, max_chars=30) == "Primera frase."
    assert _shorten_to_sentence("Texto  completo.\n", max_chars=30) == "Texto completo."
    text = "Es un BIC desde 1931.\u200b\u200b Tiene torre ."
    assert _shorten_to_sentence(text, 200) == "Es un BIC desde 1931. Tiene torre."


def test_mediawiki_batches_follow_redirects_and_continue(monkeypatch):
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append(params)
        if "excontinue" not in params:
            return {
                "continue": {"excontinue": 1},
                "query": {
                    "normalized": [{"from": "torres_de_Serranos", "to": "Torres de Serranos"}],
                    "redirects": [{"from": "Torres de Serranos", "to": "Torres de Serrans"}],
                    "pages": [{"title": "Torres de Serrans", "extract": "Puerta de la muralla."},
                              {"title": "Lonja", "missing": True}],
                },
            }
        return {"query": {"pages": [{"title": "Museo", "extract": "Un museo."}]}}

    monkeypatch.setattr(ingestion, "_http_get_json", fake_get)
    out = ingestion._fetch_wikipedia_extracts("es", ["torres_de_Serranos", "Lonja", "Museo"])
    assert out == {"torres_de_Serranos": "Puerta de la muralla.", "Museo": "Un museo."}
    assert len(calls) == 2 and calls[0]["titles"] == "Lonja|Museo|torres_de_Serranos"


def test_article_prefers_spanish_wikipedia():
    ent = {"sitelinks": {"cawiki": {"title": "Torres dels Serrans"}, "eswiki": {"title": "Torres de Serranos"}}}
    assert ingestion._choose_article({"wikipedia": "ca:Torres dels Serrans"}, ent) == ("es", "Torres de Serranos")
    assert ingestion._choose_article({"wikipedia": "ca:Pont"}, {}) == ("ca", "Pont")
    assert ingestion._choose_article({}, {}) is None


def test_address_is_structured():
    assert _osm_address({"addr:street": "Carrer de la Pau", "addr:housenumber": "7", "addr:postcode": "46002"}) == {
        "streetAddress": "Carrer de la Pau, 7", "postalCode": "46002",
    }
    assert _osm_address({}) == {}


def test_relevance_small_categories_use_global_ranking():
    entities = [{"category": ["museum"]} for _ in range(5)] + [{"category": ["zoo"]}]
    _assign_relevance(entities, [1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    assert entities[4]["relevance"] == 0.05  # most popular museum
    assert entities[0]["relevance"] == 0.95  # least popular museum
    assert entities[5]["relevance"] == 0.05  # lone zoo, most popular overall (not 0.5)
