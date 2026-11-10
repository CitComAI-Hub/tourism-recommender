from pmv_turismo.descriptions import build_description
from pmv_turismo.heritage import match_heritage, names_match, parse_inventory, utm_to_latlon
from pmv_turismo.ingestion import _extract_pois, _haversine_m, _is_falla, _wikidata_matches_poi

HEADER = (
    "igpcv;denominacion;provincia;municipio;utmeste;utmnorte;"
    "codclasificacion;clasificacion;codcategoria;categoria\n"
)


def test_utm_conversion_lands_on_the_monument():
    lat, lon = utm_to_latlon(725018, 4372845)  # Torres de Quart, ETRS89 UTM 30N
    assert _haversine_m(lat, lon, 39.4758, -0.3840) < 50


def test_inventory_reads_swapped_columns_and_skips_movable_items():
    text = HEADER + (
        '"1";"Torres de Quart";"VALENCIA";"VALÈNCIA";"4372845";"725018";1;"x";4;"Monumento"\n'
        '"2";"Finca Roja";"VALENCIA";"VALÈNCIA";"724987";"4371474";5;"x";11;"Monumento de interés local"\n'
        '"3";"Campana la Nova";"VALENCIA";"VALÈNCIA";"725000";"4372000";1;"x";9;"Individual (mueble)"\n'
        '"4";"Castillo";"ALICANTE";"ALCOSSER";"725000";"4297326";1;"x";4;"Monumento"\n'
    )
    entries = parse_inventory(text, "BIC", "VALÈNCIA")
    assert [e["name"] for e in entries] == ["Torres de Quart", "Finca Roja"]
    assert all(39.4 < e["lat"] < 39.5 and -0.4 < e["lon"] < -0.3 for e in entries)


def test_names_match_across_languages_but_not_neighbours():
    assert names_match("Torres dels Serrans", "Torres de Serranos")
    assert names_match("Catedral de València", "Santa Iglesia Catedral Basílica Metropolitana de Santa María")
    assert names_match("Església de Sant Llorenç", "Iglesia de San Lorenzo")
    assert names_match("Convento de Santa Clara", "Monasterio de Santa Clara")
    # Same words around a different kind of place
    assert not names_match("Jardí de la Generalitat", "Palau de la Generalitat")
    # A tiled altarpiece on a church façade is not the church
    assert not names_match(
        "Santuari de la Mare de Déu de Montolivet", "Retaule Ceràmic de la Mare de Déu de Monteolivet"
    )
    assert not names_match("Plaça de l'Ermita Molí de Vera", "Ermita de Vera")
    assert names_match("Plaça de Bous de València", "Plaza de Toros")


def test_heritage_match_prefers_bic_and_needs_the_name():
    poi = {"id": "way:1", "name": "Torres de Quart", "lat": 39.4758, "lon": -0.3840}
    square = {"id": "node:2", "name": "Plaça de Santa Úrsula", "lat": 39.4758, "lon": -0.3840}
    entries = [
        {"id": "b", "name": "Torres de Quart", "category": "Monumento de interés local",
         "designation": "BRL", "lat": 39.4758, "lon": -0.3840},
        {"id": "a", "name": "Torres de Quart", "category": "Monumento",
         "designation": "BIC", "lat": 39.4760, "lon": -0.3841},
    ]
    rows = match_heritage([poi, square], entries, _haversine_m)
    assert len(rows) == 1 and poi["heritage"]["designation"] == "BIC"
    assert "heritage" not in square


def test_fallas_are_not_year_round_pois():
    falla, statue = _extract_pois([
        {"type": "node", "id": 1, "lat": 39.47, "lon": -0.37,
         "tags": {"name": "Falla Na Jordana", "tourism": "artwork", "artwork_type": "falla"}},
        {"type": "node", "id": 2, "lat": 39.47, "lon": -0.37,
         "tags": {"name": "El Palleter", "tourism": "artwork", "artwork_type": "statue"}},
    ])
    assert _is_falla(falla) and not _is_falla(statue)


def test_wikidata_link_must_be_located_at_the_poi():
    poi = {"lat": 39.4700, "lon": -0.3760}
    here = {"claims": {"P625": [{"mainsnak": {"datavalue": {"value": {"latitude": 39.4705, "longitude": -0.3762}}}}]}}
    madrid = {"claims": {"P625": [{"mainsnak": {"datavalue": {"value": {"latitude": 40.4168, "longitude": -3.7038}}}}]}}
    noble_title = {"claims": {"P31": []}}  # not a place: no coordinates
    assert _wikidata_matches_poi(poi, here)
    assert not _wikidata_matches_poi(poi, madrid)
    assert not _wikidata_matches_poi(poi, noble_title)


def test_generated_description_states_only_known_facts():
    text, facts = build_description(
        {"artwork_type": "sculpture", "material": "bronze", "artist_name": "Manolo Valdés", "start_date": "2005-03"},
        "artwork",
    )
    assert text == "Escultura de bronce, obra de Manolo Valdés (2005)."
    assert facts == 3

    # Unknown values are left out instead of leaking English tag values
    text, facts = build_description({"artwork_type": "sculpture", "material": "unobtainium"}, "artwork")
    assert text == "Escultura." and facts == 0

    heritage = {"designation": "BRL", "category": "Monumento de interés local", "name": "Iglesia de San Lorenzo"}
    text, facts = build_description({"denomination": "catholic"}, "place_of_worship", heritage)
    assert text.startswith("Templo católico. Protegido como Bien de Relevancia Local (monumento de interés local)")
    assert "«Iglesia de San Lorenzo»" in text and facts == 1


def _wd(lat=None, lon=None, classes=(), **labels):
    claims = {"P31": [{"mainsnak": {"datavalue": {"value": {"id": c}}}} for c in classes]}
    if lat is not None:
        claims["P625"] = [{"mainsnak": {"datavalue": {"value": {"latitude": lat, "longitude": lon}}}}]
    return {"claims": claims, "labels": {k: {"value": v} for k, v in labels.items()}}


def test_wikidata_areas_and_items_without_coordinates():
    beach = {"name": "Platja de la Devesa", "category": "beach", "lat": 39.33, "lon": -0.30}
    assert _wikidata_matches_poi(beach, _wd(39.35, -0.30))  # ~2 km: centroid of a long beach
    church = {"name": "Església de la Santíssima Creu", "category": "place_of_worship", "lat": 39.47, "lon": -0.38}
    assert _wikidata_matches_poi(church, _wd(classes=["Q16970"], es="Iglesia de la Santísima Cruz (Valencia)"))
    museum = {"name": "Casa Museu Benlluire", "category": "museum", "lat": 39.47, "lon": -0.38}
    assert not _wikidata_matches_poi(museum, _wd(classes=["Q5"], es="Mariano Benlliure"))
    statue = {"name": "Marqués del Túria", "category": "artwork", "lat": 39.47, "lon": -0.37}
    assert not _wikidata_matches_poi(statue, _wd(es="Marquesado del Turia"))
