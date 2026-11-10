import pytest
from conftest import poi_id
from fastapi.testclient import TestClient

import pmv_turismo.api as api


@pytest.fixture
def client(workspace, monkeypatch):
    monkeypatch.setattr(api, "CONFIG_PATH", workspace / "config.yaml")
    monkeypatch.setattr(api, "STATE", api._State())
    monkeypatch.setattr(api, "encode_query", lambda cfg, text: None)
    with TestClient(api.app) as c:
        yield c


def test_health_reports_catalog(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "pois": 10, "source": "file", "behaviourModel": None}


def test_pois_filters(client):
    assert len(client.get("/pois").json()) == 10
    assert {p["name"] for p in client.get("/pois?category=museum").json()} == {"m1", "m2", "m3"}
    box = client.get("/pois?min_lat=39.459&min_lon=-0.391&max_lat=39.461&max_lon=-0.389").json()
    assert [p["name"] for p in box] == ["v1"]
    assert client.get(f"/pois/{poi_id('m1')}").json()["name"] == "m1"
    assert client.get("/pois/unknown").status_code == 404


def test_onboarding_cards_exclude_answered(client):
    cards = client.post("/onboarding/cards", json={"limit": 2, "liked": [poi_id("m1")]}).json()
    assert [c["name"] for c in cards] == ["p1", "c1"]


def test_recommendations(client):
    body = client.post("/recommendations", json={
        "liked": [poi_id("p1")], "location": {"lat": 39.48, "lon": -0.37},
        "top_n": 3, "at": "2026-06-10T11:00:00+02:00",
    }).json()
    assert len(body["results"]) == 3
    assert poi_id("p1") not in [r["id"] for r in body["results"]]


@pytest.mark.parametrize("payload", [
    {"at": "2026-06-10T11:00:00"},              # naive datetime
    {"location": {"lat": 120, "lon": 0}},        # out of range
    {"liked": ["x"] * 201},                      # too many ids
    {"top_n": 0},
])
def test_recommendations_validation(client, payload):
    assert client.post("/recommendations", json=payload).status_code == 422


def test_search_without_text_model_falls_back_to_words(client):
    # encode_query is patched to None (no text model): word search, nothing matches here
    response = client.get("/search?q=gothic churches")
    assert response.status_code == 200 and response.json() == []
    assert client.get("/search?q=a").status_code == 422


def test_no_user_storage_endpoints(client):
    for path in ("/user/preferences", "/itinerary", "/run/ingestion", "/embedding/x"):
        assert client.get(path).status_code in (404, 405)


def test_broker_source_keeps_last_catalog_on_failure(client, monkeypatch):
    import pmv_turismo.data_source as ds

    assert client.get("/health").json()["pois"] == 10
    monkeypatch.setenv("POI_SOURCE", "ngsi-ld")
    monkeypatch.setattr(api, "load_pois", lambda cfg: (_ for _ in ()).throw(RuntimeError("broker down")))
    assert client.get("/health").json()["pois"] == 10
    monkeypatch.setattr(api, "load_pois", lambda cfg: ds.load_poi_list(api.processed_dir(cfg) / "poi.jsonld")[:4])
    monkeypatch.setenv("POI_REFRESH_SECONDS", "1")
    import time
    time.sleep(1.1)
    assert client.get("/health").json()["pois"] == 4


def test_server_serves_api_and_web(workspace, monkeypatch, tmp_path):
    import importlib

    web = tmp_path / "out"
    (web / "explore").mkdir(parents=True)
    (web / "index.html").write_text("<html>home</html>")
    (web / "explore" / "index.html").write_text("<html>explore</html>")
    monkeypatch.setenv("WEB_DIR", str(web))
    monkeypatch.setattr(api, "CONFIG_PATH", workspace / "config.yaml")
    monkeypatch.setattr(api, "STATE", api._State())
    import pmv_turismo.server as server

    server = importlib.reload(server)
    with TestClient(server.app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/api/health").json()["pois"] == 10
        assert "explore" in c.get("/explore/").text
        assert c.get("/").status_code == 200


def test_publish_normalizes_and_selects_sourced():
    from pmv_turismo.publish import normalized, select

    poi = {"id": "urn:x", "type": "PointOfInterest", "@context": ["c"], "name": "A", "image": None,
           "location": {"type": "Point", "coordinates": [0, 0]}, "refSeeAlso": ["https://www.wikidata.org/wiki/Q1"]}
    n = normalized(poi)
    assert n["location"]["type"] == "GeoProperty" and n["name"] == {"type": "Property", "value": "A"}
    assert "@context" not in n and "image" not in n
    assert select([poi, {"id": "urn:y"}], "sourced") == [poi]
