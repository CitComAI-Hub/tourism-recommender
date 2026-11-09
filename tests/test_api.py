import pytest
from fastapi.testclient import TestClient

import pmv_turismo.api as api
from conftest import poi_id


@pytest.fixture
def client(workspace, monkeypatch):
    monkeypatch.setattr(api, "CONFIG_PATH", workspace / "config.yaml")
    monkeypatch.setattr(api, "STATE", api._State())
    monkeypatch.setattr(api, "encode_query", lambda cfg, text: None)
    with TestClient(api.app) as c:
        yield c


def test_health_reports_catalog(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "pois": 10, "behaviourModel": None}


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


def test_search_without_text_model(client):
    assert client.get("/search?q=gothic churches").status_code == 503


def test_no_user_storage_endpoints(client):
    for path in ("/user/preferences", "/itinerary", "/run/ingestion", "/embedding/x"):
        assert client.get(path).status_code in (404, 405)
