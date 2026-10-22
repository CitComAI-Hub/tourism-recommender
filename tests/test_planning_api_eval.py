from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from pmv_turismo import planning
from pmv_turismo.common import LOCAL_TZ
from pmv_turismo.evaluation import run_evaluation, synthetic_profiles

from conftest import poi_id

START = datetime(2026, 6, 10, 10, 0, tzinfo=LOCAL_TZ)


def _fake_table(endpoint, profile, locations):
    # 1 km between consecutive indices, 12 minutes per km.
    n = len(locations)
    dist = [[abs(i - j) * 1000.0 for j in range(n)] for i in range(n)]
    dur = [[abs(i - j) * 720.0 for j in range(n)] for i in range(n)]
    return {"distances": dist, "durations": dur}


@pytest.fixture(autouse=True)
def offline_routing(monkeypatch):
    monkeypatch.setattr(planning, "_osrm_table", _fake_table)
    monkeypatch.setattr(planning, "_osrm_route", lambda *a: {"routes": [{"geometry": "x", "legs": []}]})


def _candidate(pid, score=None, hours=None, duration=30):
    cand = {"id": pid, "name": pid, "location": {"coordinates": [-0.37, 39.47]}, "duration": duration}
    if score is not None:
        cand["score"] = score
    if hours:
        cand["openingHours"] = {"value": hours}
    return cand


def test_planner_prefers_score_over_small_detours(config):
    cands = [_candidate("near", score=0.1), _candidate("far", score=0.9)]
    out = planning.plan_itinerary(config, cands, start_at=START)
    assert [s["id"] for s in out["stops"]] == ["far", "near"]


def test_planner_equal_scores_is_nearest_neighbour(config):
    out = planning.plan_itinerary(config, [_candidate("a"), _candidate("b"), _candidate("c")], start_at=START)
    assert [s["id"] for s in out["stops"]] == ["a", "b", "c"]


def test_planner_skips_closed_pois_and_respects_budget(config):
    config["parameters"]["planning"]["max_total_minutes"] = 60
    cands = [_candidate("closed", hours="Mo-Su 18:00-20:00"), _candidate("open"), _candidate("too_far")]
    out = planning.plan_itinerary(config, cands, start_at=START)
    ids = [s["id"] for s in out["stops"]]
    assert "closed" not in ids
    assert out["totalMinutes"] <= 60


def test_visit_time_override(config):
    out = planning.plan_itinerary(config, [_candidate("a", duration=90)], visit_time_override=20, start_at=START)
    assert out["stops"][0]["visitMinutes"] == 20
    assert out["stops"][0]["arrival_time"] == "10:12"


def test_ors_requires_key_from_env(config):
    config["parameters"]["planning"]["routing"] = {"provider": "openrouteservice"}
    with pytest.raises(ValueError, match="ORS_API_KEY"):
        planning.plan_itinerary(config, [_candidate("a")], start_at=START)


@pytest.fixture
def client(workspace, monkeypatch):
    from pmv_turismo import api

    monkeypatch.setattr(api, "CONFIG_PATH", workspace / "config.yaml")
    return TestClient(api.app)


def test_api_rejects_path_traversal(client, workspace):
    for bad in ["../../etc", "a/b", "..", "x" * 65]:
        assert client.get("/recommendations", params={"user_id": bad}).status_code == 422
        assert client.post("/user/preferences", json={"user_id": bad, "profile": {}}).status_code == 422
    assert not (workspace / "processed" / "users").exists()


def test_api_onboarding_recommend_plan_flow(client, workspace):
    user = "user_1_abc"
    onboarding = {
        "user_id": user,
        "profile": {"name": "t", "preferred_categories": ["museum"], "recommendation_style": "conservative"},
        "liked_pois": [poi_id("m1")],
        "disliked_pois": [poi_id("p1")],
        "start_location": {"lat": 39.47, "lon": -0.376},
        "planner_preferences": {},
    }
    assert client.post("/user/onboarding", json=onboarding).status_code == 200
    assert client.get("/user/preferences", params={"user_id": user}).json()["profile"]["liked_pois"] == [poi_id("m1")]

    recs = client.post("/recommendations", json={"user_id": user}).json()["results"]
    assert recs and poi_id("p1") not in [r["id"] for r in recs]
    assert (workspace / "processed" / "users" / user / "recommendations.jsonld").exists()
    assert not (workspace / "processed" / "recommendations.jsonld").exists()

    # The frontend sends lightweight candidates; the API completes them.
    cands = [{"id": r["id"], "location": r["location"], "category": r["category"]} for r in recs[:3]]
    itinerary = client.post(
        "/itinerary", json={"user_id": user, "candidates": cands, "start_at": START.isoformat()}
    ).json()
    assert len(itinerary["stops"]) == 3
    assert all(s["poi"].get("score") is not None for s in itinerary["stops"])


def test_api_get_does_not_create_user_dirs(client, workspace):
    assert client.get("/itinerary", params={"user_id": "nobody"}).status_code == 404
    assert not (workspace / "processed" / "users").exists()


def test_evaluation_report(config, workspace):
    config["evaluation"] = {"k": 3, "seed": 1, "synthetic_users": 4}
    report = run_evaluation(config)
    assert report["profiles"] == 4
    metrics = report["metrics"]
    assert {"style:explorer", "semantic_mean", "popularity", "random"} <= set(metrics)
    assert metrics["semantic_mean"]["hit@3"] >= metrics["random"]["hit@3"]
    assert set(report["onboarding"]) == {"active_learning", "random"}
    assert (workspace / "processed" / "evaluation_report.txt").exists()


def test_synthetic_profiles_are_deterministic():
    pois = [{"id": f"{c}{i}", "category": [c]} for c in "abc" for i in range(5)]
    assert synthetic_profiles(pois, 3, seed=7) == synthetic_profiles(pois, 3, seed=7)
