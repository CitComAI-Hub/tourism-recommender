from datetime import datetime

import numpy as np
import pytest
from conftest import POIS, poi_id

from pmv_turismo.behaviour import BehaviourModel, first_visits, train
from pmv_turismo.common import LOCAL_TZ, load_poi_list, opening_hours_cover_duration
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.recommendation import Recommender

WEDNESDAY_11 = datetime(2026, 6, 10, 11, 0, tzinfo=LOCAL_TZ)
MONDAY_11 = datetime(2026, 6, 8, 11, 0, tzinfo=LOCAL_TZ)


def test_opening_hours():
    hours = {"value": "Tu-Su 10:00-19:00; Mo off"}
    assert opening_hours_cover_duration(hours, WEDNESDAY_11, 60)
    assert not opening_hours_cover_duration(hours, WEDNESDAY_11.replace(hour=18, minute=30), 60)
    assert not opening_hours_cover_duration(hours, MONDAY_11, 30)
    assert opening_hours_cover_duration("Mo-Fr 09:00-14:00,17:00-20:00", WEDNESDAY_11.replace(hour=17), 60)
    assert opening_hours_cover_duration("24/7", MONDAY_11, 600)
    assert opening_hours_cover_duration(None, MONDAY_11, 60)
    # A weekday not mentioned is closed (OSM semantics)
    assert not opening_hours_cover_duration("Tu-Su 10:00-20:00", MONDAY_11, 30)
    # Day lists, holidays and spaces around the dash
    sunday_11 = MONDAY_11.replace(day=14)
    assert opening_hours_cover_duration("Tu-Sa 10:00-14:00, 15:00-19:00; Su,PH 10:00-14:00", sunday_11, 60)
    assert opening_hours_cover_duration("Sa,Su 10:00 - 14:00", sunday_11, 60)
    # Not understood (month ranges): permissive
    assert opening_hours_cover_duration("Jan off; Mar-Dec: Tu-Sa 11:00-14:00", MONDAY_11, 30)


def _catalog(config, workspace):
    pois = load_poi_list(workspace / "processed" / "poi.jsonld")
    return pois, load_embedding_index(embeddings_path(config))


def _photo(short, user):
    lat, lon = next(p[4] for p in POIS if p[0] == short)
    return {"user": user, "lat": lat, "lon": lon}


def _model(pois, index):
    # Users who visit a museum also visit the nearby church; nobody visits v1.
    photos = [_photo(s, f"u{u}") for u in range(6) for s in ("m1", "c1")]
    photos += [_photo(s, f"w{u}") for u in range(3) for s in ("p1", "p2")]
    photos += [_photo("m2", "x0")]
    units = index.units[[index.index_of(p["id"]) for p in pois]]
    return train(pois, units, first_visits(photos, pois), "test")


def test_without_model_popularity_comes_from_wikipedia(config, workspace):
    rec = Recommender(*_catalog(config, workspace), config=config)
    cards = rec.onboarding_cards(3)
    # relevance is inverse popularity: m1 (0.1) and p1 (0.2) are the best known
    assert [c["name"] for c in cards] == ["m1", "p1", "c1"]
    assert set(rec.popularity_source) == {"wikipedia"}


def test_cards_skip_seen_and_filter_categories(config, workspace):
    rec = Recommender(*_catalog(config, workspace), config=config)
    cards = rec.onboarding_cards(2, seen=[poi_id("m1")], categories=["museum"])
    assert [c["name"] for c in cards] == ["m2", "m3"]


def test_recommend_excludes_feedback_and_prefers_nearby(config, workspace):
    rec = Recommender(*_catalog(config, workspace), config=config)
    results = rec.recommend(liked=[poi_id("p3")], disliked=[poi_id("m1")])
    names = [r["name"] for r in results]
    assert "p3" not in names and "m1" not in names
    # v1 is the farthest from p3 and among the least known: it must come last
    assert names[-1] == "v1"
    assert all("reasons" in r and r["rank"] == i + 1 for i, r in enumerate(results))


def test_recommend_uses_location_and_categories(config, workspace):
    rec = Recommender(*_catalog(config, workspace), config=config)
    near_v1 = {"lat": 39.460, "lon": -0.390}
    results = rec.recommend(location=near_v1, categories=["viewpoint", "park"])
    assert {r["name"] for r in results} <= {"v1", "p1", "p2", "p3"}
    assert results[0]["name"] == "v1"
    assert results[0]["distanceKm"] == pytest.approx(0.0, abs=1e-6)


def test_opening_hours_filter(config, workspace):
    pois, index = _catalog(config, workspace)
    for p in pois:
        if p["name"] == "m2":
            p["openingHours"] = "Tu-Su 10:00-19:00"
    config["recommendation"]["filter_opening_hours"] = True
    rec = Recommender(pois, index, config=config)
    assert "m2" not in [r["name"] for r in rec.recommend(at=MONDAY_11)]
    assert "m2" in [r["name"] for r in rec.recommend(at=WEDNESDAY_11)]


def test_behaviour_model_roundtrip_and_sources(config, workspace, tmp_path):
    pois, index = _catalog(config, workspace)
    model = _model(pois, index)
    model.save(tmp_path / "m.npz")
    loaded = BehaviourModel.load(tmp_path / "m.npz")
    assert loaded.meta["users"] == 10 and loaded.ids == model.ids
    rec = Recommender(pois, index, loaded, config)
    source = dict(zip([p["name"] for p in rec.pois], rec.popularity_source, strict=False))
    assert source["m1"] == "visits" and source["v1"] == "predicted"
    # the most visited places lead the onboarding
    assert {c["name"] for c in rec.onboarding_cards(2)} == {"m1", "c1"}


def test_adapted_similarity_follows_co_visits(config, workspace):
    pois, index = _catalog(config, workspace)
    rec = Recommender(pois, index, _model(pois, index), config)
    i, j, k = (rec.position[poi_id(s)] for s in ("m1", "c1", "p1"))
    raw = rec.units @ rec.units.T
    ad = rec.adapted @ rec.adapted.T
    # m1 and c1 are described differently but visited together
    assert ad[i, j] > raw[i, j]
    assert ad[i, j] > ad[i, k]


def test_search_ranks_by_text_similarity(config, workspace):
    rec = Recommender(*_catalog(config, workspace), config=config)
    churches = np.array([0.0, 0.0, 1.0, 0.0])
    assert [r["name"] for r in rec.search(churches, limit=3)][0] == "c1"
    assert rec.search(None) == []


def test_text_search_by_words(config, workspace):
    pois, index = _catalog(config, workspace)
    pois[0]["name"], pois[0]["description"] = "Museu de Belles Arts", "Pintura gòtica"
    pois[6]["description"] = "Església gòtica del segle XIV"
    rec = Recommender(pois, index, config=config)
    names = [r["name"] for r in rec.text_search("gotica belles")]
    assert names[0] == "Museu de Belles Arts" and "c1" in names
    assert rec.text_search("xy") == []


def test_model_of_another_catalog_is_ignored(config, workspace):
    pois, index = _catalog(config, workspace)
    model = _model(pois, index)
    model.ids = [f"urn:other:{i}" for i in range(len(model.ids))]
    rec = Recommender(pois, index, model, config)
    assert rec.model_meta is None and set(rec.popularity_source) == {"wikipedia"}


def _trip(user, day, *shorts):
    return [{**_photo(s, user), "ts": f"2010-05-{day:02d} {10 + h:02d}:00:00"} for h, s in enumerate(shorts)]


def test_transitions_follow_visit_order(config, workspace):
    pois, index = _catalog(config, workspace)
    # Five visitors go from m1 to c3 (900 m away, otherwise unvisited); one to m2.
    photos = [p for u in range(5) for p in _trip(f"u{u}", 1, "m1", "c3")]
    photos += _trip("x", 2, "m1", "m2") + [_photo("p1", "y")]  # y: no timestamp, no order
    units = index.units[[index.index_of(p["id"]) for p in pois]]
    model = train(pois, units, first_visits(photos, pois), "test")
    pos = {p["name"]: i for i, p in enumerate(pois)}
    assert model.transitions[pos["m1"], pos["c3"]] == pytest.approx(5 / 6)
    assert model.transitions[pos["c3"]].sum() == 0  # nobody went anywhere after c3

    rec = Recommender(pois, index, model, config)
    after = rec.recommend(visited=[poi_id("m1")])
    names = [r["name"] for r in after]
    assert "m1" not in names
    assert names[0] == "c3"  # not the nearest, but where m1's visitors go next
    assert {"type": "next", "poi": "m1"} in after[0]["reasons"]
    # without the visit, c3 is just a little-known place farther away than m2
    assert [r["name"] for r in rec.recommend(liked=[poi_id("m1")])][0] != "c3"

    rec.next_weight = 0.0
    assert [r["name"] for r in rec.recommend(visited=[poi_id("m1")])][0] != "c3"


def test_predicted_popularity_calibration(config, workspace):
    pois, index = _catalog(config, workspace)
    model = _model(pois, index)
    rec = Recommender(pois, index, model, config)
    unvisited = np.array([s == "predicted" for s in rec.popularity_source])
    raw = rec.popularity[unvisited].copy()
    assert rec.predicted_calibration == 0.0
    rec.set_predicted_calibration(1.0)
    mapped = rec.popularity[unvisited]
    # never reverses the order; spread over the real counts (zeros included)
    assert np.all(np.diff(mapped[np.argsort(raw, kind="stable")]) >= 0)
    real = np.log1p(model.visitors)
    assert mapped.min() >= real.min() and mapped.max() <= real.max()
    # visited POIs keep their real counts
    assert np.array_equal(rec.popularity[~unvisited], np.log1p(model.visitors[~unvisited]))
    rec.set_predicted_calibration(0.0)
    assert np.allclose(rec.popularity[unvisited], raw)


def test_stated_interests_reorder_cards_and_recommendations(config, workspace):
    rec = Recommender(*_catalog(config, workspace), config=config)
    churches = np.array([0.0, 0.0, 1.0, 0.0])
    interest = rec.interest_scores("iglesias", churches)
    assert interest is not None and interest.min() == 0.0 and interest.max() == 1.0
    # m1 is the best known place, but a stated interest in churches puts c1 first
    assert rec.onboarding_cards(1)[0]["name"] == "m1"
    assert rec.onboarding_cards(1, interest=interest)[0]["name"] == "c1"
    top = rec.recommend(liked=[poi_id("m2")], interest=interest, top_n=3)
    assert top[0]["name"] == "c1" and top[0]["reasons"][0] == {"type": "interest"}
    assert rec.interest_scores("  ") is None


def test_stated_interests_without_text_model_use_words(config, workspace):
    pois, index = _catalog(config, workspace)
    next(p for p in pois if p["id"] == poi_id("v1"))["description"] = "Mirador con vistas"
    rec = Recommender(pois, index, config=config)
    interest = rec.interest_scores("miradores vistas", None)
    assert interest is not None and rec.ids[int(np.argmax(interest))] == poi_id("v1")
    assert rec.interest_scores("zzzz", None) is None


def test_model_without_recorded_features_uses_embedding_and_category(config, workspace):
    pois, index = _catalog(config, workspace)
    units = index.units[[index.index_of(p["id"]) for p in pois]]
    photos = [_photo(s, f"u{u}") for u in range(6) for s in ("m1", "c1")]
    model = train(pois, units, first_visits(photos, pois), "test", features=("embedding", "category"))
    del model.meta["features"]
    assert model.predicted_popularity(pois, units).shape == (len(pois),)
