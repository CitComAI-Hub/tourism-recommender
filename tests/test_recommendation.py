from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pmv_turismo.common import LOCAL_TZ, load_poi_list, opening_hours_cover_duration
from pmv_turismo.embeddings import embeddings_path, load_embedding_index
from pmv_turismo.recommendation import (
    apply_style,
    build_interest_vectors,
    recommend,
    select_active_learning_pois,
    semantic_scores,
)

from conftest import poi_id

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


def test_single_dislike_does_not_collapse_semantic_scores():
    rng = np.random.default_rng(0)
    # Embeddings with a high shared baseline similarity, like real POI texts.
    base = rng.normal(size=16)
    cands = base + 0.6 * rng.normal(size=(50, 16))
    cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    scores = semantic_scores(cands, cands[:1], cands[1:2], dislike_weight=0.5)
    assert scores.min() == 0.0 and scores.max() == 1.0
    assert np.mean(scores == 0.0) < 0.05


def test_interest_vectors_keep_distinct_tastes():
    likes = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    interests = build_interest_vectors(likes, max_interests=3)
    assert interests.shape == (2, 2)


def test_recommend_is_pure_and_respects_feedback(config, workspace):
    output = recommend(config, liked_poi_ids=[poi_id("m1")], disliked_poi_ids=[poi_id("p1")])
    ids = [r["id"] for r in output["results"]]
    assert ids[0] == poi_id("m1") and output["results"][0]["is_liked"]
    assert poi_id("p1") not in ids
    assert not (workspace / "processed" / "recommendations.jsonld").exists()


def test_recommend_can_exclude_likes(config):
    output = recommend(config, liked_poi_ids=[poi_id("m1")], include_liked=False)
    assert poi_id("m1") not in [r["id"] for r in output["results"]]


def test_semantic_style_prefers_similar_pois(config):
    config["recommendation"].update({"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "theta_mmr": 1.0, "top_n": 2})
    output = recommend(config, liked_poi_ids=[poi_id("m1")], include_liked=False)
    assert {r["id"] for r in output["results"]} == {poi_id("m2"), poi_id("m3")}


def test_apply_style_does_not_mutate_config(config):
    styled = apply_style(config, "hidden_gems")
    assert styled["recommendation"]["hidden_gems_bias"] == 1.0
    assert config["recommendation"]["hidden_gems_bias"] == 0.0


def test_opening_hours_filter(config):
    pois = load_poi_list(Path(config["data"]["processed_dir"]) / "poi.jsonld")
    for poi in pois:
        if poi["id"] == poi_id("m2"):
            poi["openingHours"] = {"value": "Mo-Su 18:00-20:00"}
    config["recommendation"]["filter_opening_hours"] = True
    output = recommend(config, poi_data=pois, liked_poi_ids=[poi_id("m1")], requested_at=WEDNESDAY_11)
    ids = [r["id"] for r in output["results"]]
    assert poi_id("m2") not in ids and poi_id("m3") in ids


def test_active_learning_rounds(config):
    pois = load_poi_list(Path(config["data"]["processed_dir"]) / "poi.jsonld")
    index = load_embedding_index(embeddings_path(config))

    first = select_active_learning_pois(pois, index, k=3)
    assert len({p["category"][0] for p in first}) >= 2  # diverse cold start

    seen = [p["id"] for p in first]
    second = select_active_learning_pois(
        pois, index, k=2, liked_poi_ids=[poi_id("m1")], disliked_poi_ids=[poi_id("p1")], seen_poi_ids=seen
    )
    assert len(second) == 2
    assert not set(seen) & {p["id"] for p in second}


def test_active_learning_explores_after_dislikes(config):
    pois = load_poi_list(Path(config["data"]["processed_dir"]) / "poi.jsonld")
    index = load_embedding_index(embeddings_path(config))

    seen = [poi_id("m1"), poi_id("p1")]
    picks = select_active_learning_pois(pois, index, k=2, disliked_poi_ids=seen, seen_poi_ids=seen)
    assert {p["category"][0] for p in picks} <= {"place_of_worship", "viewpoint"}


def test_active_learning_refines_liked_region(config):
    pois = load_poi_list(Path(config["data"]["processed_dir"]) / "poi.jsonld")
    index = load_embedding_index(embeddings_path(config))

    seen = [poi_id("m1"), poi_id("p1"), poi_id("c1")]
    picks = select_active_learning_pois(
        pois, index, k=1, liked_poi_ids=[poi_id("m1")],
        disliked_poi_ids=[poi_id("p1"), poi_id("c1")], seen_poi_ids=seen,
    )
    assert picks[0]["category"][0] == "museum"


@pytest.mark.parametrize("missing", [[], ["does-not-exist"]])
def test_recommend_requires_known_likes(config, missing):
    with pytest.raises(ValueError):
        recommend(config, liked_poi_ids=missing)


def test_recommend_falls_back_to_preferred_categories(config):
    config["recommendation"].update({"theta_mmr": 1.0, "user_profile": {"preferred_categories": ["park"]}})
    output = recommend(config, liked_poi_ids=[], disliked_poi_ids=[poi_id("m1")])
    ids = [r["id"] for r in output["results"]]
    assert output["semantic_interest_count"] == 0
    assert poi_id("m1") not in ids
    assert {r["category"][0] for r in output["results"][:3]} == {"park"}


def test_preferred_categories_boost_ranking(config):
    config["recommendation"].update({"category_weight": 2.0, "theta_mmr": 1.0, "top_n": 3})
    config["recommendation"]["user_profile"] = {"preferred_categories": ["place_of_worship"]}
    output = recommend(config, liked_poi_ids=[poi_id("m1")], include_liked=False)
    assert all(r["category"][0] == "place_of_worship" for r in output["results"])
