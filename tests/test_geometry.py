import numpy as np

from pmv_turismo.behaviour import assign_visits
from pmv_turismo.geometry import assign_photos, osm_ref


def _poi(pid, lon, lat):
    return {"id": f"urn:ngsi-ld:PointOfInterest:{pid}", "location": {"coordinates": [lon, lat]}}


# A park (square ~220 m wide), a statue inside it, a bridge outside it and a lone node.
PARK = [[-0.380, 39.470], [-0.3775, 39.470], [-0.3775, 39.472], [-0.380, 39.472], [-0.380, 39.470]]
POIS = [
    _poi("way:1", -0.37875, 39.471),
    _poi("node:2", -0.3790, 39.4705),
    _poi("way:3", -0.370, 39.471),
    _poi("node:4", -0.360, 39.471),
]
SHAPES = {
    "urn:ngsi-ld:PointOfInterest:way:1": {"kind": "polygon", "rings": [PARK]},
    "urn:ngsi-ld:PointOfInterest:way:3": {"kind": "line", "rings": [[[-0.371, 39.471], [-0.369, 39.471]]]},
}


def test_osm_ref():
    assert osm_ref("urn:ngsi-ld:PointOfInterest:way:42") == ("way", 42)
    assert osm_ref("urn:ngsi-ld:PointOfInterest:foo") is None


def test_photos_match_real_shapes():
    lat = np.array([39.4711, 39.4705, 39.4712, 39.4711, 39.4740, 39.47105])
    lon = np.array([-0.3781, -0.3790, -0.3699, -0.3600, -0.3600, -0.37745])
    hits = assign_photos(POIS, SHAPES, lat, lon)
    assert set(hits[0]) == {0, 1, 5}   # inside the park, incl. the statue and 5 m outside its edge
    assert set(hits[1]) == {1}         # only near the statue
    assert set(hits[2]) == {2}         # along the bridge line
    assert set(hits[3]) == {3}         # the lone node; photo 4 is 300 m away


def test_visit_counts_every_matching_poi():
    photos = [{"user": "a", "lat": 39.4705, "lon": -0.3790}, {"user": "b", "lat": 39.4740, "lon": -0.3600}]
    assert assign_visits(photos, POIS, SHAPES) == {"a": {0, 1}}
