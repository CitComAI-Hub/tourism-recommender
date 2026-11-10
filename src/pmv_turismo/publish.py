"""Publish the catalog to an NGSI-LD broker as Smart Data Models PointOfInterest.

Batch upsert in keyValues with the SDM context sent as Link, so the entities are
the same ones the service reads back (data_source.py). Run it against the
provider's own broker (inside its cluster), not through the connector's gateway.

Subsets, to publish only what can be trusted until the catalog is audited:
  sourced  POIs linked to a checked Wikidata item or to the GVA heritage inventory
  all      the whole catalog

Settings (environment): NGSI_LD_URL, NGSILD_TENANT, POI_CONTEXT,
PUBLISH_SUBSET (sourced), PUBLISH_DRY_RUN (1 = only count, the default).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from pmv_turismo.common import load_poi_list, processed_dir
from pmv_turismo.data_source import SDM_POI_CONTEXT

logger = logging.getLogger(__name__)
BATCH = 100


def normalized(poi: Dict[str, Any]) -> Dict[str, Any]:
    """keyValues -> normalized NGSI-LD (batch operations of Orion-LD 1.6 reject keyValues)."""
    out: Dict[str, Any] = {"id": poi["id"], "type": poi.get("type", "PointOfInterest")}
    for key, value in poi.items():
        if key in ("id", "type", "@context") or value is None:
            continue
        kind = "GeoProperty" if key == "location" else "Property"
        out[key] = {"type": kind, "value": value}
    return out


def is_sourced(poi: Dict[str, Any]) -> bool:
    links = poi.get("refSeeAlso") or []
    return bool(poi.get("heritageInventoryId")) or any("wikidata.org" in str(u) for u in links)


def select(pois: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    if subset == "all":
        return pois
    if subset == "sourced":
        return [p for p in pois if is_sourced(p)]
    raise ValueError(f"Unknown subset: {subset}")


def publish_from_config(config: dict) -> None:
    """CLI step `publish`."""
    subset = os.environ.get("PUBLISH_SUBSET", "sourced")
    dry_run = os.environ.get("PUBLISH_DRY_RUN", "1") != "0"
    pois = load_poi_list(processed_dir(config) / "poi.jsonld")
    chosen = select(pois, subset)
    generated = sum(1 for p in chosen if p.get("descriptionGenerated"))
    print(f"publish: {len(chosen)} of {len(pois)} POIs in subset '{subset}' ({generated} with generated descriptions)")
    if dry_run:
        print("publish: dry run (set PUBLISH_DRY_RUN=0 to write to the broker)")
        return

    from filip.clients.ngsi_ld.cb import ContextBrokerLDClient
    from filip.models.base import FiwareLDHeader
    from filip.models.ngsi_ld.context import ActionTypeLD, ContextLDEntity

    header = FiwareLDHeader()
    if os.environ.get("NGSILD_TENANT"):
        header.ngsild_tenant = os.environ["NGSILD_TENANT"]
    header.set_context(os.environ.get("POI_CONTEXT", SDM_POI_CONTEXT))
    client = ContextBrokerLDClient(url=os.environ["NGSI_LD_URL"], fiware_header=header)
    entities = [ContextLDEntity(**normalized(p)) for p in chosen]
    for start in range(0, len(entities), BATCH):
        client.entity_batch_operation(entities=entities[start:start + BATCH],
                                      action_type=ActionTypeLD.UPSERT, options="update")
    print(f"publish: {len(entities)} POIs upserted into {os.environ['NGSI_LD_URL']}")
