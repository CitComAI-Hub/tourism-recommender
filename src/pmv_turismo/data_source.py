"""Where the service reads the POI catalog from.

- `file` (default): `poi.jsonld` of the processed directory (built by the
  offline pipeline and shipped with the service).
- `ngsi-ld`: the `PointOfInterest` entities of an NGSI-LD broker, read with
  FiLiP. Same pattern as the MIAIS and the Container Location Optimizer: any
  broker (level 1) or a data space connector (level 2), used when
  `DS_ISSUER_URL` is set; every request then carries a data space access token
  (ds-consumer-auth).

The POI embeddings always come from the shipped file: POIs of the broker that
have none (not built by this pipeline) are left out of the recommendations.

Settings (environment):
  POI_SOURCE           file | ngsi-ld
  NGSI_LD_URL          broker or connector endpoint (ngsi-ld)
  NGSILD_TENANT        optional tenant
  POI_TYPE             entity type (default PointOfInterest)
  POI_CONTEXT          @context sent as Link (default: the SDM PointOfInterest one)
  POI_REFRESH_SECONDS  how long a broker read is reused (default 3600)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from pmv_turismo.common import load_poi_list, processed_dir

logger = logging.getLogger(__name__)

SDM_POI_CONTEXT = "https://raw.githubusercontent.com/smart-data-models/dataModel.PointOfInterest/master/context.jsonld"


def source_kind() -> str:
    return os.environ.get("POI_SOURCE", "file").strip().lower()


def refresh_seconds() -> int:
    return int(os.environ.get("POI_REFRESH_SECONDS", "3600"))


def load_pois(config: dict[str, Any]) -> list[dict[str, Any]]:
    kind = source_kind()
    if kind == "file":
        return load_poi_list(processed_dir(config) / "poi.jsonld")
    if kind == "ngsi-ld":
        return _broker_pois()
    raise ValueError(f"Unknown POI_SOURCE: {kind}")


def _broker_pois() -> list[dict[str, Any]]:
    from filip.clients.ngsi_ld.cb import ContextBrokerLDClient
    from filip.models.base import FiwareLDHeader

    url = os.environ["NGSI_LD_URL"]
    session = None
    if os.environ.get("DS_ISSUER_URL"):
        from ds_consumer_auth import data_space_session

        session = data_space_session()
    header = FiwareLDHeader()
    if os.environ.get("NGSILD_TENANT"):
        header.ngsild_tenant = os.environ["NGSILD_TENANT"]
    # Without the SDM context the broker expands the type and attributes to other URIs.
    header.set_context(os.environ.get("POI_CONTEXT", SDM_POI_CONTEXT))
    client = ContextBrokerLDClient(url=url, session=session, fiware_header=header, create_tenant=False)
    entities = client.get_entity_list(entity_type=os.environ.get("POI_TYPE", "PointOfInterest"), options="keyValues")
    pois = [e.model_dump(exclude_none=True, by_alias=True) for e in entities]
    valid = [p for p in pois if _has_point(p)]
    if len(valid) < len(pois):
        logger.warning("%d POIs without a Point location ignored", len(pois) - len(valid))
    logger.info("Read %d POIs from %s%s", len(valid), url, " through the data space" if session else "")
    return valid


def _has_point(poi: dict[str, Any]) -> bool:
    loc = poi.get("location")
    if isinstance(loc, dict) and "value" in loc:
        loc = loc["value"]
        poi["location"] = loc
    try:
        return loc["type"] == "Point" and len(loc["coordinates"]) >= 2
    except (KeyError, TypeError):
        return False
