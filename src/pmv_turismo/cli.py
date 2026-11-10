"""Offline pipeline: build the catalog, its embeddings and the behaviour model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pmv_turismo.behaviour import train_from_config
from pmv_turismo.catalog import build_catalog
from pmv_turismo.config import Config
from pmv_turismo.embeddings import generate_embeddings
from pmv_turismo.geometry import save_geometries
from pmv_turismo.ingestion import run_ingestion
from pmv_turismo.publish import publish_from_config

STEPS = {
    "ingestion": run_ingestion,
    "catalog": build_catalog,
    "embeddings": generate_embeddings,
    "geometry": save_geometries,
    "behaviour": train_from_config,
}
# Not part of "all": writing to a broker is always explicit.
EXPLICIT = {"publish": publish_from_config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tourism recommender: offline pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--step", choices=[*STEPS, *EXPLICIT, "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config.load(args.config).data
    if args.step in EXPLICIT:
        EXPLICIT[args.step](config)
        return
    for name, run in STEPS.items():
        if args.step in (name, "all"):
            run(config)


if __name__ == "__main__":
    main()
