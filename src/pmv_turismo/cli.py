from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmv_turismo.common import processed_dir
from pmv_turismo.config import Config
from pmv_turismo.ingestion import run_ingestion
from pmv_turismo.catalog import build_catalog
from pmv_turismo.embeddings import generate_embeddings
from pmv_turismo.recommendation import apply_style, recommend, write_recommendations
from pmv_turismo.planning import plan_itinerary, write_itinerary
from pmv_turismo.evaluation import format_report, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PMV IA Turismo CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=[
            "ingestion",
            "catalog",
            "embeddings",
            "recommendation",
            "planning",
            "evaluation",
            "all",
        ],
        default="all",
        help="Run a single step or all steps",
    )
    return parser.parse_args()


def _run_recommendation(config: dict) -> None:
    rec_cfg = config.get("recommendation", {})
    if not rec_cfg.get("liked_pois"):
        print("recommendation: skipped (set recommendation.liked_pois in the config to run it from the CLI)")
        return
    styled = apply_style(config, rec_cfg.get("style"))
    output = recommend(styled)
    write_recommendations(output, processed_dir(config) / "recommendations.jsonld")
    print(f"recommendation: {len(output['results'])} results")


def _run_planning(config: dict) -> None:
    rec_path = processed_dir(config) / "recommendations.jsonld"
    if not rec_path.exists():
        print(f"planning: skipped ({rec_path} not found; run the recommendation step first)")
        return
    candidates = json.loads(rec_path.read_text(encoding="utf-8")).get("results", [])
    output = plan_itinerary(config, candidates=candidates)
    write_itinerary(output, processed_dir(config) / "itinerary.jsonld")
    print(f"planning: {len(output['stops'])} stops, {output['totalMinutes']} min")


def main() -> None:
    args = parse_args()
    config = Config.load(args.config).data
    modules = config.get("modules", {})
    steps = [
        ("ingestion", run_ingestion),
        ("catalog", build_catalog),
        ("embeddings", generate_embeddings),
        ("recommendation", _run_recommendation),
        ("planning", _run_planning),
        ("evaluation", lambda cfg: print(format_report(run_evaluation(cfg)))),
    ]
    for name, run in steps:
        if args.step in (name, "all") and modules.get(name, {}).get("enabled"):
            run(config)


if __name__ == "__main__":
    main()
