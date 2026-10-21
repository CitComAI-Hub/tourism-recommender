from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def build_catalog(config: dict) -> None:
    """Load processed POI JSON-LD and write a simple catalog summary."""
    data_cfg = config.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    processed_path = processed_dir / "poi.jsonld"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed POI file not found: {processed_path}")

    data = json.loads(processed_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("poi.jsonld must be a list of entities")

    categories = Counter()
    sources = Counter()
    for item in data:
        for cat in item.get("category", []) or []:
            categories[cat] += 1
        source = item.get("source")
        if source:
            sources[source] += 1

    lines = [f"pois={len(data)}"]
    if sources:
        lines.append("sources:")
        for key, count in sources.most_common():
            lines.append(f"- {key}: {count}")
    if categories:
        lines.append("top_categories:")
        for key, count in categories.most_common(20):
            lines.append(f"- {key}: {count}")

    summary_path = processed_dir / "catalog_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
