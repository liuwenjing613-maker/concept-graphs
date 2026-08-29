#!/usr/bin/env python3
"""Audit actual returned models and recorded usage for relation prediction caches."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cells = []
    grand_models = Counter()
    grand_usage = Counter()
    for scene in ("room0", "office0"):
        for variant in ("b0", "o1", "o2", "o3"):
            paths = sorted((args.pilot_root / scene / "relations" / variant / scene / "predictions").glob("*.json"))
            models = Counter()
            usage = Counter()
            for path in paths:
                record = json.loads(path.read_text(encoding="utf-8"))
                model = str(record.get("model_returned") or record.get("model_requested") or "unknown")
                models[model] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = (record.get("usage") or {}).get(key)
                    if isinstance(value, (int, float)):
                        usage[key] += value
            grand_models.update(models)
            grand_usage.update(usage)
            cells.append(
                {
                    "scene_id": scene,
                    "variant": variant.upper(),
                    "prediction_files": len(paths),
                    "model_returned_counts": dict(models),
                    "usage": dict(usage),
                }
            )
    output = {
        "schema_version": "1.0.0",
        "protocol": "read model_returned and usage from every persisted relation prediction",
        "cells": cells,
        "totals": {
            "prediction_files": sum(cell["prediction_files"] for cell in cells),
            "model_returned_counts": dict(grand_models),
            "usage": dict(grand_usage),
        },
        "interpretation": "Relation metrics are a mixed-model cache and cannot be described as a pure Terra experiment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output["totals"], indent=2))


if __name__ == "__main__":
    main()
