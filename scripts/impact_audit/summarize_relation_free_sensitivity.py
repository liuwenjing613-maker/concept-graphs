#!/usr/bin/env python3
"""Recompute Oracle recovery ratios without the mixed-model relation component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = ("B0", "O1", "O2", "O3")
METRICS = ("instance_ap_mean_25_50", "node_f1", "semantic_accuracy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-summary", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scenes = {}
    for item in args.scene_summary:
        scene, path = item.split("=", 1)
        source = json.loads(Path(path).read_text(encoding="utf-8"))
        q = {}
        for condition in CONDITIONS:
            components = source["conditions"][condition]["components"]
            q[condition] = sum(float(components[metric]) for metric in METRICS) / len(METRICS)
        numerator = q["O2"] - q["B0"]
        denominator = q["O3"] - q["B0"]
        scenes[scene] = {
            "q_relation_free": q,
            "numerator": numerator,
            "denominator": denominator,
            "rho_relation_free": numerator / denominator,
        }

    macro_q = {
        condition: sum(scene["q_relation_free"][condition] for scene in scenes.values()) / len(scenes)
        for condition in CONDITIONS
    }
    macro_numerator = macro_q["O2"] - macro_q["B0"]
    macro_denominator = macro_q["O3"] - macro_q["B0"]
    output = {
        "schema_version": "1.0.0",
        "protocol": {
            "excluded_component": "relation_recall_at_1",
            "reason": "relation predictions mix gpt-5.6-sol and gpt-5.6-terra cached outputs",
            "included_components": list(METRICS),
            "aggregation": "equal-weight arithmetic mean, then equal scene macro",
        },
        "per_scene": scenes,
        "macro": {
            "q_relation_free": macro_q,
            "numerator": macro_numerator,
            "denominator": macro_denominator,
            "rho_relation_free": macro_numerator / macro_denominator,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
