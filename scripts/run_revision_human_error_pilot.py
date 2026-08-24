#!/usr/bin/env python3
"""Run the frozen 3-false-merge + 3-false-split human causal pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conceptgraph.revision.benchmark.human_error_pilot import run_human_error_pilot


def _mapping(values: list[str], name: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{name} must use SCENE=PATH: {value}")
        scene, raw_path = value.split("=", 1)
        scene = scene.strip()
        if not scene or scene in result:
            raise ValueError(f"invalid or duplicate {name} scene: {scene!r}")
        result[scene] = Path(raw_path).expanduser().resolve()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--scene-base-run",
        action="append",
        default=[],
        metavar="SCENE=PATH",
        help="Repeat once for each of the one or two pilot scenes.",
    )
    parser.add_argument(
        "--edge-stream",
        action="append",
        default=[],
        metavar="SCENE=PATH",
        help="Optional frozen make_edges stream; repeat per scene.",
    )
    parser.add_argument("--expert-queue", required=True, type=Path)
    parser.add_argument("--r1-labels", required=True, type=Path)
    parser.add_argument("--r2-labels", required=True, type=Path)
    args = parser.parse_args()

    try:
        scene_base_runs = _mapping(args.scene_base_run, "scene-base-run")
        edge_streams = _mapping(args.edge_stream, "edge-stream")
        aggregate = run_human_error_pilot(
            manifest_path=args.manifest.resolve(),
            scene_base_runs=scene_base_runs,
            output_root=args.output_root.resolve(),
            source_paths={
                "expert_queue": args.expert_queue.resolve(),
                "r1_labels": args.r1_labels.resolve(),
                "r2_labels": args.r2_labels.resolve(),
            },
            edge_stream_roots=edge_streams,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "INVALID_INPUT", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "status": aggregate["status"],
                "pilot_uid": aggregate["pilot_uid"],
                "human_confirmed_case_count": aggregate[
                    "human_confirmed_case_count"
                ],
                "replayable_case_count": aggregate["replayable_case_count"],
                "deferred_non_association_root_count": aggregate[
                    "deferred_non_association_root_count"
                ],
                "natural_replay_still_wrong_count": aggregate[
                    "natural_replay_still_wrong_count"
                ],
                "sparse_causal_repair_correct_count": aggregate[
                    "sparse_causal_repair_correct_count"
                ],
                "strict_contrast_pass_count": aggregate[
                    "strict_contrast_pass_count"
                ],
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if aggregate["pilot_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
