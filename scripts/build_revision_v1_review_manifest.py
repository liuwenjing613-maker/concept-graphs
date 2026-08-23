from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.benchmark.experiment_v1 import selected_metric_paths


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _review_row(root: Path, metrics_path: Path) -> dict[str, Any]:
    case_root = metrics_path.parent
    metrics = _read(metrics_path)
    case_path = case_root / "case.json"
    case = _read(case_path) if case_path.exists() else {}
    names = (
        "case.json",
        "incident.json",
        "constraint.json",
        "dependency.json",
        "pre_anchor_snapshot.json",
        "corruption_trace.json",
        "replay_decision_trace.json",
        "relation_rebuild.json",
        "runtime_verification.json",
        "benchmark_metrics.json",
    )
    return {
        "case_uid": metrics.get("case_uid", case_root.name),
        "failure_type": metrics.get("failure_type"),
        "outcome": "SUCCESS" if metrics.get("pass") else "FAILED_OR_DEFERRED",
        "failure_taxonomy": metrics.get("failure_taxonomy") or [],
        "anchor_event_uid": case.get("anchor_association_event_uid"),
        "anchor_observation_uid": case.get("obs_uid"),
        "artifact_refs": [
            str((case_root / name).relative_to(root))
            for name in names
            if (case_root / name).exists()
        ],
        "review_checks": (
            "anchor_and_constraint",
            "before_after_membership",
            "corruption_and_replay_decisions",
            "relation_diff",
            "runtime_invariants",
        ),
        "review_status": "PENDING_MANUAL_REVIEW",
    }


def build(root: Path, *, seed: int = 20260823, count: int = 5) -> dict[str, Any]:
    paths, selection_integrity = selected_metric_paths(root)
    success = [path for path in paths if bool(_read(path).get("pass"))]
    failed = [path for path in paths if not bool(_read(path).get("pass"))]
    rng = random.Random(seed)
    rng.shuffle(success)
    rng.shuffle(failed)
    selected = success[:count] + failed[:count]
    return {
        "schema_version": "1.0.0",
        "seed": seed,
        "requested_per_outcome": count,
        "available_success": len(success),
        "available_failed_or_deferred": len(failed),
        "selected_success": min(count, len(success)),
        "selected_failed_or_deferred": min(count, len(failed)),
        "selection_uses_outcome_only_for_review_not_evaluation": True,
        "selection_integrity": selection_integrity,
        "cases": [_review_row(root, path) for path in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze V1 manual-review samples")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    result = build(root, seed=args.seed, count=args.count)
    destination = Path(args.output).resolve() if args.output else root / "random_review_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
