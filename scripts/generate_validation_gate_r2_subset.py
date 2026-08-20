#!/usr/bin/env python3
"""Select the independent-review subset for Audit Validity Gate v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_STAGES = ("detection", "association", "fusion", "object_identity")
COHORTS = ("calibration_random", "diagnostic_priority")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_no}")
        rows.append(value)
    return rows


def identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["scene_id"]), str(row.get("case_uid") or row["finding_uid"])


def stable_key(row: dict[str, Any], seed: int) -> tuple[str, str, str]:
    scene, uid = identity(row)
    digest = hashlib.sha256(f"{seed}:{scene}:{uid}".encode()).hexdigest()
    return digest, scene, uid


def select_subset(rows: list[dict[str, Any]], *, per_cohort: int = 16, seed: int = 20260820) -> list[dict[str, Any]]:
    if per_cohort < len(REQUIRED_STAGES):
        raise ValueError(f"per_cohort must be at least {len(REQUIRED_STAGES)}")
    selected: list[dict[str, Any]] = []
    for cohort in COHORTS:
        pool = sorted((row for row in rows if row.get("cohort") == cohort), key=lambda row: stable_key(row, seed))
        if len(pool) < per_cohort:
            raise ValueError(f"not enough {cohort} cases: need {per_cohort}, found {len(pool)}")
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[tuple[str, str]] = set()

        def add_first(candidates: list[dict[str, Any]]) -> None:
            for candidate in candidates:
                key = identity(candidate)
                if key not in chosen_ids:
                    chosen.append(candidate)
                    chosen_ids.add(key)
                    return

        for stage in REQUIRED_STAGES:
            stage_rows = [row for row in pool if row.get("stage") == stage]
            if not stage_rows:
                raise ValueError(f"{cohort} has no case for required stage {stage}")
            add_first(stage_rows)
        for scene in sorted({str(row["scene_id"]) for row in pool}):
            add_first([row for row in pool if str(row["scene_id"]) == scene])
        for row in pool:
            if len(chosen) >= per_cohort:
                break
            add_first([row])
        selected.extend(chosen)

    result: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, 1):
        copied = dict(row)
        copied["r2_subset_rank"] = rank
        copied["r2_selection_seed"] = seed
        copied["r2_selection_method"] = "sha256_stratified_required_stage_and_scene"
        copied["reviewer_id"] = None
        for field in (
            "evidence_sufficient",
            "finding_correct",
            "root_stage_correct",
            "physical_interpretation",
            "downstream_harm",
            "harm_confidence",
            "repair_action",
            "repair_locality",
            "repair_confidence",
            "alternative_explanation",
            "review_seconds",
        ):
            copied[field] = None
        copied["notes"] = ""
        result.append(copied)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worklist", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-cohort", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    selected = select_subset(read_jsonl(args.worklist), per_cohort=args.per_cohort, seed=args.seed)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "selected": len(selected)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
