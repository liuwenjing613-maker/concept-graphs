#!/usr/bin/env python3
"""Freeze two real capability holdouts before automatic constraint generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(encoding="utf-8", newline=None) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), ""):
            digest.update(block.encode("utf-8"))
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _evidence_files(case_dir: Path) -> list[Path]:
    named = [
        case_dir / "case.json",
        case_dir / "review_evidence.json",
        case_dir / "view_selection.json",
        case_dir / "review_final_objects_detail.png",
        case_dir / "review_final_objects_relative.png",
    ]
    named.extend(sorted(case_dir.glob("review_observation_Q*.png")))
    result = []
    seen = set()
    for path in named:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        seen.add(resolved)
        result.append(resolved)
    return result


def _expected_capability(endpoint_error_type: str) -> dict[str, Any]:
    if endpoint_error_type == "SEMANTIC_IDENTITY_ERROR":
        return {
            "candidate_family": "RELABEL",
            "promotion_expected": "DEFER",
            "reason": (
                "RELABEL may be proposed, but automatic mutation is forbidden until "
                "entity binding, executable relabel replay, and independent semantic "
                "endpoint evaluation are all available"
            ),
        }
    if endpoint_error_type == "GEOMETRY_CORRUPTION":
        return {
            "candidate_family": "RESTORE_OBSERVATION_GEOMETRY",
            "promotion_expected": "DEFER",
            "reason": (
                "raw-to-processed mask loss is outside identity primitives and needs "
                "hash-bound point/mask evidence plus a geometry-specific executor"
            ),
        }
    return {
        "candidate_family": "UNSUPPORTED",
        "promotion_expected": "DEFER",
        "reason": "no promoted primitive is registered for this endpoint type",
    }


def _case_record(
    queue_row: dict[str, Any],
    label_row: dict[str, Any],
) -> dict[str, Any]:
    case_dir = Path(str(queue_row["case_dir"])).resolve()
    artifacts = [
        {
            "logical_name": path.name,
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _evidence_files(case_dir)
    ]
    incident_uid = str(queue_row["incident_uid"])
    endpoint_type = str(label_row["final_error_type"])
    scene = str(queue_row["scene_id"])
    return {
        "case_uid": (
            f"holdout_{scene}_{endpoint_type.lower()}_{incident_uid.removeprefix('incident_')[:8]}"
        ),
        "incident_uid": incident_uid,
        "scene_id": scene,
        "representative_finding_uid": str(queue_row["representative_finding_uid"]),
        "inference_inputs": {
            "representative_trigger_observation_uids": [
                str(item)
                for item in queue_row.get("representative_trigger_observation_uids", ())
            ],
            "evidence_artifacts": artifacts,
            "human_endpoint_label_excluded": True,
            "human_notes_excluded": True,
            "final_owner_uids_excluded": True,
        },
        "posthoc_gold": {
            "evidence_sufficient": label_row.get("evidence_sufficient"),
            "final_state": label_row.get("final_state"),
            "endpoint_error_type": endpoint_type,
            "notes": label_row.get("notes"),
        },
        "expected_capability_gate": _expected_capability(endpoint_type),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--r1-labels", required=True, type=Path)
    parser.add_argument("--incident", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.incident) != 2 or len(set(args.incident)) != 2:
        raise ValueError("exactly two distinct holdout incidents are required")

    queue_rows = _read_jsonl(args.queue)
    label_rows = _read_jsonl(args.r1_labels)
    queue = {str(row["incident_uid"]): row for row in queue_rows}
    labels = {str(row["incident_uid"]): row for row in label_rows}
    cases = []
    for incident_uid in args.incident:
        if incident_uid not in queue or incident_uid not in labels:
            raise KeyError(f"unknown incident: {incident_uid}")
        label = labels[incident_uid]
        if (
            label.get("evidence_sufficient") != "YES"
            or label.get("final_state") != "WRONG"
        ):
            raise ValueError(
                f"holdout is not a confirmed endpoint error: {incident_uid}"
            )
        cases.append(_case_record(queue[incident_uid], label))

    if len({case["scene_id"] for case in cases}) != 2:
        raise ValueError("holdouts must cover two development scenes")
    if len({case["posthoc_gold"]["endpoint_error_type"] for case in cases}) != 2:
        raise ValueError("holdouts must cover two distinct capability families")

    manifest = {
        "schema_version": "2.0.0",
        "holdout_uid": "revision_v2_capability_holdouts_20260824",
        "frozen_before_generator_outcomes": True,
        "selection_uses_generator_outputs": False,
        "selection_role": (
            "TWO_REAL_CAPABILITY_AND_ABSTENTION_PROBES; "
            "NOT_A_SCENE_GENERALIZATION_OR_POPULATION_ESTIMATE"
        ),
        "selection_policy": {
            "semantic_probe": (
                "one clear cross-scene semantic correction with multiple views"
            ),
            "geometry_probe": (
                "one explicit raw-mask-correct/processed-mask-damaged incident"
            ),
            "no_replacement_after_outcomes": True,
        },
        "source_artifacts": {
            "expert_queue": {
                "path": str(args.queue.resolve()),
                "sha256_utf8_canonical_lf": _text_sha256(args.queue),
            },
            "r1_labels": {
                "path": str(args.r1_labels.resolve()),
                "sha256_utf8_canonical_lf": _text_sha256(args.r1_labels),
            },
        },
        "cases": cases,
    }
    _write(args.output, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "case_uids": [case["case_uid"] for case in cases],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
