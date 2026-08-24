#!/usr/bin/env python3
"""Freeze two unseen holdouts without reading human answer fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


_SELECTION_FIELDS = {
    "incident_uid",
    "scene_id",
    "case_dir",
    "representative_finding_uid",
    "representative_trigger_observation_uids",
}
_SCENES = ("office0", "room0")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_queue_projection(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            projected = {key: source.get(key) for key in _SELECTION_FIELDS}
            if not projected["incident_uid"] or not projected["scene_id"]:
                raise ValueError(f"queue line {line_number} lacks selection fields")
            rows.append(projected)
    return rows


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


def _consumed_incidents(
    human6: dict[str, Any], prior_holdouts: dict[str, Any]
) -> set[str]:
    consumed = {str(case["incident_uid"]) for case in human6.get("cases") or ()}
    consumed.update(
        str(case["incident_uid"]) for case in prior_holdouts.get("cases") or ()
    )
    if len(consumed) != 8:
        raise ValueError(f"expected 8 unique consumed incidents, found {len(consumed)}")
    return consumed


def _rank(protocol_salt: str, incident_uid: str) -> str:
    return hashlib.sha256(f"{protocol_salt}|{incident_uid}".encode("utf-8")).hexdigest()


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


def _frozen_case(row: dict[str, Any], rank: str) -> dict[str, Any]:
    case_dir = Path(str(row["case_dir"])).resolve()
    artifacts = [
        {
            "logical_name": path.name,
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _evidence_files(case_dir)
    ]
    incident_uid = str(row["incident_uid"])
    return {
        "case_uid": (
            f"fresh_holdout_{row['scene_id']}_"
            f"{incident_uid.removeprefix('incident_')[:8]}"
        ),
        "incident_uid": incident_uid,
        "scene_id": str(row["scene_id"]),
        "selection_rank_sha256": rank,
        "representative_finding_uid": str(row["representative_finding_uid"]),
        "inference_inputs": {
            "representative_trigger_observation_uids": [
                str(item)
                for item in (row.get("representative_trigger_observation_uids") or ())
            ],
            "evidence_artifacts": artifacts,
        },
        "answer_access_contract": {
            "human_label_not_read_by_freezer": True,
            "human_note_not_read_by_freezer": True,
            "endpoint_error_type_not_read_by_freezer": True,
            "final_owner_not_read_by_freezer": True,
            "generator_output_not_available_at_freeze": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--r1-labels", required=True, type=Path)
    parser.add_argument("--human6-manifest", required=True, type=Path)
    parser.add_argument("--prior-holdout-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protocol-salt",
        default="ali_my_revision_kernel_v2_fresh_holdout_20260825",
    )
    args = parser.parse_args()

    human6 = _read_json(args.human6_manifest.resolve())
    prior_holdouts = _read_json(args.prior_holdout_manifest.resolve())
    consumed = _consumed_incidents(human6, prior_holdouts)
    queue = _read_queue_projection(args.queue.resolve())
    if len(queue) != 40:
        raise ValueError(f"expected 40 queue incidents, found {len(queue)}")
    if len({str(row["incident_uid"]) for row in queue}) != len(queue):
        raise ValueError("queue incident UIDs are not unique")

    eligible = [
        row
        for row in queue
        if str(row["incident_uid"]) not in consumed and str(row["scene_id"]) in _SCENES
    ]
    selected = []
    eligible_counts = {}
    for scene in _SCENES:
        scene_rows = [row for row in eligible if str(row["scene_id"]) == scene]
        if not scene_rows:
            raise ValueError(f"no unseen candidate for scene {scene}")
        eligible_counts[scene] = len(scene_rows)
        selected.append(
            min(
                scene_rows,
                key=lambda row: (
                    _rank(args.protocol_salt, str(row["incident_uid"])),
                    str(row["incident_uid"]),
                ),
            )
        )

    cases = [
        _frozen_case(row, _rank(args.protocol_salt, str(row["incident_uid"])))
        for row in selected
    ]
    if len({case["incident_uid"] for case in cases}) != 2:
        raise ValueError("fresh holdout selection is not distinct")
    if any(case["incident_uid"] in consumed for case in cases):
        raise ValueError("fresh holdout overlaps a consumed incident")

    manifest = {
        "schema_version": "1.0.0",
        "holdout_uid": "revision_kernel_v2_fresh_blind_holdouts_20260825",
        "status": "FROZEN",
        "frozen_before_posthoc_label_access": True,
        "frozen_before_generator_outcomes": True,
        "selection_uses_human_answer_fields": False,
        "selection_uses_generator_outputs": False,
        "selection_role": (
            "TWO_UNSEEN_INCIDENT_PROBES; NOT_A_POPULATION_OR_SCENE_"
            "GENERALIZATION_ESTIMATE"
        ),
        "selection_policy": {
            "excluded_consumed_incident_count": len(consumed),
            "eligible_unseen_incident_count": len(eligible),
            "eligible_count_by_scene": eligible_counts,
            "scene_strata": list(_SCENES),
            "rank": "minimum SHA256(protocol_salt|incident_uid) per scene",
            "protocol_salt": args.protocol_salt,
            "no_outcome_based_replacement": True,
            "allowed_queue_fields": sorted(_SELECTION_FIELDS),
        },
        "consumed_incident_uids": sorted(consumed),
        "source_artifacts": {
            "expert_queue": {
                "path": str(args.queue.resolve()),
                "sha256_utf8_canonical_lf": _text_sha256(args.queue.resolve()),
            },
            "r1_labels_hash_only_not_parsed": {
                "path": str(args.r1_labels.resolve()),
                "sha256_utf8_canonical_lf": _text_sha256(args.r1_labels.resolve()),
            },
            "human6_manifest": {
                "path": str(args.human6_manifest.resolve()),
                "sha256": _sha256(args.human6_manifest.resolve()),
            },
            "prior_holdout_manifest": {
                "path": str(args.prior_holdout_manifest.resolve()),
                "sha256": _sha256(args.prior_holdout_manifest.resolve()),
            },
        },
        "cases": cases,
    }
    _write(args.output.resolve(), manifest)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(args.output.resolve()),
                "manifest_sha256": _sha256(args.output.resolve()),
                "case_uids": [case["case_uid"] for case in cases],
                "incident_uids": [case["incident_uid"] for case in cases],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
