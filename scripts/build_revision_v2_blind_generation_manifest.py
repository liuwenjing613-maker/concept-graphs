#!/usr/bin/env python3
"""Materialize label-free inputs for the automatic constraint generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--blind-input-uid",
        default="revision_v2_auto_constraint_blind_inputs_20260824",
    )
    args = parser.parse_args()

    identity = _read(args.identity_manifest)
    holdout = _read(args.holdout_manifest)
    replayable = sorted(
        (
            case
            for case in identity.get("cases", ())
            if case.get("causal_disposition") == "REPLAYABLE_ASSOCIATION_CAUSE"
        ),
        key=lambda case: str(case["anchor_association_event_uid"]),
    )
    if len(replayable) != 3:
        raise ValueError("expected exactly three replayable association incidents")
    holdout_cases = sorted(
        holdout.get("cases", ()), key=lambda case: str(case["incident_uid"])
    )
    if len(holdout_cases) != 2:
        raise ValueError("expected exactly two frozen capability holdouts")

    cases = []
    for case in replayable:
        cases.append(
            {
                "blind_case_uid": f"blind_identity_{len(cases) + 1:03d}",
                "input_family": "IDENTITY_ASSOCIATION",
                "scene_id": str(case["scene_id"]),
                "anchor_association_event_uid": str(
                    case["anchor_association_event_uid"]
                ),
                "source_case_uid_excluded": True,
                "endpoint_error_type_excluded": True,
                "human_label_excluded": True,
                "human_notes_excluded": True,
                "final_membership_excluded": True,
            }
        )
    for case in holdout_cases:
        artifacts = []
        for artifact in case["inference_inputs"]["evidence_artifacts"]:
            path = Path(str(artifact["path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _sha256(path)
            if actual != artifact["sha256"]:
                raise ValueError(f"holdout artifact drift: {path}")
            artifacts.append(
                {
                    "logical_name": str(artifact["logical_name"]),
                    "path": str(path),
                    "sha256": actual,
                    "bytes": path.stat().st_size,
                }
            )
        cases.append(
            {
                "blind_case_uid": f"blind_capability_{len(cases) - 2:03d}",
                "input_family": "CAPABILITY_PROBE",
                "scene_id": str(case["scene_id"]),
                "incident_uid": str(case["incident_uid"]),
                "representative_trigger_observation_uids": [
                    str(item)
                    for item in case["inference_inputs"][
                        "representative_trigger_observation_uids"
                    ]
                ],
                "evidence_artifacts": artifacts,
                "source_case_uid_excluded": True,
                "endpoint_error_type_excluded": True,
                "expected_capability_excluded": True,
                "human_label_excluded": True,
                "human_notes_excluded": True,
                "final_owner_uids_excluded": True,
            }
        )

    output = {
        "schema_version": "2.0.0",
        "blind_input_uid": str(args.blind_input_uid),
        "frozen_before_generator_responses": True,
        "selection_uses_generator_outputs": False,
        "case_count": len(cases),
        "source_manifest_integrity_only": {
            "identity_manifest_sha256": _sha256(args.identity_manifest.resolve()),
            "holdout_manifest_sha256": _sha256(args.holdout_manifest.resolve()),
        },
        "request_exclusions": [
            "source case UID",
            "endpoint error type",
            "human endpoint label",
            "human notes",
            "expected candidate family",
            "expected constraint",
            "final owner UID",
            "final membership",
        ],
        "cases": cases,
    }
    _write(args.output, output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "case_count": len(cases),
                "blind_case_uids": [case["blind_case_uid"] for case in cases],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
