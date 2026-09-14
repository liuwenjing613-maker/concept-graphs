#!/usr/bin/env python3
"""Create a causal-trace/replay queue only from confirmed R1 endpoint errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FINAL_STATES = {"CORRECT", "WRONG", "UNCLEAR"}
ERROR_TYPES = {
    "NOT_APPLICABLE",
    "FALSE_MERGE",
    "FALSE_SPLIT",
    "SPURIOUS_OBJECT",
    "MISSING_OBJECT",
    "WRONG_MEMBERSHIP",
    "GEOMETRY_CORRUPTION",
    "SEMANTIC_IDENTITY_ERROR",
    "OTHER",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def key(row: dict[str, Any]) -> tuple[str, str]:
    value = str(row.get("scene_id") or ""), str(
        row.get("incident_uid") or row.get("case_uid") or ""
    )
    if not all(value):
        raise ValueError("incident row needs scene_id and incident_uid")
    return value


def validate_label(row: dict[str, Any], incident_key: tuple[str, str]) -> None:
    prefix = f"{incident_key[0]}/{incident_key[1]}"
    evidence = row.get("evidence_sufficient")
    state = row.get("final_state")
    error_type = row.get("final_error_type")
    if row.get("reviewer_id") != "R1":
        raise ValueError(f"reviewer_id must be R1 for {prefix}")
    if evidence not in {"YES", "NO"} or state not in FINAL_STATES or error_type not in ERROR_TYPES:
        raise ValueError(f"invalid endpoint label enum for {prefix}")
    if evidence == "NO" and state != "UNCLEAR":
        raise ValueError(f"evidence NO requires UNCLEAR for {prefix}")
    if evidence == "YES" and state == "UNCLEAR":
        raise ValueError(f"evidence YES cannot be UNCLEAR for {prefix}")
    if state == "WRONG" and error_type == "NOT_APPLICABLE":
        raise ValueError(f"WRONG requires an endpoint error type for {prefix}")
    if state != "WRONG" and error_type != "NOT_APPLICABLE":
        raise ValueError(f"non-WRONG requires NOT_APPLICABLE for {prefix}")
    if error_type == "OTHER" and not str(row.get("notes") or "").strip():
        raise ValueError(f"OTHER requires notes for {prefix}")
    try:
        seconds = float(row.get("review_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid review_seconds for {prefix}") from exc
    if seconds < 0:
        raise ValueError(f"negative review_seconds for {prefix}")


def generate(worklist: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    work_index = {key(row): row for row in worklist}
    label_index = {key(row): row for row in labels}
    if len(work_index) != len(worklist) or len(label_index) != len(labels):
        raise ValueError("duplicate incident key")
    if set(work_index) != set(label_index):
        raise ValueError("R1 must be complete before creating the expert queue")
    output = []
    for incident_key in sorted(work_index):
        meta = work_index[incident_key]
        label = label_index[incident_key]
        validate_label(label, incident_key)
        if label.get("evidence_sufficient") != "YES" or label.get("final_state") != "WRONG":
            continue
        output.append(
            {
                "schema_version": "1.0.0",
                "scene_id": incident_key[0],
                "incident_uid": incident_key[1],
                "endpoint_error_type": label.get("final_error_type"),
                "representative_finding_uid": meta.get("representative_finding_uid"),
                "linked_finding_uids": meta.get("member_finding_uids") or [],
                "candidate_checker_ids": meta.get("checker_ids") or [],
                "candidate_stages": meta.get("stages") or [],
                "representative_trigger_observation_uids": meta.get(
                    "representative_trigger_observation_uids"
                ) or meta.get("trigger_observation_uids") or [],
                "trigger_observation_uids": meta.get("all_trigger_observation_uids")
                or meta.get("trigger_observation_uids")
                or [],
                "final_owner_uids": meta.get("final_owner_uids") or [],
                "case_dir": meta.get("case_dir"),
                "expert_status": "PENDING_CAUSAL_TRACE",
                "earliest_causal_stage": None,
                "causal_chain": None,
                "root_evidence_refs": [],
                "repair_hypothesis": None,
                "intervention_plan": None,
                "replay_status": "NOT_RUN",
                "replay_output": None,
                "repair_verified": None,
                "expert_notes": None,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.validation_root.resolve()
    try:
        worklist = read_jsonl(root / "labels" / "r1_worklist.jsonl")
        labels = read_jsonl(root / "labels" / "labels_r1.jsonl")
        rows = generate(worklist, labels)
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "NOT_READY", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    output = root / "expert" / "confirmed_endpoint_error_queue.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "status": "READY",
                "confirmed_endpoint_error_count": len(rows),
                "queue": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
