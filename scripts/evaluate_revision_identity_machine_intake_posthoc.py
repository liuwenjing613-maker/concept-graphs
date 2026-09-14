#!/usr/bin/env python3
"""Post-hoc evaluation of machine-intake decisions against available labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from conceptgraph.revision.evidence_split import sha256_file


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_labels(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            incident_uid = str(value.get("incident_uid") or value.get("case_uid") or "")
            if not incident_uid:
                raise ValueError(f"{path}:{line_number} lacks incident identity")
            if incident_uid in rows:
                raise ValueError(f"duplicate human label: {incident_uid}")
            rows[incident_uid] = value
    return rows


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path, action="append")
    parser.add_argument("--machine-intake-audit", required=True, type=Path)
    parser.add_argument("--human-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    decision_paths = [path.resolve() for path in args.decision]
    decisions: dict[str, dict[str, Any]] = {}
    for path in decision_paths:
        aggregate = _read_object(path)
        if aggregate.get("runtime_human_or_gold_loaded") is not False:
            raise ValueError(f"decision was not oracle isolated: {path}")
        for row in aggregate.get("cases") or ():
            case_uid = str(row["case_uid"])
            if case_uid in decisions:
                raise ValueError(f"duplicate decision case: {case_uid}")
            decisions[case_uid] = dict(row)

    audit_path = args.machine_intake_audit.resolve()
    audit = _read_object(audit_path)
    if audit.get("role") != "PRIVATE_MACHINE_INTAKE_TRACE_NOT_INFERENCE_INPUT":
        raise ValueError("unexpected machine-intake audit role")
    if audit.get("human_labels_loaded") is not False:
        raise ValueError("machine intake was not label blind")
    source_by_blind = {
        str(row["blind_case_uid"]): {
            "source_incident_uid": str(row["source_incident_uid"]),
            "scene_id": str(row["scene_id"]),
        }
        for row in audit.get("accepted_cases") or ()
    }
    if set(decisions) != set(source_by_blind):
        raise ValueError(
            "decision/intake case mismatch: "
            f"decisions={sorted(decisions)}, intake={sorted(source_by_blind)}"
        )

    labels_path = args.human_labels.resolve()
    labels = _read_labels(labels_path)
    rows = []
    for blind_case_uid in sorted(decisions):
        decision = decisions[blind_case_uid]
        source = source_by_blind[blind_case_uid]
        label = labels.get(source["source_incident_uid"])
        final_state = str(label["final_state"]) if label else None
        expected_action = (
            "NO_OP"
            if final_state == "CORRECT"
            else "REPAIR"
            if final_state in {"ERROR", "WRONG"}
            else None
        )
        shadow_status = str(decision["shadow_status"])
        production_decision = str(decision["production_selective_decision"]["decision"])
        rows.append(
            {
                "blind_case_uid": blind_case_uid,
                "source_incident_uid": source["source_incident_uid"],
                "scene_id": source["scene_id"],
                "human_label_available": label is not None,
                "human_final_state": final_state,
                "human_final_error_type": (
                    str(label["final_error_type"]) if label else None
                ),
                "expected_action_when_defined": expected_action,
                "shadow_status": shadow_status,
                "production_decision": production_decision,
                "shadow_repair_false_positive": bool(
                    expected_action == "NO_OP"
                    and shadow_status == "SHADOW_REPAIR_RECOMMENDED"
                ),
                "shadow_confirmed_error_repair": bool(
                    expected_action == "REPAIR"
                    and shadow_status == "SHADOW_REPAIR_RECOMMENDED"
                ),
                "unsafe_production_commit": bool(
                    expected_action == "NO_OP" and production_decision == "COMMIT"
                ),
                "production_abstained": production_decision == "DEFER",
                "evaluation_role": "POSTHOC_ONLY_NOT_RUNTIME_SELECTION",
            }
        )

    labeled = [row for row in rows if row["human_label_available"]]
    aggregate = {
        "schema_version": "1.0.0",
        "evaluation_role": "POSTHOC_MACHINE_INTAKE_ONLY_NOT_RUNTIME_SELECTION",
        "runtime_decisions_were_label_blind": True,
        "decision_paths": [str(path) for path in decision_paths],
        "decision_sha256": {str(path): sha256_file(path) for path in decision_paths},
        "machine_intake_audit_path": str(audit_path),
        "machine_intake_audit_sha256": sha256_file(audit_path),
        "human_labels_path": str(labels_path),
        "human_labels_sha256": sha256_file(labels_path),
        "case_count": len(rows),
        "labeled_case_count": len(labeled),
        "unlabeled_case_count": len(rows) - len(labeled),
        "labeled_clean_case_count": sum(
            row["expected_action_when_defined"] == "NO_OP" for row in rows
        ),
        "labeled_error_case_count": sum(
            row["expected_action_when_defined"] == "REPAIR" for row in rows
        ),
        "shadow_repair_recommendation_count": sum(
            row["shadow_status"] == "SHADOW_REPAIR_RECOMMENDED" for row in rows
        ),
        "shadow_inconclusive_count": sum(
            row["shadow_status"] == "SHADOW_INCONCLUSIVE" for row in rows
        ),
        "shadow_repair_false_positive_count": sum(
            row["shadow_repair_false_positive"] for row in rows
        ),
        "shadow_confirmed_error_repair_count": sum(
            row["shadow_confirmed_error_repair"] for row in rows
        ),
        "production_commit_count": sum(
            row["production_decision"] == "COMMIT" for row in rows
        ),
        "unsafe_production_commit_count": sum(
            row["unsafe_production_commit"] for row in rows
        ),
        "labeled_production_abstention_count": sum(
            row["human_label_available"] and row["production_abstained"] for row in rows
        ),
        "labeled_clean_protected_count": sum(
            row["expected_action_when_defined"] == "NO_OP"
            and row["production_abstained"]
            for row in rows
        ),
        "labeled_error_unrepaired_count": sum(
            row["expected_action_when_defined"] == "REPAIR"
            and row["production_abstained"]
            for row in rows
        ),
        "cases": rows,
    }
    _write(args.output.resolve(), aggregate)
    print(
        json.dumps(
            {
                key: aggregate[key]
                for key in (
                    "case_count",
                    "labeled_case_count",
                    "unlabeled_case_count",
                    "labeled_clean_case_count",
                    "labeled_error_case_count",
                    "shadow_repair_recommendation_count",
                    "shadow_inconclusive_count",
                    "shadow_repair_false_positive_count",
                    "shadow_confirmed_error_repair_count",
                    "production_commit_count",
                    "unsafe_production_commit_count",
                    "labeled_clean_protected_count",
                    "labeled_error_unrepaired_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
