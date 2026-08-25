#!/usr/bin/env python3
"""Post-hoc development evaluation of frozen oracle-free identity decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from conceptgraph.revision.auto_constraints import semantic_constraint_fingerprint
from conceptgraph.revision.evidence_split import sha256_file


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _constraint_compatibility(
    candidate: dict[str, Any] | None, reference: dict[str, Any]
) -> dict[str, Any]:
    if candidate is None:
        return {
            "compatible": False,
            "relation": "NO_CANDIDATE",
            "mismatched_fields": [],
        }
    constraint_type = str(reference.get("type"))
    fields = [
        "type",
        "obs_uid",
        "applies_at_event_uid",
        "active_from_sequence",
        "active_until_sequence",
    ]
    if constraint_type == "ASSIGN_OBSERVATION":
        fields.extend(
            [
                "target_entity_uid",
                "target_lineage_uid",
                "target_origin_obs_uid",
            ]
        )
    elif constraint_type == "CREATE_INSTANCE":
        fields.extend(["created_entity_uid", "created_lineage_uid"])
    else:
        return {
            "compatible": False,
            "relation": "UNSUPPORTED_REFERENCE_TYPE",
            "mismatched_fields": ["type"],
        }
    mismatched = [
        field for field in fields if candidate.get(field) != reference.get(field)
    ]
    exact = semantic_constraint_fingerprint(
        candidate
    ) == semantic_constraint_fingerprint(reference)
    return {
        "compatible": not mismatched,
        "relation": (
            "EXACT_EXECUTION_SEMANTICS"
            if exact
            else (
                "COMPATIBLE_SAFETY_STRENGTHENING" if not mismatched else "INCOMPATIBLE"
            )
        ),
        "mismatched_fields": mismatched,
        "candidate_extra_separation_binding": bool(
            constraint_type == "CREATE_INSTANCE"
            and candidate.get("separate_from_identity_uids")
        ),
        "candidate_created_identity_bound": bool(
            constraint_type == "CREATE_INSTANCE"
            and candidate.get("created_identity_uid")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path, action="append")
    parser.add_argument("--human-manifest", required=True, type=Path)
    parser.add_argument("--identity-evidence-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    decision_paths = [path.resolve() for path in args.decision]
    decisions = {}
    for path in decision_paths:
        aggregate = _read(path)
        if aggregate.get("runtime_human_or_gold_loaded") is not False:
            raise ValueError(f"decision was not oracle isolated: {path}")
        for row in aggregate.get("cases") or ():
            case_uid = str(row["case_uid"])
            if case_uid in decisions:
                raise ValueError(f"duplicate decision case: {case_uid}")
            decisions[case_uid] = dict(row)

    evidence_path = args.identity_evidence_manifest.resolve()
    evidence = _read(evidence_path)
    source_case_by_blind = {
        str(row["blind_case_uid"]): str(row["development_source_case_uid"])
        for row in evidence.get("cases") or ()
    }
    human_path = args.human_manifest.resolve()
    human = _read(human_path)
    human_cases = {str(row["case_uid"]): row for row in human.get("cases") or ()}

    rows = []
    for blind_case_uid, decision in sorted(decisions.items()):
        source_case_uid = source_case_by_blind[blind_case_uid]
        source = human_cases[source_case_uid]
        constraints = list(source.get("constraints") or ())
        if len(constraints) != 1:
            raise ValueError(f"{source_case_uid}: expected one development constraint")
        gold_fingerprint = semantic_constraint_fingerprint(constraints[0])
        selected = decision.get("shadow_replay_constraint")
        selected_fingerprint = (
            semantic_constraint_fingerprint(selected) if selected else None
        )
        private_path = Path(str(decision["execution_private_path"])).resolve()
        if sha256_file(private_path) != str(decision["execution_private_sha256"]):
            raise ValueError(f"execution private hash drift: {private_path}")
        private = _read(private_path)
        candidate_constraints = [
            dict(row["constraint"]) for row in private.get("candidate_replays") or ()
        ]
        candidate_compatibility = [
            _constraint_compatibility(candidate, constraints[0])
            for candidate in candidate_constraints
        ]
        candidate_target_recall = any(
            item["compatible"] for item in candidate_compatibility
        )
        selected_compatibility = _constraint_compatibility(selected, constraints[0])
        recommended = decision["shadow_status"] == "SHADOW_REPAIR_RECOMMENDED"
        matches = bool(recommended and selected_compatibility["compatible"])
        rows.append(
            {
                "blind_case_uid": blind_case_uid,
                "development_source_case_uid": source_case_uid,
                "endpoint_error_type": str(source["endpoint_error_type"]),
                "shadow_status": str(decision["shadow_status"]),
                "candidate_target_recall": candidate_target_recall,
                "candidate_execution_compatibility": candidate_compatibility,
                "selected_constraint_fingerprint": selected_fingerprint,
                "development_gold_constraint_fingerprint": gold_fingerprint,
                "selected_exact_fingerprint_match": bool(
                    selected_fingerprint == gold_fingerprint
                ),
                "selected_execution_compatibility": selected_compatibility,
                "repair_recommended": recommended,
                "recommended_repair_matches_development_gold": matches,
                "production_decision": str(
                    decision["production_selective_decision"]["decision"]
                ),
                "evaluation_role": "POSTHOC_DEVELOPMENT_ONLY",
            }
        )

    recommendation_count = sum(row["repair_recommended"] for row in rows)
    correct_count = sum(
        row["recommended_repair_matches_development_gold"] for row in rows
    )
    incorrect_count = recommendation_count - correct_count
    case_count = len(rows)
    aggregate = {
        "schema_version": "1.0.0",
        "evaluation_role": "POSTHOC_DEVELOPMENT_ONLY_NOT_RUNTIME_SELECTION",
        "decision_paths": [str(path) for path in decision_paths],
        "decision_sha256": {str(path): sha256_file(path) for path in decision_paths},
        "human_manifest_path": str(human_path),
        "human_manifest_sha256": sha256_file(human_path),
        "identity_evidence_manifest_path": str(evidence_path),
        "identity_evidence_manifest_sha256": sha256_file(evidence_path),
        "case_count": case_count,
        "candidate_target_recall_count": sum(
            row["candidate_target_recall"] for row in rows
        ),
        "shadow_repair_recommendation_count": recommendation_count,
        "correct_shadow_repair_recommendation_count": correct_count,
        "incorrect_shadow_repair_recommendation_count": incorrect_count,
        "shadow_selective_precision": (
            correct_count / recommendation_count if recommendation_count else None
        ),
        "confirmed_error_repair_recall": (
            correct_count / case_count if case_count else None
        ),
        "stable_noop_miss_count": sum(
            row["shadow_status"] == "SHADOW_NOOP_PREFERRED" for row in rows
        ),
        "inconclusive_count": sum(
            row["shadow_status"] == "SHADOW_INCONCLUSIVE" for row in rows
        ),
        "production_commit_count": sum(
            row["production_decision"] == "COMMIT" for row in rows
        ),
        "production_defer_count": sum(
            row["production_decision"] == "DEFER" for row in rows
        ),
        "unsafe_shadow_recommendation_count": incorrect_count,
        "cases": rows,
    }
    _write(args.output.resolve(), aggregate)
    print(
        json.dumps(
            {
                key: aggregate[key]
                for key in (
                    "case_count",
                    "candidate_target_recall_count",
                    "shadow_repair_recommendation_count",
                    "correct_shadow_repair_recommendation_count",
                    "incorrect_shadow_repair_recommendation_count",
                    "shadow_selective_precision",
                    "confirmed_error_repair_recall",
                    "stable_noop_miss_count",
                    "production_commit_count",
                    "production_defer_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
