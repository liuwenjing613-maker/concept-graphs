#!/usr/bin/env python3
"""Post-hoc no-op preservation evaluation for frozen clean identity controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path, action="append")
    parser.add_argument("--posthoc-key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    decisions = {}
    decision_paths = [path.resolve() for path in args.decision]
    for path in decision_paths:
        aggregate = _read(path)
        if aggregate.get("runtime_human_or_gold_loaded") is not False:
            raise ValueError(f"decision was not oracle isolated: {path}")
        for row in aggregate.get("cases") or ():
            case_uid = str(row["case_uid"])
            if case_uid in decisions:
                raise ValueError(f"duplicate decision case: {case_uid}")
            decisions[case_uid] = dict(row)

    key_path = args.posthoc_key.resolve()
    key = _read(key_path)
    if key.get("role") != "POSTHOC_ONLY_DO_NOT_FEED_TO_RUNTIME":
        raise ValueError("invalid clean-control post-hoc key role")
    gold = {str(row["blind_case_uid"]): row for row in key.get("cases") or ()}
    if set(decisions) != set(gold):
        raise ValueError(
            f"decision/key case mismatch: decisions={sorted(decisions)}, "
            f"key={sorted(gold)}"
        )

    rows = []
    for case_uid in sorted(decisions):
        decision = decisions[case_uid]
        reference = gold[case_uid]
        if reference.get("expected_runtime_action") != "NO_OP":
            raise ValueError(f"{case_uid}: post-hoc reference is not NO_OP")
        shadow_status = str(decision["shadow_status"])
        repair_recommended = shadow_status == "SHADOW_REPAIR_RECOMMENDED"
        no_op_preferred = shadow_status == "SHADOW_NOOP_PREFERRED"
        inconclusive = shadow_status == "SHADOW_INCONCLUSIVE"
        production_decision = str(decision["production_selective_decision"]["decision"])
        rows.append(
            {
                "blind_case_uid": case_uid,
                "scene_id": str(reference["scene_id"]),
                "development_source_case_uid": str(reference["source_case_uid"]),
                "shadow_status": shadow_status,
                "strict_noop_preferred": no_op_preferred,
                "inconclusive_but_no_repair": inconclusive,
                "unsafe_shadow_repair_false_positive": repair_recommended,
                "production_decision": production_decision,
                "unsafe_production_commit": production_decision == "COMMIT",
                "evaluation_role": "POSTHOC_DEVELOPMENT_CLEAN_CONTROL",
            }
        )

    case_count = len(rows)
    false_positives = sum(row["unsafe_shadow_repair_false_positive"] for row in rows)
    aggregate = {
        "schema_version": "1.0.0",
        "evaluation_role": "POSTHOC_DEVELOPMENT_CLEAN_CONTROL_ONLY",
        "decision_paths": [str(path) for path in decision_paths],
        "decision_sha256": {str(path): sha256_file(path) for path in decision_paths},
        "posthoc_key_path": str(key_path),
        "posthoc_key_sha256": sha256_file(key_path),
        "clean_case_count": case_count,
        "strict_noop_preference_count": sum(
            row["strict_noop_preferred"] for row in rows
        ),
        "inconclusive_no_repair_count": sum(
            row["inconclusive_but_no_repair"] for row in rows
        ),
        "shadow_repair_false_positive_count": false_positives,
        "shadow_no_repair_rate": (
            (case_count - false_positives) / case_count if case_count else None
        ),
        "strict_noop_preference_rate": (
            sum(row["strict_noop_preferred"] for row in rows) / case_count
            if case_count
            else None
        ),
        "production_commit_count": sum(
            row["production_decision"] == "COMMIT" for row in rows
        ),
        "unsafe_production_commit_count": sum(
            row["unsafe_production_commit"] for row in rows
        ),
        "cases": rows,
    }
    _write(args.output.resolve(), aggregate)
    print(
        json.dumps(
            {
                key: aggregate[key]
                for key in (
                    "clean_case_count",
                    "strict_noop_preference_count",
                    "inconclusive_no_repair_count",
                    "shadow_repair_false_positive_count",
                    "shadow_no_repair_rate",
                    "production_commit_count",
                    "unsafe_production_commit_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
