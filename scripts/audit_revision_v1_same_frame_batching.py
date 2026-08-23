from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.benchmark.experiment_v1 import selected_metric_paths


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _target_origin_difference(
    provenance: ProvenanceIndex,
    association: dict[str, Any],
    trace_row: dict[str, Any],
) -> dict[str, Any] | None:
    """Compare an untruncated immutable target identity across a frame boundary."""

    version_uid = association.get("target_object_version_before")
    if version_uid not in provenance.object_versions:
        return {
            "recorded_version_uid": version_uid,
            "replayed_origin": trace_row.get("natural_target_origin_obs_uid"),
            "reason": "recorded_target_version_missing",
        }
    version = provenance.get_object_version(str(version_uid))
    members = list(version.get("member_observation_uids") or ())
    expected_origin = version.get("origin_observation_uid") or (
        members[0] if members else None
    )
    replayed_origin = trace_row.get("natural_target_origin_obs_uid")
    if str(expected_origin or "") == str(replayed_origin or ""):
        return None
    return {
        "recorded": expected_origin,
        "replayed": replayed_origin,
    }


def audit(base_run: Path, output_root: Path) -> dict[str, Any]:
    provenance = ProvenanceIndex(base_run)
    cases = []
    mismatch_count = 0
    compared_count = 0
    pre_anchor_replayed_count = 0
    metric_paths, selection_integrity = selected_metric_paths(output_root)
    for metrics_path in metric_paths:
        metrics = _read(metrics_path)
        if metrics.get("status") != "COMPLETED":
            continue
        case_root = metrics_path.parent
        case = _read(case_root / "case.json")
        frame = int(case["frame_idx"])
        trace = _read(case_root / "replay_decision_trace.json")["persistent_sparse"]
        snapshot = _read(case_root / "pre_anchor_snapshot.json")
        watermark = int(snapshot["watermark_event_sequence"])
        mismatches = []
        pre_anchor_replayed = []
        for row in trace:
            association = provenance.get_association_for_obs(str(row["obs_uid"]))
            sequence = provenance.sequence(association)
            if sequence <= watermark:
                pre_anchor_replayed_count += 1
                pre_anchor_replayed.append(
                    {
                        "obs_uid": row["obs_uid"],
                        "event_uid": row["event_uid"],
                        "event_sequence": sequence,
                    }
                )
                continue
            if int(row["frame_idx"]) != frame:
                continue
            compared_count += 1
            expected_create = str(association["decision"]) == "CREATE_OBJECT"
            actual_create = row.get("natural_match") is None
            mismatch: dict[str, Any] = {}
            if row.get("native_default_source") != "RECORDED_FRAME_START_ASSOCIATION":
                mismatch["native_default_source"] = {
                    "expected": "RECORDED_FRAME_START_ASSOCIATION",
                    "actual": row.get("native_default_source"),
                }
            if expected_create != actual_create:
                mismatch["decision_kind"] = {
                    "recorded": association["decision"],
                    "recomputed_natural_match": row.get("natural_match"),
                }
            if not expected_create and not actual_create:
                difference = _target_origin_difference(
                    provenance, association, row
                )
                if difference is not None:
                    mismatch["target_origin"] = difference
            if mismatch:
                mismatch_count += 1
                mismatches.append(
                    {
                        "obs_uid": row["obs_uid"],
                        "event_uid": row["event_uid"],
                        "differences": mismatch,
                        "natural_candidates": row.get("natural_candidates"),
                    }
                )
        cases.append(
            {
                "case_uid": case["case_uid"],
                "anchor_frame": frame,
                "same_frame_compared": sum(int(row["frame_idx"]) == frame for row in trace),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "pre_anchor_replayed_count": len(pre_anchor_replayed),
                "pre_anchor_replayed": pre_anchor_replayed,
            }
        )
    return {
        "schema_version": "1.0.0",
        "pass": mismatch_count == 0 and pre_anchor_replayed_count == 0,
        "selection_integrity": selection_integrity,
        "case_count": len(cases),
        "compared_decision_count": compared_count,
        "mismatch_count": mismatch_count,
        "pre_anchor_replayed_count": pre_anchor_replayed_count,
        "cases": cases,
        "interpretation": (
            "A failure means the suffix replay crossed its snapshot watermark or did not "
            "reuse a decision frozen in the native mapper's frame-start match matrix."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit mid-frame replay decision fidelity")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    result = audit(Path(args.base_run).resolve(), root)
    destination = Path(args.output).resolve() if args.output else root / "same_frame_batching_audit.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
