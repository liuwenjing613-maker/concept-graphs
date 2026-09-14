#!/usr/bin/env python3
"""Evaluate causal relation impact against a frozen, identity-stable gold scope."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.relation_impact import (
    canonical_edge_map,
    edge_diff,
    gold_key,
    gold_metrics,
    indexed_predictions,
    membership_changed_entities,
    restrict_edges,
    split_stable_gold,
)
from conceptgraph.revision.relations import load_edge_stream


_BRANCHES = ("native", "natural", "sparse")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triples(values: Iterable[Iterable[Any]]) -> set[tuple[int, int, str]]:
    result = set()
    for value in values:
        source, target, predicate = value
        result.add((int(source), int(target), str(predicate)))
    return result


def _formal_map_index(path: Path) -> dict[str, int]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        raise ValueError("formal map does not contain an objects list")
    result = {str(item["id"]): index for index, item in enumerate(payload["objects"])}
    if len(result) != len(payload["objects"]):
        raise ValueError("formal map object UUIDs are not unique")
    return result


def _edge_change_is_local(
    diff: Mapping[str, Any], localization_entities: set[str]
) -> bool:
    changed_rows = [
        row
        for section in ("added", "removed", "support_changed")
        for row in diff[section]
    ]
    return all(
        bool({str(row["edge"][0]), str(row["edge"][2])} & localization_entities)
        for row in changed_rows
    )


def _case_evaluation(
    *,
    case_root: Path,
    entity_to_index: Mapping[str, int],
    gold_rows: list[dict[str, Any]],
    gold_keys: set[tuple[int, int, str]],
    reference_predictions: set[tuple[int, int, str]],
    expected_input_edges: int,
) -> dict[str, Any]:
    case = _read(case_root / "case_manifest.json")
    metrics = _read(case_root / "metrics.json")
    relation = _read(case_root / "relation_rebuild.json")
    if str(case["scene_id"]) != "room0":
        raise ValueError(f"relation gold is room0-only: {case_root.name}")

    states = {
        name: _read(case_root / "branches" / f"{name}.json") for name in _BRANCHES
    }
    edge_maps = {
        name: canonical_edge_map(states[name].get("edges") or ()) for name in _BRANCHES
    }
    relation_branches = relation.get("branches") or {}
    if set(relation_branches) != set(_BRANCHES):
        raise ValueError(f"missing relation branch in {case_root}")

    native_membership = states["native"].get("membership") or {}
    changed_entities = set()
    for name in ("natural", "sparse"):
        changed_entities.update(
            membership_changed_entities(
                native_membership, states[name].get("membership") or {}
            )
        )
    affected_entities = {
        str(item)
        for item in (case.get("evaluation") or {}).get(
            "affected_native_entity_uids", ()
        )
    }
    localization_entities = changed_entities | affected_entities
    impacted_indices = {
        int(entity_to_index[entity])
        for entity in localization_entities
        if entity in entity_to_index
    }
    stable_rows, impacted_rows = split_stable_gold(gold_rows, impacted_indices)
    full_gold_admissible = not impacted_rows

    branch_results = {}
    for name in _BRANCHES:
        predictions, unmapped = indexed_predictions(edge_maps[name], entity_to_index)
        stable_metrics = gold_metrics(stable_rows, predictions)
        full_metrics = gold_metrics(gold_rows, predictions)
        relation_branch = relation_branches[name]
        branch_results[name] = {
            "edge_count": len(edge_maps[name]),
            "total_support": sum(edge_maps[name].values()),
            "mapped_prediction_count": len(predictions),
            "unmapped_edge_count": len(unmapped),
            "unmapped_edges": unmapped,
            "unknown_prediction_count_outside_gold_scope": (
                len(predictions - gold_keys) + len(unmapped)
            ),
            "stable_gold_metrics": stable_metrics,
            "full_gold_metrics_if_admissible": (
                full_metrics if full_gold_admissible else None
            ),
            "relation_backend_validation": relation_branch["validation"],
            "input_edge_observations": int(relation_branch["input_edge_observations"]),
            "predictions": predictions,
        }

    natural_diff = edge_diff(edge_maps["native"], edge_maps["natural"])
    sparse_diff = edge_diff(edge_maps["native"], edge_maps["sparse"])
    native_outside = restrict_edges(edge_maps["native"], localization_entities)
    outside_exact = {
        name: restrict_edges(edge_maps[name], localization_entities) == native_outside
        for name in _BRANCHES
    }
    native_stable_metrics = branch_results["native"]["stable_gold_metrics"]
    stable_gold_exact = {
        name: branch_results[name]["stable_gold_metrics"] == native_stable_metrics
        for name in _BRANCHES
    }
    all_branch_valid = all(
        bool(branch_results[name]["relation_backend_validation"]["pass"])
        for name in _BRANCHES
    )
    all_input_counts_exact = all(
        branch_results[name]["input_edge_observations"] == expected_input_edges
        for name in _BRANCHES
    )
    native_calibration_exact = (
        branch_results["native"]["predictions"] == reference_predictions
    )

    checks = {
        "causal_contrast_pass": (
            metrics.get("status") == "PASS" and bool(metrics.get("contrast_pass"))
        ),
        "relation_rebuild_pass": relation.get("status") == "PASS",
        "all_branch_backend_validation_pass": all_branch_valid,
        "all_branch_input_edge_counts_exact": all_input_counts_exact,
        "native_predictions_exactly_calibrate_to_frozen_source": (
            native_calibration_exact
        ),
        "natural_relation_state_exact_to_native": natural_diff["exact"],
        "all_relation_changes_touch_identity_localization": (
            _edge_change_is_local(sparse_diff, localization_entities)
        ),
        "outside_relation_state_and_support_exact": all(outside_exact.values()),
        "stable_gold_metrics_exact_across_branches": all(stable_gold_exact.values()),
    }
    serializable_branches = {}
    for name, row in branch_results.items():
        serializable_branches[name] = {
            key: value for key, value in row.items() if key != "predictions"
        }

    return {
        "case_uid": str(case["case_uid"]),
        "incident_uid": str(case["incident_uid"]),
        "endpoint_error_type": str(case["endpoint_error_type"]),
        "pass": all(checks.values()),
        "checks": checks,
        "identity_localization": {
            "affected_native_entity_uids": sorted(affected_entities),
            "membership_changed_entity_uids": sorted(changed_entities),
            "localization_entity_uids": sorted(localization_entities),
            "impacted_formal_map_indices": sorted(impacted_indices),
        },
        "gold_scope": {
            "frozen_label_count": len(gold_rows),
            "stable_label_count": len(stable_rows),
            "identity_impacted_label_count": len(impacted_rows),
            "full_gold_comparison_admissible": full_gold_admissible,
            "impacted_gold_uids": sorted(str(row["gold_uid"]) for row in impacted_rows),
        },
        "branches": serializable_branches,
        "causal_relation_delta": {
            "natural_vs_native": natural_diff,
            "sparse_vs_native": sparse_diff,
            "outside_exact_to_native": outside_exact,
            "stable_gold_metrics_exact_to_native": stable_gold_exact,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--case-uid", action="append", required=True)
    parser.add_argument("--formal-map", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--source-evaluation", required=True, type=Path)
    parser.add_argument("--edge-stream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    case_uids = list(dict.fromkeys(str(item) for item in args.case_uid))
    if len(case_uids) != 2:
        raise ValueError("this preregistered impact slice requires exactly two cases")

    paths = {
        "formal_map": args.formal_map.resolve(),
        "gold": args.gold.resolve(),
        "source_evaluation": args.source_evaluation.resolve(),
        "edge_manifest": (args.edge_stream.resolve() / "manifest.json"),
        "edge_frames": (args.edge_stream.resolve() / "frames.jsonl"),
        "pilot_aggregate": (args.pilot_root.resolve() / "aggregate_metrics.json"),
    }
    source_hashes_before = {name: _sha256(path) for name, path in paths.items()}

    entity_to_index = _formal_map_index(paths["formal_map"])
    gold = _read(paths["gold"])
    source_evaluation = _read(paths["source_evaluation"])
    if _sha256(paths["source_evaluation"]) != str(
        (gold.get("source_artifact") or {}).get("sha256")
    ):
        raise ValueError("gold source evaluation hash does not match frozen binding")
    if str(
        ((source_evaluation.get("inputs") or {}).get("map_sha256") or "")
    ) != _sha256(paths["formal_map"]):
        raise ValueError("formal map does not match relation evaluation source")

    edge_manifest, _edge_frames = load_edge_stream(args.edge_stream.resolve())
    expected_input_edges = int(edge_manifest["input_edge_observations"])
    gold_rows = [dict(item) for item in gold.get("relations") or ()]
    gold_keys = {gold_key(row) for row in gold_rows}
    if len(gold_keys) != len(gold_rows):
        raise ValueError("frozen gold has duplicate pair-direction keys")
    reference_predictions = _triples(
        (source_evaluation.get("audit") or {}).get("predicted_triples") or ()
    )
    reference_metrics = gold_metrics(gold_rows, reference_predictions)

    cases = [
        _case_evaluation(
            case_root=args.pilot_root.resolve() / case_uid,
            entity_to_index=entity_to_index,
            gold_rows=gold_rows,
            gold_keys=gold_keys,
            reference_predictions=reference_predictions,
            expected_input_edges=expected_input_edges,
        )
        for case_uid in case_uids
    ]
    source_hashes_after = {name: _sha256(path) for name, path in paths.items()}
    effect_counts = {
        "relation_state_exact": sum(
            bool(case["causal_relation_delta"]["sparse_vs_native"]["exact"])
            for case in cases
        ),
        "support_only_change": sum(
            not case["causal_relation_delta"]["sparse_vs_native"]["exact"]
            and not case["causal_relation_delta"]["sparse_vs_native"]["added"]
            and not case["causal_relation_delta"]["sparse_vs_native"]["removed"]
            and bool(
                case["causal_relation_delta"]["sparse_vs_native"]["support_changed"]
            )
            for case in cases
        ),
        "edge_set_change": sum(
            bool(case["causal_relation_delta"]["sparse_vs_native"]["added"])
            or bool(case["causal_relation_delta"]["sparse_vs_native"]["removed"])
            for case in cases
        ),
    }
    checks = {
        "exactly_two_preregistered_cases": len(cases) == 2,
        "all_case_gates_pass": all(bool(case["pass"]) for case in cases),
        "reference_native_gold_metrics_reproduced": (
            reference_metrics
            == {
                "label_count": 44,
                "tp": 20,
                "fp": 0,
                "fn": 2,
                "tn": 22,
                "accuracy": 0.9545454545454546,
                "precision_within_labeled_pair_scope": 1.0,
                "recall_within_labeled_pair_scope": 0.9090909090909091,
                "specificity_within_labeled_pair_scope": 1.0,
                "f1_within_labeled_pair_scope": 0.9523809523809523,
                "balanced_accuracy": 0.9545454545454546,
            }
        ),
        "all_44_gold_labels_identity_stable_for_this_slice": all(
            case["gold_scope"]["identity_impacted_label_count"] == 0 for case in cases
        ),
        "source_artifacts_unchanged": source_hashes_before == source_hashes_after,
    }
    result = {
        "schema_version": "1.0.0",
        "evaluation_role": ("CAUSAL_RELATION_IMPACT_AND_IDENTITY_STABLE_FROZEN_GOLD"),
        "production_commit_permitted": False,
        "pass": all(checks.values()),
        "checks": checks,
        "protocol": {
            "relation_input": (
                "Frozen 200-frame make_edges stream; 2425 immutable directed "
                "edge observations."
            ),
            "aggregation": "ali-dev process_edges unchanged",
            "causal_comparison": "native versus natural versus sparse",
            "gold_policy": (
                "Any gold row touching an identity-changed formal-map endpoint "
                "is UNKNOWN and excluded from comparable metrics."
            ),
            "support_is_part_of_relation_state": True,
            "unknown_predictions_are_not_negatives": True,
        },
        "source_artifacts": {
            name: {"path": str(paths[name]), "sha256": source_hashes_before[name]}
            for name in sorted(paths)
        },
        "edge_stream": {
            "frame_count": int(edge_manifest["frame_count"]),
            "input_edge_observations": expected_input_edges,
            "frames_sha256": str(edge_manifest["frames_sha256"]),
        },
        "formal_map_object_count": len(entity_to_index),
        "frozen_gold": {
            "gold_uid": str(gold["gold_uid"]),
            "label_count": len(gold_rows),
            "reference_prediction_count": len(reference_predictions),
            "reference_metrics": reference_metrics,
        },
        "effect_summary": effect_counts,
        "cases": cases,
        "claim_limits": {
            "relation_accuracy_improvement_claimed": False,
            "relation_preservation_on_identity_stable_gold_claimed": True,
            "relation_correctness_on_identity_changed_endpoints_claimed": False,
            "population_generalization_claimed": False,
            "reason": (
                "This slice detects causal relation change and stable-scope "
                "preservation; it contains no new human relation label on the "
                "revised identity endpoints."
            ),
        },
    }
    _write(args.output.resolve(), result)
    if args.audit_output is not None:
        _write(args.audit_output.resolve(), result)
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "checks": checks,
                "effect_summary": effect_counts,
                "reference_metrics": reference_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
