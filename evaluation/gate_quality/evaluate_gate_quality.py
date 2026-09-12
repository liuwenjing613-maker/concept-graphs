from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.observation_gt.association_metrics import (
    ACTION_ATTACH,
    CORRECT_CANONICAL,
    CORRECT_FRAGMENT,
    CORRECT_IDENTITY_RESULTS,
    DETERMINATE_CLEAN_ERRORS,
    DETERMINATE_CLEAN_RESULTS,
    WRONG_FUSION,
    action_from_association,
    action_from_index,
    classify_action,
)
from evaluation.observation_gt.instance_identity import (
    IDENTITY_RELIABLE,
    PredictionIdentityConfig,
    canonical_predictions,
    infer_state_identities,
)

from . import PROTOCOL_VERSION
from .metrics import (
    confusion_cell,
    summarize_detection,
    summarize_error_subtypes,
    summarize_post_gate,
)


ATTACH_DETERMINATE_RESULTS = {
    CORRECT_CANONICAL,
    CORRECT_FRAGMENT,
    WRONG_FUSION,
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_unique(
    rows: Iterable[Mapping[str, Any]], field: str, *, description: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in result:
            raise ValueError(f"duplicate {description}: {key}")
        result[key] = row
    return result


def _gt_class_by_instance(
    labels: Iterable[Mapping[str, Any]],
) -> dict[int, str]:
    votes: dict[int, Counter[str]] = {}
    for label in labels:
        if label.get("quality_status") != "CLEAN":
            continue
        gt_id = label.get("assigned_gt_instance")
        class_name = label.get("top1_gt_class")
        if gt_id is None or class_name is None:
            continue
        votes.setdefault(int(gt_id), Counter())[str(class_name)] += 1
    return {
        gt_id: min(counts, key=lambda item: (-counts[item], item))
        for gt_id, counts in votes.items()
    }


def _candidate_state(
    association: Mapping[str, Any],
    versions_by_uid: Mapping[str, Mapping[str, Any]],
    *,
    current_obs_uid: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    object_uids = [str(item) for item in association.get("object_uids_before") or ()]
    version_uids = [
        str(item) for item in association.get("candidate_object_version_uids") or ()
    ]
    aligned = len(object_uids) == len(version_uids)
    missing = [uid for uid in version_uids if uid not in versions_by_uid]
    mismatches: list[dict[str, str]] = []
    current_member_leaks: list[str] = []
    state: dict[str, tuple[str, ...]] = {}
    if aligned and not missing:
        for object_uid, version_uid in zip(object_uids, version_uids):
            version = versions_by_uid[version_uid]
            version_object_uid = str(version.get("object_uid") or "")
            if version_object_uid != object_uid:
                mismatches.append(
                    {
                        "object_uid": object_uid,
                        "version_uid": version_uid,
                        "version_object_uid": version_object_uid,
                    }
                )
            members = tuple(
                str(item) for item in version.get("member_observation_uids") or ()
            )
            if current_obs_uid in members:
                current_member_leaks.append(object_uid)
            state[object_uid] = tuple(
                item for item in members if item != current_obs_uid
            )
    valid = aligned and not missing and not mismatches and not current_member_leaks
    return state, {
        "object_count_before": len(object_uids),
        "candidate_version_count": len(version_uids),
        "counts_aligned": aligned,
        "missing_object_version_uids": missing,
        "object_version_uid_mismatches": mismatches,
        "current_observation_member_leaks": current_member_leaks,
        "candidate_snapshot_valid": valid,
    }


def build_event_rows(
    associations: Iterable[Mapping[str, Any]],
    gate_events: Iterable[Mapping[str, Any]],
    object_versions: Iterable[Mapping[str, Any]],
    observation_labels: Iterable[Mapping[str, Any]],
    identity_config: PredictionIdentityConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    associations = list(associations)
    gate_events = list(gate_events)
    object_versions = list(object_versions)
    observation_labels = list(observation_labels)
    gates_by_obs = _index_unique(
        gate_events, "current_observation_uid", description="gate observation UID"
    )
    versions_by_uid = _index_unique(
        object_versions, "object_version_uid", description="object version UID"
    )
    labels_by_uid = _index_unique(
        observation_labels, "obs_uid", description="observation label UID"
    )
    gt_classes = _gt_class_by_instance(observation_labels)
    rows: list[dict[str, Any]] = []
    seen_association_obs: set[str] = set()
    duplicate_association_obs: list[str] = []
    snapshot_failures: list[str] = []
    current_member_leak_events: list[str] = []

    for association in sorted(
        associations, key=lambda item: int(item.get("event_sequence", 0))
    ):
        obs_uid = str(association["obs_uid"])
        if obs_uid in seen_association_obs:
            duplicate_association_obs.append(obs_uid)
        seen_association_obs.add(obs_uid)
        label = labels_by_uid.get(obs_uid)
        gate = gates_by_obs.get(obs_uid)
        object_uids = [
            str(item) for item in association.get("object_uids_before") or ()
        ]
        state, snapshot = _candidate_state(
            association, versions_by_uid, current_obs_uid=obs_uid
        )
        if not snapshot["candidate_snapshot_valid"]:
            snapshot_failures.append(obs_uid)
        if snapshot["current_observation_member_leaks"]:
            current_member_leak_events.append(obs_uid)
        identities = infer_state_identities(
            state,
            labels_by_uid,
            identity_config,
            object_uids=object_uids,
        )
        canonical = canonical_predictions(identities)
        if gate is None:
            baseline_action = action_from_association(association)
        else:
            baseline_action = action_from_index(
                gate.get("baseline_match_index"), object_uids
            )
        final_action = action_from_association(association)
        baseline_result = (
            classify_action(label, baseline_action, identities, canonical)
            if label is not None
            else None
        )
        final_result = (
            classify_action(label, final_action, identities, canonical)
            if label is not None
            else None
        )
        baseline_category = str((baseline_result or {}).get("category") or "MISSING")
        final_category = str((final_result or {}).get("category") or "MISSING")
        wrong_fusion_evaluable = baseline_category in ATTACH_DETERMINATE_RESULTS
        wrong_fusion_error = (
            baseline_category == WRONG_FUSION if wrong_fusion_evaluable else None
        )
        clean_action_evaluable = baseline_category in DETERMINATE_CLEAN_RESULTS
        clean_action_error = (
            baseline_category in DETERMINATE_CLEAN_ERRORS
            if clean_action_evaluable
            else None
        )
        final_identity_correct = (
            final_category in CORRECT_IDENTITY_RESULTS
            if final_category in DETERMINATE_CLEAN_RESULTS
            else None
        )
        observation_gt = (
            int(label["assigned_gt_instance"])
            if label is not None and label.get("assigned_gt_instance") is not None
            else None
        )
        target_uid = (
            str(baseline_action.get("object_uid"))
            if baseline_action.get("kind") == ACTION_ATTACH
            else None
        )
        target_identity = identities.get(target_uid) if target_uid is not None else None
        target_gt = (
            int(target_identity["gt_instance"])
            if target_identity is not None
            and target_identity.get("identity_status") == IDENTITY_RELIABLE
            and target_identity.get("gt_instance") is not None
            else None
        )
        wrong_fusion_subtype = None
        if wrong_fusion_error:
            observation_class = (
                gt_classes.get(observation_gt) if observation_gt is not None else None
            )
            target_class = gt_classes.get(target_gt) if target_gt is not None else None
            if observation_class is None or target_class is None:
                wrong_fusion_subtype = "UNKNOWN_CLASS"
            elif observation_class == target_class:
                wrong_fusion_subtype = "SAME_CLASS_DIFFERENT_INSTANCE"
            else:
                wrong_fusion_subtype = "CROSS_CLASS"
        reliable_same_gt_candidates = [
            object_uid
            for object_uid, identity in identities.items()
            if observation_gt is not None
            and identity.get("identity_status") == IDENTITY_RELIABLE
            and identity.get("gt_instance") is not None
            and int(identity["gt_instance"]) == observation_gt
        ]
        trigger = (gate or {}).get("trigger") or {}
        row = {
            "association_event_uid": association.get("event_uid"),
            "event_sequence": association.get("event_sequence"),
            "frame_uid": association.get("frame_uid"),
            "obs_uid": obs_uid,
            "quality_status": (label or {}).get("quality_status"),
            "quality_reason": (label or {}).get("quality_reason"),
            "observation_gt_instance": observation_gt,
            "observation_gt_class": gt_classes.get(observation_gt),
            "candidate_snapshot_validation": snapshot,
            "baseline_action": baseline_action,
            "baseline_result": baseline_result,
            "baseline_target_gt_instance": target_gt,
            "baseline_target_gt_class": gt_classes.get(target_gt),
            "final_action": final_action,
            "final_result": final_result,
            "final_identity_correct": final_identity_correct,
            "gate_triggered": gate is not None,
            "gate_event_id": (gate or {}).get("event_id"),
            "gate_trigger_kind": trigger.get("kind") if gate is not None else None,
            "gate_trigger_reasons": trigger.get("reasons") if gate is not None else [],
            "gate_changed": bool((gate or {}).get("changed")),
            "gate_decision_source": (gate or {}).get("decision_source"),
            "gate_route_reason": (gate or {}).get("route_reason"),
            "association_top1_score": association.get("top1_score"),
            "association_top2_score": association.get("top2_score"),
            "association_margin": association.get("margin"),
            "association_sim_threshold": association.get("sim_threshold"),
            "wrong_fusion_evaluable": wrong_fusion_evaluable,
            "wrong_fusion_error": wrong_fusion_error,
            "wrong_fusion_confusion_cell": (
                confusion_cell(bool(wrong_fusion_error), gate is not None)
                if wrong_fusion_evaluable
                else None
            ),
            "wrong_fusion_subtype": wrong_fusion_subtype,
            "clean_action_evaluable": clean_action_evaluable,
            "clean_action_error": clean_action_error,
            "clean_action_confusion_cell": (
                confusion_cell(bool(clean_action_error), gate is not None)
                if clean_action_evaluable
                else None
            ),
            "reliable_same_gt_candidate_uids": sorted(reliable_same_gt_candidates),
            "correct_candidate_available": bool(reliable_same_gt_candidates),
        }
        rows.append(row)

    unmatched_gate_obs = sorted(set(gates_by_obs) - seen_association_obs)
    audit = {
        "association_count": len(associations),
        "unique_association_observation_count": len(seen_association_obs),
        "duplicate_association_observation_count": len(duplicate_association_obs),
        "duplicate_association_observation_uids": duplicate_association_obs[:100],
        "gate_event_count": len(gate_events),
        "matched_gate_event_count": len(set(gates_by_obs) & seen_association_obs),
        "unmatched_gate_event_count": len(unmatched_gate_obs),
        "unmatched_gate_observation_uids": unmatched_gate_obs[:100],
        "observation_label_count": len(observation_labels),
        "association_without_label_count": sum(
            row["quality_status"] is None for row in rows
        ),
        "candidate_snapshot_validation_failure_count": len(snapshot_failures),
        "candidate_snapshot_validation_failure_observation_uids": snapshot_failures[:100],
        "current_observation_member_leak_count": len(current_member_leak_events),
        "current_observation_member_leak_uids": current_member_leak_events[:100],
    }
    return rows, audit


def gate_log_audit(run_dir: Path, gate_event_count: int) -> dict[str, Any]:
    summary_path = run_dir / "blocking_association_gate/summary.json"
    if not summary_path.is_file():
        return {
            "summary_exists": False,
            "status": None,
            "processed": None,
            "event_count": gate_event_count,
            "complete_for_absence_as_not_triggered": False,
        }
    summary = read_json(summary_path)
    processed = (summary.get("stats") or {}).get("processed")
    complete = summary.get("status") == "completed" and processed == gate_event_count
    return {
        "summary_exists": True,
        "summary_path": str(summary_path),
        "status": summary.get("status"),
        "processed": processed,
        "event_count": gate_event_count,
        "complete_for_absence_as_not_triggered": bool(complete),
    }


def build_metrics(rows: list[Mapping[str, Any]], audit: Mapping[str, Any]) -> dict[str, Any]:
    trigger_counts = Counter(
        str(row.get("gate_trigger_kind") or "NOT_TRIGGERED") for row in rows
    )
    baseline_categories = Counter(
        str((row.get("baseline_result") or {}).get("category") or "MISSING")
        for row in rows
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "wrong_fusion_detection": summarize_detection(
            rows,
            evaluable_field="wrong_fusion_evaluable",
            error_field="wrong_fusion_error",
        ),
        "wrong_fusion_post_gate": summarize_post_gate(
            rows,
            evaluable_field="wrong_fusion_evaluable",
            error_field="wrong_fusion_error",
        ),
        "wrong_fusion_error_recall_by_subtype": summarize_error_subtypes(rows),
        "all_clean_action_error_detection": summarize_detection(
            rows,
            evaluable_field="clean_action_evaluable",
            error_field="clean_action_error",
        ),
        "all_clean_action_post_gate": summarize_post_gate(
            rows,
            evaluable_field="clean_action_evaluable",
            error_field="clean_action_error",
        ),
        "gate_trigger_kind_counts_over_all_associations": dict(sorted(trigger_counts.items())),
        "baseline_result_category_counts": dict(sorted(baseline_categories.items())),
        "audit": dict(audit),
    }


def write_confusion_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    scopes = (
        ("wrong_fusion_detection", metrics["wrong_fusion_detection"]),
        ("all_clean_action_error_detection", metrics["all_clean_action_error_detection"]),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scope", "cell", "count"))
        writer.writeheader()
        for scope, summary in scopes:
            for cell, count in summary["confusion_matrix"].items():
                writer.writerow({"scope": scope, "cell": cell, "count": count})


def run(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    observation_gt_dir = args.observation_gt_dir.resolve()
    output = (args.out or (run_dir / "gate_quality_metrics")).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}; choose a fresh --out"
        )
    associations_path = run_dir / "evidence/associations.jsonl"
    versions_path = run_dir / "evidence/object_versions.jsonl"
    gates_path = run_dir / "blocking_association_gate/events.jsonl"
    labels_path = observation_gt_dir / "observation_labels.jsonl"
    required = (associations_path, versions_path, gates_path, labels_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    observation_protocol_path = observation_gt_dir / "protocol.json"
    if observation_protocol_path.is_file():
        observation_protocol = read_json(observation_protocol_path)
        if observation_protocol.get("incomplete_smoke_run"):
            raise ValueError("refusing incomplete observation-GT smoke output")
    else:
        observation_protocol = None
    associations = read_jsonl(associations_path)
    object_versions = read_jsonl(versions_path)
    gate_events = read_jsonl(gates_path)
    labels = read_jsonl(labels_path)
    log_audit = gate_log_audit(run_dir, len(gate_events))
    if (
        not log_audit["complete_for_absence_as_not_triggered"]
        and not args.allow_incomplete_gate_log
    ):
        raise ValueError(
            "gate summary is not complete, so absence from events.jsonl cannot be "
            "interpreted as not-triggered; pass --allow-incomplete-gate-log only for diagnostics"
        )
    identity_config = PredictionIdentityConfig(
        min_prediction_identity_purity=args.min_prediction_identity_purity
    )
    identity_config.validate()
    rows, event_audit = build_event_rows(
        associations,
        gate_events,
        object_versions,
        labels,
        identity_config,
    )
    audit = {**event_audit, "gate_log": log_audit}
    metrics = build_metrics(rows, audit)
    staging = output.with_name(output.name + f".building-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    try:
        write_jsonl(staging / "gate_quality_events.jsonl", rows)
        write_jsonl(
            staging / "missed_wrong_fusions.jsonl",
            [
                row
                for row in rows
                if row.get("wrong_fusion_evaluable")
                and row.get("wrong_fusion_error")
                and not row.get("gate_triggered")
            ],
        )
        write_jsonl(
            staging / "caught_wrong_fusions.jsonl",
            [
                row
                for row in rows
                if row.get("wrong_fusion_evaluable")
                and row.get("wrong_fusion_error")
                and row.get("gate_triggered")
            ],
        )
        write_jsonl(
            staging / "false_interventions.jsonl",
            [
                row
                for row in rows
                if row.get("wrong_fusion_evaluable")
                and not row.get("wrong_fusion_error")
                and row.get("gate_triggered")
            ],
        )
        write_confusion_csv(staging / "confusion_matrix.csv", metrics)
        write_json(staging / "metrics.json", metrics)
        protocol = {
            "protocol": PROTOCOL_VERSION,
            "run_dir": str(run_dir),
            "observation_gt_dir": str(observation_gt_dir),
            "observation_gt_protocol": (
                (observation_protocol or {}).get("protocol")
            ),
            "evaluation_unit": "all association proposals",
            "gate_trigger_definition": (
                "current_observation_uid is present in blocking_association_gate/events.jsonl"
            ),
            "primary_positive_definition": (
                "baseline pre-gate ATTACH is WRONG_FUSION under causal prediction identity"
            ),
            "primary_negative_definition": (
                "baseline pre-gate ATTACH is CORRECT_CANONICAL or CORRECT_FRAGMENT"
            ),
            "prediction_identity": (
                "weighted vote over CLEAN members in candidate object version before event; "
                "current observation is explicitly excluded"
            ),
            "min_prediction_identity_purity": args.min_prediction_identity_purity,
            "absence_as_not_triggered_validated": log_audit[
                "complete_for_absence_as_not_triggered"
            ],
            "allow_incomplete_gate_log": bool(args.allow_incomplete_gate_log),
            "input_fingerprints": {
                "associations": sha256_file(associations_path),
                "object_versions": sha256_file(versions_path),
                "gate_events": sha256_file(gates_path),
                "observation_labels": sha256_file(labels_path),
            },
            "code_fingerprints": {
                "evaluate_gate_quality.py": sha256_file(Path(__file__)),
                "metrics.py": sha256_file(Path(__file__).with_name("metrics.py")),
            },
        }
        write_json(staging / "protocol.json", protocol)
        staging.replace(output)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic-gate error detection over all association proposals"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observation-gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-prediction-identity-purity", type=float, default=0.80)
    parser.add_argument(
        "--allow-incomplete-gate-log",
        action="store_true",
        help="diagnostic only; permits treating absent gate events as not triggered",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
