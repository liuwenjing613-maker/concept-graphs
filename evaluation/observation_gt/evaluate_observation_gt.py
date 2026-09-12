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

import numpy as np
from PIL import Image

from conceptgraph.revision.counterfactual_projection import ProjectionEvidenceLoader

from . import PROTOCOL_VERSION
from .association_metrics import (
    action_from_association,
    action_from_index,
    classify_action,
    classify_vlm_staged_decision,
    summarize_decisions,
    summarize_transitions,
)
from .instance_identity import (
    PredictionIdentityConfig,
    canonical_predictions,
    infer_state_identities,
)
from .observation_labeler import (
    GTReference,
    ObservationGTLabeler,
    ObservationLabelConfig,
    aggregate_projection_diagnostics,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    default=_json_default,
                )
                + "\n"
            )


def load_config(path: Path | None) -> dict[str, Any]:
    return dict(read_json(path)) if path is not None else {}


def load_reference(reference_dir: Path) -> tuple[GTReference, dict[str, Any], Path]:
    manifest_path = reference_dir / "manifest.json"
    reference_path = reference_dir / "reference.npz"
    manifest = dict(read_json(manifest_path))
    actual_hash = sha256_file(reference_path)
    expected_hash = str(manifest.get("reference_sha256") or "")
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"GT reference hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return GTReference.load(reference_path, manifest), manifest, reference_path


def observation_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["obs_uid"]): row
        for row in read_jsonl(run_dir / "evidence/observations.jsonl")
    }


def render_overlay(rgb_path: Path, mask: np.ndarray, output: Path) -> None:
    image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    if image.shape[:2] != mask.shape:
        raise ValueError(f"RGB/mask shape mismatch for diagnostic {rgb_path}")
    overlay = image.copy()
    color = np.asarray([255.0, 48.0, 48.0], dtype=np.float32)
    overlay[mask] = 0.58 * image[mask] + 0.42 * color
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output)


def label_observations(
    run_dir: Path,
    reference: GTReference,
    config: ObservationLabelConfig,
    *,
    verify_hashes: bool,
    max_frames: int | None,
    diagnostics_dir: Path,
    diagnostic_samples_per_status: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loader = ProjectionEvidenceLoader(run_dir, verify_hashes=verify_hashes)
    metadata = observation_metadata(run_dir)
    labeler = ObservationGTLabeler(reference, config)
    frames = sorted(loader.frames.values(), key=lambda row: int(row["frame_idx"]))
    if max_frames is not None:
        frames = frames[:max_frames]
    labels: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_counts: Counter[str] = Counter()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for frame_number, frame_row in enumerate(frames, 1):
        frame = loader.load_frame(str(frame_row["frame_uid"]))
        for obs_uid, mask in zip(frame.observed_mask_uids, frame.observed_masks):
            meta = metadata.get(obs_uid, {})
            label = labeler.label_mask(
                obs_uid=obs_uid,
                frame_uid=frame.frame_uid,
                class_name=meta.get("class_name"),
                mask=mask,
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                pose_camera_to_world=frame.pose,
            )
            label["frame_index"] = int(frame.frame_index)
            label["source_frame_id"] = frame_row.get("source_frame_id")
            label["filtered_det_idx"] = meta.get("filtered_det_idx")
            label["observation_status"] = meta.get("status")
            labels.append(label)
            status = str(label["quality_status"])
            if diagnostic_counts[status] < diagnostic_samples_per_status:
                status_dir = diagnostics_dir / status.lower()
                status_dir.mkdir(parents=True, exist_ok=True)
                output = status_dir / f"{obs_uid}.png"
                render_overlay(frame.rgb_path, mask, output)
                diagnostic_counts[status] += 1
                diagnostic_rows.append(
                    {
                        "obs_uid": obs_uid,
                        "quality_status": status,
                        "quality_reason": label["quality_reason"],
                        "overlay_path": str(output.relative_to(diagnostics_dir.parent)),
                        "top1_gt_instance": label["top1_gt_instance"],
                        "top1_gt_class": label["top1_gt_class"],
                        "top1_purity": label["top1_purity"],
                        "gt_support_ratio": label["gt_support_ratio"],
                    }
                )
        if frame_number % 25 == 0 or frame_number == len(frames):
            print(
                f"labelled frames {frame_number}/{len(frames)}; observations={len(labels)}",
                flush=True,
            )
    return labels, diagnostic_rows


def _action_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return dict(first) == dict(second)


def evaluate_gate_events(
    run_dir: Path,
    labels_by_uid: Mapping[str, Mapping[str, Any]],
    identity_config: PredictionIdentityConfig,
    *,
    maximum_frame_index: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    associations = read_jsonl(run_dir / "evidence/associations.jsonl")
    mapping_events = read_jsonl(run_dir / "evidence/mapping_events.jsonl")
    object_versions = read_jsonl(run_dir / "evidence/object_versions.jsonl")
    versions_by_uid = {
        str(row["object_version_uid"]): row for row in object_versions
    }
    if len(versions_by_uid) != len(object_versions):
        raise ValueError("duplicate object_version_uid in object_versions.jsonl")
    gate_path = run_dir / "blocking_association_gate/events.jsonl"
    gate_root = gate_path.parent
    gate_events = read_jsonl(gate_path) if gate_path.is_file() else []
    gate_by_obs: dict[str, dict[str, Any]] = {}
    for event in gate_events:
        obs_uid = str(event["current_observation_uid"])
        if obs_uid in gate_by_obs:
            raise ValueError(f"multiple gate events for observation {obs_uid}")
        gate_by_obs[obs_uid] = event
    merge_decisions = [
        read_json(path)
        for path in sorted(
            (gate_root / "vlm_instance_merge" / "events").glob("*/decision.json")
        )
    ]
    merge_by_parent: dict[str, list[dict[str, Any]]] = {}
    merge_by_pair: dict[str, list[dict[str, Any]]] = {}
    for merge in merge_decisions:
        parent = str(merge.get("parent_event") or "")
        pair_key = str(merge.get("pair_key") or "")
        if parent:
            merge_by_parent.setdefault(parent, []).append(merge)
        if pair_key:
            merge_by_pair.setdefault(pair_key, []).append(merge)
    event_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    snapshot_validation_failures: list[dict[str, Any]] = []
    final_action_mismatches: list[str] = []
    evaluated_gate_uids: set[str] = set()
    missing_staged_decisions: list[str] = []

    for row in sorted(associations, key=lambda item: int(item["event_sequence"])):
        if maximum_frame_index is not None and _frame_index(row) > maximum_frame_index:
            continue
        obs_uid = str(row["obs_uid"])
        gate = gate_by_obs.get(obs_uid)
        if gate is None:
            continue
        label = labels_by_uid.get(obs_uid)
        if label is None:
            continue
        object_uids = [str(item) for item in row.get("object_uids_before") or ()]
        version_uids = [
            str(item) for item in row.get("candidate_object_version_uids") or ()
        ]
        missing_versions = [uid for uid in version_uids if uid not in versions_by_uid]
        aligned = len(object_uids) == len(version_uids)
        object_version_mismatches = []
        state: dict[str, tuple[str, ...]] = {}
        if aligned and not missing_versions:
            for object_uid, version_uid in zip(object_uids, version_uids):
                version = versions_by_uid[version_uid]
                if str(version.get("object_uid")) != object_uid:
                    object_version_mismatches.append(
                        {
                            "object_uid": object_uid,
                            "version_uid": version_uid,
                            "version_object_uid": version.get("object_uid"),
                        }
                    )
                state[object_uid] = tuple(
                    str(item)
                    for item in version.get("member_observation_uids") or ()
                )
        validation = {
            "object_count_before": len(object_uids),
            "candidate_version_count": len(version_uids),
            "counts_aligned": aligned,
            "missing_object_version_uids": missing_versions,
            "object_version_uid_mismatches": object_version_mismatches,
            "candidate_snapshot_valid": (
                aligned and not missing_versions and not object_version_mismatches
            ),
            "association_object_count_after_premerge": len(
                row.get("association_object_uids_before") or object_uids
            ),
        }
        if not validation["candidate_snapshot_valid"]:
            snapshot_validation_failures.append(
                {"obs_uid": obs_uid, "event_uid": row.get("event_uid"), **validation}
            )
        identities = infer_state_identities(
            state,
            labels_by_uid,
            identity_config,
            object_uids=object_uids,
        )
        canonical = canonical_predictions(identities)
        candidate_object_uids = []
        candidate_alias_to_object_uid: dict[str, str] = {}
        invalid_candidate_indices = []
        for alias, index_value in sorted(
            (gate.get("candidate_alias_to_object_index") or {}).items()
        ):
            try:
                candidate_index = int(index_value)
            except (TypeError, ValueError):
                invalid_candidate_indices.append(
                    {"alias": str(alias), "index": index_value}
                )
                continue
            if 0 <= candidate_index < len(object_uids):
                candidate_uid = object_uids[candidate_index]
                candidate_object_uids.append(candidate_uid)
                candidate_alias_to_object_uid[str(alias)] = candidate_uid
            else:
                invalid_candidate_indices.append(
                    {"alias": str(alias), "index": candidate_index}
                )
        baseline_action = action_from_index(gate.get("baseline_match_index"), object_uids)
        gate_final_action = action_from_index(gate.get("final_match_index"), object_uids)
        actual_action = action_from_association(row)
        final_matches_actual = _action_equal(gate_final_action, actual_action)
        if not final_matches_actual:
            final_action_mismatches.append(obs_uid)
        final_action = actual_action
        baseline_result = classify_action(
            label,
            baseline_action,
            identities,
            canonical,
            candidate_object_uids,
        )
        execution_result = classify_action(
            label,
            final_action,
            identities,
            canonical,
            candidate_object_uids,
        )
        gate_event_id = str(gate.get("event_id") or "")
        staged_path = gate_root / "events" / gate_event_id / "staged_decision.json"
        if staged_path.is_file():
            staged_decision = dict(read_json(staged_path))
        else:
            staged_decision = {}
            missing_staged_decisions.append(gate_event_id)
        final_result = classify_vlm_staged_decision(
            label,
            staged_decision,
            identities,
            canonical,
            candidate_alias_to_object_uid,
            execution_result,
        )
        linked_merge_reviews = []
        for merge in merge_by_parent.get(gate_event_id, []):
            pair_key = str(merge.get("pair_key") or "")
            lifecycle = sorted(
                merge_by_pair.get(pair_key, []),
                key=lambda item: (int(item.get("frame_idx") or -1), str(item.get("event_id"))),
            )
            executed = next(
                (item for item in lifecycle if item.get("execution") == "MERGED"),
                None,
            )
            linked_merge_reviews.append(
                {
                    "event_id": merge.get("event_id"),
                    "pair_key": pair_key,
                    "object_uids": [
                        (merge.get("object_A") or {}).get("object_uid"),
                        (merge.get("object_B") or {}).get("object_uid"),
                    ],
                    "model_choice": (merge.get("model_output") or {}).get("choice"),
                    "execution_at_review": merge.get("execution"),
                    "eventually_merged": executed is not None,
                    "executed_event_id": executed.get("event_id") if executed else None,
                    "executed_frame_idx": executed.get("frame_idx") if executed else None,
                    "executed_decision_source": (
                        executed.get("decision_source") if executed else None
                    ),
                }
            )
        event_row = {
            "obs_uid": obs_uid,
            "association_event_uid": row.get("event_uid"),
            "event_sequence": row.get("event_sequence"),
            "frame_uid": row.get("frame_uid"),
            "frame_index": _frame_index(row),
            "quality_status": label.get("quality_status"),
            "quality_reason": label.get("quality_reason"),
            "observation_gt_instance": label.get("assigned_gt_instance"),
            "observation_top1_gt_instance": label.get("top1_gt_instance"),
            "observation_top1_purity": label.get("top1_purity"),
            "object_count_before": len(object_uids),
            "candidate_snapshot_validation": validation,
            "candidate_alias_to_object_index": gate.get("candidate_alias_to_object_index"),
            "candidate_alias_to_object_uid": candidate_alias_to_object_uid,
            "candidate_object_uids": candidate_object_uids,
            "invalid_candidate_indices": invalid_candidate_indices,
            "baseline_action": baseline_action,
            "gate_final_requested_action": gate_final_action,
            "final_action": final_action,
            "actual_action": actual_action,
            "final_action_matches_association": final_matches_actual,
            "baseline_result": baseline_result,
            "execution_result": execution_result,
            "final_result": final_result,
            "vlm_staged_decision": {
                key: staged_decision.get(key)
                for key in (
                    "kind",
                    "reason_code",
                    "target_alias",
                    "same_aliases",
                    "choice",
                    "merge_execution",
                )
                if key in staged_decision
            },
            "linked_merge_reviews": linked_merge_reviews,
            "changed": bool(gate.get("changed")),
            "decision_source": gate.get("decision_source"),
            "route_reason": gate.get("route_reason"),
            "trigger": gate.get("trigger"),
            "canonical_prediction_by_gt": {
                str(gt_id): uid for gt_id, uid in sorted(canonical.items())
            },
        }
        event_rows.append(event_row)
        evaluated_gate_uids.add(obs_uid)
        for object_uid, identity in identities.items():
            identity_rows.append(
                {
                    "obs_uid": obs_uid,
                    "association_event_uid": row.get("event_uid"),
                    "event_sequence": row.get("event_sequence"),
                    "is_canonical_for_dominant_gt": (
                        identity.get("gt_instance") is not None
                        and canonical.get(int(identity["gt_instance"])) == object_uid
                    ),
                    **identity,
                }
            )
    eligible_gate_uids = {
        obs_uid
        for obs_uid, gate in gate_by_obs.items()
        if maximum_frame_index is None
        or int((gate.get("timeline") or {}).get("c_frame", maximum_frame_index + 1))
        <= maximum_frame_index
    }
    missing_gates = sorted(eligible_gate_uids - evaluated_gate_uids)
    audit = {
        "association_count": len(associations),
        "mapping_event_count": len(mapping_events),
        "object_version_count": len(object_versions),
        "gate_event_count": len(gate_events),
        "evaluated_gate_event_count": len(event_rows),
        "missing_gate_observation_uids": missing_gates,
        "candidate_snapshot_validation_failure_count": len(snapshot_validation_failures),
        "candidate_snapshot_validation_failures": snapshot_validation_failures[:100],
        "final_action_mismatch_count": len(final_action_mismatches),
        "final_action_mismatch_observation_uids": final_action_mismatches[:100],
        "missing_staged_decision_count": len(missing_staged_decisions),
        "missing_staged_decision_event_ids": missing_staged_decisions[:100],
        "merge_review_decision_count": sum(
            str((row.get("vlm_staged_decision") or {}).get("kind") or "")
            == "MERGE_REVIEW"
            for row in event_rows
        ),
    }
    return event_rows, identity_rows, audit


def _frame_index(row: Mapping[str, Any]) -> int:
    frame_uid = str(row.get("frame_uid") or "")
    try:
        return int(frame_uid.rsplit("_f", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse frame index from {frame_uid!r}") from exc


def summarize_label_quality(labels: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    labels = list(labels)
    counts = Counter(str(row["quality_status"]) for row in labels)
    reasons = Counter(str(row["quality_reason"]) for row in labels)
    total = len(labels)
    return {
        "observation_count": total,
        "status_counts": dict(sorted(counts.items())),
        "status_rates": {
            key: value / total if total else None for key, value in sorted(counts.items())
        },
        "reason_counts": dict(sorted(reasons.items())),
        "scorable_observation_rate": (
            (counts["CLEAN"] + counts["MIXED"]) / total if total else None
        ),
    }


def summarize_by_field(
    rows: Iterable[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(field) or "MISSING"), []).append(row)
    return {
        key: {
            "event_count": len(group),
            "baseline": summarize_decisions(group, "baseline_result"),
            "final": summarize_decisions(group, "final_result"),
            "transitions": summarize_transitions(group),
        }
        for key, group in sorted(groups.items())
    }


def build_threshold_sensitivity(
    labels: Iterable[Mapping[str, Any]], config: ObservationLabelConfig
) -> dict[str, Any]:
    rows = list(labels)
    grid = []
    for purity in (0.70, 0.80, 0.90):
        for minimum_points in (25, 50, 100):
            clean_count = 0
            for row in rows:
                if float(row.get("valid_depth_ratio", 0.0)) < config.min_valid_depth_ratio:
                    continue
                if int(row.get("matched_gt_points", 0)) < minimum_points:
                    continue
                if float(row.get("gt_support_ratio", 0.0)) < config.min_gt_support_ratio:
                    continue
                if float(row.get("top1_purity", 0.0)) < purity:
                    continue
                if float(row.get("top2_purity", 0.0)) > config.max_top2_purity:
                    continue
                if float(row.get("purity_margin", 0.0)) < config.min_purity_margin:
                    continue
                if int(row.get("core_matched_gt_points", 0)) < config.min_core_matched_gt_points:
                    continue
                if not row.get("core_agrees_with_full_mask"):
                    continue
                clean_count += 1
            grid.append(
                {
                    "min_top1_purity": purity,
                    "min_matched_gt_points": minimum_points,
                    "clean_count": clean_count,
                    "clean_rate_over_all_observations": clean_count / len(rows) if rows else None,
                }
            )
    return {
        "fixed_gt_match_distance_m": config.gt_match_distance_m,
        "purity_and_minimum_point_grid": grid,
        "distance_sensitivity_status": "requires independent reruns",
        "recommended_distance_values_m": [0.03, 0.05, 0.07],
    }


def code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "__init__.py",
        "observation_labeler.py",
        "instance_identity.py",
        "association_metrics.py",
        "evaluate_observation_gt.py",
    )
    return {name: sha256_file(root / name) for name in names}


def collection_fingerprint(paths: Iterable[Path], root: Path) -> dict[str, Any]:
    paths = sorted(path for path in paths if path.is_file())
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(paths),
        "combined_sha256": digest.hexdigest(),
    }


def input_fingerprint(run_dir: Path, reference_dir: Path) -> dict[str, Any]:
    paths = {
        "reference_npz": reference_dir / "reference.npz",
        "reference_manifest": reference_dir / "manifest.json",
        "config_params": run_dir / "config_params.json",
        "frames": run_dir / "evidence/frames.jsonl",
        "observations": run_dir / "evidence/observations.jsonl",
        "associations": run_dir / "evidence/associations.jsonl",
        "mapping_events": run_dir / "evidence/mapping_events.jsonl",
        "object_versions": run_dir / "evidence/object_versions.jsonl",
        "gate_events": run_dir / "blocking_association_gate/events.jsonl",
    }
    result = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
        if path.is_file()
    }
    gate_root = run_dir / "blocking_association_gate"
    result["staged_decisions"] = collection_fingerprint(
        (gate_root / "events").glob("*/staged_decision.json"), gate_root
    )
    result["instance_merge_decisions"] = collection_fingerprint(
        (gate_root / "vlm_instance_merge" / "events").glob("*/decision.json"),
        gate_root,
    )
    return result


def write_transition_csv(path: Path, transitions: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("baseline", "final", "count"))
        writer.writeheader()
        writer.writerows(transitions)


def run(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    output = (args.out or (run_dir / "observation_gt_metrics")).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}; choose a fresh --out"
        )
    staging = output.with_name(output.name + f".building-{uuid.uuid4().hex[:8]}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        raw_config = load_config(args.config)
        label_config = ObservationLabelConfig.from_mapping(raw_config)
        identity_config = PredictionIdentityConfig(
            min_prediction_identity_purity=float(
                raw_config.get("min_prediction_identity_purity", 0.80)
            )
        )
        identity_config.validate()
        verify_hashes = bool(raw_config.get("verify_evidence_hashes", True))
        diagnostic_count = int(raw_config.get("diagnostic_samples_per_status", 8))
        reference, manifest, reference_path = load_reference(reference_dir)
        labels, diagnostic_rows = label_observations(
            run_dir,
            reference,
            label_config,
            verify_hashes=verify_hashes,
            max_frames=args.max_frames,
            diagnostics_dir=staging / "diagnostics",
            diagnostic_samples_per_status=diagnostic_count,
        )
        labels_by_uid = {str(row["obs_uid"]): row for row in labels}
        if len(labels_by_uid) != len(labels):
            raise ValueError("duplicate observation labels")
        max_frame_index = (
            max((int(row["frame_index"]) for row in labels), default=-1)
            if args.max_frames is not None
            else None
        )
        event_rows, identity_rows, event_audit = evaluate_gate_events(
            run_dir,
            labels_by_uid,
            identity_config,
            maximum_frame_index=max_frame_index,
        )
        label_summary = summarize_label_quality(labels)
        projection_summary = aggregate_projection_diagnostics(labels)
        baseline_summary = summarize_decisions(event_rows, "baseline_result")
        final_summary = summarize_decisions(event_rows, "final_result")
        transition_summary = summarize_transitions(event_rows)
        threshold_sensitivity = build_threshold_sensitivity(labels, label_config)
        metrics = {
            "protocol": PROTOCOL_VERSION,
            "incomplete_smoke_run": args.max_frames is not None,
            "primary_denominator": {
                "name": "all_vlm_gate_events",
                "event_count": len(event_rows),
                "policy": (
                    "baseline and final primary rates use the same complete gate-event "
                    "set; AMBIGUOUS and UNSCORABLE remain explicit outcome buckets"
                ),
            },
            "label_quality": label_summary,
            "projection_diagnostics": projection_summary,
            "baseline": baseline_summary,
            "final": final_summary,
            "vlm_transitions": transition_summary,
            "by_decision_source": summarize_by_field(event_rows, "decision_source"),
            "by_route_reason": summarize_by_field(event_rows, "route_reason"),
            "event_audit": event_audit,
        }
        protocol = {
            "protocol": PROTOCOL_VERSION,
            "run_dir": str(run_dir),
            "reference_dir": str(reference_dir),
            "scene": manifest.get("scene"),
            "pose_convention": "frame pose is camera-to-world",
            "projection": "processed mask pixels + depth -> camera -> world; exact GT cKDTree k=1",
            "distance_acceptance": "strictly less than gt_match_distance_m",
            "prediction_identity": "causal weighted vote over CLEAN member observations before the event",
            "sampling": "deterministic evenly spaced valid mask pixels up to max_sampled_points_per_mask",
            "observation_scope": (
                "all detections are classified as CLEAN, MIXED, or UNSCORABLE; "
                "every instance declared by the reference participates, including "
                "wall, ceiling, floor, and other"
            ),
            "instance_id_policy": (
                "instance validity comes from manifest instance_id_to_semantic_id; "
                "native instance id 0 is supported and no class*1000 encoding is assumed"
            ),
            "unlabeled_gt_policy": (
                "matched reference points without a declared instance contribute to "
                "GT support and the purity denominator but cannot be assigned an identity"
            ),
            "primary_metric_denominator": (
                "all VLM gate events; baseline and final use the identical event set"
            ),
            "primary_outcomes": {
                "SUCCESS": (
                    "correct CLEAN identity/merge-review intent or correct MIXED discard"
                ),
                "FAILURE": (
                    "wrong CLEAN identity/discard action or retained MIXED observation"
                ),
                "AMBIGUOUS": "CLEAN event whose action target/newness cannot be resolved",
                "UNSCORABLE": "observation lacks reliable GT/depth evidence",
                "UNRESOLVED": "action or category cannot be normalized",
            },
            "new_action_ambiguity_policy": (
                "CLEAN NEW is AMBIGUOUS_NEW only when a shown gate candidate has "
                "UNKNOWN/MIXED prediction identity; unrelated uncertain predictions "
                "elsewhere in the map are diagnostics only. A reliable same-GT "
                "prediction always makes NEW a duplicate, with shown-vs-unshown "
                "source recorded in new_action_context."
            ),
            "merge_review_policy": (
                "When staged_decision.kind is MERGE_REVIEW, score the semantic SAME "
                "claim over same_aliases against causal candidate GT identities. A "
                "temporary DISCARD while pair approval is pending is preserved as an "
                "execution fallback diagnostic, not classified as FALSE_DISCARD. For "
                "MIXED/UNSCORABLE observations, retain the observation-action category "
                "and expose MERGE_REVIEW as a parallel diagnostic."
            ),
            "subgroup_denominators": (
                "CLEAN identity metrics use all CLEAN gate events; MIXED discard recall "
                "uses all MIXED gate events; diagnostic route tables use their named subsets"
            ),
            "observation_label_config": label_config.as_dict(),
            "prediction_identity_config": {
                "min_prediction_identity_purity": identity_config.min_prediction_identity_purity
            },
            "verify_evidence_hashes": verify_hashes,
            "max_frames": args.max_frames,
            "incomplete_smoke_run": args.max_frames is not None,
            "reference_manifest_protocol": manifest.get("protocol"),
            "reference_point_count": int(len(reference.xyz)),
            "reference_sha256": sha256_file(reference_path),
            "code_fingerprint": code_fingerprint(),
            "input_fingerprint": input_fingerprint(run_dir, reference_dir),
        }
        write_jsonl(staging / "observation_labels.jsonl", labels)
        write_jsonl(staging / "prediction_identities.jsonl", identity_rows)
        write_jsonl(staging / "event_decisions.jsonl", event_rows)
        write_jsonl(staging / "diagnostics/index.jsonl", diagnostic_rows)
        write_json(staging / "metrics.json", metrics)
        write_json(staging / "protocol.json", protocol)
        write_json(staging / "threshold_sensitivity.json", threshold_sensitivity)
        write_transition_csv(
            staging / "transition_matrix.csv",
            transition_summary["transition_counts"],
        )
        staging.replace(output)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return output
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causal observation-to-GT and VLM association evaluation"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="smoke-test only; output is explicitly marked incomplete",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    output = run(args)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
