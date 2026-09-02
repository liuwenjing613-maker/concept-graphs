#!/usr/bin/env python3
"""Compile human-confirmed room0 routing errors into causal/future-evidence episodes."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ERROR_LABELS = {
    "WRONG_ATTACH_EXISTING",
    "SHOULD_HAVE_BEEN_NEW",
    "WRONG_NEW_FALSE_SPLIT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-packet-root", type=Path, required=True)
    parser.add_argument("--large-events", type=Path, required=True)
    parser.add_argument("--r2-packet-root", type=Path, required=True)
    parser.add_argument("--r2-overlay", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--observation-gt", type=Path, required=True)
    parser.add_argument("--routing-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--future-window", type=int, default=30)
    parser.add_argument("--view-angle-deg", type=float, default=15.0)
    parser.add_argument("--translation-scale", type=float, default=0.5)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def frame_index(frame_uid: str) -> int:
    return int(frame_uid.rsplit("_f", 1)[1])


def reliable_gt(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    return bool(
        row.get("gt_assignment_eligible")
        and row.get("gt_top_id") is not None
        and float(row.get("gt_purity") or 0) >= 0.90
        and float(row.get("gt_supported_fraction") or 0) >= 0.90
        and int(row.get("gt_top_pixels") or 0) >= 25
    )


def compile_r2_base(r2_root: Path, overlay_path: Path) -> list[dict[str, Any]]:
    worklist = read_jsonl(r2_root / "worklist.jsonl")
    private_rows = {
        str(row["case_uid"]): row
        for row in read_jsonl(r2_root / "private_v2_r2_worklist.jsonl")
    }
    labels = {
        str(row["case_uid"]): row
        for row in read_jsonl(r2_root / "labels" / "event_labels.jsonl")
    }
    overlays = {
        str(row["case_uid"]): row
        for row in read_jsonl(overlay_path)
    }
    rows: list[dict[str, Any]] = []
    for work in worklist:
        if work.get("repeat_of"):
            continue
        case_uid = str(work["case_uid"])
        label = dict(labels[case_uid])
        adjudication = overlays.get(case_uid)
        if adjudication:
            label.update(adjudication["adjudicated"])
        private = private_rows[case_uid]
        case_private = read_json(Path(str(work["case_dir"])) / "case_private.json")
        code_to_uid = {
            str(candidate["code"]): str(candidate["object_uid"])
            for candidate in case_private["candidates"]
        }
        human = label["derived"]
        legal_uids = sorted(
            {
                code_to_uid[code]
                for code in human["legal_target_codes_shown"]
            }
            | set(human["legal_target_uids_outside"])
        )
        rows.append({
            "schema_version": "experiment0-human-routing-event/1.0",
            "source_batch": "V2_R2_ADJUDICATED_CALIBRATION",
            "adjudicated": bool(adjudication),
            "case_uid": case_uid,
            "repeat_of": None,
            "event_uid": label["event_uid"],
            "obs_uid": case_private["obs_uid"],
            "scene": label["scene"],
            "source_frame": label["source_frame"],
            "event_frame_idx": label["event_frame_idx"],
            "sample_kind": work.get("sample_kind"),
            "queue_memberships": ["CALIBRATION_R2"],
            "original_action_type": case_private["original_action_type"],
            "original_target_code": case_private.get("original_target_code"),
            "original_target_uid": case_private.get("original_target_uid"),
            "created_object_uid": case_private.get("created_object_uid"),
            "frozen_candidates": [
                {
                    "code": candidate["code"],
                    "object_uid": candidate["object_uid"],
                    "object_version_uid": candidate["object_version_uid"],
                    "aggregate_score": candidate.get("aggregate_score"),
                    "spatial_score": candidate.get("spatial_score"),
                    "visual_score": candidate.get("visual_score"),
                }
                for candidate in case_private["candidates"]
            ],
            "human": {
                "observation_quality": label["blind"]["observation_quality"],
                "matching_candidate_codes": label["blind"]["matching_candidate_codes"],
                "identity_evidence_status": label["blind"]["identity_evidence_status"],
                "physical_instance_note": label["blind"].get("physical_instance_note"),
                "target_pre_state": label["final"]["target_pre_state"],
                "full_map_status": label["final"]["full_map_status"],
                "confidence": label["final"]["confidence"],
                "causal_note": label["final"].get("causal_note"),
                "notes": label["final"].get("notes"),
                "annotation_status": human["annotation_status"],
                "routing_label": human["routing_label"],
                "correct_action_type": human["correct_action_type"],
                "legal_target_uids": legal_uids,
                "identity_routing_eligible": human["identity_routing_eligible"],
                "main_set": human["main_set"],
                "sensitivity_set": human["sensitivity_set"],
                "episode_review": human["episode_review"],
                "is_error": human["is_error"],
            },
            "private_gt_audit": {
                "auto_evaluable": private.get("private_auto_evaluable"),
                "auto_routing_label": private.get("private_auto_routing_label"),
                "auto_episode_role": private.get("private_auto_episode_role"),
                "causal_group_uid": private.get("private_causal_group_uid"),
                "obs_gt_id": private.get("private_obs_gt_id"),
                "obs_gt_label": private.get("private_obs_gt_label"),
                "obs_gt_purity": private.get("private_obs_gt_purity"),
                "legal_candidate_uids": private.get("private_legal_candidate_uids") or [],
                "all_legal_candidates_displayed": (
                    case_private.get("private_full_map_gt_audit") or {}
                ).get("all_legal_candidates_displayed"),
            },
        })
    return rows


def camera_position(frame: dict[str, Any]) -> np.ndarray:
    return np.asarray(frame["pose"], dtype=np.float64)[:3, 3]


def unit_view(camera: np.ndarray, center: np.ndarray) -> np.ndarray:
    vector = center - camera
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(3, dtype=np.float64)


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return math.degrees(math.acos(value))


def independent_views(
    candidates: list[dict[str, Any]],
    frames: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    object_diagonal: float,
    angle_threshold: float,
    translation_scale: float,
) -> list[dict[str, Any]]:
    representatives: list[dict[str, Any]] = []
    translation_threshold = max(0.05, translation_scale * object_diagonal)
    for item in candidates:
        observation = observations.get(str(item["obs_uid"]))
        frame = frames.get(str(item["frame_uid"]))
        if observation is None or frame is None:
            continue
        center = np.asarray(observation.get("bbox_3d_center") or [], dtype=np.float64)
        if center.shape != (3,):
            continue
        camera = camera_position(frame)
        view = unit_view(camera, center)
        independent = True
        nearest_angle = None
        nearest_translation = None
        for rep in representatives:
            angle = angle_degrees(view, rep["_view"])
            translation = float(np.linalg.norm(camera - rep["_camera"]))
            if nearest_angle is None or angle < nearest_angle:
                nearest_angle = angle
            if nearest_translation is None or translation < nearest_translation:
                nearest_translation = translation
            if angle < angle_threshold and translation < translation_threshold:
                independent = False
                break
        if not independent:
            continue
        representatives.append({
            "obs_uid": item["obs_uid"],
            "processed_frame_idx": item["processed_frame_idx"],
            "raw_frame": item["raw_frame"],
            "gt_purity": item["gt_purity"],
            "gt_top_pixels": item["gt_top_pixels"],
            "nearest_previous_angle_deg": nearest_angle,
            "nearest_previous_translation_m": nearest_translation,
            "_camera": camera,
            "_view": view,
        })
    for rep in representatives:
        rep.pop("_camera")
        rep.pop("_view")
    return representatives


def future_evidence(
    event: dict[str, Any],
    association: dict[str, Any],
    gt_by_id: dict[int, list[dict[str, Any]]],
    frames: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    window: int,
    angle_threshold: float,
    translation_scale: float,
) -> dict[str, Any]:
    gt_id = event["private_gt_audit"].get("obs_gt_id")
    current_observation = observations.get(event["obs_uid"])
    if gt_id is None or current_observation is None:
        return {"status": "GT_ID_UNAVAILABLE", "independent_views_window": [], "independent_views_suffix": []}
    current_frame = frame_index(str(association["frame_uid"]))
    extent = np.asarray(current_observation.get("bbox_3d_extent") or [], dtype=np.float64)
    diagonal = float(np.linalg.norm(extent)) if extent.shape == (3,) else 0.2
    by_frame: dict[int, dict[str, Any]] = {}
    for gt in gt_by_id.get(int(gt_id), []):
        if not reliable_gt(gt):
            continue
        obs_uid = str(gt["obs_uid"])
        observation = observations.get(obs_uid)
        if observation is None:
            continue
        processed = frame_index(str(observation["frame_uid"]))
        if processed <= current_frame:
            continue
        candidate = {
            "obs_uid": obs_uid,
            "frame_uid": observation["frame_uid"],
            "processed_frame_idx": processed,
            "raw_frame": int(gt.get("raw_frame", processed)),
            "gt_purity": float(gt.get("gt_purity") or 0),
            "gt_top_pixels": int(gt.get("gt_top_pixels") or 0),
        }
        existing = by_frame.get(processed)
        score = (candidate["gt_top_pixels"], candidate["gt_purity"])
        old_score = (
            (existing["gt_top_pixels"], existing["gt_purity"])
            if existing else (-1, -1.0)
        )
        if score > old_score:
            by_frame[processed] = candidate
    suffix = [by_frame[key] for key in sorted(by_frame)]
    within = [row for row in suffix if row["processed_frame_idx"] <= current_frame + window]
    window_views = independent_views(
        within, frames, observations, diagonal, angle_threshold, translation_scale
    )
    suffix_views = independent_views(
        suffix, frames, observations, diagonal, angle_threshold, translation_scale
    )
    return {
        "status": "OK",
        "gt_id": int(gt_id),
        "object_diagonal_m": diagonal,
        "future_window_mapper_updates": window,
        "reliable_future_observations_window": len(within),
        "reliable_future_frames_suffix": len(suffix),
        "independent_views_window_count": len(window_views),
        "independent_views_suffix_count": len(suffix_views),
        "has_two_views_window": len(window_views) >= 2,
        "has_three_views_window": len(window_views) >= 3,
        "proposal_view_obs_uids": [row["obs_uid"] for row in window_views[:2]],
        "validation_view_obs_uid": window_views[2]["obs_uid"] if len(window_views) >= 3 else None,
        "independent_views_window": window_views,
        "independent_views_suffix": suffix_views,
    }


def candidate_rank(
    event: dict[str, Any],
    association: dict[str, Any],
    observation: dict[str, Any] | None,
    evidence_root: Path,
) -> dict[str, Any]:
    if event["human"]["correct_action_type"] == "NEW":
        return {
            "correct_candidate": "NEW",
            "best_existing_rank": None,
            "covered_at_1_plus_new": True,
            "covered_at_3_plus_new": True,
            "covered_at_5_plus_new": True,
        }
    targets = set(event["human"]["legal_target_uids"])
    object_uids = [str(uid) for uid in association.get("object_uids_before") or []]
    if observation is None or not targets:
        return {"correct_candidate": "ATTACH_EXISTING", "status": "TARGET_OR_OBSERVATION_MISSING"}
    ref = association.get("aggregate_sim_ref") or {}
    path = (evidence_root.parent / str(ref.get("path"))).resolve()
    if not path.is_file():
        return {"correct_candidate": "ATTACH_EXISTING", "status": "SIMILARITY_MATRIX_MISSING"}
    matrix = np.load(path, allow_pickle=False)[str(ref["key"])]
    row_index = int(observation["filtered_det_idx"])
    values = np.asarray(matrix[row_index], dtype=np.float64)
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = {object_uids[index]: rank for rank, index in enumerate(order, 1)}
    target_ranks = sorted(ranks[uid] for uid in targets if uid in ranks)
    best = target_ranks[0] if target_ranks else None
    return {
        "correct_candidate": "ATTACH_EXISTING",
        "status": "OK" if best is not None else "HUMAN_TARGET_NOT_IN_TMINUS_OBJECTS",
        "human_legal_target_uids": sorted(targets),
        "target_ranks": target_ranks,
        "best_existing_rank": best,
        "covered_at_1_plus_new": best is not None and best <= 1,
        "covered_at_3_plus_new": best is not None and best <= 3,
        "covered_at_5_plus_new": best is not None and best <= 5,
    }


def dominant_label_by_gt(gt_rows: list[dict[str, Any]]) -> dict[int, str]:
    counters: dict[int, Counter[str]] = defaultdict(Counter)
    for row in gt_rows:
        if row.get("gt_top_id") is not None and row.get("gt_top_label"):
            counters[int(row["gt_top_id"])][str(row["gt_top_label"])] += 1
    return {gt_id: counts.most_common(1)[0][0] for gt_id, counts in counters.items()}


def main() -> int:
    args = parse_args()
    large_root = args.large_packet_root.resolve()
    r2_root = args.r2_packet_root.resolve()
    evidence_root = args.evidence_root.resolve()
    output_root = args.output_root.resolve()

    large_rows = read_jsonl(args.large_events.resolve())
    for row in large_rows:
        row["source_batch"] = "ROOM0_LARGE_R1"
        row["adjudicated"] = False
    r2_rows = compile_r2_base(r2_root, args.r2_overlay.resolve())
    all_rows = large_rows + r2_rows
    by_event: dict[str, dict[str, Any]] = {}
    duplicate_events: list[str] = []
    for row in all_rows:
        event_uid = str(row["event_uid"])
        if event_uid in by_event:
            duplicate_events.append(event_uid)
        else:
            by_event[event_uid] = row
    if duplicate_events:
        raise ValueError(f"duplicate independent event_uids: {sorted(set(duplicate_events))}")

    associations = read_jsonl(evidence_root / "associations.jsonl")
    associations_by_event = {str(row["event_uid"]): row for row in associations}
    association_by_obs = {str(row["obs_uid"]): row for row in associations}
    observations = {
        str(row["obs_uid"]): row
        for row in read_jsonl(evidence_root / "observations.jsonl")
    }
    frames = {
        str(row["frame_uid"]): row
        for row in read_jsonl(evidence_root / "frames.jsonl")
    }
    versions = {
        str(row["object_version_uid"]): row
        for row in read_jsonl(evidence_root / "object_versions.jsonl")
    }
    gt_rows = read_jsonl(args.observation_gt.resolve())
    gt_by_obs = {str(row["obs_uid"]): row for row in gt_rows}
    gt_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        if row.get("gt_top_id") is not None:
            gt_by_id[int(row["gt_top_id"])].append(row)
    gt_labels = dominant_label_by_gt(gt_rows)
    routing_audit = {
        str(row["event_uid"]): row
        for row in read_jsonl(args.routing_audit.resolve())
    }

    human_errors = sorted(
        (row for row in by_event.values() if row["human"]["is_error"]),
        key=lambda row: associations_by_event[str(row["event_uid"])]["event_sequence"],
    )
    episodes: list[dict[str, Any]] = []
    for event in human_errors:
        event_uid = str(event["event_uid"])
        association = associations_by_event[event_uid]
        audit = routing_audit.get(event_uid) or {}
        human_route = event["human"]["routing_label"]
        auto_route = audit.get("private_auto_routing_label")
        target_state = event["human"]["target_pre_state"]
        if event["original_action_type"] == "ATTACH_EXISTING":
            if target_state == "CLEAN_SINGLE_INSTANCE":
                causal_role = "ROOT_CONFIRMED_BY_HUMAN_TARGET_STATE"
            elif target_state == "ALREADY_CONTAMINATED":
                causal_role = "CASCADE_CONFIRMED_BY_HUMAN_TARGET_STATE"
            else:
                causal_role = "ROOT_OR_CASCADE_PENDING"
        elif auto_route == human_route and audit.get("private_auto_episode_role") == "ROOT_CANDIDATE":
            causal_role = "ROOT_SUPPORTED_BY_HUMAN_ROUTE_AND_GT_LINEAGE"
        elif auto_route == human_route and audit.get("private_auto_episode_role") == "CASCADE_CANDIDATE":
            causal_role = "CASCADE_SUPPORTED_BY_HUMAN_ROUTE_AND_GT_LINEAGE"
        else:
            causal_role = "ROOT_OR_CASCADE_PENDING"

        obs_gt = gt_by_obs.get(event["obs_uid"])
        obs_gt_id = (
            int(obs_gt["gt_top_id"])
            if obs_gt is not None and obs_gt.get("gt_top_id") is not None
            else event["private_gt_audit"].get("obs_gt_id")
        )
        if auto_route == human_route and audit.get("private_causal_group_uid"):
            group_uid = "audit:" + str(audit["private_causal_group_uid"])
        elif event["original_action_type"] == "ATTACH_EXISTING":
            group_uid = "human-attach:" + ":".join([
                str(event.get("original_target_uid")), str(obs_gt_id)
            ])
        else:
            group_uid = "human-new:" + ":".join([
                ",".join(event["human"]["legal_target_uids"]), str(obs_gt_id)
            ])

        target_version_uid = association.get("target_object_version_before")
        anchor_candidate = None
        if (
            causal_role == "CASCADE_CONFIRMED_BY_HUMAN_TARGET_STATE"
            and obs_gt_id is not None
            and target_version_uid in versions
        ):
            candidates = []
            for member_obs_uid in versions[target_version_uid].get("member_observation_uids") or []:
                gt = gt_by_obs.get(str(member_obs_uid))
                member_assoc = association_by_obs.get(str(member_obs_uid))
                if (
                    gt is not None
                    and gt.get("gt_top_id") is not None
                    and int(gt["gt_top_id"]) == int(obs_gt_id)
                    and member_assoc is not None
                    and int(member_assoc["event_sequence"]) < int(association["event_sequence"])
                ):
                    candidates.append((int(member_assoc["event_sequence"]), member_obs_uid, gt, member_assoc))
            if candidates:
                _, member_obs_uid, gt, member_assoc = sorted(candidates)[0]
                anchor_candidate = {
                    "event_uid": member_assoc["event_uid"],
                    "obs_uid": member_obs_uid,
                    "processed_frame_idx": frame_index(member_assoc["frame_uid"]),
                    "raw_frame": gt.get("raw_frame"),
                    "gt_purity": gt.get("gt_purity"),
                    "has_human_label": str(member_assoc["event_uid"]) in by_event,
                    "human_label": (
                        by_event[str(member_assoc["event_uid"])]["human"]["routing_label"]
                        if str(member_assoc["event_uid"]) in by_event else None
                    ),
                }

        target_gt_id = audit.get("private_target_gt_id")
        obs_label = gt_labels.get(int(obs_gt_id)) if obs_gt_id is not None else None
        target_label = gt_labels.get(int(target_gt_id)) if target_gt_id is not None else None
        class_relation = (
            "SAME_CLASS" if obs_label and target_label and obs_label == target_label
            else "DIFFERENT_CLASS" if obs_label and target_label
            else "UNKNOWN"
        )
        future = future_evidence(
            event,
            association,
            gt_by_id,
            frames,
            observations,
            args.future_window,
            args.view_angle_deg,
            args.translation_scale,
        )
        coverage = candidate_rank(
            event, association, observations.get(event["obs_uid"]), evidence_root
        )
        episodes.append({
            "schema_version": "experiment0-human-causal-episode/1.0",
            "case_uid": event["case_uid"],
            "event_uid": event_uid,
            "event_sequence": association["event_sequence"],
            "processed_frame_idx": frame_index(association["frame_uid"]),
            "raw_frame": int(event["source_frame"].replace("frame", "")),
            "source_batch": event["source_batch"],
            "queue_memberships": event["queue_memberships"],
            "routing_label": human_route,
            "correct_action_type": event["human"]["correct_action_type"],
            "original_action_type": event["original_action_type"],
            "original_target_uid": event.get("original_target_uid"),
            "created_object_uid": event.get("created_object_uid"),
            "human_legal_target_uids": event["human"]["legal_target_uids"],
            "human_target_pre_state": target_state,
            "human_confidence": event["human"]["confidence"],
            "causal_role": causal_role,
            "causal_group_uid": group_uid,
            "earliest_same_identity_member_in_target": anchor_candidate,
            "offline_identity_audit": {
                "obs_gt_id": obs_gt_id,
                "obs_gt_label": obs_label,
                "obs_gt_purity": obs_gt.get("gt_purity") if obs_gt else None,
                "original_target_gt_id": target_gt_id,
                "original_target_gt_label": target_label,
                "class_relation": class_relation,
                "private_auto_routing_label": auto_route,
                "private_auto_episode_role": audit.get("private_auto_episode_role"),
                "private_causal_group_uid": audit.get("private_causal_group_uid"),
                "human_route_confirms_private_auto": auto_route == human_route,
            },
            "candidate_coverage": coverage,
            "future_evidence": future,
            "replay_plan": {
                "B0": "original decision; no repair",
                "B1": "revise only root membership; freeze later routing decisions",
                "B2": "revise root and replay dependency closure",
                "B3": "revise root and replay full suffix reference",
                "eligible_for_first_oracle_replay": causal_role.startswith("ROOT_")
                and causal_role != "ROOT_OR_CASCADE_PENDING"
                and future.get("has_two_views_window") is True,
            },
        })

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_group[episode["causal_group_uid"]].append(episode)
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: row["event_sequence"])
        roots = [
            row for row in group_rows
            if row["causal_role"].startswith("ROOT_")
            and row["causal_role"] != "ROOT_OR_CASCADE_PENDING"
        ]
        root_uid = roots[0]["event_uid"] if roots else None
        for index, row in enumerate(group_rows):
            row["linked_human_root_event_uid"] = root_uid
            if (
                index > 0
                and root_uid is not None
                and row["event_uid"] != root_uid
                and row["causal_role"].startswith("ROOT_")
                and row["causal_role"] != "ROOT_OR_CASCADE_PENDING"
            ):
                row["causal_role_before_group_chronology"] = row["causal_role"]
                row["causal_role"] = "CASCADE_BY_GROUP_CHRONOLOGY"
                row["group_role_override_reason"] = (
                    "A prior human-confirmed error exists in the same causal group; "
                    "the later event cannot be counted as an independent root."
                )
                row["replay_plan"]["eligible_for_first_oracle_replay"] = False

    def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        root_rows = [
            row for row in rows
            if row["causal_role"].startswith("ROOT_")
            and row["causal_role"] != "ROOT_OR_CASCADE_PENDING"
        ]
        cascade_rows = [row for row in rows if row["causal_role"].startswith("CASCADE_")]
        pending_rows = [row for row in rows if row["causal_role"].endswith("PENDING")]
        future_ok = [row for row in rows if row["future_evidence"].get("status") == "OK"]
        two = [row for row in rows if row["future_evidence"].get("has_two_views_window")]
        three = [row for row in rows if row["future_evidence"].get("has_three_views_window")]
        top5 = [row for row in rows if row["candidate_coverage"].get("covered_at_5_plus_new")]
        replay = [row for row in rows if row["replay_plan"]["eligible_for_first_oracle_replay"]]
        return {
            "events": len(rows),
            "routing_label_counts": dict(sorted(Counter(row["routing_label"] for row in rows).items())),
            "causal_role_counts": dict(sorted(Counter(row["causal_role"] for row in rows).items())),
            "root_events": len(root_rows),
            "cascade_events": len(cascade_rows),
            "pending_events": len(pending_rows),
            "offline_gt_available": len(future_ok),
            "two_independent_future_views_within_window": len(two),
            "three_independent_future_views_within_window": len(three),
            "two_view_rate_among_gt_available": ratio(len(two), len(future_ok)),
            "three_view_rate_among_gt_available": ratio(len(three), len(future_ok)),
            "top5_plus_new_coverage": ratio(len(top5), len(rows)),
            "oracle_replay_first_case_count": len(replay),
            "oracle_replay_first_case_uids": [row["case_uid"] for row in replay],
        }

    probability_errors = [
        row for row in episodes if "PROBABILITY_SAMPLE" in row["queue_memberships"]
    ]
    large_errors = [row for row in episodes if row["source_batch"] == "ROOM0_LARGE_R1"]
    r2_errors = [row for row in episodes if row["source_batch"] == "V2_R2_ADJUDICATED_CALIBRATION"]
    report = {
        "schema_version": "experiment0-room0-human-episode-summary/1.0",
        "status": "READY_FOR_ORACLE_REPLAY_CASE_SELECTION",
        "inputs": {
            "combined_independent_human_events": len(by_event),
            "large_events": len(large_rows),
            "r2_adjudicated_events": len(r2_rows),
            "human_confirmed_error_events": len(episodes),
            "note": "The original v2 trial is excluded because its raw quality labels require an unmaterialized adjudication overlay.",
        },
        "all_human_confirmed_errors": subset_summary(episodes),
        "large_batch_errors": subset_summary(large_errors),
        "natural_probability_errors": subset_summary(probability_errors),
        "r2_calibration_errors": subset_summary(r2_errors),
        "error_case_table": [
            {
                "case_uid": row["case_uid"],
                "source_batch": row["source_batch"],
                "routing_label": row["routing_label"],
                "causal_role": row["causal_role"],
                "causal_group_uid": row["causal_group_uid"],
                "future_views_30": row["future_evidence"].get("independent_views_window_count"),
                "top5_plus_new": row["candidate_coverage"].get("covered_at_5_plus_new"),
                "class_relation": row["offline_identity_audit"]["class_relation"],
                "first_oracle_replay": row["replay_plan"]["eligible_for_first_oracle_replay"],
            }
            for row in episodes
        ],
        "interpretation": [
            "Human route labels, not private projected-GT routes, define the error set.",
            "Private GT is used only for offline lineage and future-view availability audits.",
            "Continuous observations in one view cluster count once; proposal uses the first two representatives and validation uses the third.",
            "room0 is a development scene, so these results select the next minimal experiment but cannot establish cross-scene prevalence.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / "combined_human_routing_events.jsonl", by_event.values())
    write_jsonl_atomic(output_root / "human_error_episodes.jsonl", episodes)
    write_json_atomic(output_root / "episode_summary.json", report)

    natural = report["natural_probability_errors"]
    all_summary = report["all_human_confirmed_errors"]
    lines = [
        "# Experiment 0 room0：人工标签驱动的 episode 与未来证据编译",
        "",
        "## 结论",
        "",
        "标注已从页面答案转换为事件时节点 UID、root/cascade 角色、候选排名和未来独立视角。后续 oracle replay 只读取这份事件表。",
        "",
        "## 人工确认错误",
        "",
        f"- 合并后的独立人工事件：{len(by_event)}",
        f"- 人工确认错误事件：{all_summary['events']}",
        f"- root：{all_summary['root_events']}；cascade：{all_summary['cascade_events']}；待定：{all_summary['pending_events']}",
        f"- 30 个 mapper update 内至少 2 个独立未来视角：{all_summary['two_independent_future_views_within_window']}",
        f"- `top-5 + NEW` 覆盖率：{all_summary['top5_plus_new_coverage']}",
        "",
        "## 自然概率队列中的错误",
        "",
        f"- 错误事件：{natural['events']}；root：{natural['root_events']}；cascade：{natural['cascade_events']}；待定：{natural['pending_events']}",
        f"- 30 updates 内两视角覆盖：{natural['two_independent_future_views_within_window']}/{natural['offline_gt_available']}",
        f"- 首批可进入 B0/B1/B2/B3 oracle replay：{natural['oracle_replay_first_case_uids']}",
        "",
        "## 下一步约束",
        "",
        "1. 先跑最少的人工确认 root，不能把 cascade 当独立样本。",
        "2. B0/B1/B2/B3 使用同一错误前 snapshot、同一未来 observation 顺序和相同特征。",
        "3. proposal 与 validation 视角不重叠；不足三视角时只做 evidence ceiling，不声称安全提交。",
        "4. room0 只用于开发；并行准备 room2 从 frame 0 空图的同配置日志。",
        "",
    ]
    (output_root / "ROOM0_HUMAN_EPISODE_COMPILATION_CN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "combined_events": len(by_event),
        "human_errors": len(episodes),
        "natural_errors": natural["events"],
        "natural_roots": natural["root_events"],
        "natural_cascades": natural["cascade_events"],
        "natural_pending": natural["pending_events"],
        "natural_two_view": natural["two_independent_future_views_within_window"],
        "first_oracle_replay": natural["oracle_replay_first_case_uids"],
        "output_root": str(output_root),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
