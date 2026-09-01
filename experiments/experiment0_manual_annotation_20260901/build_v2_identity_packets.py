#!/usr/bin/env python3
"""Build hash-bound ATTACH/NEW identity-routing packets for the v2 schema trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

import build_event_packets as legacy


SCHEMA_VERSION = "experiment0-identity-routing-packet/2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--history-views", type=int, default=6)
    return parser.parse_args()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tminus_snapshot_sha256(association: dict[str, Any]) -> str:
    return stable_json_sha256(
        {
            "frame_uid": association.get("frame_uid"),
            "event_uid": association.get("event_uid"),
            "object_uids_before": association.get("object_uids_before") or [],
            "candidate_object_version_uids": (
                association.get("candidate_object_version_uids") or []
            ),
            "aggregate_sim_sha256": (
                association.get("aggregate_sim_ref") or {}
            ).get("sha256"),
        }
    )


def choose_candidates(
    association: dict[str, Any],
    similarity: list[dict[str, Any]],
    versions: dict[str, dict[str, Any]],
    forced_legal_uids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        similarity, key=lambda row: (-float(row["aggregate_score"]), row["object_uid"])
    )
    by_uid = {str(row["object_uid"]): row for row in ranked}
    selected_uids = [str(row["object_uid"]) for row in ranked[:top_k]]
    if association.get("decision") == "MERGE_TO_OBJECT":
        target_uid = str(association.get("target_object_uid") or "")
        if target_uid and target_uid not in selected_uids:
            selected_uids.append(target_uid)
    for uid in forced_legal_uids:
        uid = str(uid)
        if uid not in selected_uids:
            selected_uids.append(uid)

    selected = []
    for uid in selected_uids:
        row = by_uid.get(uid)
        if row is None:
            raise ValueError(f"forced candidate missing from similarity row: {uid}")
        if str(row.get("object_version_uid") or "") not in versions:
            raise ValueError(f"candidate version missing: {uid}")
        selected.append(dict(row))
    random.Random(str(association["event_uid"])).shuffle(selected)
    for index, row in enumerate(selected):
        row["code"] = chr(ord("A") + index)
    return selected


def build_case(
    args: argparse.Namespace,
    state: dict[str, Any],
    work_item: dict[str, Any],
    association: dict[str, Any],
) -> dict[str, Any]:
    case_uid = str(work_item["case_uid"])
    case_dir = args.output_root / "cases" / case_uid
    case_dir.mkdir(parents=True, exist_ok=True)
    exp_root: Path = state["exp_root"]
    frames = state["frames"]
    observations = state["observations"]
    versions = state["versions"]
    observation = observations.get(str(association.get("obs_uid") or ""))
    if observation is None:
        raise ValueError(f"kept observation missing for {association.get('event_uid')}")

    recorded_snapshot = work_item.get("tminus_snapshot_sha256")
    computed_snapshot = tminus_snapshot_sha256(association)
    if recorded_snapshot and recorded_snapshot != computed_snapshot:
        raise ValueError(f"t^- snapshot binding mismatch for {association.get('event_uid')}")

    frame = frames[str(observation["frame_uid"])]
    current_context, current_crop = legacy.render_observation_assets(
        exp_root, observation, frames, case_dir, "current", (230, 66, 112)
    )
    current_points = legacy.sampled_points(
        legacy.load_points(exp_root, observation), 800, str(observation["obs_uid"])
    )

    similarity = legacy.load_similarity_row(exp_root, association)
    forced_legal_uids = [
        str(value) for value in work_item.get("private_legal_candidate_uids") or []
    ]
    candidates = choose_candidates(
        association, similarity, versions, forced_legal_uids, args.top_k
    )
    # A genuine first-frame NEW has an empty t^- map.  It is still a valid and
    # useful CORRECT_NEW calibration cell; the UI then offers only
    # NONE_SHOWN/UNCERTAIN for the identity question.

    decision = str(association.get("decision") or "")
    if decision == "MERGE_TO_OBJECT":
        original_action_type = "ATTACH_EXISTING"
        original_target_uid = str(association.get("target_object_uid") or "")
        original_target_code = next(
            (row["code"] for row in candidates if row["object_uid"] == original_target_uid),
            None,
        )
        if original_target_code is None:
            raise ValueError("original ATTACH target is not displayed")
    elif decision == "CREATE_OBJECT":
        original_action_type = "NEW"
        original_target_uid = None
        original_target_code = None
    else:
        raise ValueError(f"unsupported decision: {decision}")

    public_candidates = []
    private_candidates = []
    for candidate in candidates:
        code = str(candidate["code"])
        version = versions[str(candidate["object_version_uid"])]
        history_name = f"candidate_{code}_history.jpg"
        history_meta = legacy.render_history_sheet(
            exp_root,
            version,
            observations,
            frames,
            case_dir / history_name,
            args.history_views,
        )
        selected_history = legacy.select_history_observations(
            version, observations, args.history_views
        )
        history_parts = [legacy.load_points(exp_root, row) for row in selected_history]
        nonempty = [part for part in history_parts if len(part)]
        history_points = (
            np.concatenate(nonempty, axis=0)
            if nonempty
            else np.empty((0, 3), dtype=float)
        )
        history_points = legacy.sampled_points(
            history_points, 1200, str(candidate["object_version_uid"])
        )
        pcd_name = f"candidate_{code}_3d.jpg"
        legacy.render_3d_comparison(
            current_points,
            history_points,
            case_dir / pcd_name,
            f"Candidate {code}",
        )
        public_candidates.append(
            {
                "code": code,
                "history_asset": history_name,
                "pcd_asset": pcd_name,
                "history_observation_count": len(
                    version.get("member_observation_uids") or []
                ),
                "history_frame_count": legacy.history_frame_count(version, observations),
                "displayed_history_count": len(history_meta),
            }
        )
        private_candidates.append(
            {
                **candidate,
                "history_displayed": history_meta,
                "history_member_observation_uids": list(
                    version.get("member_observation_uids") or []
                ),
                "private_forced_by_full_map_gt_audit": (
                    str(candidate["object_uid"]) in forced_legal_uids
                ),
            }
        )

    shown_legal = sorted(
        str(candidate["object_uid"])
        for candidate in candidates
        if str(candidate["object_uid"]) in forced_legal_uids
    )
    if shown_legal != sorted(forced_legal_uids):
        raise ValueError("not all private legal candidates are displayed")

    asset_names = [current_context, current_crop]
    for candidate in public_candidates:
        asset_names.extend([candidate["history_asset"], candidate["pcd_asset"]])
    asset_hashes = {
        name: legacy.sha256_file(case_dir / name) for name in sorted(asset_names)
    }
    event_frame_idx = legacy.event_frame(association, observations)
    public = {
        "schema_version": SCHEMA_VERSION,
        "scene": args.scene,
        "case_uid": case_uid,
        "event_uid": association.get("event_uid"),
        "event_frame_idx": event_frame_idx,
        "source_frame": frame.get("source_frame_id"),
        "mapper_latest_frame_at_event": event_frame_idx,
        "packet_built_when_mapper_latest_frame": state["latest_frame"],
        "tminus_snapshot_sha256": computed_snapshot,
        "current": {
            "context_asset": current_context,
            "crop_asset": current_crop,
            "mask_area": observation.get("processed_mask_area"),
            "valid_depth_ratio": observation.get("valid_depth_ratio"),
            "stored_point_count": observation.get("pcd_stored_points"),
        },
        "candidates": public_candidates,
        "displayed_asset_sha256": asset_hashes,
        "annotation_notice": (
            "Original ATTACH/NEW action, mapper target, UID, score, corrected GT and "
            "private sampling stratum are hidden until the blind identity judgement is saved."
        ),
    }
    code_by_uid = {
        str(candidate["object_uid"]): str(candidate["code"])
        for candidate in candidates
    }
    private = {
        "schema_version": SCHEMA_VERSION,
        "scene": args.scene,
        "case_uid": case_uid,
        "event_uid": association.get("event_uid"),
        "obs_uid": association.get("obs_uid"),
        "association_event": association,
        "original_action_type": original_action_type,
        "original_target_code": original_target_code,
        "original_target_uid": original_target_uid,
        "created_object_uid": (
            association.get("target_object_uid") if original_action_type == "NEW" else None
        ),
        "candidates": private_candidates,
        "private_full_map_gt_audit": {
            "legal_candidate_uids": forced_legal_uids,
            "legal_candidate_codes": [code_by_uid[uid] for uid in forced_legal_uids],
            "all_legal_candidates_displayed": True,
            "human_label_still_required": True,
        },
        "sampling": {
            key: value for key, value in work_item.items() if key not in {"case_uid", "event_uid"}
        },
        "source_public_sha256": None,
    }
    public_path = case_dir / "case_public.json"
    private_path = case_dir / "case_private.json"
    legacy.write_json_atomic(public_path, public)
    private["source_public_sha256"] = legacy.sha256_file(public_path)
    legacy.write_json_atomic(private_path, private)
    return {
        "case_uid": case_uid,
        "event_uid": association.get("event_uid"),
        "case_dir": str(case_dir.resolve()),
        "source_frame": frame.get("source_frame_id"),
        "sample_kind": work_item.get("sample_kind"),
        "repeat_of": work_item.get("repeat_of"),
        "schema_version": SCHEMA_VERSION,
    }


def main() -> int:
    args = parse_args()
    args.evidence_root = args.evidence_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state = legacy.load_exp_state(args.evidence_root)
    associations = {
        str(row.get("event_uid")): row for row in state["associations"]
    }
    private_worklist = legacy.read_jsonl(args.worklist)
    manifest_rows = []
    failures = []
    for item in private_worklist:
        event_uid = str(item.get("event_uid") or "")
        association = associations.get(event_uid)
        if association is None:
            failures.append(
                {
                    "case_uid": item.get("case_uid"),
                    "event_uid": event_uid,
                    "error": "association event missing",
                }
            )
            continue
        try:
            manifest_rows.append(build_case(args, state, item, association))
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            failures.append(
                {
                    "case_uid": item.get("case_uid"),
                    "event_uid": event_uid,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    manifest_rows.sort(key=lambda row: row["case_uid"])
    worklist_path = args.output_root / "worklist.jsonl"
    legacy.write_jsonl_atomic(worklist_path, manifest_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if not failures else "READY_WITH_FAILURES",
        "purpose": "SCHEMA_AND_ANNOTATION_VALIDITY_ONLY_NOT_PREVALENCE",
        "scene": args.scene,
        "evidence_root": str(args.evidence_root),
        "evidence_manifest_sha256": legacy.sha256_file(
            args.evidence_root / "manifest.json"
        ),
        "source_private_worklist": str(args.worklist.resolve()),
        "source_private_worklist_sha256": legacy.sha256_file(args.worklist),
        "mapper_complete": state["complete"],
        "mapper_latest_frame": state["latest_frame"],
        "ready_through_frame": state["ready_frame"],
        "case_count": len(manifest_rows),
        "failure_count": len(failures),
        "worklist_sha256": legacy.sha256_file(worklist_path),
        "failures": failures,
    }
    legacy.write_json_atomic(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
