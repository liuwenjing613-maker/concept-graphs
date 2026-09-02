#!/usr/bin/env python3
"""Read-only multidimensional audit for Experiment 0 human routing labels.

This script is intentionally analysis-only.  It reads the frozen event packets,
the corrected offline observation GT, the immutable mapper evidence ledgers, and
existing B0/B0R/B1/B2/B3 replay products.  It never mutates mapper state or
production configuration.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EVENT_SEQUENCE_RE = re.compile(r"_e(\d+)$")
FRAME_RE = re.compile(r"_f(\d+)_")
VERSION_RE = re.compile(r"@v(\d+)$")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def event_sequence(uid: Any) -> int | None:
    match = EVENT_SEQUENCE_RE.search(str(uid or ""))
    return int(match.group(1)) if match else None


def frame_index(uid: Any) -> int | None:
    match = FRAME_RE.search(str(uid or ""))
    return int(match.group(1)) if match else None


def version_index(uid: Any) -> int:
    match = VERSION_RE.search(str(uid or ""))
    return int(match.group(1)) if match else -1


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def mean_vector(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    width = len(vectors[0])
    return [sum(float(row[index]) for row in vectors) / len(vectors) for index in range(width)]


def sorted_counter(counter: Counter[Any], denominator: int | None = None) -> list[dict[str, Any]]:
    total = denominator if denominator is not None else sum(counter.values())
    return [
        {"value": str(value), "count": count, "fraction": safe_ratio(count, total)}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def grouped_counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def median_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def source_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_unix": stat.st_mtime,
    }
    if path.suffix == ".jsonl":
        with path.open("rb") as handle:
            info["line_count"] = sum(1 for _ in handle)
    return info


def normalize_causal_role(value: Any) -> str:
    text = str(value or "")
    # The compiled label ROOT_OR_CASCADE_PENDING is not an observed root.
    # Check PENDING before the ROOT prefix so the uncertainty is preserved.
    if "PENDING" in text:
        return "PENDING"
    if text.startswith("ROOT"):
        return "ROOT"
    if text.startswith("CASCADE"):
        return "CASCADE"
    return "PENDING"


def purity_bin(value: Any) -> str:
    if value is None:
        return "MISSING"
    value = float(value)
    if value >= 0.95:
        return ">=0.95"
    if value >= 0.80:
        return "0.80-0.95"
    return "<0.80"


class Audit:
    def __init__(self, project_root: Path, output_dir: Path) -> None:
        self.project_root = project_root
        self.output_dir = output_dir
        self.exp_root = project_root / "results/experiments/experiment0_manual_annotation_20260901"
        self.analysis_root = self.exp_root / "v2_large_room0_r1/analysis_20260902"
        self.evidence_root = (
            project_root
            / "results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps"
            / "online_label_trigger_v1_room0_dev_pcd/evidence"
        )
        self.paths = {
            "events": self.analysis_root / "episodes/combined_human_routing_events.jsonl",
            "episodes": self.analysis_root / "episodes/human_error_episodes.jsonl",
            "gt": self.exp_root / "corrected_gt_audit_room0/observation_gt.jsonl",
            "associations": self.evidence_root / "associations.jsonl",
            "observations": self.evidence_root / "observations.jsonl",
            "versions": self.evidence_root / "object_versions.jsonl",
            "final_membership": self.evidence_root / "final_membership.json",
            "create_partition_audit": self.analysis_root / "oracle_create_partition_audit/create_partition_audit.json",
            "routing_audit_records": self.exp_root / "identity_routing_v2_audit_room0/routing_records_private.jsonl",
            "routing_audit_summary": self.exp_root / "identity_routing_v2_audit_room0/summary.json",
            "large_worklist_manifest": self.exp_root / "v2_large_room0_r1/worklist_manifest.json",
        }
        for name, path in self.paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing required {name}: {path}")

        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.error_by_case: dict[str, dict[str, Any]] = {}
        self.gt: dict[str, dict[str, Any]] = {}
        self.observations: dict[str, dict[str, Any]] = {}
        self.assoc_by_event: dict[str, dict[str, Any]] = {}
        self.assoc_by_obs: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, dict[str, Any]] = {}
        self.versions_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.final_membership: dict[str, list[str]] = {}
        self.native_owner_by_obs: dict[str, str] = {}
        self.packet_private_by_case: dict[str, dict[str, Any]] = {}
        self.composition_cache: dict[str, dict[str, Any]] = {}
        self.labeled_event_uids: set[str] = set()
        self.routing_audit_event_uids: set[str] = set()

    def load(self) -> None:
        self.events = read_jsonl(self.paths["events"])
        self.errors = read_jsonl(self.paths["episodes"])
        self.error_by_case = {str(row["case_uid"]): row for row in self.errors}
        self.gt = {str(row["obs_uid"]): row for row in read_jsonl(self.paths["gt"])}
        self.routing_audit_event_uids = {
            str(row["event_uid"]) for row in read_jsonl(self.paths["routing_audit_records"])
        }

        labeled_event_uids = {str(row["event_uid"]) for row in self.events}
        self.labeled_event_uids = labeled_event_uids
        with self.paths["associations"].open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                slim = {
                    "event_uid": row.get("event_uid"),
                    "event_sequence": row.get("event_sequence"),
                    "obs_uid": row.get("obs_uid"),
                    "decision": row.get("decision"),
                    "target_object_uid": row.get("target_object_uid"),
                    "target_object_version_before": row.get("target_object_version_before"),
                    "target_object_version_after": row.get("target_object_version_after"),
                    "top1_score": row.get("top1_score"),
                    "top2_score": row.get("top2_score"),
                    "margin": row.get("margin"),
                    "sim_threshold": row.get("sim_threshold"),
                    "match_method": row.get("match_method"),
                }
                if str(row.get("event_uid")) in labeled_event_uids:
                    slim["top_candidates"] = row.get("top_candidates") or []
                    slim["candidate_object_version_uids"] = row.get("candidate_object_version_uids") or []
                self.assoc_by_event[str(row["event_uid"])] = slim
                self.assoc_by_obs[str(row["obs_uid"])] = slim

        associated_obs = set(self.assoc_by_obs)
        with self.paths["observations"].open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                uid = str(row.get("obs_uid") or "")
                if uid not in associated_obs:
                    continue
                self.observations[uid] = {
                    "obs_uid": uid,
                    "frame_uid": row.get("frame_uid"),
                    "bbox_3d_center": row.get("bbox_3d_center"),
                    "bbox_3d_extent": row.get("bbox_3d_extent"),
                    "class_id": row.get("class_id"),
                    "class_name": row.get("class_name"),
                    "detection_label": row.get("detection_label"),
                    "confidence": row.get("confidence"),
                    "processed_mask_area": row.get("processed_mask_area"),
                    "boundary_touch_ratio": row.get("boundary_touch_ratio"),
                }

        with self.paths["versions"].open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                uid = str(row["object_version_uid"])
                self.versions[uid] = row
                self.versions_by_object[str(row["object_uid"])].append(row)
        for rows in self.versions_by_object.values():
            rows.sort(key=lambda row: (event_sequence(row.get("trigger_event_uid")) or -1, version_index(row.get("object_version_uid"))))

        membership_rows = read_json(self.paths["final_membership"])
        self.final_membership = {
            str(row["object_uid"]): [str(uid) for uid in row.get("member_observation_uids") or []]
            for row in membership_rows
            if str(row.get("status") or "active") == "active"
        }
        for owner_uid, members in self.final_membership.items():
            for obs_uid in members:
                if obs_uid in self.native_owner_by_obs:
                    raise ValueError(f"duplicate native owner for {obs_uid}")
                self.native_owner_by_obs[obs_uid] = owner_uid

        for path in self.exp_root.rglob("case_private.json"):
            try:
                packet = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            case_uid = str(packet.get("case_uid") or packet.get("public", {}).get("case_uid") or "")
            if case_uid and case_uid not in self.packet_private_by_case:
                packet["_source_path"] = str(path)
                self.packet_private_by_case[case_uid] = packet

    def version_at(self, object_uid: str, sequence: int, exact_uid: str | None = None) -> dict[str, Any] | None:
        if exact_uid and exact_uid in self.versions:
            return self.versions[exact_uid]
        eligible = [
            row
            for row in self.versions_by_object.get(object_uid, [])
            if (event_sequence(row.get("trigger_event_uid")) or -1) < sequence
        ]
        return eligible[-1] if eligible else None

    def composition(self, members: list[str], cache_key: str | None = None, bbox_extent: Any = None) -> dict[str, Any]:
        if cache_key and cache_key in self.composition_cache:
            return self.composition_cache[cache_key]

        member_uids = [str(uid) for uid in members]
        eligible_rows: list[tuple[str, dict[str, Any]]] = []
        gt_missing = 0
        for uid in member_uids:
            row = self.gt.get(uid)
            if not row or not row.get("gt_assignment_eligible") or row.get("gt_top_id") is None:
                gt_missing += 1
                continue
            eligible_rows.append((uid, row))

        gt_counts: Counter[int] = Counter(int(row["gt_top_id"]) for _, row in eligible_rows)
        label_counts: Counter[str] = Counter(str(row.get("gt_top_label") or "UNKNOWN") for _, row in eligible_rows)
        sorted_ids = sorted(gt_counts, key=lambda value: (-gt_counts[value], value))
        dominant_id = sorted_ids[0] if sorted_ids else None
        second_id = sorted_ids[1] if len(sorted_ids) > 1 else None
        second_count = gt_counts.get(second_id, 0) if second_id is not None else 0
        gt_frames: dict[int, set[int]] = defaultdict(set)
        gt_members: dict[int, list[str]] = defaultdict(list)
        gt_centers: dict[int, list[list[float]]] = defaultdict(list)
        gt_pixel_counts_top2: Counter[int] = Counter()
        detector_classes: Counter[str] = Counter()
        frame_id_sets: dict[int, set[int]] = defaultdict(set)
        purities: list[float] = []
        pixel_mixed_count = 0
        two_foreground_count = 0
        pixel_mixed_examples: list[dict[str, Any]] = []
        pixel_mixed_frames: set[int] = set()

        for uid, row in eligible_rows:
            gt_id = int(row["gt_top_id"])
            frame = int(row.get("frame_idx") if row.get("frame_idx") is not None else frame_index(uid) or -1)
            gt_frames[gt_id].add(frame)
            gt_members[gt_id].append(uid)
            gt_pixel_counts_top2[gt_id] += int(row.get("gt_top_pixels") or 0)
            if row.get("gt_second_id") is not None:
                gt_pixel_counts_top2[int(row["gt_second_id"])] += int(row.get("gt_second_pixels") or 0)
            frame_id_sets[frame].add(gt_id)
            if row.get("gt_purity") is not None:
                purities.append(float(row["gt_purity"]))
            pixel_mixed_count += int(bool(row.get("mask_mixed")))
            two_foreground_count += int(bool(row.get("mask_two_foreground")))
            if row.get("mask_mixed") or row.get("mask_two_foreground"):
                pixel_mixed_frames.add(frame)
                pixel_mixed_examples.append(
                    {
                        "obs_uid": uid,
                        "frame_idx": frame,
                        "gt_top_id": row.get("gt_top_id"),
                        "gt_top_label": row.get("gt_top_label"),
                        "gt_purity": row.get("gt_purity"),
                        "gt_second_id": row.get("gt_second_id"),
                        "gt_second_label": row.get("gt_second_label"),
                        "gt_second_fraction": row.get("gt_second_fraction"),
                        "mask_mixed": row.get("mask_mixed"),
                        "mask_two_foreground": row.get("mask_two_foreground"),
                    }
                )
            obs = self.observations.get(uid, {})
            detector_classes[str(obs.get("detection_label") or obs.get("class_name") or "UNKNOWN")] += 1
            center = obs.get("bbox_3d_center")
            if isinstance(center, list) and len(center) == 3 and all(value is not None for value in center):
                gt_centers[gt_id].append([float(value) for value in center])

        top_component_ids = sorted_ids[:2]
        coobserved_frames: list[int] = []
        if len(top_component_ids) == 2:
            expected = set(top_component_ids)
            coobserved_frames = sorted(frame for frame, ids in frame_id_sets.items() if expected.issubset(ids))
        centroid_means = {str(gt_id): mean_vector(gt_centers[gt_id]) for gt_id in sorted_ids}
        centroid_distance = None
        if len(top_component_ids) == 2:
            left = centroid_means.get(str(top_component_ids[0]))
            right = centroid_means.get(str(top_component_ids[1]))
            if left and right:
                centroid_distance = distance(left, right)
        diagonal = None
        if isinstance(bbox_extent, list) and len(bbox_extent) == 3:
            diagonal = math.sqrt(sum(float(value) ** 2 for value in bbox_extent))

        ordered = sorted(
            eligible_rows,
            key=lambda pair: (
                int(pair[1].get("frame_idx") if pair[1].get("frame_idx") is not None else frame_index(pair[0]) or -1),
                int(self.assoc_by_obs.get(pair[0], {}).get("event_sequence") or 10**12),
                pair[0],
            ),
        )
        transitions = 0
        previous_id = None
        for _, row in ordered:
            current_id = int(row["gt_top_id"])
            if previous_id is not None and current_id != previous_id:
                transitions += 1
            previous_id = current_id

        coverage = safe_ratio(len(eligible_rows), len(member_uids))
        second_frames = len(gt_frames.get(second_id, set())) if second_id is not None else 0
        if not eligible_rows or (coverage is not None and coverage < 0.80):
            state = "UNCERTAIN"
            state_reason = "NO_OR_LOW_GT_COVERAGE"
        elif len(gt_counts) == 1:
            state = "CLEAN_SINGLE_INSTANCE"
            state_reason = "ONE_ELIGIBLE_GT_ID"
        elif second_count >= 2 and second_frames >= 2:
            state = "ALREADY_CONTAMINATED"
            state_reason = "SECOND_GT_PERSISTS_AT_LEAST_2_OBS_AND_2_FRAMES"
        else:
            state = "UNCERTAIN"
            state_reason = "MULTI_GT_LOW_SUPPORT"

        sorted_pixel_ids = sorted(
            gt_pixel_counts_top2, key=lambda value: (-gt_pixel_counts_top2[value], value)
        )
        known_top2_pixels = sum(gt_pixel_counts_top2.values())
        dominant_pixel_count = (
            gt_pixel_counts_top2[sorted_pixel_ids[0]] if sorted_pixel_ids else 0
        )
        non_dominant_pixel_fraction = (
            1 - safe_ratio(dominant_pixel_count, known_top2_pixels)
            if known_top2_pixels
            else None
        )
        persistent_pixel_contamination = bool(
            pixel_mixed_count >= 2
            and len(pixel_mixed_frames) >= 2
            and non_dominant_pixel_fraction is not None
            and non_dominant_pixel_fraction >= 0.05
        )
        if persistent_pixel_contamination:
            pixel_state = "ALREADY_CONTAMINATED"
        elif pixel_mixed_count == 0:
            pixel_state = "NO_PIXEL_MIXTURE_DETECTED"
        else:
            pixel_state = "UNCERTAIN_LOW_SUPPORT_PIXEL_MIXTURE"

        if not eligible_rows or (coverage is not None and coverage < 0.80):
            joint_state = "UNCERTAIN"
            joint_state_reason = "NO_OR_LOW_GT_COVERAGE"
        elif state == "ALREADY_CONTAMINATED" or persistent_pixel_contamination:
            joint_state = "ALREADY_CONTAMINATED"
            joint_state_reason = (
                "PERSISTENT_MULTI_GT_MEMBERSHIP"
                if state == "ALREADY_CONTAMINATED"
                else "PERSISTENT_PIXEL_MIXTURE_2OBS_2FRAMES_AND_5PCT"
            )
        elif state == "CLEAN_SINGLE_INSTANCE" and pixel_mixed_count == 0:
            joint_state = "CLEAN_SINGLE_INSTANCE"
            joint_state_reason = "ONE_MEMBER_GT_AND_NO_PIXEL_MIXTURE"
        else:
            joint_state = "UNCERTAIN"
            joint_state_reason = "LOW_SUPPORT_MEMBERSHIP_OR_PIXEL_MIXTURE"

        component_details: list[dict[str, Any]] = []
        for gt_id in sorted_ids:
            frames = sorted(gt_frames[gt_id])
            centers = gt_centers[gt_id]
            labels = Counter(
                str(self.gt[uid].get("gt_top_label") or "UNKNOWN")
                for uid in gt_members[gt_id]
                if uid in self.gt
            )
            classes = Counter(
                str(self.observations.get(uid, {}).get("detection_label") or self.observations.get(uid, {}).get("class_name") or "UNKNOWN")
                for uid in gt_members[gt_id]
            )
            component_details.append(
                {
                    "gt_id": gt_id,
                    "count": gt_counts[gt_id],
                    "fraction": safe_ratio(gt_counts[gt_id], len(eligible_rows)),
                    "gt_labels": sorted_counter(labels),
                    "unique_frame_count": len(frames),
                    "frame_range": [frames[0], frames[-1]] if frames else None,
                    "detector_class_counts": sorted_counter(classes),
                    "centroid_mean": mean_vector(centers),
                }
            )

        result = {
            "member_observation_count": len(member_uids),
            "eligible_gt_observation_count": len(eligible_rows),
            "missing_or_ineligible_gt_count": gt_missing,
            "gt_coverage": coverage,
            "gt_distinct_id_count": len(gt_counts),
            "gt_id_counts": sorted_counter(gt_counts, len(eligible_rows)),
            "projected_gt_pixel_counts_top2": sorted_counter(
                gt_pixel_counts_top2, sum(gt_pixel_counts_top2.values())
            ),
            "projected_gt_known_pixel_count_top2": sum(gt_pixel_counts_top2.values()),
            "projected_gt_non_dominant_pixel_fraction_top2": non_dominant_pixel_fraction,
            "gt_label_counts": sorted_counter(label_counts, len(eligible_rows)),
            "dominant_gt_id": dominant_id,
            "dominant_gt_fraction": safe_ratio(gt_counts.get(dominant_id, 0), len(eligible_rows)) if dominant_id is not None else None,
            "second_gt_id": second_id,
            "second_gt_count": second_count,
            "second_gt_fraction": safe_ratio(second_count, len(eligible_rows)),
            "second_gt_unique_frame_count": second_frames,
            "objective_target_pre_state": state,
            "objective_target_pre_state_reason": state_reason,
            "membership_identity_pre_state": state,
            "membership_identity_pre_state_reason": state_reason,
            "pixel_contamination_state": pixel_state,
            "persistent_pixel_contamination_2obs_2frames_5pct": persistent_pixel_contamination,
            "pixel_mixed_unique_frame_count": len(pixel_mixed_frames),
            "objective_target_pre_state_joint": joint_state,
            "objective_target_pre_state_joint_reason": joint_state_reason,
            "strict_multi_gt": len(gt_counts) >= 2,
            "material_multi_gt_5pct": bool(len(gt_counts) >= 2 and safe_ratio(second_count, len(eligible_rows)) is not None and safe_ratio(second_count, len(eligible_rows)) >= 0.05),
            "persistent_multi_gt_2obs_2frames": bool(len(gt_counts) >= 2 and second_count >= 2 and second_frames >= 2),
            "pixel_mixed_mask_count": pixel_mixed_count,
            "pixel_mixed_mask_fraction": safe_ratio(pixel_mixed_count, len(eligible_rows)),
            "pixel_mixed_mask_examples": sorted(
                pixel_mixed_examples, key=lambda item: (int(item["frame_idx"]), str(item["obs_uid"]))
            )[:10],
            "two_foreground_mask_count": two_foreground_count,
            "mean_mask_gt_purity": statistics.mean(purities) if purities else None,
            "detector_class_counts": sorted_counter(detector_classes, len(eligible_rows)),
            "component_details": component_details,
            "top2_coobserved_frame_count": len(coobserved_frames),
            "top2_coobserved_frame_examples": coobserved_frames[:10],
            "temporal_gt_transition_count": transitions,
            "top2_centroid_distance_m": centroid_distance,
            "object_bbox_diagonal_m": diagonal,
            "top2_centroid_distance_over_object_diagonal": safe_ratio(centroid_distance, diagonal) if centroid_distance is not None and diagonal else None,
        }
        if cache_key:
            self.composition_cache[cache_key] = result
        return result

    def version_composition(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return self.composition(
            [str(uid) for uid in row.get("member_observation_uids") or []],
            cache_key=str(row["object_version_uid"]),
            bbox_extent=row.get("bbox_extent"),
        )

    def earliest_contamination(self, selected_version: dict[str, Any] | None) -> dict[str, Any] | None:
        if selected_version is None:
            return None
        selected_seq = event_sequence(selected_version.get("trigger_event_uid")) or 10**12
        selected_version_number = version_index(selected_version.get("object_version_uid"))
        history = [
            row
            for row in self.versions_by_object.get(str(selected_version["object_uid"]), [])
            if (event_sequence(row.get("trigger_event_uid")) or -1) <= selected_seq
            and version_index(row.get("object_version_uid")) <= selected_version_number
        ]
        first_strict = None
        first_persistent = None
        first_pixel_mixed = None
        first_persistent_pixel_contamination = None
        previous_members: set[str] = set()
        previous_ids: set[int] = set()
        previous_version_uid: str | None = None
        previous_comp: dict[str, Any] | None = None
        for row in history:
            comp = self.version_composition(row) or {}
            ids = {int(item["value"]) for item in comp.get("gt_id_counts") or []}
            members = {str(uid) for uid in row.get("member_observation_uids") or []}
            if comp.get("pixel_mixed_mask_count") and first_pixel_mixed is None:
                first_pixel_mixed = {
                    "object_version_uid": row.get("object_version_uid"),
                    "trigger_event_uid": row.get("trigger_event_uid"),
                    "trigger_event_sequence": event_sequence(row.get("trigger_event_uid")),
                    "operation": row.get("operation"),
                    "pixel_mixed_mask_count": comp.get("pixel_mixed_mask_count"),
                    "pixel_mixed_mask_examples": comp.get("pixel_mixed_mask_examples") or [],
                    "linear_predecessor_version_uid": previous_version_uid,
                    "linear_predecessor_pixel_mixed_mask_count": (
                        previous_comp.get("pixel_mixed_mask_count") if previous_comp else None
                    ),
                }
            if (
                comp.get("persistent_pixel_contamination_2obs_2frames_5pct")
                and first_persistent_pixel_contamination is None
            ):
                first_persistent_pixel_contamination = {
                    "object_version_uid": row.get("object_version_uid"),
                    "trigger_event_uid": row.get("trigger_event_uid"),
                    "trigger_event_sequence": event_sequence(row.get("trigger_event_uid")),
                    "operation": row.get("operation"),
                    "pixel_mixed_mask_count": comp.get("pixel_mixed_mask_count"),
                    "pixel_mixed_unique_frame_count": comp.get("pixel_mixed_unique_frame_count"),
                    "projected_gt_non_dominant_pixel_fraction_top2": comp.get(
                        "projected_gt_non_dominant_pixel_fraction_top2"
                    ),
                    "pixel_mixed_mask_examples": comp.get("pixel_mixed_mask_examples") or [],
                    "linear_predecessor_version_uid": previous_version_uid,
                }
            if comp.get("strict_multi_gt") and first_strict is None:
                first_strict = self._contamination_transition(row, comp, ids - previous_ids, members - previous_members)
                first_strict["linear_predecessor_version_uid"] = previous_version_uid
                first_strict["linear_predecessor_composition"] = previous_comp
            if comp.get("persistent_multi_gt_2obs_2frames") and first_persistent is None:
                first_persistent = self._contamination_transition(row, comp, ids - previous_ids, members - previous_members)
                first_persistent["linear_predecessor_version_uid"] = previous_version_uid
                first_persistent["linear_predecessor_composition"] = previous_comp
            previous_members = members
            previous_ids = ids
            previous_version_uid = str(row.get("object_version_uid") or "") or None
            previous_comp = comp
        return {
            "first_pixel_mixed_mask": first_pixel_mixed,
            "first_persistent_pixel_contamination": first_persistent_pixel_contamination,
            "first_strict_multi_gt": first_strict,
            "first_persistent_multi_gt": first_persistent,
        }

    def _contamination_transition(
        self,
        row: dict[str, Any],
        comp: dict[str, Any],
        introduced_ids: set[int],
        added_members: set[str],
    ) -> dict[str, Any]:
        added_gt = Counter()
        examples: list[dict[str, Any]] = []
        introduced_examples: list[dict[str, Any]] = []
        for uid in sorted(added_members, key=lambda value: (frame_index(value) or -1, value)):
            gt = self.gt.get(uid)
            if not gt or not gt.get("gt_assignment_eligible") or gt.get("gt_top_id") is None:
                continue
            gt_id = int(gt["gt_top_id"])
            added_gt[gt_id] += 1
            assoc = self.assoc_by_obs.get(uid, {})
            example = {
                "obs_uid": uid,
                "frame_idx": gt.get("frame_idx"),
                "gt_id": gt_id,
                "gt_label": gt.get("gt_top_label"),
                "gt_purity": gt.get("gt_purity"),
                "gt_second_id": gt.get("gt_second_id"),
                "gt_second_label": gt.get("gt_second_label"),
                "gt_second_fraction": gt.get("gt_second_fraction"),
                "mask_mixed": gt.get("mask_mixed"),
                "mask_two_foreground": gt.get("mask_two_foreground"),
                "association_event_uid": assoc.get("event_uid"),
                "association_event_sequence": assoc.get("event_sequence"),
                "association_decision": assoc.get("decision"),
                "association_target_object_uid": assoc.get("target_object_uid"),
                "association_top1_score": assoc.get("top1_score"),
                "association_top2_score": assoc.get("top2_score"),
                "association_margin": assoc.get("margin"),
                "association_sim_threshold": assoc.get("sim_threshold"),
                "association_top1_minus_threshold": (
                    float(assoc["top1_score"]) - float(assoc["sim_threshold"])
                    if assoc.get("top1_score") is not None and assoc.get("sim_threshold") is not None
                    else None
                ),
            }
            if len(examples) < 8:
                examples.append(example)
            if gt_id in introduced_ids and len(introduced_examples) < 8:
                introduced_examples.append(example)
        operation = str(row.get("operation") or "")
        return {
            "object_version_uid": row.get("object_version_uid"),
            "trigger_event_uid": row.get("trigger_event_uid"),
            "trigger_event_sequence": event_sequence(row.get("trigger_event_uid")),
            "trigger_frame_idx": frame_index(row.get("frame_uid")),
            "operation": row.get("operation"),
            "trace_kind": (
                "DIRECT_OBSERVATION_ASSOCIATION"
                if operation == "OBS_ASSOCIATE" and introduced_examples
                else "OBJECT_MERGE_REQUIRES_PARENT_DAG_BACKTRACE"
                if operation == "OBJECT_MERGE"
                else "OTHER_OR_AMBIGUOUS_TRANSITION"
            ),
            "parent_version_uids": row.get("parent_version_uids") or [],
            "introduced_gt_ids": sorted(int(value) for value in introduced_ids),
            "added_member_count": len(added_members),
            "added_member_gt_counts": sorted_counter(added_gt),
            "added_member_examples": examples,
            "introduced_gt_member_examples": introduced_examples,
            "composition_at_transition": {
                key: comp.get(key)
                for key in (
                    "member_observation_count",
                    "eligible_gt_observation_count",
                    "gt_id_counts",
                    "objective_target_pre_state",
                    "objective_target_pre_state_joint",
                    "persistent_multi_gt_2obs_2frames",
                    "persistent_pixel_contamination_2obs_2frames_5pct",
                    "projected_gt_pixel_counts_top2",
                )
            },
        }

    def packet_display_audit(self, case_uid: str, object_uid: str | None) -> dict[str, Any] | None:
        if not object_uid:
            return None
        packet = self.packet_private_by_case.get(case_uid)
        if not packet:
            return None
        candidates = packet.get("candidates") or packet.get("public", {}).get("candidates") or []
        candidate = next((row for row in candidates if str(row.get("object_uid")) == object_uid), None)
        if candidate is None:
            return None
        displayed = [str(row["obs_uid"]) for row in candidate.get("history_displayed") or []]
        members = [str(uid) for uid in candidate.get("history_member_observation_uids") or []]
        displayed_comp = self.composition(displayed, cache_key=f"display:{case_uid}:{object_uid}")
        member_comp = self.composition(members, cache_key=f"packet:{case_uid}:{object_uid}")
        full_ids = {item["value"] for item in member_comp.get("gt_id_counts") or []}
        shown_ids = {item["value"] for item in displayed_comp.get("gt_id_counts") or []}
        return {
            "source_path": packet.get("_source_path"),
            "candidate_code": candidate.get("code"),
            "packet_object_version_uid": candidate.get("object_version_uid"),
            "displayed_history_count": len(displayed),
            "packet_member_count": len(members),
            "displayed_observation_uids": displayed,
            "displayed_composition": displayed_comp,
            "packet_full_composition": member_comp,
            "gt_ids_omitted_from_display": sorted(full_ids - shown_ids),
        }

    def current_mask(self, event: dict[str, Any]) -> dict[str, Any]:
        uid = str(event["obs_uid"])
        gt = self.gt.get(uid, {})
        obs = self.observations.get(uid, {})
        return {
            "human_observation_quality": event.get("human", {}).get("observation_quality"),
            "human_identity_evidence_status": event.get("human", {}).get("identity_evidence_status"),
            "human_identity_routing_eligible": event.get("human", {}).get("identity_routing_eligible"),
            "gt_assignment_eligible": gt.get("gt_assignment_eligible"),
            "gt_id": gt.get("gt_top_id"),
            "gt_label": gt.get("gt_top_label"),
            "gt_purity": gt.get("gt_purity"),
            "gt_purity_bin": purity_bin(gt.get("gt_purity")),
            "gt_second_id": gt.get("gt_second_id"),
            "gt_second_fraction": gt.get("gt_second_fraction"),
            "mask_mixed": gt.get("mask_mixed"),
            "mask_two_foreground": gt.get("mask_two_foreground"),
            "detector_class": obs.get("detection_label") or obs.get("class_name"),
            "bbox_3d_center": obs.get("bbox_3d_center"),
            "boundary_touch_ratio": obs.get("boundary_touch_ratio"),
        }

    def relevant_versions(
        self, event: dict[str, Any], assoc: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        sequence = int(assoc.get("event_sequence") or event_sequence(event.get("event_uid")) or -1)
        original_uid = str(event.get("original_target_uid") or "")
        original = None
        if original_uid:
            original = self.version_at(original_uid, sequence, str(assoc.get("target_object_version_before") or "") or None)

        legal_uids = [str(uid) for uid in event.get("human", {}).get("legal_target_uids") or []]
        legal_versions = []
        for uid in legal_uids:
            row = self.version_at(uid, sequence)
            if row:
                legal_versions.append(row)
        return original, legal_versions

    def native_self_heal(
        self,
        event: dict[str, Any],
        original_version: dict[str, Any] | None,
        legal_versions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        obs_uid = str(event["obs_uid"])
        owner_uid = self.native_owner_by_obs.get(obs_uid)
        if owner_uid is None:
            return {"status": "UNDETERMINED_CURRENT_OBSERVATION_HAS_NO_FINAL_OWNER"}
        owner_members = self.final_membership.get(owner_uid, [])
        owner_comp = self.composition(owner_members, cache_key=f"native:{owner_uid}")
        current_gt = self.gt.get(obs_uid, {}).get("gt_top_id")
        current_gt_count = 0
        if current_gt is not None:
            for item in owner_comp.get("gt_id_counts") or []:
                if str(item["value"]) == str(current_gt):
                    current_gt_count = int(item["count"])
                    break
        owner_precision = safe_ratio(current_gt_count, owner_comp.get("eligible_gt_observation_count") or 0)
        owner_joint_state = owner_comp.get("objective_target_pre_state_joint")
        correct_action = str(event.get("human", {}).get("correct_action_type") or "")
        original_members = set(str(uid) for uid in (original_version or {}).get("member_observation_uids") or [])
        legal_member_sets = [set(str(uid) for uid in row.get("member_observation_uids") or []) for row in legal_versions]
        original_overlap = len(original_members.intersection(owner_members))
        legal_overlaps = [len(members.intersection(owner_members)) for members in legal_member_sets]

        if correct_action == "NEW":
            structural = original_overlap == 0
            if structural and owner_joint_state == "CLEAN_SINGLE_INSTANCE":
                status = "SELF_HEALED_JOINT_CLEAN_SEPARATION"
            elif structural:
                status = "SELF_HEALED_SEPARATION_BUT_FINAL_OWNER_CONTAMINATED_OR_UNCERTAIN"
            else:
                status = "NOT_SELF_HEALED_REMAINS_WITH_ORIGINAL_LINEAGE"
        elif correct_action == "ATTACH_EXISTING":
            structural = bool(legal_overlaps and max(legal_overlaps) > 0)
            if structural and owner_joint_state == "CLEAN_SINGLE_INSTANCE":
                status = "SELF_HEALED_JOINT_CLEAN_JOIN_TO_LEGAL_LINEAGE"
            elif structural:
                status = "SELF_HEALED_TOPOLOGY_ONLY_FINAL_OWNER_CONTAMINATED_OR_UNCERTAIN"
            else:
                status = "NOT_SELF_HEALED_NO_JOIN_TO_LEGAL_LINEAGE"
        else:
            structural = False
            status = "UNDETERMINED_CORRECT_ACTION"
        return {
            "status": status,
            "structural_self_heal": structural,
            "native_final_owner_uid": owner_uid,
            "native_final_owner_member_count": len(owner_members),
            "native_final_owner_current_gt_precision": owner_precision,
            "native_final_owner_joint_state": owner_joint_state,
            "native_final_owner_composition": owner_comp,
            "original_prestate_member_overlap": original_overlap,
            "legal_prestate_member_overlaps": legal_overlaps,
        }

    def future_stability(self, episode: dict[str, Any], current_obs_uid: str) -> dict[str, Any]:
        future = episode.get("future_evidence") or {}
        suffix = [str(row["obs_uid"]) for row in future.get("independent_views_suffix") or []]
        window = [str(row["obs_uid"]) for row in future.get("independent_views_window") or []]
        owners = [self.native_owner_by_obs.get(uid) for uid in [current_obs_uid] + suffix]
        known = [owner for owner in owners if owner]
        owner_counts = Counter(known)
        current_owner = self.native_owner_by_obs.get(current_obs_uid)
        return {
            "status": future.get("status"),
            "independent_views_window_count": future.get("independent_views_window_count"),
            "independent_views_suffix_count": future.get("independent_views_suffix_count"),
            "has_two_views_window": future.get("has_two_views_window"),
            "has_three_views_window": future.get("has_three_views_window"),
            "window_observation_uids": window,
            "suffix_observation_count_with_current": 1 + len(suffix),
            "native_final_distinct_owner_count": len(owner_counts),
            "native_final_owner_counts": sorted_counter(owner_counts, len(known)),
            "current_owner_share_among_suffix_views": safe_ratio(owner_counts.get(current_owner, 0), len(known)) if current_owner else None,
            "note": "Uses post-run ownership and is diagnostic future information, not online repair input.",
        }

    def infer_primitive(
        self,
        event: dict[str, Any],
        current: dict[str, Any],
        original_comp: dict[str, Any] | None,
        legal_comps: list[dict[str, Any]],
        earliest_original: dict[str, Any] | None,
    ) -> str:
        quality = str(current.get("human_observation_quality") or "")
        if quality.startswith("MIXED") or current.get("mask_two_foreground"):
            return "MASK_SPLIT_OR_GRANULARITY_REPAIR_BEFORE_IDENTITY_ROUTING"
        correct_action = str(event.get("human", {}).get("correct_action_type") or "")
        if correct_action == "NEW":
            root_transition = (earliest_original or {}).get("first_strict_multi_gt") or {}
            root_examples = root_transition.get("introduced_gt_member_examples") or []
            if root_transition.get("trace_kind") == "DIRECT_OBSERVATION_ASSOCIATION" and any(
                bool(item.get("mask_mixed") or item.get("mask_two_foreground"))
                for item in root_examples
            ):
                return "BACKTRACK_TO_SEGMENTATION_ROOT_SPLIT_OBSERVATION_QUARANTINE_AND_REPLAY"
            if original_comp and original_comp.get("objective_target_pre_state_joint") == "ALREADY_CONTAMINATED":
                return "BACKTRACK_OR_SPLIT_CONTAMINATED_TARGET_PLUS_PERSISTENT_NEW_BOUNDARY"
            if original_comp and original_comp.get("objective_target_pre_state_joint") == "CLEAN_SINGLE_INSTANCE":
                return "PERSISTENT_CREATE_BOUNDARY"
            return "TEMPORARY_IDENTITY_CLUSTER_AND_DELAYED_DECISION"
        if correct_action == "ATTACH_EXISTING":
            states = [str(comp.get("objective_target_pre_state_joint")) for comp in legal_comps]
            current_gt = current.get("gt_id")
            prior_same_gt_count = 0
            if original_comp is not None and current_gt is not None:
                prior_same_gt_count = next(
                    (
                        int(item.get("count") or 0)
                        for item in original_comp.get("gt_id_counts") or []
                        if str(item.get("value")) == str(current_gt)
                    ),
                    0,
                )
            original_has_prior_same_id_contamination = bool(
                original_comp
                and original_comp.get("objective_target_pre_state_joint") == "ALREADY_CONTAMINATED"
                and prior_same_gt_count > 0
            )
            if any(state == "ALREADY_CONTAMINATED" for state in states):
                if original_has_prior_same_id_contamination:
                    return "SPLIT_LEGAL_CANDIDATE_AND_EXTRACT_WRONG_LINEAGE_THEN_REUNIFY"
                return "SPLIT_CONTAMINATED_CANDIDATE_THEN_REDIRECT_OR_REUNIFY"
            if any(state == "CLEAN_SINGLE_INSTANCE" for state in states):
                if original_has_prior_same_id_contamination:
                    return "EXTRACT_PRIOR_SAME_ID_FROM_WRONG_TARGET_THEN_REDIRECT_LINEAGE"
                return "DIRECT_REDIRECT_TO_VERIFIED_CLEAN_TARGET"
            if original_has_prior_same_id_contamination:
                return "EXTRACT_PRIOR_SAME_ID_TO_TEMP_CLUSTER_AND_DELAY_LEGAL_COMMIT"
            return "TEMPORARY_IDENTITY_CLUSTER_AND_DELAYED_DECISION"
        return "NO_REPAIR_OR_UNDETERMINED"

    def build_event_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for event in self.events:
            case_uid = str(event["case_uid"])
            assoc = self.assoc_by_event.get(str(event["event_uid"]), {})
            original_version, legal_versions = self.relevant_versions(event, assoc)
            original_comp = self.version_composition(original_version)
            legal_comps = [self.version_composition(row) or {} for row in legal_versions]
            current = self.current_mask(event)
            human = event.get("human", {})
            is_error = bool(human.get("is_error"))
            episode = self.error_by_case.get(case_uid)

            packet_target = self.packet_display_audit(case_uid, str(event.get("original_target_uid") or "") or None)
            frozen_target = next(
                (
                    candidate
                    for candidate in event.get("frozen_candidates") or []
                    if str(candidate.get("object_uid")) == str(event.get("original_target_uid"))
                ),
                None,
            )
            exact_target_uid = str((original_version or {}).get("object_version_uid") or "") or None
            frozen_target_uid = str((frozen_target or {}).get("object_version_uid") or "") or None
            stale_delta = None
            if exact_target_uid and frozen_target_uid and exact_target_uid in self.versions and frozen_target_uid in self.versions:
                stale_delta = len(self.versions[exact_target_uid].get("member_observation_uids") or []) - len(
                    self.versions[frozen_target_uid].get("member_observation_uids") or []
                )

            score = {
                "top1_score": assoc.get("top1_score"),
                "top2_score": assoc.get("top2_score"),
                "top1_top2_margin": assoc.get("margin"),
                "sim_threshold": assoc.get("sim_threshold"),
                "top1_minus_new_threshold": (
                    float(assoc["top1_score"]) - float(assoc["sim_threshold"])
                    if assoc.get("top1_score") is not None and assoc.get("sim_threshold") is not None
                    else None
                ),
                "correct_existing_best_rank": (episode or {}).get("candidate_coverage", {}).get("best_existing_rank"),
                "correct_existing_target_ranks": (episode or {}).get("candidate_coverage", {}).get("target_ranks"),
                "covered_at_5_plus_new": (episode or {}).get("candidate_coverage", {}).get("covered_at_5_plus_new"),
                "new_was_overridden": bool(human.get("correct_action_type") == "NEW" and event.get("original_action_type") == "ATTACH_EXISTING"),
            }
            row = {
                "case_uid": case_uid,
                "source_batch": event.get("source_batch"),
                "sample_kind": event.get("sample_kind"),
                "event_uid": event.get("event_uid"),
                "event_sequence": assoc.get("event_sequence") or event_sequence(event.get("event_uid")),
                "event_frame_idx": event.get("event_frame_idx"),
                "obs_uid": event.get("obs_uid"),
                "is_error": is_error,
                "routing_label": human.get("routing_label"),
                "original_action_type": event.get("original_action_type"),
                "correct_action_type": human.get("correct_action_type"),
                "original_target_uid": event.get("original_target_uid"),
                "human_target_pre_state": human.get("target_pre_state"),
                "human_confidence": human.get("confidence"),
                "current_mask": current,
                "association_scores": score,
                "original_target_exact_tminus_version_uid": exact_target_uid,
                "original_target_frozen_packet_version_uid": frozen_target_uid,
                "frozen_vs_exact_version_mismatch": bool(exact_target_uid and frozen_target_uid and exact_target_uid != frozen_target_uid),
                "frozen_vs_exact_member_count_delta": stale_delta,
                "original_target_composition": original_comp,
                "legal_target_versions": [str(item["object_version_uid"]) for item in legal_versions],
                "legal_target_compositions": legal_comps,
                "packet_original_target_display_audit": packet_target,
            }

            if is_error and episode:
                earliest_original = self.earliest_contamination(original_version)
                earliest_strict = (earliest_original or {}).get("first_strict_multi_gt") or {}
                earliest_examples = earliest_strict.get("introduced_gt_member_examples") or []
                direct_trace = earliest_strict.get("trace_kind") == "DIRECT_OBSERVATION_ASSOCIATION"
                earliest_assoc_uid = (
                    next(
                        (str(item.get("association_event_uid")) for item in earliest_examples if item.get("association_event_uid")),
                        None,
                    )
                    if direct_trace
                    else None
                )
                earliest_assoc_sequence = (
                    next(
                        (int(item.get("association_event_sequence")) for item in earliest_examples if item.get("association_event_sequence") is not None),
                        None,
                    )
                    if direct_trace
                    else None
                )
                row.update(
                    {
                        "causal_role_raw": episode.get("causal_role"),
                        "causal_role": normalize_causal_role(episode.get("causal_role")),
                        "causal_group_uid": episode.get("causal_group_uid"),
                        "linked_human_root_event_uid": episode.get("linked_human_root_event_uid"),
                        "offline_class_relation": episode.get("offline_identity_audit", {}).get("class_relation"),
                        "offline_observation_gt_id": episode.get("offline_identity_audit", {}).get("obs_gt_id"),
                        "future_identity_stability": self.future_stability(episode, str(event["obs_uid"])),
                        "native_self_heal": self.native_self_heal(event, original_version, legal_versions),
                        "earliest_original_target_contamination": earliest_original,
                        "ledger_earliest_strict_contamination_association_event_uid": earliest_assoc_uid,
                        "ledger_earliest_strict_contamination_trace_kind": earliest_strict.get("trace_kind"),
                        "ledger_earliest_strict_contamination_transition_event_uid": earliest_strict.get("trigger_event_uid"),
                        "ledger_earliest_strict_contamination_in_174_labels": (
                            earliest_assoc_uid in self.labeled_event_uids if earliest_assoc_uid else None
                        ),
                        "anchor_minus_earliest_contamination_event_gap": (
                            int(row["event_sequence"]) - earliest_assoc_sequence
                            if earliest_assoc_sequence is not None and row.get("event_sequence") is not None
                            else None
                        ),
                        "earliest_legal_target_contamination": [self.earliest_contamination(item) for item in legal_versions],
                        "repair_primitive_inference": self.infer_primitive(
                            event, current, original_comp, legal_comps, earliest_original
                        ),
                    }
                )
                error_rows.append(row)

            rows.append(row)

            relevant: dict[tuple[str, str], dict[str, Any]] = {}
            for candidate in event.get("frozen_candidates") or []:
                object_uid = str(candidate.get("object_uid") or "")
                version = self.version_at(object_uid, int(row["event_sequence"]), str(candidate.get("object_version_uid") or "") or None)
                if version:
                    relevant[(object_uid, "FROZEN_DISPLAY")] = version
            if original_version:
                relevant[(str(original_version["object_uid"]), "ORIGINAL_SELECTED_EXACT_TMINUS")] = original_version
            for version in legal_versions:
                relevant[(str(version["object_uid"]), "CORRECT_LEGAL_EXACT_TMINUS")] = version
            if is_error:
                for (object_uid, role), version in relevant.items():
                    candidate_rows.append(
                        {
                            "case_uid": case_uid,
                            "event_sequence": row["event_sequence"],
                            "candidate_role": role,
                            "object_uid": object_uid,
                            "object_version_uid": version.get("object_version_uid"),
                            "composition": self.version_composition(version),
                            "earliest_contamination": self.earliest_contamination(version),
                            "packet_display_audit": self.packet_display_audit(case_uid, object_uid),
                        }
                    )
        return rows, error_rows, candidate_rows

    def replay_audit(self) -> dict[str, Any]:
        replay_runs: list[dict[str, Any]] = []
        metrics_paths = sorted(self.analysis_root.glob("oracle_minimal_replay*/**/metrics.json"))
        for path in metrics_paths:
            metrics = read_json(path)
            branches: dict[str, Any] = {}
            for name, branch in (metrics.get("branches") or {}).items():
                collateral = branch.get("collateral") or {}
                root = branch.get("root_action") or {}
                runtime = branch.get("runtime_invariants") or {}
                branches[name] = {
                    "root_action_correct": root.get("correct"),
                    "endpoint_correct": branch.get("endpoint_correct"),
                    "geometry_valid": branch.get("geometry_valid"),
                    "active_object_count": branch.get("active_object_count"),
                    "replayed_observation_count": branch.get("replayed_observation_count"),
                    "closure_effective_observation_count": branch.get("closure_effective_observation_count"),
                    "changed_outside_observation_count": collateral.get("changed_outside_observation_count"),
                    "outside_partition_exact_to_native": collateral.get("outside_partition_exact_to_native"),
                    "runtime_invariants_pass": runtime.get("pass"),
                    "runtime_ms": branch.get("runtime_ms"),
                }
            replay_runs.append(
                {
                    "case_uid": metrics.get("case_uid"),
                    "run_name": path.parent.parent.name,
                    "metrics_path": str(path),
                    "routing_label": metrics.get("routing_label"),
                    "correct_action_type": metrics.get("correct_action_type"),
                    "b0r_exact_partition_parity": metrics.get("b0r_exact_partition_parity"),
                    "interpretation": metrics.get("interpretation"),
                    "branches": branches,
                }
            )

        create = read_json(self.paths["create_partition_audit"])
        full_instance: list[dict[str, Any]] = []
        latest_metrics_by_case: dict[str, Path] = {}
        for path in metrics_paths:
            latest_metrics_by_case[str(read_json(path).get("case_uid"))] = path
        for case in create.get("cases") or []:
            case_uid = str(case["case_uid"])
            metrics_path = latest_metrics_by_case.get(case_uid)
            for branch_name, branch in (case.get("branches") or {}).items():
                new_owner = branch.get("new_owner") or {}
                target_owner = branch.get("target_owner") or {}
                expected_id = new_owner.get("expected_gt_id")
                new_expected = int(new_owner.get("expected_gt_observation_count") or 0)
                target_expected = 0
                for item in target_owner.get("gt_id_counts") or []:
                    if str(item.get("value")) == str(expected_id):
                        target_expected = int(item.get("count") or 0)
                        break
                eligible_new = int(new_owner.get("eligible_gt_observation_count") or 0)
                dominant_target = max((int(item.get("count") or 0) for item in target_owner.get("gt_id_counts") or []), default=0)
                eligible_target = int(target_owner.get("eligible_gt_observation_count") or 0)
                global_expected = sum(
                    1
                    for gt in self.gt.values()
                    if gt.get("gt_assignment_eligible") and str(gt.get("gt_top_id")) == str(expected_id)
                )
                global_owner_counts: Counter[str] = Counter()
                new_owner_uid = branch.get("new_owner_uid")
                target_owner_uid = branch.get("target_owner_uid")
                branch_membership: dict[str, list[str]] = {}
                if metrics_path and expected_id is not None:
                    branch_path = metrics_path.parent / "branches" / f"{branch_name}.json.gz"
                    if branch_path.exists():
                        with gzip.open(branch_path, "rt", encoding="utf-8") as handle:
                            state = json.load(handle)
                        branch_membership = {
                            str(owner_uid): [str(uid) for uid in members]
                            for owner_uid, members in (state.get("membership") or {}).items()
                        }
                        for owner_uid, members in branch_membership.items():
                            count = sum(
                                1
                                for uid in members
                                if uid in self.gt
                                and self.gt[uid].get("gt_assignment_eligible")
                                and str(self.gt[uid].get("gt_top_id")) == str(expected_id)
                            )
                            if count:
                                global_owner_counts[str(owner_uid)] += count
                precision = safe_ratio(new_expected, eligible_new)
                global_recall = safe_ratio(global_owner_counts.get(str(new_owner_uid), 0), global_expected)
                pair_recall = safe_ratio(new_expected, new_expected + target_expected)
                new_joint_comp = self.composition(
                    branch_membership.get(str(new_owner_uid), []),
                    cache_key=f"replay:{case_uid}:{branch_name}:new:{new_owner_uid}",
                )
                target_joint_comp = self.composition(
                    branch_membership.get(str(target_owner_uid), []),
                    cache_key=f"replay:{case_uid}:{branch_name}:target:{target_owner_uid}",
                )
                full_instance.append(
                    {
                        "case_uid": case_uid,
                        "branch": branch_name,
                        "expected_gt_id": expected_id,
                        "new_owner_uid": new_owner_uid,
                        "new_owner_member_count": new_owner.get("member_observation_count"),
                        "new_owner_expected_gt_count": new_expected,
                        "new_cluster_precision": precision,
                        "new_owner_joint_state": new_joint_comp.get("objective_target_pre_state_joint"),
                        "new_owner_pixel_mixed_mask_count": new_joint_comp.get("pixel_mixed_mask_count"),
                        "new_owner_non_dominant_pixel_fraction_top2": new_joint_comp.get(
                            "projected_gt_non_dominant_pixel_fraction_top2"
                        ),
                        "same_gt_residual_in_original_target_count": target_expected,
                        "two_owner_capture_recall": pair_recall,
                        "global_expected_gt_observation_count": global_expected,
                        "new_owner_global_observation_recall": global_recall,
                        "new_owner_f1_global": (
                            2 * precision * global_recall / (precision + global_recall)
                            if precision is not None and global_recall is not None and precision + global_recall
                            else None
                        ),
                        "global_expected_gt_owner_count": len(global_owner_counts),
                        "global_expected_gt_owner_counts": sorted_counter(global_owner_counts, global_expected),
                        "original_target_member_count": target_owner.get("member_observation_count"),
                        "original_target_eligible_gt_count": eligible_target,
                        "original_target_dominant_gt_purity": safe_ratio(dominant_target, eligible_target),
                        "original_target_residual_contamination_rate": (
                            1 - safe_ratio(dominant_target, eligible_target)
                            if safe_ratio(dominant_target, eligible_target) is not None
                            else None
                        ),
                        "original_target_joint_state": target_joint_comp.get(
                            "objective_target_pre_state_joint"
                        ),
                        "original_target_pixel_mixed_mask_count": target_joint_comp.get(
                            "pixel_mixed_mask_count"
                        ),
                        "original_target_non_dominant_pixel_fraction_top2": target_joint_comp.get(
                            "projected_gt_non_dominant_pixel_fraction_top2"
                        ),
                        "endpoint_correct": branch.get("endpoint_correct"),
                        "changed_outside_observation_count": branch.get("changed_outside_observation_count"),
                        "runtime_invariants_pass": branch.get("runtime_invariants_pass"),
                        "note": "Precision/recall are observation-membership metrics. two_owner_capture_recall uses new+original-target owners; global recall uses all corrected-GT observations of that physical instance.",
                    }
                )
        return {"replay_runs": replay_runs, "create_full_instance_metrics": full_instance}

    def cross_statistics(
        self,
        event_rows: list[dict[str, Any]],
        error_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        attach_rows = [row for row in event_rows if row.get("original_action_type") == "ATTACH_EXISTING"]
        attach_errors = [row for row in error_rows if row.get("original_action_type") == "ATTACH_EXISTING"]
        gt_available_errors = [row for row in error_rows if row.get("current_mask", {}).get("gt_assignment_eligible")]

        action_pairs = Counter(
            (str(row.get("original_action_type")), str(row.get("correct_action_type"))) for row in event_rows
        )
        error_action_pairs = Counter(
            (str(row.get("original_action_type")), str(row.get("correct_action_type"))) for row in error_rows
        )
        human_membership_pairs = Counter(
            (
                str(row.get("human_target_pre_state")),
                str((row.get("original_target_composition") or {}).get("objective_target_pre_state")),
            )
            for row in attach_errors
        )
        human_joint_pairs = Counter(
            (
                str(row.get("human_target_pre_state")),
                str(
                    (row.get("original_target_composition") or {}).get(
                        "objective_target_pre_state_joint"
                    )
                ),
            )
            for row in attach_errors
        )
        native_heal_counts = Counter(str(row.get("native_self_heal", {}).get("status")) for row in error_rows)
        primitive_counts = Counter(str(row.get("repair_primitive_inference")) for row in error_rows)

        correct_candidate_states = []
        for row in error_rows:
            comps = row.get("legal_target_compositions") or []
            if comps:
                states = [str(comp.get("objective_target_pre_state_joint")) for comp in comps]
                if "CLEAN_SINGLE_INSTANCE" in states:
                    correct_candidate_states.append("HAS_CLEAN_LEGAL_TARGET")
                elif "ALREADY_CONTAMINATED" in states:
                    correct_candidate_states.append("ALL_AVAILABLE_LEGAL_TARGETS_CONTAMINATED_OR_UNCERTAIN")
                else:
                    correct_candidate_states.append("LEGAL_TARGET_UNCERTAIN")
            elif row.get("correct_action_type") == "ATTACH_EXISTING":
                correct_candidate_states.append("LEGAL_TARGET_VERSION_MISSING")

        display_relevant = [
            row.get("packet_original_target_display_audit")
            for row in attach_errors
            if row.get("packet_original_target_display_audit")
        ]
        display_omitted = [row for row in display_relevant if row.get("gt_ids_omitted_from_display")]
        frozen_candidates = [row for row in candidate_rows if row.get("candidate_role") == "FROZEN_DISPLAY"]
        contaminated_frozen = [
            row
            for row in frozen_candidates
            if row.get("composition", {}).get("objective_target_pre_state_joint") == "ALREADY_CONTAMINATED"
        ]
        contaminated_attach_errors = [
            row
            for row in attach_errors
            if row.get("original_target_composition", {}).get("objective_target_pre_state_joint")
            == "ALREADY_CONTAMINATED"
        ]
        earliest_root_not_labeled = [
            row
            for row in contaminated_attach_errors
            if row.get("ledger_earliest_strict_contamination_in_174_labels") is False
        ]
        merge_backtrace_required = [
            row
            for row in contaminated_attach_errors
            if row.get("ledger_earliest_strict_contamination_trace_kind")
            == "OBJECT_MERGE_REQUIRES_PARENT_DAG_BACKTRACE"
        ]
        direct_root_by_event: dict[str, dict[str, Any]] = {}
        for row in contaminated_attach_errors:
            event_uid = row.get("ledger_earliest_strict_contamination_association_event_uid")
            if event_uid:
                direct_root_by_event.setdefault(str(event_uid), row)
        mixed_direct_roots = []
        gt_gate_failed_direct_roots = []
        for event_uid, row in direct_root_by_event.items():
            transition = (row.get("earliest_original_target_contamination") or {}).get(
                "first_strict_multi_gt"
            ) or {}
            examples = transition.get("introduced_gt_member_examples") or []
            if any(bool(item.get("mask_mixed") or item.get("mask_two_foreground")) for item in examples):
                mixed_direct_roots.append(event_uid)
            if any(
                not (
                    item.get("gt_purity") is not None
                    and float(item.get("gt_purity")) >= 0.90
                    and not bool(item.get("mask_mixed") or item.get("mask_two_foreground"))
                )
                for item in examples
            ):
                gt_gate_failed_direct_roots.append(event_uid)

        routing_summary = read_json(self.paths["routing_audit_summary"])
        worklist_manifest = read_json(self.paths["large_worklist_manifest"])

        root_cross = Counter(
            (
                str(row.get("causal_role")),
                str(
                    (row.get("original_target_composition") or {}).get(
                        "objective_target_pre_state_joint"
                    )
                    or "NOT_APPLICABLE"
                ),
            )
            for row in error_rows
        )
        quality_cross = Counter(
            (
                str(row.get("current_mask", {}).get("human_observation_quality")),
                str(row.get("routing_label")),
            )
            for row in error_rows
        )
        class_cross = Counter(
            (str(row.get("offline_class_relation")), str(row.get("routing_label"))) for row in error_rows
        )
        future2 = sum(bool(row.get("future_identity_stability", {}).get("has_two_views_window")) for row in error_rows)
        future3 = sum(bool(row.get("future_identity_stability", {}).get("has_three_views_window")) for row in error_rows)
        future_available = sum(row.get("future_identity_stability", {}).get("status") == "OK" for row in error_rows)

        correct_attach_rows = [
            row for row in event_rows if row.get("correct_action_type") == "ATTACH_EXISTING"
        ]
        multi_legal_rows = [
            row for row in correct_attach_rows if len(row.get("legal_target_versions") or []) > 1
        ]
        multi_legal_details: list[dict[str, Any]] = []
        same_gt_fragmentation_sets: set[tuple[str, ...]] = set()
        for row in multi_legal_rows:
            observation_gt_id = row.get("current_mask", {}).get("gt_id")
            observation_gt_key = str(observation_gt_id) if observation_gt_id is not None else None
            versions = [str(uid) for uid in (row.get("legal_target_versions") or [])]
            compositions = row.get("legal_target_compositions") or []
            candidates: list[dict[str, Any]] = []
            supporting_centroids: list[list[float]] = []
            supporting_object_uids: list[str] = []
            for index, version_uid in enumerate(versions):
                composition = compositions[index] if index < len(compositions) else {}
                support_count = 0
                if observation_gt_key is not None:
                    support_count = next(
                        (
                            int(item.get("count") or 0)
                            for item in composition.get("gt_id_counts") or []
                            if str(item.get("value")) == observation_gt_key
                        ),
                        0,
                    )
                matching_component = next(
                    (
                        item
                        for item in composition.get("component_details") or []
                        if observation_gt_key is not None
                        and str(item.get("gt_id")) == observation_gt_key
                    ),
                    None,
                )
                centroid = (matching_component or {}).get("centroid_mean")
                object_uid = version_uid.split("@", 1)[0]
                if support_count > 0:
                    supporting_object_uids.append(object_uid)
                    if centroid:
                        supporting_centroids.append([float(value) for value in centroid])
                candidates.append(
                    {
                        "object_uid": object_uid,
                        "object_version_uid": version_uid,
                        "observation_gt_member_count": support_count,
                        "member_observation_count": composition.get("member_observation_count"),
                        "joint_state": composition.get("objective_target_pre_state_joint"),
                        "pixel_mixed_mask_count": composition.get("pixel_mixed_mask_count"),
                        "observation_gt_centroid_mean": centroid,
                    }
                )
            pair_distances = [
                distance(supporting_centroids[left], supporting_centroids[right])
                for left in range(len(supporting_centroids))
                for right in range(left + 1, len(supporting_centroids))
            ]
            supporting_count = len(supporting_object_uids)
            if supporting_count >= 2:
                same_gt_fragmentation_sets.add(tuple(sorted(supporting_object_uids)))
            multi_legal_details.append(
                {
                    "case_uid": row.get("case_uid"),
                    "event_uid": row.get("event_uid"),
                    "event_sequence": row.get("event_sequence"),
                    "observation_gt_id": observation_gt_id,
                    "legal_candidate_count": len(versions),
                    "offline_same_gt_supporting_candidate_count": supporting_count,
                    "offline_same_gt_supporting_candidate_pair_distances_m": pair_distances,
                    "classification": (
                        "SAME_GT_MULTI_OBJECT_FRAGMENTATION_SIGNAL"
                        if supporting_count >= 2
                        else "HUMAN_MULTI_MATCH_NOT_CONFIRMED_BY_OFFLINE_GT"
                    ),
                    "is_error": bool(row.get("is_error")),
                    "routing_label": row.get("routing_label"),
                    "candidates": candidates,
                }
            )

        return {
            "scope": {
                "independent_event_count": len(event_rows),
                "error_count": len(error_rows),
                "attach_event_count": len(attach_rows),
                "attach_error_count": len(attach_errors),
                "error_with_corrected_gt_count": len(gt_available_errors),
            },
            "current_mask": {
                "all_human_quality_counts": grouped_counts(row.get("current_mask", {}).get("human_observation_quality") for row in event_rows),
                "error_human_quality_counts": grouped_counts(row.get("current_mask", {}).get("human_observation_quality") for row in error_rows),
                "error_identity_evidence_counts": grouped_counts(row.get("current_mask", {}).get("human_identity_evidence_status") for row in error_rows),
                "error_gt_purity_bin_counts": grouped_counts(row.get("current_mask", {}).get("gt_purity_bin") for row in error_rows),
                "error_corrected_gt_mask_mixed_true_count": sum(bool(row.get("current_mask", {}).get("mask_mixed")) for row in error_rows),
                "error_corrected_gt_two_foreground_true_count": sum(bool(row.get("current_mask", {}).get("mask_two_foreground")) for row in error_rows),
                "quality_by_error_type": {f"{left} | {right}": count for (left, right), count in sorted(quality_cross.items())},
            },
            "routing": {
                "all_original_to_correct_action": {f"{left} -> {right}": count for (left, right), count in sorted(action_pairs.items())},
                "error_original_to_correct_action": {f"{left} -> {right}": count for (left, right), count in sorted(error_action_pairs.items())},
                "error_routing_label_counts": grouped_counts(row.get("routing_label") for row in error_rows),
            },
            "target_prestate": {
                "attach_event_membership_state_counts": grouped_counts(
                    (row.get("original_target_composition") or {}).get("objective_target_pre_state") for row in attach_rows
                ),
                "attach_event_joint_state_counts": grouped_counts(
                    (row.get("original_target_composition") or {}).get("objective_target_pre_state_joint")
                    for row in attach_rows
                ),
                "attach_error_membership_state_counts": grouped_counts(
                    (row.get("original_target_composition") or {}).get("objective_target_pre_state") for row in attach_errors
                ),
                "attach_error_joint_state_counts": grouped_counts(
                    (row.get("original_target_composition") or {}).get("objective_target_pre_state_joint")
                    for row in attach_errors
                ),
                "attach_error_human_vs_membership_state": {
                    f"{human} -> {objective}": count
                    for (human, objective), count in sorted(human_membership_pairs.items())
                },
                "attach_error_human_vs_joint_state": {
                    f"{human} -> {objective}": count
                    for (human, objective), count in sorted(human_joint_pairs.items())
                },
                "error_correct_legal_candidate_state_counts": grouped_counts(correct_candidate_states),
                "strict_multi_gt_attach_error_count": sum(
                    bool((row.get("original_target_composition") or {}).get("strict_multi_gt")) for row in attach_errors
                ),
                "persistent_multi_gt_attach_error_count": sum(
                    bool((row.get("original_target_composition") or {}).get("persistent_multi_gt_2obs_2frames")) for row in attach_errors
                ),
                "persistent_pixel_contamination_attach_error_count": sum(
                    bool(
                        (row.get("original_target_composition") or {}).get(
                            "persistent_pixel_contamination_2obs_2frames_5pct"
                        )
                    )
                    for row in attach_errors
                ),
            },
            "causality": {
                "root_cascade_pending_counts": grouped_counts(row.get("causal_role") for row in error_rows),
                "causal_role_by_original_target_state": {
                    f"{role} | {state}": count for (role, state), count in sorted(root_cross.items())
                },
                "contaminated_attach_error_count": len(contaminated_attach_errors),
                "earliest_strict_contamination_event_not_in_174_count": len(earliest_root_not_labeled),
                "unique_direct_earliest_contamination_event_not_in_174_count": len(
                    {
                        str(row.get("ledger_earliest_strict_contamination_association_event_uid"))
                        for row in earliest_root_not_labeled
                    }
                ),
                "earliest_strict_contamination_event_not_in_174_cases": [
                    {
                        "case_uid": row.get("case_uid"),
                        "anchor_event_uid": row.get("event_uid"),
                        "earliest_event_uid": row.get("ledger_earliest_strict_contamination_association_event_uid"),
                        "event_gap": row.get("anchor_minus_earliest_contamination_event_gap"),
                    }
                    for row in earliest_root_not_labeled
                ],
                "object_merge_transition_requires_parent_dag_backtrace_count": len(merge_backtrace_required),
                "object_merge_transition_requires_parent_dag_backtrace_cases": [
                    {
                        "case_uid": row.get("case_uid"),
                        "anchor_event_uid": row.get("event_uid"),
                        "transition_event_uid": row.get(
                            "ledger_earliest_strict_contamination_transition_event_uid"
                        ),
                    }
                    for row in merge_backtrace_required
                ],
                "unique_direct_contamination_root_count": len(direct_root_by_event),
                "unique_direct_root_with_mixed_or_two_foreground_mask_count": len(mixed_direct_roots),
                "unique_direct_root_with_mixed_or_two_foreground_mask_event_uids": sorted(mixed_direct_roots),
            },
            "sampling_blind_spot": {
                "direct_contamination_roots_present_in_private_routing_audit_count": sum(
                    event_uid in self.routing_audit_event_uids for event_uid in direct_root_by_event
                ),
                "direct_contamination_roots_absent_from_private_routing_audit_count": sum(
                    event_uid not in self.routing_audit_event_uids for event_uid in direct_root_by_event
                ),
                "direct_contamination_roots_failing_090_or_mixed_protocol_gate_count": len(
                    set(gt_gate_failed_direct_roots)
                ),
                "routing_audit_observation_gt_unreliable_count": routing_summary.get("audit_counts", {}).get(
                    "observation_gt_unreliable"
                ),
                "routing_audit_gt_purity_threshold": routing_summary.get("thresholds", {}).get("gt_purity"),
                "probability_population_count": worklist_manifest.get("probability_population_count"),
                "probability_sample_count": worklist_manifest.get("probability_sample_count"),
                "probability_inclusion_probability": worklist_manifest.get("probability_inclusion_probability"),
                "error_harvest_count": worklist_manifest.get("error_harvest_count"),
                "note": (
                    "The private routing audit first requires a high-purity current observation. "
                    "Mixed-mask segmentation roots can therefore be absent from error harvest and only enter "
                    "through the low-rate probability queue."
                ),
            },
            "candidate_ranking": {
                "covered_at_5_plus_new_count": sum(
                    bool(row.get("association_scores", {}).get("covered_at_5_plus_new")) for row in error_rows
                ),
                "covered_at_5_plus_new_denominator": len(error_rows),
                "correct_existing_rank_counts": grouped_counts(
                    row.get("association_scores", {}).get("correct_existing_best_rank")
                    for row in error_rows
                    if row.get("correct_action_type") == "ATTACH_EXISTING"
                ),
                "new_overridden_count": sum(bool(row.get("association_scores", {}).get("new_was_overridden")) for row in error_rows),
                "new_overridden_top1_minus_threshold_median": median_or_none(
                    row.get("association_scores", {}).get("top1_minus_new_threshold")
                    for row in error_rows
                    if row.get("association_scores", {}).get("new_was_overridden")
                ),
                "error_top1_top2_margin_median": median_or_none(
                    row.get("association_scores", {}).get("top1_top2_margin") for row in error_rows
                ),
            },
            "multi_legal_target_analysis": {
                "correct_attach_event_count": len(correct_attach_rows),
                "legal_candidate_count_distribution": grouped_counts(
                    len(row.get("legal_target_versions") or []) for row in correct_attach_rows
                ),
                "multi_legal_target_event_count": len(multi_legal_rows),
                "multi_legal_target_event_rate_among_correct_attach": safe_ratio(
                    len(multi_legal_rows), len(correct_attach_rows)
                ),
                "multi_legal_target_error_event_count": sum(
                    bool(row.get("is_error")) for row in multi_legal_rows
                ),
                "same_gt_multi_object_fragmentation_signal_event_count": sum(
                    detail["classification"] == "SAME_GT_MULTI_OBJECT_FRAGMENTATION_SIGNAL"
                    for detail in multi_legal_details
                ),
                "unique_same_gt_fragmentation_candidate_set_count": len(
                    same_gt_fragmentation_sets
                ),
                "human_multi_match_not_confirmed_by_offline_gt_event_count": sum(
                    detail["classification"]
                    == "HUMAN_MULTI_MATCH_NOT_CONFIRMED_BY_OFFLINE_GT"
                    for detail in multi_legal_details
                ),
                "cases": multi_legal_details,
                "note": (
                    "A human multi-target label is an over-segmentation signal only when at least two "
                    "distinct graph objects contain members of the current corrected-GT instance. "
                    "Selections with only one offline-supporting candidate remain annotation ambiguity, "
                    "part-whole ambiguity, or false-positive matching rather than confirmed fragmentation."
                ),
            },
            "future_views": {
                "offline_future_status_ok_count": future_available,
                "two_independent_views_within_30_count": future2,
                "three_independent_views_within_30_count": future3,
                "window_count_median_when_available": median_or_none(
                    row.get("future_identity_stability", {}).get("independent_views_window_count")
                    for row in error_rows
                    if row.get("future_identity_stability", {}).get("status") == "OK"
                ),
                "suffix_count_median_when_available": median_or_none(
                    row.get("future_identity_stability", {}).get("independent_views_suffix_count")
                    for row in error_rows
                    if row.get("future_identity_stability", {}).get("status") == "OK"
                ),
                "native_final_owner_stable_count": sum(
                    row.get("future_identity_stability", {}).get("native_final_distinct_owner_count") == 1
                    for row in error_rows
                    if row.get("future_identity_stability", {}).get("status") == "OK"
                ),
                "note": "Future views and final ownership are diagnostic only and would leak future information if used at the anchor.",
            },
            "class_relation": {
                "counts": grouped_counts(row.get("offline_class_relation") for row in error_rows),
                "by_error_type": {f"{left} | {right}": count for (left, right), count in sorted(class_cross.items())},
            },
            "native_self_heal": {
                "status_counts": dict(sorted(native_heal_counts.items())),
                "structural_self_heal_count": sum(bool(row.get("native_self_heal", {}).get("structural_self_heal")) for row in error_rows),
                "note": "Computed from the native final map; this is a post-run diagnostic, not causal proof or online evidence.",
            },
            "page_evidence": {
                "exact_target_version_mismatch_count_all_attach": sum(
                    bool(row.get("frozen_vs_exact_version_mismatch")) for row in attach_rows
                ),
                "exact_target_version_mismatch_count_attach_errors": sum(
                    bool(row.get("frozen_vs_exact_version_mismatch")) for row in attach_errors
                ),
                "display_auditable_attach_error_count": len(display_relevant),
                "display_omitted_at_least_one_full_history_gt_id_count": len(display_omitted),
                "note": "A version mismatch can arise from an earlier event in the same frame; member-count delta is reported per event and is not automatically a labeling bug.",
            },
            "displayed_candidate_pool": {
                "unique_frozen_candidate_count": len(frozen_candidates),
                "membership_state_counts": grouped_counts(
                    row.get("composition", {}).get("objective_target_pre_state") for row in frozen_candidates
                ),
                "joint_state_counts": grouped_counts(
                    row.get("composition", {}).get("objective_target_pre_state_joint")
                    for row in frozen_candidates
                ),
                "already_contaminated_count": len(contaminated_frozen),
                "error_case_count_with_at_least_one_contaminated_frozen_candidate": len(
                    {str(row.get("case_uid")) for row in contaminated_frozen}
                ),
                "note": "Descriptive for the displayed candidates of the 14 error cases, not a prevalence estimate for the full stream.",
            },
            "repair_primitive_inference": dict(sorted(primitive_counts.items())),
        }

    def write_csv(self, path: Path, error_rows: list[dict[str, Any]]) -> None:
        fields = [
            "case_uid",
            "sequence",
            "frame",
            "routing_label",
            "original_action",
            "correct_action",
            "mask_quality",
            "identity_evidence",
            "obs_gt",
            "obs_purity",
            "causal_role",
            "class_relation",
            "human_target_state",
            "membership_target_state",
            "joint_target_state",
            "target_gt_counts",
            "correct_rank",
            "top1_minus_threshold",
            "top1_top2_margin",
            "future_views_30",
            "future_views_suffix",
            "native_self_heal",
            "repair_primitive_inference",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in error_rows:
                target = row.get("original_target_composition") or {}
                writer.writerow(
                    {
                        "case_uid": row.get("case_uid"),
                        "sequence": row.get("event_sequence"),
                        "frame": row.get("event_frame_idx"),
                        "routing_label": row.get("routing_label"),
                        "original_action": row.get("original_action_type"),
                        "correct_action": row.get("correct_action_type"),
                        "mask_quality": row.get("current_mask", {}).get("human_observation_quality"),
                        "identity_evidence": row.get("current_mask", {}).get("human_identity_evidence_status"),
                        "obs_gt": row.get("current_mask", {}).get("gt_id"),
                        "obs_purity": row.get("current_mask", {}).get("gt_purity"),
                        "causal_role": row.get("causal_role"),
                        "class_relation": row.get("offline_class_relation"),
                        "human_target_state": row.get("human_target_pre_state"),
                        "membership_target_state": target.get("objective_target_pre_state"),
                        "joint_target_state": target.get("objective_target_pre_state_joint"),
                        "target_gt_counts": json.dumps(target.get("gt_id_counts"), ensure_ascii=False),
                        "correct_rank": row.get("association_scores", {}).get("correct_existing_best_rank"),
                        "top1_minus_threshold": row.get("association_scores", {}).get("top1_minus_new_threshold"),
                        "top1_top2_margin": row.get("association_scores", {}).get("top1_top2_margin"),
                        "future_views_30": row.get("future_identity_stability", {}).get("independent_views_window_count"),
                        "future_views_suffix": row.get("future_identity_stability", {}).get("independent_views_suffix_count"),
                        "native_self_heal": row.get("native_self_heal", {}).get("status"),
                        "repair_primitive_inference": row.get("repair_primitive_inference"),
                    }
                )

    def run(self) -> None:
        started = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load()
        event_rows, error_rows, candidate_rows = self.build_event_rows()
        replay = self.replay_audit()
        cross = self.cross_statistics(event_rows, error_rows, candidate_rows)
        write_jsonl(self.output_dir / "event_multidimensional_table.jsonl", event_rows)
        write_jsonl(self.output_dir / "error_multidimensional_table.jsonl", error_rows)
        write_jsonl(self.output_dir / "error_candidate_compositions.jsonl", candidate_rows)
        write_json(self.output_dir / "cross_statistics.json", cross)
        write_json(self.output_dir / "replay_full_instance_audit.json", replay)
        self.write_csv(self.output_dir / "error_multidimensional_table.csv", error_rows)
        manifest = {
            "schema_version": "experiment0-multidimensional-audit/1.0",
            "analysis_only": True,
            "production_logic_changed": False,
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "runtime_seconds": time.time() - started,
            "source_files": {name: source_info(path) for name, path in self.paths.items()},
            "outputs": sorted(path.name for path in self.output_dir.iterdir()),
            "definitions": {
                "membership_clean": "eligible corrected GT members contain exactly one dominant observation-level GT id",
                "membership_contaminated": "second observation-level GT id has >=2 observations across >=2 frames",
                "joint_clean": "membership-clean and no corrected-GT mixed/two-foreground mask member",
                "joint_contaminated": "membership-contaminated, or >=2 mixed masks across >=2 frames with >=5% aggregate non-dominant top-2 GT pixels",
                "joint_uncertain": "GT coverage <80%, no eligible GT, or membership/pixel mixture below the persistent threshold",
                "material_mixture": "second GT member fraction >=5%",
                "two_owner_capture_recall": "new-owner positives / (new-owner positives + residual positives in original target owner)",
                "global_observation_recall": "new-owner positives / all corrected-GT observations of the physical instance in the 7507-observation ledger",
            },
            "warnings": [
                "Corrected GT is offline diagnostic evidence and must not be passed to the online mapper.",
                "Human target_pre_state and objective GT composition answer related but non-identical questions; disagreements require case review.",
                "Native final ownership and future views are post-anchor information and are never treated as online features here.",
                "No new file hashes are computed; existing immutable ledgers are identified by absolute path, size, mtime, and line count.",
            ],
        }
        write_json(self.output_dir / "run_manifest.json", manifest)
        print(json.dumps({"status": "OK", "events": len(event_rows), "errors": len(error_rows), "output_dir": str(self.output_dir), "runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/home/chenkejun/beauty/conceptgraphs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_multidimensional_analysis_20260902"
        ),
    )
    args = parser.parse_args()
    Audit(args.project_root.resolve(), args.output_dir.resolve()).run()


if __name__ == "__main__":
    main()
