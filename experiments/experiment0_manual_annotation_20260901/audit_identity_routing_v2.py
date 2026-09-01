#!/usr/bin/env python3
"""Compile high-precision ATTACH/NEW identity-routing audit records.

This is a read-only evaluator.  It uses corrected offline instance GT only to
select and audit human-review cases; GT-derived strata must remain private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BG_LABELS = {"wall", "floor", "ceiling", "unknown", "undefined", "background"}
GT_PURITY = 0.90
GT_SUPPORT = 0.90
GT_TOP_PIXELS = 25
HISTORY_MIN_OBS = 3
HISTORY_MIN_FRAMES = 3
HISTORY_DOMINANT_RATIO = 0.80
ROUTING_ERRORS = {
    "WRONG_ATTACH_EXISTING",
    "SHOULD_HAVE_BEEN_NEW",
    "WRONG_NEW_FALSE_SPLIT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--observation-gt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield value


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def raw_frame(row: dict[str, Any]) -> int:
    return int(row.get("raw_frame", row.get("frame_idx", -1)))


def processed_frame(frame_uid: str) -> int:
    try:
        return int(frame_uid.rsplit("_f", 1)[1][:6])
    except (IndexError, ValueError):
        return -1


def reliable_gt(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    label = str(row.get("gt_top_label") or "").strip().lower()
    return bool(
        row.get("gt_assignment_eligible")
        and float(row.get("gt_purity") or 0) >= GT_PURITY
        and float(row.get("gt_supported_fraction") or 0) >= GT_SUPPORT
        and int(row.get("gt_top_pixels") or 0) >= GT_TOP_PIXELS
        and row.get("gt_top_id") is not None
        and label not in BG_LABELS
    )


@dataclass(frozen=True)
class HistoricalIdentity:
    gt_id: int
    reliable_observations: int
    unique_frames: int
    dominant_ratio: float
    supported_gt_ids: tuple[int, ...]


def historical_identity(
    version_uid: str,
    current_raw_frame: int,
    versions: dict[str, dict[str, Any]],
    gt_by_obs: dict[str, dict[str, Any]],
) -> tuple[HistoricalIdentity | None, str, set[int]]:
    version = versions.get(version_uid)
    if version is None:
        return None, "missing_version", set()
    rows = []
    for obs_uid in version.get("member_observation_uids") or []:
        gt = gt_by_obs.get(str(obs_uid))
        if reliable_gt(gt) and raw_frame(gt) < current_raw_frame:
            rows.append(gt)
    supported_ids = {int(row["gt_top_id"]) for row in rows}
    if len(rows) < HISTORY_MIN_OBS:
        return None, "history_lt3_reliable", supported_ids
    frames = {raw_frame(row) for row in rows}
    if len(frames) < HISTORY_MIN_FRAMES:
        return None, "history_lt3_frames", supported_ids
    counts = Counter(int(row["gt_top_id"]) for row in rows)
    gt_id, count = counts.most_common(1)[0]
    ratio = count / len(rows)
    if ratio < HISTORY_DOMINANT_RATIO:
        return None, "history_ambiguous", supported_ids
    return (
        HistoricalIdentity(
            gt_id=gt_id,
            reliable_observations=len(rows),
            unique_frames=len(frames),
            dominant_ratio=ratio,
            supported_gt_ids=tuple(sorted(supported_ids)),
        ),
        "ok",
        supported_ids,
    )


def causal_group_key(record: dict[str, Any]) -> str | None:
    routing_label = record.get("private_auto_routing_label")
    obs_gt_id = record.get("private_obs_gt_id")
    if routing_label in {"WRONG_ATTACH_EXISTING", "SHOULD_HAVE_BEEN_NEW"}:
        return f"attach:{record.get('original_target_object_uid')}:{obs_gt_id}"
    if routing_label == "WRONG_NEW_FALSE_SPLIT":
        legal = ",".join(record.get("private_legal_candidate_uids") or [])
        return f"new:{obs_gt_id}:{legal}"
    return None


def compile_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence_root = args.evidence_root.resolve()
    gt_path = args.observation_gt.resolve()
    associations_path = evidence_root / "associations.jsonl"
    versions_path = evidence_root / "object_versions.jsonl"
    gt_by_obs = {str(row["obs_uid"]): row for row in read_jsonl(gt_path)}
    versions = {
        str(row["object_version_uid"]): row for row in read_jsonl(versions_path)
    }
    associations = list(read_jsonl(associations_path))

    records: list[dict[str, Any]] = []
    audit = Counter()
    decision_counts = Counter()
    identity_cache: dict[
        tuple[str, int], tuple[HistoricalIdentity | None, str, set[int]]
    ] = {}

    for association in associations:
        decision = str(association.get("decision") or "")
        decision_counts[decision] += 1
        if decision not in {"MERGE_TO_OBJECT", "CREATE_OBJECT"}:
            audit["unknown_decision"] += 1
            continue

        obs_uid = str(association.get("obs_uid") or "")
        obs_gt = gt_by_obs.get(obs_uid)
        if obs_gt is None:
            audit["missing_observation_gt"] += 1
            continue
        if not reliable_gt(obs_gt):
            audit["observation_gt_unreliable"] += 1
            continue

        object_uids = [str(value) for value in association.get("object_uids_before") or []]
        version_uids = [
            str(value) for value in association.get("candidate_object_version_uids") or []
        ]
        if len(object_uids) != len(version_uids):
            audit["candidate_version_alignment_mismatch"] += 1
            continue
        if len(set(object_uids)) != len(object_uids):
            audit["duplicate_candidate_object_uid"] += 1
            continue
        if any(version_uid not in versions for version_uid in version_uids):
            audit["missing_candidate_version"] += 1
            continue

        current_raw_frame = raw_frame(obs_gt)
        obs_gt_id = int(obs_gt["gt_top_id"])
        identities: list[HistoricalIdentity | None] = []
        identity_reasons: list[str] = []
        support_sets: list[set[int]] = []
        for version_uid in version_uids:
            cache_key = (version_uid, current_raw_frame)
            if cache_key not in identity_cache:
                identity_cache[cache_key] = historical_identity(
                    version_uid, current_raw_frame, versions, gt_by_obs
                )
            identity, reason, supported_ids = identity_cache[cache_key]
            identities.append(identity)
            identity_reasons.append(reason)
            support_sets.append(supported_ids)

        legal_candidate_uids = sorted(
            object_uids[index]
            for index, identity in enumerate(identities)
            if identity is not None and identity.gt_id == obs_gt_id
        )
        same_gt_support_candidate_uids = sorted(
            object_uids[index]
            for index, supported_ids in enumerate(support_sets)
            if obs_gt_id in supported_ids
        )
        unknown_candidate_count = sum(identity is None for identity in identities)

        target_uid = str(association.get("target_object_uid") or "") or None
        target_identity: HistoricalIdentity | None = None
        target_identity_reason: str | None = None
        target_version_uid: str | None = None
        routing_label: str | None = None
        exclusion_reason: str | None = None

        if decision == "MERGE_TO_OBJECT":
            if target_uid not in object_uids:
                audit["merge_target_not_in_tminus_candidates"] += 1
                continue
            target_index = object_uids.index(str(target_uid))
            target_identity = identities[target_index]
            target_identity_reason = identity_reasons[target_index]
            target_version_uid = version_uids[target_index]
            if target_identity is None:
                exclusion_reason = f"target_{target_identity_reason}"
            elif target_identity.gt_id == obs_gt_id:
                routing_label = "CORRECT_ATTACH"
            elif legal_candidate_uids:
                routing_label = "WRONG_ATTACH_EXISTING"
            elif not same_gt_support_candidate_uids:
                routing_label = "SHOULD_HAVE_BEEN_NEW"
            else:
                exclusion_reason = "same_gt_only_in_short_or_ambiguous_history"
        else:
            if legal_candidate_uids:
                routing_label = "WRONG_NEW_FALSE_SPLIT"
            elif not same_gt_support_candidate_uids:
                routing_label = "CORRECT_NEW"
            else:
                exclusion_reason = "same_gt_only_in_short_or_ambiguous_history"

        frame_uid = str(association.get("frame_uid") or "")
        snapshot_payload = {
            "frame_uid": frame_uid,
            "event_uid": association.get("event_uid"),
            "object_uids_before": object_uids,
            "candidate_object_version_uids": version_uids,
            "aggregate_sim_sha256": (
                association.get("aggregate_sim_ref") or {}
            ).get("sha256"),
        }
        record = {
            "schema_version": "experiment0-private-routing-audit/2.0",
            "scene": args.scene,
            "event_uid": association.get("event_uid"),
            "obs_uid": obs_uid,
            "frame_uid": frame_uid,
            "processed_frame_idx": processed_frame(frame_uid),
            "raw_frame": current_raw_frame,
            "decision": decision,
            "original_action_type": (
                "ATTACH_EXISTING" if decision == "MERGE_TO_OBJECT" else "NEW"
            ),
            "original_target_object_uid": target_uid if decision == "MERGE_TO_OBJECT" else None,
            "created_object_uid": target_uid if decision == "CREATE_OBJECT" else None,
            "target_object_version_before": target_version_uid,
            "candidate_count": len(object_uids),
            "strict_identity_candidate_count": len(object_uids) - unknown_candidate_count,
            "unknown_identity_candidate_count": unknown_candidate_count,
            "private_obs_gt_id": obs_gt_id,
            "private_obs_gt_label": obs_gt.get("gt_top_label"),
            "private_obs_gt_purity": float(obs_gt["gt_purity"]),
            "private_target_gt_id": target_identity.gt_id if target_identity else None,
            "private_target_history_observations": (
                target_identity.reliable_observations if target_identity else None
            ),
            "private_target_history_frames": (
                target_identity.unique_frames if target_identity else None
            ),
            "private_target_history_dominant_ratio": (
                target_identity.dominant_ratio if target_identity else None
            ),
            "private_target_identity_reason": target_identity_reason,
            "private_legal_candidate_uids": legal_candidate_uids,
            "private_same_gt_support_candidate_uids": same_gt_support_candidate_uids,
            "private_auto_routing_label": routing_label,
            "private_auto_evaluable": routing_label is not None,
            "private_auto_exclusion_reason": exclusion_reason,
            "top1_score": association.get("top1_score"),
            "top2_score": association.get("top2_score"),
            "margin": association.get("margin"),
            "sim_threshold": association.get("sim_threshold"),
            "tminus_snapshot_sha256": stable_json_sha256(snapshot_payload),
        }
        records.append(record)
        if routing_label is None:
            audit[f"unresolved_{exclusion_reason}"] += 1
        else:
            audit[f"routing_{routing_label}"] += 1

    records.sort(key=lambda row: (row["processed_frame_idx"], str(row["event_uid"])))
    by_causal_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = causal_group_key(record)
        record["private_causal_group_uid"] = (
            hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] if key else None
        )
        if key:
            by_causal_group[key].append(record)
    for group in by_causal_group.values():
        group.sort(key=lambda row: (row["processed_frame_idx"], str(row["event_uid"])))
        for index, record in enumerate(group):
            record["private_auto_episode_role"] = (
                "ROOT_CANDIDATE" if index == 0 else "CASCADE_CANDIDATE"
            )
    for record in records:
        if "private_auto_episode_role" not in record:
            record["private_auto_episode_role"] = (
                "NOT_ERROR"
                if record.get("private_auto_routing_label") not in ROUTING_ERRORS
                else "UNRESOLVED"
            )

    stratum_counts = Counter(
        str(row["private_auto_routing_label"])
        for row in records
        if row["private_auto_routing_label"] is not None
    )
    root_candidate_counts = Counter(
        str(row["private_auto_routing_label"])
        for row in records
        if row["private_auto_episode_role"] == "ROOT_CANDIDATE"
    )
    summary = {
        "schema_version": "experiment0-private-routing-audit-summary/2.0",
        "status": "READY",
        "scene": args.scene,
        "evidence_root": str(evidence_root),
        "observation_gt": str(gt_path),
        "source_sha256": {
            "associations": sha256_file(associations_path),
            "object_versions": sha256_file(versions_path),
            "observation_gt": sha256_file(gt_path),
        },
        "association_count": len(associations),
        "decision_counts": dict(sorted(decision_counts.items())),
        "audit_counts": dict(sorted(audit.items())),
        "compiled_record_count": len(records),
        "evaluable_record_count": sum(row["private_auto_evaluable"] for row in records),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "root_candidate_counts": dict(sorted(root_candidate_counts.items())),
        "causal_group_count": len(by_causal_group),
        "thresholds": {
            "gt_purity": GT_PURITY,
            "gt_support": GT_SUPPORT,
            "gt_top_pixels": GT_TOP_PIXELS,
            "history_min_observations": HISTORY_MIN_OBS,
            "history_min_frames": HISTORY_MIN_FRAMES,
            "history_dominant_ratio": HISTORY_DOMINANT_RATIO,
        },
        "interpretation": (
            "Private GT strata are high-precision case-selection aids, not final human labels. "
            "Root/cascade fields are chronological candidates pending human episode review."
        ),
    }
    return records, summary


def main() -> int:
    args = parse_args()
    records, summary = compile_records(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records_path = args.output_root / "routing_records_private.jsonl"
    write_jsonl_atomic(records_path, records)
    summary["routing_records_sha256"] = sha256_file(records_path)
    write_json_atomic(args.output_root / "summary.json", summary)
    (args.output_root / "READY").write_text("READY\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
