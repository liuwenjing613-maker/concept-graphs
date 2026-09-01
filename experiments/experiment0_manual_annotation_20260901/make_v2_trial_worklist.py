#!/usr/bin/env python3
"""Build a private, balanced schema-trial worklist for identity routing v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROUTING_STRATA = (
    "CORRECT_ATTACH",
    "WRONG_ATTACH_EXISTING",
    "SHOULD_HAVE_BEEN_NEW",
    "CORRECT_NEW",
    "WRONG_NEW_FALSE_SPLIT",
)
DEFAULT_EXCLUSIONS = {
    "calibration_018": "OUT_OF_SCOPE_MIXED_MULTIPLE_INSTANCES",
    "calibration_020": "OUT_OF_SCOPE_BACKGROUND_OR_FRAGMENT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-records", type=Path, required=True)
    parser.add_argument("--v1-calibration-worklist", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-routing-stratum", type=int, default=2)
    parser.add_argument("--hidden-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


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


def row_sha256(row: dict[str, Any]) -> str:
    encoded = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clarity_key(row: dict[str, Any]) -> tuple[Any, ...]:
    root_priority = int(row.get("private_auto_episode_role") == "ROOT_CANDIDATE")
    purity = float(row.get("private_obs_gt_purity") or 0)
    history_frames = int(row.get("private_target_history_frames") or 0)
    history_observations = int(row.get("private_target_history_observations") or 0)
    candidate_count = int(row.get("candidate_count") or 0)
    return (
        root_priority,
        purity,
        history_frames,
        history_observations,
        candidate_count > 0,
        -candidate_count,
        -int(row.get("processed_frame_idx") or 0),
        str(row.get("event_uid") or ""),
    )


def diversity_key(row: dict[str, Any]) -> str:
    causal = row.get("private_causal_group_uid")
    if causal:
        return f"causal:{causal}"
    label = str(row.get("private_auto_routing_label") or "")
    if label == "CORRECT_ATTACH":
        return f"target:{row.get('original_target_object_uid')}"
    return f"identity:{row.get('private_obs_gt_id')}"


def select_diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=clarity_key, reverse=True)
    selected: list[dict[str, Any]] = []
    seen = set()
    for row in ranked:
        key = diversity_key(row)
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) == count:
            return selected
    for row in ranked:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    return selected


def as_work_item(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    return {
        "case_uid": f"v2_trial_source_{source_index:03d}",
        "event_uid": row["event_uid"],
        "sample_kind": "V2_SCHEMA_TRIAL_PRIVATE_BALANCED",
        "private_auto_routing_label": row["private_auto_routing_label"],
        "private_auto_episode_role": row.get("private_auto_episode_role"),
        "private_causal_group_uid": row.get("private_causal_group_uid"),
        "private_legal_candidate_uids": row.get("private_legal_candidate_uids") or [],
        "private_same_gt_support_candidate_uids": (
            row.get("private_same_gt_support_candidate_uids") or []
        ),
        "private_obs_gt_id": row.get("private_obs_gt_id"),
        "private_obs_gt_label": row.get("private_obs_gt_label"),
        "private_obs_gt_purity": row.get("private_obs_gt_purity"),
        "private_routing_audit_row_sha256": row_sha256(row),
        "tminus_snapshot_sha256": row.get("tminus_snapshot_sha256"),
    }


def main() -> int:
    args = parse_args()
    if args.per_routing_stratum < 1:
        raise ValueError("--per-routing-stratum must be positive")
    if args.hidden_repeats < 0:
        raise ValueError("--hidden-repeats cannot be negative")

    routing_rows = read_jsonl(args.routing_records)
    selected_rows: list[dict[str, Any]] = []
    availability = Counter(
        str(row.get("private_auto_routing_label"))
        for row in routing_rows
        if row.get("private_auto_evaluable")
    )
    selected_counts = Counter()
    shortfalls = {}
    for stratum in ROUTING_STRATA:
        eligible = [
            row
            for row in routing_rows
            if row.get("private_auto_routing_label") == stratum
        ]
        chosen = select_diverse(eligible, args.per_routing_stratum)
        selected_rows.extend(chosen)
        selected_counts[stratum] = len(chosen)
        if len(chosen) < args.per_routing_stratum:
            shortfalls[stratum] = {
                "requested": args.per_routing_stratum,
                "available": len(eligible),
            }

    if any(selected_counts[stratum] == 0 for stratum in ROUTING_STRATA):
        missing = [stratum for stratum in ROUTING_STRATA if selected_counts[stratum] == 0]
        raise ValueError("v2 schema trial 缺少路由单元：" + ", ".join(missing))

    items = [as_work_item(row, index) for index, row in enumerate(selected_rows, 1)]
    calibration_rows = {
        str(row.get("case_uid")): row for row in read_jsonl(args.v1_calibration_worklist)
    }
    for old_case_uid, exclusion_label in DEFAULT_EXCLUSIONS.items():
        old = calibration_rows.get(old_case_uid)
        if old is None:
            raise ValueError(f"v1 worklist 缺少 {old_case_uid}")
        items.append(
            {
                "case_uid": f"v2_trial_source_{len(items) + 1:03d}",
                "event_uid": old["event_uid"],
                "sample_kind": "V2_SCHEMA_TRIAL_PRIVATE_EXCLUSION",
                "private_auto_routing_label": exclusion_label,
                "private_auto_episode_role": "NOT_APPLICABLE",
                "private_causal_group_uid": None,
                "private_legal_candidate_uids": [],
                "private_same_gt_support_candidate_uids": [],
                "private_source_v1_case_uid": old_case_uid,
                "private_source_v1_worklist_row_sha256": row_sha256(old),
            }
        )

    error_priority = (
        "WRONG_NEW_FALSE_SPLIT",
        "WRONG_ATTACH_EXISTING",
        "SHOULD_HAVE_BEEN_NEW",
    )
    repeat_sources = []
    for stratum in error_priority:
        source = next(
            (item for item in items if item["private_auto_routing_label"] == stratum),
            None,
        )
        if source is not None:
            repeat_sources.append(source)
    if len(repeat_sources) < args.hidden_repeats:
        for item in items:
            if item not in repeat_sources:
                repeat_sources.append(item)
            if len(repeat_sources) == args.hidden_repeats:
                break
    repeat_sources = repeat_sources[: args.hidden_repeats]
    repeats = []
    for source in repeat_sources:
        repeated = dict(source)
        repeated["case_uid"] = f"v2_trial_repeat_source_{len(repeats) + 1:03d}"
        repeated["sample_kind"] = "V2_SCHEMA_TRIAL_HIDDEN_REPEAT"
        repeated["repeat_of"] = source["case_uid"]
        repeats.append(repeated)
    items.extend(repeats)

    rng = random.Random(args.seed)
    rng.shuffle(items)
    source_to_public_uid = {}
    for index, item in enumerate(items, 1):
        old_uid = item["case_uid"]
        new_uid = f"v2_trial_{index:03d}"
        source_to_public_uid[old_uid] = new_uid
        item["case_uid"] = new_uid
    for item in items:
        repeat_of = item.get("repeat_of")
        if repeat_of:
            item["repeat_of"] = source_to_public_uid[repeat_of]

    args.output_root.mkdir(parents=True, exist_ok=True)
    worklist_path = args.output_root / "private_v2_trial_worklist.jsonl"
    write_jsonl_atomic(worklist_path, items)
    final_counts = Counter(item["private_auto_routing_label"] for item in items)
    manifest = {
        "schema_version": "experiment0-v2-trial-worklist/2.0",
        "status": "READY" if not shortfalls else "READY_WITH_DECLARED_SHORTFALLS",
        "purpose": "SCHEMA_AND_ANNOTATION_VALIDITY_ONLY_NOT_PREVALENCE",
        "routing_strata": list(ROUTING_STRATA),
        "requested_per_routing_stratum": args.per_routing_stratum,
        "routing_availability": dict(sorted(availability.items())),
        "base_selected_counts": dict(sorted(selected_counts.items())),
        "shortfalls": shortfalls,
        "base_case_count": len(items) - len(repeats),
        "hidden_repeat_count": len(repeats),
        "total_case_count": len(items),
        "private_final_counts_including_repeats": dict(sorted(final_counts.items())),
        "seed": args.seed,
        "source_sha256": {
            "routing_records": sha256_file(args.routing_records),
            "v1_calibration_worklist": sha256_file(args.v1_calibration_worklist),
        },
        "worklist_sha256": sha256_file(worklist_path),
        "warning": (
            "The balanced trial cannot estimate natural error prevalence. "
            "All private strata must remain hidden from the reviewer."
        ),
    }
    write_json_atomic(args.output_root / "trial_worklist_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
