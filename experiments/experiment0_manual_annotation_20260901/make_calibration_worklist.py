#!/usr/bin/env python3
"""Freeze a 20+4 room0 calibration list without exposing its strata in the UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SEED = "experiment0-room0-calibration-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--event-records", type=Path, required=True)
    parser.add_argument("--observation-gt", type=Path, required=True)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def hash_key(row: dict[str, Any], salt: str) -> str:
    value = f"{SEED}:{salt}:{row.get('event_uid')}:{row.get('obs_uid')}"
    return hashlib.sha256(value.encode()).hexdigest()


def diverse_pick(rows: list[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    """Round-robin clusters so one repeated object does not fill a stratum."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster = str(row.get("cluster_uid") or row.get("target_object_uid") or "unknown")
        groups[cluster].append(row)
    for values in groups.values():
        values.sort(key=lambda row: hash_key(row, salt))
    group_order = sorted(groups, key=lambda key: hashlib.sha256(f"{SEED}:{salt}:{key}".encode()).hexdigest())
    picked: list[dict[str, Any]] = []
    while group_order and len(picked) < count:
        remaining = []
        for group in group_order:
            if groups[group] and len(picked) < count:
                picked.append(groups[group].pop(0))
            if groups[group]:
                remaining.append(group)
        group_order = remaining
    if len(picked) < count:
        raise ValueError(f"not enough rows for {salt}: requested {count}, got {len(picked)}")
    return picked


def main() -> int:
    args = parse_args()
    event_records = [
        row
        for row in read_jsonl(args.event_records)
        if row.get("event_family") == "merge" and str(row.get("scene")) == args.scene
    ]
    associations = read_jsonl(args.associations)
    association_by_event = {str(row.get("event_uid")): row for row in associations}
    gt_by_obs = {str(row.get("obs_uid")): row for row in read_jsonl(args.observation_gt)}

    errors = [row for row in event_records if int(row.get("is_error") or 0) == 1]
    correct = [row for row in event_records if int(row.get("is_error") or 0) == 0]
    chosen_error = diverse_pick(errors, 8, "error")
    chosen_correct = diverse_pick(correct, 8, "correct")
    already = {str(row["event_uid"]) for row in chosen_error + chosen_correct}

    excluded_candidates = []
    for association in associations:
        if association.get("decision") != "MERGE_TO_OBJECT":
            continue
        event_uid = str(association.get("event_uid") or "")
        if event_uid in already:
            continue
        gt = gt_by_obs.get(str(association.get("obs_uid")))
        if gt is None:
            continue
        purity = float(gt.get("gt_purity") or 0)
        if bool(gt.get("mask_mixed")) or purity < 0.80:
            excluded_candidates.append({
                "event_uid": event_uid,
                "obs_uid": association.get("obs_uid"),
                "target_object_uid": association.get("target_object_uid"),
                "cluster_uid": association.get("target_object_uid"),
                "raw_frame": gt.get("raw_frame"),
                "private_auto_mask_mixed": bool(gt.get("mask_mixed")),
                "private_auto_gt_purity": purity,
            })
    chosen_excluded = diverse_pick(excluded_candidates, 4, "excluded")

    base_rows = []
    groups = [
        ("AUTO_ERROR", chosen_error),
        ("AUTO_CORRECT", chosen_correct),
        ("AUTO_MASK_EXCLUSION", chosen_excluded),
    ]
    sequence = 1
    for group_name, rows in groups:
        for row in rows:
            event_uid = str(row["event_uid"])
            association = association_by_event.get(event_uid)
            if association is None:
                raise ValueError(f"association missing for {event_uid}")
            case_uid = f"calibration_{sequence:03d}"
            base_rows.append({
                "case_uid": case_uid,
                "event_uid": event_uid,
                "sample_kind": "CALIBRATION_BALANCED_PRIVATE",
                "private_selection_group": group_name,
                "private_auto_is_error": row.get("is_error"),
                "private_auto_action_has_existing": row.get("correct_candidate_exists"),
                "private_auto_gt_purity": row.get("obs_gt_purity", row.get("private_auto_gt_purity")),
                "private_auto_mask_mixed": row.get("private_auto_mask_mixed"),
            })
            sequence += 1

    repeats = []
    repeat_sources = diverse_pick(base_rows, 4, "repeat")
    for source in repeat_sources:
        repeats.append({
            **source,
            "case_uid": f"calibration_{sequence:03d}",
            "repeat_of": source["case_uid"],
            "sample_kind": "CALIBRATION_HIDDEN_REPEAT",
        })
        sequence += 1

    worklist = base_rows + repeats
    # The server uses this fixed hash order; the reviewer never sees group fields.
    worklist.sort(key=lambda row: hashlib.sha256(f"{SEED}:order:{row['case_uid']}".encode()).hexdigest())
    write_jsonl_atomic(args.output, worklist)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "base_cases": len(base_rows),
        "hidden_repeats": len(repeats),
        "total": len(worklist),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "private_groups": {name: len(rows) for name, rows in groups},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
