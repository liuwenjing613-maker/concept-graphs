#!/usr/bin/env python3
"""Build a blind large-batch worklist without contaminating prevalence sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ERROR_ROUTES = {
    "WRONG_ATTACH_EXISTING",
    "SHOULD_HAVE_BEEN_NEW",
    "WRONG_NEW_FALSE_SPLIT",
}
CONTROL_ROUTES = {"CORRECT_ATTACH", "CORRECT_NEW"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--routing-records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--previous-worklist", type=Path, action="append", default=[])
    parser.add_argument("--probability-count", type=int, default=150)
    parser.add_argument("--matched-controls-per-error", type=int, default=2)
    parser.add_argument("--hidden-repeat-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--case-prefix", default="room0_large_r1")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


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


def stable_rank(seed: int, event_uid: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}:{seed}:{event_uid}".encode()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def work_item(
    association: dict[str, Any],
    route: dict[str, Any] | None,
    memberships: list[str],
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "case_uid": "pending",
        "event_uid": association["event_uid"],
        "sample_kind": "+".join(sorted(memberships)),
        "private_queue_memberships": sorted(memberships),
        "private_association_row_sha256": row_sha256(association),
        "private_legal_candidate_uids": [],
        "private_same_gt_support_candidate_uids": [],
    }
    if route is not None:
        item.update(
            {
                "private_auto_evaluable": route.get("private_auto_evaluable"),
                "private_auto_routing_label": route.get(
                    "private_auto_routing_label"
                ),
                "private_auto_episode_role": route.get("private_auto_episode_role"),
                "private_causal_group_uid": route.get("private_causal_group_uid"),
                "private_legal_candidate_uids": route.get(
                    "private_legal_candidate_uids"
                )
                or [],
                "private_same_gt_support_candidate_uids": route.get(
                    "private_same_gt_support_candidate_uids"
                )
                or [],
                "private_obs_gt_id": route.get("private_obs_gt_id"),
                "private_obs_gt_label": route.get("private_obs_gt_label"),
                "private_obs_gt_purity": route.get("private_obs_gt_purity"),
                "private_routing_audit_row_sha256": row_sha256(route),
                "tminus_snapshot_sha256": route.get("tminus_snapshot_sha256"),
            }
        )
    return item


def control_distance(error: dict[str, Any], control: dict[str, Any]) -> tuple[Any, ...]:
    error_frame = int(error.get("processed_frame_idx") or 0)
    control_frame = int(control.get("processed_frame_idx") or 0)
    error_margin = float(error.get("margin") or 0.0)
    control_margin = float(control.get("margin") or 0.0)
    return (
        abs(error_frame // 40 - control_frame // 40),
        abs(int(error.get("candidate_count") or 0) - int(control.get("candidate_count") or 0)),
        abs(error_margin - control_margin),
        abs(error_frame - control_frame),
        str(control["event_uid"]),
    )


def main() -> int:
    args = parse_args()
    if args.probability_count <= 0:
        raise ValueError("--probability-count must be positive")
    if args.matched_controls_per_error < 0:
        raise ValueError("--matched-controls-per-error cannot be negative")
    if not 0 <= args.hidden_repeat_fraction < 1:
        raise ValueError("--hidden-repeat-fraction must be in [0, 1)")

    associations = read_jsonl(args.associations)
    association_by_event = {str(row["event_uid"]): row for row in associations}
    if len(association_by_event) != len(associations):
        raise ValueError("associations contain duplicate event_uid")
    routes = read_jsonl(args.routing_records)
    route_by_event = {str(row["event_uid"]): row for row in routes}

    previous_event_uids: set[str] = set()
    for path in args.previous_worklist:
        previous_event_uids.update(str(row["event_uid"]) for row in read_jsonl(path))
    population = [
        row
        for row in associations
        if str(row["event_uid"]) not in previous_event_uids
    ]
    if args.probability_count > len(population):
        raise ValueError("probability sample exceeds available population")

    probability_rows = sorted(
        population,
        key=lambda row: stable_rank(args.seed, str(row["event_uid"]), "probability"),
    )[: args.probability_count]
    probability_uids = {str(row["event_uid"]) for row in probability_rows}

    errors = [
        row
        for row in routes
        if row.get("private_auto_evaluable")
        and row.get("private_auto_routing_label") in ERROR_ROUTES
        and str(row["event_uid"]) not in previous_event_uids
    ]
    errors.sort(key=lambda row: (int(row.get("processed_frame_idx") or 0), str(row["event_uid"])))
    controls = [
        row
        for row in routes
        if row.get("private_auto_evaluable")
        and row.get("private_auto_routing_label") in CONTROL_ROUTES
        and str(row["event_uid"]) not in previous_event_uids
    ]
    selected_control_uids: set[str] = set()
    control_matches: dict[str, list[str]] = {}
    for error in errors:
        wanted_decision = error.get("decision")
        available = [
            row
            for row in controls
            if row.get("decision") == wanted_decision
            and str(row["event_uid"]) not in selected_control_uids
        ]
        chosen = sorted(available, key=lambda row: control_distance(error, row))[
            : args.matched_controls_per_error
        ]
        chosen_uids = [str(row["event_uid"]) for row in chosen]
        selected_control_uids.update(chosen_uids)
        control_matches[str(error["event_uid"])] = chosen_uids

    memberships: dict[str, set[str]] = {}
    for uid in probability_uids:
        memberships.setdefault(uid, set()).add("PROBABILITY_SAMPLE")
    for row in errors:
        memberships.setdefault(str(row["event_uid"]), set()).add("ERROR_HARVEST")
    for uid in selected_control_uids:
        memberships.setdefault(uid, set()).add("MATCHED_CONTROL")

    matched_to: dict[str, str] = {}
    for error_uid, control_uids in control_matches.items():
        for control_uid in control_uids:
            matched_to[control_uid] = error_uid

    base_items = []
    for event_uid, queues in memberships.items():
        item = work_item(
            association_by_event[event_uid],
            route_by_event.get(event_uid),
            sorted(queues),
        )
        if event_uid in probability_uids:
            item["private_probability_rank"] = stable_rank(
                args.seed, event_uid, "probability"
            )
        if event_uid in matched_to:
            item["private_matched_to_error_event_uid"] = matched_to[event_uid]
        base_items.append(item)
    base_items.sort(key=lambda item: stable_rank(args.seed, item["event_uid"], "base-order"))

    repeat_count = int(math.ceil(len(base_items) * args.hidden_repeat_fraction))
    repeat_sources = sorted(
        base_items,
        key=lambda item: stable_rank(args.seed, item["event_uid"], "repeat"),
    )[:repeat_count]
    for index, item in enumerate(base_items, 1):
        item["case_uid"] = f"source_{index:04d}"
    repeats = []
    for index, source in enumerate(repeat_sources, 1):
        repeated = dict(source)
        repeated["case_uid"] = f"repeat_{index:04d}"
        repeated["sample_kind"] = "HIDDEN_REPEAT"
        repeated["private_queue_memberships"] = ["HIDDEN_REPEAT"]
        repeated["repeat_of"] = source["case_uid"]
        repeats.append(repeated)

    items = base_items + repeats
    random.Random(args.seed).shuffle(items)
    old_to_public = {}
    for index, item in enumerate(items, 1):
        old_uid = item["case_uid"]
        item["case_uid"] = f"{args.case_prefix}_{index:04d}"
        old_to_public[old_uid] = item["case_uid"]
    for item in items:
        if item.get("repeat_of"):
            item["repeat_of"] = old_to_public[str(item["repeat_of"])]

    args.output_root.mkdir(parents=True, exist_ok=True)
    worklist_path = args.output_root / "private_large_worklist.jsonl"
    write_jsonl_atomic(worklist_path, items)
    queue_counts = Counter()
    for queues in memberships.values():
        queue_counts.update(queues)
    route_counts = Counter(
        str(route_by_event[uid].get("private_auto_routing_label"))
        for uid in memberships
        if uid in route_by_event
    )
    manifest = {
        "schema_version": "experiment0-large-worklist/1.0",
        "status": "READY",
        "purpose": "ROOM0_DEVELOPMENT_LARGE_ANNOTATION",
        "statistical_contract": {
            "probability_queue": (
                "Uniform deterministic sample of the declared event population; "
                "only this queue may estimate room0 event prevalence."
            ),
            "harvest_queue": (
                "Private GT-audit error candidates plus decision-matched controls; "
                "must not estimate prevalence."
            ),
            "human_labels_required": True,
            "room0_is_development_not_unseen_validation": True,
        },
        "seed": args.seed,
        "case_prefix": args.case_prefix,
        "association_count": len(associations),
        "previous_unique_event_count": len(previous_event_uids),
        "probability_population_count": len(population),
        "probability_sample_count": len(probability_rows),
        "probability_inclusion_probability": len(probability_rows) / len(population),
        "error_harvest_count": len(errors),
        "matched_control_count": len(selected_control_uids),
        "matched_controls_per_error_requested": args.matched_controls_per_error,
        "unique_base_case_count": len(base_items),
        "hidden_repeat_count": len(repeats),
        "total_case_count": len(items),
        "queue_membership_counts": dict(sorted(queue_counts.items())),
        "private_auto_route_counts_in_unique_batch": dict(sorted(route_counts.items())),
        "source_sha256": {
            "associations": sha256_file(args.associations),
            "routing_records": sha256_file(args.routing_records),
            "previous_worklists": {
                str(path.resolve()): sha256_file(path) for path in args.previous_worklist
            },
        },
        "worklist_sha256": sha256_file(worklist_path),
    }
    write_json_atomic(args.output_root / "worklist_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
