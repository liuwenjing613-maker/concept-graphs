#!/usr/bin/env python3
"""Build a freshness-audited private worklist for Experiment 0 v2 R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROUTING_STRATA = {
    "CORRECT_ATTACH",
    "WRONG_ATTACH_EXISTING",
    "SHOULD_HAVE_BEEN_NEW",
    "CORRECT_NEW",
    "WRONG_NEW_FALSE_SPLIT",
}
EXCLUSION_STRATA = {
    "OUT_OF_SCOPE_MIXED_MULTIPLE_INSTANCES",
    "OUT_OF_SCOPE_BACKGROUND_OR_FRAGMENT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-spec", type=Path, required=True)
    parser.add_argument("--routing-records", type=Path, required=True)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--previous-worklist", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
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


def route_work_item(
    route: dict[str, Any], selection: dict[str, Any], source_index: int
) -> dict[str, Any]:
    expected = selection["private_auto_routing_label"]
    actual = route.get("private_auto_routing_label")
    if actual != expected:
        raise ValueError(
            f"{route['event_uid']}: selection expects {expected}, routing audit says {actual}"
        )
    if not route.get("private_auto_evaluable"):
        raise ValueError(f"{route['event_uid']}: selected routing row is not evaluable")
    return {
        "case_uid": f"r2_source_{source_index:03d}",
        "event_uid": route["event_uid"],
        "sample_kind": selection.get("sample_kind", "V2_R2_PRIVATE_BALANCED"),
        "private_auto_routing_label": actual,
        "private_auto_episode_role": route.get("private_auto_episode_role"),
        "private_causal_group_uid": route.get("private_causal_group_uid"),
        "private_legal_candidate_uids": route.get("private_legal_candidate_uids") or [],
        "private_same_gt_support_candidate_uids": (
            route.get("private_same_gt_support_candidate_uids") or []
        ),
        "private_obs_gt_id": route.get("private_obs_gt_id"),
        "private_obs_gt_label": route.get("private_obs_gt_label"),
        "private_obs_gt_purity": route.get("private_obs_gt_purity"),
        "private_routing_audit_row_sha256": row_sha256(route),
        "private_challenge_tags": sorted(set(selection.get("challenge_tags") or [])),
        "private_selection_rationale": selection.get("rationale"),
        "tminus_snapshot_sha256": route.get("tminus_snapshot_sha256"),
    }


def exclusion_work_item(
    association: dict[str, Any], selection: dict[str, Any], source_index: int
) -> dict[str, Any]:
    label = selection["private_auto_routing_label"]
    if label not in EXCLUSION_STRATA:
        raise ValueError(f"unsupported manual exclusion label: {label}")
    return {
        "case_uid": f"r2_source_{source_index:03d}",
        "event_uid": association["event_uid"],
        "sample_kind": selection.get("sample_kind", "V2_R2_PRIVATE_EXCLUSION"),
        "private_auto_routing_label": label,
        "private_auto_episode_role": "NOT_APPLICABLE",
        "private_causal_group_uid": None,
        "private_legal_candidate_uids": [],
        "private_same_gt_support_candidate_uids": [],
        "private_obs_gt_id": selection.get("private_obs_gt_id"),
        "private_obs_gt_label": selection.get("private_obs_gt_label"),
        "private_obs_gt_purity": selection.get("private_obs_gt_purity"),
        "private_exclusion_source": "CURATED_FROM_CORRECTED_OBSERVATION_GT_AND_VISUAL_REVIEW",
        "private_challenge_tags": sorted(set(selection.get("challenge_tags") or [])),
        "private_selection_rationale": selection.get("rationale"),
        "private_association_row_sha256": row_sha256(association),
    }


def main() -> int:
    args = parse_args()
    spec = read_json(args.selection_spec)
    selections = spec.get("base_cases")
    if not isinstance(selections, list) or not selections:
        raise ValueError("selection spec must contain nonempty base_cases")

    routes = {row["event_uid"]: row for row in read_jsonl(args.routing_records)}
    associations = {row["event_uid"]: row for row in read_jsonl(args.associations)}
    previous_rows = []
    for path in args.previous_worklist:
        previous_rows.extend(read_jsonl(path))
    previous_event_uids = {str(row["event_uid"]) for row in previous_rows}

    base_event_uids = [str(row["event_uid"]) for row in selections]
    if len(base_event_uids) != len(set(base_event_uids)):
        raise ValueError("base_cases contains duplicate event_uid")
    overlap = sorted(set(base_event_uids) & previous_event_uids)
    if overlap:
        raise ValueError("R2 selection overlaps previous worklists: " + ", ".join(overlap))

    base_items = []
    for index, selection in enumerate(selections, 1):
        event_uid = str(selection["event_uid"])
        label = str(selection["private_auto_routing_label"])
        association = associations.get(event_uid)
        if association is None:
            raise ValueError(f"association missing: {event_uid}")
        if label in ROUTING_STRATA:
            route = routes.get(event_uid)
            if route is None:
                raise ValueError(f"routing audit row missing: {event_uid}")
            item = route_work_item(route, selection, index)
        elif label in EXCLUSION_STRATA:
            item = exclusion_work_item(association, selection, index)
        else:
            raise ValueError(f"unsupported private stratum: {label}")
        base_items.append(item)

    expected_counts = Counter(spec.get("expected_base_counts") or {})
    actual_counts = Counter(item["private_auto_routing_label"] for item in base_items)
    if expected_counts and actual_counts != expected_counts:
        raise ValueError(
            f"base stratum counts differ: actual={dict(actual_counts)}, "
            f"expected={dict(expected_counts)}"
        )

    repeats = []
    repeat_event_uids = [str(value) for value in spec.get("hidden_repeat_event_uids") or []]
    by_event = {item["event_uid"]: item for item in base_items}
    for event_uid in repeat_event_uids:
        source = by_event.get(event_uid)
        if source is None:
            raise ValueError(f"hidden repeat source is not a base case: {event_uid}")
        repeated = dict(source)
        repeated["case_uid"] = f"r2_repeat_source_{len(repeats) + 1:03d}"
        repeated["sample_kind"] = "V2_R2_HIDDEN_REPEAT"
        repeated["repeat_of"] = source["case_uid"]
        repeats.append(repeated)

    items = base_items + repeats
    random.Random(int(spec.get("seed", 20260902))).shuffle(items)
    prefix = str(spec.get("case_uid_prefix") or "v2_r2")
    source_to_public = {}
    for index, item in enumerate(items, 1):
        old_uid = item["case_uid"]
        item["case_uid"] = f"{prefix}_{index:03d}"
        source_to_public[old_uid] = item["case_uid"]
    for item in items:
        repeat_of = item.get("repeat_of")
        if repeat_of:
            item["repeat_of"] = source_to_public[repeat_of]

    args.output_root.mkdir(parents=True, exist_ok=True)
    worklist_path = args.output_root / "private_v2_r2_worklist.jsonl"
    write_jsonl_atomic(worklist_path, items)

    challenge_counts = Counter()
    for item in base_items:
        challenge_counts.update(item.get("private_challenge_tags") or [])
    causal_by_stratum: dict[str, set[str]] = defaultdict(set)
    for item in base_items:
        causal = item.get("private_causal_group_uid")
        if causal:
            causal_by_stratum[item["private_auto_routing_label"]].add(str(causal))
    limitations = list(spec.get("declared_limitations") or [])
    for stratum, count in actual_counts.items():
        if stratum in ROUTING_STRATA and count > 1:
            causal_count = len(causal_by_stratum.get(stratum, set()))
            if causal_count and causal_count < count:
                limitations.append(
                    f"{stratum}: {count} base events cover only {causal_count} causal group(s)."
                )

    manifest = {
        "schema_version": "experiment0-v2-r2-worklist/1.0",
        "status": "READY" if not limitations else "READY_WITH_DECLARED_LIMITATIONS",
        "purpose": spec.get("purpose", "R2_SCHEMA_CALIBRATION_NOT_PREVALENCE"),
        "case_uid_prefix": prefix,
        "seed": int(spec.get("seed", 20260902)),
        "base_case_count": len(base_items),
        "hidden_repeat_count": len(repeats),
        "total_case_count": len(items),
        "unique_base_event_count": len(set(base_event_uids)),
        "previous_event_overlap_count": 0,
        "base_stratum_counts": dict(sorted(actual_counts.items())),
        "final_stratum_counts_including_repeats": dict(
            sorted(Counter(item["private_auto_routing_label"] for item in items).items())
        ),
        "challenge_tag_counts": dict(sorted(challenge_counts.items())),
        "causal_group_counts_by_stratum": {
            key: len(value) for key, value in sorted(causal_by_stratum.items())
        },
        "hidden_repeat_event_uids": repeat_event_uids,
        "declared_limitations": sorted(set(limitations)),
        "source_sha256": {
            "selection_spec": sha256_file(args.selection_spec),
            "routing_records": sha256_file(args.routing_records),
            "associations": sha256_file(args.associations),
            "previous_worklists": {
                str(path.resolve()): sha256_file(path) for path in args.previous_worklist
            },
        },
        "worklist_sha256": sha256_file(worklist_path),
        "warning": (
            "Private strata, challenge tags, repeats and selection rationales must remain "
            "hidden from the reviewer. This balanced set cannot estimate prevalence."
        ),
    }
    write_json_atomic(args.output_root / "r2_worklist_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
