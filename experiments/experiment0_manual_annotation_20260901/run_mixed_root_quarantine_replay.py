#!/usr/bin/env python3
"""Replay a mixed-mask root while quarantining the unsplittable observation.

This is an experiment-only counterfactual.  The anchor observation is omitted
from the replay input and from the comparison endpoint; every later observation
is routed by the unchanged online matcher.  Corrected instance GT is loaded only
after replay to audit the final partition.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.benchmark.human_error_pilot import (
    HumanSceneContext,
    evaluate_collateral,
)
from conceptgraph.revision.constraints import ReplayMode
from conceptgraph.revision.models import DependencyClosure
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.snapshot import AnchorStateBuilder

from run_human_oracle_minimal_replay import _full_suffix_closure, _partition_hash


SCHEMA_VERSION = "mixed-root-quarantine-replay/1.0"
DEFAULT_ANCHOR_OBS = "room0_20260831T111035Z_5c9d86fa_f000138_r0016"
DEFAULT_TARGET_VERSION = "30c6ac88-91e0-44bd-8667-6fd8df4c12a6@v000015"
DEFAULT_GT_IDS = (15, 19)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _owner_index(
    membership: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for entity_uid, members in membership.items():
        for obs_uid in members or ():
            owners.setdefault(str(obs_uid), []).append(str(entity_uid))
    return owners


def quarantine_observation(
    state: Mapping[str, Any], obs_uid: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an endpoint view in which exactly one observation is unowned."""

    value = copy.deepcopy(dict(state))
    original_membership = value.get("membership") or {}
    owners_before = [
        str(entity_uid)
        for entity_uid, members in original_membership.items()
        if obs_uid in set(str(item) for item in members or ())
    ]
    if len(owners_before) != 1:
        raise ValueError(f"quarantine anchor must have one native owner: {owners_before}")

    membership: dict[str, list[str]] = {}
    for entity_uid, members in original_membership.items():
        kept = sorted(set(str(item) for item in members or ()) - {obs_uid})
        if kept:
            membership[str(entity_uid)] = kept
    value["membership"] = membership

    objects = []
    object_rows_changed = 0
    for source in value.get("objects") or ():
        row = copy.deepcopy(dict(source))
        changed = False
        for key in ("member_observation_uids", "obs_uids"):
            if key not in row:
                continue
            before = [str(item) for item in row.get(key) or ()]
            after = [item for item in before if item != obs_uid]
            if len(after) != len(before):
                row[key] = after
                changed = True
        object_rows_changed += int(changed)
        objects.append(row)
    value["objects"] = objects
    value["state_hash"] = _partition_hash(membership)
    return value, {
        "anchor_obs_uid": obs_uid,
        "native_owner_uid": owners_before[0],
        "native_owner_count": len(owners_before),
        "object_rows_changed": object_rows_changed,
        "membership_contains_anchor_after": any(
            obs_uid in set(members) for members in membership.values()
        ),
    }


def closure_without_observation(
    closure: DependencyClosure,
    *,
    obs_uid: str,
    event_uids_for_observation: Iterable[str],
) -> DependencyClosure:
    excluded_events = set(str(item) for item in event_uids_for_observation)
    return DependencyClosure.build(
        event_uids=(
            item for item in closure.event_uids if item not in excluded_events
        ),
        version_uids=closure.version_uids,
        entity_uids=closure.entity_uids,
        obs_uids=(item for item in closure.obs_uids if item != obs_uid),
        edge_uids=closure.edge_uids,
        start_sequence=closure.start_sequence,
        end_sequence=closure.end_sequence,
    )


def _reliable_gt_rows(
    observation_gt: Mapping[str, Mapping[str, Any]],
    *,
    gt_ids: set[int],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(obs_uid): row
        for obs_uid, row in observation_gt.items()
        if row.get("gt_assignment_eligible")
        and row.get("gt_top_id") is not None
        and int(row["gt_top_id"]) in gt_ids
        and float(row.get("gt_purity") or 0.0) >= 0.8
        and not bool(row.get("mask_mixed"))
        and not bool(row.get("mask_two_foreground"))
    }


def audit_gt_partition(
    membership: Mapping[str, Iterable[str]],
    observation_gt: Mapping[str, Mapping[str, Any]],
    *,
    gt_ids: Iterable[int] = DEFAULT_GT_IDS,
    target_probe_obs_uid: str | None = None,
) -> dict[str, Any]:
    """Audit complete reliable-instance partitions, not a few hand-picked probes."""

    expected_ids = set(int(item) for item in gt_ids)
    reliable = _reliable_gt_rows(observation_gt, gt_ids=expected_ids)
    owners = _owner_index(membership)
    entity_members = {
        str(entity_uid): set(str(item) for item in members or ())
        for entity_uid, members in membership.items()
    }
    reliable_any_gt = {
        str(obs_uid): row
        for obs_uid, row in observation_gt.items()
        if row.get("gt_assignment_eligible")
        and row.get("gt_top_id") is not None
        and float(row.get("gt_purity") or 0.0) >= 0.8
        and not bool(row.get("mask_mixed"))
        and not bool(row.get("mask_two_foreground"))
    }

    per_gt: dict[str, Any] = {}
    best_entities: dict[int, str | None] = {}
    for gt_id in sorted(expected_ids):
        expected = {
            obs_uid
            for obs_uid, row in reliable.items()
            if int(row["gt_top_id"]) == gt_id
        }
        candidates = []
        for entity_uid, members in entity_members.items():
            true_members = members & expected
            if not true_members:
                continue
            eligible_members = members & set(reliable_any_gt)
            precision = len(true_members) / max(len(eligible_members), 1)
            recall = len(true_members) / max(len(expected), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            candidates.append(
                {
                    "entity_uid": entity_uid,
                    "true_member_count": len(true_members),
                    "eligible_member_count": len(eligible_members),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        candidates.sort(
            key=lambda row: (
                -float(row["f1"]),
                -int(row["true_member_count"]),
                str(row["entity_uid"]),
            )
        )
        best = candidates[0] if candidates else None
        best_entities[gt_id] = str(best["entity_uid"]) if best else None
        owned = sum(bool(owners.get(obs_uid)) for obs_uid in expected)
        duplicate = sum(len(owners.get(obs_uid, ())) > 1 for obs_uid in expected)
        per_gt[str(gt_id)] = {
            "expected_reliable_observation_count": len(expected),
            "owned_observation_count": owned,
            "duplicate_owner_observation_count": duplicate,
            "owner_entity_count": len(
                {
                    entity_uid
                    for obs_uid in expected
                    for entity_uid in owners.get(obs_uid, ())
                }
            ),
            "best_entity": best,
            "top_entities": candidates[:5],
        }

    target_owner_uids = (
        owners.get(str(target_probe_obs_uid), []) if target_probe_obs_uid else []
    )
    target_owner_composition: dict[str, int] = {}
    if len(target_owner_uids) == 1:
        target_members = entity_members[target_owner_uids[0]]
        target_owner_composition = dict(
            sorted(
                Counter(
                    int(reliable_any_gt[item]["gt_top_id"])
                    for item in target_members
                    if item in reliable_any_gt
                ).items()
            )
        )
    distinct = (
        len(expected_ids) == len(set(best_entities.values()))
        and None not in set(best_entities.values())
    )
    return {
        "evaluation_policy": (
            "POST_REPLAY_CORRECTED_GT_ONLY; purity>=0.8; "
            "exclude MIXED and two-foreground observations"
        ),
        "reliable_expected_observation_count": len(reliable),
        "per_gt": per_gt,
        "best_entities_are_distinct": distinct,
        "best_entity_by_gt": {
            str(key): value for key, value in sorted(best_entities.items())
        },
        "target_probe_obs_uid": target_probe_obs_uid,
        "target_probe_owner_uids": target_owner_uids,
        "target_owner_reliable_gt_composition": target_owner_composition,
    }


def _affected_native_observations(
    membership: Mapping[str, Iterable[str]],
    observation_gt: Mapping[str, Mapping[str, Any]],
    gt_ids: set[int],
) -> set[str]:
    reliable = _reliable_gt_rows(observation_gt, gt_ids=gt_ids)
    owners = _owner_index(membership)
    affected_entities = {
        entity_uid
        for obs_uid in reliable
        for entity_uid in owners.get(obs_uid, ())
    }
    return {
        str(obs_uid)
        for entity_uid in affected_entities
        for obs_uid in membership.get(entity_uid, ())
    }


def _markdown(result: Mapping[str, Any]) -> str:
    native = result["branches"]["B0_NATIVE"]["gt_partition"]
    quarantine = result["branches"]["Q1_QUARANTINE"]["gt_partition"]
    lines = [
        "# Room0 frame138 混合 mask 隔离反事实",
        "",
        f"- 执行状态：**{result['execution_status']}**",
        f"- 科学结论：**{result['scientific_outcome']}**",
        f"- anchor：`{result['anchor']['obs_uid']}`",
        "- 干预：只隔离 anchor，不给后续 observation 注入 GT 或人工路线。",
        "- 评测：回放完成后才读取校正 GT；MIXED observation 不参与完整实例指标。",
        "",
        "| 分支 | GT15 best P/R/F1 | GT19 best P/R/F1 | 两实例最佳实体不同 |",
        "|---|---|---|---:|",
    ]
    for name, audit in (("B0 原始", native), ("Q1 隔离", quarantine)):
        values = []
        for gt_id in (15, 19):
            best = audit["per_gt"][str(gt_id)]["best_entity"]
            if best is None:
                values.append("N/A")
            else:
                values.append(
                    "{:.3f}/{:.3f}/{:.3f}".format(
                        best["precision"], best["recall"], best["f1"]
                    )
                )
        lines.append(
            f"| {name} | {values[0]} | {values[1]} | "
            f"{audit['best_entities_are_distinct']} |"
        )
    lines.extend(
        [
            "",
            "## 完整性边界",
            "",
            f"- anchor 最终 owner 数：{result['anchor']['quarantine_owner_count']}（期望 0）。",
            f"- 其余 observation 不变量：{result['runtime_invariants']['pass']}。",
            f"- 受影响集合之外改动：{result['collateral']['outside_changed']}。",
            f"- 源证据结束校验未变：{result['source_evidence_unchanged']}。",
            "- 这是人工/离线发现混合根因后的 oracle 隔离上限，不代表自动混合检测器已经完成。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--observation-gt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--beauty-summary", type=Path)
    parser.add_argument("--anchor-obs", default=DEFAULT_ANCHOR_OBS)
    parser.add_argument("--target-version", default=DEFAULT_TARGET_VERSION)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("[1/5] build immutable room0 context", flush=True)
    context = HumanSceneContext.build("room0", args.room_run.resolve())
    native_state = copy.deepcopy(context.native_state)
    provenance = context.provenance

    anchor_obs = str(args.anchor_obs)
    association = provenance.get_association_for_obs(anchor_obs)
    anchor_event_uid = str(association["event_uid"])
    anchor_sequence = provenance.sequence(association)
    anchor_frame = int(str(association["frame_uid"]).rsplit("_f", 1)[-1])
    target_version = provenance.get_object_version(str(args.target_version))
    target_probe = str(
        target_version.get("origin_observation_uid")
        or target_version["member_observation_uids"][0]
    )
    if anchor_obs in set(target_version.get("member_observation_uids") or ()):
        raise ValueError("target version is not a strict pre-anchor version")

    print("[2/5] build strict pre-anchor snapshot", flush=True)
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(anchor_frame)
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        anchor_event_uid,
        [str(args.target_version)],
        strict=True,
        prefix_state=prefix_state,
        prefix_objects=prefix_objects,
    )
    full_closure = _full_suffix_closure(
        context,
        anchor_event_sequence=anchor_sequence,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
    )
    anchor_related_events = [
        str(row["event_uid"])
        for row in provenance.events.values()
        if str(row.get("obs_uid") or "") == anchor_obs
    ]
    quarantine_closure = closure_without_observation(
        full_closure,
        obs_uid=anchor_obs,
        event_uids_for_observation=anchor_related_events,
    )
    quarantine_current, quarantine_edit = quarantine_observation(
        native_state, anchor_obs
    )
    _atomic_json(
        output_root / "intake.json",
        {
            "schema_version": SCHEMA_VERSION,
            "anchor_obs_uid": anchor_obs,
            "anchor_event_uid": anchor_event_uid,
            "anchor_event_sequence": anchor_sequence,
            "anchor_frame": anchor_frame,
            "target_version_uid": str(args.target_version),
            "target_probe_obs_uid": target_probe,
            "snapshot": snapshot.as_dict(),
            "full_suffix_observation_count": len(full_closure.obs_uids),
            "quarantine_suffix_observation_count": len(quarantine_closure.obs_uids),
            "quarantine_edit": quarantine_edit,
        },
    )
    print(
        f"[snapshot] pass={snapshot.validation['pass']} "
        f"suffix={len(quarantine_closure.obs_uids)} anchor_removed=1",
        flush=True,
    )

    print("[3/5] replay unchanged matcher with anchor quarantined", flush=True)
    replay_state = context.engine.replay_suffix_from_snapshot(
        mode=ReplayMode.NATURAL_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=quarantine_closure,
        current_state=quarantine_current,
    )
    replay_state["branch"] = "Q1_MIXED_ANCHOR_QUARANTINE_NATURAL_SUFFIX"
    _atomic_gzip_json(output_root / "Q1_state.json.gz", replay_state)

    print("[4/5] audit complete partitions with post-replay GT", flush=True)
    gt_rows = _read_jsonl(args.observation_gt.resolve())
    observation_gt = {str(row["obs_uid"]): row for row in gt_rows}
    native_audit = audit_gt_partition(
        native_state["membership"],
        observation_gt,
        target_probe_obs_uid=target_probe,
    )
    replay_audit = audit_gt_partition(
        replay_state["membership"],
        observation_gt,
        target_probe_obs_uid=target_probe,
    )
    known_without_anchor = set(provenance.observations) - {anchor_obs}
    invariants = InvariantVerifier().verify(
        state=replay_state,
        constraints=(),
        known_observation_uids=known_without_anchor,
    )
    replay_owners = _owner_index(replay_state["membership"])
    affected = _affected_native_observations(
        native_state["membership"], observation_gt, set(DEFAULT_GT_IDS)
    ) | {anchor_obs}
    collateral = evaluate_collateral(
        native_state["membership"], replay_state["membership"], affected
    )

    native_gt19 = native_audit["per_gt"]["19"]["best_entity"] or {}
    replay_gt19 = replay_audit["per_gt"]["19"]["best_entity"] or {}
    f1_gain = float(replay_gt19.get("f1", 0.0)) - float(
        native_gt19.get("f1", 0.0)
    )
    separated = bool(replay_audit["best_entities_are_distinct"])
    scientific_outcome = (
        "QUARANTINE_SUPPORTS_NATURAL_RECOVERY"
        if separated and f1_gain > 0.05
        else "QUARANTINE_INSUFFICIENT_FOR_NATURAL_RECOVERY"
    )

    print("[5/5] verify source and write report", flush=True)
    source_unchanged = context.source_hashes_before == provenance.source_hashes()
    quarantine_owner_count = len(replay_owners.get(anchor_obs, ()))
    execution_pass = bool(
        snapshot.validation["pass"]
        and invariants["pass"]
        and quarantine_owner_count == 0
        and collateral["outside_changed"] == 0
        and source_unchanged
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "execution_status": "PASS" if execution_pass else "FAIL",
        "scientific_outcome": scientific_outcome,
        "intervention_semantics": (
            "OMIT_ONE_MIXED_ANCHOR; NO_GT_OR_ROUTE_FOR_LATER_OBSERVATIONS"
        ),
        "gt_use": "POST_REPLAY_EVALUATION_ONLY",
        "anchor": {
            "obs_uid": anchor_obs,
            "event_uid": anchor_event_uid,
            "frame": anchor_frame,
            "target_version_uid": str(args.target_version),
            "target_probe_obs_uid": target_probe,
            "quarantine_owner_count": quarantine_owner_count,
        },
        "snapshot_pass": bool(snapshot.validation["pass"]),
        "branches": {
            "B0_NATIVE": {
                "partition_hash": _partition_hash(native_state["membership"]),
                "gt_partition": native_audit,
            },
            "Q1_QUARANTINE": {
                "partition_hash": _partition_hash(replay_state["membership"]),
                "replayed_observation_count": replay_state.get(
                    "replayed_observations"
                ),
                "closure_initial_observation_count": replay_state.get(
                    "closure_initial_observation_count"
                ),
                "closure_effective_observation_count": replay_state.get(
                    "closure_effective_observation_count"
                ),
                "runtime_ms": replay_state.get("runtime_ms"),
                "gt_partition": replay_audit,
            },
        },
        "gt19_best_entity_f1_gain": f1_gain,
        "runtime_invariants": invariants,
        "collateral": collateral,
        "source_evidence_unchanged": source_unchanged,
        "source_hash_policy": "HASH_ONCE_AT_CONTEXT_BUILD_AND_ONCE_AT_END",
        "total_wall_ms": (time.perf_counter() - started) * 1000.0,
    }
    _atomic_json(output_root / "metrics.json", result)
    report = _markdown(result)
    report_path = output_root / "ROOM0_FRAME138_MIXED_QUARANTINE_REPLAY_CN.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    if args.beauty_summary:
        beauty_path = args.beauty_summary.resolve()
        beauty_path.parent.mkdir(parents=True, exist_ok=True)
        beauty_path.write_text(report, encoding="utf-8", newline="\n")
    print(
        f"[done] execution={'PASS' if execution_pass else 'FAIL'} "
        f"outcome={scientific_outcome} gt19_f1_gain={f1_gain:.4f}",
        flush=True,
    )
    return 0 if execution_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
