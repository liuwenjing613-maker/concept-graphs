#!/usr/bin/env python3
"""Oracle ceiling: filter recurring mixed masks, then create on first clean GT19.

This experiment deliberately uses corrected GT to choose which observations are
quarantined and to select one clean CREATE_INSTANCE trigger.  It is therefore
an offline upper-bound experiment, not an online detector or a deployable rule.
All non-quarantined observations after the one-shot trigger are routed by the
unchanged matcher.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.benchmark.human_error_pilot import (
    HumanSceneContext,
    evaluate_collateral,
)
from conceptgraph.revision.constraints import ReplayMode, SparseRepairConstraint
from conceptgraph.revision.models import DependencyClosure
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.snapshot import AnchorStateBuilder

from analyze_mixed_root_temporal_chain import (
    DEFAULT_TARGET_ORIGIN,
    is_pair_mixed,
    is_pure_reliable,
    read_jsonl,
)
from run_human_oracle_minimal_replay import _full_suffix_closure, _partition_hash
from run_mixed_root_quarantine_replay import (
    DEFAULT_ANCHOR_OBS,
    DEFAULT_GT_IDS,
    DEFAULT_TARGET_VERSION,
    _affected_native_observations,
    _atomic_gzip_json,
    _atomic_json,
    _owner_index,
    _read_gzip_json,
    audit_gt_partition,
)


SCHEMA_VERSION = "mixed-interval-clean-create-replay/1.0"
DEFAULT_TRIGGER_OBS = "room0_20260831T111035Z_5c9d86fa_f000253_r0011"


def quarantine_observations(
    state: Mapping[str, Any], obs_uids: Iterable[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove a verified set of observations from final endpoint ownership."""

    excluded = set(str(item) for item in obs_uids)
    value = copy.deepcopy(dict(state))
    native_owners = _owner_index(value.get("membership") or {})
    invalid = {
        obs_uid: native_owners.get(obs_uid, [])
        for obs_uid in sorted(excluded)
        if len(native_owners.get(obs_uid, [])) != 1
    }
    if invalid:
        raise ValueError(f"quarantine observations require one native owner: {invalid}")

    membership: dict[str, list[str]] = {}
    removed_memberships = 0
    for entity_uid, members in (value.get("membership") or {}).items():
        before = [str(item) for item in members or ()]
        after = sorted(set(before) - excluded)
        removed_memberships += len(before) - len(after)
        if after:
            membership[str(entity_uid)] = after
    value["membership"] = membership

    objects = []
    object_rows_changed = 0
    object_member_references_removed = 0
    for source in value.get("objects") or ():
        row = copy.deepcopy(dict(source))
        changed = False
        for key in ("member_observation_uids", "obs_uids"):
            if key not in row:
                continue
            before = [str(item) for item in row.get(key) or ()]
            after = [item for item in before if item not in excluded]
            if len(after) != len(before):
                row[key] = after
                object_member_references_removed += len(before) - len(after)
                changed = True
        object_rows_changed += int(changed)
        objects.append(row)
    value["objects"] = objects
    value["state_hash"] = _partition_hash(membership)
    owners_after = _owner_index(membership)
    still_owned = sorted(obs_uid for obs_uid in excluded if owners_after.get(obs_uid))
    return value, {
        "requested_observation_count": len(excluded),
        "removed_membership_count": removed_memberships,
        "object_rows_changed": object_rows_changed,
        "object_member_references_removed": object_member_references_removed,
        "still_owned_observation_uids": still_owned,
    }


def closure_without_observations(
    closure: DependencyClosure,
    *,
    obs_uids: Iterable[str],
    event_uids: Iterable[str],
) -> DependencyClosure:
    excluded_obs = set(str(item) for item in obs_uids)
    excluded_events = set(str(item) for item in event_uids)
    return DependencyClosure.build(
        event_uids=(
            item for item in closure.event_uids if item not in excluded_events
        ),
        version_uids=closure.version_uids,
        entity_uids=closure.entity_uids,
        obs_uids=(item for item in closure.obs_uids if item not in excluded_obs),
        edge_uids=closure.edge_uids,
        start_sequence=closure.start_sequence,
        end_sequence=closure.end_sequence,
    )


def create_constraint(
    *,
    trigger_obs: str,
    trigger_event_uid: str,
    trigger_sequence: int,
    target_lineage_uid: str,
) -> SparseRepairConstraint:
    experiment_uid = "room0-gt15-gt19-mixed-filter-clean-create"
    identity_uid = f"oracle-ceiling-identity:{experiment_uid}"
    entity_uid = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"oracle-ceiling-entity:{experiment_uid}")
    )
    return SparseRepairConstraint.from_mapping(
        {
            "type": "CREATE_INSTANCE",
            "obs_uid": trigger_obs,
            "created_lineage_uid": identity_uid,
            "created_identity_uid": identity_uid,
            "created_entity_uid": entity_uid,
            "separate_from_identity_uids": [target_lineage_uid],
            "applies_at_event_uid": trigger_event_uid,
            "active_from_sequence": trigger_sequence,
            "active_until_sequence": trigger_sequence,
            "source": "offline_gt_oracle_ceiling_experiment0",
            "evidence_refs": [
                experiment_uid,
                trigger_event_uid,
                trigger_obs,
            ],
        }
    )


def select_old_preferred_boundary_near_ties(
    state: Mapping[str, Any],
    *,
    old_entity_uid: str,
    new_entity_uid: str,
    after_frame: int,
    margin_delta: float,
) -> dict[str, Any]:
    """Select score-only old/new boundary ambiguities; no GT is consulted."""

    pair_rows: list[dict[str, Any]] = []
    for decision in state.get("decision_trace") or ():
        frame_idx = int(decision.get("frame_idx", -1))
        if frame_idx <= after_frame:
            continue
        eligible = {
            str(row["entity_uid"]): row
            for row in decision.get("natural_candidates") or ()
            if row.get("entity_uid") and row.get("eligible")
        }
        if old_entity_uid not in eligible or new_entity_uid not in eligible:
            continue
        old_row = eligible[old_entity_uid]
        new_row = eligible[new_entity_uid]
        old_score = float(old_row["score"])
        new_score = float(new_row["score"])
        old_preferred = decision.get("applied_match") == old_row.get("index")
        pair_rows.append(
            {
                "obs_uid": str(decision["obs_uid"]),
                "event_uid": str(decision["event_uid"]),
                "frame_idx": frame_idx,
                "old_candidate_index": old_row.get("index"),
                "new_candidate_index": new_row.get("index"),
                "old_score": old_score,
                "new_score": new_score,
                "old_minus_new_score": old_score - new_score,
                "old_preferred": old_preferred,
                "selected": bool(
                    old_preferred
                    and 0.0 < old_score - new_score <= margin_delta
                ),
            }
        )
    selected = [row for row in pair_rows if row["selected"]]
    unselected_old_margins = sorted(
        float(row["old_minus_new_score"])
        for row in pair_rows
        if row["old_preferred"] and not row["selected"]
    )
    return {
        "selection_semantics": (
            "POST_Q3_SCORE_ONLY; BOTH_BOUNDARY_CANDIDATES_ELIGIBLE; "
            "OLD_SELECTED; 0<OLD_MINUS_NEW<=DELTA; NO_GT"
        ),
        "margin_delta": margin_delta,
        "both_eligible_count": len(pair_rows),
        "old_preferred_count": sum(bool(row["old_preferred"]) for row in pair_rows),
        "selected_count": len(selected),
        "selected": selected,
        "minimum_unselected_old_preferred_margin": (
            unselected_old_margins[0] if unselected_old_margins else None
        ),
        "all_boundary_pair_rows": pair_rows,
    }


def assign_constraint(
    *,
    obs_uid: str,
    event_uid: str,
    event_sequence: int,
    target_lineage_uid: str,
    target_origin_obs_uid: str,
    target_entity_uid: str,
    margin_delta: float,
) -> SparseRepairConstraint:
    return SparseRepairConstraint.from_mapping(
        {
            "type": "ASSIGN_OBSERVATION",
            "obs_uid": obs_uid,
            "target_lineage_uid": target_lineage_uid,
            "target_origin_obs_uid": target_origin_obs_uid,
            "target_entity_uid": target_entity_uid,
            "applies_at_event_uid": event_uid,
            "active_from_sequence": event_sequence,
            "active_until_sequence": event_sequence,
            "source": "exploratory_boundary_margin_hysteresis_ablation",
            "evidence_refs": [
                "q3-old-new-eligible-score-margin",
                f"margin_delta:{margin_delta}",
                event_uid,
                obs_uid,
            ],
        }
    )


def branch_summary(
    state: Mapping[str, Any],
    observation_gt: Mapping[str, Mapping[str, Any]],
    *,
    target_probe: str,
) -> dict[str, Any]:
    return {
        "partition_hash": _partition_hash(state["membership"]),
        "status": state.get("status"),
        "replayed_observation_count": state.get("replayed_observations"),
        "runtime_ms": state.get("runtime_ms"),
        "gt_partition": audit_gt_partition(
            state["membership"],
            observation_gt,
            target_probe_obs_uid=target_probe,
        ),
    }


def natural_continuation_audit(
    state: Mapping[str, Any],
    observation_gt: Mapping[str, Mapping[str, Any]],
    *,
    trigger_obs: str,
    trigger_frame: int,
) -> dict[str, Any]:
    owners = _owner_index(state["membership"])
    trigger_owners = owners.get(trigger_obs, [])
    later_gt19 = sorted(
        str(obs_uid)
        for obs_uid, row in observation_gt.items()
        if int(row.get("frame_idx", -1)) > trigger_frame
        and is_pure_reliable(row, set(DEFAULT_GT_IDS))
        and int(row["gt_top_id"]) == 19
    )
    same_owner = 0
    if len(trigger_owners) == 1:
        trigger_owner = trigger_owners[0]
        same_owner = sum(trigger_owner in owners.get(obs_uid, []) for obs_uid in later_gt19)
    return {
        "trigger_owner_uids": trigger_owners,
        "later_pure_gt19_observation_count": len(later_gt19),
        "later_pure_gt19_same_owner_count": same_owner,
        "later_pure_gt19_same_owner_fraction": (
            same_owner / len(later_gt19) if later_gt19 else None
        ),
        "later_pure_gt19_observation_uids": later_gt19,
    }


def markdown(result: Mapping[str, Any]) -> str:
    rows = []
    for key, name in (
        ("B0_NATIVE", "B0 原始"),
        ("Q2_FILTER_ONLY", "Q2 仅过滤混合"),
        ("Q3_FILTER_PLUS_CLEAN_CREATE", "Q3 过滤 + 单次纯净 CREATE"),
        (
            "Q4_FILTER_CREATE_MARGIN_HYSTERESIS",
            "Q4 Q3 + 低间隔身份滞回消融",
        ),
    ):
        audit = result["branches"][key]["gt_partition"]
        values = []
        for gt_id in (15, 19):
            best = audit["per_gt"][str(gt_id)]["best_entity"]
            values.append(
                "N/A"
                if best is None
                else "{:.3f}/{:.3f}/{:.3f}".format(
                    best["precision"], best["recall"], best["f1"]
                )
            )
        rows.append(
            f"| {name} | {values[0]} | {values[1]} | "
            f"{audit['best_entities_are_distinct']} |"
        )

    q3_continuation = result["natural_continuation"][
        "Q3_FILTER_PLUS_CLEAN_CREATE"
    ]
    q4_continuation = result["natural_continuation"][
        "Q4_FILTER_CREATE_MARGIN_HYSTERESIS"
    ]
    margin = result["margin_hysteresis_ablation"]["selection"]
    selected = margin["selected"][0]
    return "\n".join(
        [
            "# Room0 混合区间过滤 + 首个纯净 GT19 CREATE 上限实验",
            "",
            f"- 执行状态：**{result['execution_status']}**",
            f"- 科学结论：**{result['scientific_outcome']}**",
            f"- 离线过滤的混合 observation：{result['oracle_selection']['mixed_observation_count']} 个。",
            f"- 单次 CREATE 触发：`{result['oracle_selection']['clean_create_trigger_obs_uid']}`。",
            "- 触发后不再提供逐帧身份路线；其余保留 observation 由原 matcher 自然关联。",
            "",
            "| 分支 | GT15 best P/R/F1 | GT19 best P/R/F1 | 最佳实体不同 |",
            "|---|---|---|---:|",
            *rows,
            "",
            "## 关键验证",
            "",
            f"- Q3：CREATE 后剩余纯净 GT19 共 {q3_continuation['later_pure_gt19_observation_count']} 个，自然进入新实体 {q3_continuation['later_pure_gt19_same_owner_count']} 个。",
            f"- 唯一残差：`{selected['obs_uid']}`；旧实体仅高 {selected['old_minus_new_score']:.6f} 分。",
            f"- Q4 只对该低间隔边界歧义做一次身份滞回后，新实体覆盖 {q4_continuation['later_pure_gt19_same_owner_count']}/{q4_continuation['later_pure_gt19_observation_count']} 个后续纯净 GT19。",
            f"- 同一 Q3 trace 中，两边都合格的候选共 {margin['both_eligible_count']} 个；旧实体胜出 {margin['old_preferred_count']} 个；0.03 内仅 {margin['selected_count']} 个，下一旧实体胜出间隔为 {margin['minimum_unselected_old_preferred_margin']:.6f}。",
            f"- 过滤分支不变量：{result['runtime_invariants']['Q2_FILTER_ONLY']['pass']}。",
            f"- 修复分支不变量：{result['runtime_invariants']['Q3_FILTER_PLUS_CLEAN_CREATE']['pass']}。",
            f"- 滞回消融分支不变量：{result['runtime_invariants']['Q4_FILTER_CREATE_MARGIN_HYSTERESIS']['pass']}。",
            f"- 受影响集合外改动：Q2={result['collateral']['Q2_FILTER_ONLY']['changed_outside_observation_count']}，Q3={result['collateral']['Q3_FILTER_PLUS_CLEAN_CREATE']['changed_outside_observation_count']}，Q4={result['collateral']['Q4_FILTER_CREATE_MARGIN_HYSTERESIS']['changed_outside_observation_count']}。",
            f"- 源证据结束校验未变：{result['source_evidence_unchanged']}。",
            "",
            "## 结论边界",
            "",
            "- 这是使用校正 GT 选择混合帧与 CREATE 时机的 oracle ceiling，只回答修复机制在理想质量门控下是否足够。",
            "- 它不能证明在线系统已经能识别混合 mask，也不能直接作为最终方法效果。",
            "- Q4 的残差选择不读取 GT，但 0.03 阈值是在本例结果上事后选定；它只是新假设，必须用其他实例/场景验证后才能成为方法。",
            "- CREATE 的 association boundary 在本实验关闭，postprocess identity boundary 保留；否则新实例无法接收后续自然证据。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--observation-gt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--beauty-summary", type=Path)
    parser.add_argument("--anchor-obs", default=DEFAULT_ANCHOR_OBS)
    parser.add_argument("--target-version", default=DEFAULT_TARGET_VERSION)
    parser.add_argument("--trigger-obs", default=DEFAULT_TRIGGER_OBS)
    parser.add_argument("--margin-delta", type=float, default=0.03)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse completed Q2/Q3/Q4 gzip states in output-root when present.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("[1/9] build immutable room0 context", flush=True)
    context = HumanSceneContext.build("room0", args.room_run.resolve())
    provenance = context.provenance
    native_state = copy.deepcopy(context.native_state)

    gt_rows = read_jsonl(args.observation_gt.resolve())
    observation_gt = {str(row["obs_uid"]): row for row in gt_rows}
    affected_ids = set(DEFAULT_GT_IDS)
    mixed_obs = sorted(
        str(row["obs_uid"])
        for row in gt_rows
        if int(row.get("frame_idx", -1)) >= 138
        and is_pair_mixed(row, affected_ids)
    )
    trigger_obs = str(args.trigger_obs)
    trigger_gt = observation_gt.get(trigger_obs)
    if trigger_gt is None or not is_pure_reliable(trigger_gt, affected_ids):
        raise ValueError("CREATE trigger must be a corrected-GT pure reliable row")
    if int(trigger_gt["gt_top_id"]) != 19:
        raise ValueError("CREATE trigger must be GT19")
    if trigger_obs in set(mixed_obs):
        raise ValueError("CREATE trigger cannot also be quarantined")

    anchor_obs = str(args.anchor_obs)
    anchor_association = provenance.get_association_for_obs(anchor_obs)
    anchor_event_uid = str(anchor_association["event_uid"])
    anchor_sequence = provenance.sequence(anchor_association)
    anchor_frame = int(str(anchor_association["frame_uid"]).rsplit("_f", 1)[-1])
    target_version = provenance.get_object_version(str(args.target_version))
    target_probe = str(
        target_version.get("origin_observation_uid")
        or target_version["member_observation_uids"][0]
    )
    target_lineage_uid = str(target_version["lineage_uid"])
    if target_probe != DEFAULT_TARGET_ORIGIN:
        raise ValueError(f"unexpected target origin: {target_probe}")

    trigger_association = provenance.get_association_for_obs(trigger_obs)
    trigger_event_uid = str(trigger_association["event_uid"])
    trigger_sequence = provenance.sequence(trigger_association)
    trigger_frame = int(trigger_gt["frame_idx"])
    primitive = create_constraint(
        trigger_obs=trigger_obs,
        trigger_event_uid=trigger_event_uid,
        trigger_sequence=trigger_sequence,
        target_lineage_uid=target_lineage_uid,
    )
    primitive_value = primitive.as_dict()
    created_lineage_uid = str(primitive_value["created_lineage_uid"])
    created_entity_uid = str(primitive_value["created_entity_uid"])
    target_owner_uids = _owner_index(native_state["membership"]).get(
        target_probe, []
    )
    if len(target_owner_uids) != 1:
        raise ValueError(f"target probe must have one native owner: {target_owner_uids}")
    old_entity_uid = target_owner_uids[0]

    print("[2/9] build strict pre-frame138 snapshot and filtered suffix", flush=True)
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
    excluded_events = [
        str(row["event_uid"])
        for row in provenance.events.values()
        if str(row.get("obs_uid") or "") in set(mixed_obs)
    ]
    filtered_closure = closure_without_observations(
        full_closure,
        obs_uids=mixed_obs,
        event_uids=excluded_events,
    )
    filtered_current, quarantine_edit = quarantine_observations(
        native_state, mixed_obs
    )
    if quarantine_edit["still_owned_observation_uids"]:
        raise ValueError("quarantined observations remain owned")

    component_policy = {
        "positive_lineage_redirect": True,
        "create_association_boundary": False,
        "create_postprocess_boundary": True,
    }
    intake = {
        "schema_version": SCHEMA_VERSION,
        "oracle_selection_semantics": "CORRECTED_GT_PRESELECTED_OFFLINE_UPPER_BOUND",
        "anchor_obs_uid": anchor_obs,
        "anchor_event_uid": anchor_event_uid,
        "anchor_frame": anchor_frame,
        "target_version_uid": str(args.target_version),
        "target_origin_obs_uid": target_probe,
        "mixed_observation_count": len(mixed_obs),
        "mixed_observation_uids": mixed_obs,
        "clean_create_trigger_obs_uid": trigger_obs,
        "clean_create_trigger_frame": trigger_frame,
        "clean_create_trigger_event_uid": trigger_event_uid,
        "full_suffix_observation_count": len(full_closure.obs_uids),
        "filtered_suffix_observation_count": len(filtered_closure.obs_uids),
        "quarantine_edit": quarantine_edit,
        "constraint": primitive_value,
        "component_policy": component_policy,
        "snapshot": snapshot.as_dict(),
    }
    _atomic_json(output_root / "intake.json", intake)
    print(
        f"[intake] snapshot_pass={snapshot.validation['pass']} "
        f"mixed_removed={len(mixed_obs)} suffix={len(filtered_closure.obs_uids)}",
        flush=True,
    )

    common = {
        "snapshot_objects": snapshot.objects,
        "snapshot_runtime_ms": snapshot.state["runtime_ms"],
        "snapshot_timing": snapshot.state.get("timing"),
        "anchor_frame": snapshot.anchor_frame,
        "snapshot_watermark_event_sequence": snapshot.watermark_event_sequence,
        "closure": filtered_closure,
        "current_state": filtered_current,
    }
    q2_path = output_root / "Q2_filter_only_state.json.gz"
    if args.resume_existing and q2_path.exists():
        print("[3/9] reuse Q2 filter-only state", flush=True)
        q2 = _read_gzip_json(q2_path)
    else:
        print("[3/9] replay Q2 filter-only natural suffix", flush=True)
        q2 = context.engine.replay_suffix_from_snapshot(
            mode=ReplayMode.NATURAL_REPLAY,
            **common,
        )
        q2["branch"] = "Q2_ORACLE_MIXED_FILTER_ONLY_NATURAL_SUFFIX"
        _atomic_gzip_json(q2_path, q2)
    print(
        f"[Q2] status={q2.get('status')} replayed={q2.get('replayed_observations')} "
        f"runtime_ms={q2.get('runtime_ms')}",
        flush=True,
    )

    q3_path = output_root / "Q3_filter_plus_clean_create_state.json.gz"
    if args.resume_existing and q3_path.exists():
        print("[4/9] reuse Q3 filter-plus-create state", flush=True)
        q3 = _read_gzip_json(q3_path)
    else:
        print("[4/9] replay Q3 filtered suffix with one clean CREATE", flush=True)
        q3 = context.engine.replay_local_from_snapshot(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=[primitive],
            component_policy=component_policy,
            **common,
        )
        q3["branch"] = "Q3_ORACLE_MIXED_FILTER_PLUS_ONE_CLEAN_CREATE"
        _atomic_gzip_json(q3_path, q3)
    print(
        f"[Q3] status={q3.get('status')} replayed={q3.get('replayed_observations')} "
        f"runtime_ms={q3.get('runtime_ms')}",
        flush=True,
    )

    print("[5/9] select score-only boundary near-tie from Q3", flush=True)
    margin_selection = select_old_preferred_boundary_near_ties(
        q3,
        old_entity_uid=old_entity_uid,
        new_entity_uid=created_entity_uid,
        after_frame=trigger_frame,
        margin_delta=float(args.margin_delta),
    )
    if margin_selection["selected_count"] != 1:
        raise ValueError(
            "expected exactly one evidence-only boundary near-tie, got "
            f"{margin_selection['selected_count']}"
        )
    margin_row = margin_selection["selected"][0]
    margin_association = provenance.get_association_for_obs(
        str(margin_row["obs_uid"])
    )
    margin_primitive = assign_constraint(
        obs_uid=str(margin_row["obs_uid"]),
        event_uid=str(margin_row["event_uid"]),
        event_sequence=provenance.sequence(margin_association),
        target_lineage_uid=created_lineage_uid,
        target_origin_obs_uid=trigger_obs,
        target_entity_uid=created_entity_uid,
        margin_delta=float(args.margin_delta),
    )
    intake["margin_hysteresis_selection"] = margin_selection
    intake["margin_hysteresis_constraint"] = margin_primitive.as_dict()
    _atomic_json(output_root / "intake.json", intake)
    _atomic_json(output_root / "margin_selection.json", margin_selection)
    print(
        f"[margin] selected={margin_row['obs_uid']} "
        f"old_minus_new={margin_row['old_minus_new_score']:.6f} "
        f"next_old_margin="
        f"{margin_selection['minimum_unselected_old_preferred_margin']}",
        flush=True,
    )

    q4_path = output_root / "Q4_filter_create_margin_state.json.gz"
    q4_constraints = [primitive, margin_primitive]
    if args.resume_existing and q4_path.exists():
        print("[6/9] reuse Q4 margin-hysteresis ablation state", flush=True)
        q4 = _read_gzip_json(q4_path)
    else:
        print("[6/9] replay Q4 with CREATE plus score-margin assignment", flush=True)
        q4 = context.engine.replay_local_from_snapshot(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=q4_constraints,
            component_policy=component_policy,
            **common,
        )
        q4["branch"] = "Q4_ORACLE_FILTER_CREATE_PLUS_MARGIN_HYSTERESIS_ABLATION"
        _atomic_gzip_json(q4_path, q4)
    print(
        f"[Q4] status={q4.get('status')} replayed={q4.get('replayed_observations')} "
        f"runtime_ms={q4.get('runtime_ms')}",
        flush=True,
    )

    print("[7/9] audit complete reliable partitions with corrected GT", flush=True)
    native_summary = branch_summary(
        native_state, observation_gt, target_probe=target_probe
    )
    q2_summary = branch_summary(q2, observation_gt, target_probe=target_probe)
    q3_summary = branch_summary(q3, observation_gt, target_probe=target_probe)
    q4_summary = branch_summary(q4, observation_gt, target_probe=target_probe)
    q3_continuation = natural_continuation_audit(
        q3,
        observation_gt,
        trigger_obs=trigger_obs,
        trigger_frame=trigger_frame,
    )
    q4_continuation = natural_continuation_audit(
        q4,
        observation_gt,
        trigger_obs=trigger_obs,
        trigger_frame=trigger_frame,
    )

    print("[8/9] verify ownership, collateral, and invariants", flush=True)
    known_without_mixed = set(provenance.observations) - set(mixed_obs)
    verifier = InvariantVerifier()
    q2_invariants = verifier.verify(
        state=q2,
        constraints=(),
        known_observation_uids=known_without_mixed,
    )
    q3_invariants = verifier.verify(
        state=q3,
        constraints=[primitive],
        known_observation_uids=known_without_mixed,
    )
    q4_invariants = verifier.verify(
        state=q4,
        constraints=q4_constraints,
        known_observation_uids=known_without_mixed,
    )
    affected = _affected_native_observations(
        native_state["membership"], observation_gt, affected_ids
    ) | set(mixed_obs)
    q2_collateral = evaluate_collateral(
        filtered_current["membership"], q2["membership"], affected
    )
    q3_collateral = evaluate_collateral(
        filtered_current["membership"], q3["membership"], affected
    )
    q4_collateral = evaluate_collateral(
        filtered_current["membership"], q4["membership"], affected
    )
    q3_owners = _owner_index(q3["membership"])
    target_owners = q3_owners.get(target_probe, [])
    trigger_owners = q3_owners.get(trigger_obs, [])
    trigger_separated = bool(
        len(target_owners) == 1
        and len(trigger_owners) == 1
        and target_owners[0] != trigger_owners[0]
    )

    print("[9/9] final source check and report", flush=True)
    source_unchanged = context.source_hashes_before == provenance.source_hashes()
    q2_distinct = bool(
        q2_summary["gt_partition"]["best_entities_are_distinct"]
    )
    q3_distinct = bool(
        q3_summary["gt_partition"]["best_entities_are_distinct"]
    )
    q4_distinct = bool(
        q4_summary["gt_partition"]["best_entities_are_distinct"]
    )
    q3_continuation_complete = bool(
        q3_continuation["later_pure_gt19_observation_count"] > 0
        and q3_continuation["later_pure_gt19_same_owner_count"]
        == q3_continuation["later_pure_gt19_observation_count"]
    )
    q4_continuation_complete = bool(
        q4_continuation["later_pure_gt19_observation_count"] > 0
        and q4_continuation["later_pure_gt19_same_owner_count"]
        == q4_continuation["later_pure_gt19_observation_count"]
    )
    execution_pass = bool(
        snapshot.validation["pass"]
        and q2_invariants["pass"]
        and q3_invariants["pass"]
        and q4_invariants["pass"]
        and q2_collateral["changed_outside_observation_count"] == 0
        and q3_collateral["changed_outside_observation_count"] == 0
        and q4_collateral["changed_outside_observation_count"] == 0
        and source_unchanged
    )
    create_mechanism_success = bool(
        q3_distinct and trigger_separated and q3_continuation_complete
    )
    margin_ablation_success = bool(
        q4_distinct and trigger_separated and q4_continuation_complete
    )
    if q2_distinct:
        outcome = "ORACLE_MIXED_FILTER_ONLY_SUFFICIENT"
    elif create_mechanism_success:
        outcome = "ORACLE_FILTER_PLUS_CLEAN_CREATE_SUFFICIENT"
    elif q3_distinct and margin_ablation_success:
        outcome = (
            "ORACLE_CORE_SEPARATION_WITH_ONE_RESIDUAL;"
            "EXPLORATORY_MARGIN_HYSTERESIS_CLOSES_RESIDUAL"
        )
    else:
        outcome = "ORACLE_FILTER_PLUS_CLEAN_CREATE_INSUFFICIENT"
    result = {
        "schema_version": SCHEMA_VERSION,
        "execution_status": "PASS" if execution_pass else "FAIL",
        "scientific_outcome": outcome,
        "claim_scope": "OFFLINE_ORACLE_CEILING_NOT_ONLINE_METHOD",
        "oracle_selection": {
            "mixed_observation_count": len(mixed_obs),
            "clean_create_trigger_obs_uid": trigger_obs,
            "clean_create_trigger_frame": trigger_frame,
            "gt_use": "PRESELECT_INTERVENTION_AND_POST_REPLAY_EVALUATION",
        },
        "component_policy": component_policy,
        "snapshot_pass": bool(snapshot.validation["pass"]),
        "branches": {
            "B0_NATIVE": native_summary,
            "Q2_FILTER_ONLY": q2_summary,
            "Q3_FILTER_PLUS_CLEAN_CREATE": q3_summary,
            "Q4_FILTER_CREATE_MARGIN_HYSTERESIS": q4_summary,
        },
        "natural_continuation": {
            "Q3_FILTER_PLUS_CLEAN_CREATE": q3_continuation,
            "Q4_FILTER_CREATE_MARGIN_HYSTERESIS": q4_continuation,
        },
        "margin_hysteresis_ablation": {
            "selection": margin_selection,
            "constraint": margin_primitive.as_dict(),
            "uses_gt_for_residual_selection": False,
            "threshold_selected_post_hoc": True,
            "claim_scope": "EXPLORATORY_SINGLE_SCENE_ABLATION",
        },
        "trigger_target_separated": trigger_separated,
        "runtime_invariants": {
            "Q2_FILTER_ONLY": q2_invariants,
            "Q3_FILTER_PLUS_CLEAN_CREATE": q3_invariants,
            "Q4_FILTER_CREATE_MARGIN_HYSTERESIS": q4_invariants,
        },
        "collateral": {
            "Q2_FILTER_ONLY": q2_collateral,
            "Q3_FILTER_PLUS_CLEAN_CREATE": q3_collateral,
            "Q4_FILTER_CREATE_MARGIN_HYSTERESIS": q4_collateral,
        },
        "source_evidence_unchanged": source_unchanged,
        "source_hash_policy": "HASH_ONCE_AT_CONTEXT_BUILD_AND_ONCE_AT_END",
        "total_wall_ms": (time.perf_counter() - started) * 1000.0,
    }
    _atomic_json(output_root / "metrics.json", result)
    report = markdown(result)
    report_path = output_root / "ROOM0_MIXED_FILTER_CLEAN_CREATE_REPLAY_CN.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    if args.beauty_summary:
        beauty_path = args.beauty_summary.resolve()
        beauty_path.parent.mkdir(parents=True, exist_ok=True)
        beauty_path.write_text(report, encoding="utf-8", newline="\n")
    print(
        f"[done] execution={'PASS' if execution_pass else 'FAIL'} "
        f"outcome={outcome} Q3="
        f"{q3_continuation['later_pure_gt19_same_owner_count']}/"
        f"{q3_continuation['later_pure_gt19_observation_count']} Q4="
        f"{q4_continuation['later_pure_gt19_same_owner_count']}/"
        f"{q4_continuation['later_pure_gt19_observation_count']}",
        flush=True,
    )
    return 0 if execution_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
