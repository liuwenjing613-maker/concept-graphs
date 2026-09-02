#!/usr/bin/env python3
"""Run the first label-driven room0 oracle replay on three human-confirmed roots.

This script intentionally keeps four branches distinct:

* B0: immutable native endpoint (the annotated error is left untouched).
* B1: membership-only root edit with every later routing decision frozen.
* B2: the same root constraint with typed dependency-closure replay.
* B3: the same root constraint with a complete post-anchor suffix replay.

B1 is an identity-partition diagnostic, not a geometry-valid map.  B2/B3 execute
the real ConceptGraphs mapper and are checked with the runtime invariants.
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.benchmark.human_error_pilot import (
    HumanSceneContext,
    evaluate_collateral,
    evaluate_endpoint_groups,
)
from conceptgraph.revision.constraints import ReplayMode, SparseRepairConstraint
from conceptgraph.revision.models import DependencyClosure
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.snapshot import AnchorStateBuilder


SCHEMA_VERSION = "human-label-oracle-replay/1.0"
DEFAULT_CASES = (
    "v2_r2_007",
    "room0_large_r1_0169",
    "room0_large_r1_0143",
)


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
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _partition_signature(
    membership: Mapping[str, Iterable[str]],
) -> tuple[tuple[str, ...], ...]:
    groups = {
        tuple(sorted(set(str(item) for item in members or ())))
        for members in membership.values()
    }
    return tuple(sorted(group for group in groups if group))


def _partition_hash(membership: Mapping[str, Iterable[str]]) -> str:
    payload = json.dumps(
        _partition_signature(membership),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _owner_index(
    membership: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for entity_uid, members in membership.items():
        for obs_uid in members or ():
            owners.setdefault(str(obs_uid), []).append(str(entity_uid))
    return owners


def _unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _candidate_version_for_entity(
    association: Mapping[str, Any], entity_uid: str
) -> str:
    pairs = [
        str(version_uid)
        for object_uid, version_uid in zip(
            association.get("object_uids_before") or (),
            association.get("candidate_object_version_uids") or (),
        )
        if str(object_uid) == str(entity_uid)
    ]
    if len(pairs) != 1:
        raise ValueError(
            f"expected one frozen candidate version for {entity_uid}, got {pairs}"
        )
    return pairs[0]


def _compile_case(
    episode: Mapping[str, Any], context: HumanSceneContext
) -> dict[str, Any]:
    provenance = context.provenance
    case_uid = str(episode["case_uid"])
    event_uid = str(episode["event_uid"])
    association = provenance.get_event(event_uid)
    if str(association.get("event_type", "ASSOCIATION_DECISION")) not in {
        "ASSOCIATION_DECISION",
        "",
    }:
        raise ValueError(f"{case_uid}: anchor is not an association event")
    obs_uid = str(association["obs_uid"])
    event_sequence = provenance.sequence(association)
    frame_idx = int(str(association["frame_uid"]).rsplit("_f", 1)[-1])
    original = str(episode["original_action_type"])
    correct = str(episode["correct_action_type"])
    recorded = str(association["decision"])
    expected_recorded = "CREATE_OBJECT" if original == "NEW" else "MERGE_TO_OBJECT"
    if recorded != expected_recorded:
        raise ValueError(
            f"{case_uid}: human original action {original} disagrees with {recorded}"
        )

    human_targets = [str(item) for item in episode.get("human_legal_target_uids") or ()]
    if correct == "ATTACH_EXISTING":
        if len(human_targets) != 1:
            raise ValueError(
                f"{case_uid}: first replay requires one human legal target, got {human_targets}"
            )
        target_entity_uid = human_targets[0]
        target_version_uid = _candidate_version_for_entity(
            association, target_entity_uid
        )
    elif correct == "NEW":
        if human_targets:
            raise ValueError(f"{case_uid}: NEW must not carry a legal target")
        target_entity_uid = str(association.get("target_object_uid") or "")
        target_version_uid = str(
            association.get("target_object_version_before")
            or _candidate_version_for_entity(association, target_entity_uid)
        )
    else:
        raise ValueError(f"{case_uid}: unsupported correct action {correct}")

    target_version = provenance.get_object_version(target_version_uid)
    target_lineage_uid = str(
        target_version.get("lineage_uid") or target_version["object_uid"]
    )
    target_origin_obs_uid = str(
        target_version.get("origin_observation_uid")
        or (target_version.get("member_observation_uids") or [""])[0]
    )
    if not target_origin_obs_uid:
        raise ValueError(f"{case_uid}: target version has no stable observation probe")

    future = episode.get("future_evidence") or {}
    proposal = [str(item) for item in future.get("proposal_view_obs_uids") or ()]
    validation = future.get("validation_view_obs_uid")
    corrected_probes = _unique([obs_uid, *proposal, validation])
    supplemental_suffix_probes: list[str] = []
    if len(corrected_probes) < 3:
        # Some high-confidence annotations have only one independent view inside
        # the short audit window, but a second GT-audited view later in the
        # suffix.  These probes are endpoint evaluation evidence only: they do
        # not alter the sparse constraint, replay scope, or online decisions.
        for row in future.get("independent_views_suffix") or ():
            probe = str(row.get("obs_uid") or "")
            if probe and probe not in corrected_probes:
                corrected_probes.append(probe)
                supplemental_suffix_probes.append(probe)
            if len(corrected_probes) >= 3:
                break
    if len(corrected_probes) < 3:
        raise ValueError(
            f"{case_uid}: requires anchor plus at least two future evidence views"
        )
    for probe in corrected_probes:
        if probe not in provenance.observations:
            raise ValueError(f"{case_uid}: unknown label-derived probe {probe}")

    evidence_refs = _unique([case_uid, event_uid, *proposal, validation])
    if correct == "ATTACH_EXISTING":
        constraint_value = {
            "type": "ASSIGN_OBSERVATION",
            "obs_uid": obs_uid,
            "target_lineage_uid": target_lineage_uid,
            "target_origin_obs_uid": target_origin_obs_uid,
            "target_entity_uid": target_entity_uid,
            "applies_at_event_uid": event_uid,
            "active_from_sequence": event_sequence,
            "active_until_sequence": event_sequence,
            "source": "human_annotation_oracle_experiment1",
            "evidence_refs": evidence_refs,
        }
        evaluation_groups = {
            "same_instance_human_evidence": _unique(
                [target_origin_obs_uid, *corrected_probes]
            )
        }
        desired_relation = "SAME_OWNER"
    else:
        created_identity_uid = f"human-oracle-identity:{case_uid}"
        created_entity_uid = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"human-oracle-entity:{case_uid}")
        )
        constraint_value = {
            "type": "CREATE_INSTANCE",
            "obs_uid": obs_uid,
            "created_lineage_uid": created_identity_uid,
            "created_identity_uid": created_identity_uid,
            "created_entity_uid": created_entity_uid,
            "separate_from_identity_uids": [target_lineage_uid],
            "applies_at_event_uid": event_uid,
            "active_from_sequence": event_sequence,
            "active_until_sequence": event_sequence,
            "source": "human_annotation_oracle_experiment1",
            "evidence_refs": evidence_refs,
        }
        evaluation_groups = {
            "new_instance_human_evidence": corrected_probes,
            "human_confirmed_clean_original_target": [target_origin_obs_uid],
        }
        desired_relation = "DIFFERENT_OWNER"

    primitive = SparseRepairConstraint.from_mapping(constraint_value)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_uid": case_uid,
        "source_batch": episode.get("source_batch"),
        "queue_memberships": episode.get("queue_memberships") or [],
        "causal_role": episode.get("causal_role"),
        "routing_label": episode.get("routing_label"),
        "anchor_event_uid": event_uid,
        "anchor_obs_uid": obs_uid,
        "anchor_event_sequence": event_sequence,
        "anchor_frame": frame_idx,
        "original_action_type": original,
        "correct_action_type": correct,
        "target_entity_uid": target_entity_uid,
        "target_version_uid": target_version_uid,
        "target_lineage_uid": target_lineage_uid,
        "target_origin_obs_uid": target_origin_obs_uid,
        "snapshot_seed_version_uids": [target_version_uid],
        "constraint": primitive.as_dict(),
        "label_derived_evaluation": {
            "oracle_status": "HUMAN_ROUTE_PLUS_OFFLINE_GT_AUDITED_FUTURE_VIEWS",
            "groups": evaluation_groups,
            "desired_owner_relation": desired_relation,
            "corrected_instance_probe_obs_uids": corrected_probes,
            "proposal_view_obs_uids": proposal,
            "validation_view_obs_uid": validation,
            "supplemental_suffix_evaluation_obs_uids": supplemental_suffix_probes,
            "target_probe_obs_uid": target_origin_obs_uid,
        },
    }


def _membership_only_b1(
    native_state: Mapping[str, Any], compiled: Mapping[str, Any]
) -> dict[str, Any]:
    membership = {
        str(entity_uid): list(members or ())
        for entity_uid, members in (native_state.get("membership") or {}).items()
    }
    anchor = str(compiled["anchor_obs_uid"])
    for entity_uid in list(membership):
        membership[entity_uid] = [item for item in membership[entity_uid] if item != anchor]
        if not membership[entity_uid]:
            del membership[entity_uid]
    if str(compiled["correct_action_type"]) == "ATTACH_EXISTING":
        target_probe = str(compiled["target_origin_obs_uid"])
        owners = _owner_index(membership).get(target_probe, [])
        if len(owners) != 1:
            raise ValueError(
                f"{compiled['case_uid']}: B1 target probe has owners {owners}"
            )
        membership[owners[0]].append(anchor)
        membership[owners[0]] = sorted(set(membership[owners[0]]))
    else:
        membership[f"b1-membership-only:{compiled['case_uid']}"] = [anchor]
    return {
        "schema_version": SCHEMA_VERSION,
        "branch": "B1_ROOT_MEMBERSHIP_ONLY_LATER_ROUTING_FROZEN",
        "scope": "MEMBERSHIP_PARTITION_ONLY_NO_GEOMETRY_RECOMPUTATION",
        "status": "COMPLETED",
        "membership": membership,
        "objects": [],
        "edges": [],
        "runtime_ms": 0.0,
        "replayed_observations": 0,
        "state_hash": _partition_hash(membership),
        "geometry_valid": False,
        "formal_map_state": False,
    }


def _full_suffix_closure(
    context: HumanSceneContext,
    *,
    anchor_event_sequence: int,
    snapshot_watermark_event_sequence: int,
) -> DependencyClosure:
    provenance = context.provenance
    events = [
        row
        for row in provenance.events.values()
        if provenance.sequence(row) > int(snapshot_watermark_event_sequence)
    ]
    event_uids = [str(row["event_uid"]) for row in events]
    obs_uids = [
        str(row["obs_uid"])
        for row in provenance.association_rows
        if provenance.sequence(row) > int(snapshot_watermark_event_sequence)
    ]
    version_uids: list[str] = []
    entity_uids: set[str] = set()
    for row in provenance.object_version_rows:
        trigger = row.get("trigger_event_uid")
        if trigger in provenance.events and provenance.sequence(
            provenance.get_event(str(trigger))
        ) > int(snapshot_watermark_event_sequence):
            version_uids.append(str(row["object_version_uid"]))
        entity_uids.add(str(row["object_uid"]))
        if row.get("lineage_uid"):
            entity_uids.add(str(row["lineage_uid"]))
    edge_uids = [
        str(row.get("edge_uid") or row["event_uid"])
        for row in events
        if str(row.get("event_type", "")).startswith("EDGE_")
    ]
    return DependencyClosure.build(
        event_uids=event_uids,
        version_uids=version_uids,
        entity_uids=entity_uids,
        obs_uids=obs_uids,
        edge_uids=edge_uids,
        start_sequence=int(anchor_event_sequence),
        end_sequence=max(
            (provenance.sequence(row) for row in events),
            default=int(anchor_event_sequence),
        ),
    )


def _affected_native_observations(
    native_membership: Mapping[str, Iterable[str]],
    evaluation_groups: Mapping[str, Iterable[str]],
) -> set[str]:
    owners = _owner_index(native_membership)
    entities = {
        entity_uid
        for members in evaluation_groups.values()
        for obs_uid in members
        for entity_uid in owners.get(str(obs_uid), ())
    }
    affected = {
        str(obs_uid)
        for entity_uid in entities
        for obs_uid in native_membership.get(entity_uid, ())
    }
    affected.update(
        str(obs_uid) for members in evaluation_groups.values() for obs_uid in members
    )
    return affected


def _anchor_action_audit(
    branch_name: str,
    state: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    expected = (
        "FORCE_TARGET"
        if str(compiled["correct_action_type"]) == "ATTACH_EXISTING"
        else "FORCE_CREATE"
    )
    if branch_name == "B1":
        return {
            "correct": True,
            "expected_constraint_action": expected,
            "observed_constraint_actions": ["DIRECT_MEMBERSHIP_EDIT"],
            "basis": "B1_DEFINITION",
        }
    rows = [
        row
        for row in state.get("decision_trace") or ()
        if str(row.get("event_uid")) == str(compiled["anchor_event_uid"])
    ]
    actions = [str((row.get("constraint") or {}).get("action")) for row in rows]
    return {
        "correct": expected in actions,
        "expected_constraint_action": expected,
        "observed_constraint_actions": actions,
        "decision_count": len(rows),
        "basis": "EXECUTED_DECISION_TRACE",
    }


def _branch_metrics(
    *,
    branch_name: str,
    state: Mapping[str, Any],
    compiled: Mapping[str, Any],
    native_state: Mapping[str, Any],
    affected_native: set[str],
    verifier: InvariantVerifier,
    primitive: SparseRepairConstraint,
    known_observations: Iterable[str],
) -> dict[str, Any]:
    evaluation = compiled["label_derived_evaluation"]
    endpoint = evaluate_endpoint_groups(
        state.get("membership") or {},
        evaluation["groups"],
        str(evaluation["desired_owner_relation"]),
        probes=evaluation["corrected_instance_probe_obs_uids"]
        + [evaluation["target_probe_obs_uid"]],
    )
    collateral = evaluate_collateral(
        native_state.get("membership") or {},
        state.get("membership") or {},
        affected_native,
    )
    runtime_invariants: dict[str, Any]
    if branch_name == "B1":
        runtime_invariants = {
            "pass": None,
            "status": "NOT_APPLICABLE_MEMBERSHIP_ONLY_BRANCH",
        }
    else:
        runtime_invariants = verifier.verify(
            state=state,
            constraints=[primitive] if branch_name in {"B2", "B3"} else (),
            known_observation_uids=known_observations,
        )
    timing = state.get("timing") or {}
    return {
        "branch": branch_name,
        "partition_hash": _partition_hash(state.get("membership") or {}),
        "active_object_count": len(state.get("membership") or {}),
        "root_action": _anchor_action_audit(branch_name, state, compiled),
        "label_derived_endpoint": endpoint,
        "endpoint_correct": bool(endpoint["correct"]),
        "collateral": collateral,
        "runtime_invariants": runtime_invariants,
        "replayed_observation_count": int(state.get("replayed_observations", 0)),
        "runtime_ms": float(state.get("runtime_ms", 0.0)),
        "closure_initial_observation_count": state.get(
            "closure_initial_observation_count"
        ),
        "closure_effective_observation_count": state.get(
            "closure_effective_observation_count"
        ),
        "closure_expanded_observation_count": state.get(
            "closure_expanded_observation_count"
        ),
        "overlay_diagnostics": state.get("overlay_diagnostics"),
        "suffix_total_wall_ms": timing.get("suffix_total_wall_ms"),
        "geometry_valid": branch_name != "B1",
    }


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _markdown_report(aggregate: Mapping[str, Any]) -> str:
    lines = [
        "# Room0 人工标注驱动的最小 Oracle 回放",
        "",
        f"- 状态：**{aggregate['status']}**",
        f"- 人工根错误：{aggregate['case_count']} 例",
        f"- 源证据仅在结束时校验一次：{aggregate['source_evidence_unchanged']}",
        "- B1 只是成员归属对照，不是几何有效的地图。B2/B3 才运行真实 mapper。",
        "",
        "| 病例 | 人工类型 | B0 endpoint | B1 endpoint | B2 endpoint | B3 endpoint | B2/B3 invariant |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate["cases"]:
        branches = row["branches"]
        lines.append(
            "| {case} | {kind} | {b0} | {b1} | {b2} | {b3} | {inv2}/{inv3} |".format(
                case=row["case_uid"],
                kind=row["routing_label"],
                b0=branches["B0"]["endpoint_correct"],
                b1=branches["B1"]["endpoint_correct"],
                b2=branches["B2"]["endpoint_correct"],
                b3=branches["B3"]["endpoint_correct"],
                inv2=branches["B2"]["runtime_invariants"]["pass"],
                inv3=branches["B3"]["runtime_invariants"]["pass"],
            )
        )
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "- 约束由人工路由标签编译；未用私有 GT 替代人工目标。",
            "- 未来视角是标注后的 oracle 证据上限，不代表已有自动发现器。",
            "- B2 使用类型化前向依赖闭包；B3 使用锚点后全后缀，两者不混称。",
            "- 如果 B2 与 B3 同时失败，说明当前约束传播机制不足，不能归因为闭包裁剪。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--beauty-summary", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--intake-only",
        action="store_true",
        help="Compile human constraints and strict snapshots, but do not replay.",
    )
    parser.add_argument(
        "--disable-create-association-boundary",
        action="store_true",
        help=(
            "Ablate the per-observation CREATE_INSTANCE association boundary while "
            "retaining the postprocess merge boundary. The default replay policy is "
            "unchanged when this flag is absent."
        ),
    )
    args = parser.parse_args()

    component_policy = {
        "positive_lineage_redirect": True,
        "create_association_boundary": not bool(
            args.disable_create_association_boundary
        ),
        "create_postprocess_boundary": True,
    }
    if args.disable_create_association_boundary:
        print(
            "[policy] create_association_boundary=False; "
            "create_postprocess_boundary=True",
            flush=True,
        )

    selected = tuple(args.case or DEFAULT_CASES)
    episodes = {
        str(row["case_uid"]): row for row in _read_jsonl(args.episodes.resolve())
    }
    missing = sorted(set(selected) - set(episodes))
    if missing:
        raise ValueError(f"missing selected human episodes: {missing}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[start] build room0 context: {args.room_run.resolve()}", flush=True)
    run_started = time.perf_counter()
    context = HumanSceneContext.build("room0", args.room_run.resolve())
    verifier = InvariantVerifier()
    native_state = copy.deepcopy(context.native_state)
    native_partition_hash = _partition_hash(native_state["membership"])
    print(
        f"[context] native objects={len(native_state['membership'])} "
        f"observations={sum(len(v) for v in native_state['membership'].values())}",
        flush=True,
    )

    compiled_rows = [_compile_case(episodes[uid], context) for uid in selected]
    compiled_rows.sort(key=lambda row: (int(row["anchor_frame"]), row["case_uid"]))
    _atomic_json(output_root / "compiled_human_oracle_cases.json", compiled_rows)
    print(
        "[intake] "
        + ", ".join(
            f"{row['case_uid']}@f{row['anchor_frame']}:{row['constraint']['type']}"
            for row in compiled_rows
        ),
        flush=True,
    )

    case_rows = []
    for index, compiled in enumerate(compiled_rows, 1):
        case_uid = str(compiled["case_uid"])
        case_root = output_root / case_uid
        case_root.mkdir(parents=True, exist_ok=True)
        primitive = SparseRepairConstraint.from_mapping(compiled["constraint"])
        print(
            f"[case {index}/{len(compiled_rows)}] {case_uid}: build strict pre-anchor snapshot",
            flush=True,
        )
        prefix_state, prefix_objects = context.prefix_cache.prefix_before(
            int(compiled["anchor_frame"])
        )
        snapshot = AnchorStateBuilder(
            context.provenance, context.engine
        ).build_pre_anchor_state(
            str(compiled["anchor_event_uid"]),
            compiled["snapshot_seed_version_uids"],
            strict=True,
            prefix_state=prefix_state,
            prefix_objects=prefix_objects,
        )
        closure = context.dependency_graph.forward_closure(
            anchor_event_uid=str(compiled["anchor_event_uid"]),
            seed_version_uids=compiled["snapshot_seed_version_uids"],
        )
        label_probes = set(
            compiled["label_derived_evaluation"][
                "corrected_instance_probe_obs_uids"
            ]
        )
        closure_probe_coverage = sorted(label_probes & set(closure.obs_uids))
        intake = {
            "compiled_case": compiled,
            "snapshot": snapshot.as_dict(),
            "dependency_closure": closure.as_dict(),
            "label_probe_count": len(label_probes),
            "label_probes_in_dependency_closure": closure_probe_coverage,
            "label_probe_dependency_coverage": len(closure_probe_coverage)
            / len(label_probes),
        }
        _atomic_json(case_root / "intake.json", intake)
        print(
            f"[case {index}/{len(compiled_rows)}] snapshot_pass={snapshot.validation['pass']} "
            f"closure_obs={len(closure.obs_uids)} "
            f"label_probe_coverage={len(closure_probe_coverage)}/{len(label_probes)}",
            flush=True,
        )
        if args.intake_only:
            case_rows.append(
                {
                    "case_uid": case_uid,
                    "routing_label": compiled["routing_label"],
                    "status": "INTAKE_PASS",
                    "snapshot_pass": snapshot.validation["pass"],
                    "closure_observation_count": len(closure.obs_uids),
                    "label_probe_dependency_coverage": len(closure_probe_coverage)
                    / len(label_probes),
                }
            )
            continue

        evaluation_groups = compiled["label_derived_evaluation"]["groups"]
        affected_native = _affected_native_observations(
            native_state["membership"], evaluation_groups
        )
        b0 = copy.deepcopy(native_state)
        b0["branch"] = "B0_NATIVE_FROZEN_ENDPOINT"
        b1 = _membership_only_b1(native_state, compiled)

        common = {
            "snapshot_objects": snapshot.objects,
            "snapshot_runtime_ms": snapshot.state["runtime_ms"],
            "snapshot_timing": snapshot.state.get("timing"),
            "anchor_frame": snapshot.anchor_frame,
            "snapshot_watermark_event_sequence": snapshot.watermark_event_sequence,
            "current_state": native_state,
        }
        print(f"[case {index}/{len(compiled_rows)}] run B0 replay control", flush=True)
        b0r = context.engine.replay_suffix_from_snapshot(
            mode=ReplayMode.NATURAL_REPLAY,
            closure=closure,
            **common,
        )
        b0r["branch"] = "B0R_NO_CONSTRAINT_DEPENDENCY_REPLAY_CONTROL"
        print(
            f"[case {index}/{len(compiled_rows)}] B0R replayed={b0r['replayed_observations']} "
            f"partition_parity={_partition_hash(b0r['membership']) == native_partition_hash}",
            flush=True,
        )

        print(f"[case {index}/{len(compiled_rows)}] run B2 dependency closure", flush=True)
        b2 = context.engine.replay_local_from_snapshot(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            closure=closure,
            constraints=[primitive],
            component_policy=component_policy,
            **common,
        )
        b2["branch"] = "B2_TYPED_DEPENDENCY_CLOSURE_REPLAY"
        print(
            f"[case {index}/{len(compiled_rows)}] B2 replayed={b2['replayed_observations']} "
            f"effective_scope={b2.get('closure_effective_observation_count')}",
            flush=True,
        )

        full_closure = _full_suffix_closure(
            context,
            anchor_event_sequence=int(compiled["anchor_event_sequence"]),
            snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        )
        print(
            f"[case {index}/{len(compiled_rows)}] run B3 full suffix "
            f"scope_obs={len(full_closure.obs_uids)}",
            flush=True,
        )
        b3 = context.engine.replay_local_from_snapshot(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            closure=full_closure,
            constraints=[primitive],
            component_policy=component_policy,
            **common,
        )
        b3["branch"] = "B3_FULL_POST_ANCHOR_SUFFIX_REPLAY"

        branches = {"B0": b0, "B1": b1, "B0R": b0r, "B2": b2, "B3": b3}
        branch_metrics = {
            name: _branch_metrics(
                branch_name=name,
                state=state,
                compiled=compiled,
                native_state=native_state,
                affected_native=affected_native,
                verifier=verifier,
                primitive=primitive,
                known_observations=context.provenance.observations,
            )
            for name, state in branches.items()
        }
        b0r_parity = (
            branch_metrics["B0R"]["partition_hash"]
            == branch_metrics["B0"]["partition_hash"]
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "case_uid": case_uid,
            "routing_label": compiled["routing_label"],
            "correct_action_type": compiled["correct_action_type"],
            "snapshot_pass": bool(snapshot.validation["pass"]),
            "dependency_closure_observation_count": len(closure.obs_uids),
            "full_suffix_observation_count": len(full_closure.obs_uids),
            "label_probe_dependency_coverage": len(closure_probe_coverage)
            / len(label_probes),
            "b0r_exact_partition_parity": b0r_parity,
            "branches": branch_metrics,
            "interpretation": {
                "b1_is_membership_only": True,
                "b2_is_typed_dependency_closure": True,
                "b3_is_full_post_anchor_suffix": True,
                "future_views_are_oracle_evaluation_evidence": True,
                "component_policy": component_policy,
                "association_boundary_ablation": bool(
                    args.disable_create_association_boundary
                ),
            },
        }
        _atomic_json(case_root / "metrics.json", result)
        _atomic_json(case_root / "full_suffix_closure.json", full_closure.as_dict())
        for name, state in branches.items():
            if name == "B0":
                continue
            _atomic_gzip_json(case_root / "branches" / f"{name}.json.gz", state)
        case_rows.append(result)
        print(
            f"[case {index}/{len(compiled_rows)}] endpoints "
            + " ".join(
                f"{name}={branch_metrics[name]['endpoint_correct']}"
                for name in ("B0", "B1", "B2", "B3")
            )
            + f" invariants B2={branch_metrics['B2']['runtime_invariants']['pass']}"
            + f" B3={branch_metrics['B3']['runtime_invariants']['pass']}",
            flush=True,
        )
        del b0, b1, b0r, b2, b3, branches
        _release_cuda()

    source_hashes_after = context.provenance.source_hashes()
    source_unchanged = context.source_hashes_before == source_hashes_after
    if args.intake_only:
        status = "INTAKE_PASS" if all(row["status"] == "INTAKE_PASS" for row in case_rows) else "FAIL"
    else:
        controls_pass = all(row["b0r_exact_partition_parity"] for row in case_rows)
        invariants_pass = all(
            row["branches"][name]["runtime_invariants"]["pass"]
            for row in case_rows
            for name in ("B2", "B3")
        )
        status = "PASS" if controls_pass and invariants_pass and source_unchanged else "FAIL"
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "intake_only": bool(args.intake_only),
        "case_count": len(case_rows),
        "source_evidence_unchanged": source_unchanged,
        "source_hash_policy": "HASH_ONCE_AT_CONTEXT_BUILD_AND_ONCE_AFTER_ALL_CASES",
        "native_partition_hash": native_partition_hash,
        "cases": case_rows,
        "run_total_wall_ms": (time.perf_counter() - run_started) * 1000.0,
    }
    _atomic_json(output_root / "aggregate_metrics.json", aggregate)
    if not args.intake_only:
        report = _markdown_report(aggregate)
        report_path = output_root / "ROOM0_HUMAN_ORACLE_MINIMAL_REPLAY_CN.md"
        report_path.write_text(report, encoding="utf-8", newline="\n")
        if args.beauty_summary:
            beauty_path = args.beauty_summary.resolve()
            beauty_path.parent.mkdir(parents=True, exist_ok=True)
            beauty_path.write_text(report, encoding="utf-8", newline="\n")
    print(
        f"[done] status={status} cases={len(case_rows)} "
        f"source_unchanged={source_unchanged} output={output_root}",
        flush=True,
    )
    return 0 if status in {"PASS", "INTAKE_PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
