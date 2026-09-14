from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..constraints import ReplayMode, SparseRepairConstraint
from ..dependency_graph import TypedDependencyGraph
from ..evaluate import edge_metrics, evaluate_state
from ..index import ProvenanceIndex
from ..relations import AliDevBaselineRelationBackend, load_baseline_frame_records
from ..replay import CounterfactualReplayEngine
from ..runtime_verify import InvariantVerifier
from ..snapshot import AnchorStateBuilder, IncrementalPrefixCache
from ..sparse_replay import (
    SparseCounterfactualReplayEngine,
    SparseReplayDeferred,
    SparseReplayError,
)
from .cases import compile_sparse_constraints


MEMBER_F1_ATOL = 1e-12
BBOX_IOU_ATOL = 1e-6


@dataclass
class SceneExperimentContext:
    provenance: ProvenanceIndex
    engine: SparseCounterfactualReplayEngine
    reference_state: dict[str, Any]
    dependency_graph: TypedDependencyGraph
    prefix_cache: IncrementalPrefixCache

    @classmethod
    def build(cls, base_run: str | Path) -> "SceneExperimentContext":
        provenance = ProvenanceIndex(base_run)
        engine = SparseCounterfactualReplayEngine(provenance)
        return cls(
            provenance=provenance,
            engine=engine,
            reference_state=CounterfactualReplayEngine(provenance).clean_state(),
            dependency_graph=TypedDependencyGraph(provenance),
            prefix_cache=IncrementalPrefixCache(engine),
        )


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    temporary.replace(destination)


def _seed_versions(case: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(value)
            for value in (
                case.get("clean_target_object_version_uid"),
                case.get("target_object_version_uid"),
            )
            if value
        }
    )


def _affected_observations(case: Mapping[str, Any]) -> set[str]:
    return {
        str(obs_uid)
        for members in (case.get("affected_clean_groups") or {}).values()
        for obs_uid in members
    }


def _recovery(corrupted_f1: float, repaired_f1: float) -> float | None:
    denominator = 1.0 - float(corrupted_f1)
    if denominator <= 1e-12:
        return None
    return 1.0 - (1.0 - float(repaired_f1)) / denominator


def aligned_relation_metrics(
    reference_state: Mapping[str, Any], candidate_state: Mapping[str, Any]
) -> dict[str, Any]:
    reference_by_partition: dict[tuple[str, ...], list[str]] = {}
    for entity_uid, members in (reference_state.get("membership") or {}).items():
        partition = tuple(sorted(str(obs_uid) for obs_uid in members))
        reference_by_partition.setdefault(partition, []).append(str(entity_uid))
    entity_alignment: dict[str, str] = {}
    ambiguous = []
    unaligned = []
    for entity_uid, members in (candidate_state.get("membership") or {}).items():
        candidate_uid = str(entity_uid)
        partition = tuple(sorted(str(obs_uid) for obs_uid in members))
        targets = reference_by_partition.get(partition, [])
        if len(targets) == 1:
            entity_alignment[candidate_uid] = targets[0]
        elif len(targets) > 1:
            ambiguous.append(candidate_uid)
        else:
            unaligned.append(candidate_uid)

    # A candidate entity with a different member partition must not inherit a
    # matching clean UUID by accident.  Give it an evaluator-only namespace so an
    # incident edge remains observably different.  Exact partitions still align
    # independent executions whose runtime UUIDs legitimately differ.
    def aligned_uid(value: Any) -> str:
        uid = str(value)
        return entity_alignment.get(uid, f"__unaligned_candidate__:{uid}")

    aligned = dict(candidate_state)
    aligned["membership"] = reference_state.get("membership") or {}
    aligned["edges"] = [
        {
            **edge,
            "source_entity_uid": aligned_uid(edge["source_entity_uid"]),
            "target_entity_uid": aligned_uid(edge["target_entity_uid"]),
        }
        for edge in candidate_state.get("edges") or ()
    ]
    result = edge_metrics(reference_state, aligned)
    raw = edge_metrics(reference_state, candidate_state)
    result["alignment_basis"] = "EXACT_MEMBER_PARTITION"
    result["entity_alignment"] = entity_alignment
    result["ambiguous_entity_alignments"] = ambiguous
    result["unaligned_candidate_entities"] = sorted(unaligned)
    result["exact_partition_alignment_count"] = len(entity_alignment)
    result["raw_entity_id_edge_state_match"] = raw["edge_state_match"]
    result["raw_entity_id_edge_set_f1"] = raw["edge_set_f1_to_clean"]
    return result


def _method_equivalent(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        math.isclose(
            float(first["membership"]["member_f1"]),
            float(second["membership"]["member_f1"]),
            rel_tol=0.0,
            abs_tol=MEMBER_F1_ATOL,
        )
        and math.isclose(
            float(first["geometry"]["bbox_iou_to_clean"]),
            float(second["geometry"]["bbox_iou_to_clean"]),
            rel_tol=0.0,
            abs_tol=BBOX_IOU_ATOL,
        )
        and bool(first["relation"]["edge_state_match"])
        == bool(second["relation"]["edge_state_match"])
    )


def classify_repair_outcome(
    *,
    corrupted_method: Mapping[str, Any],
    persistent_method: Mapping[str, Any],
    verification_pass: bool,
) -> dict[str, Any]:
    corrupted_f1 = float(corrupted_method["membership"]["member_f1"])
    persistent_f1 = float(persistent_method["membership"]["member_f1"])
    corrupted_iou = float(corrupted_method["geometry"]["bbox_iou_to_clean"])
    persistent_iou = float(persistent_method["geometry"]["bbox_iou_to_clean"])
    corrupted_relation_exact = bool(corrupted_method["relation"]["edge_state_match"])
    persistent_relation_exact = bool(
        persistent_method["relation"]["edge_state_match"]
    )
    damage = {
        "membership": corrupted_f1 < 1.0 - MEMBER_F1_ATOL,
        "geometry": corrupted_iou < 1.0 - BBOX_IOU_ATOL,
        "relation": not corrupted_relation_exact,
    }
    improvements = {
        "membership": persistent_f1 > corrupted_f1 + MEMBER_F1_ATOL,
        "geometry": persistent_iou > corrupted_iou + BBOX_IOU_ATOL,
        "relation": damage["relation"] and persistent_relation_exact,
    }
    non_worse = (
        persistent_f1 >= corrupted_f1 - MEMBER_F1_ATOL
        and persistent_iou >= corrupted_iou - BBOX_IOU_ATOL
        and (not corrupted_relation_exact or persistent_relation_exact)
    )
    global_membership = persistent_method.get("membership_global") or {}
    collateral_safe = bool(
        global_membership.get(
            "partition_exact",
            float(global_membership.get("member_f1", 1.0)) >= 1.0 - MEMBER_F1_ATOL,
        )
    )
    return {
        "damage_dimensions": damage,
        "improved_dimensions": improvements,
        "non_worse": non_worse,
        "collateral_safe": collateral_safe,
        "pass": bool(
            verification_pass
            and any(damage.values())
            and any(improvements.values())
            and non_worse
            and collateral_safe
        ),
        "thresholds": {
            "member_f1_atol": MEMBER_F1_ATOL,
            "bbox_iou_atol": BBOX_IOU_ATOL,
        },
    }


def _attach_relations(
    provenance: ProvenanceIndex,
    states: Mapping[str, dict[str, Any]],
    *,
    edge_stream_root: str | Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_started = time.perf_counter()
    for name, state in states.items():
        branch_started = time.perf_counter()
        objects, records = load_baseline_frame_records(
            provenance,
            state["membership"],
            edge_stream_root=edge_stream_root,
        )
        rebuilt = AliDevBaselineRelationBackend().rebuild(
            objects=objects,
            frame_records=records,
        )
        state["edges"] = rebuilt["output_edges"]
        branch_runtime_ms = (time.perf_counter() - branch_started) * 1000.0
        state_timing = dict(state.get("timing") or {})
        state_timing["relation_rebuild_wall_ms"] = branch_runtime_ms
        state["timing"] = state_timing
        rebuilt["runtime_ms"] = branch_runtime_ms
        result[name] = rebuilt
    return {
        "schema_version": "1.0.0",
        "strategy": "GLOBAL_BASELINE_EDGE_REFERENCE",
        "edge_stream_root": str(edge_stream_root) if edge_stream_root else None,
        "all_structural_validations_pass": all(
            value["validation"]["pass"] for value in result.values()
        ),
        "total_wall_ms": (time.perf_counter() - all_started) * 1000.0,
        "branches": result,
    }


def _failure_taxonomy(
    *,
    corrupted_metrics: Mapping[str, Any],
    historical_no_repair_metrics: Mapping[str, Any],
    natural_recompute_metrics: Mapping[str, Any],
    anchor_metrics: Mapping[str, Any],
    persistent_metrics: Mapping[str, Any],
    anchor_state: Mapping[str, Any],
    persistent_state: Mapping[str, Any],
    verification: Mapping[str, Any],
    global_metrics: Mapping[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    outcome = classify_repair_outcome(
        corrupted_method=corrupted_metrics,
        persistent_method=persistent_metrics,
        verification_pass=bool(verification.get("pass", False)),
    )
    any_damage = any(outcome["damage_dimensions"].values())
    any_improvement = any(outcome["improved_dimensions"].values())
    if not any_damage:
        failures.append("CORRUPTION_SELF_HEALED_NO_FINAL_EFFECT")
    elif not any_improvement:
        failures.append("CONSTRAINT_INSUFFICIENT")
    if float(anchor_metrics["membership"]["member_f1"]) < 0.999999:
        failures.append("NATURAL_REPROPAGATION_FAILURE")
    if not (persistent_state.get("overlay_diagnostics") or {}).get("overlay_pass", False):
        failures.append("CLOSURE_TOO_SMALL")
    if int(persistent_state.get("closure_expansion_count", 0)):
        failures.append("CLOSURE_EXPANDED")
    if not verification.get("pass", False):
        failures.append("RUNTIME_INVARIANT_FAILURE")
    if not outcome["collateral_safe"]:
        failures.append("COLLATERAL_DAMAGE")
    if global_metrics is not None:
        local_f1 = float(persistent_metrics["membership"]["member_f1"])
        global_f1 = float(global_metrics["membership"]["member_f1"])
        if abs(local_f1 - global_f1) > 1e-12:
            failures.append("POSTPROCESS_DIVERGENCE")
    if (
        persistent_state.get("constraint_hit_count", 0) > 0
        and persistent_state.get("constraint_override_count", 0) == 0
        and _method_equivalent(historical_no_repair_metrics, persistent_metrics)
    ):
        failures.append("CONSTRAINT_NON_CAUSAL_ABLATION_EQUIVALENT")
    if _method_equivalent(natural_recompute_metrics, persistent_metrics):
        failures.append("NATURAL_RECOMPUTE_BASELINE_EQUIVALENT")
    return sorted(set(failures))


def run_case(
    *,
    base_run: str | Path,
    output_root: str | Path,
    case: Mapping[str, Any],
    edge_stream_root: str | Path | None = None,
    run_global_sparse: bool = False,
    run_global_corruption: bool = False,
    context: SceneExperimentContext | None = None,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    context = context or SceneExperimentContext.build(base_run)
    provenance = context.provenance
    source_hashes_before = provenance.source_hashes()
    case_root = Path(output_root) / str(case["case_uid"])
    case_root.mkdir(parents=True, exist_ok=True)
    write_json(case_root / "case.json", case)
    write_json(
        case_root / "incident.json",
        {
            "case_uid": case["case_uid"],
            "failure_type": case["failure_type"],
            "suspect_observation_uid": case["obs_uid"],
            "anchor_ground_truth_event_uid": case["anchor_association_event_uid"],
            "benchmark_only": True,
        },
    )
    constraints = compile_sparse_constraints(case, provenance)
    write_json(case_root / "constraint.json", [item.as_dict() for item in constraints])
    seeds = _seed_versions(case)
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=str(case["anchor_association_event_uid"]),
        seed_version_uids=seeds,
    )
    write_json(case_root / "dependency.json", closure.as_dict())

    engine = context.engine
    try:
        prefix_state, prefix_objects = context.prefix_cache.prefix_before(
            int(case["frame_idx"])
        )
        snapshot = AnchorStateBuilder(provenance, engine).build_pre_anchor_state(
            str(case["anchor_association_event_uid"]),
            seeds,
            strict=True,
            prefix_state=prefix_state,
            prefix_objects=prefix_objects,
        )
    except (SparseReplayError, SparseReplayDeferred) as exc:
        failure = {
            "schema_version": "1.0.0",
            "case_uid": case["case_uid"],
            "failure_type": case["failure_type"],
            "status": "FAILED",
            "failure_taxonomy": ["PRE_ANCHOR_RECONSTRUCTION_MISMATCH"],
            "error": str(exc),
            "pass": False,
        }
        write_json(case_root / "benchmark_metrics.json", failure)
        return failure
    write_json(case_root / "pre_anchor_snapshot.json", snapshot.as_dict())

    reference_state = copy.deepcopy(context.reference_state)
    corrupted_state = engine.replay_suffix_from_snapshot(
        mode=ReplayMode.TEMPORAL_CORRUPTION,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        current_state=reference_state,
        corruption_plan=case["corruption_plan"],
    )
    natural_recompute_state = engine.replay_suffix_from_snapshot(
        mode=ReplayMode.NATURAL_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        current_state=corrupted_state,
    )
    historical_no_repair_state = copy.deepcopy(corrupted_state)
    historical_no_repair_state["branch"] = "historical_anchor_no_repair"
    historical_no_repair_state["scope"] = "historical_anchor_suffix_no_repair"
    anchor_state = engine.replay_local_from_snapshot(
        mode=ReplayMode.ANCHOR_ONLY_REPAIR,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        constraints=constraints,
        current_state=corrupted_state,
        snapshot_timing=snapshot.state.get("timing"),
        historical_anchor_plan=case["corruption_plan"],
    )
    persistent_state = engine.replay_local_from_snapshot(
        mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        constraints=constraints,
        current_state=corrupted_state,
        snapshot_timing=snapshot.state.get("timing"),
        historical_anchor_plan=case["corruption_plan"],
    )

    global_sparse_state = None
    if run_global_sparse:
        global_sparse_state = engine.replay_global(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=constraints,
            historical_anchor_plan=case["corruption_plan"],
        )
    global_corrupted_state = None
    if run_global_corruption:
        global_corrupted_state = engine.replay_global(
            mode=ReplayMode.TEMPORAL_CORRUPTION,
            corruption_plan=case["corruption_plan"],
        )

    states = {
        "reference": reference_state,
        "temporal_corrupted_local": corrupted_state,
        "historical_anchor_no_repair": historical_no_repair_state,
        "natural_recompute_ablation": natural_recompute_state,
        "anchor_only_local": anchor_state,
        "persistent_sparse_local": persistent_state,
    }
    if global_sparse_state is not None:
        states["persistent_sparse_global"] = global_sparse_state
    if global_corrupted_state is not None:
        states["temporal_corrupted_global"] = global_corrupted_state
    relation = _attach_relations(
        provenance, states, edge_stream_root=edge_stream_root
    )
    write_json(case_root / "relation_rebuild.json", relation)

    affected = _affected_observations(case)
    method_metrics = {
        name: evaluate_state(
            reference_state,
            state,
            affected_observations=affected,
        )
        for name, state in states.items()
    }
    for name, state in states.items():
        method_metrics[name]["relation"] = aligned_relation_metrics(
            reference_state, state
        )
    verifier = InvariantVerifier()
    anchor_verification_started = time.perf_counter()
    anchor_verification = verifier.verify(
        state=anchor_state,
        constraints=constraints,
        source_hashes_before=source_hashes_before,
        source_hashes_after=provenance.source_hashes(),
        known_observation_uids=provenance.observations,
    )
    anchor_verification_ms = (
        time.perf_counter() - anchor_verification_started
    ) * 1000.0
    anchor_verification["runtime_ms"] = anchor_verification_ms
    anchor_state.setdefault("timing", {})[
        "runtime_invariant_verification_wall_ms"
    ] = anchor_verification_ms
    persistent_verification_started = time.perf_counter()
    persistent_verification = verifier.verify(
        state=persistent_state,
        constraints=constraints,
        source_hashes_before=source_hashes_before,
        source_hashes_after=provenance.source_hashes(),
        known_observation_uids=provenance.observations,
    )
    persistent_verification_ms = (
        time.perf_counter() - persistent_verification_started
    ) * 1000.0
    persistent_verification["runtime_ms"] = persistent_verification_ms
    persistent_state.setdefault("timing", {})[
        "runtime_invariant_verification_wall_ms"
    ] = persistent_verification_ms
    verification = {
        "anchor_only": anchor_verification,
        "persistent_sparse": persistent_verification,
        "pass": anchor_verification["pass"] and persistent_verification["pass"],
    }
    write_json(case_root / "runtime_verification.json", verification)

    corrupted_f1 = float(
        method_metrics["temporal_corrupted_local"]["membership"]["member_f1"]
    )
    anchor_f1 = float(method_metrics["anchor_only_local"]["membership"]["member_f1"])
    persistent_f1 = float(
        method_metrics["persistent_sparse_local"]["membership"]["member_f1"]
    )
    global_metrics = method_metrics.get("persistent_sparse_global")
    taxonomy = _failure_taxonomy(
        corrupted_metrics=method_metrics["temporal_corrupted_local"],
        historical_no_repair_metrics=method_metrics["historical_anchor_no_repair"],
        natural_recompute_metrics=method_metrics["natural_recompute_ablation"],
        anchor_metrics=method_metrics["anchor_only_local"],
        persistent_metrics=method_metrics["persistent_sparse_local"],
        anchor_state=anchor_state,
        persistent_state=persistent_state,
        verification=verification,
        global_metrics=global_metrics,
    )
    no_constraint_equivalent = _method_equivalent(
        method_metrics["historical_anchor_no_repair"],
        method_metrics["persistent_sparse_local"],
    )
    natural_recompute_equivalent = _method_equivalent(
        method_metrics["natural_recompute_ablation"],
        method_metrics["persistent_sparse_local"],
    )
    anchor_persistent_equivalent = _method_equivalent(
        method_metrics["anchor_only_local"],
        method_metrics["persistent_sparse_local"],
    )
    corrupted_method = method_metrics["temporal_corrupted_local"]
    persistent_method = method_metrics["persistent_sparse_local"]
    outcome = classify_repair_outcome(
        corrupted_method=corrupted_method,
        persistent_method=persistent_method,
        verification_pass=bool(verification["pass"]),
    )
    metrics = {
        "schema_version": "1.0.0",
        "implementation_semantics": "RECORDED_ANCHOR_THEN_SPARSE_OVERRIDE",
        "case_uid": case["case_uid"],
        "failure_type": case["failure_type"],
        "scene_id": case["scene_id"],
        "evaluation_role": case.get("evaluation_role", "UNSPECIFIED"),
        "status": "COMPLETED",
        "methods": method_metrics,
        "recovery": {
            "anchor_only_member_recovery": _recovery(corrupted_f1, anchor_f1),
            "persistent_sparse_member_recovery": _recovery(
                corrupted_f1, persistent_f1
            ),
        },
        "constraint_diagnostics": {
            "primitive_count": len(constraints),
            "anchor_hit_count": anchor_state["constraint_hit_count"],
            "anchor_override_count": anchor_state["constraint_override_count"],
            "persistent_hit_count": persistent_state["constraint_hit_count"],
            "persistent_override_count": persistent_state["constraint_override_count"],
            "persistent_native_override_count": persistent_state[
                "constraint_native_override_count"
            ],
            "persistent_historical_override_count": persistent_state[
                "constraint_historical_override_count"
            ],
            "no_constraint_equivalent": no_constraint_equivalent,
            "natural_recompute_equivalent": natural_recompute_equivalent,
            "anchor_persistent_equivalent": anchor_persistent_equivalent,
            "supports_sparse_constraint_causal_claim": bool(
                persistent_state["constraint_historical_override_count"] > 0
                and not no_constraint_equivalent
            ),
        },
        "locality": {
            "closure_event_count": len(closure.event_uids),
            "total_event_count": len(provenance.events),
            "closure_event_fraction": len(closure.event_uids)
            / max(1, len(provenance.events)),
            "closure_observation_count": len(closure.obs_uids),
            "total_observation_count": len(provenance.association_rows),
            "closure_observation_fraction": len(closure.obs_uids)
            / max(1, len(provenance.association_rows)),
            "effective_closure_observation_count": persistent_state.get(
                "closure_effective_observation_count", len(closure.obs_uids)
            ),
            "effective_closure_observation_fraction": persistent_state.get(
                "closure_effective_observation_count", len(closure.obs_uids)
            )
            / max(1, len(provenance.association_rows)),
            "expanded_observation_count": persistent_state.get(
                "closure_expanded_observation_count", 0
            ),
            "expanded_entity_uids": persistent_state.get(
                "closure_expanded_entity_uids", []
            ),
            "closure_entity_count": len(closure.entity_uids),
            "closure_expansion_count": persistent_state.get(
                "closure_expansion_count", 0
            ),
        },
        "snapshot_validation": snapshot.validation,
        "runtime_verification": verification,
        "failure_taxonomy": taxonomy,
        "damage_dimensions": outcome["damage_dimensions"],
        "improved_dimensions": outcome["improved_dimensions"],
        "collateral_safe": outcome["collateral_safe"],
        "outcome_classification": {
            "schema_version": "1.2.0",
            "thresholds": outcome["thresholds"],
        },
        "global_sparse_executed": global_sparse_state is not None,
        "global_corruption_executed": global_corrupted_state is not None,
        "source_hashes_unchanged": source_hashes_before == provenance.source_hashes(),
        "timing": {
            "snapshot": dict(snapshot.state.get("timing") or {}),
            "relation_rebuild_total_wall_ms": float(
                relation.get("total_wall_ms", 0.0)
            ),
            "runtime_invariant_verification_wall_ms": {
                "anchor_only": anchor_verification_ms,
                "persistent_sparse": persistent_verification_ms,
            },
            "basis": {
                "case_total_wall_ms_excluding_final_metrics_write": (
                    "CASE_ENTRY_THROUGH_BRANCH_ARTIFACT_WRITES"
                )
            },
        },
        "pass": outcome["pass"],
    }
    write_json(
        case_root / "corruption_trace.json",
        {
            "plan": case["corruption_plan"],
            "decision_trace": [
                row
                for row in corrupted_state["decision_trace"]
                if row.get("intervention_overrode_natural")
            ],
            "intervention_count": corrupted_state["intervention_count"],
        },
    )
    write_json(
        case_root / "replay_decision_trace.json",
        {
            "historical_anchor_no_repair": historical_no_repair_state[
                "decision_trace"
            ],
            "natural_recompute": natural_recompute_state["decision_trace"],
            "anchor_only": anchor_state["decision_trace"],
            "persistent_sparse": persistent_state["decision_trace"],
        },
    )
    write_json(
        case_root / "transaction.json",
        {
            "status": "NOT_LIVE_COMMIT",
            "reason": "V1 Phase 1-3 event-stream validation only",
            "runtime_gate_pass": verification["pass"],
        },
    )
    for name, state in states.items():
        write_json(case_root / "branches" / f"{name}.json", state)
    metrics["timing"]["case_total_wall_ms_excluding_final_metrics_write"] = (
        time.perf_counter() - case_started
    ) * 1000.0
    write_json(case_root / "benchmark_metrics.json", metrics)
    return metrics


def percentile(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    return float(np.percentile(values, q)) if values else None


def distribution_summary(values: Iterable[float]) -> dict[str, float | None]:
    values = [float(value) for value in values]
    return {
        "mean": float(np.mean(values)) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def selected_metric_paths(root: Path) -> tuple[list[Path], dict[str, Any]]:
    discovered = {
        path.parent.name: path
        for path in sorted(root.glob("*/benchmark_metrics.json"))
    }
    manifest_path = root / "manifests" / "cases.json"
    if not manifest_path.exists():
        return list(discovered.values()), {
            "uses_frozen_manifest": False,
            "manifest_case_count": None,
            "missing_case_uids": [],
            "unexpected_case_uids_ignored": [],
        }
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [str(row["case_uid"]) for row in cases]
    if len(selected) != len(set(selected)):
        raise ValueError("frozen case manifest contains duplicate case_uid values")
    selected_set = set(selected)
    return [discovered[uid] for uid in selected if uid in discovered], {
        "uses_frozen_manifest": True,
        "manifest_case_count": len(selected),
        "missing_case_uids": [uid for uid in selected if uid not in discovered],
        "unexpected_case_uids_ignored": sorted(set(discovered) - selected_set),
    }


def aggregate_results(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    paths, selection_integrity = selected_metric_paths(root)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    taxonomy: dict[str, int] = {}
    for row in rows:
        for label in row.get("failure_taxonomy") or ():
            taxonomy[label] = taxonomy.get(label, 0) + 1
    recoveries = [
        row["recovery"]["persistent_sparse_member_recovery"]
        for row in completed
        if row["recovery"]["persistent_sparse_member_recovery"] is not None
    ]
    costs = [row["methods"]["persistent_sparse_local"]["cost"] for row in completed]
    suffix_runtimes = [
        float(cost.get("suffix_runtime_ms", cost.get("runtime_ms", 0.0)))
        for cost in costs
    ]
    snapshot_runtimes = [float(cost.get("snapshot_runtime_ms", 0.0)) for cost in costs]
    cold_runtimes = [
        float(
            cost.get(
                "cold_snapshot_plus_suffix_runtime_ms",
                float(cost.get("runtime_ms", 0.0))
                + float(cost.get("snapshot_runtime_ms", 0.0)),
            )
        )
        for cost in costs
    ]
    locality_summary = {}
    for name in (
        "closure_event_fraction",
        "closure_observation_fraction",
        "effective_closure_observation_fraction",
    ):
        values = [float(row["locality"][name]) for row in completed]
        locality_summary[name] = distribution_summary(values)
    locality_summary["expanded_case_count"] = sum(
        int(row["locality"].get("closure_expansion_count", 0)) > 0
        for row in completed
    )
    locality_summary["expanded_observation_total"] = sum(
        int(row["locality"].get("expanded_observation_count", 0))
        for row in completed
    )
    method_names = sorted(
        {
            name
            for row in completed
            for name in (row.get("methods") or {})
        }
    )
    method_summary = {}
    for name in method_names:
        available = [row["methods"][name] for row in completed if name in row["methods"]]
        member_f1 = [float(item["membership"]["member_f1"]) for item in available]
        bbox_iou = [float(item["geometry"]["bbox_iou_to_clean"]) for item in available]
        relation_exact = [bool(item["relation"]["edge_state_match"]) for item in available]
        relation_informative = [
            bool(item["relation"].get("informative")) for item in available
        ]
        method_summary[name] = {
            "case_count": len(available),
            "member_f1_mean": float(np.mean(member_f1)) if member_f1 else None,
            "member_f1_median": float(np.median(member_f1)) if member_f1 else None,
            "bbox_iou_mean": float(np.mean(bbox_iou)) if bbox_iou else None,
            "bbox_iou_median": float(np.median(bbox_iou)) if bbox_iou else None,
            "relation_state_exact_rate": (
                float(np.mean(relation_exact)) if relation_exact else None
            ),
            "relation_informative_count": sum(relation_informative),
            "relation_informative_rate": (
                float(np.mean(relation_informative))
                if relation_informative
                else None
            ),
        }

    geometry_recoveries = []
    relation_recoveries = []
    for row in completed:
        corrupted = row["methods"]["temporal_corrupted_local"]
        repaired = row["methods"]["persistent_sparse_local"]
        corrupted_iou = float(corrupted["geometry"]["bbox_iou_to_clean"])
        repaired_iou = float(repaired["geometry"]["bbox_iou_to_clean"])
        if corrupted_iou < 1.0 - BBOX_IOU_ATOL:
            geometry_recoveries.append(
                1.0 - (1.0 - repaired_iou) / max(1e-12, 1.0 - corrupted_iou)
            )
        if not bool(corrupted["relation"]["edge_state_match"]):
            relation_recoveries.append(
                1.0 if bool(repaired["relation"]["edge_state_match"]) else 0.0
            )

    by_failure_type = {}
    for failure_type in sorted(
        set(str(row.get("failure_type")) for row in rows if row.get("failure_type"))
    ):
        typed = [row for row in rows if str(row.get("failure_type")) == failure_type]
        typed_completed = [row for row in typed if row.get("status") == "COMPLETED"]
        typed_recovery = [
            row["recovery"]["persistent_sparse_member_recovery"]
            for row in typed_completed
            if row["recovery"]["persistent_sparse_member_recovery"] is not None
        ]
        by_failure_type[failure_type] = {
            "case_count": len(typed),
            "completed_count": len(typed_completed),
            "pass_count": sum(bool(row.get("pass")) for row in typed),
            "damaging_corruption_count": sum(
                any((row.get("damage_dimensions") or {}).values())
                for row in typed_completed
            ),
            "causal_constraint_support_count": sum(
                bool(
                    (row.get("constraint_diagnostics") or {}).get(
                        "supports_sparse_constraint_causal_claim"
                    )
                )
                for row in typed_completed
            ),
            "median_persistent_member_recovery": (
                float(np.median(typed_recovery)) if typed_recovery else None
            ),
        }
    result = {
        "schema_version": "1.0.0",
        "outcome_classification": {
            "schema_version": "1.2.0",
            "thresholds": {
                "member_f1_atol": MEMBER_F1_ATOL,
                "bbox_iou_atol": BBOX_IOU_ATOL,
            },
        },
        "selection_integrity": selection_integrity,
        "case_count": len(rows),
        "completed_count": len(completed),
        "pass_count": sum(bool(row.get("pass")) for row in rows),
        "median_persistent_member_recovery": (
            float(np.median(recoveries)) if recoveries else None
        ),
        "median_persistent_geometry_recovery": (
            float(np.median(geometry_recoveries)) if geometry_recoveries else None
        ),
        "mean_relation_recovery_on_damaged_cases": (
            float(np.mean(relation_recoveries)) if relation_recoveries else None
        ),
        "method_summary": method_summary,
        "by_failure_type": by_failure_type,
        # Legacy key retained, now explicitly identical to suffix-only.
        "runtime_ms": distribution_summary(suffix_runtimes),
        "runtime": {
            "suffix_only_ms": distribution_summary(suffix_runtimes),
            "snapshot_cumulative_ms": distribution_summary(snapshot_runtimes),
            "cold_snapshot_plus_suffix_ms": distribution_summary(cold_runtimes),
            "amortized_cache_cost_available": False,
            "amortized_cache_cost_note": (
                "Current artifacts store cumulative per-case prefix time but not the "
                "incremental cache and same-frame components separately; do not infer "
                "an amortized speedup from this field."
            ),
        },
        "locality": locality_summary,
        "mean_closure_event_fraction": (
            float(np.mean([row["locality"]["closure_event_fraction"] for row in completed]))
            if completed
            else None
        ),
        "constraint_no_override_count": sum(
            row["constraint_diagnostics"]["persistent_override_count"] == 0
            for row in completed
        ),
        "no_constraint_equivalent_count": sum(
            bool(row["constraint_diagnostics"]["no_constraint_equivalent"])
            for row in completed
        ),
        "natural_recompute_equivalent_count": sum(
            bool(row["constraint_diagnostics"]["natural_recompute_equivalent"])
            for row in completed
        ),
        "anchor_persistent_equivalent_count": sum(
            _method_equivalent(
                row["methods"]["anchor_only_local"],
                row["methods"]["persistent_sparse_local"],
            )
            for row in completed
        ),
        "causal_constraint_support_count": sum(
            bool(
                row["constraint_diagnostics"][
                    "supports_sparse_constraint_causal_claim"
                ]
            )
            for row in completed
        ),
        "collateral_safe_count": sum(
            bool(
                (row["methods"]["persistent_sparse_local"].get(
                    "membership_global"
                ) or {}).get("partition_exact", True)
            )
            for row in completed
        ),
        "collateral_damage_count": sum(
            not bool(
                (row["methods"]["persistent_sparse_local"].get(
                    "membership_global"
                ) or {}).get("partition_exact", True)
            )
            for row in completed
        ),
        "damaging_corruption_count": sum(
            any(row["damage_dimensions"].values()) for row in completed
        ),
        "damage_dimension_counts": {
            dimension: sum(
                bool((row.get("damage_dimensions") or {}).get(dimension))
                for row in completed
            )
            for dimension in ("membership", "geometry", "relation")
        },
        "improved_dimension_counts": {
            dimension: sum(
                bool((row.get("improved_dimensions") or {}).get(dimension))
                for row in completed
            )
            for dimension in ("membership", "geometry", "relation")
        },
        "snapshot_validation_pass_count": sum(
            bool(row["snapshot_validation"]["pass"]) for row in completed
        ),
        "implementation_semantics_mismatch_count": sum(
            row.get("implementation_semantics")
            != "RECORDED_ANCHOR_THEN_SPARSE_OVERRIDE"
            for row in completed
        ),
        "failure_taxonomy": dict(sorted(taxonomy.items())),
        "cases": [
            {
                "case_uid": row.get("case_uid"),
                "failure_type": row.get("failure_type"),
                "status": row.get("status"),
                "pass": row.get("pass"),
                "failure_taxonomy": row.get("failure_taxonomy", []),
                "corrupted_member_f1": (
                    row.get("methods", {})
                    .get("temporal_corrupted_local", {})
                    .get("membership", {})
                    .get("member_f1")
                ),
                "persistent_member_f1": (
                    row.get("methods", {})
                    .get("persistent_sparse_local", {})
                    .get("membership", {})
                    .get("member_f1")
                ),
            }
            for row in rows
        ],
    }
    write_json(root / "revision_v1_metrics.json", result)
    return result
