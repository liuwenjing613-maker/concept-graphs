from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cases import ControlledCaseBuilder, apply_controlled_membership_corruption
from .evaluate import edge_metrics, evaluate_case
from .index import ProvenanceIndex
from .relations import (
    AliDevBaselineRelationBackend,
    load_baseline_frame_records,
    remap_frame_records,
)
from .replay import CounterfactualReplayEngine
from .tracing import CausalTracer
from .transactions import ShadowTransactionManager


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    temporary.replace(destination)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def run_controlled_case(
    *,
    base_run: str | Path,
    output_root: str | Path,
    case: Mapping[str, Any],
    run_global: bool = True,
    edge_stream_root: str | Path | None = None,
) -> dict[str, Any]:
    provenance = ProvenanceIndex(base_run)
    source_hashes_before = provenance.source_hashes()
    case_root = Path(output_root) / str(case["case_uid"])
    case_root.mkdir(parents=True, exist_ok=True)
    write_json(case_root / "case.json", case)
    write_json(case_root / "corruption.json", case["corruption_plan"])
    write_json(case_root / "constraint.json", case["oracle_constraint"])

    trace = CausalTracer(provenance).trace(case)
    write_json(case_root / "trace.json", trace)
    write_json(case_root / "dependency.json", trace["dependency_closure"])

    engine = CounterfactualReplayEngine(provenance)
    clean_state = engine.clean_state()
    corrupted_state = engine.replay_local(case, branch="corrupted")
    refusion_state = engine.final_member_refusion(case)
    local_state = engine.replay_local(case, branch="repaired")
    global_state = (
        engine.replay_global(case, branch="repaired") if run_global else clean_state
    )
    if not run_global:
        global_state = dict(global_state)
        global_state["branch"] = "global_replay_skipped"
        global_state["scope"] = "not_executed"

    branch_states = {
        "clean": clean_state,
        "corrupted": corrupted_state,
        "final_member_refusion": refusion_state,
        "local_replay": local_state,
        "global_replay": global_state,
    }
    relation_results = {}
    for branch_name, state in branch_states.items():
        relation_backend = AliDevBaselineRelationBackend()
        relation_backend.invalidate(trace["dependency_closure"]["entity_uids"])
        relation_objects, frame_records = load_baseline_frame_records(
            provenance,
            state["membership"],
            edge_stream_root=edge_stream_root,
        )
        relation_result = relation_backend.rebuild(
            objects=relation_objects, frame_records=frame_records
        )
        state["edges"] = relation_result["output_edges"]
        relation_results[branch_name] = relation_result
    relation_summary = {
        "schema_version": "0.2.0",
        "strategy": "GLOBAL_BASELINE_EDGE_REPLAY",
        "node_replay_scope": "dependency-local",
        "edge_replay_scope": "global baseline reconstruction",
        "edge_stream_root": str(edge_stream_root) if edge_stream_root else None,
        "all_structural_validations_pass": all(
            result["validation"]["pass"] for result in relation_results.values()
        ),
        "informative": bool(relation_results["clean"]["informative"]),
        "branches": relation_results,
    }
    write_json(case_root / "edge_rebuild_summary.json", relation_summary)

    branch_root = case_root / "branches"
    write_json(branch_root / "clean.json", clean_state)
    write_json(branch_root / "corrupted.json", corrupted_state)
    write_json(branch_root / "final_member_refusion.json", refusion_state)
    write_json(branch_root / "local_replay.json", local_state)
    write_json(branch_root / "global_replay.json", global_state)

    transaction_manager = ShadowTransactionManager(provenance, output_root)
    transaction = transaction_manager.prepare(
        case=case,
        trace=trace,
        constraint=case["oracle_constraint"],
    )
    outcome = transaction_manager.verify_and_commit(
        transaction=transaction,
        baseline_state=clean_state,
        derived_state=local_state,
    )
    metrics = evaluate_case(
        case=case,
        clean_state=clean_state,
        corrupted_state=corrupted_state,
        refusion_state=refusion_state,
        local_state=local_state,
        global_state=global_state,
        verification=outcome["verification"],
    )
    metrics["relation_rebuild"] = relation_summary
    metrics["transaction_status"] = outcome["transaction"]["commit_status"]
    metrics["source_hashes_before"] = source_hashes_before
    metrics["source_hashes_after"] = provenance.source_hashes()
    write_json(case_root / "metrics.json", metrics)
    write_json(
        case_root / "before_after_summary.json",
        {
            "case_uid": case["case_uid"],
            "clean_state_hash": clean_state["state_hash"],
            "corrupted_state_hash": corrupted_state["state_hash"],
            "repaired_state_hash": local_state["state_hash"],
            "corrupted_member_f1": metrics["methods"]["corrupted"]["membership"][
                "member_f1"
            ],
            "repaired_member_f1": metrics["methods"]["counterfactual_local_replay"][
                "membership"
            ]["member_f1"],
            "verification_pass": outcome["verification"]["pass"],
            "transaction_status": outcome["transaction"]["commit_status"],
        },
    )
    return metrics


def select_cases(
    base_run: str | Path,
    failure_types: Iterable[str],
    *,
    edge_stream_root: str | Path | None = None,
    require_relation_change: bool = False,
    candidate_limit: int = 100,
) -> list[dict[str, Any]]:
    """Select deterministic cases, optionally requiring a real corrupted edge-set change.

    Relation-sensitive selection is intentionally an evaluation stress set, not an
    estimate of naturally occurring failure frequency. Every selected case is later
    rerun through the unchanged baseline relation backend by ``run_controlled_case``.
    """
    provenance = ProvenanceIndex(base_run)
    builder = ControlledCaseBuilder(provenance)
    failures = [str(item).upper() for item in failure_types]
    if not require_relation_change:
        return [builder.select(item) for item in failures]
    if edge_stream_root is None:
        raise ValueError("relation-sensitive selection requires --edge-stream")
    if candidate_limit < 1:
        raise ValueError("candidate limit must be at least one")

    engine = CounterfactualReplayEngine(provenance)
    clean_state = engine.clean_state()
    clean_objects, clean_records = load_baseline_frame_records(
        provenance,
        clean_state["membership"],
        edge_stream_root=edge_stream_root,
    )
    clean_relation = AliDevBaselineRelationBackend().rebuild(
        objects=clean_objects,
        frame_records=clean_records,
    )
    clean_state = dict(clean_state)
    clean_state["edges"] = clean_relation["output_edges"]

    selected = []
    for failure_type in failures:
        candidates = builder.ranked_candidates(failure_type, limit=candidate_limit)
        chosen = None
        best_f1 = 1.0
        static_impact_candidates = 0
        for rank, case in enumerate(candidates, 1):
            static_membership = apply_controlled_membership_corruption(
                clean_state["membership"], case
            )
            static_objects, static_records = remap_frame_records(
                clean_records,
                static_membership,
            )
            static_relation = AliDevBaselineRelationBackend().rebuild(
                objects=static_objects,
                frame_records=static_records,
            )
            static_state = {
                "membership": static_membership,
                "edges": static_relation["output_edges"],
            }
            static_metrics = edge_metrics(clean_state, static_state)
            if static_metrics["edge_state_match"]:
                continue
            static_impact_candidates += 1

            corrupted_state = engine.replay_local(case, branch="corrupted")
            objects, records = remap_frame_records(
                clean_records,
                corrupted_state["membership"],
            )
            relation = AliDevBaselineRelationBackend().rebuild(
                objects=objects,
                frame_records=records,
            )
            corrupted_state = dict(corrupted_state)
            corrupted_state["edges"] = relation["output_edges"]
            metrics = edge_metrics(clean_state, corrupted_state)
            best_f1 = min(best_f1, float(metrics["edge_set_f1_to_clean"]))
            if not metrics["edge_state_match"]:
                chosen = dict(case)
                chosen["selection_metadata"] = {
                    "criterion": "CORRUPTED_RELATION_STATE_DIFFERS_FROM_CLEAN",
                    "evaluation_role": "EDGE_SENSITIVE_STRESS_SET",
                    "candidate_rank": rank,
                    "candidates_screened": rank,
                    "candidate_limit": candidate_limit,
                    "static_impact_candidates_tested": static_impact_candidates,
                    "clean_edge_count": metrics["clean_edge_count"],
                    "corrupted_edge_count": metrics["candidate_edge_count"],
                    "corrupted_edge_f1_to_clean": metrics["edge_set_f1_to_clean"],
                    "false_positive_edge_count": metrics["false_positive_edge_count"],
                    "false_negative_edge_count": metrics["false_negative_edge_count"],
                    "support_mismatch_edge_count": metrics[
                        "support_mismatch_edge_count"
                    ],
                    "support_absolute_error": metrics["support_absolute_error"],
                    "static_corrupted_edge_f1_to_clean": static_metrics[
                        "edge_set_f1_to_clean"
                    ],
                    "static_support_mismatch_edge_count": static_metrics[
                        "support_mismatch_edge_count"
                    ],
                }
                break
        if chosen is None:
            raise RuntimeError(
                f"no edge-sensitive {failure_type} case among {len(candidates)} "
                f"candidates ({static_impact_candidates} direct-impact candidates; "
                f"best replay edge F1={best_f1:.6f})"
            )
        selected.append(chosen)
    return selected


def build_aggregate_report(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    metrics_files = sorted(root.glob("*/metrics.json"))
    cases = [read_json(path) for path in metrics_files]
    if not cases:
        raise FileNotFoundError(f"no case metrics under {root}")
    pass_count = sum(bool(case.get("pass")) for case in cases)
    local_f1 = [
        case["methods"]["counterfactual_local_replay"]["membership"]["member_f1"]
        for case in cases
    ]
    corrupt_f1 = [
        case["methods"]["corrupted"]["membership"]["member_f1"] for case in cases
    ]
    runtime_ratios = [
        case["local_vs_global"]["runtime_ratio"]
        for case in cases
        if case["methods"]["global_replay_reference"]["cost"]["runtime_ms"] > 0
    ]
    informative_relations = [
        case for case in cases if case["relation_diagnostics"]["informative"]
    ]
    aggregate = {
        "schema_version": "0.2.0",
        "case_count": len(cases),
        "pass_count": pass_count,
        "all_cases_pass": pass_count == len(cases),
        "mean_corrupted_member_f1": sum(corrupt_f1) / len(corrupt_f1),
        "mean_repaired_member_f1": sum(local_f1) / len(local_f1),
        "mean_local_global_runtime_ratio": (
            sum(runtime_ratios) / len(runtime_ratios) if runtime_ratios else None
        ),
        "informative_relation_case_count": len(informative_relations),
        "relation_corruption_changed_case_count": sum(
            bool(case["relation_diagnostics"]["corruption_changes_relation"])
            for case in informative_relations
        ),
        "all_local_relations_match_clean": all(
            bool(case["relation_diagnostics"]["local_matches_clean"])
            for case in informative_relations
        ),
        "cases": [
            {
                "case_uid": case["case_uid"],
                "failure_type": case["failure_type"],
                "pass": case["pass"],
                "corrupted_member_f1": case["methods"]["corrupted"]["membership"][
                    "member_f1"
                ],
                "repaired_member_f1": case["methods"]["counterfactual_local_replay"][
                    "membership"
                ]["member_f1"],
                "local_runtime_ms": case["methods"]["counterfactual_local_replay"][
                    "cost"
                ]["runtime_ms"],
                "global_runtime_ms": case["methods"]["global_replay_reference"]["cost"][
                    "runtime_ms"
                ],
                "global_reference_executed": case["methods"]["global_replay_reference"][
                    "cost"
                ]["runtime_ms"]
                > 0,
                "transaction_status": case["transaction_status"],
                "clean_edge_count": case["methods"]["clean"]["relation"][
                    "clean_edge_count"
                ],
                "corrupted_edge_f1": case["methods"]["corrupted"]["relation"][
                    "edge_set_f1_to_clean"
                ],
                "repaired_edge_f1": case["methods"][
                    "counterfactual_local_replay"
                ]["relation"]["edge_set_f1_to_clean"],
                "corrupted_edge_state_match": case["methods"]["corrupted"][
                    "relation"
                ]["edge_state_match"],
                "corrupted_support_mismatch_count": case["methods"]["corrupted"][
                    "relation"
                ]["support_mismatch_edge_count"],
                "repaired_edge_state_match": case["methods"][
                    "counterfactual_local_replay"
                ]["relation"]["edge_state_match"],
                "repaired_support_mismatch_count": case["methods"][
                    "counterfactual_local_replay"
                ]["relation"]["support_mismatch_edge_count"],
            }
            for case in cases
        ],
    }
    write_json(root / "revision_metrics.json", aggregate)
    lines = [
        "# Revision Kernel controlled-validation report",
        "",
        f"- Cases: {len(cases)}",
        f"- Passed: {pass_count}/{len(cases)}",
        f"- Mean corrupted member F1: {aggregate['mean_corrupted_member_f1']:.6f}",
        f"- Mean repaired member F1: {aggregate['mean_repaired_member_f1']:.6f}",
        "- Mean local/global runtime ratio: "
        + (
            f"{aggregate['mean_local_global_runtime_ratio']:.6f}"
            if aggregate["mean_local_global_runtime_ratio"] is not None
            else "n/a (global reference not executed)"
        ),
        f"- Informative relation cases: {len(informative_relations)}/{len(cases)}",
        "- Local relation recovery matches clean: "
        + str(aggregate["all_local_relations_match_clean"]),
        "",
        "| Failure | Case | Corrupt F1 | Repaired F1 | Local ms | Global ms | Commit | Pass |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in aggregate["cases"]:
        global_time = (
            f"{row['global_runtime_ms']:.1f}" if row["global_reference_executed"] else "n/a"
        )
        lines.append(
            f"| {row['failure_type']} | `{row['case_uid']}` | "
            f"{row['corrupted_member_f1']:.6f} | {row['repaired_member_f1']:.6f} | "
            f"{row['local_runtime_ms']:.1f} | {global_time} | "
            f"{row['transaction_status']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "| Failure | Case | Clean edges | Corrupt edge F1 | Corrupt support mismatches | Repaired edge F1 | Repaired exact state |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in aggregate["cases"]:
        lines.append(
            f"| {row['failure_type']} | `{row['case_uid']}` | "
            f"{row['clean_edge_count']} | {row['corrupted_edge_f1']:.6f} | "
            f"{row['corrupted_support_mismatch_count']} | "
            f"{row['repaired_edge_f1']:.6f} | "
            f"{row['repaired_edge_state_match']} |"
        )
    lines.extend(
        [
            "",
            "Node replay is dependency-local. Edge replay is global reconstruction through the unchanged ali-dev `process_edges` path.",
            "An empty edge stream remains explicitly non-informative. A supplied frozen edge stream is hash-verified and replayed for every clean/corrupt/refusion/local/global branch.",
            "",
        ]
    )
    (root / "revision_report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate
