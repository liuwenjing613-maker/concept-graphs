from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cases import ControlledCaseBuilder
from .evaluate import evaluate_case
from .index import ProvenanceIndex
from .relations import AliDevBaselineRelationBackend, load_baseline_frame_records
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

    relation_backend = AliDevBaselineRelationBackend()
    relation_backend.invalidate(trace["dependency_closure"]["entity_uids"])
    relation_objects, frame_records = load_baseline_frame_records(
        provenance, local_state["membership"]
    )
    relation_result = relation_backend.rebuild(
        objects=relation_objects, frame_records=frame_records
    )
    local_state["edges"] = relation_result["output_edges"]
    write_json(case_root / "edge_rebuild_summary.json", relation_result)

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
    metrics["relation_rebuild"] = relation_result
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


def select_cases(base_run: str | Path, failure_types: Iterable[str]) -> list[dict[str, Any]]:
    builder = ControlledCaseBuilder(ProvenanceIndex(base_run))
    return [builder.select(item) for item in failure_types]


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
    aggregate = {
        "schema_version": "0.1.0",
        "case_count": len(cases),
        "pass_count": pass_count,
        "all_cases_pass": pass_count == len(cases),
        "mean_corrupted_member_f1": sum(corrupt_f1) / len(corrupt_f1),
        "mean_repaired_member_f1": sum(local_f1) / len(local_f1),
        "mean_local_global_runtime_ratio": (
            sum(runtime_ratios) / len(runtime_ratios) if runtime_ratios else None
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
            "Node replay is dependency-local. Edge replay is global reconstruction through the unchanged ali-dev `process_edges` path.",
            "An empty real edge stream is reported as non-informative; the non-empty semantic path is covered separately by regression tests.",
            "",
        ]
    )
    (root / "revision_report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate
