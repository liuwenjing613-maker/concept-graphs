from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.benchmark.cases import compile_sparse_constraints
from conceptgraph.revision.benchmark.experiment_v1 import (
    _attach_relations,
    aligned_relation_metrics,
    percentile,
    write_json,
)
from conceptgraph.revision.constraints import ReplayMode
from conceptgraph.revision.evaluate import (
    evaluate_state,
    geometry_metrics,
    symmetric_membership_metrics,
)
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.replay import CounterfactualReplayEngine
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine


def _read(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _runtime_comparison(
    local_metrics: dict[str, Any], global_state: dict[str, Any]
) -> dict[str, Any]:
    cost = local_metrics["cost"]
    suffix_ms = float(cost.get("suffix_runtime_ms", cost.get("runtime_ms", 0.0)))
    snapshot_ms = float(cost.get("snapshot_runtime_ms", 0.0))
    cold_ms = float(
        cost.get("cold_snapshot_plus_suffix_runtime_ms", suffix_ms + snapshot_ms)
    )
    global_ms = max(1e-12, float(global_state["runtime_ms"]))
    suffix_ratio = suffix_ms / global_ms
    return {
        # Compatibility field; never describe it as total local cost.
        "runtime_ratio_local_over_global": suffix_ratio,
        "suffix_runtime_ratio_local_over_global": suffix_ratio,
        "cold_runtime_ratio_local_over_global": cold_ms / global_ms,
        "local_suffix_runtime_ms": suffix_ms,
        "local_snapshot_cumulative_runtime_ms": snapshot_ms,
        "local_cold_runtime_ms": cold_ms,
        "global_runtime_ms": global_ms,
        "runtime_ratio_basis": {
            "runtime_ratio_local_over_global": "SUFFIX_ONLY_LEGACY_FIELD",
            "cold_runtime_ratio_local_over_global": "NON_AMORTIZED_COLD_UPPER_BOUND",
        },
    }


def _validate_frozen_selection_request(
    manifest: dict[str, Any],
    *,
    per_type: int,
    primary_manifest: Path,
    cases: list[dict[str, Any]] | None = None,
) -> None:
    mismatches = []
    if int(manifest.get("per_type", -1)) != int(per_type):
        mismatches.append(
            f"per_type frozen={manifest.get('per_type')} requested={per_type}"
        )
    frozen_primary = manifest.get("primary_manifest")
    if not frozen_primary or Path(str(frozen_primary)).resolve() != primary_manifest.resolve():
        mismatches.append(
            f"primary_manifest frozen={frozen_primary} requested={primary_manifest}"
        )
    if cases is not None:
        frozen_uids = [str(uid) for uid in manifest.get("selected_case_uids") or ()]
        case_uids = [str(row.get("case_uid")) for row in cases]
        if len(case_uids) != len(set(case_uids)):
            mismatches.append("cases.json contains duplicate case_uid values")
        if frozen_uids != case_uids:
            mismatches.append(
                "cases.json ordered case IDs differ from frozen selection manifest"
            )
    if mismatches:
        raise RuntimeError(
            "requested global-reference selection conflicts with frozen manifest: "
            + "; ".join(mismatches)
        )


def _prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = Path(args.output_root)
    manifest_path = output / "global_reference_selection_manifest.json"
    cases_path = output / "cases.json"
    if manifest_path.exists() or cases_path.exists():
        if not (manifest_path.exists() and cases_path.exists()):
            raise RuntimeError("frozen global-reference manifest is incomplete")
        manifest = dict(_read(manifest_path))
        cases = list(_read(cases_path))
        _validate_frozen_selection_request(
            manifest,
            per_type=args.per_type,
            primary_manifest=(
                Path(args.primary_root).resolve()
                / "manifests"
                / "case_selection_manifest.json"
            ),
            cases=cases,
        )
        return cases
    primary_root = Path(args.primary_root).resolve()
    primary_manifest = _read(primary_root / "manifests" / "case_selection_manifest.json")
    if bool(primary_manifest.get("outcome_screened")):
        raise RuntimeError("global reference source manifest was outcome-screened")
    selected = []
    counts: dict[str, int] = {}
    for case in _read(primary_root / "manifests" / "cases.json"):
        failure_type = str(case["failure_type"])
        rank = counts.get(failure_type, 0)
        counts[failure_type] = rank + 1
        if rank < args.per_type:
            selected.append(case)
    write_json(
        manifest_path,
        {
            "schema_version": "1.0.0",
            "evaluation_role": "GLOBAL_SAME_CONSTRAINT_REFERENCE",
            "outcome_screened": False,
            "frozen_before_global_outcomes": True,
            "selection_rule": "first N per failure type in frozen primary manifest order",
            "per_type": args.per_type,
            "primary_manifest": str(
                (primary_root / "manifests" / "case_selection_manifest.json").resolve()
            ),
            "selected_case_uids": [str(row["case_uid"]) for row in selected],
        },
    )
    write_json(cases_path, selected)
    for case in selected:
        write_json(output / "cases" / f"{case['case_uid']}.json", case)
    return selected


def _worker(args: argparse.Namespace) -> int:
    case = _read(args.worker_case)
    provenance = ProvenanceIndex(args.base_run)
    source_hashes = provenance.source_hashes()
    constraints = compile_sparse_constraints(case, provenance)
    engine = SparseCounterfactualReplayEngine(provenance)
    global_state = engine.replay_global(
        mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        constraints=constraints,
        historical_anchor_plan=case["corruption_plan"],
    )
    reference = CounterfactualReplayEngine(provenance).clean_state()
    relation = _attach_relations(
        provenance,
        {"reference": reference, "persistent_sparse_global": global_state},
        edge_stream_root=args.edge_stream,
    )
    affected = {
        str(obs_uid)
        for members in (case.get("affected_clean_groups") or {}).values()
        for obs_uid in members
    }
    global_metrics = evaluate_state(
        reference, global_state, affected_observations=affected
    )
    global_metrics["relation"] = aligned_relation_metrics(reference, global_state)
    local_root = Path(args.primary_root) / str(case["case_uid"])
    local_state = _read(local_root / "branches" / "persistent_sparse_local.json")
    local_metrics = _read(local_root / "benchmark_metrics.json")["methods"][
        "persistent_sparse_local"
    ]
    all_observations = {
        str(item)
        for state in (global_state, local_state)
        for members in state["membership"].values()
        for item in members
    }
    local_global_membership = symmetric_membership_metrics(
        global_state["membership"],
        local_state["membership"],
    )
    local_global_geometry = geometry_metrics(
        global_state,
        local_state,
        observation_scope=all_observations,
    )
    local_global_relation = aligned_relation_metrics(global_state, local_state)
    verification = InvariantVerifier().verify(
        state=global_state,
        constraints=constraints,
        source_hashes_before=source_hashes,
        source_hashes_after=provenance.source_hashes(),
        known_observation_uids=provenance.observations,
    )
    checks = {
        "global_runtime_invariants": verification["pass"],
        "local_global_membership_exact": local_global_membership["partition_exact"],
        "local_global_bbox_iou_ge_0_999": local_global_geometry[
            "bbox_iou_to_clean"
        ]
        >= 0.999,
        "local_global_relation_exact": local_global_relation["edge_state_match"],
        "source_hashes_unchanged": source_hashes == provenance.source_hashes(),
    }
    result = {
        "schema_version": "1.0.0",
        "case_uid": case["case_uid"],
        "failure_type": case["failure_type"],
        "evaluation_role": "GLOBAL_SAME_CONSTRAINT_REFERENCE",
        "pass": all(checks.values()),
        "checks": checks,
        "local": local_metrics,
        "global": global_metrics,
        "local_vs_global": {
            "membership": local_global_membership,
            "geometry": local_global_geometry,
            "relation": local_global_relation,
            **_runtime_comparison(local_metrics, global_state),
        },
        "verification": verification,
        "relation_rebuild": relation,
    }
    case_root = Path(args.output_root) / str(case["case_uid"])
    write_json(case_root / "global_state.json", global_state)
    write_json(case_root / "global_reference_metrics.json", result)
    print(json.dumps({"case_uid": case["case_uid"], "pass": result["pass"]}))
    return 0 if result["pass"] else 1


def _reaudit_stored(args: argparse.Namespace) -> dict[str, Any]:
    """Refresh evaluator-only fields without rerunning the expensive global map."""

    output_root = Path(args.output_root)
    primary_root = Path(args.primary_root)
    provenance = ProvenanceIndex(args.base_run)
    source_hashes = provenance.source_hashes()
    audits = []
    for case in list(_read(output_root / "cases.json")):
        uid = str(case["case_uid"])
        case_root = output_root / uid
        result_path = case_root / "global_reference_metrics.json"
        state_path = case_root / "global_state.json"
        if not result_path.exists() or not state_path.exists():
            raise FileNotFoundError(f"incomplete stored global reference for {uid}")
        before = _read(result_path)
        global_state = _read(state_path)
        local_root = primary_root / uid
        reference = _read(local_root / "branches" / "reference.json")
        local_state = _read(local_root / "branches" / "persistent_sparse_local.json")
        local_metrics = _read(local_root / "benchmark_metrics.json")["methods"][
            "persistent_sparse_local"
        ]
        constraints = compile_sparse_constraints(case, provenance)
        affected = {
            str(obs_uid)
            for members in (case.get("affected_clean_groups") or {}).values()
            for obs_uid in members
        }
        global_metrics = evaluate_state(
            reference, global_state, affected_observations=affected
        )
        global_metrics["relation"] = aligned_relation_metrics(reference, global_state)
        all_observations = {
            str(item)
            for state in (global_state, local_state)
            for members in state["membership"].values()
            for item in members
        }
        local_global_membership = symmetric_membership_metrics(
            global_state["membership"],
            local_state["membership"],
        )
        local_global_geometry = geometry_metrics(
            global_state, local_state, observation_scope=all_observations
        )
        local_global_relation = aligned_relation_metrics(global_state, local_state)
        verification = InvariantVerifier().verify(
            state=global_state,
            constraints=constraints,
            source_hashes_before=source_hashes,
            source_hashes_after=global_state.get("source_hashes"),
            known_observation_uids=provenance.observations,
        )
        checks = {
            "global_runtime_invariants": verification["pass"],
            "local_global_membership_exact": local_global_membership[
                "partition_exact"
            ],
            "local_global_bbox_iou_ge_0_999": local_global_geometry[
                "bbox_iou_to_clean"
            ]
            >= 0.999,
            "local_global_relation_exact": local_global_relation["edge_state_match"],
            "source_hashes_unchanged": source_hashes
            == global_state.get("source_hashes"),
        }
        result = {
            "schema_version": "1.1.0",
            "case_uid": uid,
            "failure_type": case["failure_type"],
            "evaluation_role": "GLOBAL_SAME_CONSTRAINT_REFERENCE",
            "pass": all(checks.values()),
            "checks": checks,
            "local": local_metrics,
            "global": global_metrics,
            "local_vs_global": {
                "membership": local_global_membership,
                "geometry": local_global_geometry,
                "relation": local_global_relation,
                **_runtime_comparison(local_metrics, global_state),
            },
            "verification": verification,
            "relation_rebuild": before.get("relation_rebuild"),
            "reaudited_from_immutable_stored_states": True,
        }
        write_json(result_path, result)
        audits.append(
            {
                "case_uid": uid,
                "pass_before": bool(before.get("pass")),
                "pass_after": bool(result["pass"]),
                "checks_before": before.get("checks"),
                "checks_after": checks,
            }
        )
    aggregate = _aggregate(output_root)
    audit = {
        "schema_version": "1.0.0",
        "case_count": len(audits),
        "changed_pass_count": sum(
            row["pass_before"] != row["pass_after"] for row in audits
        ),
        "cases": audits,
        "aggregate": aggregate,
    }
    write_json(output_root / "stored_state_reaudit.json", audit)
    return audit


def _aggregate(output_root: Path) -> dict[str, Any]:
    manifest_cases = list(_read(output_root / "cases.json"))
    selected = [str(row["case_uid"]) for row in manifest_cases]
    if len(selected) != len(set(selected)):
        raise ValueError("frozen global-reference manifest contains duplicate case_uid values")
    discovered = {
        path.parent.name: path
        for path in sorted(output_root.glob("*/global_reference_metrics.json"))
    }
    selected_set = set(selected)
    missing = [uid for uid in selected if uid not in discovered]
    rows = [_read(discovered[uid]) for uid in selected if uid in discovered]
    ratios = [
        float(row["local_vs_global"]["runtime_ratio_local_over_global"])
        for row in rows
    ]
    cold_ratios = [
        float(
            row["local_vs_global"].get(
                "cold_runtime_ratio_local_over_global",
                row["local_vs_global"]["runtime_ratio_local_over_global"],
            )
        )
        for row in rows
    ]
    result = {
        "schema_version": "1.0.0",
        "selection_integrity": {
            "uses_frozen_manifest": True,
            "manifest_case_count": len(selected),
            "missing_case_uids": missing,
            "unexpected_case_uids_ignored": sorted(set(discovered) - selected_set),
        },
        "case_count": len(rows),
        "pass_count": sum(bool(row["pass"]) for row in rows),
        "member_exact_count": sum(
            bool(row["checks"]["local_global_membership_exact"]) for row in rows
        ),
        "geometry_fidelity_count": sum(
            bool(row["checks"]["local_global_bbox_iou_ge_0_999"]) for row in rows
        ),
        "relation_exact_count": sum(
            bool(row["checks"]["local_global_relation_exact"]) for row in rows
        ),
        "relation_informative_count": sum(
            bool(
                ((row.get("local_vs_global") or {}).get("relation") or {}).get(
                    "informative"
                )
            )
            for row in rows
        ),
        "runtime_ratio_local_over_global": {
            "p50": percentile(ratios, 50),
            "p95": percentile(ratios, 95),
            "max": max(ratios) if ratios else None,
            "mean": float(np.mean(ratios)) if ratios else None,
            "basis": "SUFFIX_ONLY_LEGACY_FIELD",
        },
        "cold_runtime_ratio_local_over_global": {
            "p50": percentile(cold_ratios, 50),
            "p95": percentile(cold_ratios, 95),
            "max": max(cold_ratios) if cold_ratios else None,
            "mean": float(np.mean(cold_ratios)) if cold_ratios else None,
            "basis": "NON_AMORTIZED_SNAPSHOT_PLUS_SUFFIX_UPPER_BOUND",
        },
        "cases": [
            {
                "case_uid": row["case_uid"],
                "failure_type": row["failure_type"],
                "pass": row["pass"],
                "checks": row["checks"],
            }
            for row in rows
        ],
    }
    write_json(output_root / "global_reference_metrics.json", result)
    return result


def _parallel(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    jobs = min(args.jobs, len(gpus), len(cases))
    if jobs < 1:
        raise ValueError("at least one worker and GPU are required")
    logs = Path(args.output_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(cases)
    active: list[dict[str, Any]] = []
    failures = []
    cursor = 0
    try:
        while cursor < len(pending) or active:
            while cursor < len(pending) and len(active) < jobs:
                case = pending[cursor]
                gpu = gpus[cursor % len(gpus)]
                cursor += 1
                uid = str(case["case_uid"])
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--base-run",
                    args.base_run,
                    "--primary-root",
                    args.primary_root,
                    "--output-root",
                    args.output_root,
                    "--worker-case",
                    str(Path(args.output_root) / "cases" / f"{uid}.json"),
                ]
                if args.edge_stream:
                    command.extend(["--edge-stream", args.edge_stream])
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                handle = (logs / f"{uid}.log").open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active.append(
                    {"process": process, "handle": handle, "case_uid": uid, "gpu": gpu}
                )
            time.sleep(1.0)
            remaining = []
            for row in active:
                code = row["process"].poll()
                if code is None:
                    remaining.append(row)
                    continue
                row["handle"].close()
                if code:
                    failures.append(
                        {"case_uid": row["case_uid"], "gpu": row["gpu"], "exit_code": code}
                    )
                print(
                    json.dumps(
                        {"case_uid": row["case_uid"], "gpu": row["gpu"], "exit_code": code}
                    ),
                    flush=True,
                )
            active = remaining
    finally:
        for row in active:
            if row["process"].poll() is None:
                row["process"].terminate()
        for row in active:
            if row["process"].poll() is None:
                try:
                    row["process"].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    row["process"].kill()
                    row["process"].wait()
            if not row["handle"].closed:
                row["handle"].close()
    aggregate = _aggregate(Path(args.output_root))
    aggregate["worker_failure_count"] = len(failures)
    aggregate["worker_failures"] = failures
    write_json(Path(args.output_root) / "orchestration.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 same-constraint global reference")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--edge-stream")
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--gpus", default="1,3,5")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reaudit-stored", action="store_true")
    parser.add_argument("--worker-case")
    args = parser.parse_args()
    if args.worker_case:
        raise SystemExit(_worker(args))
    if args.reaudit_stored:
        result = _reaudit_stored(args)
        print(json.dumps(result, indent=2))
        raise SystemExit(
            0
            if result["aggregate"]["pass_count"]
            == result["aggregate"]["selection_integrity"]["manifest_case_count"]
            else 1
        )
    cases = _prepare(args)
    if args.prepare_only:
        print(json.dumps({"prepared": len(cases)}))
        return
    result = _parallel(args, cases)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["worker_failure_count"] == 0 else 1)


if __name__ == "__main__":
    main()
