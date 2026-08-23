from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.benchmark import BatchCaseSampler
from conceptgraph.revision.benchmark.experiment_v1 import (
    SceneExperimentContext,
    aggregate_results,
    run_case,
    write_json,
)
from conceptgraph.revision.index import ProvenanceIndex


def _read(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_frozen_primary_request(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    base_run: str | Path,
    scene: str,
    seed: int,
    count_per_type: int,
    global_sparse_per_type: int,
    global_corruption_per_type: int,
) -> None:
    mismatches = []
    scalar_checks = {
        "scene": (manifest.get("scene"), str(scene)),
        "seed": (manifest.get("seed"), int(seed)),
        "global_sparse_per_type": (
            manifest.get("global_sparse_per_type"),
            int(global_sparse_per_type),
        ),
        "global_corruption_per_type": (
            manifest.get("global_corruption_per_type"),
            int(global_corruption_per_type),
        ),
    }
    for name, (frozen, requested) in scalar_checks.items():
        if frozen != requested:
            mismatches.append(f"{name} frozen={frozen} requested={requested}")
    frozen_base = manifest.get("base_run")
    if not frozen_base or Path(str(frozen_base)).resolve() != Path(base_run).resolve():
        mismatches.append(f"base_run frozen={frozen_base} requested={base_run}")
    subset_counts = {
        str(row.get("failure_type")): int(row.get("requested_count", -1))
        for row in manifest.get("subsets") or ()
    }
    for failure_type in ("FALSE_SPLIT", "WRONG_MEMBERSHIP", "FALSE_MERGE"):
        if subset_counts.get(failure_type) != int(count_per_type):
            mismatches.append(
                f"{failure_type}.requested_count "
                f"frozen={subset_counts.get(failure_type)} requested={count_per_type}"
            )
    frozen_uids = [
        str(uid)
        for subset in manifest.get("subsets") or ()
        for uid in subset.get("selected_case_uids") or ()
    ]
    case_uids = [str(row.get("case_uid")) for row in cases]
    if len(case_uids) != len(set(case_uids)):
        mismatches.append("cases.json contains duplicate case_uid values")
    if frozen_uids != case_uids:
        mismatches.append("cases.json ordered case IDs differ from frozen manifest")
    if int(manifest.get("case_count", -1)) != len(cases):
        mismatches.append(
            f"case_count frozen={manifest.get('case_count')} actual={len(cases)}"
        )
    if mismatches:
        raise RuntimeError(
            "requested primary selection conflicts with frozen manifest: "
            + "; ".join(mismatches)
        )


def _prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_root = Path(args.output_root) / "manifests"
    manifest_path = manifest_root / "case_selection_manifest.json"
    cases_path = manifest_root / "cases.json"
    if manifest_path.exists() or cases_path.exists():
        if not (manifest_path.exists() and cases_path.exists()):
            raise RuntimeError("frozen manifest is incomplete; do not silently regenerate it")
        manifest = dict(_read(manifest_path))
        cases = list(_read(cases_path))
        _validate_frozen_primary_request(
            manifest,
            cases,
            base_run=args.base_run,
            scene=args.scene,
            seed=args.seed,
            count_per_type=args.count_per_type,
            global_sparse_per_type=args.global_sparse_per_type,
            global_corruption_per_type=args.global_corruption_per_type,
        )
        return cases
    provenance = ProvenanceIndex(args.base_run)
    sampler = BatchCaseSampler(
        provenance,
        scene=args.scene,
        seed=args.seed,
        frame_count=args.frame_count,
    )
    cases, manifest = sampler.sample_matrix(count_per_type=args.count_per_type)
    by_type: dict[str, int] = {}
    for case in cases:
        failure_type = str(case["failure_type"])
        rank = by_type.get(failure_type, 0)
        by_type[failure_type] = rank + 1
        case["run_global_sparse"] = rank < args.global_sparse_per_type
        case["run_global_corruption"] = rank < args.global_corruption_per_type
    manifest["base_run"] = str(Path(args.base_run).resolve())
    manifest["source_hashes"] = provenance.source_hashes()
    manifest["global_sparse_per_type"] = args.global_sparse_per_type
    manifest["global_corruption_per_type"] = args.global_corruption_per_type
    manifest["frozen_before_outcomes"] = True
    write_json(manifest_path, manifest)
    write_json(cases_path, cases)
    case_root = manifest_root / "cases"
    for case in cases:
        write_json(case_root / f"{case['case_uid']}.json", case)
    return cases


def _worker(args: argparse.Namespace) -> int:
    case = _read(args.worker_case)
    try:
        result = run_case(
            base_run=args.base_run,
            output_root=args.output_root,
            case=case,
            edge_stream_root=args.edge_stream,
            run_global_sparse=bool(case.get("run_global_sparse")),
            run_global_corruption=bool(case.get("run_global_corruption")),
        )
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "case_uid": case.get("case_uid"),
            "failure_type": case.get("failure_type"),
            "status": "FAILED",
            "failure_taxonomy": ["UNHANDLED_EXECUTION_FAILURE"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pass": False,
        }
        write_json(
            Path(args.output_root) / str(case["case_uid"]) / "benchmark_metrics.json",
            failure,
        )
        raise
    print(json.dumps({"case_uid": case["case_uid"], "pass": result["pass"]}))
    return 0


def _worker_shard(args: argparse.Namespace) -> int:
    cases = sorted(_read(args.worker_shard), key=lambda row: (int(row["frame_idx"]), row["case_uid"]))
    context = SceneExperimentContext.build(args.base_run)
    failures = []
    for index, case in enumerate(cases, 1):
        try:
            result = run_case(
                base_run=args.base_run,
                output_root=args.output_root,
                case=case,
                edge_stream_root=args.edge_stream,
                run_global_sparse=bool(case.get("run_global_sparse")),
                run_global_corruption=bool(case.get("run_global_corruption")),
                context=context,
            )
            print(
                json.dumps(
                    {
                        "shard_progress": index,
                        "shard_total": len(cases),
                        "case_uid": case["case_uid"],
                        "pass": result.get("pass"),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "schema_version": "1.0.0",
                "case_uid": case.get("case_uid"),
                "failure_type": case.get("failure_type"),
                "status": "FAILED",
                "failure_taxonomy": ["UNHANDLED_EXECUTION_FAILURE"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pass": False,
            }
            write_json(
                Path(args.output_root)
                / str(case["case_uid"])
                / "benchmark_metrics.json",
                failure,
            )
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    return 1 if failures else 0


def _run_parallel(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU identifier is required")
    jobs = min(args.jobs, len(gpus), len(cases))
    ordered = sorted(cases, key=lambda row: (int(row["frame_idx"]), row["case_uid"]))
    shards = [[] for _ in range(jobs)]
    for index, case in enumerate(ordered):
        shards[index % jobs].append(case)
    shard_root = Path(args.output_root) / "manifests" / "shards"
    active: list[tuple[subprocess.Popen[Any], Any, str, str]] = []
    failures = []
    logs = Path(args.output_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        shard_uid = f"shard_{index:02d}"
        shard_path = shard_root / f"{shard_uid}.json"
        write_json(shard_path, shard)
        gpu = gpus[index]
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--base-run",
            args.base_run,
            "--output-root",
            args.output_root,
            "--scene",
            args.scene,
            "--worker-shard",
            str(shard_path),
        ]
        if args.edge_stream:
            command.extend(["--edge-stream", args.edge_stream])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        log_handle = (logs / f"{shard_uid}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        active.append((process, log_handle, shard_uid, gpu))
    try:
        while active:
            time.sleep(1.0)
            remaining = []
            for process, handle, uid, gpu in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, handle, uid, gpu))
                    continue
                handle.close()
                if code:
                    failures.append({"shard_uid": uid, "exit_code": code, "gpu": gpu})
                print(
                    json.dumps(
                        {
                            "completed_shard": uid,
                            "total_shards": jobs,
                            "exit_code": code,
                            "gpu": gpu,
                        }
                    ),
                    flush=True,
                )
            active = remaining
    finally:
        # A stopped parent must not leave CPU-heavy replay shards behind.  This is
        # also important scientifically: a later rerun must not overlap stale workers.
        for process, handle, _, _ in active:
            if process.poll() is None:
                process.terminate()
        for process, handle, _, _ in active:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not handle.closed:
                handle.close()
    aggregate = aggregate_results(args.output_root)
    orchestration = {
        "case_count": len(cases),
        "worker_failure_count": len(failures),
        "worker_failures": failures,
        "aggregate": aggregate,
    }
    write_json(Path(args.output_root) / "orchestration.json", orchestration)
    return orchestration


def main() -> None:
    parser = argparse.ArgumentParser(description="Revision Kernel V1 batch runner")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--edge-stream")
    parser.add_argument("--count-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--frame-count", type=int, default=200)
    parser.add_argument("--global-sparse-per-type", type=int, default=0)
    parser.add_argument("--global-corruption-per-type", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-case")
    parser.add_argument("--worker-shard")
    args = parser.parse_args()
    if args.worker_case:
        raise SystemExit(_worker(args))
    if args.worker_shard:
        raise SystemExit(_worker_shard(args))
    cases = _prepare(args)
    if args.prepare_only:
        print(json.dumps({"prepared": len(cases), "output_root": args.output_root}))
        return
    result = _run_parallel(args, cases)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["worker_failure_count"] == 0 else 1)


if __name__ == "__main__":
    main()
