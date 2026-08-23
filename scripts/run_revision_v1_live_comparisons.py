from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _aggregate(root: Path) -> dict[str, Any]:
    cases = list(_read(root / "cases.json"))
    selected = [str(row["case_uid"]) for row in cases]
    if len(selected) != len(set(selected)):
        raise ValueError("frozen live-fidelity manifest contains duplicate case_uid values")
    discovered = {
        path.stem: path for path in sorted((root / "comparisons").glob("*.json"))
    }
    selected_set = set(selected)
    missing = [uid for uid in selected if uid not in discovered]
    rows = [_read(discovered[uid]) for uid in selected if uid in discovered]
    return {
        "schema_version": "1.0.0",
        "selection_integrity": {
            "uses_frozen_manifest": True,
            "manifest_case_count": len(selected),
            "missing_case_uids": missing,
            "unexpected_case_uids_ignored": sorted(set(discovered) - selected_set),
        },
        "case_count": len(rows),
        "pass_count": sum(bool(row.get("pass")) for row in rows),
        "single_injection_exact_count": sum(
            bool((row.get("checks") or {}).get("single_injection_exact")) for row in rows
        ),
        "decision_trace_exact_count": sum(
            bool((row.get("checks") or {}).get("downstream_decision_kind_exact"))
            for row in rows
        ),
        "membership_exact_count": sum(
            bool((row.get("checks") or {}).get("final_membership_partition_exact"))
            for row in rows
        ),
        "object_payload_exact_count": sum(
            bool((row.get("checks") or {}).get("final_object_payload_exact"))
            for row in rows
        ),
        "postprocess_exact_count": sum(
            bool((row.get("checks") or {}).get("postprocess_counts_exact")) for row in rows
        ),
        "cases": [
            {
                "case_uid": row.get("case_uid"),
                "pass": row.get("pass"),
                "checks": row.get("checks"),
                "decision_kind_mismatch_count": (
                    row.get("decisions") or {}
                ).get("decision_kind_mismatch_count"),
                "target_origin_mismatch_count": (
                    row.get("decisions") or {}
                ).get("target_origin_mismatch_count"),
                "member_f1": (row.get("membership") or {}).get("member_f1"),
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all frozen live/simulator comparisons")
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--gpus", default="1,3,5")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.selection_root).resolve()
    cases = list(_read(root / "cases.json"))
    live_runs = dict(_read(root / "live_orchestration.json")["live_runs"])
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    jobs = min(args.jobs, len(gpus), len(cases))
    if jobs < 1:
        raise ValueError("at least one worker and GPU are required")
    comparisons = root / "comparisons"
    logs = root / "logs"
    comparisons.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    active: list[dict[str, Any]] = []
    cursor = 0
    failures = []
    try:
        while cursor < len(cases) or active:
            while cursor < len(cases) and len(active) < jobs:
                case = cases[cursor]
                gpu = gpus[cursor % len(gpus)]
                cursor += 1
                uid = str(case["case_uid"])
                if uid not in live_runs:
                    raise RuntimeError(f"missing live run for {uid}")
                output = comparisons / f"{uid}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("compare_revision_live_simulator_v1.py")),
                    "--base-run",
                    args.base_run,
                    "--live-run",
                    live_runs[uid],
                    "--case",
                    str(root / "cases" / f"{uid}.json"),
                    "--output",
                    str(output),
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                handle = (logs / f"compare_{uid}.log").open("w", encoding="utf-8")
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
    result = _aggregate(root)
    result["worker_failure_count"] = len(failures)
    result["worker_failures"] = failures
    _write(root / "live_fidelity_metrics.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass_count"] == len(cases) else 1)


if __name__ == "__main__":
    main()
