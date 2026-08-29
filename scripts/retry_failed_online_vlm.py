#!/usr/bin/env python3
"""Serially retry invalid VLM calls against their original frozen packets."""

from __future__ import annotations

import argparse
import getpass
import json
import time
from pathlib import Path
from typing import Any

from validate_unified_vlm_v2 import (
    FrozenRun,
    call_vlm,
    prepare_case,
    write_json,
    write_root_html,
)

RETRYABLE_STATUSES = {"API_OR_PARSE_ERROR", "DEFER_INVALID_OUTPUT"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--online-subdir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    online_root = args.experiment_root / args.online_subdir
    case_root = online_root / "vlm_object_state_v2"
    dispatch_records = _read_json(online_root / "dispatch_order.json")["records"]
    ordered_uids = [row["ticket_uid"] for row in dispatch_records]
    initial_failures = {
        uid: _read_json(case_root / uid / "validation.json")
        for uid in ordered_uids
        if (case_root / uid / "case_manifest.json").is_file()
        and _read_json(case_root / uid / "validation.json").get("status")
        in RETRYABLE_STATUSES
    }
    retry_uids = list(initial_failures)
    print(f"SERIAL_RETRY_TARGETS {len(retry_uids)} {retry_uids}", flush=True)
    api_key = ""
    if retry_uids:
        api_key = getpass.getpass("API key (memory only): ").strip()
        if not api_key:
            raise SystemExit("API key cannot be empty")
    run = FrozenRun(args.experiment_root, online_subdir=args.online_subdir)
    retry_results: list[dict[str, Any]] = []
    attempted = 0
    for uid in retry_uids:
        case = prepare_case(run, uid, case_root)
        result: dict[str, Any] = {}
        attempt = 0
        for attempt in range(1, max(1, args.max_attempts) + 1):
            attempted += 1
            result = call_vlm(
                case,
                api_key,
                args.base_url,
                args.model,
                args.timeout_seconds,
                reasoning_effort=args.reasoning_effort,
            )
            print(
                "RETRY",
                uid,
                attempt,
                result.get("status"),
                round(float(result.get("elapsed_seconds", 0.0)), 2),
                flush=True,
            )
            if result.get("status") == "VALID":
                break
            if attempt < args.max_attempts:
                time.sleep(max(0.0, args.retry_delay_seconds))
        retry_results.append(
            {
                "ticket_uid": uid,
                "attempts": attempt,
                "final_status": result.get("status"),
            }
        )

    write_root_html(case_root, ordered_ticket_uids=ordered_uids)
    validation_counts: dict[str, int] = {}
    output_counts: dict[str, int] = {}
    valid_count = 0
    for uid in ordered_uids:
        validation = _read_json(case_root / uid / "validation.json")
        status = str(validation.get("status", "MISSING"))
        validation_counts[status] = validation_counts.get(status, 0) + 1
        if status != "VALID":
            continue
        valid_count += 1
        output = _read_json(case_root / uid / "vlm_output.json")
        decision = f"{output.get('identity_target')}+{output.get('semantic_target')}"
        output_counts[decision] = output_counts.get(decision, 0) + 1

    summary_path = online_root / "run_summary.json"
    summary = _read_json(summary_path)
    initial_calls = int(
        summary.get(
            "vlm_initial_api_calls_attempted",
            summary.get("vlm_api_calls_attempted", len(ordered_uids)),
        )
    )
    previous_retries = int(summary.get("vlm_retry_calls_attempted", 0))
    summary.update(
        {
            "vlm_initial_api_calls_attempted": initial_calls,
            "vlm_retry_calls_attempted": previous_retries + attempted,
            "vlm_api_calls_attempted": initial_calls + previous_retries + attempted,
            "vlm_valid_output_count": valid_count,
            "vlm_validation_status_counts": validation_counts,
            "vlm_output_counts": output_counts,
            "status": (
                "COMPLETED"
                if valid_count == len(ordered_uids)
                else "COMPLETED_WITH_INVALID_VLM_OUTPUTS"
            ),
        }
    )
    write_json(summary_path, summary)
    retry_summary_path = online_root / "vlm_retry_summary.json"
    if retry_uids:
        previous_retry_summary = (
            _read_json(retry_summary_path) if retry_summary_path.is_file() else {}
        )
        rounds = list(previous_retry_summary.get("rounds") or [])
        rounds.append(
            {
                "targets": retry_uids,
                "initial_failures": initial_failures,
                "api_calls_attempted": attempted,
                "results": retry_results,
            }
        )
        all_targets = list(
            dict.fromkeys(
                uid
                for retry_round in rounds
                for uid in retry_round.get("targets", [])
            )
        )
        write_json(
            retry_summary_path,
            {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "serial": True,
                "same_frozen_packets": True,
                "targets": all_targets,
                "api_calls_attempted": sum(
                    int(retry_round.get("api_calls_attempted", 0))
                    for retry_round in rounds
                ),
                "rounds": rounds,
                "final_validation_status_counts": validation_counts,
            },
        )
    print(
        f"FINAL {valid_count}/{len(ordered_uids)} "
        f"validation={validation_counts} outputs={output_counts}",
        flush=True,
    )
    return 0 if valid_count == len(ordered_uids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
