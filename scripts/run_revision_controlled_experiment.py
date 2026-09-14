from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.experiment import (
    build_aggregate_report,
    read_json,
    run_controlled_case,
    select_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled Revision Kernel validation")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--case-config")
    parser.add_argument(
        "--edge-stream",
        help="Frozen make_edges-compatible stream directory or manifest.json",
    )
    parser.add_argument(
        "--require-relation-change",
        action="store_true",
        help="Select cases whose corrupted branch changes edge topology or support",
    )
    parser.add_argument(
        "--edge-candidate-limit",
        type=int,
        default=100,
        help="Maximum candidates screened per failure type for relation impact",
    )
    parser.add_argument(
        "--failure-type",
        choices=["FALSE_SPLIT", "WRONG_MEMBERSHIP", "FALSE_MERGE", "ALL"],
        default="ALL",
    )
    parser.add_argument("--skip-global", action="store_true")
    args = parser.parse_args()

    if args.case_config:
        cases = [read_json(args.case_config)]
    else:
        failures = (
            ["FALSE_SPLIT", "WRONG_MEMBERSHIP", "FALSE_MERGE"]
            if args.failure_type == "ALL"
            else [args.failure_type]
        )
        cases = select_cases(
            args.base_run,
            failures,
            edge_stream_root=args.edge_stream,
            require_relation_change=args.require_relation_change,
            candidate_limit=args.edge_candidate_limit,
        )
    results = []
    for case in cases:
        result = run_controlled_case(
            base_run=args.base_run,
            output_root=args.output_root,
            case=case,
            run_global=not args.skip_global,
            edge_stream_root=args.edge_stream,
        )
        results.append(
            {"case_uid": case["case_uid"], "pass": result["pass"]}
        )
    aggregate = build_aggregate_report(args.output_root)
    print(json.dumps({"cases": results, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
