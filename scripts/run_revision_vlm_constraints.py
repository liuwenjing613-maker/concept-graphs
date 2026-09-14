from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.experiment import read_json, write_json
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.vlm import (
    VLMIncidentBuilder,
    aggregate_votes,
    run_parallel_votes,
)


EXPECTED_ACTION = {
    "FALSE_SPLIT": "SAME_INSTANCE",
    "WRONG_MEMBERSHIP": "MOVE_OBSERVATION",
    "FALSE_MERGE": "SEPARATE_MEMBER_GROUPS",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM constraint votes without key persistence")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--key-count", type=int, default=5)
    args = parser.parse_args()
    if args.key_count != 5:
        raise ValueError("the controlled matrix is fixed to five independent votes")

    keys = [getpass.getpass(f"API key {index + 1}/5: ") for index in range(5)]
    provenance = ProvenanceIndex(args.base_run)
    builder = VLMIncidentBuilder(provenance)
    root = Path(args.run_root)
    cases = {
        path.parent.name: read_json(path) for path in sorted(root.glob("*/case.json"))
    }
    ordered = sorted(cases.values(), key=lambda case: str(case["failure_type"]))
    if len(ordered) != 3:
        raise RuntimeError("expected exactly three controlled cases")
    evidence = {
        str(case["case_uid"]): builder.build(
            case, root / str(case["case_uid"]) / "vlm" / "evidence_images"
        )
        for case in ordered
    }
    # Give the two harder two-entity incidents an independent second vote.
    schedule = [ordered[0], ordered[1], ordered[2], ordered[0], ordered[2]]
    jobs = [(str(case["case_uid"]), evidence[str(case["case_uid"])]) for case in schedule]
    votes = run_parallel_votes(
        jobs=jobs,
        api_keys=keys,
        base_url=args.base_url,
        model=args.model,
    )
    keys[:] = [""] * len(keys)
    aggregated = aggregate_votes(votes)
    evaluation = {}
    for case_uid, result in aggregated.items():
        case = cases[case_uid]
        expected = EXPECTED_ACTION[str(case["failure_type"])]
        evaluation[case_uid] = {
            "predicted_action": result["action"],
            "expected_action": expected,
            "correct": result["action"] == expected,
            "oracle_used_during_inference": False,
        }
    output = {
        "method": "VLM typed-constraint generator",
        "model": args.model,
        "base_url": args.base_url,
        "api_keys_persisted": False,
        "source_labels_allowed_during_inference": False,
        "votes": votes,
        "aggregate": aggregated,
        "posthoc_oracle_evaluation": evaluation,
        "accuracy": sum(item["correct"] for item in evaluation.values()) / len(evaluation),
    }
    write_json(root / "vlm_constraint_results.json", output)
    print(json.dumps({"aggregate": aggregated, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
