from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.experiment import read_json, write_json
from conceptgraph.revision.vlm import normalize_incident_constraint


EXPECTED_ACTION = {
    "FALSE_SPLIT": "SAME_INSTANCE",
    "WRONG_MEMBERSHIP": "MOVE_OBSERVATION",
    "FALSE_MERGE": "SEPARATE_MEMBER_GROUPS",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and evaluate blinded VLM constraints")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    result_path = root / "vlm_constraint_results.json"
    results = read_json(result_path)
    cases = {
        path.parent.name: read_json(path) for path in sorted(root.glob("*/case.json"))
    }
    compiled_votes = []
    for vote in results["votes"]:
        case = cases[str(vote["case_uid"])]
        observed = (
            "CREATE"
            if case["corruption_plan"]["corruption_type"] == "FORCE_CREATE"
            else "ASSOCIATE"
        )
        compiled = normalize_incident_constraint(
            vote["constraint"], observed_current_decision=observed
        )
        compiled_votes.append(
            {
                "case_uid": vote["case_uid"],
                "vote_index": vote["vote_index"],
                "compiled_constraint": compiled,
            }
        )
    evaluation = {}
    for case_uid, case in cases.items():
        votes = [row for row in compiled_votes if row["case_uid"] == case_uid]
        counts = Counter(row["compiled_constraint"]["action"] for row in votes)
        predicted = counts.most_common(1)[0][0]
        expected = EXPECTED_ACTION[str(case["failure_type"])]
        confidence = sum(
            float(row["compiled_constraint"].get("confidence", 0.0))
            for row in votes
            if row["compiled_constraint"]["action"] == predicted
        ) / counts[predicted]
        evidence_complete = all(
            bool(row["compiled_constraint"].get("evidence_image_ids")) for row in votes
        )
        automatic_commit_allowed = (
            predicted != "DEFER"
            and confidence >= 0.85
            and evidence_complete
            and counts[predicted] == len(votes)
        )
        evaluation[case_uid] = {
            "failure_type": case["failure_type"],
            "compiled_action": predicted,
            "expected_action": expected,
            "correct": predicted == expected,
            "safe_abstention": predicted == "DEFER",
            "vote_counts": dict(counts),
            "mean_confidence": confidence,
            "evidence_complete": evidence_complete,
            "automatic_commit_allowed_by_blind_gate": automatic_commit_allowed,
            "oracle_used_by_gate": False,
        }
    results["compiled_votes"] = compiled_votes
    results["compiled_evaluation"] = evaluation
    results["raw_exact_action_accuracy"] = results.get("accuracy")
    results["compiled_action_accuracy"] = sum(row["correct"] for row in evaluation.values()) / len(evaluation)
    results["safe_abstention_rate"] = sum(row["safe_abstention"] for row in evaluation.values()) / len(evaluation)
    results["unsafe_committed_case_count_posthoc"] = sum(
        row["automatic_commit_allowed_by_blind_gate"] and not row["correct"]
        for row in evaluation.values()
    )
    write_json(result_path, results)
    print(
        json.dumps(
            {
                "compiled_evaluation": evaluation,
                "compiled_action_accuracy": results["compiled_action_accuracy"],
                "safe_abstention_rate": results["safe_abstention_rate"],
                "unsafe_committed_case_count_posthoc": results[
                    "unsafe_committed_case_count_posthoc"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
