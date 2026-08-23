from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.experiment import read_json, write_json
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.replay import CounterfactualReplayEngine


def signature(state):
    return {
        "membership": state["membership"],
        "objects": sorted(
            (
                {
                    key: row[key]
                    for key in (
                        "entity_uid",
                        "member_observation_uids",
                        "n_points",
                        "bbox_center",
                        "bbox_extent",
                        "point_digest",
                    )
                }
                for row in state["objects"]
            ),
            key=lambda row: row["entity_uid"],
        ),
        "decisions": [
            {
                key: row[key]
                for key in (
                    "obs_uid",
                    "natural_match",
                    "applied_match",
                    "desired_entity_uid",
                    "forced",
                )
            }
            for row in state["decision_trace"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat local replay and compare exact state")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = CounterfactualReplayEngine(ProvenanceIndex(args.base_run))
    case = read_json(args.case)
    first = engine.replay_local(case, branch="repaired")
    second = engine.replay_local(case, branch="repaired")
    first_signature = signature(first)
    second_signature = signature(second)
    output = {
        "pass": first_signature == second_signature,
        "case_uid": case["case_uid"],
        "first_runtime_ms": first["runtime_ms"],
        "second_runtime_ms": second["runtime_ms"],
        "membership_equal": first["membership"] == second["membership"],
        "object_state_equal": first_signature["objects"] == second_signature["objects"],
        "decision_trace_equal": first_signature["decisions"] == second_signature["decisions"],
        "replayed_observations": first["replayed_observations"],
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["pass"] else 1)


if __name__ == "__main__":
    main()
