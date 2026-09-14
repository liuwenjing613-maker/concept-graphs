from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.cases import canonical_obs_key
from conceptgraph.revision.evaluate import membership_metrics
from conceptgraph.revision.experiment import read_json, write_json
from conceptgraph.revision.index import ProvenanceIndex


def stable_membership(index: ProvenanceIndex) -> dict[str, list[str]]:
    return {
        str(row["object_uid"]): [
            canonical_obs_key(str(obs)) for obs in row.get("member_observation_uids") or ()
        ]
        for row in index.final_membership
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare live controlled corruption to clean")
    parser.add_argument("--clean-run", required=True)
    parser.add_argument("--corrupted-run", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean = ProvenanceIndex(args.clean_run)
    corrupted = ProvenanceIndex(args.corrupted_run)
    case = read_json(args.case)
    clean_members = stable_membership(clean)
    corrupt_members = stable_membership(corrupted)
    affected = {
        canonical_obs_key(str(obs))
        for members in case["affected_clean_groups"].values()
        for obs in members
    }
    metrics = membership_metrics(
        clean_members, corrupt_members, observation_scope=affected
    )
    event_path = (
        corrupted.experiment_root
        / "revision"
        / str(case["case_uid"])
        / "corruption_events.jsonl"
    )
    events = [json.loads(line) for line in event_path.open(encoding="utf-8") if line.strip()]
    output = {
        "pass": len(events) == 1 and metrics["member_f1"] < 1.0,
        "case_uid": case["case_uid"],
        "injection_count": len(events),
        "injection": events[0] if events else None,
        "clean_integrity": clean.validate(),
        "corrupted_integrity": corrupted.validate(),
        "clean_object_count": len(clean.final_membership),
        "corrupted_object_count": len(corrupted.final_membership),
        "affected_membership": metrics,
        "run_independent_observation_keys": True,
        "clean_source_hashes": clean.source_hashes(),
        "corrupted_source_hashes": corrupted.source_hashes(),
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["pass"] else 1)


if __name__ == "__main__":
    main()
