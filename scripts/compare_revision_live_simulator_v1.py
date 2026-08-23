from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.constraints import ReplayMode
from conceptgraph.revision.evaluate import (
    symmetric_membership_metrics as _symmetric_membership_metrics,
)
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.replay import CounterfactualReplayEngine
from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine
from scripts.validate_revision_v1_global_clean_parity import _raw_object_parity


def _read(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def observation_key(obs_uid: str) -> str:
    value = str(obs_uid)
    marker = value.rfind("_f")
    return value[marker:] if marker >= 0 else value


def _canonical_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    result["membership"] = {
        str(entity_uid): [observation_key(item) for item in members]
        for entity_uid, members in (state.get("membership") or {}).items()
    }
    result["objects"] = [
        {
            **row,
            "member_observation_uids": [
                observation_key(item)
                for item in row.get("member_observation_uids") or ()
            ],
        }
        for row in state.get("objects") or ()
    ]
    return result


def _membership_observation_scope(*states: Mapping[str, Any]) -> set[str]:
    """Use the union so neither branch can hide extra observations from scoring."""

    return {
        str(item)
        for state in states
        for members in (state.get("membership") or {}).values()
        for item in members
    }


def _strict_symmetric_membership_metrics(
    live_membership: Mapping[str, list[str]],
    simulator_membership: Mapping[str, list[str]],
) -> dict[str, Any]:
    result = _symmetric_membership_metrics(live_membership, simulator_membership)
    result["comparison_scope"] = "UNION_OF_LIVE_AND_SIMULATOR_OBSERVATIONS"
    result["comparison_observation_count"] = result["observation_count"]
    result["missing_in_live"] = result["missing_from_first"]
    result["missing_in_simulator"] = result["missing_from_second"]
    result["live_duplicate_observations"] = result[
        "first_duplicate_observations"
    ]
    result["simulator_duplicate_observations"] = result[
        "second_duplicate_observations"
    ]
    result["live_observation_count"] = result["first_observation_count"]
    result["simulator_observation_count"] = result[
        "second_observation_count"
    ]
    return result


def _decision_fidelity(
    live: ProvenanceIndex,
    simulator_trace: list[Mapping[str, Any]],
) -> dict[str, Any]:
    live_by_obs = {
        observation_key(str(row["obs_uid"])): row for row in live.association_rows
    }
    simulated_by_obs = {
        observation_key(str(row["obs_uid"])): row for row in simulator_trace
    }
    missing_live = sorted(set(simulated_by_obs) - set(live_by_obs))
    missing_simulator = sorted(set(live_by_obs) - set(simulated_by_obs))
    mismatches = []
    decision_kind_mismatch_count = 0
    target_mismatch_count = 0
    for obs_uid in sorted(set(live_by_obs) & set(simulated_by_obs)):
        live_row = live_by_obs[obs_uid]
        simulator_row = simulated_by_obs[obs_uid]
        live_create = str(live_row.get("decision")) == "CREATE_OBJECT"
        simulator_create = simulator_row.get("applied_match") is None
        differences: dict[str, Any] = {}
        if live_create != simulator_create:
            decision_kind_mismatch_count += 1
            differences["decision_kind"] = {
                "live": live_row.get("decision"),
                "simulator_applied_match": simulator_row.get("applied_match"),
            }
        elif not live_create:
            version_uid = live_row.get("target_object_version_before")
            live_origin = None
            if version_uid in live.object_versions:
                version = live.get_object_version(str(version_uid))
                members = list(version.get("member_observation_uids") or ())
                live_origin = version.get("origin_observation_uid") or (
                    members[0] if members else None
                )
            simulator_origin = simulator_row.get("applied_target_origin_obs_uid")
            if observation_key(str(live_origin or "")) != observation_key(
                str(simulator_origin or "")
            ):
                target_mismatch_count += 1
                differences["target_origin"] = {
                    "live": live_origin,
                    "simulator": simulator_origin,
                }
        if differences:
            mismatches.append(
                {
                    "observation_key": obs_uid,
                    "live_event_uid": live_row.get("event_uid"),
                    "live_decision": live_row.get("decision"),
                    "simulator_event_uid": simulator_row.get("event_uid"),
                    "simulator_applied_match": simulator_row.get("applied_match"),
                    "simulator_natural_match": simulator_row.get("natural_match"),
                    "differences": differences,
                    "simulator_candidates": simulator_row.get("natural_candidates"),
                }
            )
    return {
        "pass": not missing_live and not missing_simulator and not mismatches,
        "compared_observation_count": len(set(live_by_obs) & set(simulated_by_obs)),
        "missing_in_live": missing_live,
        "missing_in_simulator": missing_simulator,
        "decision_kind_mismatch_count": decision_kind_mismatch_count,
        "target_origin_mismatch_count": target_mismatch_count,
        "first_mismatches": mismatches[:20],
    }


def _live_postprocess_counts(live_run: str | Path) -> dict[str, int]:
    path = Path(live_run) / "parity_trace.json"
    rows = _read(path)
    return {
        name: sum(bool((row.get(name) or {}).get("executed")) for row in rows)
        for name in ("denoise", "filter", "merge")
    }


def _corruption_records(live_run: str | Path, case_uid: str) -> list[dict[str, Any]]:
    path = Path(live_run) / "revision" / str(case_uid) / "corruption_events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compare(
    *,
    base_run: str | Path,
    live_run: str | Path,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    base = ProvenanceIndex(base_run)
    live = ProvenanceIndex(live_run)
    simulator, simulator_objects = SparseCounterfactualReplayEngine(
        base
    ).replay_global_with_objects(
        mode=ReplayMode.TEMPORAL_CORRUPTION,
        corruption_plan=case["corruption_plan"],
    )
    live_state = CounterfactualReplayEngine(live).clean_state()
    live_pcd_paths = sorted(Path(live_run).glob("pcd_*.pkl.gz"))
    if len(live_pcd_paths) != 1:
        raise FileNotFoundError(
            f"expected one frozen live map, found {len(live_pcd_paths)}"
        )
    with gzip.open(live_pcd_paths[0], "rb") as handle:
        live_payload = pickle.load(handle)
    canonical_simulator = _canonical_state(simulator)
    canonical_live = _canonical_state(live_state)
    all_observations = _membership_observation_scope(
        canonical_live, canonical_simulator
    )
    membership = _strict_symmetric_membership_metrics(
        canonical_live["membership"],
        canonical_simulator["membership"],
    )
    if membership["observation_count"] != len(all_observations):
        raise RuntimeError("symmetric membership scorer did not preserve union scope")
    objects = _raw_object_parity(
        live_payload["objects"],
        simulator_objects,
        member_normalizer=lambda item: observation_key(str(item)),
    )
    decisions = _decision_fidelity(live, simulator["decision_trace"])
    live_postprocess = _live_postprocess_counts(live_run)
    simulator_postprocess = dict(simulator["postprocess_counts"])
    records = _corruption_records(live_run, str(case["case_uid"]))
    simulator_interventions = [
        row
        for row in simulator["decision_trace"]
        if row.get("intervention_overrode_natural")
    ]
    injection = {
        "pass": len(records) == 1
        and len(simulator_interventions) == 1
        and observation_key(str(records[0].get("obs_uid", "")))
        == observation_key(str(case["obs_uid"])),
        "live_record_count": len(records),
        "simulator_intervention_count": len(simulator_interventions),
        "live_records": records,
        "simulator_interventions": simulator_interventions,
    }
    checks = {
        "single_injection_exact": injection["pass"],
        "downstream_decision_kind_exact": decisions["pass"],
        "final_membership_partition_exact": membership["partition_exact"],
        "final_object_payload_exact": objects["pass"],
        "postprocess_counts_exact": live_postprocess == simulator_postprocess,
    }
    return {
        "schema_version": "1.0.0",
        "case_uid": case["case_uid"],
        "base_run": str(Path(base_run).resolve()),
        "live_run": str(Path(live_run).resolve()),
        "pass": all(checks.values()),
        "checks": checks,
        "injection": injection,
        "decisions": decisions,
        "membership": membership,
        "objects": objects,
        "postprocess": {
            "live": live_postprocess,
            "simulator": simulator_postprocess,
        },
        "relation": {
            "informative": False,
            "reason": "live fidelity runs use make_edges=false; relation is evaluated separately",
        },
        "simulator_runtime_ms": simulator["runtime_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare an independent live one-event corruption with V1 simulation"
    )
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--live-run", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    case = _read(args.case)
    result = compare(base_run=args.base_run, live_run=args.live_run, case=case)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
