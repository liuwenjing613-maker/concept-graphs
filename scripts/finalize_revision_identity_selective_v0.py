#!/usr/bin/env python3
"""Finalize anonymous critic results into a single-threshold selective decision."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from conceptgraph.revision.autonomous_identity import signed_pairwise_preference
from conceptgraph.revision.candidate_verifier import CandidateEvidenceScore
from conceptgraph.revision.evidence_split import sha256_file
from conceptgraph.revision.selective_commit import (
    CalibrationArtifact,
    SelectiveCandidate,
    decide_selective_commit,
)


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _round_for_request(request_uid: str, base_request_uid: str) -> int | None:
    if request_uid == base_request_uid:
        return 0
    match = re.fullmatch(
        re.escape(base_request_uid) + r"_EVIDENCE_ROUND_(\d+)", request_uid
    )
    return int(match.group(1)) if match else None


def _load_pass_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    results = {}
    for path in paths:
        aggregate = _read(path.resolve())
        for row in aggregate.get("results") or ():
            if row.get("status") != "PASS":
                continue
            request_uid = str(row["request_uid"])
            if request_uid in results:
                raise ValueError(f"duplicate successful critic result: {request_uid}")
            results[request_uid] = dict(row)
    return results


def _latest_complete_pair(
    *,
    base_requests: list[Mapping[str, Any]],
    critic_results: Mapping[str, Mapping[str, Any]],
) -> tuple[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    if len(base_requests) != 2:
        raise ValueError("each pair must have exactly two order-swapped requests")
    by_round: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for base in base_requests:
        base_uid = str(base["request_uid"])
        for request_uid, result in critic_results.items():
            round_index = _round_for_request(request_uid, base_uid)
            if round_index is not None:
                by_round.setdefault(round_index, []).append((base, result))
    complete = {
        round_index: rows
        for round_index, rows in by_round.items()
        if len(rows) == len(base_requests)
        and {int(base["order_index"]) for base, _ in rows} == {0, 1}
    }
    if not complete:
        raise ValueError(
            "no evidence round has both successful order-swapped critic results"
        )
    latest = max(complete)
    return latest, sorted(
        complete[latest], key=lambda item: int(item[0]["order_index"])
    )


def _finalize_case(
    *,
    protocol: Mapping[str, Any],
    case_row: Mapping[str, Any],
    critic_results: Mapping[str, Mapping[str, Any]],
    calibration: CalibrationArtifact,
) -> dict[str, Any]:
    case_uid = str(case_row["case_uid"])
    result_path = Path(str(case_row["result_path"])).resolve()
    frozen = _read(result_path)
    case_dir = result_path.parent
    private_path = case_dir / "execution.private.json"
    private = _read(private_path)
    if frozen.get("status") == "NO_DISTINCT_EXECUTABLE_REPAIR":
        decision = decide_selective_commit(
            incident_uid=case_uid,
            candidates=[],
            calibration=calibration,
        )
        return {
            "schema_version": "1.0.0",
            "case_uid": case_uid,
            "scene_id": str(case_row["scene_id"]),
            "shadow_status": "SHADOW_NO_DISTINCT_REPAIR",
            "shadow_selected_partition_hash": str(private["noop_partition_hash"]),
            "shadow_candidate_uid": None,
            "shadow_replay_constraint": None,
            "production_selective_decision": decision,
            "pair_audits": [],
            "runtime_human_or_gold_loaded": False,
            "production_commit_permitted": False,
            "frozen_case_result_path": str(result_path),
            "frozen_case_result_sha256": sha256_file(result_path),
            "execution_private_path": str(private_path),
            "execution_private_sha256": sha256_file(private_path),
        }
    noop_hash = str(private["noop_partition_hash"])
    base_requests = [
        row
        for row in protocol.get("critic_requests") or ()
        if str(row.get("case_uid")) == case_uid
    ]
    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for row in base_requests:
        by_pair.setdefault(str(row["pair_uid"]), []).append(row)

    score_by_partition = {
        str(row["partition_hash"]): row
        for row in frozen.get("primary_candidate_scores") or ()
    }
    implementation_by_partition = {}
    for row in private.get("candidate_replays") or ():
        partition = str(row["partition_hash"])
        current = implementation_by_partition.get(partition)
        if current is None or (
            not current["runtime_validity"]["valid"]
            and row["runtime_validity"]["valid"]
        ):
            implementation_by_partition[partition] = row

    candidates = []
    pair_audits = []
    for pair_uid, requests in sorted(by_pair.items()):
        evidence_round, rows = _latest_complete_pair(
            base_requests=requests, critic_results=critic_results
        )
        parent_uid = str(rows[0][0]["request_uid"])
        mapping = private["critic_state_mappings"][parent_uid]
        candidate_hash = str(mapping["candidate_partition_hash"])
        preferred_hashes = []
        critic_audits = []
        for base, result in rows:
            base_uid = str(base["request_uid"])
            label_to_hash = private["critic_state_mappings"][base_uid][
                "label_to_partition_hash"
            ]
            critic = result["response"]["critic"]
            preferred_state = str(critic["preferred_state"])
            preferred_hash = (
                None
                if preferred_state == "DEFER"
                else str(label_to_hash[preferred_state])
            )
            preferred_hashes.append(preferred_hash)
            critic_audits.append(
                {
                    "order_index": int(base["order_index"]),
                    "request_uid": str(result["request_uid"]),
                    "preferred_state": preferred_state,
                    "preferred_partition_hash": preferred_hash,
                    "cited_evidence_ids": list(critic["cited_evidence_ids"]),
                    "confidence_diagnostic_only": critic["confidence_diagnostic_only"],
                    "confidence_raw_diagnostic": critic.get(
                        "confidence_raw_diagnostic"
                    ),
                    "reason": str(critic["reason"]),
                    "needed_evidence": list(critic["needed_evidence"]),
                }
            )
        pairwise = signed_pairwise_preference(
            preferred_partition_hashes=preferred_hashes,
            candidate_partition_hash=candidate_hash,
            noop_partition_hash=noop_hash,
        )
        raw_score = score_by_partition[candidate_hash]
        noop_primary = float(raw_score["primary_score"]) - float(
            raw_score["score_advantage_over_noop"]
        )
        score = CandidateEvidenceScore.build(
            incident_uid=case_uid,
            candidate_uid=str(raw_score["candidate_uid"]),
            capability="IDENTITY",
            primary_statistic=str(raw_score["primary_statistic"]),
            primary_score=float(raw_score["primary_score"]),
            noop_primary_score=noop_primary,
            valid=bool(raw_score["valid"]),
            verification_observation_count=int(
                raw_score["verification_observation_count"]
            ),
            diagnostics={
                **dict(raw_score["diagnostics"]),
                "source_score_uid": str(raw_score["score_uid"]),
                "critic_evidence_round": evidence_round,
                "order_swapped_request_count": len(rows),
            },
            vlm_pairwise_preference=pairwise,
        )
        implementation = implementation_by_partition[candidate_hash]
        candidates.append(
            SelectiveCandidate(
                score=score,
                candidate_constraint=dict(implementation["constraint"]),
            )
        )
        pair_audits.append(
            {
                "pair_uid": pair_uid,
                "candidate_partition_hash": candidate_hash,
                "noop_partition_hash": noop_hash,
                "evidence_round_used": evidence_round,
                "vlm_pairwise_preference": pairwise,
                "order_swapped_critics": critic_audits,
                "candidate_score": score.as_dict(),
            }
        )

    decision = decide_selective_commit(
        incident_uid=case_uid,
        candidates=candidates,
        calibration=calibration,
    )
    fully_supported = [
        audit for audit in pair_audits if audit["vlm_pairwise_preference"] == 1.0
    ]
    fully_rejected = [
        audit for audit in pair_audits if audit["vlm_pairwise_preference"] == -1.0
    ]
    if len(fully_supported) == 1:
        shadow_status = "SHADOW_REPAIR_RECOMMENDED"
        selected_partition = fully_supported[0]["candidate_partition_hash"]
        selected = implementation_by_partition[selected_partition]
        shadow_candidate_uid = str(
            score_by_partition[selected_partition]["candidate_uid"]
        )
        shadow_constraint = dict(selected["constraint"])
    elif len(fully_rejected) == len(pair_audits):
        shadow_status = "SHADOW_NOOP_PREFERRED"
        selected_partition = noop_hash
        shadow_candidate_uid = None
        shadow_constraint = None
    else:
        shadow_status = "SHADOW_INCONCLUSIVE"
        selected_partition = None
        shadow_candidate_uid = None
        shadow_constraint = None
    return {
        "schema_version": "1.0.0",
        "case_uid": case_uid,
        "scene_id": str(case_row["scene_id"]),
        "shadow_status": shadow_status,
        "shadow_selected_partition_hash": selected_partition,
        "shadow_candidate_uid": shadow_candidate_uid,
        "shadow_replay_constraint": shadow_constraint,
        "production_selective_decision": decision,
        "pair_audits": pair_audits,
        "runtime_human_or_gold_loaded": False,
        "production_commit_permitted": decision["decision"] == "COMMIT",
        "frozen_case_result_path": str(result_path),
        "frozen_case_result_sha256": sha256_file(result_path),
        "execution_private_path": str(private_path),
        "execution_private_sha256": sha256_file(private_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", required=True, type=Path)
    parser.add_argument(
        "--critic-results",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    protocol_path = args.freeze_protocol.resolve()
    protocol = _read(protocol_path)
    if protocol.get("runtime_human_or_gold_loaded") is not False:
        raise ValueError("freeze protocol did not pass runtime oracle isolation")
    calibration_path = args.calibration.resolve()
    calibration = CalibrationArtifact.from_mapping(_read(calibration_path))
    critic_paths = [path.resolve() for path in args.critic_results]
    critic_results = _load_pass_results(critic_paths)
    cases = [
        _finalize_case(
            protocol=protocol,
            case_row=case,
            critic_results=critic_results,
            calibration=calibration,
        )
        for case in protocol.get("cases") or ()
    ]
    aggregate = {
        "schema_version": "1.0.0",
        "role": "DEVELOPMENT_SHADOW_WITH_FAIL_CLOSED_PRODUCTION_DECISION",
        "freeze_protocol_path": str(protocol_path),
        "freeze_protocol_sha256": sha256_file(protocol_path),
        "critic_result_paths": [str(path) for path in critic_paths],
        "critic_result_sha256": {str(path): sha256_file(path) for path in critic_paths},
        "calibration": calibration.as_dict(),
        "case_count": len(cases),
        "shadow_repair_recommended_count": sum(
            row["shadow_status"] == "SHADOW_REPAIR_RECOMMENDED" for row in cases
        ),
        "shadow_noop_preferred_count": sum(
            row["shadow_status"] == "SHADOW_NOOP_PREFERRED" for row in cases
        ),
        "shadow_no_distinct_repair_count": sum(
            row["shadow_status"] == "SHADOW_NO_DISTINCT_REPAIR" for row in cases
        ),
        "shadow_inconclusive_count": sum(
            row["shadow_status"] == "SHADOW_INCONCLUSIVE" for row in cases
        ),
        "production_commit_count": sum(
            row["production_selective_decision"]["decision"] == "COMMIT"
            for row in cases
        ),
        "production_defer_count": sum(
            row["production_selective_decision"]["decision"] == "DEFER" for row in cases
        ),
        "runtime_human_or_gold_loaded": False,
        "semantic_commit_threshold_count": 1,
        "cases": cases,
    }
    _write(args.output.resolve(), aggregate)
    print(
        json.dumps(
            {
                key: aggregate[key]
                for key in (
                    "case_count",
                    "shadow_repair_recommended_count",
                    "shadow_noop_preferred_count",
                    "shadow_no_distinct_repair_count",
                    "shadow_inconclusive_count",
                    "production_commit_count",
                    "production_defer_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
