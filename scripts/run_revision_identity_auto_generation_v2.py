#!/usr/bin/env python3
"""Run five-vote identity generation from pre-frozen evidence bundles."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from conceptgraph.revision.auto_constraints import (
    IncidentBinding,
    aggregate_candidate_votes,
    canonicalize_vote,
    compile_blind_candidate,
)
from conceptgraph.revision.vlm import VLMIncidentEvidence, run_parallel_votes


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schedule(case_uids: list[str]) -> list[list[str]]:
    if len(case_uids) != 3:
        raise ValueError("balanced schedule requires exactly three cases")
    first, second, third = case_uids
    rounds = [
        [first, second, third, first, second],
        [third, first, second, third, first],
        [second, third, first, second, third],
    ]
    counts = Counter(uid for round_cases in rounds for uid in round_cases)
    if any(counts[uid] != 5 for uid in case_uids):
        raise AssertionError(f"unbalanced vote schedule: {counts}")
    return rounds


def _load_frozen_case(row: dict[str, Any]):
    request_path = Path(str(row["request_path"])).resolve()
    if _sha256(request_path) != str(row["request_sha256"]):
        raise ValueError(f"request drift: {request_path}")
    request = _read(request_path)
    if _text_sha256(str(request["prompt"])) != str(request["prompt_sha256"]):
        raise ValueError(f"prompt drift: {request_path}")
    image_paths = []
    for image in request.get("images") or ():
        path = Path(str(image["path"])).resolve()
        if not path.is_file() or _sha256(path) != str(image["sha256"]):
            raise ValueError(f"image drift: {path}")
        image_paths.append(path)
    binding_path = Path(str(request["binding_private_path"])).resolve()
    if _sha256(binding_path) != str(row["binding_private_sha256"]):
        raise ValueError(f"binding drift: {binding_path}")
    binding = IncidentBinding.from_mapping(_read(binding_path))
    evidence = VLMIncidentEvidence(
        incident_uid=str(request["blind_case_uid"]),
        prompt=str(request["prompt"]),
        image_paths=tuple(image_paths),
        image_manifest=tuple(request["images"]),
        system_prompt=(
            "You perform conservative physical-instance decisions for a 3D scene "
            "graph. Use only finite aliases and cited frozen evidence. Current-map "
            "detector output is observational evidence, not a correct endpoint. "
            "Never use semantic class alone as identity proof. Return one JSON object."
        ),
    )
    return request, binding, evidence


def _majority_diagnostic(votes: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = []
    errors = []
    for index, row in enumerate(votes):
        try:
            parsed.append(canonicalize_vote(row.get("constraint", row)))
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"vote_{index}:{type(exc).__name__}:{exc}")
    counts = Counter(row["signature"] for row in parsed)
    signature, count = counts.most_common(1)[0] if counts else (None, 0)
    selected = next(
        (row for row in parsed if row["signature"] == signature),
        None,
    )
    return {
        "vote_count": len(votes),
        "valid_vote_count": len(parsed),
        "signature_counts": dict(sorted(counts.items())),
        "majority_count": count,
        "four_of_five_consensus": count >= 4 and len(votes) == 5,
        "selected_majority_proposal": selected,
        "parse_errors": errors,
        "diagnostic_only_not_compiled": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--key-count", type=int, default=5)
    args = parser.parse_args()
    if args.key_count != 5:
        raise ValueError("the frozen design requires exactly five credential slots")

    manifest_path = args.evidence_manifest.resolve()
    manifest = _read(manifest_path)
    if (
        manifest.get("role") != "DEVELOPMENT_NOT_HOLDOUT"
        or not manifest.get("frozen_before_model_responses")
        or int(manifest.get("case_count", -1)) != 3
    ):
        raise ValueError("evidence manifest is not a frozen three-case development set")
    cases = list(manifest["cases"])
    case_uids = [str(row["blind_case_uid"]) for row in cases]
    if len(set(case_uids)) != 3:
        raise ValueError("blind case UIDs must be unique")

    requests = {}
    bindings = {}
    evidence = {}
    for row in cases:
        uid = str(row["blind_case_uid"])
        requests[uid], bindings[uid], evidence[uid] = _load_frozen_case(row)

    schedule = _schedule(case_uids)
    output_root = args.output_root.resolve()
    result_path = output_root / "identity_generation_v2.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite: {result_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "schema_version": "1.0.0",
        "experiment_uid": "identity_auto_generation_v2_20260824",
        "role": "DEVELOPMENT_NOT_HOLDOUT",
        "evidence_manifest_path": str(manifest_path),
        "evidence_manifest_sha256": _sha256(manifest_path),
        "frozen_before_responses": True,
        "case_count": 3,
        "votes_per_case": 5,
        "total_vote_count": 15,
        "parallel_credential_slots": 5,
        "credential_slots_are_not_independent_models": True,
        "model": args.model,
        "base_url": args.base_url,
        "api_keys_persisted": False,
        "gold_loaded_by_generator": False,
        "strict_compiler_requires_five_of_five_structural_unanimity": True,
        "four_of_five_is_diagnostic_only": True,
        "schedule": [
            {
                "round_index": index,
                "credential_slot_to_case": {
                    str(slot): uid for slot, uid in enumerate(round_cases)
                },
            }
            for index, round_cases in enumerate(schedule)
        ],
        "request_audit": [
            {
                "blind_case_uid": uid,
                "prompt_sha256": requests[uid]["prompt_sha256"],
                "allowed_evidence_image_ids": requests[uid][
                    "allowed_evidence_image_ids"
                ],
                "bundle_uid": requests[uid]["bundle_uid"],
                "source_case_uid_not_in_prompt": requests[uid][
                    "source_case_uid_not_in_prompt"
                ],
                "human_verdict_not_in_prompt": requests[uid][
                    "human_verdict_not_in_prompt"
                ],
                "expected_action_not_in_prompt": requests[uid][
                    "expected_action_not_in_prompt"
                ],
                "repaired_ownership_not_in_prompt": requests[uid][
                    "repaired_ownership_not_in_prompt"
                ],
            }
            for uid in case_uids
        ],
    }
    protocol_path = output_root / "inference_protocol.frozen.json"
    _write(protocol_path, protocol)
    protocol_sha256 = _sha256(protocol_path)

    keys: list[str] = []
    votes: list[dict[str, Any]] = []
    try:
        keys = [
            getpass.getpass(f"API key {index + 1}/{args.key_count}: ")
            for index in range(args.key_count)
        ]
        if not all(keys):
            raise ValueError("all five in-memory API keys are required")
        for round_index, round_cases in enumerate(schedule):
            jobs = [(uid, evidence[uid]) for uid in round_cases]
            round_votes = run_parallel_votes(
                jobs=jobs,
                api_keys=keys,
                base_url=args.base_url,
                model=args.model,
            )
            for row in round_votes:
                row["round_index"] = round_index
                row["credential_slot"] = int(row.pop("vote_index"))
                votes.append(row)
    finally:
        keys[:] = [""] * len(keys)

    strict_aggregate = {}
    majority_diagnostic = {}
    compiled = {}
    for uid in case_uids:
        case_votes = [row for row in votes if row["case_uid"] == uid]
        allowed = requests[uid]["allowed_evidence_image_ids"]
        strict_aggregate[uid] = aggregate_candidate_votes(
            case_votes,
            allowed_evidence_ids=allowed,
            minimum_votes=5,
        )
        majority_diagnostic[uid] = _majority_diagnostic(case_votes)
        compiled[uid] = compile_blind_candidate(
            strict_aggregate[uid],
            bindings[uid],
        )

    result = {
        "schema_version": "1.0.0",
        "experiment_uid": protocol["experiment_uid"],
        "inference_protocol_path": str(protocol_path),
        "inference_protocol_sha256": protocol_sha256,
        "inference_protocol_frozen_before_responses": True,
        "api_keys_persisted": False,
        "gold_loaded_by_generator": False,
        "vote_count": len(votes),
        "votes": votes,
        "strict_aggregate": strict_aggregate,
        "majority_diagnostic": majority_diagnostic,
        "compiled_candidates": compiled,
    }
    _write(result_path, result)
    print(
        json.dumps(
            {
                "status": "PASS",
                "result_path": str(result_path),
                "vote_count": len(votes),
                "cases": {
                    uid: {
                        "strict_action": (
                            strict_aggregate[uid].get("selected_proposal") or {}
                        ).get("action"),
                        "strict_ready": strict_aggregate[uid]["ready_for_binding"],
                        "majority_action": (
                            majority_diagnostic[uid].get("selected_majority_proposal")
                            or {}
                        ).get("action"),
                        "majority_count": majority_diagnostic[uid]["majority_count"],
                        "compiled_stage": compiled[uid]["stage"],
                        "defer_reasons": compiled[uid]["defer_reasons"],
                    }
                    for uid in case_uids
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
