#!/usr/bin/env python3
"""Run the five-case blind automatic-constraint generation experiment."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.auto_constraints import (
    IncidentBinding,
    aggregate_candidate_votes,
    compile_blind_candidate,
    forbidden_inference_paths,
)
from conceptgraph.revision.cases import canonical_obs_key
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.vlm import (
    VLMIncidentBuilder,
    VLMIncidentEvidence,
    run_parallel_votes,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity_alias(
    provenance: ProvenanceIndex,
    *,
    alias: str,
    object_uid: str,
    version_uid: str,
) -> dict[str, Any]:
    version = provenance.get_object_version(version_uid)
    members = provenance.get_member_observations(version_uid)
    lineage_uid = version.get("lineage_uid")
    origin_obs_uid = version.get("origin_observation_uid") or (
        members[0] if members else None
    )
    complete = bool(
        version.get("object_uid") == object_uid
        and lineage_uid
        and origin_obs_uid
        and members
    )
    return {
        "alias": alias,
        "entity_uid": object_uid,
        "lineage_uid": lineage_uid,
        "origin_obs_uid": origin_obs_uid,
        "identity_uids": [lineage_uid] if lineage_uid else [],
        "provenance_lineage_uids": [lineage_uid] if lineage_uid else [],
        "member_observation_count": len(members),
        "complete": complete,
    }


def _build_identity_case(
    case: Mapping[str, Any],
    *,
    provenance: ProvenanceIndex,
    output_dir: Path,
) -> tuple[VLMIncidentEvidence, IncidentBinding, dict[str, Any]]:
    event_uid = str(case["anchor_association_event_uid"])
    association = provenance.get_event(event_uid)
    observed = (
        "CREATE"
        if str(association.get("decision", "")).upper() == "CREATE_OBJECT"
        else "ASSOCIATE"
    )
    evidence = VLMIncidentBuilder(provenance).build(
        {
            "anchor_association_event_uid": event_uid,
            "observed_current_decision": observed,
        },
        output_dir / "evidence_images",
    )
    candidate_versions = list(association.get("candidate_object_version_uids") or ())
    objects_before = list(association.get("object_uids_before") or ())
    version_by_object = {
        str(object_uid): str(candidate_versions[index])
        for index, object_uid in enumerate(objects_before)
        if index < len(candidate_versions)
    }
    aliases = {}
    for rank, candidate in enumerate(association.get("top_candidates") or (), 1):
        if rank > 2:
            break
        object_uid = str(candidate.get("object_uid", ""))
        version_uid = version_by_object.get(object_uid)
        if not object_uid or not version_uid:
            continue
        alias = f"CANDIDATE_{rank}_CONTEXT"
        aliases[alias] = _identity_alias(
            provenance,
            alias=alias,
            object_uid=object_uid,
            version_uid=version_uid,
        )

    mapping_event = provenance.get_event(str(association["mapping_event_uid"]))
    created_entity_uid = None
    created_identity_uid = None
    if mapping_event.get("event_type") == "OBJECT_CREATE":
        created_entity_uid = mapping_event.get("object_uid")
        outputs = mapping_event.get("output_object_version_uids") or ()
        if len(outputs) == 1:
            created_version = provenance.get_object_version(str(outputs[0]))
            created_identity_uid = created_version.get("lineage_uid")
            aliases["ANCHOR"] = _identity_alias(
                provenance,
                alias="ANCHOR",
                object_uid=str(created_entity_uid),
                version_uid=str(outputs[0]),
            )

    image_hashes = [
        {
            **row,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for row, path in zip(evidence.image_manifest, evidence.image_paths)
    ]
    binding = IncidentBinding.from_mapping(
        {
            "case_uid": str(case["blind_case_uid"]),
            "obs_uid": str(association["obs_uid"]),
            "obs_key": canonical_obs_key(str(association["obs_uid"])),
            "event_uid": event_uid,
            "event_sequence": provenance.sequence(association),
            "observed_current_decision": observed,
            "aliases": aliases,
            "created_entity_uid": created_entity_uid,
            "created_identity_uid": created_identity_uid,
            "evidence_refs": [
                str(case["blind_case_uid"]),
                event_uid,
                *[row["image_id"] + ":" + row["sha256"] for row in image_hashes],
            ],
        }
    )
    audit = {
        "blind_case_uid": case["blind_case_uid"],
        "input_family": case["input_family"],
        "prompt_sha256": _text_sha256(evidence.prompt),
        "prompt": evidence.prompt,
        "image_manifest": image_hashes,
        "allowed_evidence_image_ids": [
            row["image_id"] for row in evidence.image_manifest
        ],
        "binding": binding.as_dict(),
    }
    return evidence, binding, audit


def _sanitized_trigger_observations(
    review_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for row in review_evidence.get("trigger_observations") or ():
        obs_uid = str(row["obs_uid"])
        pre_dbscan = row.get("pre_dbscan") or {}
        post_dbscan = row.get("post_dbscan") or {}
        result.append(
            {
                "observation_alias": str(row.get("observation_alias") or "ANCHOR"),
                "obs_key": canonical_obs_key(obs_uid),
                "status": row.get("status"),
                "class_name": row.get("class_name"),
                "confidence": row.get("confidence"),
                "raw_mask_area": row.get("raw_mask_area"),
                "pre_subtract_mask_area": row.get("pre_subtract_mask_area"),
                "processed_mask_area": row.get("processed_mask_area"),
                "removed_pixel_count": row.get("removed_pixel_count"),
                "valid_depth_ratio": row.get("valid_depth_ratio"),
                "n_points": row.get("n_points"),
                "pre_dbscan_cluster_count": pre_dbscan.get("cluster_count"),
                "pre_dbscan_largest_cluster_ratio": pre_dbscan.get(
                    "largest_cluster_ratio"
                ),
                "post_dbscan_n_points": post_dbscan.get("n_points"),
                "bbox_3d_extent": row.get("bbox_3d_extent"),
            }
        )
    return result


def _build_capability_case(
    case: Mapping[str, Any],
) -> tuple[VLMIncidentEvidence, IncidentBinding, dict[str, Any]]:
    artifacts = {
        str(row["logical_name"]): row for row in case.get("evidence_artifacts") or ()
    }
    case_row = _read(Path(str(artifacts["case.json"]["path"])))
    review = _read(Path(str(artifacts["review_evidence.json"]["path"])))
    detector_evidence = {
        "checker_id": case_row.get("checker_id"),
        "stage": case_row.get("stage"),
        "subtype": case_row.get("subtype"),
        "scope": case_row.get("scope"),
        "certainty": case_row.get("certainty"),
        "proven_facts": case_row.get("proven_facts") or [],
        "hypotheses": case_row.get("hypotheses") or [],
        "vetoes": case_row.get("vetoes") or [],
        "missing_evidence": case_row.get("missing_evidence") or [],
    }
    image_rows = [
        row
        for row in case.get("evidence_artifacts") or ()
        if str(row["logical_name"]).lower().endswith((".png", ".jpg", ".jpeg"))
        and str(row["logical_name"]).startswith("review_")
    ]
    image_paths = tuple(Path(str(row["path"])).resolve() for row in image_rows)
    image_manifest = tuple(
        {
            "image_id": f"I{index + 1:02d}",
            "logical_name": str(row["logical_name"]),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
        }
        for index, row in enumerate(image_rows)
    )
    trigger_observations = _sanitized_trigger_observations(review)
    obs_keys = [
        canonical_obs_key(str(obs_uid))
        for obs_uid in case["representative_trigger_observation_uids"]
    ]
    prompt_payload = {
        "blind_incident_uid": hashlib.sha256(
            str(case["incident_uid"]).encode("utf-8")
        ).hexdigest()[:16],
        "detector_evidence": detector_evidence,
        "trigger_observations": trigger_observations,
        "observation_keys": obs_keys,
        "images": list(image_manifest),
        "allowed_actions": [
            "RELABEL",
            "RESTORE_OBSERVATION_GEOMETRY",
            "PARTITION_OBSERVATION",
            "DEFER",
        ],
    }
    forbidden = forbidden_inference_paths(prompt_payload)
    if forbidden:
        raise ValueError(
            "capability payload contains forbidden fields: " + str(forbidden)
        )
    prompt = (
        "Diagnose the safest repair capability using only this machine-generated "
        "detector evidence and the supplied review panels. No human endpoint label, "
        "human note, expected answer, final owner, or final membership is provided. "
        "Choose RELABEL only when the physical object is clear but its semantic name "
        "is wrong; include entity_alias='ANCHOR' and a concise label. Choose "
        "RESTORE_OBSERVATION_GEOMETRY when preprocessing discarded valid raw mask "
        "support; include the exact obs_key. Choose PARTITION_OBSERVATION only when "
        "one observation itself must be split into physical instances, and otherwise "
        "DEFER. This is candidate-family diagnosis, not permission to mutate. Return "
        "exactly one JSON object with action, confidence, entity_alias, label, "
        "obs_key, evidence_image_ids, and reason. Cite only listed image IDs.\n\n"
        "BLIND INCIDENT:\n" + json.dumps(prompt_payload, indent=2, sort_keys=True)
    )
    evidence = VLMIncidentEvidence(
        incident_uid=str(prompt_payload["blind_incident_uid"]),
        prompt=prompt,
        image_paths=image_paths,
        image_manifest=image_manifest,
        system_prompt=(
            "You conservatively diagnose typed repair candidates for a 3D scene "
            "graph. Distinguish semantic relabeling, restoration of observation "
            "geometry lost by preprocessing, point-level observation partitioning, "
            "and genuine ambiguity. Prefer DEFER over an unsupported diagnosis."
        ),
    )
    primary_obs_uid = str(case["representative_trigger_observation_uids"][0])
    binding = IncidentBinding.from_mapping(
        {
            "case_uid": str(case["blind_case_uid"]),
            "obs_uid": primary_obs_uid,
            "obs_key": canonical_obs_key(primary_obs_uid),
            "event_uid": str(case["incident_uid"]),
            "event_sequence": -1,
            "observed_current_decision": "CREATE",
            "aliases": {},
            "evidence_refs": [
                str(case["blind_case_uid"]),
                *[row["image_id"] + ":" + row["sha256"] for row in image_manifest],
            ],
        }
    )
    audit = {
        "blind_case_uid": case["blind_case_uid"],
        "input_family": case["input_family"],
        "prompt_sha256": _text_sha256(prompt),
        "prompt": prompt,
        "image_manifest": list(image_manifest),
        "allowed_evidence_image_ids": [row["image_id"] for row in image_manifest],
        "binding": binding.as_dict(),
    }
    return evidence, binding, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", required=True, type=Path)
    parser.add_argument("--office-run", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--key-count", type=int, default=5)
    parser.add_argument("--votes-per-case", type=int, default=3)
    parser.add_argument(
        "--experiment-uid",
        default="revision_v2_blind_auto_constraint_generation_20260824",
    )
    args = parser.parse_args()
    if args.key_count != 5 or args.votes_per_case != 3:
        raise ValueError(
            "the frozen design requires five key slots and three votes per case"
        )

    manifest = _read(args.blind_manifest)
    cases = list(manifest.get("cases") or ())
    if len(cases) != 5 or len({case["blind_case_uid"] for case in cases}) != 5:
        raise ValueError("blind manifest must contain exactly five unique cases")
    output_root = args.output_root.resolve()
    result_path = output_root / "blind_generation.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite generator result: {result_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    provenance = {
        "office0": ProvenanceIndex(args.office_run),
        "room0": ProvenanceIndex(args.room_run),
    }
    evidence: dict[str, VLMIncidentEvidence] = {}
    bindings: dict[str, IncidentBinding] = {}
    audits: dict[str, dict[str, Any]] = {}
    for case in cases:
        blind_uid = str(case["blind_case_uid"])
        if case["input_family"] == "IDENTITY_ASSOCIATION":
            built = _build_identity_case(
                case,
                provenance=provenance[str(case["scene_id"])],
                output_dir=output_root / blind_uid,
            )
        elif case["input_family"] == "CAPABILITY_PROBE":
            built = _build_capability_case(case)
        else:
            raise ValueError(f"unknown input family: {case['input_family']}")
        evidence[blind_uid], bindings[blind_uid], audits[blind_uid] = built

    forbidden_strings = (
        "FALSE_MERGE",
        "FALSE_SPLIT",
        "SEMANTIC_IDENTITY_ERROR",
        "GEOMETRY_CORRUPTION",
        "SPURIOUS_OBJECT",
        "posthoc_gold",
        "expected_capability",
        "final_owner_uids",
        "final_membership",
    )
    leakage = {
        blind_uid: [
            token for token in forbidden_strings if token.lower() in item.prompt.lower()
        ]
        for blind_uid, item in evidence.items()
    }
    if any(leakage.values()):
        raise RuntimeError(f"forbidden prompt leakage: {leakage}")

    ordered_uids = [str(case["blind_case_uid"]) for case in cases]
    schedule = []
    for round_index in range(args.votes_per_case):
        rotated = ordered_uids[round_index:] + ordered_uids[:round_index]
        schedule.append(
            {
                "round_index": round_index,
                "credential_slot_to_blind_case": {
                    str(slot): blind_uid for slot, blind_uid in enumerate(rotated)
                },
            }
        )
    protocol = {
        "schema_version": "2.0.0",
        "experiment_uid": str(args.experiment_uid),
        "blind_manifest_path": str(args.blind_manifest.resolve()),
        "blind_manifest_sha256": _sha256(args.blind_manifest.resolve()),
        "frozen_before_responses": True,
        "case_count": len(cases),
        "votes_per_case": args.votes_per_case,
        "parallel_credential_slots": args.key_count,
        "credential_rotation": True,
        "credential_slots_are_not_claimed_as_independent_models": True,
        "model": args.model,
        "base_url": args.base_url,
        "api_keys_persisted": False,
        "temperature_assumption": "provider default; no independence claim",
        "blind_request_audit": [audits[uid] for uid in ordered_uids],
        "forbidden_prompt_leakage": leakage,
        "schedule": schedule,
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
        for round_row in schedule:
            rotated = [
                round_row["credential_slot_to_blind_case"][str(slot)]
                for slot in range(args.key_count)
            ]
            jobs = [(blind_uid, evidence[blind_uid]) for blind_uid in rotated]
            round_votes = run_parallel_votes(
                jobs=jobs,
                api_keys=keys,
                base_url=args.base_url,
                model=args.model,
            )
            for row in round_votes:
                row["round_index"] = int(round_row["round_index"])
                row["credential_slot"] = int(row.pop("vote_index"))
                votes.append(row)
    finally:
        keys[:] = [""] * len(keys)

    aggregate: dict[str, dict[str, Any]] = {}
    compiled: dict[str, dict[str, Any]] = {}
    for blind_uid in ordered_uids:
        case_votes = [row for row in votes if row["case_uid"] == blind_uid]
        allowed = audits[blind_uid]["allowed_evidence_image_ids"]
        aggregate[blind_uid] = aggregate_candidate_votes(
            case_votes,
            allowed_evidence_ids=allowed,
            minimum_votes=args.votes_per_case,
        )
        compiled[blind_uid] = compile_blind_candidate(
            aggregate[blind_uid], bindings[blind_uid]
        )

    result = {
        "schema_version": "2.0.0",
        "experiment_uid": protocol["experiment_uid"],
        "inference_protocol_path": str(protocol_path),
        "inference_protocol_sha256": protocol_sha256,
        "inference_protocol_frozen_before_responses": True,
        "api_keys_persisted": False,
        "gold_loaded_by_generator": False,
        "vote_count": len(votes),
        "votes": votes,
        "aggregate": aggregate,
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
                        "action": (aggregate[uid].get("selected_proposal") or {}).get(
                            "action"
                        ),
                        "ready_for_binding": aggregate[uid]["ready_for_binding"],
                        "stage": compiled[uid]["stage"],
                        "defer_reasons": compiled[uid]["defer_reasons"],
                    }
                    for uid in ordered_uids
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
