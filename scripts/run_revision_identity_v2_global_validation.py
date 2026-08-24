#!/usr/bin/env python3
"""Validate V2 identity contracts with full replay, parity, and legal-merge gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.benchmark.human_error_pilot import (
    _resolve_groups,
    evaluate_endpoint_groups,
)
from conceptgraph.revision.constraints import (
    ConstraintType,
    ReplayMode,
    SparseRepairConstraint,
)
from conceptgraph.revision.evaluate import (
    geometry_metrics,
    symmetric_membership_metrics,
)
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine


def _read(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def _case(manifest: Mapping[str, Any], case_uid: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in manifest.get("cases") or ()
        if str(item.get("case_uid")) == str(case_uid)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest case {case_uid}, found {len(matches)}")
    return matches[0]


def _boundary_signature(
    state: Mapping[str, Any]
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    signature = []
    for row in state.get("identity_boundaries") or ():
        left = tuple(sorted(str(item) for item in row.get("left_identity_uids") or ()))
        right = tuple(
            sorted(str(item) for item in row.get("right_identity_uids") or ())
        )
        signature.append(tuple(sorted((left, right))))
    return sorted(signature)


def _geometry_exact_enough(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["bbox_iou_to_clean"]) >= 0.999
        and float(metrics["center_error_to_clean"]) <= 1e-4
        and float(metrics["extent_error_to_clean"]) <= 1e-4
        and abs(float(metrics["point_support"]) - 1.0) <= 1e-3
    )


def _mechanism_positive(
    state: Mapping[str, Any], primitive: SparseRepairConstraint
) -> bool:
    if primitive.constraint_type == ConstraintType.CREATE_INSTANCE:
        return (
            int(state.get("persistent_create_instance_merge_veto_count", 0))
            + int(state.get("persistent_create_instance_association_veto_count", 0))
            > 0
        )
    return int(state.get("persistent_lineage_redirect_override_count", 0)) > 0


def _all_observations(*states: Mapping[str, Any]) -> set[str]:
    return {
        str(obs_uid)
        for state in states
        for members in (state.get("membership") or {}).values()
        for obs_uid in members or ()
    }


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read(args.manifest)
    case = _case(manifest, args.case_uid)
    if str(case.get("causal_disposition")) != "REPLAYABLE_ASSOCIATION_CAUSE":
        raise ValueError("parity requires a replayable association-rooted case")
    primitive = SparseRepairConstraint.from_mapping(case["constraints"][0])
    provenance = ProvenanceIndex(args.base_run)
    hashes_before = provenance.source_hashes()
    engine = SparseCounterfactualReplayEngine(provenance)
    global_state = (
        _read(args.stored_global_state)
        if args.stored_global_state is not None
        else engine.replay_global(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=[primitive],
        )
    )
    local_state = _read(args.local_state)

    scope = _all_observations(local_state, global_state)
    membership = symmetric_membership_metrics(
        local_state["membership"], global_state["membership"]
    )
    geometry = geometry_metrics(
        local_state,
        global_state,
        observation_scope=scope,
    )
    groups = _resolve_groups(provenance, case["evaluation"]["groups"])
    desired = str(case["evaluation"]["desired_owner_relation"])
    probes = [str(item) for item in case["evaluation"].get("probe_obs_uids") or ()]
    local_endpoint = evaluate_endpoint_groups(
        local_state["membership"], groups, desired, probes=probes
    )
    global_endpoint = evaluate_endpoint_groups(
        global_state["membership"], groups, desired, probes=probes
    )
    verification = InvariantVerifier().verify(
        state=global_state,
        constraints=[primitive],
        source_hashes_before=hashes_before,
        source_hashes_after=provenance.source_hashes(),
        known_observation_uids=provenance.observations,
    )
    local_boundaries = _boundary_signature(local_state)
    global_boundaries = _boundary_signature(global_state)
    checks = {
        "local_endpoint_correct": bool(local_endpoint["correct"]),
        "global_endpoint_correct": bool(global_endpoint["correct"]),
        "local_global_membership_partition_exact": bool(membership["partition_exact"]),
        "local_global_geometry_equivalent": _geometry_exact_enough(geometry),
        "identity_boundary_signature_exact": local_boundaries == global_boundaries,
        "local_mechanism_positive": _mechanism_positive(local_state, primitive),
        "global_mechanism_positive": _mechanism_positive(global_state, primitive),
        "global_runtime_invariants": bool(verification["pass"]),
        "source_hashes_unchanged": hashes_before == provenance.source_hashes(),
    }
    local_ms = float(
        (local_state.get("timing") or {}).get(
            "suffix_total_wall_ms", local_state.get("runtime_ms", 0.0)
        )
    )
    global_ms = float(global_state.get("runtime_ms", 0.0))
    result = {
        "schema_version": "2.0.0",
        "evaluation_role": "IDENTITY_V2_LOCAL_GLOBAL_SAME_CONSTRAINT_PARITY",
        "case_uid": str(case["case_uid"]),
        "scene_id": str(case["scene_id"]),
        "endpoint_error_type": str(case["endpoint_error_type"]),
        "pass": all(checks.values()),
        "checks": checks,
        "local_endpoint": local_endpoint,
        "global_endpoint": global_endpoint,
        "local_vs_global": {
            "membership": membership,
            "geometry": geometry,
            "local_identity_boundaries": local_boundaries,
            "global_identity_boundaries": global_boundaries,
            "local_suffix_wall_ms": local_ms,
            "global_full_replay_runtime_ms": global_ms,
            "local_suffix_over_global_ratio": local_ms / max(global_ms, 1e-12),
        },
        "verification": verification,
        "component_policy": global_state.get("component_policy"),
        "source_hashes": hashes_before,
    }
    output = Path(args.output_root) / str(case["case_uid"])
    _write(output / "global_state.json", global_state)
    _write(output / "parity_result.json", result)
    return result


def _owners(membership: Mapping[str, Iterable[str]], obs_uid: str) -> set[str]:
    return {
        str(entity_uid)
        for entity_uid, members in membership.items()
        if str(obs_uid) in {str(item) for item in members or ()}
    }


def _identity_observations(row: Mapping[str, Any], side: str) -> set[str]:
    return {
        str(item)
        for item in (
            (row.get(f"{side}_identity") or {}).get("evidence_observation_uids") or ()
        )
    }


def _sha256_uid_set(values: Iterable[str]) -> str:
    payload = "".join(
        f"{item}\n" for item in sorted(set(str(value) for value in values))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compact_merge_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_idx": row.get("frame_idx"),
        "candidate_rank": row.get("candidate_rank"),
        "operation": row.get("operation"),
        "decision": row.get("decision"),
        "reject_reasons": list(row.get("reject_reasons") or ()),
        "source_entity_uid": row.get("source_entity_uid"),
        "target_entity_uid": row.get("target_entity_uid"),
        "source_identity_uids": list(
            (row.get("source_identity") or {}).get("effective_identity_uids") or ()
        ),
        "target_identity_uids": list(
            (row.get("target_identity") or {}).get("effective_identity_uids") or ()
        ),
    }


def run_legal_merge(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read(args.manifest)
    negative_manifest = _read(args.negative_manifest)
    negative = negative_manifest["legal_merge_negative"]
    protected_uids = [str(item) for item in negative["protected_case_uids"]]
    protected_cases = [_case(manifest, uid) for uid in protected_uids]
    constraints = [
        SparseRepairConstraint.from_mapping(case["constraints"][0])
        for case in protected_cases
    ]

    provenance = ProvenanceIndex(args.base_run)
    hashes_before = provenance.source_hashes()
    event = provenance.get_event(str(negative["event_uid"]))
    source_members = event.get("source_member_set_before") or ()
    target_members = event.get("target_member_set_before") or ()
    frozen_event_checks = {
        "event_type_object_merge": str(event.get("event_type")) == "OBJECT_MERGE",
        "event_sequence_exact": int(event.get("event_sequence", -1))
        == int(negative["event_sequence"]),
        "source_member_count_exact": len(source_members)
        == int(negative["source_member_count_before"]),
        "target_member_count_exact": len(target_members)
        == int(negative["target_member_count_before"]),
        "source_member_hash_exact": _sha256_uid_set(source_members)
        == str(negative["source_member_uid_sha256_lf"]),
        "target_member_hash_exact": _sha256_uid_set(target_members)
        == str(negative["target_member_uid_sha256_lf"]),
    }
    if not all(frozen_event_checks.values()):
        raise ValueError(f"frozen legal merge source mismatch: {frozen_event_checks}")

    global_state = (
        _read(args.stored_global_state)
        if args.stored_global_state is not None
        else SparseCounterfactualReplayEngine(provenance).replay_global(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=constraints,
        )
    )
    trigger_a, trigger_b = [
        str(item) for item in negative["human_label"]["trigger_observation_uids"]
    ]
    frame_rows = [
        row
        for row in global_state.get("postprocess_decision_trace") or ()
        if int(row.get("frame_idx", -1)) == int(negative["frame_idx"])
    ]
    cross_rows = []
    for row in frame_rows:
        source = _identity_observations(row, "source")
        target = _identity_observations(row, "target")
        if (trigger_a in source and trigger_b in target) or (
            trigger_b in source and trigger_a in target
        ):
            cross_rows.append(row)
    accepted = [
        row
        for row in cross_rows
        if str(row.get("operation")) == "OBJECT_MERGE_CANDIDATE"
        and str(row.get("decision")) in {"ACCEPT", "MERGE"}
        and not (row.get("reject_reasons") or ())
    ]
    boundary_rejections = [
        row
        for row in cross_rows
        if "persistent_create_instance_boundary" in (row.get("reject_reasons") or ())
    ]
    owner_a = _owners(global_state["membership"], trigger_a)
    owner_b = _owners(global_state["membership"], trigger_b)
    verification = InvariantVerifier().verify(
        state=global_state,
        constraints=constraints,
        source_hashes_before=hashes_before,
        source_hashes_after=provenance.source_hashes(),
        known_observation_uids=provenance.observations,
    )
    checks = {
        **frozen_event_checks,
        "cross_trigger_candidate_found": bool(cross_rows),
        "cross_trigger_merge_accepted": bool(accepted),
        "cross_trigger_not_boundary_rejected": not boundary_rejections,
        "trigger_a_has_one_owner": len(owner_a) == 1,
        "trigger_b_has_one_owner": len(owner_b) == 1,
        "merged_owner_relation_same_owner": bool(owner_a and owner_a == owner_b),
        "two_protected_identity_boundaries_active": len(
            global_state.get("identity_boundaries") or ()
        )
        == 2,
        "protected_runtime_invariants": bool(verification["pass"]),
        "source_hashes_unchanged": hashes_before == provenance.source_hashes(),
    }
    result = {
        "schema_version": "2.0.0",
        "evaluation_role": "IDENTITY_V2_LEGAL_MERGE_NEGATIVE_GLOBAL",
        "case_uid": str(negative["case_uid"]),
        "protected_case_uids": protected_uids,
        "pass": all(checks.values()),
        "checks": checks,
        "matched_cross_trigger_rows": [_compact_merge_row(row) for row in cross_rows],
        "accepted_cross_trigger_rows": [_compact_merge_row(row) for row in accepted],
        "boundary_rejected_cross_trigger_rows": [
            _compact_merge_row(row) for row in boundary_rejections
        ],
        "trigger_owners": {
            trigger_a: sorted(owner_a),
            trigger_b: sorted(owner_b),
        },
        "identity_boundaries": global_state.get("identity_boundaries"),
        "verification": verification,
        "runtime_ms": global_state.get("runtime_ms"),
        "source_hashes": hashes_before,
    }
    output = Path(args.output_root) / str(negative["case_uid"])
    _write(output / "global_state.json", global_state)
    _write(output / "legal_merge_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("parity", "legal-merge"), required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-uid")
    parser.add_argument("--local-state", type=Path)
    parser.add_argument("--negative-manifest", type=Path)
    parser.add_argument("--stored-global-state", type=Path)
    args = parser.parse_args()
    if args.task == "parity":
        if not args.case_uid or args.local_state is None:
            parser.error("parity requires --case-uid and --local-state")
        result = run_parity(args)
    else:
        if args.negative_manifest is None:
            parser.error("legal-merge requires --negative-manifest")
        result = run_legal_merge(args)
    print(
        json.dumps(
            {
                "task": args.task,
                "case_uid": result["case_uid"],
                "pass": result["pass"],
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
