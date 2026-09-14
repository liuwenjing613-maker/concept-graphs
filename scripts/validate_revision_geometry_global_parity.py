#!/usr/bin/env python3
"""Validate one hash-bound geometry repair with full local/global replay parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.constraints import (
    ConstraintType,
    ReplayMode,
    SparseRepairConstraint,
)
from conceptgraph.revision.evaluate import (
    geometry_metrics,
    symmetric_membership_metrics,
)
from conceptgraph.revision.geometry import file_sha256
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


def _owners(
    membership: Mapping[str, Iterable[str]], obs_uid: str
) -> dict[str, tuple[str, ...]]:
    return {
        str(entity_uid): tuple(sorted(str(item) for item in members or ()))
        for entity_uid, members in membership.items()
        if obs_uid in {str(item) for item in members or ()}
    }


def _all_observations(*states: Mapping[str, Any]) -> set[str]:
    return {
        str(obs_uid)
        for state in states
        for members in (state.get("membership") or {}).values()
        for obs_uid in members or ()
    }


def _geometry_equivalent(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["bbox_iou_to_clean"]) >= 0.999
        and float(metrics["center_error_to_clean"]) <= 1e-4
        and float(metrics["extent_error_to_clean"]) <= 1e-4
        and abs(float(metrics["point_support"]) - 1.0) <= 1e-3
    )


def _trace(state: Mapping[str, Any], obs_uid: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in state.get("decision_trace") or ()
        if str(row.get("obs_uid")) == obs_uid
    ]


def _trace_contract_ok(
    rows: list[Mapping[str, Any]], primitive: SparseRepairConstraint
) -> bool:
    if len(rows) != 1:
        return False
    row = rows[0]
    decision = row.get("constraint") or {}
    restoration = row.get("geometry_restoration") or {}
    contract = primitive.geometry_contract or {}
    derivation = contract.get("derivation") or {}
    return bool(
        row.get("native_default_source") == "RECOMPUTED_AFTER_GEOMETRY_OVERLAY"
        and decision.get("action") == "KEEP_NATURAL"
        and decision.get("reason")
        == "geometry_payload_overlay_applied_before_association"
        and restoration.get("applied")
        and restoration.get("source_binding_pass")
        and restoration.get("payload_uid") == contract.get("payload_uid")
        and restoration.get("replacement_pcd_sha256")
        == (contract.get("replacement_pcd_ref") or {}).get("sha256")
        and restoration.get("replacement_mask_sha256")
        == (contract.get("replacement_mask_ref") or {}).get("sha256")
        and restoration.get("replacement_points_sha256")
        == derivation.get("replacement_points_sha256")
        and restoration.get("replacement_colors_sha256")
        == derivation.get("replacement_colors_sha256")
        and restoration.get("replacement_mask_array_sha256")
        == derivation.get("replacement_mask_array_sha256")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--local-state", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--stored-global-state", type=Path)
    parser.add_argument(
        "--expected-build-role",
        default="DEVELOPMENT_GEOMETRY_CAPABILITY_NOT_HOLDOUT",
    )
    parser.add_argument(
        "--evaluation-role",
        default="DEVELOPMENT_GEOMETRY_LOCAL_GLOBAL_PARITY",
    )
    args = parser.parse_args()

    manifest = _read(args.build_manifest)
    if manifest.get("evaluation_role") != args.expected_build_role:
        raise ValueError("geometry build manifest role does not match protocol")
    if not manifest.get("pass"):
        raise ValueError("geometry build manifest did not pass")
    primitive = SparseRepairConstraint.from_mapping(manifest["constraint"])
    if primitive.constraint_type != ConstraintType.RESTORE_OBSERVATION_GEOMETRY:
        raise ValueError("expected RESTORE_OBSERVATION_GEOMETRY")
    obs_uid = str(primitive.obs_uid)

    provenance = ProvenanceIndex(args.base_run)
    source_hashes_before = provenance.source_hashes()
    source_artifacts_before = {
        str(item["role"]): str(item["sha256"])
        for item in (primitive.geometry_contract or {}).get("source_artifacts") or ()
    }
    local_state = _read(args.local_state)
    global_state = (
        _read(args.stored_global_state)
        if args.stored_global_state is not None
        else SparseCounterfactualReplayEngine(provenance).replay_global(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=[primitive],
        )
    )

    source_hashes_after = provenance.source_hashes()
    source_artifacts_after = {
        str(item["role"]): file_sha256(item["path"])
        for item in (primitive.geometry_contract or {}).get("source_artifacts") or ()
    }
    scope = _all_observations(local_state, global_state)
    membership = symmetric_membership_metrics(
        local_state["membership"], global_state["membership"]
    )
    geometry = geometry_metrics(
        local_state,
        global_state,
        observation_scope=scope,
    )
    local_owners = _owners(local_state["membership"], obs_uid)
    global_owners = _owners(global_state["membership"], obs_uid)
    local_rows = _trace(local_state, obs_uid)
    global_rows = _trace(global_state, obs_uid)
    verification = InvariantVerifier().verify(
        state=global_state,
        constraints=[primitive],
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        known_observation_uids=provenance.observations,
    )

    local_target_origin = (
        local_rows[0].get("applied_target_origin_obs_uid")
        if len(local_rows) == 1
        else None
    )
    global_target_origin = (
        global_rows[0].get("applied_target_origin_obs_uid")
        if len(global_rows) == 1
        else None
    )
    checks = {
        "build_manifest_pass": bool(manifest.get("pass")),
        "local_global_membership_partition_exact": bool(membership["partition_exact"]),
        "local_global_geometry_equivalent": _geometry_equivalent(geometry),
        "local_owner_unique": len(local_owners) == 1,
        "global_owner_unique": len(global_owners) == 1,
        "local_global_owner_members_exact": sorted(local_owners.values())
        == sorted(global_owners.values()),
        "local_geometry_trace_exact": _trace_contract_ok(local_rows, primitive),
        "global_geometry_trace_exact": _trace_contract_ok(global_rows, primitive),
        "local_overlay_hit_exactly_once": int(
            local_state.get("geometry_restoration_hit_count", 0)
        )
        == 1,
        "global_overlay_hit_exactly_once": int(
            global_state.get("geometry_restoration_hit_count", 0)
        )
        == 1,
        "local_similarity_recomputed_exactly_once": int(
            local_state.get("geometry_similarity_recompute_count", 0)
        )
        == 1,
        "global_similarity_recomputed_exactly_once": int(
            global_state.get("geometry_similarity_recompute_count", 0)
        )
        == 1,
        "local_global_applied_target_origin_exact": bool(local_target_origin)
        and local_target_origin == global_target_origin,
        "global_runtime_invariants": bool(verification["pass"]),
        "provenance_hashes_unchanged": source_hashes_before == source_hashes_after,
        "geometry_source_artifacts_unchanged": source_artifacts_before
        == source_artifacts_after,
    }
    result = {
        "schema_version": "1.0.0",
        "evaluation_role": str(args.evaluation_role),
        "production_commit_permitted": False,
        "obs_uid": obs_uid,
        "constraint_uid": primitive.constraint_uid,
        "payload_uid": (primitive.geometry_contract or {}).get("payload_uid"),
        "pass": all(checks.values()),
        "checks": checks,
        "local_vs_global": {
            "membership": membership,
            "geometry": geometry,
            "local_owner": local_owners,
            "global_owner": global_owners,
            "local_applied_target_origin_obs_uid": local_target_origin,
            "global_applied_target_origin_obs_uid": global_target_origin,
            "local_suffix_wall_ms": (local_state.get("timing") or {}).get(
                "suffix_total_wall_ms"
            ),
            "global_full_replay_runtime_ms": global_state.get("runtime_ms"),
        },
        "local_trace": local_rows,
        "global_trace": global_rows,
        "verification": verification,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_artifact_hashes_before": source_artifacts_before,
        "source_artifact_hashes_after": source_artifacts_after,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "global_state.json", global_state)
    _write(args.output_root / "parity_result.json", result)
    if args.audit_output is not None:
        _write(args.audit_output.resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
