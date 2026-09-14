#!/usr/bin/env python3
"""Validate local geometry restoration, its exact no-op, and collateral scope."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from conceptgraph.revision.benchmark.human_error_pilot import (
    HumanSceneContext,
    evaluate_collateral,
)
from conceptgraph.revision.constraints import ReplayMode, SparseRepairConstraint
from conceptgraph.revision.evaluate import (
    geometry_metrics,
    symmetric_membership_metrics,
)
from conceptgraph.revision.geometry import (
    ObservationGeometryContract,
    array_sha256,
    canonical_json_sha256,
    file_sha256,
)
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.snapshot import AnchorStateBuilder


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def _owners(membership: Mapping[str, list[str]], obs_uid: str) -> list[str]:
    return sorted(
        str(entity_uid)
        for entity_uid, members in membership.items()
        if obs_uid in set(str(item) for item in members or ())
    )


def _owner_row(state: Mapping[str, Any], obs_uid: str) -> dict[str, Any] | None:
    rows = [
        dict(row)
        for row in state.get("objects") or ()
        if obs_uid
        in set(str(item) for item in row.get("member_observation_uids") or ())
    ]
    return rows[0] if len(rows) == 1 else None


def _geometry_equivalent(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics["bbox_iou_to_clean"] >= 0.999999
        and metrics["center_error_to_clean"] <= 1e-9
        and metrics["extent_error_to_clean"] <= 1e-9
        and abs(metrics["point_support"] - 1.0) <= 1e-9
    )


def _noop_contract(
    *,
    context: HumanSceneContext,
    restoration: ObservationGeometryContract,
) -> ObservationGeometryContract:
    row = context.provenance.get_observation(restoration.obs_uid)
    detection = context.engine.materializer.materialize(restoration.obs_uid)
    points = np.asarray(detection["pcd"].points, dtype=np.float64)
    colors = np.asarray(detection["pcd"].colors, dtype=np.float64)
    mask = np.asarray(detection["mask"][0], dtype=bool)
    original_pcd = next(
        item
        for item in restoration.source_artifacts
        if item["role"] == "original_observation_pcd"
    )
    processed_mask = next(
        item
        for item in restoration.source_artifacts
        if item["role"] == "processed_mask"
    )
    return ObservationGeometryContract.build(
        obs_uid=restoration.obs_uid,
        replacement_pcd_ref={
            key: value for key, value in original_pcd.items() if key != "role"
        },
        replacement_mask_ref={
            key: value for key, value in processed_mask.items() if key != "role"
        },
        source_observation_sha256=canonical_json_sha256(row),
        source_artifacts=restoration.source_artifacts,
        derivation={
            "algorithm": "EXACT_EXISTING_PAYLOAD_NOOP_V1",
            "random_perturbation": False,
            "replacement_points_sha256": array_sha256(points),
            "replacement_colors_sha256": array_sha256(colors),
            "replacement_mask_array_sha256": array_sha256(mask),
        },
    )


def _constraint(
    contract: ObservationGeometryContract,
    *,
    event_uid: str,
    event_sequence: int,
    source: str,
) -> SparseRepairConstraint:
    return SparseRepairConstraint.from_mapping(
        {
            "type": "RESTORE_OBSERVATION_GEOMETRY",
            "obs_uid": contract.obs_uid,
            "geometry_contract": contract.as_dict(),
            "applies_at_event_uid": event_uid,
            "active_from_sequence": event_sequence,
            "source": source,
            "evidence_refs": [
                contract.obs_uid,
                event_uid,
                contract.payload_uid,
            ],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument(
        "--expected-build-role",
        default="DEVELOPMENT_GEOMETRY_CAPABILITY_NOT_HOLDOUT",
    )
    parser.add_argument(
        "--evaluation-role",
        default="DEVELOPMENT_GEOMETRY_LOCAL_CAUSAL_VALIDATION",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    result_path = output_root / "local_validation.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")

    build = _read(args.build_manifest)
    if build.get("evaluation_role") != args.expected_build_role or not build.get(
        "pass"
    ):
        raise ValueError("geometry build manifest role/pass does not match protocol")
    primitive = SparseRepairConstraint.from_mapping(build["constraint"])
    contract = ObservationGeometryContract.from_mapping(
        primitive.geometry_contract or {}
    )
    obs_uid = contract.obs_uid
    scene_id = str(args.scene_id or build.get("scene_id") or "")
    if not scene_id:
        raise ValueError("scene_id must be supplied or bound in the build manifest")
    if build.get("scene_id") and str(build["scene_id"]) != scene_id:
        raise ValueError("scene binding drift between build and local validation")
    context = HumanSceneContext.build(scene_id, args.base_run.resolve())
    provenance = context.provenance
    association = provenance.get_association_for_obs(obs_uid)
    event_uid = str(association["event_uid"])
    event_sequence = provenance.sequence(association)
    if primitive.applies_at_event_uid != event_uid:
        raise ValueError("geometry constraint anchor drift")
    noop_contract = _noop_contract(context=context, restoration=contract)
    noop_primitive = _constraint(
        noop_contract,
        event_uid=event_uid,
        event_sequence=event_sequence,
        source="exact_existing_geometry_payload_noop_control",
    )

    seed_versions = sorted(
        set(
            str(item)
            for item in (
                list(association.get("candidate_object_version_uids") or ())
                + [association.get("target_object_version_before")]
            )
            if item
        )
    )
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=event_uid,
        seed_version_uids=seed_versions,
    )
    frame_idx = int(str(association["frame_uid"]).rsplit("_f", 1)[-1])
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(frame_idx)
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        event_uid,
        seed_versions,
        strict=True,
        prefix_state=prefix_state,
        prefix_objects=prefix_objects,
    )
    native = copy.deepcopy(context.native_state)
    natural = context.engine.replay_suffix_from_snapshot(
        mode=ReplayMode.NATURAL_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        current_state=native,
    )
    noop = context.engine.replay_local_from_snapshot(
        mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        constraints=[noop_primitive],
        current_state=native,
    )
    sparse = context.engine.replay_local_from_snapshot(
        mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        constraints=[primitive],
        current_state=native,
    )

    all_observations = {
        str(item)
        for state in (natural, noop, sparse)
        for members in state["membership"].values()
        for item in members
    }
    noop_membership = symmetric_membership_metrics(
        natural["membership"], noop["membership"]
    )
    noop_geometry = geometry_metrics(
        natural,
        noop,
        observation_scope=all_observations,
    )
    verifier = InvariantVerifier()
    noop_invariants = verifier.verify(
        state=noop,
        constraints=[noop_primitive],
        source_hashes_before=context.source_hashes_before,
        source_hashes_after=provenance.source_hashes(),
        known_observation_uids=provenance.observations,
    )
    sparse_invariants = verifier.verify(
        state=sparse,
        constraints=[primitive],
        source_hashes_before=context.source_hashes_before,
        source_hashes_after=provenance.source_hashes(),
        known_observation_uids=provenance.observations,
    )

    affected = set(closure.obs_uids)
    affected.add(obs_uid)
    for entity_uid in closure.entity_uids:
        row = provenance.final_by_object.get(str(entity_uid))
        if row is not None:
            affected.update(
                str(item) for item in row.get("member_observation_uids") or ()
            )
    collateral = evaluate_collateral(
        native["membership"], sparse["membership"], affected
    )
    baseline_detection = context.engine.materializer.materialize(obs_uid)
    restored_detection = context.engine.materializer.materialize(
        obs_uid, geometry_contract=contract
    )
    baseline_points = np.asarray(baseline_detection["pcd"].points, dtype=np.float64)
    restored_points = np.asarray(restored_detection["pcd"].points, dtype=np.float64)
    restored_mask = np.asarray(restored_detection["mask"][0], dtype=bool)
    payload = contract.load_payload(base_root=provenance.experiment_root)
    sparse_traces = [
        row
        for row in sparse.get("decision_trace") or ()
        if str(row.get("obs_uid")) == obs_uid
    ]
    noop_traces = [
        row
        for row in noop.get("decision_trace") or ()
        if str(row.get("obs_uid")) == obs_uid
    ]
    source_artifact_hashes_after = {
        item["role"]: file_sha256(item["path"]) for item in contract.source_artifacts
    }
    endpoint_checks = {
        "replacement_points_exact": np.array_equal(restored_points, payload["points"]),
        "replacement_mask_exact": np.array_equal(restored_mask, payload["mask"]),
        "point_support_increased": len(restored_points) > len(baseline_points),
        "point_support_gain_ge_10x": len(restored_points) >= 10 * len(baseline_points),
        "restored_geometry_non_degenerate": bool(
            np.all(np.ptp(restored_points, axis=0) > 0.02)
        ),
        "sparse_observation_has_unique_owner": len(
            _owners(sparse["membership"], obs_uid)
        )
        == 1,
        "geometry_overlay_hit_exactly_once": sparse.get(
            "geometry_restoration_hit_count"
        )
        == 1,
        "geometry_similarity_recomputed_exactly_once": sparse.get(
            "geometry_similarity_recompute_count"
        )
        == 1,
        "geometry_trace_exactly_once": len(sparse_traces) == 1
        and bool((sparse_traces[0].get("geometry_restoration") or {}).get("applied")),
        "constraint_kept_recomputed_natural": len(sparse_traces) == 1
        and (sparse_traces[0].get("constraint") or {}).get("action") == "KEEP_NATURAL",
        "constraint_not_identity_forced": len(sparse_traces) == 1
        and (sparse_traces[0].get("constraint") or {}).get("reason")
        == "geometry_payload_overlay_applied_before_association",
    }
    noop_checks = {
        "membership_partition_exact": bool(noop_membership["partition_exact"]),
        "geometry_exact": _geometry_equivalent(noop_geometry),
        "runtime_invariants": bool(noop_invariants["pass"]),
        "overlay_hit_exactly_once": noop.get("geometry_restoration_hit_count") == 1,
        "similarity_recomputed_exactly_once": noop.get(
            "geometry_similarity_recompute_count"
        )
        == 1,
        "trace_exactly_once": len(noop_traces) == 1,
    }
    source_checks = {
        "provenance_hashes_unchanged": context.source_hashes_before
        == provenance.source_hashes(),
        "geometry_source_artifacts_unchanged": all(
            source_artifact_hashes_after[item["role"]] == item["sha256"]
            for item in contract.source_artifacts
        ),
    }
    required_endpoint_checks = (
        "replacement_points_exact",
        "replacement_mask_exact",
        "point_support_increased",
        "restored_geometry_non_degenerate",
        "sparse_observation_has_unique_owner",
        "geometry_overlay_hit_exactly_once",
        "geometry_similarity_recomputed_exactly_once",
        "geometry_trace_exactly_once",
        "constraint_kept_recomputed_natural",
        "constraint_not_identity_forced",
    )
    checks = {
        "build_manifest_pass": bool(build["pass"]),
        "snapshot_validation": bool(snapshot.validation["pass"]),
        "geometry_endpoint": all(
            bool(endpoint_checks[key]) for key in required_endpoint_checks
        ),
        "geometry_noop": all(noop_checks.values()),
        "sparse_runtime_invariants": bool(sparse_invariants["pass"]),
        "collateral_safe": bool(collateral["safe"]),
        "sources_immutable": all(source_checks.values()),
    }
    result = {
        "schema_version": "1.0.0",
        "evaluation_role": str(args.evaluation_role),
        "scene_id": scene_id,
        "production_commit_permitted": False,
        "obs_uid": obs_uid,
        "event_uid": event_uid,
        "constraint_uid": primitive.constraint_uid,
        "payload_uid": contract.payload_uid,
        "pass": all(checks.values()),
        "checks": checks,
        "endpoint_checks": endpoint_checks,
        "required_endpoint_checks": list(required_endpoint_checks),
        "noop_checks": noop_checks,
        "source_checks": source_checks,
        "source_artifact_hashes_after": source_artifact_hashes_after,
        "baseline_observation_geometry": {
            "point_count": int(len(baseline_points)),
            "mask_area": int(np.asarray(baseline_detection["mask"][0]).sum()),
            "aabb_extent": np.ptp(baseline_points, axis=0).tolist(),
        },
        "restored_observation_geometry": {
            "point_count": int(len(restored_points)),
            "mask_area": int(restored_mask.sum()),
            "aabb_extent": np.ptp(restored_points, axis=0).tolist(),
        },
        "association_effect": {
            "historical_default_match": (
                sparse_traces[0].get("historical_default_match")
                if sparse_traces
                else None
            ),
            "restored_natural_match": (
                sparse_traces[0].get("natural_match") if sparse_traces else None
            ),
            "applied_match": (
                sparse_traces[0].get("applied_match") if sparse_traces else None
            ),
            "changed_recorded_decision": bool(
                sparse_traces
                and sparse_traces[0].get("historical_default_match")
                != sparse_traces[0].get("applied_match")
            ),
            "threshold_semantics": (
                sparse_traces[0].get("threshold_semantics") if sparse_traces else None
            ),
        },
        "native_owner_uids": _owners(native["membership"], obs_uid),
        "natural_owner_uids": _owners(natural["membership"], obs_uid),
        "sparse_owner_uids": _owners(sparse["membership"], obs_uid),
        "sparse_owner": _owner_row(sparse, obs_uid),
        "collateral": collateral,
        "affected_observation_count": len(affected),
        "closure": closure.as_dict(),
        "snapshot_validation": snapshot.validation,
        "noop_membership": noop_membership,
        "noop_geometry": noop_geometry,
        "noop_invariants": noop_invariants,
        "sparse_invariants": sparse_invariants,
        "sparse_anchor_trace": sparse_traces,
        "noop_anchor_trace": noop_traces,
        "timing": {
            "natural": natural.get("timing"),
            "noop": noop.get("timing"),
            "sparse": sparse.get("timing"),
        },
    }
    _write(output_root / "natural_state.json", natural)
    _write(output_root / "noop_state.json", noop)
    _write(output_root / "sparse_state.json", sparse)
    _write(result_path, result)
    audit = {
        key: value
        for key, value in result.items()
        if key
        in {
            "schema_version",
            "evaluation_role",
            "production_commit_permitted",
            "obs_uid",
            "event_uid",
            "constraint_uid",
            "payload_uid",
            "pass",
            "checks",
            "endpoint_checks",
            "noop_checks",
            "source_checks",
            "baseline_observation_geometry",
            "restored_observation_geometry",
            "association_effect",
            "native_owner_uids",
            "natural_owner_uids",
            "sparse_owner_uids",
            "affected_observation_count",
            "timing",
        }
    }
    audit["result_path"] = str(result_path)
    if args.audit_output is not None:
        _write(args.audit_output.resolve(), audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
