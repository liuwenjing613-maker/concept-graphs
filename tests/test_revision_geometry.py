from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conceptgraph.revision.constraints import (
    CandidateTarget,
    ConstraintAction,
    ConstraintEngine,
    SparseRepairConstraint,
)
from conceptgraph.revision.geometry import (
    GeometryContractError,
    ObservationGeometryContract,
    array_sha256,
    canonical_json_sha256,
    file_sha256,
)
from conceptgraph.revision.runtime_verify import InvariantVerifier


def _contract(tmp_path: Path):
    observation = {"obs_uid": "scene_f000001_r0001", "status": "kept"}
    sources = []
    for role in (
        "raw_mask",
        "processed_mask",
        "depth",
        "rgb",
        "original_observation_pcd",
    ):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(role.encode("ascii"))
        sources.append(
            {
                "role": role,
                "path": str(path),
                "sha256": file_sha256(path),
                "format": "bin",
            }
        )
    points = np.asarray(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.2, 1.1]],
        dtype=np.float64,
    )
    colors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mask = np.asarray([[True, False], [True, True]], dtype=bool)
    pcd_path = tmp_path / "replacement_pcd.npz"
    mask_path = tmp_path / "replacement_mask.npz"
    np.savez_compressed(pcd_path, points=points, colors=colors)
    np.savez_compressed(mask_path, mask=mask)
    derivation = {
        "algorithm": "RAW_MASK_DEPTH_WORLD_PCD_V1",
        "random_perturbation": False,
        "replacement_points_sha256": array_sha256(points),
        "replacement_colors_sha256": array_sha256(colors),
        "replacement_mask_array_sha256": array_sha256(mask),
    }
    contract = ObservationGeometryContract.build(
        obs_uid=observation["obs_uid"],
        replacement_pcd_ref={
            "path": str(pcd_path),
            "sha256": file_sha256(pcd_path),
            "format": "npz",
        },
        replacement_mask_ref={
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
            "format": "npz",
            "key": "mask",
        },
        source_observation_sha256=canonical_json_sha256(observation),
        source_artifacts=sources,
        derivation=derivation,
    )
    return observation, contract, points, mask


def _constraint(contract: ObservationGeometryContract) -> SparseRepairConstraint:
    return SparseRepairConstraint.from_mapping(
        {
            "type": "RESTORE_OBSERVATION_GEOMETRY",
            "obs_uid": contract.obs_uid,
            "geometry_contract": contract.as_dict(),
            "applies_at_event_uid": "event_1",
            "active_from_sequence": 1,
        }
    )


def test_geometry_contract_loads_exact_hash_bound_payload(tmp_path):
    observation, contract, points, mask = _contract(tmp_path)
    assert contract.verify_source_bindings(observation)["pass"]
    payload = contract.load_payload()
    assert np.array_equal(payload["points"], points)
    assert np.array_equal(payload["mask"], mask)
    reparsed = ObservationGeometryContract.from_mapping(contract.as_dict())
    assert reparsed == contract


def test_geometry_contract_fails_closed_on_payload_drift(tmp_path):
    _, contract, _, _ = _contract(tmp_path)
    Path(contract.replacement_pcd_ref["path"]).write_bytes(b"drift")
    with pytest.raises(GeometryContractError, match="artifact drift"):
        contract.load_payload()


def test_geometry_constraint_keeps_recomputed_natural_match(tmp_path):
    _, contract, _, _ = _contract(tmp_path)
    primitive = _constraint(contract)
    candidate = CandidateTarget.build(
        index=2,
        entity_uid="entity_2",
        lineage_uids=("lineage_2",),
        score=1.5,
    )
    decision = ConstraintEngine([primitive]).resolve_for_observation(
        obs_uid=contract.obs_uid,
        event_uid="event_1",
        event_sequence=1,
        natural_match=2,
        natural_candidates=[candidate],
    )
    assert decision.action == ConstraintAction.KEEP_NATURAL
    assert decision.target_index == 2
    assert decision.reason == "geometry_payload_overlay_applied_before_association"


def test_runtime_verifier_requires_exact_geometry_trace(tmp_path):
    _, contract, points, mask = _contract(tmp_path)
    primitive = _constraint(contract)
    restoration = {
        "applied": True,
        "source_binding_pass": True,
        "payload_uid": contract.payload_uid,
        "replacement_pcd_sha256": contract.replacement_pcd_ref["sha256"],
        "replacement_mask_sha256": contract.replacement_mask_ref["sha256"],
        "replacement_points_sha256": array_sha256(points),
        "replacement_colors_sha256": contract.derivation["replacement_colors_sha256"],
        "replacement_mask_array_sha256": array_sha256(mask),
    }
    decision = {
        "obs_uid": contract.obs_uid,
        "applied_match": 0,
        "geometry_restoration": restoration,
        "constraint": {
            "action": "KEEP_NATURAL",
            "target_index": 0,
            "constraint_uids": [primitive.constraint_uid],
            "forbidden_indices": [],
        },
    }
    state = {
        "membership": {"entity_1": [contract.obs_uid]},
        "objects": [
            {
                "entity_uid": "entity_1",
                "member_observation_uids": [contract.obs_uid],
                "n_points": len(points),
                "bbox_center": [0.0, 0.0, 1.0],
                "bbox_extent": [0.1, 0.2, 0.1],
            }
        ],
        "edges": [],
        "decision_trace": [decision],
    }
    verifier = InvariantVerifier()
    valid = verifier.verify(
        state=state,
        constraints=[primitive],
        known_observation_uids=[contract.obs_uid],
    )
    assert valid["pass"]

    restoration["payload_uid"] = "geometry_payload_wrong"
    invalid = verifier.verify(
        state=state,
        constraints=[primitive],
        known_observation_uids=[contract.obs_uid],
    )
    assert not invalid["pass"]
    assert not invalid["checks"]["R6_constraint_satisfied"]
