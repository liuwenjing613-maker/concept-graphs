from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conceptgraph.revision.constraints import (
    CandidateTarget,
    ConstraintAction,
    ConstraintConflictError,
    ConstraintEngine,
    SparseRepairConstraint,
)
from conceptgraph.revision.partition import (
    ObservationPartitionContract,
    apply_observation_partition,
    observation_payload_sha256,
    partition_assignment_sha256,
)


def _payload() -> dict[str, np.ndarray]:
    return {
        "points": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "colors": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 1.0],
            ],
            dtype=np.float32,
        ),
        "point_ids": np.arange(4, dtype=np.int32),
    }


def _contract(
    *,
    payload: dict[str, np.ndarray] | None = None,
    assignment: np.ndarray | None = None,
) -> ObservationPartitionContract:
    source = payload or _payload()
    labels = (
        np.asarray([0, 0, 1, 1], dtype=np.int64) if assignment is None else assignment
    )
    return ObservationPartitionContract.from_mapping(
        {
            "obs_uid": "scene_run_f000010_r0002",
            "source_point_count": 4,
            "source_payload_sha256": observation_payload_sha256(source),
            "assignment_sha256": partition_assignment_sha256(labels),
            "parts": [
                {
                    "part_index": 0,
                    "part_uid": "table",
                    "identity_uid": "identity_table",
                    "label": "table",
                },
                {
                    "part_index": 1,
                    "part_uid": "floor",
                    "identity_uid": "identity_floor",
                    "label": "floor",
                },
            ],
            "evidence_refs": [
                "mask_sha256:abc",
                "point_labels_sha256:def",
            ],
        }
    )


def test_partition_observation_is_exhaustive_disjoint_and_provenance_preserving():
    payload = _payload()
    assignment = np.asarray([0, 0, 1, 1], dtype=np.int32)
    result = apply_observation_partition(
        _contract(payload=payload, assignment=assignment),
        payload=payload,
        assignment=assignment,
    )

    assert result.validation["pass"]
    assert result.validation["exhaustive"]
    assert result.validation["disjoint"]
    assert result.validation["atomic"]
    assert [part.point_count for part in result.parts] == [2, 2]
    assert result.parts[0].payload["point_ids"].tolist() == [0, 1]
    assert result.parts[1].payload["point_ids"].tolist() == [2, 3]
    assert result.parts[0].provenance_observation_uids == ("scene_run_f000010_r0002",)
    assert result.parts[0].identity_uid != result.parts[1].identity_uid
    assert result.parts[0].obs_uid.endswith("::partition::table")


def test_partition_observation_rejects_payload_or_assignment_hash_mismatch():
    payload = _payload()
    assignment = np.asarray([0, 0, 1, 1], dtype=np.int32)
    contract = _contract(payload=payload, assignment=assignment)

    changed = {name: value.copy() for name, value in payload.items()}
    changed["points"][0, 0] = 99.0
    with pytest.raises(ValueError, match="payload hash mismatch"):
        apply_observation_partition(
            contract,
            payload=changed,
            assignment=assignment,
        )

    with pytest.raises(ValueError, match="assignment hash mismatch"):
        contract.validate_assignment(np.asarray([0, 1, 0, 1], dtype=np.int32))


def test_partition_observation_rejects_missing_part_even_with_matching_hash():
    payload = _payload()
    assignment = np.asarray([0, 0, 0, 0], dtype=np.int32)
    contract = _contract(payload=payload, assignment=assignment)
    with pytest.raises(ValueError, match="not exhaustive"):
        contract.validate_assignment(assignment)


def test_partition_contract_requires_distinct_effective_identities():
    value = _contract().as_dict()
    value["parts"][1]["identity_uid"] = value["parts"][0]["identity_uid"]
    value.pop("partition_uid")
    with pytest.raises(ValueError, match="pairwise distinct"):
        ObservationPartitionContract.from_mapping(value)


def test_partition_constraint_roundtrip_and_association_stage_defers():
    contract = _contract()
    constraint = SparseRepairConstraint.from_mapping(
        {
            "type": "PARTITION_OBSERVATION",
            "obs_uid": contract.obs_uid,
            "partition_contract": contract.as_dict(),
            "applies_at_event_uid": "event_10",
            "source": "human_point_gold",
            "evidence_refs": ["point_labels_sha256:def"],
        }
    )
    restored = SparseRepairConstraint.from_mapping(constraint.as_dict())
    assert restored.constraint_uid == constraint.constraint_uid
    assert restored.partition_contract["partition_uid"] == contract.partition_uid

    decision = ConstraintEngine([restored]).resolve_for_observation(
        obs_uid=contract.obs_uid,
        event_uid="event_10",
        event_sequence=10,
        natural_match=0,
        natural_candidates=[
            CandidateTarget.build(
                index=0,
                entity_uid="entity",
                lineage_uids=["identity_table"],
                score=2.0,
            )
        ],
    )
    assert decision.action == ConstraintAction.DEFER
    assert (
        decision.reason
        == "partition_observation_requires_pre_association_payload_stage"
    )


def test_partition_constraint_rejects_obs_mismatch_and_scope_conflict():
    contract = _contract()
    with pytest.raises(ValueError, match="does not match"):
        SparseRepairConstraint.from_mapping(
            {
                "type": "PARTITION_OBSERVATION",
                "obs_uid": "different_obs",
                "partition_contract": contract.as_dict(),
            }
        )

    partition = SparseRepairConstraint.from_mapping(
        {
            "type": "PARTITION_OBSERVATION",
            "obs_uid": contract.obs_uid,
            "partition_contract": contract.as_dict(),
            "applies_at_event_uid": "event_10",
        }
    )
    create = SparseRepairConstraint.from_mapping(
        {
            "type": "CREATE_INSTANCE",
            "obs_uid": contract.obs_uid,
            "created_identity_uid": "new_identity",
            "applies_at_event_uid": "event_10",
        }
    )
    with pytest.raises(ConstraintConflictError, match="cannot share"):
        ConstraintEngine([partition, create])


def test_prevoxel_partition_excludes_contamination_atomically(tmp_path):
    payload = _payload()
    assignment = np.asarray([0, 0, 1, 1], dtype=np.uint16)
    assignment_path = tmp_path / "assignment.npz"
    np.savez_compressed(assignment_path, assignment=assignment)

    value = _contract(payload=payload, assignment=assignment).as_dict()
    value.pop("partition_uid")
    value["source_stage"] = "PRE_VOXEL_SAMPLED_PAYLOAD"
    value["assignment_ref"] = {
        "path": str(assignment_path),
        "format": "npz",
        "key": "assignment",
        "sha256": hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
        "assignment_sha256": value["assignment_sha256"],
    }
    value["parts"][1]["disposition"] = "EXCLUDE_AS_CONTAMINATION"
    contract = ObservationPartitionContract.from_mapping(value)
    result = apply_observation_partition(
        contract,
        payload=payload,
        assignment=assignment,
    )

    assert contract.source_stage == "PRE_VOXEL_SAMPLED_PAYLOAD"
    assert len(result.parts) == 1
    assert len(result.excluded_parts) == 1
    assert result.parts[0].point_count == 2
    assert result.excluded_parts[0].point_count == 2
    assert result.validation["assigned_point_count"] == 4
    assert result.validation["emitted_point_count"] == 2
    assert result.validation["excluded_point_count"] == 2
    assert result.validation["exhaustive"]
    assert result.validation["disjoint"]
    assert ObservationPartitionContract.from_mapping(contract.as_dict()) == contract


def test_partition_v1_roundtrip_remains_supported():
    value = _contract().as_dict()
    value.pop("partition_uid")
    value.pop("source_stage")
    value.pop("assignment_ref")
    value["schema_version"] = "1.0.0"
    contract = ObservationPartitionContract.from_mapping(value)
    assert contract.schema_version == "1.0.0"
    assert all(part.disposition == "EMIT_OBSERVATION" for part in contract.parts)


def test_prevoxel_partition_requires_hash_bound_assignment_ref():
    value = _contract().as_dict()
    value.pop("partition_uid")
    value["source_stage"] = "PRE_VOXEL_SAMPLED_PAYLOAD"
    value["assignment_ref"] = None
    with pytest.raises(ValueError, match="requires assignment_ref"):
        ObservationPartitionContract.from_mapping(value)
