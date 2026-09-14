from copy import deepcopy

from conceptgraph.revision.constraints import SparseRepairConstraint
from conceptgraph.revision.runtime_verify import InvariantVerifier


def _state() -> dict:
    decision = {
        "obs_uid": "obs_anchor",
        "applied_match": 0,
        "constraint": {
            "action": "FORCE_TARGET",
            "target_index": 0,
            "forbidden_indices": [],
            "constraint_uids": [],
        },
    }
    return {
        "membership": {"entity_a": ["obs_anchor"]},
        "objects": [
            {
                "entity_uid": "entity_a",
                "member_observation_uids": ["obs_anchor"],
                "n_points": 3,
                "bbox_center": [0.0, 0.0, 0.0],
                "bbox_extent": [1.0, 1.0, 1.0],
            }
        ],
        "edges": [],
        "decision_trace": [decision],
    }


def _constraint() -> SparseRepairConstraint:
    return SparseRepairConstraint.from_mapping(
        {
            "type": "ASSIGN_OBSERVATION",
            "obs_uid": "obs_anchor",
            "target_lineage_uid": "lineage_a",
        }
    )


def test_runtime_verifier_checks_membership_payload_and_constraint_semantics() -> None:
    state = _state()
    constraint = _constraint()
    state["decision_trace"][0]["constraint"]["constraint_uids"] = [
        constraint.constraint_uid
    ]
    result = InvariantVerifier().verify(
        state=state,
        constraints=[constraint],
        known_observation_uids=["obs_anchor"],
    )
    assert result["pass"] is True
    assert result["checks"]["R2_evidence_and_object_refs"] is True

    corrupted = deepcopy(state)
    corrupted["objects"][0]["member_observation_uids"] = ["unknown_obs"]
    corrupted["decision_trace"][0]["applied_match"] = 1
    result = InvariantVerifier().verify(
        state=corrupted,
        constraints=[constraint],
        known_observation_uids=["obs_anchor"],
    )
    assert result["pass"] is False
    assert result["checks"]["R2_evidence_and_object_refs"] is False
    assert result["checks"]["R6_constraint_satisfied"] is False
