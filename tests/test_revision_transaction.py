from pathlib import Path

from conceptgraph.revision.models import (
    DependencyClosure,
    RepairConstraint,
    RevisionTransaction,
)
from conceptgraph.revision.transactions import ShadowTransactionManager


class _FakeProvenance:
    observations = {"obs": {"obs_uid": "obs"}, "outside": {"obs_uid": "outside"}}
    final_by_object = {
        "A": {"member_observation_uids": ["obs"]},
        "B": {"member_observation_uids": ["outside"]},
    }

    def source_hashes(self):
        return {"ledger": "frozen"}

    def get_current_version(self, entity):
        return {"object_version_uid": entity + "@v1"} if entity in self.final_by_object else None


def _object(entity, obs, offset=0.0):
    return {
        "entity_uid": entity,
        "member_observation_uids": [obs],
        "n_points": 10,
        "bbox_center": [offset, 0.0, 0.0],
        "bbox_extent": [1.0, 1.0, 1.0],
        "aabb_min": [offset - 0.5, -0.5, -0.5],
        "aabb_max": [offset + 0.5, 0.5, 0.5],
    }


def test_shadow_commit_is_independent_and_preserves_outside_closure(tmp_path: Path):
    provenance = _FakeProvenance()
    baseline = {
        "membership": {"A": ["obs"], "B": ["outside"]},
        "objects": [_object("A", "obs"), _object("B", "outside", 2.0)],
        "edges": [],
        "source_hashes": provenance.source_hashes(),
    }
    derived = {
        "membership": {"A": ["obs"], "B": ["outside"]},
        "objects": [_object("A", "obs", 0.1), _object("B", "outside", 2.0)],
        "edges": [],
    }
    closure = DependencyClosure.build(
        event_uids=["event"],
        version_uids=["A@v1"],
        entity_uids=["A"],
        obs_uids=["obs"],
        edge_uids=[],
        start_sequence=1,
        end_sequence=2,
    )
    transaction = RevisionTransaction(
        case_uid="case",
        causal_anchor_event_uid="event",
        base_event_watermark=2,
        base_entity_versions={"A": "A@v1"},
        read_set=("obs",),
        write_set=("A",),
        dependency_closure=closure,
        repair_constraint=RepairConstraint(constraint_type="DEFER"),
    )
    manager = ShadowTransactionManager(provenance, tmp_path)
    result = manager.verify_and_commit(
        transaction=transaction,
        baseline_state=baseline,
        derived_state=derived,
    )
    assert result["transaction"]["commit_status"] == "COMMITTED"
    assert result["verification"]["pass"] is True
    assert baseline["objects"][0]["bbox_center"] == [0.0, 0.0, 0.0]
    assert (tmp_path / "case" / "shadow" / "derived_state.json").is_file()
    assert (tmp_path / "case" / "derived" / "derived_state.json").is_file()
