from pathlib import Path

import numpy as np

from conceptgraph.revision.sparse_replay import (
    SparseCounterfactualReplayEngine,
    _expand_observation_scope,
)


class _FrameLedger:
    def __init__(self, root: Path):
        self.experiment_root = root
        self.object_versions = {
            "entity_a@v000001": {
                "object_uid": "entity_a",
                "lineage_uid": "lineage_a",
                "origin_observation_uid": "obs_a",
                "member_observation_uids": ["obs_a"],
            },
            "entity_b@v000001": {
                "object_uid": "entity_b",
                "lineage_uid": "lineage_b",
                "origin_observation_uid": "obs_b",
                "member_observation_uids": ["obs_b"],
            },
        }
        self.associations = {
            "obs_before": {"event_sequence": 9},
            "obs_anchor": {
                "event_sequence": 10,
                "decision": "CREATE_OBJECT",
                "object_uids_before": ["entity_a", "entity_b"],
                "candidate_object_version_uids": [
                    "entity_a@v000001",
                    "entity_b@v000001",
                ],
                "aggregate_sim_ref": {
                    "path": "evidence/similarities/frame_000001.npz",
                    "key": "aggregate_sim",
                },
            },
            "obs_after": {"event_sequence": 11},
        }

    def get_association_for_obs(self, uid):
        return self.associations[uid]

    def get_object_version(self, uid):
        return self.object_versions[uid]

    @staticmethod
    def sequence(row):
        return int(row["event_sequence"])


def test_dynamic_scope_expands_to_whole_current_entity_without_other_entities():
    scoped = {"obs_a"}
    membership = {
        "entity_a": ["obs_a", "obs_b"],
        "entity_b": ["obs_c"],
    }

    added, entities = _expand_observation_scope(scoped, membership)

    assert added == 1
    assert entities == {"entity_a"}
    assert scoped == {"obs_a", "obs_b"}


def test_dynamic_scope_can_expand_a_specific_partial_overlay_entity():
    scoped = {"obs_a"}
    membership = {
        "entity_a": ["obs_a"],
        "entity_b": ["obs_b", "obs_c"],
    }

    added, entities = _expand_observation_scope(
        scoped,
        membership,
        entity_uids=["entity_b"],
    )

    assert added == 2
    assert entities == {"entity_b"}
    assert scoped == {"obs_a", "obs_b", "obs_c"}


def test_suffix_rows_are_strictly_after_the_snapshot_watermark(tmp_path):
    engine = SparseCounterfactualReplayEngine.__new__(
        SparseCounterfactualReplayEngine
    )
    engine.provenance = _FrameLedger(tmp_path)
    rows = [
        {"obs_uid": "obs_before"},
        {"obs_uid": "obs_anchor"},
        {"obs_uid": "obs_after"},
    ]

    selected = engine._rows_strictly_after_watermark(rows, 9)

    assert [row["obs_uid"] for row in selected] == ["obs_anchor", "obs_after"]


def test_frozen_frame_matrix_excludes_objects_created_earlier_same_frame(tmp_path):
    matrix_path = tmp_path / "evidence" / "similarities" / "frame_000001.npz"
    matrix_path.parent.mkdir(parents=True)
    np.savez(matrix_path, aggregate_sim=np.asarray([[0.5, 1.5]], dtype=float))
    engine = SparseCounterfactualReplayEngine.__new__(
        SparseCounterfactualReplayEngine
    )
    engine.provenance = _FrameLedger(tmp_path)
    engine._similarity_cache = {}
    engine._obs_lineages = {}
    objects = [
        {"id": "entity_a", "obs_uids": ["obs_a"]},
        {"id": "entity_b", "obs_uids": ["obs_b"]},
        {"id": "same_frame_new", "obs_uids": ["obs_new"]},
    ]

    natural, scores = engine._frozen_recorded_frame_details(
        [{"obs_uid": "obs_anchor", "filtered_det_idx": 0}], objects
    )

    assert natural == [None]
    assert scores.shape == (1, 3)
    assert scores[0, :2].tolist() == [0.5, 1.5]
    assert np.isneginf(scores[0, 2])
