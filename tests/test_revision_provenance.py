import json
from pathlib import Path

from conceptgraph.revision.index import LineageIndex, ProvenanceIndex


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _ledger(tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    _write_jsonl(
        evidence / "observations.jsonl",
        [{"obs_uid": "f0_r0", "status": "kept"}, {"obs_uid": "f1_r0", "status": "kept"}],
    )
    _write_jsonl(
        evidence / "associations.jsonl",
        [
            {
                "event_uid": "e1",
                "mapping_event_uid": "e2",
                "event_sequence": 1,
                "obs_uid": "f0_r0",
                "target_object_uid": "o1",
            },
            {
                "event_uid": "e3",
                "mapping_event_uid": "e4",
                "event_sequence": 3,
                "obs_uid": "f1_r0",
                "target_object_uid": "o1",
            },
        ],
    )
    _write_jsonl(
        evidence / "mapping_events.jsonl",
        [
            {
                "event_uid": "e2",
                "event_sequence": 2,
                "event_type": "OBJECT_CREATE",
                "object_uid": "o1",
                "obs_uid": "f0_r0",
                "input_object_version_uids": [],
                "output_object_version_uids": ["o1@v000001"],
            },
            {
                "event_uid": "e4",
                "event_sequence": 4,
                "event_type": "OBS_ASSOCIATE",
                "object_uid": "o1",
                "obs_uid": "f1_r0",
                "input_object_version_uids": ["o1@v000001"],
                "output_object_version_uids": ["o1@v000002"],
            },
        ],
    )
    _write_jsonl(
        evidence / "object_versions.jsonl",
        [
            {
                "object_version_uid": "o1@v000001",
                "object_uid": "o1",
                "version": 1,
                "status": "active",
                "member_observation_uids": ["f0_r0"],
                "parent_version_uids": [],
                "lineage_uid": "l1",
            },
            {
                "object_version_uid": "o1@v000002",
                "object_uid": "o1",
                "version": 2,
                "status": "active",
                "member_observation_uids": ["f0_r0", "f1_r0"],
                "parent_version_uids": ["o1@v000001"],
                "lineage_uid": "l1",
            },
        ],
    )
    _write_jsonl(evidence / "object_pair_decisions.jsonl", [])
    (evidence / "final_membership.json").write_text(
        json.dumps([{"object_uid": "o1", "member_observation_uids": ["f0_r0", "f1_r0"]}]),
        encoding="utf-8",
    )
    return evidence


def test_provenance_index_resolves_complete_chain(tmp_path):
    index = ProvenanceIndex(_ledger(tmp_path))
    assert index.get_observation("f0_r0")["status"] == "kept"
    assert index.get_association_for_obs("f1_r0")["mapping_event_uid"] == "e4"
    assert index.get_current_version("o1")["object_version_uid"] == "o1@v000002"
    assert index.get_member_observations("o1@v000002") == ("f0_r0", "f1_r0")
    assert [row["event_uid"] for row in index.events_after(2)] == ["e3", "e4"]


def test_lineage_index_resolves_ancestors_descendants_and_redirect(tmp_path):
    lineage = LineageIndex(ProvenanceIndex(_ledger(tmp_path)))
    assert lineage.resolve_descendants("o1@v000001") == ("o1@v000002",)
    assert lineage.resolve_ancestors("o1@v000002") == ("o1@v000001",)
    assert lineage.is_descendant("o1@v000002", "o1@v000001")
    assert lineage.resolve_current_entities("l1") == ("o1",)
    lineage.add_redirect(
        source_version_uid="o1@v000002",
        target_entity_uids=["derived_a", "derived_b"],
        event_type="LINEAGE_SPLIT",
        tx_id="tx1",
    )
    assert lineage.resolve_current_entities("o1@v000002") == ("derived_a", "derived_b")
