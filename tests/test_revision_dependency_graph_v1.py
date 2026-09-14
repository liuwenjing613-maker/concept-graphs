from conceptgraph.revision.dependency_graph import TypedDependencyGraph


class _Ledger:
    def __init__(self):
        self.object_version_rows = [
            {
                "object_version_uid": "version_a",
                "object_uid": "entity_12",
                "lineage_uid": "lineage_12",
                "trigger_event_uid": "map_a",
                "parent_version_uids": [],
                "member_observation_uids": ["obs_a"],
            },
            {
                "object_version_uid": "version_b",
                "object_uid": "entity_123",
                "lineage_uid": "lineage_123",
                "trigger_event_uid": "map_b",
                "parent_version_uids": [],
                "member_observation_uids": ["obs_b"],
            },
        ]
        self.association_rows = [
            {
                "event_uid": "assoc_a",
                "event_sequence": 1,
                "obs_uid": "obs_a",
                "mapping_event_uid": "map_a",
                "target_object_version_after": "version_a",
                "target_object_uid": "entity_12",
            },
            {
                "event_uid": "assoc_b",
                "event_sequence": 3,
                "obs_uid": "obs_b",
                "mapping_event_uid": "map_b",
                "target_object_version_after": "version_b",
                "target_object_uid": "entity_123",
            },
        ]
        self.mapping_event_rows = [
            {
                "event_uid": "map_a",
                "event_sequence": 2,
                "association_event_uid": "assoc_a",
                "obs_uid": "obs_a",
                "object_uid": "entity_12",
                "input_object_version_uids": [],
                "output_object_version_uids": ["version_a"],
            },
            {
                "event_uid": "map_b",
                "event_sequence": 4,
                "association_event_uid": "assoc_b",
                "obs_uid": "obs_b",
                "object_uid": "entity_123",
                "input_object_version_uids": [],
                "output_object_version_uids": ["version_b"],
            },
        ]
        self.associations = {row["event_uid"]: row for row in self.association_rows}
        self.mapping_events = {row["event_uid"]: row for row in self.mapping_event_rows}
        self.events = {**self.associations, **self.mapping_events}
        self.max_sequence = 4

    @staticmethod
    def sequence(row):
        return int(row["event_sequence"])

    def get_event(self, uid):
        return self.events[uid]


def test_explicit_dependency_does_not_match_uuid_like_substrings():
    graph = TypedDependencyGraph(_Ledger())
    closure = graph.forward_closure(
        anchor_event_uid="assoc_a", seed_lineage_uids=["lineage_12"]
    )
    assert "assoc_a" in closure.event_uids
    assert "entity_12" in closure.entity_uids
    assert "assoc_b" not in closure.event_uids
    assert "entity_123" not in closure.entity_uids
