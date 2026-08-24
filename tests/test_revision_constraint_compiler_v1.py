from conceptgraph.revision.benchmark.cases import compile_sparse_constraints


class _CreateLedger:
    def __init__(self):
        self.object_versions = {
            "entity_new@v000001": {
                "object_version_uid": "entity_new@v000001",
                "object_uid": "entity_new",
                "lineage_uid": "lineage_new",
                "origin_observation_uid": "obs_anchor",
                "member_observation_uids": ["obs_anchor"],
            }
        }
        self.anchor = {
            "event_uid": "event_anchor",
            "event_sequence": 9,
            "obs_uid": "obs_anchor",
            "mapping_event_uid": "event_mapping",
            "decision": "CREATE_OBJECT",
            "target_object_version_after": "entity_new@v000001",
        }

    def get_event(self, uid):
        assert uid == "event_anchor"
        return self.anchor

    def get_object_version(self, uid):
        return self.object_versions[uid]

    @staticmethod
    def sequence(row):
        return int(row["event_sequence"])


def test_create_action_compiles_to_first_class_create_instance_without_wrong_target():
    constraints = compile_sparse_constraints(
        {
            "anchor_association_event_uid": "event_anchor",
            "obs_uid": "obs_anchor",
            "failure_type": "FALSE_SPLIT",
        },
        _CreateLedger(),
    )

    assert len(constraints) == 1
    primitive = constraints[0]
    assert primitive.constraint_type.value == "CREATE_INSTANCE"
    assert primitive.target_key() == (None, None, None)
    assert primitive.created_lineage_uid == "lineage_new"
    assert primitive.created_entity_uid == "entity_new"
