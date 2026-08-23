from conceptgraph.revision.snapshot import AnchorStateBuilder, _snapshot_validation_pass
from scripts.audit_revision_v1_same_frame_batching import _target_origin_difference


class _SnapshotLedger:
    def __init__(self):
        self.events = {
            "event_4": {"event_sequence": 4},
            "event_6": {"event_sequence": 6},
            "event_8": {"event_sequence": 8},
            "event_12": {"event_sequence": 12},
        }
        self.object_versions = {
            "entity_a@v000004": {
                "object_uid": "entity_a",
                "lineage_uid": "lineage_a",
                "trigger_event_uid": "event_4",
            },
            "entity_a@v000006": {
                "object_uid": "entity_a",
                "lineage_uid": "lineage_a",
                "trigger_event_uid": "event_6",
            },
            "entity_b@v000008": {
                "object_uid": "entity_b",
                "lineage_uid": "lineage_b",
                "trigger_event_uid": "event_8",
            },
            "entity_a@v000012": {
                "object_uid": "entity_a",
                "lineage_uid": "lineage_a",
                "trigger_event_uid": "event_12",
            },
        }

    def get_object_version(self, uid):
        return self.object_versions[uid]

    def get_event(self, uid):
        return self.events[uid]

    @staticmethod
    def sequence(row):
        return int(row["event_sequence"])


def test_snapshot_resolves_matrix_time_version_to_latest_pre_anchor_lineage_version():
    builder = AnchorStateBuilder.__new__(AnchorStateBuilder)
    builder.provenance = _SnapshotLedger()

    resolved, resolution, skipped = builder._resolve_pre_anchor_versions(
        ["entity_a@v000004"],
        anchor_sequence=10,
    )

    assert resolved == ["entity_a@v000006"]
    assert resolution[0]["advanced_within_prefix"] is True
    assert resolution[0]["resolved_event_sequence"] == 6
    assert skipped == []


def test_snapshot_resolution_excludes_post_anchor_and_other_lineages():
    builder = AnchorStateBuilder.__new__(AnchorStateBuilder)
    builder.provenance = _SnapshotLedger()

    resolved, _, _ = builder._resolve_pre_anchor_versions(
        ["entity_a@v000004", "entity_b@v000008"],
        anchor_sequence=10,
    )

    assert set(resolved) == {"entity_a@v000006", "entity_b@v000008"}
    assert "entity_a@v000012" not in resolved


def test_snapshot_gate_rejects_one_skipped_seed_even_if_another_version_passes():
    assert not _snapshot_validation_pass(
        requested_count=2,
        resolution=[{"requested_version_uid": "version_ok"}],
        skipped=[{"version_uid": "version_missing", "reason": "unknown_version"}],
        rows=[{"version_uid": "version_ok", "pass": True}],
    )


def test_snapshot_gate_accepts_multiple_requests_resolving_to_one_valid_version():
    assert _snapshot_validation_pass(
        requested_count=2,
        resolution=[
            {"requested_version_uid": "version_a"},
            {"requested_version_uid": "version_b"},
        ],
        skipped=[],
        rows=[{"version_uid": "shared_active_version", "pass": True}],
    )


def test_same_frame_target_audit_does_not_depend_on_truncated_candidates():
    provenance = _SnapshotLedger()
    provenance.object_versions["target_version"] = {
        "object_uid": "target",
        "lineage_uid": "lineage_target",
        "trigger_event_uid": "event_4",
        "origin_observation_uid": "obs_origin",
        "member_observation_uids": ["obs_origin"],
    }
    association = {"target_object_version_before": "target_version"}
    trace = {
        "natural_match": 25,
        "natural_candidates": [],
        "natural_target_origin_obs_uid": "obs_origin",
    }

    assert _target_origin_difference(provenance, association, trace) is None
    trace["natural_target_origin_obs_uid"] = "different_origin"
    assert _target_origin_difference(provenance, association, trace) == {
        "recorded": "obs_origin",
        "replayed": "different_origin",
    }
