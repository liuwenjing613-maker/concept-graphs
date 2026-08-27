from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from conceptgraph.revision.online_mvp import (
    OnlineEvidencePacket,
    _apply_semantic_relabels,
    _semantic_relabel_gate,
    compile_unified_vlm_response,
    final_scene_metric_gate,
)
from scripts.validate_unified_vlm_v1 import read_jsonl, validate_output


def _version(
    object_uid: str, lineage_uid: str, origin: str, class_name: str = "chair"
) -> dict:
    return {
        "object_uid": object_uid,
        "lineage_uid": lineage_uid,
        "origin_observation_uid": origin,
        "member_observation_uids": [origin],
        "identity_uids": ["identity:" + lineage_uid],
        "provenance_lineage_uids": [lineage_uid],
        "class_name": class_name,
    }


def _packet() -> OnlineEvidencePacket:
    return OnlineEvidencePacket(
        ticket_uid="ticket-test",
        issue_uid="issue-test",
        freeze_frame=10,
        freeze_sequence=20,
        evidence=None,
        association={
            "event_uid": "run_e00000010",
            "event_sequence": 10,
            "obs_uid": "run_f000005_r0001",
            "decision": "CREATE_OBJECT",
            "target_object_version_after": "created@v1",
        },
        alias_version_uids={
            "CANDIDATE_1_CONTEXT": "candidate-1@v1",
            "CANDIDATE_2_CONTEXT": "candidate-2@v1",
        },
        allowed_image_ids=("I01", "I02", "I03"),
        packet_manifest={},
    )


def _ledger() -> SimpleNamespace:
    return SimpleNamespace(
        object_versions={
            "created@v1": _version(
                "created", "lineage-created", "run_f000005_r0001", "pillow"
            ),
            "candidate-1@v1": _version("candidate-1", "lineage-candidate-1", "run_f000001_r0001"),
            "candidate-2@v1": _version("candidate-2", "lineage-candidate-2", "run_f000002_r0001"),
        }
    )


def _output(
    selected: str,
    axis: str = "IDENTITY",
    suggested_label: str | None = None,
) -> dict:
    return {
        "selected_candidate": selected,
        "decision_axis": axis,
        "confidence_diagnostic": 0.9,
        "evidence_ids": ["I1", "I2"],
        "reason": "bounded test reason",
        "counterevidence": "bounded test counterevidence",
        "needed_evidence": [],
        "suggested_label_for_logging": suggested_label,
    }


def test_three_view_reader_ignores_trailing_partial_record(tmp_path: Path) -> None:
    path = tmp_path / "growing.jsonl"
    path.write_bytes(b'{"complete":1}\n{"partial":')
    assert list(read_jsonl(path)) == [{"complete": 1}]


def test_single_anchor_partition_compiles_to_existing_sparse_constraint() -> None:
    candidates = [
        {
            "id": "H1",
            "axis": "IDENTITY",
            "action": "PARTITION_ALIASES",
            "parameters": {"groups": [["A", "E1"], ["E2"]]},
            "executable": True,
        }
    ]
    compiled = compile_unified_vlm_response(
        packet=_packet(),
        result={
            "status": "VALID",
            "output": _output("H1"),
            "candidates": candidates,
            "allowed_image_ids": ["I1", "I2", "I3"],
            "prompt_version": "test",
        },
        ledger=_ledger(),
    )
    assert compiled["stage"] == "BOUND_PENDING_SHADOW"
    assert compiled["candidate_constraint"]["type"] == "ASSIGN_OBSERVATION"
    assert compiled["candidate_constraint"]["target_lineage_uid"] == "lineage-candidate-1"
    assert compiled["adapter_proposal"]["action"] == "SAME_INSTANCE"
    assert compiled["citation_check"]["allowed_image_ids"] == ["I1", "I2", "I3"]


def test_compound_partition_fails_closed_instead_of_losing_an_alias() -> None:
    candidate = {
        "id": "H4",
        "axis": "IDENTITY",
        "action": "PARTITION_ALIASES",
        "parameters": {"groups": [["A", "E1", "E2"]]},
        "executable": True,
    }
    compiled = compile_unified_vlm_response(
        packet=_packet(),
        result={
            "status": "VALID",
            "output": _output("H4"),
            "candidates": [candidate],
            "allowed_image_ids": ["I1", "I2", "I3"],
            "prompt_version": "test",
        },
        ledger=_ledger(),
    )
    assert compiled["stage"] == "DEFERRED"
    assert compiled["candidate_constraint"] is None
    assert compiled["defer_reasons"] == [
        "compound_identity_partition_requires_multi_constraint_executor"
    ]


def test_no_op_is_terminal_without_starting_shadow_replay() -> None:
    candidate = {
        "id": "H0",
        "axis": "NONE",
        "action": "NO_OP",
        "parameters": {"groups": [["A"], ["E1"], ["E2"]]},
    }
    compiled = compile_unified_vlm_response(
        packet=_packet(),
        result={
            "status": "VALID",
            "output": _output("H0", axis="NONE"),
            "candidates": [candidate],
            "prompt_version": "test",
        },
        ledger=_ledger(),
    )
    assert compiled["stage"] == "NO_OP"
    assert compiled["candidate_constraint"] is None


def test_preflight_defer_never_becomes_a_constraint() -> None:
    compiled = compile_unified_vlm_response(
        packet=_packet(),
        result={
            "status": "PREFLIGHT_DEFER",
            "defer_code": "DEFER_INSUFFICIENT_VIEWS",
            "output": None,
            "candidates": [],
            "prompt_version": "test",
        },
        ledger=_ledger(),
    )
    assert compiled["stage"] == "DEFERRED"
    assert compiled["candidate_constraint"] is None
    assert compiled["defer_reasons"] == [
        "unified_vlm_not_valid:DEFER_INSUFFICIENT_VIEWS"
    ]


def test_semantic_candidate_compiles_to_guarded_relabel() -> None:
    candidate = {
        "id": "H5",
        "axis": "SEMANTIC_LABEL",
        "action": "RELABEL_ENTITY",
        "parameters": {"entity": "E0", "from_label": "L0", "to_label": "L1"},
        "label_text": {"from": "pillow", "to": "sofa chair"},
        "executable": True,
        "mode": "SEMANTIC_PILOT",
    }
    compiled = compile_unified_vlm_response(
        packet=_packet(),
        result={
            "status": "VALID",
            "output": _output(
                "H5", axis="SEMANTIC_LABEL", suggested_label="sofa chair"
            ),
            "candidates": [candidate],
            "allowed_image_ids": ["I1", "I2", "I3"],
            "prompt_version": "test",
        },
        ledger=_ledger(),
    )
    constraint = compiled["candidate_constraint"]
    assert compiled["stage"] == "BOUND_PENDING_SHADOW"
    assert constraint["type"] == "RELABEL"
    assert constraint["entity_uid"] == "created"
    assert constraint["expected_label"] == "pillow"
    assert constraint["label"] == "sofa chair"
    assert compiled["semantic_pilot"]["accuracy_validated"] is False


def test_semantic_output_requires_two_distinct_images() -> None:
    candidate = {
        "id": "H5",
        "axis": "SEMANTIC_LABEL",
        "action": "RELABEL_ENTITY",
        "parameters": {"entity": "E0", "from_label": "L0", "to_label": "L1"},
        "label_text": {"from": "pillow", "to": "sofa chair"},
        "executable": True,
    }
    output = _output(
        "H5", axis="SEMANTIC_LABEL", suggested_label="sofa chair"
    )
    output["evidence_ids"] = ["I1"]
    errors = validate_output(output, (candidate,))
    assert "semantic relabel requires at least two distinct evidence images" in errors


def test_semantic_output_label_must_exactly_match_selected_candidate() -> None:
    candidate = {
        "id": "H5",
        "axis": "SEMANTIC_LABEL",
        "action": "RELABEL_ENTITY",
        "parameters": {"entity": "E0", "from_label": "L0", "to_label": "L1"},
        "label_text": {"from": "pillow", "to": "sofa chair"},
        "executable": True,
    }
    matching = _output(
        "H5", axis="SEMANTIC_LABEL", suggested_label="sofa chair"
    )
    assert validate_output(matching, (candidate,)) == []

    mismatched = _output(
        "H5", axis="SEMANTIC_LABEL", suggested_label="couch"
    )
    errors = validate_output(mismatched, (candidate,))
    assert (
        "semantic relabel suggested_label_for_logging must exactly match label_text.to"
        in errors
    )

    compiled = compile_unified_vlm_response(
        packet=_packet(),
        result={
            "status": "VALID",
            "output": mismatched,
            "candidates": [candidate],
            "allowed_image_ids": ["I1", "I2", "I3"],
            "prompt_version": "test",
        },
        ledger=_ledger(),
    )
    assert compiled["stage"] == "DEFERRED"
    assert compiled["defer_reasons"] == [
        "semantic_output_label_not_exact_candidate"
    ]


def test_semantic_apply_changes_only_stable_label() -> None:
    state = {
        "state_hash": "membership-hash",
        "membership": {"entity-1": ["obs-1", "obs-2"]},
        "objects": [
            {
                "entity_uid": "entity-1",
                "class_name": "pillow",
                "class_histogram": {"pillow": 2},
                "member_observation_uids": ["obs-1", "obs-2"],
                "bbox_center": [0.0, 0.0, 0.0],
            }
        ],
    }
    candidate, reports = _apply_semantic_relabels(
        state,
        (
            {
                "type": "RELABEL",
                "entity_uid": "entity-1",
                "expected_label": "pillow",
                "label": "sofa chair",
            },
        ),
    )
    assert reports[0]["applied"] is True
    assert candidate["objects"][0]["class_name"] == "sofa chair"
    assert candidate["objects"][0]["class_histogram"] == {"pillow": 2}
    assert candidate["objects"][0]["bbox_center"] == [0.0, 0.0, 0.0]
    assert candidate["membership"] == state["membership"]
    assert candidate["state_hash"] == state["state_hash"]
    assert state["objects"][0]["class_name"] == "pillow"


def test_semantic_only_final_metric_gate_does_not_require_partition_change() -> None:
    metrics = {
        "object_count": 2,
        "observation_count": 4,
        "duplicate_ownership_count": 0,
        "singleton_object_rate": 0.0,
        "weighted_semantic_purity": 1.0,
        "low_purity_object_rate": 0.0,
        "invalid_geometry_object_count": 0,
        "state_hash": "same-membership",
    }
    gate = final_scene_metric_gate(
        metrics, metrics, require_partition_change=False
    )
    assert gate["partition_changed"] is True
    assert all(gate.values())


def test_empty_semantic_gate_passes_without_claiming_a_label_change() -> None:
    state = {
        "state_hash": "membership-hash",
        "membership": {"entity-1": ["obs-1"]},
        "objects": [
            {
                "entity_uid": "entity-1",
                "class_name": "chair",
                "class_histogram": {"chair": 1},
            }
        ],
    }
    gate = _semantic_relabel_gate(
        before_state=state,
        after_state=state,
        constraints=(),
        reports=(),
    )
    assert gate["pass"] is True
    assert gate["semantic_state_changed"] is False
