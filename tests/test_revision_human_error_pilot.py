import json
from copy import deepcopy
from pathlib import Path

import pytest

from conceptgraph.revision.constraints import SparseRepairConstraint
from conceptgraph.revision.benchmark.human_error_pilot import (
    _mechanism_trace,
    evaluate_collateral,
    evaluate_endpoint_groups,
    validate_manifest,
)


MANIFEST = (
    Path(__file__).parents[1]
    / "docs"
    / "revision_v1_audits"
    / "HUMAN_CONFIRMED_SIX_CAUSAL_PILOT_MANIFEST_20260824.json"
)


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_human_pilot_has_exact_balanced_cohort_and_dispositions():
    result = validate_manifest(_manifest())

    assert result["pass"]
    assert result["case_count"] == 6
    assert result["endpoint_error_type_counts"] == {
        "FALSE_MERGE": 3,
        "FALSE_SPLIT": 3,
    }
    assert result["causal_disposition_counts"] == {
        "DEFER_NON_ASSOCIATION_ROOT": 3,
        "REPLAYABLE_ASSOCIATION_CAUSE": 3,
    }
    assert result["population_inference_to_all_confirmed_errors"] is False


def test_human_pilot_rejects_outcome_friendly_case_replacement():
    manifest = deepcopy(_manifest())
    manifest["cases"].pop()

    with pytest.raises(ValueError, match="exactly six"):
        validate_manifest(manifest)


def test_endpoint_criterion_distinguishes_false_merge_and_false_split():
    merged = {"entity_ab": ["a1", "a2", "b1"]}
    separated = {"entity_a": ["a1", "a2"], "entity_b": ["b1"]}
    groups = {"a": ["a1", "a2"], "b": ["b1"]}

    assert not evaluate_endpoint_groups(
        merged, groups, "DIFFERENT_OWNER"
    )["correct"]
    assert evaluate_endpoint_groups(
        separated, groups, "DIFFERENT_OWNER"
    )["correct"]
    assert not evaluate_endpoint_groups(separated, groups, "SAME_OWNER")["correct"]
    assert evaluate_endpoint_groups(merged, groups, "SAME_OWNER")["correct"]


def test_collateral_metric_is_entity_id_invariant_and_detects_boundary_pollution():
    native = {"native_a": ["a1", "a2"], "native_x": ["x1", "x2"]}
    safe_repair = {"new_a1": ["a1"], "new_a2": ["a2"], "renamed_x": ["x1", "x2"]}
    polluted = {"new_a1": ["a1", "x1", "x2"], "new_a2": ["a2"]}

    safe = evaluate_collateral(native, safe_repair, {"a1", "a2"})
    unsafe = evaluate_collateral(native, polluted, {"a1", "a2"})

    assert safe["safe"]
    assert safe["outside_partition_exact_to_native"]
    assert not unsafe["safe"]
    assert unsafe["cross_boundary_entity_uids"] == ["new_a1"]


def test_false_split_mechanism_requires_persistent_redirect_override():
    primitive = SparseRepairConstraint.from_mapping(
        {
            "type": "ASSIGN_OBSERVATION",
            "obs_uid": "obs_anchor",
            "target_lineage_uid": "lineage_target",
            "applies_at_event_uid": "event_anchor",
        }
    )
    state = {
        "persistent_create_instance_merge_veto_count": 0,
        "persistent_create_instance_association_veto_count": 0,
        "persistent_lineage_redirect_count": 1,
        "persistent_lineage_redirect_override_count": 1,
        "decision_trace": [
            {
                "obs_uid": "obs_anchor",
                "constraint": {
                    "action": "FORCE_TARGET",
                    "reason": "explicit_positive_constraint",
                },
            },
            {
                "obs_uid": "obs_later",
                "natural_match": None,
                "historical_default_match": None,
                "applied_match": 3,
                "persistent_lineage_redirect_source_lineages": [
                    "lineage_duplicate"
                ],
                "constraint": {
                    "action": "FORCE_TARGET",
                    "reason": "persistent_lineage_redirect",
                },
            },
        ],
        "postprocess_decision_trace": [],
    }

    mechanism = _mechanism_trace(state, primitive)

    assert mechanism["verified"]
    assert mechanism["persistent_lineage_redirects"][0]["obs_uid"] == "obs_later"

    state["persistent_lineage_redirect_override_count"] = 0
    assert not _mechanism_trace(state, primitive)["verified"]
