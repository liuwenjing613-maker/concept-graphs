from pathlib import Path

from conceptgraph.audit.layered_audit import (
    Finding,
    load_audit_config,
    resolve_root_causes,
)


def _finding(uid, checker, stage, subtype, hypotheses):
    return Finding(
        finding_uid=uid,
        checker_id=checker,
        stage=stage,
        subtype=subtype,
        scope={"object_uid": "object-a"},
        certainty="AMBIGUOUS_MAPPING_RISK",
        severity="HIGH",
        policy_context={"environment_mode": "static"},
        proven_facts=[{"name": "metric", "value": 1.0}],
        hypotheses=hypotheses,
        vetoes=["new_viewpoint_not_excluded"],
        missing_evidence=["viewpoint_coverage"],
        route="VLM_REVIEW",
    )


def test_v1_config_is_versioned_and_stage_aware():
    path = Path(__file__).parents[1] / "conceptgraph/audit/configs/v1.yaml"
    config = load_audit_config(path)
    assert config["version"] == "1.0.0"
    assert config["policy"]["missing_evidence_policy"] == "unknown_not_pass"
    assert config["enabled_checkers"]["association"] is True
    assert config["thresholds"]["duplicate_proposal"]["mask_iou"] == 0.85


def test_root_cause_prefers_earliest_stage_and_preserves_uncertainty():
    findings = [
        _finding(
            "finding_000001",
            "SEG-002",
            "segmentation",
            "BACKGROUND_LEAKAGE",
            [{"name": "segmentation_background_leakage", "support": ["multi_cluster"]}],
        ),
        _finding(
            "finding_000002",
            "ASSOC-006",
            "association",
            "LOW_GEOMETRIC_SUPPORT",
            [{"name": "false_association", "support": ["low_geometry"]}],
        ),
    ]
    roots = resolve_root_causes(findings)
    assert len(roots) == 1
    assert roots[0]["primary_stage"] == "segmentation"
    assert roots[0]["primary_hypothesis"] == "segmentation_background_leakage"
    assert roots[0]["supporting_findings"] == [
        "finding_000001",
        "finding_000002",
    ]
    assert "new_viewpoint_not_excluded" in roots[0]["vetoes"]
    assert "viewpoint_coverage" in roots[0]["missing_evidence"]


def test_finding_keeps_fact_hypothesis_veto_and_missing_evidence_separate():
    finding = _finding(
        "finding_000003",
        "OBJ-002",
        "object_identity",
        "POSSIBLE_DUPLICATE_OBJECT",
        [{"name": "false_split", "support": ["3d_overlap"]}],
    ).to_dict()
    assert finding["proven_facts"] == [{"name": "metric", "value": 1.0}]
    assert finding["hypotheses"][0]["name"] == "false_split"
    assert finding["vetoes"] == ["new_viewpoint_not_excluded"]
    assert finding["missing_evidence"] == ["viewpoint_coverage"]
    assert finding["repair_allowed"] is False
