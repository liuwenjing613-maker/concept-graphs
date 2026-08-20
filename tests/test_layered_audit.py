from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from conceptgraph.audit.layered_audit import (
    FactStore,
    Finding,
    LayeredAudit,
    _case_observation_selection,
    _select_case_candidates,
    _select_object_views,
    build_evidence_packets,
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
    assert config["version"] == "1.1.0"
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


class _FakeFacts:
    def __init__(self, tmp_path=None):
        self.context = SimpleNamespace(
            run_id="test-run",
            evidence_root=Path(tmp_path or "."),
        )
        self.obs_by_uid = {}
        self.frame_by_uid = {}
        self.ownership = {}
        self.event_by_uid = {}
        self.version_by_uid = {}
        self.versions_by_object = {}
        self._members = {}
        self._arrays = {}

    def member_observations(self, object_uid, object_version_uid=None):
        return list(self._members.get(object_uid, []))

    def array(self, ref, dtype=float):
        value = self._arrays.get(ref)
        return None if value is None else np.asarray(value, dtype=dtype)


def _obs(uid, frame_uid, *, confidence=0.5, points=10, feature=(1.0, 0.0)):
    return {
        "obs_uid": uid,
        "frame_uid": frame_uid,
        "bbox_2d": [2, 2, 20, 20],
        "confidence": confidence,
        "n_points": points,
        "processed_mask_ref": f"mask:{uid}",
        "pcd_ref": f"pcd:{uid}",
        "image_feat_ref": f"feature:{uid}",
        "_feature": np.asarray(feature, dtype=float),
    }


def test_mixed_object_views_cover_six_diagnostic_roles():
    facts = _FakeFacts()
    members = [
        _obs(f"obs-{index}", f"frame-{index:03d}", confidence=0.2 + index * 0.05, points=10 + index)
        for index in range(7)
    ]
    members[1]["n_points"] = 999
    members[2]["confidence"] = 0.99
    members[4]["_feature"] = np.asarray((0.0, 1.0))
    facts._members["object-a"] = members
    for index, item in enumerate(members):
        facts.obs_by_uid[item["obs_uid"]] = item
        facts.frame_by_uid[item["frame_uid"]] = {
            "pose": [[1, 0, 0, index if index != 5 else 100], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        }
        facts._arrays[item["image_feat_ref"]] = item["_feature"]
    selected = _select_object_views("object-a", facts, {"obs-3"}, 6)
    reasons = {reason for item in selected for reason in item["selection_reasons"]}
    assert len(selected) == 6
    assert {
        "earliest_creation_view",
        "highest_point_contribution",
        "highest_detector_confidence",
        "anomaly_trigger_view",
        "largest_semantic_conflict",
        "largest_camera_viewpoint_difference",
    } <= reasons


def test_association_selection_includes_chosen_and_all_alternate_roles():
    facts = _FakeFacts()
    suspect = _obs("obs-suspect", "frame-010")
    chosen = _obs("obs-chosen", "frame-020")
    spatial = _obs("obs-spatial", "frame-030")
    visual = _obs("obs-visual", "frame-040")
    aggregate2 = _obs("obs-aggregate2", "frame-050")
    counterfactual = _obs("obs-counterfactual", "frame-060")
    facts.obs_by_uid = {item["obs_uid"]: item for item in (suspect, chosen, spatial, visual, aggregate2, counterfactual)}
    facts._members = {
        "chosen": [chosen],
        "spatial": [spatial],
        "visual": [visual],
        "aggregate2": [aggregate2],
        "counterfactual": [counterfactual],
    }
    finding = _finding("finding-1", "ASSOC-003", "association", "RANK_CONFLICT", [])
    finding.scope = {
        "obs_uid": "obs-suspect",
        "object_uid": "chosen",
        "chosen_target_object_uid": "chosen",
        "spatial_top_object_uid": "spatial",
        "visual_top_object_uid": "visual",
        "aggregate_top_object_uids": ["chosen", "aggregate2"],
        "counterfactual_alternate_object_uid": "counterfactual",
        "alternate_object_uids": ["spatial", "visual", "aggregate2", "counterfactual"],
        "association_candidate_roles": {
            "chosen_target": "chosen",
            "spatial_top_candidate": "spatial",
            "visual_top_candidate": "visual",
            "aggregate_top1": "chosen",
            "aggregate_top2": "aggregate2",
            "counterfactual_alternate": "counterfactual",
        },
    }
    selected = _case_observation_selection(finding, facts, 6, 24)
    selected_uids = {item["observation"]["obs_uid"] for item in selected}
    assert selected_uids == {
        "obs-suspect", "obs-chosen", "obs-spatial", "obs-visual",
        "obs-aggregate2", "obs-counterfactual",
    }


def test_association_candidate_uses_recorded_object_version():
    facts = FactStore.__new__(FactStore)
    facts.object_by_uid = {"merged-object": {"member_observation_uids": ["obs-final"]}}
    facts.obs_by_uid = {
        "obs-old": {"obs_uid": "obs-old"},
        "obs-final": {"obs_uid": "obs-final"},
    }
    facts.version_by_uid = {
        "merged-object@v1": {"member_observation_uids": ["obs-old"]}
    }
    facts.versions_by_object = {
        "merged-object": [
            {"object_version_uid": "merged-object@v1", "member_observation_uids": ["obs-old"]},
            {"object_version_uid": "merged-object@v2", "member_observation_uids": []},
        ]
    }
    assert facts.member_observations("merged-object", "merged-object@v1") == [{"obs_uid": "obs-old"}]
    assert facts.member_observations("merged-object") == [{"obs_uid": "obs-final"}]


def test_inactive_candidate_falls_back_to_last_nonempty_object_version():
    facts = FactStore.__new__(FactStore)
    facts.object_by_uid = {}
    facts.obs_by_uid = {"obs-old": {"obs_uid": "obs-old"}}
    facts.version_by_uid = {}
    facts.versions_by_object = {
        "merged-object": [
            {"member_observation_uids": ["obs-old"]},
            {"member_observation_uids": []},
        ]
    }
    assert facts.member_observations("merged-object") == [{"obs_uid": "obs-old"}]


def test_packet_images_are_isolated_per_frame_and_have_timeline(tmp_path):
    facts = _FakeFacts(tmp_path)
    for index, frame_uid in enumerate(("frame-010", "frame-035")):
        rgb_path = tmp_path / f"{frame_uid}.jpg"
        depth_path = tmp_path / f"{frame_uid}.png"
        Image.new("RGB", (32, 32), (20 + index * 100, 40, 60)).save(rgb_path)
        Image.fromarray(np.full((32, 32), 1000 + index * 100, dtype=np.uint16)).save(depth_path)
        facts.frame_by_uid[frame_uid] = {
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "pose": np.eye(4).tolist(),
        }
    left = _obs("obs-left", "frame-010")
    right = _obs("obs-right", "frame-035")
    facts.obs_by_uid = {left["obs_uid"]: left, right["obs_uid"]: right}
    for item in (left, right):
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:20, 2:20] = True
        facts._arrays[item["processed_mask_ref"]] = mask
        facts._arrays[item["pcd_ref"]] = np.asarray([[0.0, 0.0, 0.0]])
        facts._arrays[item["image_feat_ref"]] = np.asarray([1.0, 0.0])
    finding = _finding("finding_000001", "GEO-004", "geometry", "CROSS_FRAME", [])
    finding.scope = {"obs_uids": ["obs-left", "obs-right"]}
    config = {
        "thresholds": {"geometry": {"center_jump_normalized": 1.0}},
        "case_builder": {
            "enabled": True,
            "max_cases": 1,
            "max_images_per_object": 6,
            "max_total_images_per_case": 24,
            "calibration_fraction": 1.0,
            "save_depth_overlay": True,
            "save_3d_overlay": False,
        },
    }
    result = build_evidence_packets([finding], facts, tmp_path / "audit_v1", config)
    case_dir = tmp_path / "audit_v1" / "cases" / finding.finding_uid
    assert result["built"] == 1
    assert (case_dir / "overview_frame-010.jpg").exists()
    assert (case_dir / "overview_frame-035.jpg").exists()
    assert (case_dir / "mask_overlay_frame-010.png").exists()
    assert (case_dir / "mask_overlay_frame-035.png").exists()
    assert (case_dir / "depth_frame-010.png").exists()
    assert (case_dir / "depth_frame-035.png").exists()
    assert (case_dir / "timeline.jpg").exists()
    assert not (case_dir / "overview.jpg").exists()
    assert not (case_dir / "mask_overlay.png").exists()


def test_dual_cohort_sampling_is_stratified_and_not_execution_order_biased():
    facts = _FakeFacts()
    findings = []
    for index in range(100):
        item = _finding(f"det-{index:03d}", "DET-001", "detection", "DUPLICATE", [{"name": "duplicate", "support": ["2d", "3d"]}])
        item.scope = {"obs_uid": f"det-obs-{index:03d}"}
        findings.append(item)
    for index in range(5):
        item = _finding(f"geo-{index:03d}", "GEO-004", "geometry", "CROSS_FRAME", [{"name": "geometry", "support": ["jump"]}])
        item.scope = {"obs_uid": f"geo-obs-{index:03d}"}
        findings.append(item)
    config = {
        "thresholds": {"duplicate_proposal": {}, "geometry": {}},
        "case_builder": {
            "calibration_fraction": 0.5,
            "min_calibration_per_stratum": 1,
            "min_priority_per_checker": 1,
            "max_priority_per_checker": 3,
            "max_priority_cases_per_entity": 1,
            "random_seed": 7,
        },
    }
    selected, manifest = _select_case_candidates(findings, facts, config, 10)
    assert len(selected) == 10
    assert set(manifest["selected_checker_counts"]) == {"DET-001", "GEO-004"}
    assert manifest["selected_cohort_counts"]["calibration_random"] == 5
    assert manifest["selected_cohort_counts"]["diagnostic_priority"] == 5
    assert manifest["cohorts"]["calibration_random"]["strata"]
    assert all(item.review_score is not None and item.review_priority is not None for item in findings)


def test_finding_cap_suppression_is_counted_and_fails_validation_gate():
    context = SimpleNamespace(
        audit_config={
            "limits": {
                "max_findings_per_rule": 2,
                "fail_if_population_censored": True,
            }
        },
        config={},
        policy={},
        environment_mode="static",
        policy_source="test",
        run_id="test-run",
        scene_id="room0",
    )
    engine = LayeredAudit(context, _FakeFacts())
    for _ in range(3):
        engine.add(
            "DET-001",
            "detection",
            "DUPLICATE",
            "AMBIGUOUS_MAPPING_RISK",
        )

    summary = engine._summary(gate_passed=True, root_causes=[], elapsed=0.0)

    assert summary["finding_population"]["DET-001"] == {
        "attempted_count": 3,
        "emitted_count": 2,
        "suppressed_count": 1,
        "population_censored": True,
    }
    assert summary["population_censoring_status"] == "FAIL"
    assert summary["weighted_precision_allowed"] is False
    assert summary["validation_gate_status"] == "FAIL"


def test_censored_population_refuses_sampling_weights():
    facts = _FakeFacts()
    findings = [
        _finding(
            f"det-{index:03d}",
            "DET-001",
            "detection",
            "DUPLICATE",
            [{"name": "duplicate", "support": ["2d", "3d"]}],
        )
        for index in range(10)
    ]
    config = {
        "thresholds": {"duplicate_proposal": {}},
        "case_builder": {
            "calibration_fraction": 0.5,
            "min_calibration_per_stratum": 1,
            "min_priority_per_checker": 1,
            "max_priority_per_checker": 10,
            "max_priority_cases_per_entity": 20,
            "random_seed": 7,
        },
    }
    population_report = {
        "DET-001": {
            "attempted_count": 20,
            "emitted_count": 10,
            "suppressed_count": 10,
            "population_censored": True,
        }
    }

    selected, manifest = _select_case_candidates(
        findings, facts, config, 8, population_report
    )

    calibration = [
        item
        for item in selected
        if item.selection_cohort == "calibration_random"
    ]
    assert calibration
    assert manifest["weighted_precision_allowed"] is False
    assert manifest["population_censoring"]["status"] == "FAIL"
    assert "forbidden" in manifest["precision_policy"].lower()
    assert all(item.selection_probability is None for item in calibration)
    assert all(item.sampling_weight is None for item in calibration)
