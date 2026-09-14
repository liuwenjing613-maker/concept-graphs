import json

import pytest

from conceptgraph.revision.benchmark.experiment_v1 import (
    _method_equivalent,
    classify_repair_outcome,
    distribution_summary,
    selected_metric_paths,
)
from conceptgraph.revision.evaluate import evaluate_state
from scripts.run_revision_v1_global_reference import (
    _aggregate as aggregate_global,
    _runtime_comparison,
    _validate_frozen_selection_request,
)
from scripts.run_revision_v1_batch import _validate_frozen_primary_request
from scripts.run_revision_v1_live_comparisons import _aggregate as aggregate_live


def _method(*, member_f1: float = 1.0, bbox_iou: float = 1.0) -> dict:
    return {
        "membership": {"member_f1": member_f1},
        "geometry": {"bbox_iou_to_clean": bbox_iou},
        "relation": {"edge_state_match": True},
    }


def test_distribution_summary_retains_tail_and_empty_semantics() -> None:
    summary = distribution_summary([0.01, 0.02, 0.20])
    assert summary["mean"] == pytest.approx(0.23 / 3.0)
    assert summary["p50"] == pytest.approx(0.02)
    assert summary["p95"] > 0.18
    assert summary["max"] == 0.20
    assert distribution_summary([]) == {
        "mean": None,
        "p50": None,
        "p95": None,
        "max": None,
    }


def test_outcome_classification_ignores_bbox_roundoff_but_keeps_real_damage() -> None:
    numerical_noise = classify_repair_outcome(
        corrupted_method=_method(bbox_iou=0.9999999885),
        persistent_method=_method(bbox_iou=0.9999999885),
        verification_pass=True,
    )
    assert numerical_noise["damage_dimensions"]["geometry"] is False
    assert numerical_noise["pass"] is False

    repaired = classify_repair_outcome(
        corrupted_method=_method(bbox_iou=0.95),
        persistent_method=_method(bbox_iou=1.0),
        verification_pass=True,
    )
    assert repaired["damage_dimensions"]["geometry"] is True
    assert repaired["improved_dimensions"]["geometry"] is True
    assert repaired["pass"] is True


def test_outcome_rejects_candidate_only_observation_as_collateral_damage() -> None:
    clean = {
        "membership": {"clean_entity": ["obs_a"]},
        "objects": [],
        "edges": [],
    }
    corrupted = {
        "membership": {"corrupted_entity": []},
        "objects": [],
        "edges": [],
    }
    repaired_with_extra = {
        "membership": {"repaired_entity": ["obs_a", "obs_extra"]},
        "objects": [],
        "edges": [],
    }
    corrupted_metrics = evaluate_state(
        clean, corrupted, affected_observations=["obs_a"]
    )
    repaired_metrics = evaluate_state(
        clean, repaired_with_extra, affected_observations=["obs_a"]
    )

    # The affected-only score is perfect, but the global symmetric partition is not.
    assert repaired_metrics["membership"]["member_f1"] == 1.0
    assert repaired_metrics["membership_global"]["partition_exact"] is False
    assert repaired_metrics["membership_global"]["missing_from_first"] == [
        "obs_extra"
    ]

    outcome = classify_repair_outcome(
        corrupted_method=corrupted_metrics,
        persistent_method=repaired_metrics,
        verification_pass=True,
    )
    assert outcome["collateral_safe"] is False
    assert outcome["pass"] is False


def test_method_equivalence_uses_declared_absolute_tolerances_only() -> None:
    assert _method_equivalent(_method(), _method(member_f1=1.0 - 5e-10)) is False
    assert _method_equivalent(_method(), _method(bbox_iou=1.0 - 5e-7)) is True


def test_cost_metrics_distinguish_suffix_runtime_from_cold_snapshot_total() -> None:
    clean = {"membership": {}, "objects": [], "edges": []}
    candidate = {
        **clean,
        "runtime_ms": 20.0,
        "snapshot_runtime_ms": 80.0,
        "replayed_observations": 0,
        "replayed_events": 0,
        "total_events": 0,
    }

    cost = evaluate_state(clean, candidate, affected_observations=[])["cost"]

    assert cost["runtime_ms"] == 20.0
    assert cost["suffix_runtime_ms"] == 20.0
    assert cost["snapshot_runtime_ms"] == 80.0
    assert cost["cold_snapshot_plus_suffix_runtime_ms"] == 100.0

    comparison = _runtime_comparison(
        {"cost": cost}, {"runtime_ms": 200.0}
    )
    assert comparison["suffix_runtime_ratio_local_over_global"] == 0.1
    assert comparison["cold_runtime_ratio_local_over_global"] == 0.5


def test_metric_selection_uses_frozen_manifest_and_ignores_diagnostics(tmp_path) -> None:
    manifest = tmp_path / "manifests" / "cases.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps([{"case_uid": "case_b"}, {"case_uid": "case_a"}]),
        encoding="utf-8",
    )
    for uid in ("case_a", "diagnostic_only"):
        path = tmp_path / uid / "benchmark_metrics.json"
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
    paths, audit = selected_metric_paths(tmp_path)
    assert [path.parent.name for path in paths] == ["case_a"]
    assert audit == {
        "uses_frozen_manifest": True,
        "manifest_case_count": 2,
        "missing_case_uids": ["case_b"],
        "unexpected_case_uids_ignored": ["diagnostic_only"],
    }


def test_frozen_global_selection_rejects_changed_scope(tmp_path) -> None:
    primary = tmp_path / "primary" / "manifests" / "case_selection_manifest.json"
    primary.parent.mkdir(parents=True)
    primary.write_text("{}", encoding="utf-8")
    manifest = {
        "per_type": 2,
        "primary_manifest": str(primary.resolve()),
        "selected_case_uids": ["case_a"],
    }

    _validate_frozen_selection_request(
        manifest,
        per_type=2,
        primary_manifest=primary,
        cases=[{"case_uid": "case_a"}],
    )
    with pytest.raises(RuntimeError):
        _validate_frozen_selection_request(
            manifest, per_type=1, primary_manifest=primary
        )
    with pytest.raises(RuntimeError):
        _validate_frozen_selection_request(
            manifest,
            per_type=2,
            primary_manifest=primary,
            cases=[{"case_uid": "case_b"}],
        )


def test_frozen_primary_selection_rejects_changed_scope_or_case_file(tmp_path) -> None:
    base_run = tmp_path / "base"
    base_run.mkdir()
    cases = [
        {"case_uid": "split"},
        {"case_uid": "wrong"},
        {"case_uid": "merge"},
    ]
    manifest = {
        "scene": "room0",
        "seed": 20260823,
        "base_run": str(base_run.resolve()),
        "global_sparse_per_type": 0,
        "global_corruption_per_type": 0,
        "case_count": 3,
        "subsets": [
            {
                "failure_type": failure_type,
                "requested_count": 1,
                "selected_case_uids": [case["case_uid"]],
            }
            for failure_type, case in zip(
                ("FALSE_SPLIT", "WRONG_MEMBERSHIP", "FALSE_MERGE"), cases
            )
        ],
    }
    kwargs = {
        "base_run": base_run,
        "scene": "room0",
        "seed": 20260823,
        "count_per_type": 1,
        "global_sparse_per_type": 0,
        "global_corruption_per_type": 0,
    }

    _validate_frozen_primary_request(manifest, cases, **kwargs)
    with pytest.raises(RuntimeError):
        _validate_frozen_primary_request(
            manifest, cases, **(kwargs | {"count_per_type": 2})
        )
    with pytest.raises(RuntimeError):
        _validate_frozen_primary_request(manifest, list(reversed(cases)), **kwargs)


def test_global_and_live_aggregates_follow_their_frozen_manifests(tmp_path) -> None:
    global_root = tmp_path / "global"
    global_root.mkdir()
    (global_root / "cases.json").write_text(
        json.dumps([{"case_uid": "selected"}, {"case_uid": "missing"}]),
        encoding="utf-8",
    )
    global_row = {
        "case_uid": "selected",
        "failure_type": "FALSE_SPLIT",
        "pass": True,
        "checks": {
            "local_global_membership_exact": True,
            "local_global_bbox_iou_ge_0_999": True,
            "local_global_relation_exact": True,
        },
        "local_vs_global": {
            "runtime_ratio_local_over_global": 0.1,
            "relation": {"informative": True},
        },
    }
    for uid in ("selected", "diagnostic"):
        path = global_root / uid / "global_reference_metrics.json"
        path.parent.mkdir()
        path.write_text(json.dumps(global_row | {"case_uid": uid}), encoding="utf-8")
    global_result = aggregate_global(global_root)
    assert global_result["case_count"] == 1
    assert global_result["selection_integrity"]["missing_case_uids"] == ["missing"]
    assert global_result["selection_integrity"]["unexpected_case_uids_ignored"] == [
        "diagnostic"
    ]

    live_root = tmp_path / "live"
    (live_root / "comparisons").mkdir(parents=True)
    (live_root / "cases.json").write_text(
        json.dumps([{"case_uid": "selected"}, {"case_uid": "missing"}]),
        encoding="utf-8",
    )
    live_row = {
        "case_uid": "selected",
        "pass": True,
        "checks": {
            "single_injection_exact": True,
            "downstream_decision_kind_exact": True,
            "final_membership_partition_exact": True,
            "final_object_payload_exact": True,
            "postprocess_counts_exact": True,
        },
    }
    for uid in ("selected", "diagnostic"):
        (live_root / "comparisons" / f"{uid}.json").write_text(
            json.dumps(live_row | {"case_uid": uid}), encoding="utf-8"
        )
    live_result = aggregate_live(live_root)
    assert live_result["case_count"] == 1
    assert live_result["selection_integrity"]["missing_case_uids"] == ["missing"]
    assert live_result["selection_integrity"]["unexpected_case_uids_ignored"] == [
        "diagnostic"
    ]
