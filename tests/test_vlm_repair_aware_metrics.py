from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_vlm_repair_aware_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("repair_aware_summary", SCRIPT)
assert SPEC and SPEC.loader
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _object_payload(count: int, matched: int, hits1: int, hits5: int):
    metrics = {
        "count": count,
        "matched_geometry": matched,
        "hits_at_1": hits1,
        "hits_at_5": hits5,
        "per_class": {
            "chair": {
                "count": count,
                "hits_at_1": hits1,
                "hits_at_5": hits5,
            }
        },
    }
    return {
        "object": {
            "main_native_clip_ft": metrics,
            "map_class_name": metrics,
        },
        "integrity": {
            "gt_objects": count,
            "predicted_objects": count + 2,
            "geometry_matched_gt_objects": matched,
            "geometry_valid_predicted_objects": matched + 1,
            "geometry_gt_objects_with_multiple_fragments": 1,
            "geometry_fragmentation_excess": 1,
        },
    }


def test_pool_object_method_recomputes_pooled_recall():
    payloads = {
        "a": _object_payload(4, 3, 2, 3),
        "b": _object_payload(6, 4, 1, 2),
    }
    pooled = summary.pool_object_method(payloads, "map_class_name")
    assert pooled["gt_objects"] == 10
    assert pooled["geometry_matched"] == 7
    assert pooled["recall_at_1"] == pytest.approx(0.3)
    assert pooled["recall_at_5"] == pytest.approx(0.5)
    assert pooled["mean_recall_at_1"] == pytest.approx(0.3)


def test_pool_geometry_labels_prediction_metrics_as_closed_scope():
    payloads = {
        "a": _object_payload(4, 3, 2, 3),
        "b": _object_payload(6, 4, 1, 2),
    }
    pooled = summary.pool_geometry(payloads)
    assert pooled["gt_objects"] == 10
    assert pooled["predicted_objects"] == 14
    assert pooled["covered_gt_objects"] == 7
    assert pooled["geometry_valid_predicted_objects"] == 9
    assert pooled["fragmentation_excess"] == 2
    assert "not unrestricted real-world precision" in pooled["scope_warning"]
