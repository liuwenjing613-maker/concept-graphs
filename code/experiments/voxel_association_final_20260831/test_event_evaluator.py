#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_association_events.py")
SPEC = importlib.util.spec_from_file_location("event_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_metric_perfect_ordering():
    result = MODULE.metric_values(np.asarray([0, 0, 1, 1]), np.asarray([0.0, 0.1, 0.8, 0.9]))
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["ap_lift"] == 0.5


def test_reliable_gt_rejects_background():
    row = {
        "gt_assignment_eligible": True,
        "gt_purity": 0.99,
        "gt_supported_fraction": 1.0,
        "gt_top_pixels": 100,
        "gt_top_id": 5,
        "gt_top_label": "wall",
    }
    assert not MODULE.reliable_gt(row)


def test_historical_identity_is_strictly_past():
    version_uid = "o@v1"
    versions = {version_uid: {"member_observation_uids": ["a", "b", "c", "d"]}}
    gt = {
        name: {
            "gt_assignment_eligible": True,
            "gt_purity": 0.99,
            "gt_supported_fraction": 1.0,
            "gt_top_pixels": 100,
            "gt_top_id": 7,
            "gt_top_label": "chair",
            "raw_frame": frame,
        }
        for name, frame in zip(("a", "b", "c", "d"), (0, 5, 10, 15))
    }
    identity, reason = MODULE.historical_identity(version_uid, 15, versions, gt)
    assert reason == "ok"
    assert identity is not None and identity.reliable_observations == 3
    identity, reason = MODULE.historical_identity(version_uid, 10, versions, gt)
    assert identity is None and reason == "history_lt3_reliable"


if __name__ == "__main__":
    test_metric_perfect_ordering()
    test_reliable_gt_rejects_background()
    test_historical_identity_is_strictly_past()
    print("PASS")
