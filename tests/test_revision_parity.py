from pathlib import Path

import yaml


def test_revision_feature_flag_is_disabled_by_default():
    config = yaml.safe_load(
        Path("conceptgraph/hydra_configs/rerun_realtime_mapping.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["revision"]["enabled"] is False
    assert config["revision"]["corruption_plan"] is None


def test_mapping_hook_is_guarded_by_controller():
    source = Path("conceptgraph/slam/rerun_realtime_mapping.py").read_text(encoding="utf-8")
    guarded = "if corruption_controller is not None:\n            match_indices = corruption_controller.apply"
    assert guarded in source
