import json

from PIL import Image

from conceptgraph.revision.selective_commit import CalibrationArtifact
from scripts import finalize_revision_identity_selective_v0 as finalize
from scripts import schedule_revision_identity_context_evidence as scheduler


class _FakeProvenance:
    def get_observation(self, obs_uid):
        assert obs_uid == "obs"
        return {"bbox_2d": [90.0, 90.0, 110.0, 110.0]}


def _calibration():
    return CalibrationArtifact.from_mapping(
        {
            "capability": "IDENTITY",
            "feature_names": ["vlm_pairwise_preference"],
            "coefficients": [1.0],
            "intercept": 0.0,
            "commit_threshold": 0.9,
            "ready_for_automatic_commit": False,
            "fit_case_count": 0,
            "fit_positive_count": 0,
            "fit_negative_count": 0,
            "target_harm_rate": 0.05,
            "source_hashes": {"unready": "a" * 64},
        }
    )


def test_padded_local_context_is_hashable_non_generative_crop(tmp_path, monkeypatch):
    source = tmp_path / "frame.jpg"
    Image.new("RGB", (200, 200), color=(20, 40, 60)).save(source)
    monkeypatch.setattr(
        scheduler,
        "_wide_frame_source",
        lambda provenance, obs_uid: source,
    )
    destination = tmp_path / "local.png"
    audit = scheduler._freeze_padded_local_context(
        provenance=_FakeProvenance(),
        obs_uid="obs",
        destination=destination,
    )
    assert audit["source_crop_bounds_xyxy"] == [50, 50, 150, 150]
    assert audit["output_size"] == [512, 512]
    assert audit["expansion_factor"] == 5.0
    with Image.open(destination) as image:
        assert image.size == (512, 512)


def test_finalizer_defers_no_distinct_executable_repair_without_critic(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    result_path = case_dir / "case_result.frozen.json"
    private_path = case_dir / "execution.private.json"
    result_path.write_text(
        json.dumps(
            {
                "case_uid": "identity_machine_test",
                "scene_id": "office0",
                "status": "NO_DISTINCT_EXECUTABLE_REPAIR",
            }
        ),
        encoding="utf-8",
    )
    private_path.write_text(
        json.dumps({"noop_partition_hash": "partition_noop"}),
        encoding="utf-8",
    )
    decision = finalize._finalize_case(
        protocol={"critic_requests": []},
        case_row={
            "case_uid": "identity_machine_test",
            "scene_id": "office0",
            "result_path": str(result_path),
        },
        critic_results={},
        calibration=_calibration(),
    )
    assert decision["shadow_status"] == "SHADOW_NO_DISTINCT_REPAIR"
    assert decision["production_selective_decision"]["decision"] == "DEFER"
    assert (
        "no_runtime_valid_candidate"
        in decision["production_selective_decision"]["defer_reasons"]
    )
