import json
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np

from conceptgraph.slam.association_gate import (
    ASSOCIATION_SYSTEM_PROMPT,
    BlockingAssociationGate,
    CREATE_SYSTEM_PROMPT,
    DISCARD_MATCH_INDEX,
    _image_data_url,
    _image_media_descriptor,
    _write_rgb,
    compute_trigger,
    deduplicate_ranked_candidates,
    route_choice,
)


def test_trigger_rules_create_only():
    association = compute_trigger([1.31, 1.18, 0.2], 0, 1.2, 0.20, 0.30, "create_only")
    assert association is not None
    assert association["kind"] == "association"
    assert abs(association["margin"] - 0.13) < 1e-8

    create = compute_trigger([1.05, 0.7], None, 1.2, 0.20, 0.30, "create_only")
    assert create is not None
    assert create["kind"] == "create"

    assert compute_trigger([0.7, 0.5], None, 1.2, 0.20, 0.30, "create_only") is None
    assert compute_trigger([1.6, 1.2], 0, 1.2, 0.20, 0.30, "create_only") is None


def test_route_choice_fallback():
    aliases = {"A": 7, "B": 9}
    assert route_choice("B", aliases, 7) == (9, "model_candidate")
    assert route_choice("NEW", aliases, 7) == (None, "model_new")
    assert route_choice("DISCARD", aliases, 7) == (DISCARD_MATCH_INDEX, "model_discard_observation")
    assert route_choice("UNCERTAIN", aliases, 7) == (7, "fallback_baseline")


def test_both_prompts_use_the_same_conservative_discard_policy():
    for prompt in (ASSOCIATION_SYSTEM_PROMPT, CREATE_SYSTEM_PROMPT):
        assert "semi-transparent RED overlay" in prompt
        assert "unhighlighted pixels are background/context only" in prompt
        assert "Never select a candidate merely because" in prompt
        assert "Dark letterbox padding" in prompt
        assert "INSTANCE re-identification, not category classification" in prompt
        assert "Category agreement is only a weak contextual cue and is never sufficient" in prompt
        assert "category or label disagreement alone is not decisive" in prompt
        assert "same individual physical object" in prompt
        assert "matching class, label, or object type is invalid" in prompt
        assert "repeated or near-identical objects" in prompt
        assert "MIXED_INSTANCES" in prompt
        assert "SEVERE_FRAGMENT" in prompt
        assert "mandatory CURRENT-observation quality gate" in prompt
        assert "use only I1 and I1-crop" in prompt
        assert "excluding masks that are fragmented merely due to occlusion" in prompt
        assert "BORDERLINE => choose UNCERTAIN and stop" in prompt
        assert "Only USABLE permits candidate comparison" in prompt
        assert "ordinary occlusion" in prompt.lower()
        assert "Candidate-image weakness is not a reason" in prompt
        assert "DISCARD" in prompt and "UNCERTAIN" in prompt
    _, association_user = BlockingAssociationGate._prompts("association", ["A", "B"])
    _, create_user = BlockingAssociationGate._prompts("create", ["A", "B", "C"])
    assert "DISCARD" in association_user
    assert "DISCARD" in create_user
    assert "judge the red-masked target" in association_user
    assert "judge the red-masked target" in create_user
    assert "using only I1 and I1-crop before looking at candidates" in association_user
    assert "using only I1 and I1-crop before looking at candidates" in create_user


def test_vlm_media_is_validated_jpeg(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    jpeg = tmp_path / "evidence.jpg"
    image = np.full((24, 32, 3), 127, dtype=np.uint8)
    _write_rgb(jpeg, image)
    data_url = _image_data_url(jpeg)
    descriptor = _image_media_descriptor(jpeg)
    assert data_url.startswith("data:image/jpeg;base64,")
    assert descriptor["mime_type"] == "image/jpeg"
    assert descriptor["jpeg_magic_hex"] == "ffd8ff"
    assert descriptor["width"] == 1024 and descriptor["height"] == 1024

    tiny = tmp_path / "tiny.jpg"
    assert cv2.imwrite(str(tiny), image)
    try:
        _image_data_url(tiny)
    except ValueError as exc:
        assert "smaller than 512px" in str(exc)
    else:
        raise AssertionError("tiny raster was not rejected")

    svg = tmp_path / "bad.jpg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    try:
        _image_data_url(svg)
    except ValueError as exc:
        assert "not a JPEG bitstream" in str(exc)
    else:
        raise AssertionError("SVG payload was not rejected")


def test_iou_prefilter_keeps_highest_score_and_changes_top2(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jpg"
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    duplicate_high = _object(source, 0, 0, 5)
    duplicate_low = _object(source, 0, 1, 5)
    distinct = _object(source, 0, 2, 55)
    kept, diagnostics = deduplicate_ranked_candidates(
        [1.31, 1.30, 1.18], [duplicate_high, duplicate_low, distinct], 0.98, 3,
    )
    assert kept == [0, 2]
    assert len(diagnostics["dropped"]) == 1
    assert diagnostics["dropped"][0]["object_index"] == 1
    assert diagnostics["dropped"][0]["representative_object_index"] == 0
    assert diagnostics["dropped"][0]["same_frame_mask_iou"] == 1.0
    filtered_trigger = compute_trigger([1.31, 1.18], 0, 1.2, 0.20, 0.30, "create_only")
    assert filtered_trigger is not None and filtered_trigger["kind"] == "association"
    assert abs(filtered_trigger["margin"] - 0.13) < 1e-8

    kept_one, _ = deduplicate_ranked_candidates(
        [1.31, 1.30], [duplicate_high, duplicate_low], 0.98, 3,
    )
    assert kept_one == [0]
    assert compute_trigger([1.31], 0, 1.2, 0.20, 0.30, "create_only") is None


def _object(path: Path, frame: int, raw_idx: int, x1: int) -> dict:
    mask = np.zeros((64, 96), dtype=bool)
    mask[12:52, x1:x1 + 25] = True
    return {
        "id": uuid.uuid4(),
        "class_name": "chair",
        "num_detections": 1,
        "color_path": [path],
        "mask": [mask],
        "xyxy": [[x1, 12, x1 + 25, 52]],
        "image_idx": [frame],
        "obs_uids": [f"run_f{frame:06d}_r{raw_idx:04d}"],
    }


def test_audit_writes_evidence_but_keeps_baseline(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jpg"
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    objects = [_object(source, 0, 0, 5), _object(source, 0, 1, 55)]
    detection = _object(source, 5, 3, 8)
    cfg = {
        "sim_threshold": 1.2,
        "association_gate": {
            "mode": "audit",
            "margin_threshold": 0.20,
            "threshold_distance": 0.30,
            "threshold_scope": "create_only",
        },
    }
    gate = BlockingAssociationGate(cfg=cfg, output_dir=tmp_path / "gate")
    routed = gate.route_frame(
        frame_idx=5,
        source_frame_id="000005",
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        detection_list=[detection],
        objects=objects,
        aggregate_sim=np.array([[1.31, 1.18]], dtype=np.float32),
        baseline_match_indices=[0],
    )
    gate.close()
    assert routed == [0]
    rows = [json.loads(line) for line in (tmp_path / "gate" / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["changed"] is False
    assert rows[0]["h_snapshot_uid"] == rows[0]["c_bound_h_snapshot_uid"]
    assert rows[0]["candidate_object_uids_distinct"] is True
    event_dir = tmp_path / "gate" / "events" / rows[0]["event_id"]
    assert (event_dir / "current_context.jpg").is_file()
    assert (event_dir / "candidate_A.jpg").is_file()
    assert (tmp_path / "gate" / "index.html").is_file()


def test_off_is_identity(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    gate = BlockingAssociationGate(
        cfg={"sim_threshold": 1.2, "association_gate": {"mode": "off"}},
        output_dir=tmp_path / "off",
    )
    baseline = [0, None]
    assert gate.route_frame(
        frame_idx=0,
        source_frame_id="0",
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        detection_list=[{}, {}],
        objects=[{}],
        aggregate_sim=np.array([[1.3], [0.9]]),
        baseline_match_indices=baseline,
    ) == baseline


def test_vlm_discard_is_a_terminal_non_mapping_action(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jpg"
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    objects = [_object(source, 0, 0, 5), _object(source, 0, 1, 55)]
    detection = _object(source, 5, 3, 8)
    gate = BlockingAssociationGate(
        cfg={
            "sim_threshold": 1.2,
            "association_gate": {
                "mode": "vlm",
                "api_key_required": False,
                "margin_threshold": 0.20,
                "candidate_iou_filter_enabled": False,
            },
        },
        output_dir=tmp_path / "gate",
    )
    gate._call_vlm = lambda payload, **kwargs: (
        {"mock": True},
        {
            "observation_quality": "MIXED_INSTANCES",
            "choice": "DISCARD",
            "confidence": 0.98,
            "reason": "MIXED_INSTANCES",
        },
        0.01,
    )
    routed = gate.route_frame(
        frame_idx=5,
        source_frame_id="5",
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        detection_list=[detection],
        objects=objects,
        aggregate_sim=np.array([[1.31, 1.18]], dtype=np.float32),
        baseline_match_indices=[0],
    )
    gate.close()
    event_dir = next((tmp_path / "gate" / "events").iterdir())
    decision = json.loads((event_dir / "decision.json").read_text())
    request = json.loads((event_dir / "actual_request_redacted.json").read_text())
    response_schema = request["response_format"]["json_schema"]["schema"]
    choices = response_schema["properties"]["choice"]["enum"]
    qualities = response_schema["properties"]["observation_quality"]["enum"]
    assert routed == [DISCARD_MATCH_INDEX]
    assert decision["route_reason"] == "model_discard_observation"
    assert decision["final_match_index"] == DISCARD_MATCH_INDEX
    assert decision["changed"] is True
    assert "DISCARD" in choices
    assert set(qualities) == {"USABLE", "BORDERLINE", "MIXED_INSTANCES", "SEVERE_FRAGMENT"}
    assert "observation_quality" in response_schema["required"]


def test_oracle_uses_processed_frame_index(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jpg"
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    gt_rows = [
        {"frame_idx": 0, "raw_frame": 0, "obs_uid": "old_f000000_r0000", "gt_assignment_eligible": True, "gt_purity": 1.0, "gt_top_id": 10},
        {"frame_idx": 0, "raw_frame": 0, "obs_uid": "old_f000000_r0001", "gt_assignment_eligible": True, "gt_purity": 1.0, "gt_top_id": 20},
        {"frame_idx": 5, "raw_frame": 25, "obs_uid": "new_f000005_r0003", "gt_assignment_eligible": True, "gt_purity": 1.0, "gt_top_id": 20},
    ]
    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text("".join(json.dumps(row) + "\n" for row in gt_rows), encoding="utf-8")
    objects = [_object(source, 0, 0, 5), _object(source, 0, 1, 55)]
    detection = _object(source, 5, 3, 8)
    gate = BlockingAssociationGate(
        cfg={
            "sim_threshold": 1.2,
            "association_gate": {
                "mode": "oracle",
                "margin_threshold": 0.20,
                "threshold_distance": 0.30,
                "oracle_gt_path": str(gt_path),
            },
        },
        output_dir=tmp_path / "gate",
    )
    routed = gate.route_frame(
        frame_idx=5,
        source_frame_id="25",
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        detection_list=[detection],
        objects=objects,
        aggregate_sim=np.array([[1.31, 1.18]], dtype=np.float32),
        baseline_match_indices=[0],
    )
    assert routed == [1]


if __name__ == "__main__":
    test_trigger_rules_create_only()
    test_route_choice_fallback()
    test_both_prompts_use_the_same_conservative_discard_policy()
    with tempfile.TemporaryDirectory(prefix="association_gate_test_") as temp_dir:
        root = Path(temp_dir)
        test_vlm_media_is_validated_jpeg(root / "media_case")
        test_iou_prefilter_keeps_highest_score_and_changes_top2(root / "iou_case")
        test_audit_writes_evidence_but_keeps_baseline(root / "audit_case")
        test_off_is_identity(root / "off_case")
        test_vlm_discard_is_a_terminal_non_mapping_action(root / "discard_case")
        test_oracle_uses_processed_frame_index(root / "oracle_case")
    print("9 association-gate tests passed")
