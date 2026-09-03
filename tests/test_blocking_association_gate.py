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
    POINT_CLOUD_READING_POLICY,
    _candidate_evidence_composite,
    _image_data_url,
    _image_media_descriptor,
    _shared_projection_ranges,
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


def test_prompts_lock_red_mask_target_and_keep_simple_identity_actions():
    for prompt in (ASSOCIATION_SYSTEM_PROMPT, CREATE_SYSTEM_PROMPT):
        assert "CURRENT CONTEXT" in prompt
        assert "H1/H2/H3 are frozen historical observations" in prompt
        assert "MANDATORY TARGET-LOCK RULE" in prompt
        assert "red mask actually covers that object" in prompt
        assert "Once a target is locked, never switch attention" in prompt
        assert "Use unmasked pixels only to understand spatial context" in prompt
        assert "Never select a candidate merely because" in prompt
        assert "physical INSTANCE matching, not category recognition" in prompt
        assert "Category agreement is weak evidence" in prompt
        assert "Category disagreement is also not decisive" in prompt
        assert "same individual physical object" in prompt
        assert "Repeated or adjacent objects require stronger evidence" in prompt
        assert "XY, XZ, and YZ are three orthographic projections" in prompt
        assert "same online world coordinate system" in prompt
        assert "same world bounds and metric scale" in prompt
        assert "centroid proximity alone" in prompt
        assert "UNCERTAIN" in prompt
        assert "DISCARD" not in prompt
        assert "observation_quality" not in prompt
    _, association_user = BlockingAssociationGate._prompts("association", ["A", "B"])
    _, create_user = BlockingAssociationGate._prompts("create", ["A", "B", "C"])
    assert "A, B, NEW, UNCERTAIN" in association_user
    assert "A, B, C, NEW, UNCERTAIN" in create_user
    assert "DISCARD" not in association_user + create_user
    assert "up to three historical red-mask views" in association_user
    assert "shared-scale XY/XZ/YZ" in create_user


def test_candidate_card_contains_rgb_and_annotation_style_3d_views():
    histories = [np.full((180, 140 + index * 20, 3), 80 + index * 20, dtype=np.uint8) for index in range(3)]
    current = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
    candidate = np.array([[0.0, 0.0, 0.0], [0.12, 0.22, 0.32], [0.25, 0.12, 0.42]])
    ranges = _shared_projection_ranges([current, candidate, candidate + 4.0])
    card = _candidate_evidence_composite(
        histories, ["H1 BEST MASK", "H2 RECENT", "H3 DIVERSE"],
        current, candidate, "A", ranges,
    )
    assert card.shape == (1024, 1024, 3)
    assert np.any((card[:, :, 0] > 180) & (card[:, :, 1] < 110))  # magenta current
    assert np.any((card[:, :, 0] < 80) & (card[:, :, 1] > 120) & (card[:, :, 2] > 140))  # cyan candidate
    assert len(ranges) == 3
    assert all(extent > 4.0 for _, _, extent in ranges)
    assert "same world bounds and metric scale" in POINT_CLOUD_READING_POLICY


def test_three_representative_history_members(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source), np.full((64, 96, 3), 120, dtype=np.uint8))
    obj = _object(source, 0, 0, 5)
    masks = []
    for width in (10, 30, 14, 20, 25):
        mask = np.zeros((64, 96), dtype=bool)
        mask[10:40, 5:5 + width] = True
        masks.append(mask)
    obj.update({
        "color_path": [source] * 5,
        "mask": masks,
        "xyxy": [[5, 10, 5 + width, 40] for width in (10, 30, 14, 20, 25)],
        "image_idx": [0, 2, 4, 6, 8],
        "obs_uids": [f"run_f{frame:06d}_r0000" for frame in (0, 2, 4, 6, 8)],
        "conf": [0.8, 0.7, 0.99, 0.85, 0.9],
        "num_detections": 5,
    })
    selected = BlockingAssociationGate._representative_members(obj)
    assert [role for role, _ in selected] == ["H1 BEST MASK", "H2 RECENT", "H3 DIVERSE"]
    assert len({index for _, index in selected}) == 3
    assert selected[0][1] == 1  # largest processed mask
    assert selected[1][1] == 4  # latest view with a sufficiently large mask


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
        "pcd_np": np.array([
            [x1 / 20.0, 0.00, 0.00],
            [x1 / 20.0 + 0.10, 0.20, 0.30],
            [x1 / 20.0 + 0.20, 0.10, 0.40],
        ], dtype=np.float64),
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
    card = cv2.imread(str(event_dir / "candidate_A.jpg"), cv2.IMREAD_COLOR)
    assert card is not None and card.shape[:2] == (1024, 1024)
    decision = json.loads((event_dir / "decision.json").read_text())
    candidate_evidence = next(item for item in decision["evidence"] if item["role"] == "candidate-A")
    candidate_b = next(item for item in decision["evidence"] if item["role"] == "candidate-B")
    assert len(candidate_evidence["selected_history"]) == 1
    assert candidate_evidence["point_cloud_projections"] == ["XY", "XZ", "YZ"]
    assert candidate_evidence["point_cloud_sources"] == {"current": "detection['pcd']", "candidate": "obj['pcd']"}
    assert candidate_evidence["point_cloud_event_shared_ranges_uid"] == candidate_b["point_cloud_event_shared_ranges_uid"]
    assert candidate_evidence["current_point_count_rendered"] == 3
    assert candidate_evidence["candidate_point_count_rendered"] == 3
    assert (tmp_path / "gate" / "index.html").is_file()
    case_id = decision["human_annotation_case_id"]
    blind_case = tmp_path / "gate" / "human_annotation_blind" / "cases" / case_id
    assert (blind_case / "case.json").is_file()
    assert (blind_case / "candidate_A.jpg").is_file()
    assert not any("vlm" in path.name.lower() or path.name == "decision.json" for path in blind_case.iterdir())
    blind_payload = json.loads((blind_case / "case.json").read_text())
    assert "event_id" not in blind_payload and "baseline" not in json.dumps(blind_payload).lower()
    assert blind_payload["allowed_choices"] == ["A", "B", "NEW", "UNCERTAIN"]
    annotation_index = (
        tmp_path / "gate" / "human_annotation_blind" / "index.html"
    ).read_text(encoding="utf-8")
    assert "下一例/跳过" in annotation_index
    assert "下一未标注" in annotation_index
    assert "导出已标注 JSONL" in annotation_index
    assert "localStorage" in annotation_index
    assert "class=\"choice" in annotation_index
    assert "http-equiv=\"refresh\"" not in annotation_index
    assert "vlm_output.json" not in annotation_index
    assert "decision.json" not in annotation_index
    annotation_readme = (
        tmp_path / "gate" / "human_annotation_blind" / "README.md"
    ).read_text(encoding="utf-8")
    assert "Open `index.html`" in annotation_readme
    assert "You may skip any case" in annotation_readme
    assert "Export labeled JSONL" in annotation_readme
    assert (tmp_path / "gate" / "human_annotation_case_map.jsonl").is_file()


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


def test_discard_route_is_retained_but_not_a_formal_vlm_action(tmp_path: Path):
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
            "candidate_assessments": [
                {"code": "A", "relation": "SAME", "evidence": "matching history and 3D"},
                {"code": "B", "relation": "DIFFERENT", "evidence": "separated in 3D"},
            ],
            "choice": "A",
            "confidence": 0.98,
            "reason": "A is supported by both evidence types",
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
    assert routed == [0]
    assert decision["route_reason"] == "model_candidate"
    assert decision["final_match_index"] == 0
    assert "DISCARD" not in choices
    assert "observation_quality" not in response_schema["properties"]
    assert "candidate_assessments" in response_schema["required"]
    assert route_choice("DISCARD", {"A": 0, "B": 1}, 0) == (
        DISCARD_MATCH_INDEX, "model_discard_observation",
    )


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
    test_prompts_lock_red_mask_target_and_keep_simple_identity_actions()
    test_candidate_card_contains_rgb_and_annotation_style_3d_views()
    with tempfile.TemporaryDirectory(prefix="association_gate_test_") as temp_dir:
        root = Path(temp_dir)
        test_three_representative_history_members(root / "history_case")
        test_vlm_media_is_validated_jpeg(root / "media_case")
        test_iou_prefilter_keeps_highest_score_and_changes_top2(root / "iou_case")
        test_audit_writes_evidence_but_keeps_baseline(root / "audit_case")
        test_off_is_identity(root / "off_case")
        test_discard_route_is_retained_but_not_a_formal_vlm_action(root / "discard_case")
        test_oracle_uses_processed_frame_index(root / "oracle_case")
    print("11 association-gate tests passed")
