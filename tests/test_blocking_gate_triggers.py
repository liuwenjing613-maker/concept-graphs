"""CPU-only routing smoke: no API, no full-scene run, no future map input."""
import ast
import argparse
from contextlib import nullcontext
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
import runpy
import sys

import cv2
import numpy as np

from conceptgraph.slam.association_gate import BlockingAssociationGate, compute_support_drop
from test_blocking_association_gate import _object


def fixture(root, **settings):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.jpg"
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    gate = BlockingAssociationGate(
        cfg={"sim_threshold": 1.2, "spatial_sim_type": "overlap", "association_gate": {"mode": "human", **settings}},
        output_dir=root / "gate",
    )
    gate._call_vlm = Mock(side_effect=AssertionError("CPU human smoke must never call an API"))
    objects = [_object(source, 0, 0, 5), _object(source, 0, 1, 55)]
    detection = _object(source, 9, 2, 8)
    return gate, image, objects, detection


def route(gate, image, objects, detection, frame, scores, supports, baseline):
    return gate.route_frame(
        frame_idx=frame, source_frame_id=str(frame * 5), image_rgb=image,
        detection_list=[detection], objects=objects,
        aggregate_sim=np.array([scores], dtype=np.float32),
        spatial_sim=np.array([supports], dtype=np.float32),
        baseline_match_indices=[baseline],
    )


def test_support_boundaries(root):
    assert not compute_support_drop(.1, [.95, .95])["triggered"]
    assert not compute_support_drop(.2, [.74] * 5)["triggered"]
    assert not compute_support_drop(.82, [.91] * 5)["triggered"]
    assert compute_support_drop(.55, [.75] * 3)["triggered"]
    assert not compute_support_drop(.5501, [.75] * 3)["triggered"]
    assert compute_support_drop(.64, [.93, .91, .95, .90, .92])["triggered"]
    assert compute_support_drop(.64, [.93, .91, .95, .90, .92])["reference"] == .92


def test_mask_drop_survives_iou_and_tracks_uid(root):
    gate, image, objects, det = fixture(root)
    objects[1]["mask"] = objects[0]["mask"]  # IoU=1, distinct persistent objects
    for frame, support in enumerate([.93, .91, .95, .90, .92, .94]):
        assert route(gate, image, objects, det, frame, [1.8, .3], [support, .1], 0) == [0]
    uid = str(objects[0]["id"])
    assert len(gate._support_history[uid]) == 5
    assert [f for f, _ in gate._support_history[uid]] == [1, 2, 3, 4, 5]
    before = list(gate._support_history[uid])
    gate._human_input = Mock(return_value="DISCARD")
    assert route(gate, image, objects, det, 6, [1.8, 1.7], [.6, .1], 0) == [-1]
    event = gate.events[-1]
    assert event["trigger"]["reasons"] == ["mask_change"]
    assert event["candidate_alias_to_object_index"] == {"A": 0}
    assert event["spatial_support_hidden_from_reviewer"]["history_frames"] == [1, 2, 3, 4, 5]
    assert list(gate._support_history[uid]) == before  # rejected support must not lower reference
    # Reindex the persistent objects: history follows UID, never list index.
    gate._human_input = Mock(return_value="A")
    assert route(gate, image, objects[::-1], det, 7, [.3, 1.8], [.1, .6], 1) == [1]
    assert gate.events[-1]["spatial_support_hidden_from_reviewer"]["object_uid"] == uid
    assert gate.events[-1]["candidate_alias_to_object_index"] == {"A": 1}
    assert gate.stats["processed"] == 2 and gate.stats["failures"] == 0
    assert gate._call_vlm.call_count == 0
    gate.close()


def test_combined_reasons_single_review_and_no_trigger_training(root):
    gate, image, objects, det = fixture(root)
    for frame in range(3):
        route(gate, image, objects, det, frame, [1.8, .3], [.95, .1], 0)
    gate._human_input = Mock(return_value="UNCERTAIN")
    assert route(gate, image, objects, det, 3, [1.4, 1.3], [.6, .1], 0) == [0]
    assert gate.events[-1]["trigger"]["reasons"] == ["association_margin", "mask_change"]
    assert gate._human_input.call_count == 1
    assert len(gate._support_history[str(objects[0]["id"])]) == 3
    gate.close()


def test_same_frame_is_not_history_and_removed_uid_is_not_reused(root):
    gate, image, objects, det = fixture(root)
    gate._human_input = Mock(side_effect=AssertionError("no event expected"))
    for frame in range(2):
        route(gate, image, objects, det, frame, [1.8, .3], [.95, .1], 0)
    gate.route_frame(
        frame_idx=2, source_frame_id="10", image_rgb=image,
        detection_list=[det, det], objects=objects,
        aggregate_sim=np.array([[1.8, .3], [1.8, .3]]),
        spatial_sim=np.array([[.95, .1], [.6, .1]]), baseline_match_indices=[0, 0],
    )
    rows = [json.loads(line) for line in gate.support_path.read_text().splitlines()]
    assert [row["history_count"] for row in rows[-2:]] == [2, 2]
    assert gate.stats["processed"] == 0
    old_uid = str(objects[0]["id"])
    new_obj = _object(Path(det["color_path"][0]), 3, 3, 5)
    route(gate, image, [new_obj], det, 3, [1.8], [.1], 0)
    assert old_uid not in gate._support_history
    assert len(gate._support_history[str(new_obj["id"])]) == 1
    gate.close()


def test_every_new_is_reviewed_exactly_once(root):
    gate, image, objects, det = fixture(root)
    answers = iter(["A", "NEW", "DISCARD"])
    gate._human_input = Mock(side_effect=lambda _: next(answers))
    result = gate.route_frame(
        frame_idx=0, source_frame_id="0", image_rgb=image,
        detection_list=[det, det, det], objects=objects,
        aggregate_sim=np.array([[1.05, .7], [.4, .2], [-np.inf, -np.inf]]),
        spatial_sim=np.zeros((3, 2)), baseline_match_indices=[None] * 3,
    )
    assert result == [0, None, -1]
    assert gate._human_input.call_count == 3 and gate.stats["processed"] == 3
    assert [event["trigger"]["reasons"] for event in gate.events] == [
        ["score_threshold_distance"], ["all_new"], ["all_new"],
    ]
    assert gate.events[2]["candidate_alias_to_object_index"] == {}
    assert gate._support_history == {}  # reviewer ATTACH from baseline NEW is not a normal sample
    assert gate.stats["failures"] == 0
    assert gate._call_vlm.call_count == 0
    gate.close()


def test_real_mapper_bootstrap_branch_obeys_discard(root):
    gate, image, objects, det = fixture(root)
    source = Path(__file__).parents[1] / "conceptgraph/slam/rerun_realtime_mapping.py"
    tree = ast.parse(source.read_text())
    # Execute the production empty-map branch, with external logging replaced by mocks.
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == "len(objects) == 0")
    program = ast.Module(body=[ast.For(
        target=ast.Name(id="_once", ctx=ast.Store()), iter=ast.List(elts=[ast.Constant(0)], ctx=ast.Load()),
        body=[branch], orelse=[],
    )], type_ignores=[])
    answers = iter(["DISCARD", "NEW", "UNCERTAIN"])
    gate._human_input = Mock(side_effect=lambda _: next(answers))
    detection_list = [det, objects[0], objects[1]]
    state = {
        "np": np, "association_gate": gate, "frame_idx": 0,
        "color_path": Path("000000.jpg"), "image_rgb": image,
        "detection_list": detection_list, "objects": [],
        "evidence": Mock(), "tracker": Mock(), "owandb": Mock(), "DISCARD_MATCH_INDEX": -1,
    }
    exec(compile(ast.fix_missing_locations(program), str(source), "exec"), state)
    assert [obj["id"] for obj in state["objects"]] == [obj["id"] for obj in detection_list[1:]]
    assert gate._human_input.call_count == 3
    assert all(event["candidate_alias_to_object_index"] == {} for event in gate.events)
    state["tracker"].increment_total_objects.assert_called_once_with(2)
    assert state["evidence"].record_association_object_version.call_count == 2
    assert state["evidence"].record_associations.call_args.args[-1] == [-1, None, None]
    assert (gate.output_dir / "human_review.html").is_file()
    assert not list((gate.output_dir / "events").glob("*/human_review.html"))
    gate.close()


def test_feature_switches_restore_old_gate(root):
    gate, image, objects, det = fixture(root, mask_change_enabled=False, review_all_new=False)
    gate._human_input = Mock(side_effect=AssertionError("no event expected"))
    assert route(gate, image, objects, det, 0, [.4, .2], [0, 0], None) == [None]
    # Existing NMS suppression remains unchanged when mask_change is disabled.
    objects[1]["mask"] = objects[0]["mask"]
    assert route(gate, image, objects, det, 1, [1.4, 1.3], [.1, .1], 0) == [0]
    assert route(gate, image, [], det, 2, [], [], None) == [None]
    assert gate.stats["processed"] == 0 and gate.stats["suppressed_by_iou_prefilter"] == 1
    assert gate._support_history == {}
    gate.close()


def test_spatial_validation(root):
    gate, image, objects, det = fixture(root)
    for supports in (None, np.zeros((2, 1)), np.array([[np.nan, .1]]), np.array([[1.1, .1]])):
        try:
            gate.route_frame(
                frame_idx=0, source_frame_id="0", image_rgb=image,
                detection_list=[det], objects=objects, aggregate_sim=np.array([[1.8, .3]]),
                spatial_sim=supports, baseline_match_indices=[0],
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid/missing support silently accepted")
    try:
        BlockingAssociationGate(cfg={"sim_threshold": 1.2, "spatial_sim_type": "iou", "association_gate": {"mode": "human"}}, output_dir=root / "bad")
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("wrong spatial metric accepted")
    gate.close()


def test_launcher_wires_switches_without_running_mapper(root):
    source = Path(__file__).parents[1] / "scripts/run_blocking_association_gate.py"
    for suffix, extra, enabled, threshold in (
        ("default", [], "true", "0.2"),
        ("ablation", ["--no-mask-change", "--no-review-all-new", "--support-drop-threshold", "0.25"], "false", "0.25"),
    ):
        argv = [str(source), "--mode", "human", "--exp-suffix", suffix,
                "--project-root", str(root), "--dataset-root", str(root / "dataset"), *extra]
        with patch.object(sys, "argv", argv), patch.object(sys.stdin, "isatty", return_value=True), patch("subprocess.run", return_value=Mock(returncode=0)) as run:
            try:
                runpy.run_path(str(source), run_name="__main__")
            except SystemExit as exc:
                assert exc.code == 0
            cmd = run.call_args.args[0]
            assert f"association_gate.mask_change_enabled={enabled}" in cmd
            assert f"association_gate.review_all_new={enabled}" in cmd
            assert f"association_gate.support_drop_threshold={threshold}" in cmd


def test_event_limit_does_not_train_on_unreviewed_risk(root):
    gate, image, objects, det = fixture(root, max_events=1)
    for frame in range(3):
        route(gate, image, objects, det, frame, [1.8, .3], [.95, .1], 0)
    gate._human_input = Mock(return_value="NEW")
    assert route(gate, image, objects, det, 3, [1.8, .3], [.6, .1], 0) == [None]
    assert route(gate, image, objects, det, 4, [1.8, .3], [.6, .1], 0) == [0]
    assert gate.stats["suppressed_by_max_events"] == 1
    assert gate._human_input.call_count == 1
    assert len(gate._support_history[str(objects[0]["id"])]) == 3
    gate.close()


def test_zero_candidate_request_schema(root):
    gate, image, objects, det = fixture(root)
    payload = gate._request_payload("system", "user", [], [])
    properties = payload["response_format"]["json_schema"]["schema"]["properties"]
    assessments = properties["candidate_assessments"]
    assert assessments["minItems"] == assessments["maxItems"] == 0
    assert "enum" not in assessments["items"]["properties"]["code"]
    assert properties["choice"]["enum"] == ["NEW", "UNCERTAIN"]
    assert gate._call_vlm.call_count == 0
    gate.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Retain smoke evidence and timing in a NEW directory")
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    tests = [function for name, function in list(globals().items()) if name.startswith("test_") and callable(function)]
    context = nullcontext(str(args.output_dir)) if args.output_dir else tempfile.TemporaryDirectory(prefix="blocking_gate_triggers_")
    with context as directory:
        results = []
        for test in tests:
            started = time.perf_counter()
            test(Path(directory) / test.__name__)
            results.append({"test": test.__name__, "passed": True, "seconds": time.perf_counter() - started})
            print(f"PASS {test.__name__}", flush=True)
        (Path(directory) / "validation.json").write_text(json.dumps({
            "scope": "synthetic CPU smoke; launcher mocked; production gate/evidence and bootstrap branch executed",
            "api_calls": 0, "full_scene_runs": 0, "tests": results,
        }, indent=2) + "\n")
    print(f"{len(tests)} trigger smoke tests passed; no API/GPU/full-scene run")
