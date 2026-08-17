import json
import uuid
from pathlib import Path

import numpy as np

from conceptgraph.utils.evidence import EvidenceRecorder


class FakePointCloud:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        self.colors = np.zeros_like(self.points)


class FakeBBox:
    def __init__(self, center=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0)):
        self._center = np.asarray(center, dtype=float)
        self._extent = np.asarray(extent, dtype=float)

    def get_center(self):
        return self._center

    def get_extent(self):
        return self._extent


class FakeEdges:
    def __init__(self):
        self.edges_by_index = {}


def test_evidence_recorder_writes_complete_minimal_chain(tmp_path):
    cfg = {
        "scene_id": "room0",
        "dataset_config": "replica",
        "detections_exp_suffix": "detections",
        "exp_suffix": "mapping",
        "mask_area_threshold": 10,
        "skip_bg": False,
        "max_bbox_area_ratio": 0.9,
        "mask_conf_threshold": 0.2,
        "sim_threshold": 0.5,
        "match_method": "sim_sum",
        "phys_bias": 0.0,
        "evidence_save_observation_pcd": True,
        "evidence_observation_pcd_max_points": 100,
        "evidence_top_k": 3,
    }
    recorder = EvidenceRecorder(tmp_path, cfg, cfg, enabled=True)

    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, :4, :4] = True
    masks[1, 0, 0] = True
    raw_gobs = {
        "xyxy": np.asarray([[0, 0, 4, 4], [0, 0, 1, 1]], dtype=float),
        "confidence": np.asarray([0.9, 0.9], dtype=float),
        "class_id": np.asarray([0, 0], dtype=int),
        "mask": masks,
        "classes": ["chair"],
        "image_crops": [None, None],
        "image_feats": np.zeros((2, 4), dtype=float),
        "text_feats": np.zeros((2, 4), dtype=float),
        "detection_class_labels": ["chair 0", "chair 1"],
        "labels": ["chair 0", "chair 1"],
        "edges": [],
        "captions": [
            {"id": "0", "name": "chair", "caption": "a chair"},
            {"id": "1", "name": "chair", "caption": "small artifact"},
        ],
    }
    snapshots = recorder.prepare_observations(
        raw_gobs, frame_idx=0, detection_path=tmp_path / "detections/000000.pkl.gz"
    )

    filtered_gobs = {
        **raw_gobs,
        "raw_det_idx": np.asarray([0], dtype=int),
        "obs_uid": np.asarray([raw_gobs["obs_uid"][0]], dtype=object),
    }
    point_cloud = FakePointCloud([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bbox = FakeBBox()
    kept_uids = recorder.record_observations(
        frame_idx=0,
        snapshots=snapshots,
        filtered_gobs=filtered_gobs,
        obj_pcds_and_bboxes=[{"pcd": point_cloud, "bbox": bbox}],
        image_shape=(8, 8, 3),
        bg_classes=[],
    )
    recorder.record_frame(
        frame_idx=0,
        source_frame_id="000000",
        rgb_path=tmp_path / "rgb/000000.jpg",
        depth_path=tmp_path / "depth/000000.png",
        pose=np.eye(4),
        intrinsics=np.eye(4),
        processed=True,
        skip_reason=None,
        num_raw_detections=2,
        num_kept_observations=1,
    )

    detection = {
        "id": uuid.uuid4(),
        "obs_uids": [kept_uids[0]],
        "class_id": [0],
        "class_name": "chair",
        "num_detections": 1,
        "curr_obj_num": 0,
        "pcd": point_cloud,
        "bbox": bbox,
        "consolidated_caption": "a chair",
    }
    empty = np.empty((1, 0), dtype=np.float32)
    recorder.record_associations(
        0, [detection], [], empty, empty, empty, [None]
    )
    recorder.close("completed", objects=[detection], map_edges=FakeEdges())

    evidence_dir = tmp_path / "evidence"
    summary = json.loads((evidence_dir / "evidence_summary.json").read_text())
    membership = json.loads((evidence_dir / "final_membership.json").read_text())
    observations = [
        json.loads(line)
        for line in (evidence_dir / "observations.jsonl").read_text().splitlines()
    ]

    assert summary["num_frames"] == 1
    assert summary["num_raw_detections"] == 2
    assert summary["num_kept_observations"] == 1
    assert summary["num_rejected_observations"] == 1
    assert summary["num_create_decisions"] == 1
    assert summary["missing_reference_count"] == 0
    assert summary["logging_error_count"] == 0
    assert membership[0]["member_observation_uids"] == kept_uids
    assert {item["status"] for item in observations} == {"kept", "rejected"}
    assert (evidence_dir / "similarities/frame_000000.npz").exists()
    assert (evidence_dir / "observation_pcd" / f"{kept_uids[0]}.npz").exists()


def test_new_run_removes_binary_sidecars_from_previous_run(tmp_path):
    similarity_dir = tmp_path / "evidence" / "similarities"
    observation_pcd_dir = tmp_path / "evidence" / "observation_pcd"
    similarity_dir.mkdir(parents=True)
    observation_pcd_dir.mkdir(parents=True)
    stale_similarity = similarity_dir / "frame_999999.npz"
    stale_pcd = observation_pcd_dir / "old_run_observation.npz"
    np.savez_compressed(stale_similarity, value=np.asarray([1]))
    np.savez_compressed(stale_pcd, value=np.asarray([1]))

    recorder = EvidenceRecorder(tmp_path, {"scene_id": "room0"}, {}, enabled=True)

    assert not stale_similarity.exists()
    assert not stale_pcd.exists()
    recorder.close("completed", objects=[], map_edges=FakeEdges())


def test_initialization_failure_is_bypassed(tmp_path):
    blocked_output = tmp_path / "not_a_directory"
    blocked_output.write_text("occupied by a file")
    client = object()

    recorder = EvidenceRecorder(blocked_output, {}, {}, enabled=True)

    assert recorder.enabled is False
    assert recorder.wrap_openai_client(client) is client
    assert recorder.prepare_observations({}, 0) == []
    assert recorder._errors


def test_disabled_recorder_is_transparent(tmp_path):
    client = object()
    recorder = EvidenceRecorder(tmp_path, {}, {}, enabled=False)
    assert recorder.wrap_openai_client(client) is client
    assert recorder.prepare_observations({}, 0) == []
