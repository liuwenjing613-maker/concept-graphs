#!/usr/bin/env python3
"""Build a unique-frame evidence-loss funnel for Replica online maps.

The audit follows each observable GT instance through cached raw masks, mapping
pre-processing, final-map provenance, final 3D structure, and semantics.  It is
an evaluation-only tool: GT is never used to alter the baseline map.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import pickle
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.slam.utils import (
    detections_to_obj_pcd_and_bbox,
    filter_gobs,
    init_process_pcd,
    resize_gobs,
)
from conceptgraph.utils.general_utils import ObjectClasses, load_saved_detections
from conceptgraph.utils.ious import mask_subtract_contained


SCENE_NAMES = {
    "room0": "room_0",
    "room1": "room_1",
    "room2": "room_2",
    "office0": "office_0",
    "office1": "office_1",
    "office2": "office_2",
    "office3": "office_3",
    "office4": "office_4",
}

ALIASES = {
    "arm chair": "chair",
    "armchair": "chair",
    "blinds": "curtain",
    "closet door": "door",
    "couch": "chair",
    "end table": "table",
    "coffee table": "table",
    "dining table": "table",
    "paper bag": "bag",
    "potted plant": "plant",
    "sofa": "chair",
    "sofa chair": "chair",
    "stool": "chair",
    "television": "screen",
    "tv": "screen",
}

STAGES = ("R0", "R1", "R2", "R3", "R4", "OK")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def normalize_label(label: object) -> str:
    return " ".join(
        str(label).strip().lower().replace("_", " ").replace("-", " ").split()
    )


class Canonicalizer:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.replica = {
            normalize_label(source): normalize_label(target)
            for source, target in payload["Replica2VisualGenome"].items()
        }
        self.visual_genome = {
            normalize_label(item) for item in payload["VisualGenome_list"]
        }

    def __call__(self, label: object) -> str:
        normalized = normalize_label(label)
        if normalized in ALIASES:
            return ALIASES[normalized]
        if normalized in self.replica:
            return self.replica[normalized]
        if normalized in self.visual_genome:
            return normalized
        return "unknown"


def load_labels(objects_path: Path, source_scene: str) -> dict[int, str]:
    payload = json.loads(objects_path.read_text(encoding="utf-8"))
    scans = [entry for entry in payload["scans"] if entry["scan"] == source_scene]
    if len(scans) != 1:
        raise ValueError(f"expected one objects entry for {source_scene}")
    return {int(item["id"]): str(item["label"]) for item in scans[0]["objects"]}


def observation_key(color_path: object, mask_idx: object) -> tuple[str, int]:
    return Path(str(color_path)).stem, int(mask_idx)


def baseline_provenance(payload: dict) -> tuple[dict[tuple[str, int], list[int]], Counter]:
    owners: dict[tuple[str, int], list[int]] = defaultdict(list)
    multiplicity: Counter = Counter()
    for object_index, obj in enumerate(payload["objects"]):
        lengths = {
            len(obj["color_path"]),
            len(obj["mask_idx"]),
            int(obj["num_detections"]),
        }
        if len(lengths) != 1:
            raise ValueError(f"B0 object {object_index} has inconsistent provenance")
        for color_path, mask_idx in zip(obj["color_path"], obj["mask_idx"]):
            key = observation_key(color_path, mask_idx)
            if object_index not in owners[key]:
                owners[key].append(object_index)
            multiplicity[key] += 1
    return dict(owners), multiplicity


def overlap_rows(
    masks: np.ndarray,
    semantic: np.ndarray,
    labels: dict[int, str],
    source_indices: np.ndarray | None = None,
    valid_mask_indices: set[int] | None = None,
) -> dict[int, list[dict]]:
    """Return all non-zero mask/GT overlaps without proposal-count inflation."""
    gt_areas = Counter(
        {
            int(instance_id): int(count)
            for instance_id, count in zip(*np.unique(semantic, return_counts=True))
            if int(instance_id) in labels
        }
    )
    result: dict[int, list[dict]] = defaultdict(list)
    for mask_index, mask in enumerate(np.asarray(masks, dtype=bool)):
        area = int(mask.sum())
        if area == 0:
            continue
        values, counts = np.unique(semantic[mask], return_counts=True)
        for instance_id, intersection in zip(values.tolist(), counts.tolist()):
            instance_id = int(instance_id)
            if instance_id not in labels:
                continue
            recall = int(intersection) / max(gt_areas[instance_id], 1)
            purity = int(intersection) / area
            result[instance_id].append(
                {
                    "mask_index": mask_index,
                    "source_mask_index": (
                        int(source_indices[mask_index])
                        if source_indices is not None
                        else mask_index
                    ),
                    "intersection_pixels": int(intersection),
                    "gt_visible_pixels": int(gt_areas[instance_id]),
                    "mask_pixels": area,
                    "recall": float(recall),
                    "purity": float(purity),
                    "quality": float(min(recall, purity)),
                    "geometry_valid": (
                        mask_index in valid_mask_indices
                        if valid_mask_indices is not None
                        else None
                    ),
                }
            )
    return dict(result)


def best_overlap(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["quality"],
            row["intersection_pixels"],
            row["purity"],
            -row["mask_index"],
        ),
    )


def usable(rows: list[dict], threshold: float, require_geometry: bool) -> list[dict]:
    return [
        row
        for row in rows
        if row["recall"] >= threshold
        and row["purity"] >= threshold
        and (not require_geometry or row["geometry_valid"] is True)
    ]


def voxelize(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    quantized = np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(
        np.int32
    )
    if quantized.size == 0:
        return set()
    return set(map(tuple, np.unique(quantized, axis=0).tolist()))


def prepare_3d(objects: list[dict], voxel_size: float, is_gt: bool) -> list[dict]:
    output = []
    for index, obj in enumerate(objects):
        voxels = voxelize(np.asarray(obj["pcd_np"]), voxel_size)
        if not voxels:
            continue
        values = np.asarray(list(voxels), dtype=np.int32)
        output.append(
            {
                "index": index,
                "label": str(obj.get("class_name", "unknown")),
                "gt_id": int(obj["oracle_gt_id"]) if is_gt else None,
                "voxels": voxels,
                "lower": values.min(axis=0),
                "upper": values.max(axis=0),
            }
        )
    return output


def structure_audit(
    baseline: dict,
    gt_map: dict,
    voxel_size: float,
    thresholds: list[float],
    degree_threshold: float,
) -> dict[int, dict]:
    predicted = prepare_3d(baseline["objects"], voxel_size, False)
    ground_truth = prepare_3d(gt_map["objects"], voxel_size, True)
    intersections: dict[tuple[int, int], int] = {}
    for pred_position, pred in enumerate(predicted):
        for gt_position, gt in enumerate(ground_truth):
            if np.any(pred["upper"] < gt["lower"]) or np.any(
                gt["upper"] < pred["lower"]
            ):
                continue
            count = len(pred["voxels"] & gt["voxels"])
            if count:
                intersections[(pred_position, gt_position)] = count

    result: dict[int, dict] = {}
    for gt_position, gt in enumerate(ground_truth):
        candidates = []
        for pred_position, pred in enumerate(predicted):
            intersection = intersections.get((pred_position, gt_position), 0)
            if not intersection:
                continue
            coverage = intersection / len(gt["voxels"])
            purity = intersection / len(pred["voxels"])
            iou = intersection / (
                len(gt["voxels"]) + len(pred["voxels"]) - intersection
            )
            candidates.append(
                {
                    "predicted_index": pred["index"],
                    "predicted_position": pred_position,
                    "predicted_label": pred["label"],
                    "intersection_voxels": intersection,
                    "coverage": float(coverage),
                    "purity": float(purity),
                    "iou": float(iou),
                    "quality": float(min(coverage, purity)),
                }
            )
        best = (
            max(
                candidates,
                key=lambda row: (
                    row["quality"],
                    row["intersection_voxels"],
                    row["iou"],
                ),
            )
            if candidates
            else None
        )
        fragmentation = sum(row["coverage"] >= degree_threshold for row in candidates)
        identity_degree = 0
        if best is not None:
            pred_position = best["predicted_position"]
            pred = predicted[pred_position]
            for other_gt_position in range(len(ground_truth)):
                intersection = intersections.get((pred_position, other_gt_position), 0)
                identity_degree += int(
                    intersection / len(pred["voxels"]) >= degree_threshold
                )
        strict = {}
        for threshold in thresholds:
            strict[str(threshold)] = bool(
                best is not None
                and best["coverage"] >= threshold
                and best["purity"] >= threshold
                and fragmentation == 1
                and identity_degree == 1
            )
        result[gt["gt_id"]] = {
            "gt_id": gt["gt_id"],
            "gt_label": gt["label"],
            "gt_voxels": len(gt["voxels"]),
            "best": (
                {key: value for key, value in best.items() if key != "predicted_position"}
                if best is not None
                else None
            ),
            "fragmentation_degree": fragmentation,
            "best_prediction_identity_degree": identity_degree,
            "strict_one_to_one": strict,
        }
    return result


def classify_stage(
    *,
    raw_frames: int,
    processed_frames: int,
    survived_frames: int,
    structure_reliable: bool,
    semantic_wrong: bool,
) -> str:
    if processed_frames == 0:
        return "R1" if raw_frames else "R0"
    if survived_frames == 0:
        return "R2"
    if not structure_reliable:
        return "R3"
    if semantic_wrong:
        return "R4"
    return "OK"


def run_self_test() -> dict:
    semantic = np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint16)
    masks = np.asarray(
        [
            [[1, 1, 0, 0], [1, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 1, 1, 0]],
        ],
        dtype=bool,
    )
    rows = overlap_rows(masks, semantic, {1: "a", 2: "b"})
    assert math.isclose(rows[1][0]["recall"], 0.75)
    assert math.isclose(rows[1][0]["purity"], 1.0)
    assert len(usable(rows[1], 0.5, False)) == 2
    assert classify_stage(
        raw_frames=0,
        processed_frames=0,
        survived_frames=0,
        structure_reliable=False,
        semantic_wrong=False,
    ) == "R0"
    assert classify_stage(
        raw_frames=2,
        processed_frames=0,
        survived_frames=0,
        structure_reliable=False,
        semantic_wrong=False,
    ) == "R1"
    assert classify_stage(
        raw_frames=2,
        processed_frames=1,
        survived_frames=0,
        structure_reliable=False,
        semantic_wrong=False,
    ) == "R2"
    assert classify_stage(
        raw_frames=2,
        processed_frames=1,
        survived_frames=1,
        structure_reliable=False,
        semantic_wrong=False,
    ) == "R3"
    assert classify_stage(
        raw_frames=2,
        processed_frames=1,
        survived_frames=1,
        structure_reliable=True,
        semantic_wrong=True,
    ) == "R4"
    assert classify_stage(
        raw_frames=2,
        processed_frames=1,
        survived_frames=1,
        structure_reliable=True,
        semantic_wrong=False,
    ) == "OK"
    points = np.asarray([[0.01, 0.01, 0.01], [0.06, 0.01, 0.01]])
    assert len(voxelize(points, 0.05)) == 2
    return {
        "status": "pass",
        "tests": [
            "mask_recall",
            "mask_purity",
            "proposal_deduplication_primitives",
            "R0_R4_stage_routing",
            "voxelization",
        ],
    }


def summarize_objects(rows: list[dict], threshold: float, scope: str) -> dict:
    selected = [row for row in rows if scope == "all" or not row["is_background"]]
    stage_counts = Counter(row["thresholds"][str(threshold)]["stage"] for row in selected)
    stage_views = Counter()
    stage_voxels = Counter()
    for row in selected:
        stage = row["thresholds"][str(threshold)]["stage"]
        stage_views[stage] += row["visible_frames"]
        stage_voxels[stage] += row["structure"]["gt_voxels"]
    visible_views = sum(row["visible_frames"] for row in selected)
    raw_views = sum(row["thresholds"][str(threshold)]["raw_usable_frames"] for row in selected)
    processed_views = sum(
        row["thresholds"][str(threshold)]["processed_usable_frames"] for row in selected
    )
    survived_views = sum(
        row["thresholds"][str(threshold)]["survived_usable_frames"] for row in selected
    )
    error_count = sum(stage_counts[stage] for stage in STAGES if stage != "OK")
    recoverable = stage_counts["R1"] + stage_counts["R2"] + stage_counts["R3"]
    return {
        "scope": scope,
        "threshold": threshold,
        "object_count": len(selected),
        "stage_object_counts": {stage: stage_counts[stage] for stage in STAGES},
        "stage_visible_view_counts": {stage: stage_views[stage] for stage in STAGES},
        "stage_gt_voxel_counts": {stage: stage_voxels[stage] for stage in STAGES},
        "unique_object_view_funnel": {
            "visible": visible_views,
            "raw_usable": raw_views,
            "processed_usable": processed_views,
            "survived_usable": survived_views,
            "raw_over_visible": raw_views / visible_views if visible_views else None,
            "processed_over_raw": processed_views / raw_views if raw_views else None,
            "survived_over_processed": (
                survived_views / processed_views if processed_views else None
            ),
        },
        "recoverable_error_object_share": (
            recoverable / error_count if error_count else None
        ),
        "reperception_required_object_share": (
            stage_counts["R0"] / error_count if error_count else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--scene", choices=sorted(SCENE_NAMES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--detections-root", type=Path)
    parser.add_argument("--baseline-map", type=Path)
    parser.add_argument("--gt-map", type=Path)
    parser.add_argument("--gt-sidecars", type=Path)
    parser.add_argument("--objects-json", type=Path)
    parser.add_argument("--label-mapping", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--degree-threshold", type=float, default=0.10)
    parser.add_argument("--visible-min-pixels", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    if args.self_test:
        report = run_self_test()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    required = (
        "scene",
        "config",
        "dataset_root",
        "detections_root",
        "baseline_map",
        "gt_map",
        "gt_sidecars",
        "objects_json",
        "label_mapping",
        "output_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    thresholds = sorted(set(args.threshold or [0.3, 0.5, 0.7]))
    if any(not 0.0 < item < 1.0 for item in thresholds):
        raise ValueError("thresholds must be inside (0,1)")

    started = time.perf_counter()
    output = args.output_root.resolve() / args.scene
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").write_text("running\n", encoding="utf-8")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config.resolve())
    OmegaConf.resolve(cfg)
    cfg.dataset_root = args.dataset_root.resolve()
    cfg.dataset_config = Path(str(cfg.dataset_config)).resolve()
    dataset_cfg = OmegaConf.load(cfg.dataset_config)
    if cfg.image_height is None:
        cfg.image_height = dataset_cfg.camera_params.image_height
    if cfg.image_width is None:
        cfg.image_width = dataset_cfg.camera_params.image_width
    if str(cfg.scene_id) != args.scene:
        raise ValueError(f"config scene {cfg.scene_id} != requested {args.scene}")
    labels = load_labels(args.objects_json.resolve(), SCENE_NAMES[args.scene])
    canonicalize = Canonicalizer(args.label_mapping.resolve())
    object_classes = ObjectClasses(
        classes_file_path=Path(str(cfg.classes_file)),
        bg_classes=list(cfg.bg_classes),
        skip_bg=bool(cfg.skip_bg),
    )
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=int(cfg.start),
        end=int(cfg.end),
        stride=int(cfg.stride),
        basedir=cfg.dataset_root,
        sequence=args.scene,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    baseline = load_pickle_gz(args.baseline_map.resolve())
    gt_map = load_pickle_gz(args.gt_map.resolve())
    final_owners, final_multiplicity = baseline_provenance(baseline)
    structure = structure_audit(
        baseline,
        gt_map,
        args.voxel_size,
        thresholds,
        args.degree_threshold,
    )
    gt_manifest_path = args.gt_sidecars.resolve() / "manifest.json"
    gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    expected_raw_frames = [
        int(Path(dataset.color_paths[index]).stem.replace("frame", ""))
        for index in range(len(dataset))
    ]
    if gt_manifest["frames"] != expected_raw_frames:
        raise ValueError("GT sidecar frame sequence differs from mapping dataset")
    alignment = gt_manifest["alignment_summary"]
    if alignment["max_median_abs_depth_m"] > 0.01 or alignment["min_within_5cm"] < 0.99:
        raise RuntimeError(f"GT alignment gate failed: {alignment}")

    frame_records: list[dict] = []
    per_object: dict[int, dict] = {
        gt_id: {
            "gt_id": gt_id,
            "gt_label": labels[gt_id],
            "is_background": labels[gt_id] in set(str(item) for item in cfg.bg_classes),
            "visible_frames": 0,
            "frames": [],
            "structure": structure.get(
                gt_id,
                {
                    "gt_id": gt_id,
                    "gt_label": labels[gt_id],
                    "gt_voxels": 0,
                    "best": None,
                    "fragmentation_degree": 0,
                    "best_prediction_identity_degree": 0,
                    "strict_one_to_one": {str(item): False for item in thresholds},
                },
            ),
        }
        for gt_id in labels
    }
    detection_digest = hashlib.sha256()
    reconstructed_valid_keys: set[tuple[str, int]] = set()
    frame_timings = []

    for frame_index in range(len(dataset)):
        frame_started = time.perf_counter()
        color_path = Path(dataset.color_paths[frame_index])
        raw_frame = int(color_path.stem.replace("frame", ""))
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_index]
        image_rgb = color_tensor.cpu().numpy().astype(np.uint8)
        depth_array = depth_tensor[..., 0].cpu().numpy()
        sidecar_path = args.gt_sidecars.resolve() / f"frame{raw_frame:06d}.npz"
        with np.load(sidecar_path) as sidecar:
            semantic = np.asarray(sidecar["semantic"], dtype=np.uint16)
        if semantic.shape != image_rgb.shape[:2]:
            raise ValueError(f"frame {raw_frame}: semantic/RGB shape mismatch")

        visible_ids, visible_counts = np.unique(semantic, return_counts=True)
        visible = {
            int(instance_id): int(count)
            for instance_id, count in zip(visible_ids.tolist(), visible_counts.tolist())
            if int(instance_id) in labels and int(count) >= args.visible_min_pixels
        }
        raw_gobs = load_saved_detections(args.detections_root.resolve() / color_path.stem)
        gobs = resize_gobs(copy.deepcopy(raw_gobs), image_rgb)
        raw_masks = np.asarray(gobs["mask"], dtype=bool)
        detection_digest.update(color_path.stem.encode("utf-8"))
        detection_digest.update(raw_masks.tobytes())
        for name in ("xyxy", "class_id", "confidence"):
            value = gobs.get(name)
            if value is not None:
                detection_digest.update(np.asarray(value).tobytes())
        raw_rows = overlap_rows(raw_masks, semantic, labels)

        gobs["source_mask_index"] = np.arange(len(raw_masks), dtype=np.int32)
        gobs = filter_gobs(
            gobs,
            image_rgb,
            skip_bg=bool(cfg.skip_bg),
            BG_CLASSES=object_classes.get_bg_classes_arr(),
            mask_area_threshold=float(cfg.mask_area_threshold),
            max_bbox_area_ratio=float(cfg.max_bbox_area_ratio),
            mask_conf_threshold=float(cfg.mask_conf_threshold),
        )
        if len(gobs["mask"]):
            gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])
            pcds = detections_to_obj_pcd_and_bbox(
                depth_array=depth_array,
                masks=gobs["mask"],
                cam_K=intrinsics.cpu().numpy()[:3, :3],
                image_rgb=image_rgb,
                trans_pose=dataset.poses[frame_index].cpu().numpy(),
                min_points_threshold=int(cfg.min_points_threshold),
                spatial_sim_type=str(cfg.spatial_sim_type),
                obj_pcd_max_points=int(cfg.obj_pcd_max_points),
                device=str(cfg.device),
            )
        else:
            pcds = []
        valid_indices: set[int] = set()
        for mask_index, item in enumerate(pcds):
            if item is None:
                continue
            processed_pcd = init_process_pcd(
                pcd=item["pcd"],
                downsample_voxel_size=float(cfg.downsample_voxel_size),
                dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
                dbscan_eps=float(cfg.dbscan_eps),
                dbscan_min_points=int(cfg.dbscan_min_points),
            )
            if len(processed_pcd.points) > 0:
                valid_indices.add(mask_index)
                reconstructed_valid_keys.add((color_path.stem, mask_index))
        processed_rows = overlap_rows(
            np.asarray(gobs["mask"], dtype=bool),
            semantic,
            labels,
            np.asarray(gobs.get("source_mask_index", []), dtype=np.int32),
            valid_indices,
        )

        for gt_id, visible_pixels in visible.items():
            object_row = per_object[gt_id]
            object_row["visible_frames"] += 1
            raw_for_gt = raw_rows.get(gt_id, [])
            processed_for_gt = processed_rows.get(gt_id, [])
            record = {
                "frame_index": frame_index,
                "raw_frame": raw_frame,
                "gt_id": gt_id,
                "gt_label": labels[gt_id],
                "visible_pixels": visible_pixels,
                "raw_proposals": len(raw_for_gt),
                "processed_proposals": len(processed_for_gt),
                "raw_best": best_overlap(raw_for_gt),
                "processed_best": best_overlap(processed_for_gt),
                "thresholds": {},
            }
            for threshold in thresholds:
                raw_good = usable(raw_for_gt, threshold, False)
                processed_good = usable(processed_for_gt, threshold, True)
                survived = [
                    row
                    for row in processed_good
                    if (color_path.stem, row["mask_index"]) in final_owners
                ]
                record["thresholds"][str(threshold)] = {
                    "raw_usable": bool(raw_good),
                    "processed_usable": bool(processed_good),
                    "survived_usable": bool(survived),
                    "raw_usable_proposals": len(raw_good),
                    "processed_usable_proposals": len(processed_good),
                    "survived_usable_proposals": len(survived),
                    "survived_final_object_indices": sorted(
                        {
                            owner
                            for row in survived
                            for owner in final_owners[(color_path.stem, row["mask_index"])]
                        }
                    ),
                }
            object_row["frames"].append(record)
            frame_records.append(record)

        frame_timings.append(
            {
                "frame_index": frame_index,
                "raw_frame": raw_frame,
                "raw_masks": len(raw_masks),
                "filtered_masks": len(gobs["mask"]),
                "geometry_valid_masks": len(valid_indices),
                "visible_gt_instances": len(visible),
                "elapsed_seconds": time.perf_counter() - frame_started,
            }
        )
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(dataset):
            print(
                f"{args.scene}: frames {frame_index + 1}/{len(dataset)}, "
                f"records {len(frame_records)}",
                flush=True,
            )

    final_keys = set(final_owners)
    missing_reconstructed = sorted(final_keys - reconstructed_valid_keys)
    if missing_reconstructed:
        raise RuntimeError(
            f"failed to reconstruct {len(missing_reconstructed)} B0 provenance keys; "
            f"first={missing_reconstructed[:10]}"
        )

    object_rows = []
    for gt_id, row in sorted(per_object.items()):
        if row["visible_frames"] == 0 or row["structure"]["gt_voxels"] == 0:
            continue
        row["thresholds"] = {}
        for threshold in thresholds:
            raw_frames = sum(
                frame["thresholds"][str(threshold)]["raw_usable"]
                for frame in row["frames"]
            )
            processed_frames = sum(
                frame["thresholds"][str(threshold)]["processed_usable"]
                for frame in row["frames"]
            )
            survived_frames = sum(
                frame["thresholds"][str(threshold)]["survived_usable"]
                for frame in row["frames"]
            )
            best = row["structure"]["best"]
            predicted_label = best["predicted_label"] if best else None
            gt_canonical = canonicalize(row["gt_label"])
            predicted_canonical = canonicalize(predicted_label) if predicted_label else None
            semantic_eligible = gt_canonical != "unknown"
            semantic_wrong = bool(
                semantic_eligible
                and predicted_canonical is not None
                and predicted_canonical != gt_canonical
            )
            structure_reliable = row["structure"]["strict_one_to_one"][str(threshold)]
            stage = classify_stage(
                raw_frames=raw_frames,
                processed_frames=processed_frames,
                survived_frames=survived_frames,
                structure_reliable=structure_reliable,
                semantic_wrong=semantic_wrong,
            )
            row["thresholds"][str(threshold)] = {
                "raw_usable_frames": raw_frames,
                "processed_usable_frames": processed_frames,
                "survived_usable_frames": survived_frames,
                "structure_reliable": structure_reliable,
                "predicted_label": predicted_label,
                "predicted_canonical_label": predicted_canonical,
                "gt_canonical_label": gt_canonical,
                "semantic_eligible": semantic_eligible,
                "semantic_wrong": semantic_wrong,
                "exact_label_wrong": (
                    normalize_label(predicted_label) != normalize_label(row["gt_label"])
                    if predicted_label is not None
                    else None
                ),
                "stage": stage,
            }
        row.pop("frames")
        object_rows.append(row)

    summaries = {
        str(threshold): {
            scope: summarize_objects(object_rows, threshold, scope)
            for scope in ("foreground", "all")
        }
        for threshold in thresholds
    }
    strict_semantic = [
        {
            "gt_id": row["gt_id"],
            "gt_label": row["gt_label"],
            "thresholds": {
                str(threshold): row["thresholds"][str(threshold)]
                for threshold in thresholds
                if row["thresholds"][str(threshold)]["stage"] == "R4"
            },
            "structure": row["structure"],
        }
        for row in object_rows
        if any(row["thresholds"][str(item)]["stage"] == "R4" for item in thresholds)
    ]
    recoverable_candidates = sorted(
        [
            row
            for row in object_rows
            if row["thresholds"]["0.5"]["stage"] in {"R1", "R2", "R3"}
            and not row["is_background"]
        ],
        key=lambda row: (
            row["structure"]["gt_voxels"],
            row["visible_frames"],
        ),
        reverse=True,
    )

    with (output / "frame_object_records.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in frame_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_json(output / "object_funnel.json", object_rows)
    atomic_json(output / "summary.json", summaries)
    atomic_json(output / "strict_semantic_candidates.json", strict_semantic)
    atomic_json(output / "recoverable_candidates.json", recoverable_candidates)

    csv_rows = []
    for row in object_rows:
        for threshold in thresholds:
            item = row["thresholds"][str(threshold)]
            csv_rows.append(
                {
                    "scene": args.scene,
                    "gt_id": row["gt_id"],
                    "gt_label": row["gt_label"],
                    "is_background": row["is_background"],
                    "threshold": threshold,
                    "stage": item["stage"],
                    "visible_frames": row["visible_frames"],
                    "raw_usable_frames": item["raw_usable_frames"],
                    "processed_usable_frames": item["processed_usable_frames"],
                    "survived_usable_frames": item["survived_usable_frames"],
                    "gt_voxels": row["structure"]["gt_voxels"],
                    "best_coverage": (
                        row["structure"]["best"]["coverage"]
                        if row["structure"]["best"]
                        else 0.0
                    ),
                    "best_purity": (
                        row["structure"]["best"]["purity"]
                        if row["structure"]["best"]
                        else 0.0
                    ),
                    "fragmentation_degree": row["structure"]["fragmentation_degree"],
                    "identity_degree": row["structure"][
                        "best_prediction_identity_degree"
                    ],
                    "predicted_label": item["predicted_label"],
                    "semantic_wrong": item["semantic_wrong"],
                }
            )
    with (output / "object_funnel.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": "1.0.0",
        "scene": args.scene,
        "source_scene": SCENE_NAMES[args.scene],
        "seed": args.seed,
        "frame_count": len(dataset),
        "thresholds": thresholds,
        "visible_min_pixels": args.visible_min_pixels,
        "voxel_size_m": args.voxel_size,
        "degree_threshold": args.degree_threshold,
        "classification": {
            "R0": "no usable raw mask in any unique visible frame",
            "R1": "usable raw mask exists but no usable postprocessed 3D observation",
            "R2": "usable processed observation exists but none survives in final B0",
            "R3": "usable evidence survives but final 3D node is not strict one-to-one",
            "R4": "strict one-to-one structure with wrong canonical semantic label",
            "OK": "strict one-to-one structure with correct/unevaluable canonical label",
        },
        "input_paths": {
            "config": str(args.config.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "detections_root": str(args.detections_root.resolve()),
            "baseline_map": str(args.baseline_map.resolve()),
            "gt_map": str(args.gt_map.resolve()),
            "gt_sidecars": str(args.gt_sidecars.resolve()),
            "objects_json": str(args.objects_json.resolve()),
            "label_mapping": str(args.label_mapping.resolve()),
        },
        "input_sha256": {
            "config": sha256_file(args.config.resolve()),
            "baseline_map": sha256_file(args.baseline_map.resolve()),
            "gt_map": sha256_file(args.gt_map.resolve()),
            "gt_manifest": sha256_file(gt_manifest_path),
            "objects_json": sha256_file(args.objects_json.resolve()),
            "label_mapping": sha256_file(args.label_mapping.resolve()),
            "relevant_detection_arrays_stream": detection_digest.hexdigest(),
        },
        "gt_alignment_summary": alignment,
        "provenance_conservation": {
            "final_unique_keys": len(final_keys),
            "final_entries": int(sum(final_multiplicity.values())),
            "reconstructed_valid_keys": len(reconstructed_valid_keys),
            "missing_final_keys": 0,
        },
        "object_rows": len(object_rows),
        "frame_object_rows": len(frame_records),
        "strict_semantic_candidate_count": len(strict_semantic),
        "elapsed_seconds": elapsed,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "frame_timing": frame_timings,
    }
    atomic_json(output / "manifest.json", manifest)
    (output / "INCOMPLETE").unlink(missing_ok=True)
    (output / "READY").write_text("ready\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scene": args.scene,
                "elapsed_seconds": elapsed,
                "peak_rss_mb": manifest["peak_rss_mb"],
                "summary_0p5_foreground": summaries["0.5"]["foreground"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
