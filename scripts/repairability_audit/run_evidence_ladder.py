#!/usr/bin/env python3
"""Build a chronological evidence-recovery oracle ladder from an empty map.

The ladder separates final identity repair, pre-processing recovery, raw-mask
partition/cleanup, and the full GT-mask ceiling.  Every condition consumes
frames in timestamp order; no condition starts from a completed map.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import random
import resource
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import (
    denoise_objects,
    detections_to_obj_pcd_and_bbox,
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    merge_obj2_into_obj1,
    merge_objects,
    processing_needed,
    resize_gobs,
)
from conceptgraph.slam.mapping import (
    aggregate_similarities,
    compute_spatial_similarities,
    compute_visual_similarities,
    match_detections_to_objects,
    merge_obj_matches,
)
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    cfg_to_dict,
    load_saved_detections,
)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def atomic_pickle_gz(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with gzip.open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_pickle_gz(path: Path) -> object:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def load_labels(objects_path: Path, source_scene: str) -> dict[int, str]:
    payload = json.loads(objects_path.read_text(encoding="utf-8"))
    scans = [entry for entry in payload["scans"] if entry["scan"] == source_scene]
    if len(scans) != 1:
        raise ValueError(f"expected one objects entry for {source_scene}")
    return {int(item["id"]): str(item["label"]) for item in scans[0]["objects"]}


def observation_key(color_path: object, mask_idx: object) -> tuple[str, int]:
    return Path(str(color_path)).stem, int(mask_idx)


def baseline_provenance(
    path: Path,
) -> tuple[
    dict[tuple[str, int], int],
    Counter[tuple[str, int]],
    list[dict],
    dict,
]:
    payload = load_pickle_gz(path)
    provenance: dict[tuple[str, int], int] = {}
    multiplicity: Counter[tuple[str, int]] = Counter()
    duplicate_rows = []
    for object_index, obj in enumerate(payload["objects"]):
        lengths = {
            len(obj["color_path"]),
            len(obj["mask_idx"]),
            len(obj["image_idx"]),
            int(obj["num_detections"]),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"B0 object {object_index} has inconsistent observation lengths: {lengths}"
            )
        for color_path, mask_idx in zip(obj["color_path"], obj["mask_idx"]):
            key = observation_key(color_path, mask_idx)
            if key in provenance and provenance[key] != object_index:
                raise ValueError(
                    f"B0 observation {key} appears in distinct nodes "
                    f"{provenance[key]} and {object_index}"
                )
            if key in provenance:
                duplicate_rows.append(
                    {"color_stem": key[0], "mask_idx": key[1], "object_index": object_index}
                )
            else:
                provenance[key] = object_index
            multiplicity[key] += 1
    return provenance, multiplicity, duplicate_rows, payload


def mask_assignment(
    mask: np.ndarray,
    semantic: np.ndarray,
    labels: dict[int, str],
    purity_threshold: float,
    min_overlap_pixels: int,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    visible_values = semantic[mask]
    ids, counts = np.unique(visible_values, return_counts=True)
    candidates = sorted(
        (
            (int(count), int(instance_id))
            for instance_id, count in zip(ids.tolist(), counts.tolist())
            if int(instance_id) in labels
        ),
        reverse=True,
    )
    if not candidates:
        return {
            "eligible": False,
            "gt_id": None,
            "gt_label": None,
            "intersection_pixels": 0,
            "mask_area_pixels": area,
            "purity": 0.0,
            "second_purity": 0.0,
            "visible_iou": 0.0,
            "mixed_mask": False,
        }
    top_count, top_id = candidates[0]
    second_count = candidates[1][0] if len(candidates) > 1 else 0
    gt_area = int(np.count_nonzero(semantic == top_id))
    union = area + gt_area - top_count
    purity = top_count / max(area, 1)
    second_purity = second_count / max(area, 1)
    eligible = top_count >= min_overlap_pixels and purity >= purity_threshold
    return {
        "eligible": bool(eligible),
        "gt_id": top_id,
        "gt_label": labels[top_id],
        "intersection_pixels": top_count,
        "mask_area_pixels": area,
        "purity": float(purity),
        "second_purity": float(second_purity),
        "visible_iou": float(top_count / max(union, 1)),
        "mixed_mask": bool(purity < 0.8 or second_purity >= 0.1),
    }


def subset_gobs(gobs: dict, indices: list[int]) -> dict:
    """Select detections while preserving scene-level metadata arrays/lists."""
    count = len(gobs["mask"])
    output = {}
    for key, value in gobs.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == count:
            output[key] = value[np.asarray(indices, dtype=np.int64)].copy()
        elif isinstance(value, list) and len(value) == count:
            output[key] = [copy.deepcopy(value[index]) for index in indices]
        else:
            output[key] = copy.deepcopy(value)
    return output


def mask_xyxy(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        raise ValueError("cannot box an empty mask")
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def make_synthetic_gobs(raw_gobs: dict, specs: list[dict]) -> dict:
    """Clone representative detector metadata and replace only mask geometry."""
    selected = subset_gobs(raw_gobs, [int(item["representative_index"]) for item in specs])
    selected["mask"] = np.asarray([item["mask"] for item in specs], dtype=bool)
    selected["xyxy"] = np.asarray([mask_xyxy(item["mask"]) for item in specs])
    if len(selected.get("captions", [])) != len(specs):
        selected["captions"] = [None for _ in specs]
    return selected


def process_gobs(
    *,
    gobs: dict,
    depth_array: np.ndarray,
    intrinsics: np.ndarray,
    image_rgb: np.ndarray,
    pose: np.ndarray,
    color_path: Path,
    object_classes: ObjectClasses,
    frame_idx: int,
    cfg,
) -> list[dict]:
    if len(gobs["mask"]) == 0:
        return []
    pcds = detections_to_obj_pcd_and_bbox(
        depth_array=depth_array,
        masks=np.asarray(gobs["mask"], dtype=bool),
        cam_K=intrinsics[:3, :3],
        image_rgb=image_rgb,
        trans_pose=pose,
        min_points_threshold=int(cfg.min_points_threshold),
        spatial_sim_type=str(cfg.spatial_sim_type),
        obj_pcd_max_points=int(cfg.obj_pcd_max_points),
        device=str(cfg.device),
    )
    for item_index, item in enumerate(pcds):
        if item is None:
            continue
        item["pcd"] = init_process_pcd(
            pcd=item["pcd"],
            downsample_voxel_size=float(cfg.downsample_voxel_size),
            dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
            dbscan_eps=float(cfg.dbscan_eps),
            dbscan_min_points=int(cfg.dbscan_min_points),
        )
        if len(item["pcd"].points) == 0:
            pcds[item_index] = None
            continue
        item["bbox"] = get_bounding_box(
            spatial_sim_type=str(cfg.spatial_sim_type),
            pcd=item["pcd"],
        )
    return make_detection_list_from_pcd_and_gobs(
        pcds, gobs, color_path, object_classes, frame_idx
    )


def representative_index(
    raw_masks: np.ndarray, semantic: np.ndarray, gt_id: int, candidates: list[int]
) -> int:
    return max(
        candidates,
        key=lambda index: (
            int(np.count_nonzero(raw_masks[index] & (semantic == gt_id))),
            int(np.count_nonzero(raw_masks[index])),
            -index,
        ),
    )


def addition_specs(
    *,
    raw_masks: np.ndarray,
    semantic: np.ndarray,
    labels: dict[int, str],
    discarded_indices: list[int],
    purity_threshold: float,
    min_overlap_pixels: int,
) -> tuple[list[dict], list[dict]]:
    """Union GT-identifiable raw proposals rejected before a valid observation."""
    grouped: dict[int, list[int]] = defaultdict(list)
    diagnostics = []
    for index in discarded_indices:
        assignment = mask_assignment(
            raw_masks[index], semantic, labels, purity_threshold, min_overlap_pixels
        )
        diagnostics.append({"raw_mask_index": index, **assignment})
        if assignment["eligible"]:
            grouped[int(assignment["gt_id"])].append(index)
    specs = []
    for gt_id, indices in sorted(grouped.items()):
        union = np.any(raw_masks[np.asarray(indices, dtype=np.int64)], axis=0)
        if not np.any(union):
            continue
        specs.append(
            {
                "gt_id": gt_id,
                "gt_label": labels[gt_id],
                "mask": union,
                "representative_index": representative_index(
                    raw_masks, semantic, gt_id, indices
                ),
                "source_indices": indices,
                "support_pixels": int(union.sum()),
            }
        )
    return specs, diagnostics


def partition_specs(
    *,
    raw_masks: np.ndarray,
    semantic: np.ndarray,
    labels: dict[int, str],
    mode: str,
    purity_threshold: float,
    min_overlap_pixels: int,
    skipped_labels: set[str],
) -> list[dict]:
    """Create one GT-clipped union per object from pixels already covered by raw masks."""
    if mode not in {"om_pure", "om_all"}:
        raise ValueError(mode)
    grouped: dict[int, list[int]] = defaultdict(list)
    if mode == "om_pure":
        for index, mask in enumerate(raw_masks):
            assignment = mask_assignment(
                mask, semantic, labels, purity_threshold, min_overlap_pixels
            )
            if assignment["eligible"]:
                gt_id = int(assignment["gt_id"])
                if labels[gt_id] not in skipped_labels:
                    grouped[gt_id].append(index)
    else:
        for gt_id in sorted(int(item) for item in np.unique(semantic)):
            if gt_id not in labels or labels[gt_id] in skipped_labels:
                continue
            indices = [
                index
                for index, mask in enumerate(raw_masks)
                if int(np.count_nonzero(mask & (semantic == gt_id))) > 0
            ]
            if indices:
                grouped[gt_id] = indices

    specs = []
    for gt_id, indices in sorted(grouped.items()):
        raw_union = np.any(raw_masks[np.asarray(indices, dtype=np.int64)], axis=0)
        clean = raw_union & (semantic == gt_id)
        if int(clean.sum()) < min_overlap_pixels:
            continue
        specs.append(
            {
                "gt_id": gt_id,
                "gt_label": labels[gt_id],
                "mask": clean,
                "representative_index": representative_index(
                    raw_masks, semantic, gt_id, indices
                ),
                "source_indices": indices,
                "support_pixels": int(clean.sum()),
                "visible_pixels": int(np.count_nonzero(semantic == gt_id)),
            }
        )
    return specs


def geometry_hash(serialized_objects: list[dict]) -> str:
    digest = hashlib.sha256()
    for obj in serialized_objects:
        digest.update(np.asarray(obj["pcd_np"], dtype=np.float32).tobytes())
        digest.update(np.asarray(obj["bbox_np"], dtype=np.float32).tobytes())
        for color_path, mask_idx in zip(obj["color_path"], obj["mask_idx"]):
            digest.update(f"{Path(str(color_path)).stem}:{int(mask_idx)}\n".encode())
    return digest.hexdigest()


def gt_masks_for_frame(
    semantic: np.ndarray,
    labels: dict[int, str],
    cfg,
) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    output = []
    image_area = semantic.shape[0] * semantic.shape[1]
    bg_classes = set(str(item) for item in cfg.bg_classes)
    for instance_id in sorted(int(item) for item in np.unique(semantic)):
        if instance_id not in labels:
            continue
        label = labels[instance_id]
        if bool(cfg.skip_bg) and label in bg_classes:
            continue
        mask = semantic == instance_id
        area = int(mask.sum())
        if area < max(int(cfg.mask_area_threshold), 10):
            continue
        ys, xs = np.nonzero(mask)
        xyxy = np.asarray(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
        )
        bbox_area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
        if (
            label not in bg_classes
            and cfg.max_bbox_area_ratio is not None
            and bbox_area > float(cfg.max_bbox_area_ratio) * image_area
        ):
            continue
        output.append((instance_id, label, mask, xyxy))
    return output


def make_gt_detection(
    *,
    instance_id: int,
    label: str,
    mask: np.ndarray,
    xyxy: np.ndarray,
    pcd_bbox: dict,
    color_path: Path,
    frame_idx: int,
    ordinal: int,
) -> dict:
    stable_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"oracle-o3:{color_path.stem}:{instance_id}",
    )
    return {
        "id": stable_id,
        "image_idx": [frame_idx],
        "mask_idx": [ordinal],
        "color_path": [color_path],
        "class_name": label,
        "class_id": [instance_id],
        "captions": [{"id": str(ordinal), "name": label, "caption": None}],
        "num_detections": 1,
        "mask": [mask],
        "xyxy": [xyxy],
        "conf": [1.0],
        "n_points": len(pcd_bbox["pcd"].points),
        "contain_number": [None],
        "inst_color": np.asarray(
            [
                ((instance_id * 37) % 255) / 255.0,
                ((instance_id * 67) % 255) / 255.0,
                ((instance_id * 97) % 255) / 255.0,
            ]
        ),
        "is_background": label in {"wall", "floor", "ceiling"},
        "pcd": pcd_bbox["pcd"],
        "bbox": pcd_bbox["bbox"],
        "clip_ft": torch.ones(1, dtype=torch.float32),
        "num_obj_in_class": 0,
        "curr_obj_num": ordinal,
        "new_counter": ordinal,
    }


def merge_by_token(
    objects: MapObjectList,
    token_to_index: dict[str, int],
    object_tokens: list[str],
    token: str,
    detection: dict,
    cfg,
) -> None:
    if token not in token_to_index:
        token_to_index[token] = len(objects)
        object_tokens.append(token)
        objects.append(detection)
        return
    object_index = token_to_index[token]
    objects[object_index] = merge_obj2_into_obj1(
        obj1=objects[object_index],
        obj2=detection,
        downsample_voxel_size=float(cfg.downsample_voxel_size),
        dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
        dbscan_eps=float(cfg.dbscan_eps),
        dbscan_min_points=int(cfg.dbscan_min_points),
        spatial_sim_type=str(cfg.spatial_sim_type),
        device=str(cfg.device),
        run_dbscan=False,
    )


def postprocess_labels(
    objects: MapObjectList,
    object_tokens: list[str],
    object_classes: ObjectClasses | None,
    labels: dict[int, str],
) -> None:
    class_names = object_classes.get_classes_arr() if object_classes is not None else []
    for object_index, (obj, token) in enumerate(zip(objects, object_tokens)):
        if object_classes is not None:
            most_common = Counter(int(item) for item in obj["class_id"]).most_common(1)[0][0]
            predicted = class_names[most_common]
            obj["class_name"] = predicted
            obj["predicted_class_name"] = predicted
        else:
            obj["predicted_class_name"] = obj["class_name"]
        obj["association_token"] = token
        if token.startswith("gt:"):
            instance_id = int(token.split(":", 1)[1])
            obj["oracle_gt_id"] = instance_id
            obj["oracle_gt_label"] = labels[instance_id]
        else:
            obj["oracle_gt_id"] = None
            obj["oracle_gt_label"] = None
        obj["oracle_object_index"] = object_index


MODE_DEFINITIONS = {
    "oa": (
        "Normal filtered 3D observations; GT identity is forced only for masks "
        "whose dominant-instance purity passes the gate. Unassigned detections "
        "are associated online in an isolated native map."
    ),
    "op": (
        "OA plus a per-frame union of raw proposals that failed to become a valid "
        "filtered observation and whose dominant-instance purity passes the gate."
    ),
    "om_pure": (
        "One GT-clipped per-frame union from owner-identifiable raw proposals. "
        "Uses only pixels present in raw masks; removes contamination and residual FPs."
    ),
    "om_all": (
        "One GT-clipped per-frame union from any overlapping raw proposal. Uses "
        "only raw-mask support but is a maximal partition plus FP-suppression ceiling."
    ),
    "og": "Full per-frame GT masks, GT identity, and GT semantics.",
}


def native_ingest(objects: MapObjectList, detections: MapObjectList, cfg) -> MapObjectList:
    """Run the frozen ali-dev association path without consulting final B0 lineage."""
    if len(detections) == 0:
        return objects
    if len(objects) == 0:
        objects.extend(detections)
        return objects
    spatial = compute_spatial_similarities(
        spatial_sim_type=str(cfg.spatial_sim_type),
        detection_list=detections,
        objects=objects,
        downsample_voxel_size=float(cfg.downsample_voxel_size),
    )
    visual = compute_visual_similarities(detections, objects)
    aggregate = aggregate_similarities(
        match_method=str(cfg.match_method),
        phys_bias=float(cfg.phys_bias),
        spatial_sim=spatial,
        visual_sim=visual,
    )
    matches = match_detections_to_objects(
        agg_sim=aggregate, detection_threshold=float(cfg.sim_threshold)
    )
    return merge_obj_matches(
        detection_list=detections,
        objects=objects,
        match_indices=matches,
        downsample_voxel_size=float(cfg.downsample_voxel_size),
        dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
        dbscan_eps=float(cfg.dbscan_eps),
        dbscan_min_points=int(cfg.dbscan_min_points),
        spatial_sim_type=str(cfg.spatial_sim_type),
        device=str(cfg.device),
    )


def postprocess_native(objects: MapObjectList, cfg, frame_idx: int, is_final: bool) -> MapObjectList:
    if processing_needed(
        int(cfg.denoise_interval), bool(cfg.run_denoise_final_frame), frame_idx, is_final
    ):
        objects = denoise_objects(
            downsample_voxel_size=float(cfg.downsample_voxel_size),
            dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
            dbscan_eps=float(cfg.dbscan_eps),
            dbscan_min_points=int(cfg.dbscan_min_points),
            spatial_sim_type=str(cfg.spatial_sim_type),
            device=str(cfg.device),
            objects=objects,
        )
    if processing_needed(
        int(cfg.filter_interval), bool(cfg.run_filter_final_frame), frame_idx, is_final
    ):
        objects = filter_objects(
            obj_min_points=int(cfg.obj_min_points),
            obj_min_detections=int(cfg.obj_min_detections),
            objects=objects,
        )
    if processing_needed(
        int(cfg.merge_interval), bool(cfg.run_merge_final_frame), frame_idx, is_final
    ):
        objects = merge_objects(
            merge_overlap_thresh=float(cfg.merge_overlap_thresh),
            merge_visual_sim_thresh=float(cfg.merge_visual_sim_thresh),
            merge_text_sim_thresh=float(cfg.merge_text_sim_thresh),
            objects=objects,
            downsample_voxel_size=float(cfg.downsample_voxel_size),
            dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
            dbscan_eps=float(cfg.dbscan_eps),
            dbscan_min_points=int(cfg.dbscan_min_points),
            spatial_sim_type=str(cfg.spatial_sim_type),
            device=str(cfg.device),
        )
    return objects


def run_self_test() -> dict:
    semantic = np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint16)
    masks = np.asarray(
        [
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 0]],
        ],
        dtype=bool,
    )
    labels = {1: "chair", 2: "table"}
    assigned = mask_assignment(masks[0], semantic, labels, 0.5, 1)
    assert assigned["eligible"] and assigned["gt_id"] == 1
    additions, _ = addition_specs(
        raw_masks=masks,
        semantic=semantic,
        labels=labels,
        discarded_indices=[0, 1],
        purity_threshold=0.5,
        min_overlap_pixels=1,
    )
    assert {item["gt_id"] for item in additions} == {1}
    maximal = partition_specs(
        raw_masks=masks,
        semantic=semantic,
        labels=labels,
        mode="om_all",
        purity_threshold=0.5,
        min_overlap_pixels=1,
        skipped_labels=set(),
    )
    assert {item["gt_id"] for item in maximal} == {1, 2}
    assert sum(item["support_pixels"] for item in maximal) == 6
    toy_gobs = {
        "mask": masks,
        "xyxy": np.zeros((2, 4), dtype=np.float32),
        "class_id": np.asarray([3, 4]),
        "image_feats": np.ones((2, 3), dtype=np.float32),
        "classes": ["a", "b", "c", "d", "e"],
        "captions": [],
    }
    selected = subset_gobs(toy_gobs, [1])
    assert selected["class_id"].tolist() == [4] and len(selected["classes"]) == 5
    return {
        "mask_assignment": "pass",
        "discarded_raw_union": "pass",
        "maximal_partition": "pass",
        "metadata_subset": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_DEFINITIONS))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--detections-root", type=Path)
    parser.add_argument("--baseline-map", type=Path)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument(
        "--objects-json",
        type=Path,
        default=Path(
            "/home/chenkejun/beauty/conceptgraphs/code/third_party/"
            "ReplicaSSG/files/objects.json"
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--purity-threshold", type=float, default=0.5)
    parser.add_argument("--min-overlap-pixels", type=int, default=25)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0
    for name in ("mode", "config", "dataset_root", "gt_root", "output_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.mode != "og" and args.detections_root is None:
        parser.error("all raw-evidence modes require --detections-root")
    if args.mode in {"oa", "op"} and args.baseline_map is None:
        parser.error("OA/OP require --baseline-map for post-run provenance audit only")
    if not 0.0 <= args.purity_threshold <= 1.0:
        raise ValueError("purity threshold must be in [0, 1]")
    if args.min_overlap_pixels < 1:
        raise ValueError("min overlap must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = OmegaConf.load(args.config.resolve())
    OmegaConf.resolve(cfg)
    cfg.dataset_root = args.dataset_root.resolve()
    cfg.dataset_config = Path(str(cfg.dataset_config)).resolve()
    dataset_cfg = OmegaConf.load(cfg.dataset_config)
    if cfg.image_height is None:
        cfg.image_height = dataset_cfg.camera_params.image_height
    if cfg.image_width is None:
        cfg.image_width = dataset_cfg.camera_params.image_width
    sequence = str(cfg.scene_id)
    source_scene = SCENE_NAMES[sequence]
    labels = load_labels(args.objects_json.resolve(), source_scene)
    object_classes = (
        None
        if args.mode == "og"
        else ObjectClasses(
            classes_file_path=Path(str(cfg.classes_file)),
            bg_classes=list(cfg.bg_classes),
            skip_bg=bool(cfg.skip_bg),
        )
    )

    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=int(cfg.start),
        end=int(cfg.end),
        stride=int(cfg.stride),
        basedir=cfg.dataset_root,
        sequence=sequence,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    frame_count = len(dataset)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    if frame_count < 1:
        raise ValueError("no frames requested")

    gt_scene_root = args.gt_root.resolve() / sequence
    if not (gt_scene_root / "READY").is_file():
        raise FileNotFoundError(f"GT sidecar is not ready: {gt_scene_root}")
    gt_request = json.loads((gt_scene_root / "request.json").read_text(encoding="utf-8"))
    gt_manifest_path = gt_scene_root / "manifest.json"
    gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    alignment = gt_manifest["alignment_summary"]
    if float(alignment["max_median_abs_depth_m"]) > 0.01:
        raise RuntimeError("GT alignment stop gate failed")
    if float(alignment["min_within_5cm"]) < 0.99:
        raise RuntimeError("GT 5cm alignment stop gate failed")
    expected_raw_frames = [
        int(Path(dataset.color_paths[index]).stem.replace("frame", ""))
        for index in range(frame_count)
    ]
    missing_gt = [
        frame
        for frame in expected_raw_frames
        if not (gt_scene_root / f"frame{frame:06d}.npz").is_file()
    ]
    if missing_gt:
        raise FileNotFoundError(f"missing GT sidecars for frames {missing_gt[:10]}")

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix outputs in non-empty directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").write_text(
        f"started_at_unix={time.time()}\n", encoding="utf-8"
    )

    oracle_objects = MapObjectList(device=str(cfg.device))
    native_objects = MapObjectList(device=str(cfg.device))
    token_to_index: dict[str, int] = {}
    oracle_tokens: list[str] = []
    processed_assignments = []
    processed_keys: set[tuple[str, int]] = set()
    frame_rows = []
    counters: Counter[str] = Counter()
    support_recall_values = []
    detection_digest = hashlib.sha256()
    skipped_labels = set(str(item) for item in cfg.bg_classes) if bool(cfg.skip_bg) else set()
    started = time.perf_counter()

    for frame_idx in range(frame_count):
        frame_started = time.perf_counter()
        color_path = Path(dataset.color_paths[frame_idx])
        raw_frame = int(color_path.stem.replace("frame", ""))
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        depth_array = depth_tensor[..., 0].cpu().numpy()
        image_rgb = color_tensor.cpu().numpy().astype(np.uint8)
        pose = dataset.poses[frame_idx].cpu().numpy()
        intrinsics_np = intrinsics.cpu().numpy()
        with np.load(gt_scene_root / f"frame{raw_frame:06d}.npz") as sidecar:
            semantic = np.asarray(sidecar["semantic"], dtype=np.uint16)
        if semantic.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"frame {raw_frame}: GT {semantic.shape} != RGB {image_rgb.shape[:2]}"
            )

        frame_counts = Counter()
        if args.mode == "og":
            gt_instances = gt_masks_for_frame(semantic, labels, cfg)
            masks = np.asarray([item[2] for item in gt_instances], dtype=bool)
            pcds = (
                detections_to_obj_pcd_and_bbox(
                    depth_array=depth_array,
                    masks=masks,
                    cam_K=intrinsics_np[:3, :3],
                    image_rgb=image_rgb,
                    trans_pose=pose,
                    min_points_threshold=int(cfg.min_points_threshold),
                    spatial_sim_type=str(cfg.spatial_sim_type),
                    obj_pcd_max_points=int(cfg.obj_pcd_max_points),
                    device=str(cfg.device),
                )
                if len(masks)
                else []
            )
            for ordinal, ((gt_id, label, mask, xyxy), item) in enumerate(
                zip(gt_instances, pcds)
            ):
                if item is None:
                    continue
                item["pcd"] = init_process_pcd(
                    pcd=item["pcd"],
                    downsample_voxel_size=float(cfg.downsample_voxel_size),
                    dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
                    dbscan_eps=float(cfg.dbscan_eps),
                    dbscan_min_points=int(cfg.dbscan_min_points),
                )
                if len(item["pcd"].points) == 0:
                    continue
                item["bbox"] = get_bounding_box(
                    spatial_sim_type=str(cfg.spatial_sim_type), pcd=item["pcd"]
                )
                detection = make_gt_detection(
                    instance_id=gt_id,
                    label=label,
                    mask=mask,
                    xyxy=xyxy,
                    pcd_bbox=item,
                    color_path=color_path,
                    frame_idx=frame_idx,
                    ordinal=ordinal,
                )
                merge_by_token(
                    oracle_objects,
                    token_to_index,
                    oracle_tokens,
                    f"gt:{gt_id}",
                    detection,
                    cfg,
                )
                frame_counts["gt_detections"] += 1
        else:
            loaded = load_saved_detections(args.detections_root.resolve() / color_path.stem)
            raw_gobs = resize_gobs(copy.deepcopy(loaded), image_rgb)
            raw_masks = np.asarray(raw_gobs["mask"], dtype=bool)
            detection_digest.update(color_path.stem.encode("utf-8"))
            detection_digest.update(raw_masks.tobytes())
            for field in ("xyxy", "class_id", "confidence", "image_feats"):
                if field in raw_gobs:
                    detection_digest.update(np.asarray(raw_gobs[field]).tobytes())
            frame_counts["raw_masks"] = len(raw_masks)

            if args.mode in {"oa", "op"}:
                filtered = copy.deepcopy(raw_gobs)
                filtered["source_mask_index"] = np.arange(len(raw_masks), dtype=np.int32)
                filtered = filter_gobs(
                    filtered,
                    image_rgb,
                    skip_bg=bool(cfg.skip_bg),
                    BG_CLASSES=object_classes.get_bg_classes_arr(),
                    mask_area_threshold=float(cfg.mask_area_threshold),
                    max_bbox_area_ratio=float(cfg.max_bbox_area_ratio),
                    mask_conf_threshold=float(cfg.mask_conf_threshold),
                )
                if len(filtered["mask"]):
                    filtered["mask"] = mask_subtract_contained(
                        filtered["xyxy"], filtered["mask"]
                    )
                detections = process_gobs(
                    gobs=filtered,
                    depth_array=depth_array,
                    intrinsics=intrinsics_np,
                    image_rgb=image_rgb,
                    pose=pose,
                    color_path=color_path,
                    object_classes=object_classes,
                    frame_idx=frame_idx,
                    cfg=cfg,
                )
                native_detections = MapObjectList(device=str(cfg.device))
                surviving_sources: set[int] = set()
                for detection in detections:
                    mask_idx = int(detection["mask_idx"][0])
                    source_idx = int(filtered["source_mask_index"][mask_idx])
                    surviving_sources.add(source_idx)
                    key = observation_key(color_path, mask_idx)
                    processed_keys.add(key)
                    assignment = mask_assignment(
                        np.asarray(detection["mask"][0]),
                        semantic,
                        labels,
                        args.purity_threshold,
                        args.min_overlap_pixels,
                    )
                    row = {
                        **assignment,
                        "frame_idx": frame_idx,
                        "raw_frame": raw_frame,
                        "mask_idx": mask_idx,
                        "source_mask_index": source_idx,
                    }
                    if assignment["eligible"]:
                        token = f"gt:{assignment['gt_id']}"
                        detection["id"] = uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"ladder-{args.mode}:{sequence}:{color_path.stem}:{mask_idx}",
                        )
                        merge_by_token(
                            oracle_objects,
                            token_to_index,
                            oracle_tokens,
                            token,
                            detection,
                            cfg,
                        )
                        row["route"] = "oracle_gt_identity"
                        frame_counts["processed_oracle"] += 1
                    else:
                        native_detections.append(detection)
                        row["route"] = "isolated_native_online"
                        frame_counts["processed_native"] += 1
                    processed_assignments.append(row)
                native_objects = native_ingest(native_objects, native_detections, cfg)

                if args.mode == "op":
                    discarded = sorted(set(range(len(raw_masks))) - surviving_sources)
                    specs, diagnostics = addition_specs(
                        raw_masks=raw_masks,
                        semantic=semantic,
                        labels=labels,
                        discarded_indices=discarded,
                        purity_threshold=args.purity_threshold,
                        min_overlap_pixels=args.min_overlap_pixels,
                    )
                    counters["discarded_raw_masks"] += len(discarded)
                    counters["discarded_owner_eligible_masks"] += sum(
                        item["eligible"] for item in diagnostics
                    )
                    if specs:
                        synthetic = make_synthetic_gobs(raw_gobs, specs)
                        additions = process_gobs(
                            gobs=synthetic,
                            depth_array=depth_array,
                            intrinsics=intrinsics_np,
                            image_rgb=image_rgb,
                            pose=pose,
                            color_path=color_path,
                            object_classes=object_classes,
                            frame_idx=frame_idx,
                            cfg=cfg,
                        )
                        for detection in additions:
                            spec = specs[int(detection["mask_idx"][0])]
                            detection["id"] = uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"ladder-op-add:{sequence}:{color_path.stem}:{spec['gt_id']}",
                            )
                            merge_by_token(
                                oracle_objects,
                                token_to_index,
                                oracle_tokens,
                                f"gt:{spec['gt_id']}",
                                detection,
                                cfg,
                            )
                            frame_counts["raw_additions"] += 1
            else:
                specs = partition_specs(
                    raw_masks=raw_masks,
                    semantic=semantic,
                    labels=labels,
                    mode=args.mode,
                    purity_threshold=args.purity_threshold,
                    min_overlap_pixels=args.min_overlap_pixels,
                    skipped_labels=skipped_labels,
                )
                if specs:
                    synthetic = make_synthetic_gobs(raw_gobs, specs)
                    partitioned = process_gobs(
                        gobs=synthetic,
                        depth_array=depth_array,
                        intrinsics=intrinsics_np,
                        image_rgb=image_rgb,
                        pose=pose,
                        color_path=color_path,
                        object_classes=object_classes,
                        frame_idx=frame_idx,
                        cfg=cfg,
                    )
                    for detection in partitioned:
                        spec = specs[int(detection["mask_idx"][0])]
                        detection["id"] = uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"ladder-{args.mode}:{sequence}:{color_path.stem}:{spec['gt_id']}",
                        )
                        merge_by_token(
                            oracle_objects,
                            token_to_index,
                            oracle_tokens,
                            f"gt:{spec['gt_id']}",
                            detection,
                            cfg,
                        )
                        frame_counts["partitioned_observations"] += 1
                        support_recall_values.append(
                            spec["support_pixels"] / max(spec["visible_pixels"], 1)
                        )

        is_final = frame_idx == frame_count - 1
        if processing_needed(
            int(cfg.denoise_interval),
            bool(cfg.run_denoise_final_frame),
            frame_idx,
            is_final,
        ):
            oracle_objects = denoise_objects(
                downsample_voxel_size=float(cfg.downsample_voxel_size),
                dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
                dbscan_eps=float(cfg.dbscan_eps),
                dbscan_min_points=int(cfg.dbscan_min_points),
                spatial_sim_type=str(cfg.spatial_sim_type),
                device=str(cfg.device),
                objects=oracle_objects,
            )
        native_objects = postprocess_native(native_objects, cfg, frame_idx, is_final)
        counters.update(frame_counts)
        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "raw_frame": raw_frame,
                **dict(frame_counts),
                "oracle_objects_after_frame": len(oracle_objects),
                "native_objects_after_frame": len(native_objects),
                "elapsed_seconds": time.perf_counter() - frame_started,
            }
        )
        if (frame_idx + 1) % 25 == 0 or is_final:
            print(
                f"{args.mode}: frames {frame_idx + 1}/{frame_count}, "
                f"oracle={len(oracle_objects)}, native={len(native_objects)}",
                flush=True,
            )

    postprocess_labels(oracle_objects, oracle_tokens, object_classes, labels)
    native_tokens = [f"native:{index}" for index in range(len(native_objects))]
    postprocess_labels(native_objects, native_tokens, object_classes, labels)
    serialized = oracle_objects.to_serializable() + native_objects.to_serializable()
    geometry_sha = geometry_hash(serialized)
    payload = {
        "objects": serialized,
        "cfg": cfg_to_dict(cfg),
        "class_names": (
            object_classes.get_classes_arr()
            if object_classes is not None
            else sorted(set(labels.values()))
        ),
        "class_colors": {},
        "edges": None,
        "oracle_condition": args.mode,
    }
    atomic_pickle_gz(output / f"pcd_{args.mode}.pkl.gz", payload)

    provenance_audit = None
    if args.mode in {"oa", "op"}:
        baseline_owners, baseline_multiplicity, duplicate_rows, _ = baseline_provenance(
            args.baseline_map.resolve()
        )
        expected = {
            key
            for key in baseline_owners
            if key[0] in {Path(dataset.color_paths[i]).stem for i in range(frame_count)}
        }
        missing = sorted(expected - processed_keys)
        unexpected = sorted(processed_keys - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"post-run provenance audit failed: missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )
        provenance_audit = {
            "baseline_final_lineage_used_for_mapping": False,
            "audit_performed_after_mapping": True,
            "expected_unique_processed_keys": len(expected),
            "replayed_unique_processed_keys": len(processed_keys),
            "missing": 0,
            "unexpected": 0,
            "baseline_duplicate_entries": len(duplicate_rows),
            "baseline_stored_entries": int(sum(baseline_multiplicity.values())),
        }

    elapsed = time.perf_counter() - started
    assignment_summary = None
    if processed_assignments:
        assignment_summary = {
            "physical_processed_masks": len(processed_assignments),
            "oracle_identity_masks": int(sum(row["eligible"] for row in processed_assignments)),
            "native_online_masks": int(sum(not row["eligible"] for row in processed_assignments)),
            "mixed_masks": int(sum(row["mixed_mask"] for row in processed_assignments)),
            "mean_purity": float(np.mean([row["purity"] for row in processed_assignments])),
            "mean_visible_iou": float(
                np.mean([row["visible_iou"] for row in processed_assignments])
            ),
        }
    manifest = {
        "schema_version": "2.0.0",
        "mode": args.mode,
        "mode_definition": MODE_DEFINITIONS[args.mode],
        "sequence": sequence,
        "source_scene": source_scene,
        "frame_count": frame_count,
        "raw_frames": expected_raw_frames,
        "online_from_empty_map": True,
        "future_final_lineage_used_for_mapping": False,
        "oracle_and_unassigned_native_maps_isolated": args.mode in {"oa", "op"},
        "seed": args.seed,
        "purity_threshold": args.purity_threshold,
        "min_overlap_pixels": args.min_overlap_pixels,
        "input_paths": {
            "config": str(args.config.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "detections_root": (
                str(args.detections_root.resolve()) if args.detections_root else None
            ),
            "baseline_map_audit_only": (
                str(args.baseline_map.resolve()) if args.baseline_map else None
            ),
            "gt_root": str(args.gt_root.resolve()),
            "objects_json": str(args.objects_json.resolve()),
        },
        "input_sha256": {
            "config": sha256_file(args.config.resolve()),
            "baseline_map_audit_only": (
                sha256_file(args.baseline_map.resolve()) if args.baseline_map else None
            ),
            "gt_manifest": sha256_file(gt_manifest_path),
            "objects_json": sha256_file(args.objects_json.resolve()),
            "detection_arrays_stream": (
                detection_digest.hexdigest() if args.mode != "og" else None
            ),
        },
        "gt_request": gt_request,
        "gt_alignment_summary": alignment,
        "provenance_audit": provenance_audit,
        "assignment_summary": assignment_summary,
        "operation_counts": dict(counters),
        "mean_partition_visible_support_recall": (
            float(np.mean(support_recall_values)) if support_recall_values else None
        ),
        "oracle_object_count": len(oracle_objects),
        "native_object_count": len(native_objects),
        "object_count": len(serialized),
        "observation_count": int(sum(obj["num_detections"] for obj in serialized)),
        "geometry_sha256": geometry_sha,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "frame_timing": frame_rows,
    }
    atomic_json(output / "manifest.json", manifest)
    if processed_assignments:
        atomic_json(output / "processed_assignment_diagnostics.json", processed_assignments)
    (output / "INCOMPLETE").unlink(missing_ok=True)
    (output / "READY").write_text("ready\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "mode",
                    "frame_count",
                    "oracle_object_count",
                    "native_object_count",
                    "object_count",
                    "observation_count",
                    "geometry_sha256",
                    "assignment_summary",
                    "operation_counts",
                    "elapsed_seconds",
                    "peak_rss_mb",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
