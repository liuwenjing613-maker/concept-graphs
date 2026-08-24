#!/usr/bin/env python3
"""Build a hash-bound raw-mask geometry restoration payload and constraint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from conceptgraph.revision.constraints import SparseRepairConstraint
from conceptgraph.revision.geometry import (
    ObservationGeometryContract,
    array_sha256,
    canonical_json_sha256,
    file_sha256,
)
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.materialize import ObservationMaterializer


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_frame(provenance: ProvenanceIndex, frame_uid: str) -> dict[str, Any]:
    path = provenance.evidence_root / "frames.jsonl"
    with path.open(encoding="utf-8") as handle:
        matches = [
            json.loads(line) for line in handle if line.strip() and frame_uid in line
        ]
    matches = [row for row in matches if str(row.get("frame_uid")) == frame_uid]
    if len(matches) != 1:
        raise ValueError(f"expected one frame record {frame_uid}, found {len(matches)}")
    return matches[0]


def _checked_path(
    materializer: ObservationMaterializer,
    ref: dict[str, Any],
) -> Path:
    path = materializer.resolve_ref(ref)
    expected = str(ref.get("sha256") or "")
    actual = file_sha256(path)
    if expected != actual:
        raise ValueError(f"source artifact drift: {path}; {expected} != {actual}")
    return path


def _source_ref(
    *,
    role: str,
    path: Path,
    source_ref: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "role": role,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "format": str(source_ref.get("format") or path.suffix.lstrip(".")),
    }
    for field in ("key", "index", "shape", "dtype"):
        if source_ref.get(field) is not None:
            result[field] = source_ref[field]
    return result


def _derive_geometry(
    *,
    raw_mask: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    pose: np.ndarray,
    voxel_size: float,
    dbscan_eps: float,
    dbscan_min_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = raw_mask & np.isfinite(depth) & (depth > 0)
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera_points = np.stack(
        (
            (columns - cx) * z / fx,
            (rows - cy) * z / fy,
            z,
        ),
        axis=1,
    )
    homogeneous = np.concatenate(
        (camera_points, np.ones((len(camera_points), 1), dtype=np.float64)),
        axis=1,
    )
    world_points = (pose @ homogeneous.T).T[:, :3]
    colors = np.asarray(rgb[rows, columns], dtype=np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(world_points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud = cloud.voxel_down_sample(voxel_size=float(voxel_size))
    voxel_points = np.asarray(cloud.points, dtype=np.float64)
    voxel_colors = np.asarray(cloud.colors, dtype=np.float64)
    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=float(dbscan_eps),
            min_points=int(dbscan_min_points),
        ),
        dtype=np.int64,
    )
    counts = Counter(int(value) for value in labels)
    noise_count = int(counts.pop(-1, 0))
    ordered = counts.most_common()
    if not ordered:
        raise ValueError("raw-mask geometry has no valid DBSCAN component")
    largest_label, largest_count = ordered[0]
    if largest_count < 5:
        raise ValueError("raw-mask largest component has fewer than five points")
    keep = labels == largest_label
    points = voxel_points[keep]
    selected_colors = voxel_colors[keep]
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = np.ascontiguousarray(points[order], dtype=np.float64)
    selected_colors = np.ascontiguousarray(selected_colors[order], dtype=np.float64)
    stats = {
        "valid_raw_depth_point_count": int(valid.sum()),
        "voxel_point_count": int(len(voxel_points)),
        "dbscan_cluster_count": int(len(ordered)),
        "dbscan_noise_point_count": noise_count,
        "largest_component_point_count": int(len(points)),
        "largest_component_ratio_to_voxel": float(
            len(points) / max(len(voxel_points), 1)
        ),
        "second_component_point_count": (int(ordered[1][1]) if len(ordered) > 1 else 0),
    }
    return points, selected_colors, stats


def _aabb(points: np.ndarray) -> dict[str, Any]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "center": ((minimum + maximum) / 2.0).tolist(),
        "extent": (maximum - minimum).tolist(),
        "volume": float(np.prod(maximum - minimum)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--obs-uid", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--depth-scale", type=float, default=6553.5)
    parser.add_argument(
        "--evaluation-role",
        default="DEVELOPMENT_GEOMETRY_CAPABILITY_NOT_HOLDOUT",
    )
    parser.add_argument(
        "--constraint-source",
        default="human_confirmed_raw_mask_geometry_development",
    )
    parser.add_argument(
        "--scene-id",
        help="Explicit scene binding; otherwise derived from the frozen frame UID.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    manifest_path = output_root / "build_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    provenance = ProvenanceIndex(args.base_run.resolve())
    source_hashes_before = provenance.source_hashes()
    row = provenance.get_observation(args.obs_uid)
    if row.get("status") != "kept":
        raise ValueError("geometry restoration requires a kept observation")
    association = provenance.get_association_for_obs(args.obs_uid)
    frame = _load_frame(provenance, str(row["frame_uid"]))
    materializer = ObservationMaterializer(provenance)
    derived_scene_id = str(frame["frame_uid"]).split("_", 1)[0]
    scene_id = str(args.scene_id or derived_scene_id)
    if scene_id != derived_scene_id:
        raise ValueError(f"scene binding drift: {scene_id} != {derived_scene_id}")

    raw_mask_path = _checked_path(materializer, row["raw_mask_ref"])
    processed_mask_path = _checked_path(materializer, row["processed_mask_ref"])
    original_pcd_path = _checked_path(materializer, row["pcd_ref"])
    depth_path = _checked_path(materializer, frame["depth_ref"])
    rgb_path = _checked_path(materializer, frame["rgb_ref"])
    raw_mask = np.asarray(materializer.load_ref(row["raw_mask_ref"]), dtype=bool)
    processed_mask = np.asarray(
        materializer.load_ref(row["processed_mask_ref"]), dtype=bool
    )
    depth_encoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if depth_encoded is None or rgb_bgr is None:
        raise ValueError("failed to load frozen RGB/depth artifacts")
    depth = np.asarray(depth_encoded, dtype=np.float64) / float(args.depth_scale)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if raw_mask.shape != depth.shape or rgb.shape[:2] != depth.shape:
        raise ValueError("raw mask, RGB, and depth shapes disagree")

    with (provenance.experiment_root / "config_params.json").open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)[:3, :3]
    pose = np.asarray(frame["pose"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("invalid frozen camera calibration")
    parameters = {
        "voxel_size": float(config["downsample_voxel_size"]),
        "dbscan_eps": float(config["dbscan_eps"]),
        "dbscan_min_points": int(config["dbscan_min_points"]),
    }
    first = _derive_geometry(
        raw_mask=raw_mask,
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        pose=pose,
        **parameters,
    )
    second = _derive_geometry(
        raw_mask=raw_mask,
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        pose=pose,
        **parameters,
    )
    points, colors, statistics = first
    deterministic_rederive = bool(
        np.array_equal(points, second[0])
        and np.array_equal(colors, second[1])
        and statistics == second[2]
    )
    if not deterministic_rederive:
        raise ValueError("geometry derivation is not deterministic")

    payload_dir = output_root / args.obs_uid
    pcd_path = payload_dir / "restored_observation_pcd.npz"
    mask_path = payload_dir / "restored_observation_mask.npz"
    _write_npz(pcd_path, points=points, colors=colors)
    _write_npz(mask_path, mask=raw_mask)
    source_artifacts = [
        _source_ref(
            role="raw_mask", path=raw_mask_path, source_ref=row["raw_mask_ref"]
        ),
        _source_ref(
            role="processed_mask",
            path=processed_mask_path,
            source_ref=row["processed_mask_ref"],
        ),
        _source_ref(role="depth", path=depth_path, source_ref=frame["depth_ref"]),
        _source_ref(role="rgb", path=rgb_path, source_ref=frame["rgb_ref"]),
        _source_ref(
            role="original_observation_pcd",
            path=original_pcd_path,
            source_ref=row["pcd_ref"],
        ),
    ]
    derivation = {
        "algorithm": "RAW_MASK_DEPTH_WORLD_PCD_V1",
        "random_perturbation": False,
        "depth_scale": float(args.depth_scale),
        "intrinsics": intrinsics.tolist(),
        "intrinsics_sha256": canonical_json_sha256(intrinsics.tolist()),
        "pose": pose.tolist(),
        "pose_sha256": canonical_json_sha256(pose.tolist()),
        "voxel_size": parameters["voxel_size"],
        "dbscan_eps": parameters["dbscan_eps"],
        "dbscan_min_points": parameters["dbscan_min_points"],
        "component_policy": "LARGEST_DBSCAN_COMPONENT",
        "deterministic_lexicographic_point_order": True,
        "replacement_points_sha256": array_sha256(points),
        "replacement_colors_sha256": array_sha256(colors),
        "replacement_mask_array_sha256": array_sha256(raw_mask),
        **statistics,
    }
    contract = ObservationGeometryContract.build(
        obs_uid=args.obs_uid,
        replacement_pcd_ref={
            "path": str(pcd_path),
            "sha256": file_sha256(pcd_path),
            "format": "npz",
            "shape": list(points.shape),
            "dtype": str(points.dtype),
        },
        replacement_mask_ref={
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
            "format": "npz",
            "key": "mask",
            "shape": list(raw_mask.shape),
            "dtype": str(raw_mask.dtype),
        },
        source_observation_sha256=canonical_json_sha256(row),
        source_artifacts=source_artifacts,
        derivation=derivation,
    )
    contract.verify_source_bindings(row, base_root=provenance.experiment_root)
    loaded_payload = contract.load_payload(base_root=provenance.experiment_root)
    constraint = SparseRepairConstraint.from_mapping(
        {
            "type": "RESTORE_OBSERVATION_GEOMETRY",
            "obs_uid": args.obs_uid,
            "geometry_contract": contract.as_dict(),
            "applies_at_event_uid": association["event_uid"],
            "active_from_sequence": provenance.sequence(association),
            "source": str(args.constraint_source),
            "evidence_refs": [
                args.obs_uid,
                association["event_uid"],
                contract.payload_uid,
            ],
        }
    )
    original_payload = np.load(original_pcd_path, allow_pickle=False)
    original_points = np.asarray(original_payload["points"], dtype=np.float64)
    source_hashes_after = provenance.source_hashes()
    source_artifact_hashes_after = {
        item["role"]: file_sha256(item["path"]) for item in source_artifacts
    }
    geometry_metrics = {
        "raw_mask_area": int(raw_mask.sum()),
        "processed_mask_area": int(processed_mask.sum()),
        "removed_pixel_count": int(raw_mask.sum() - processed_mask.sum()),
        "loss_ratio": float(1.0 - processed_mask.sum() / max(raw_mask.sum(), 1)),
        "restored_mask_exact_to_raw": bool(
            np.array_equal(loaded_payload["mask"], raw_mask)
        ),
        "restored_mask_contains_processed": bool(np.all(~processed_mask | raw_mask)),
        "original_observation_point_count": int(len(original_points)),
        "restored_observation_point_count": int(len(points)),
        "point_support_gain_ratio": float(len(points) / max(len(original_points), 1)),
        "original_aabb": _aabb(original_points),
        "restored_aabb": _aabb(points),
        "deterministic_rederive_exact": deterministic_rederive,
    }
    manifest = {
        "schema_version": "1.0.0",
        "evaluation_role": str(args.evaluation_role),
        "scene_id": scene_id,
        "obs_uid": args.obs_uid,
        "event_uid": association["event_uid"],
        "event_sequence": provenance.sequence(association),
        "contract": contract.as_dict(),
        "constraint": constraint.as_dict(),
        "geometry_metrics": geometry_metrics,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_hashes_unchanged": source_hashes_before == source_hashes_after,
        "source_artifact_hashes_after": source_artifact_hashes_after,
        "source_artifacts_unchanged": all(
            source_artifact_hashes_after[item["role"]] == item["sha256"]
            for item in source_artifacts
        ),
    }
    manifest["pass"] = bool(
        geometry_metrics["raw_mask_area"] > geometry_metrics["processed_mask_area"]
        and geometry_metrics["restored_observation_point_count"]
        > geometry_metrics["original_observation_point_count"]
        and geometry_metrics["deterministic_rederive_exact"]
        and manifest["source_hashes_unchanged"]
        and manifest["source_artifacts_unchanged"]
    )
    _write_json(payload_dir / "geometry_contract.json", contract.as_dict())
    _write_json(payload_dir / "constraint.json", constraint.as_dict())
    _write_json(manifest_path, manifest)
    audit = {
        "schema_version": manifest["schema_version"],
        "evaluation_role": manifest["evaluation_role"],
        "obs_uid": manifest["obs_uid"],
        "event_uid": manifest["event_uid"],
        "payload_uid": contract.payload_uid,
        "constraint_uid": constraint.constraint_uid,
        "geometry_metrics": geometry_metrics,
        "source_hashes_unchanged": manifest["source_hashes_unchanged"],
        "source_artifacts_unchanged": manifest["source_artifacts_unchanged"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "pass": manifest["pass"],
    }
    if args.audit_output is not None:
        _write_json(args.audit_output.resolve(), audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if manifest["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
