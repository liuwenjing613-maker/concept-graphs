#!/usr/bin/env python3
"""Build a pre-voxel PARTITION_OBSERVATION oracle from Replica semantic mesh."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import imageio.v2 as imageio
import open3d as o3d
from scipy.spatial import cKDTree

from conceptgraph.revision.constraints import (
    ConstraintEngine,
    SparseRepairConstraint,
)
from conceptgraph.revision.geometry import file_sha256
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.partition import (
    ObservationPartitionContract,
    ObservationPartitionPart,
    apply_observation_partition,
    observation_payload_sha256,
    partition_assignment_sha256,
)
from conceptgraph.slam.utils import (
    detections_to_obj_pcd_and_bbox,
    init_process_pcd,
)


_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
_QUAD_DTYPE = np.dtype(
    [
        ("count", "u1"),
        ("v0", "<u4"),
        ("v1", "<u4"),
        ("v2", "<u4"),
        ("v3", "<u4"),
    ]
)
_SEMANTIC_QUAD_DTYPE = np.dtype(_QUAD_DTYPE.descr + [("object_id", "<u2")])


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _resolve_ref(root: Path, ref: Mapping[str, Any]) -> Path:
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if ref.get("sha256") and file_sha256(path) != str(ref["sha256"]):
        raise ValueError(f"source artifact drift: {path}")
    return path


def _load_np_ref(root: Path, ref: Mapping[str, Any]) -> np.ndarray:
    path = _resolve_ref(root, ref)
    with np.load(path, allow_pickle=False) as archive:
        key = str(ref.get("key") or archive.files[0])
        if key not in archive.files:
            raise ValueError(f"missing {key} in {path}")
        value = np.asarray(archive[key])
        index = ref.get("index")
        return value.copy() if index is None else np.asarray(value[int(index)]).copy()


def _frame_record(provenance: ProvenanceIndex, frame_uid: str) -> dict[str, Any]:
    path = provenance.experiment_root / "evidence" / "frames.jsonl"
    matches = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("frame_uid")) == frame_uid:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"expected one frame record for {frame_uid}, found {len(matches)}"
        )
    return matches[0]


def _parse_replica_quad_mesh(
    path: Path, *, semantic: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    raw = path.read_bytes()
    marker = b"end_header\n"
    position = raw.find(marker)
    if position < 0:
        raise ValueError(f"PLY header terminator missing: {path}")
    offset = position + len(marker)
    header = raw[:offset].decode("ascii")
    vertex_match = re.search(r"element vertex (\d+)", header)
    face_match = re.search(r"element face (\d+)", header)
    if vertex_match is None or face_match is None:
        raise ValueError(f"PLY vertex/face counts missing: {path}")
    vertex_count = int(vertex_match.group(1))
    face_count = int(face_match.group(1))
    required_vertex_header = (
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    )
    if required_vertex_header not in header:
        raise ValueError(f"unsupported Replica vertex schema: {path}")
    if "property list uint8" not in header:
        raise ValueError(f"unsupported Replica face schema: {path}")
    face_dtype = _SEMANTIC_QUAD_DTYPE if semantic else _QUAD_DTYPE
    expected_size = (
        offset
        + vertex_count * _VERTEX_DTYPE.itemsize
        + face_count * face_dtype.itemsize
    )
    if expected_size != len(raw):
        raise ValueError(
            f"unexpected PLY binary size for quad parser: {path}; "
            f"expected={expected_size} actual={len(raw)}"
        )
    vertices_raw = np.frombuffer(
        raw, dtype=_VERTEX_DTYPE, count=vertex_count, offset=offset
    )
    face_offset = offset + vertex_count * _VERTEX_DTYPE.itemsize
    faces_raw = np.frombuffer(
        raw, dtype=face_dtype, count=face_count, offset=face_offset
    )
    if not np.all(faces_raw["count"] == 4):
        raise ValueError(f"Replica mesh is not all quads: {path}")
    vertices = np.column_stack(
        [vertices_raw["x"], vertices_raw["y"], vertices_raw["z"]]
    ).astype(np.float32, copy=True)
    faces = np.column_stack(
        [faces_raw["v0"], faces_raw["v1"], faces_raw["v2"], faces_raw["v3"]]
    ).astype(np.uint32, copy=True)
    object_ids = (
        np.asarray(faces_raw["object_id"], dtype=np.uint16).copy() if semantic else None
    )
    return (
        vertices,
        faces,
        object_ids,
        {
            "file_sha256": file_sha256(path),
            "vertex_count": vertex_count,
            "quad_count": face_count,
            "binary_size": len(raw),
            "header_sha256": __import__("hashlib").sha256(raw[:offset]).hexdigest(),
        },
    )


def _make_scene(
    vertices: np.ndarray, quads: np.ndarray, quad_object_ids: np.ndarray
) -> tuple[o3d.t.geometry.RaycastingScene, np.ndarray]:
    triangles = np.concatenate(
        [quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0
    ).astype(np.uint32, copy=False)
    triangle_object_ids = np.tile(np.asarray(quad_object_ids, dtype=np.uint16), 2)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene, triangle_object_ids


def _semantic_query(
    scene: o3d.t.geometry.RaycastingScene,
    triangle_object_ids: np.ndarray,
    points: np.ndarray,
) -> dict[str, np.ndarray]:
    query = scene.compute_closest_points(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    )
    closest = query["points"].numpy()
    primitive_ids = query["primitive_ids"].numpy().astype(np.int64)
    return {
        "closest_points": closest,
        "primitive_ids": primitive_ids,
        "object_ids": triangle_object_ids[primitive_ids].astype(np.uint16),
        "distance": np.linalg.norm(
            closest.astype(np.float64) - np.asarray(points, dtype=np.float64), axis=1
        ),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
    result = np.quantile(np.asarray(values, dtype=np.float64), levels)
    return {
        name: float(value)
        for name, value in zip(("min", "q50", "q90", "q95", "q99", "max"), result)
    }


def _counter_rows(
    values: np.ndarray, objects: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "replica_object_id": int(object_id),
            "class_name": str(objects[int(object_id)]["class_name"]),
            "count": int(count),
        }
        for object_id, count in sorted(
            Counter(int(item) for item in values).items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _voxel_mixing_audit(
    raw_pcd: o3d.geometry.PointCloud,
    raw_object_ids: np.ndarray,
    stored_points: np.ndarray,
    *,
    voxel_size: float,
    dbscan_eps: float,
    dbscan_min_points: int,
    scene: o3d.t.geometry.RaycastingScene,
    triangle_object_ids: np.ndarray,
) -> dict[str, Any]:
    raw_points = np.asarray(raw_pcd.points, dtype=np.float64)
    voxel_pcd = raw_pcd.voxel_down_sample(voxel_size)
    voxel_points = np.asarray(voxel_pcd.points, dtype=np.float64)
    min_bound = raw_points.min(axis=0) - voxel_size * 0.5
    voxel_keys = np.floor((raw_points - min_bound) / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(sums, inverse, raw_points)
    counts = np.bincount(inverse)
    centroids = sums / counts[:, None]
    centroid_distance, centroid_index = cKDTree(centroids).query(voxel_points, k=1)
    if float(centroid_distance.max(initial=0.0)) != 0.0:
        raise ValueError(
            "Open3D voxel output did not exactly match reconstructed centroids"
        )

    majority = np.empty(len(unique_keys), dtype=np.uint16)
    purity = np.empty(len(unique_keys), dtype=np.float64)
    mixed = np.zeros(len(unique_keys), dtype=bool)
    for group_index in range(len(unique_keys)):
        counter = Counter(int(item) for item in raw_object_ids[inverse == group_index])
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        total = sum(counter.values())
        majority[group_index] = ranked[0][0]
        purity[group_index] = ranked[0][1] / total
        mixed[group_index] = len(ranked) > 1
    voxel_majority = majority[centroid_index]
    voxel_purity = purity[centroid_index]
    voxel_mixed = mixed[centroid_index]

    cluster_labels = np.asarray(
        voxel_pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points)
    )
    cluster_counter = Counter(int(item) for item in cluster_labels)
    cluster_counter.pop(-1, None)
    if not cluster_counter:
        raise ValueError("reproduced voxel PCD has no DBSCAN cluster")
    largest_label = sorted(
        cluster_counter.items(), key=lambda item: (-item[1], item[0])
    )[0][0]
    keep = cluster_labels == largest_label
    final_points = voxel_points[keep]
    if not np.array_equal(final_points, stored_points):
        raise ValueError("reconstructed DBSCAN output does not match stored points")
    direct = _semantic_query(scene, triangle_object_ids, stored_points)["object_ids"]
    final_majority = voxel_majority[keep]
    final_purity = voxel_purity[keep]
    final_mixed = voxel_mixed[keep]
    return {
        "pre_dbscan_voxel_point_count": int(len(voxel_points)),
        "post_dbscan_stored_point_count": int(len(final_points)),
        "mixed_voxel_count_before_dbscan": int(voxel_mixed.sum()),
        "mixed_voxel_count_in_stored_payload": int(final_mixed.sum()),
        "mixed_voxel_ratio_in_stored_payload": float(final_mixed.mean()),
        "majority_vs_centroid_surface_disagreement_count": int(
            np.sum(final_majority != direct)
        ),
        "majority_purity_quantiles": _quantiles(final_purity),
        "stored_level_hard_partition_admissible": bool(not final_mixed.any()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--obs-uid", required=True)
    parser.add_argument("--mapping-mesh", required=True, type=Path)
    parser.add_argument("--semantic-mesh", required=True, type=Path)
    parser.add_argument("--semantic-info", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--retain-class", default="table")
    parser.add_argument("--max-surface-distance", default=0.003, type=float)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    provenance = ProvenanceIndex(args.base_run)
    source_hashes_before = provenance.source_hashes()
    observation = provenance.get_observation(args.obs_uid)
    if observation.get("status") != "kept":
        raise ValueError("partition source observation is not replayable")
    frame = _frame_record(provenance, str(observation["frame_uid"]))
    association = provenance.get_association_for_obs(args.obs_uid)
    config = _read_json(provenance.experiment_root / "config_params.json")

    pcd_path = _resolve_ref(provenance.experiment_root, observation["pcd_ref"])
    mask_path = _resolve_ref(
        provenance.experiment_root, observation["processed_mask_ref"]
    )
    depth_ref = frame.get("depth_ref") or {"path": frame["depth_path"]}
    rgb_ref = frame.get("rgb_ref") or {"path": frame["rgb_path"]}
    depth_path = _resolve_ref(provenance.experiment_root, depth_ref)
    rgb_path = _resolve_ref(provenance.experiment_root, rgb_ref)
    source_paths = {
        "stored_observation_pcd": pcd_path,
        "processed_mask": mask_path,
        "depth": depth_path,
        "rgb": rgb_path,
        "mapping_mesh": args.mapping_mesh.resolve(),
        "semantic_mesh": args.semantic_mesh.resolve(),
        "semantic_info": args.semantic_info.resolve(),
    }
    source_artifact_hashes_before = {
        role: file_sha256(path) for role, path in source_paths.items()
    }

    mapping_vertices, mapping_quads, _, mapping_mesh_audit = _parse_replica_quad_mesh(
        args.mapping_mesh.resolve(), semantic=False
    )
    (
        semantic_vertices,
        semantic_quads,
        semantic_object_ids,
        semantic_mesh_audit,
    ) = _parse_replica_quad_mesh(args.semantic_mesh.resolve(), semantic=True)
    geometry_exact = bool(
        np.array_equal(mapping_vertices, semantic_vertices)
        and np.array_equal(mapping_quads, semantic_quads)
    )
    if not geometry_exact:
        raise ValueError("mapping and semantic Replica mesh geometry differ")
    if semantic_object_ids is None:
        raise ValueError("semantic mesh did not expose object IDs")
    semantic_info = _read_json(args.semantic_info.resolve())
    objects = {int(item["id"]): item for item in semantic_info.get("objects") or ()}
    scene, triangle_object_ids = _make_scene(
        semantic_vertices, semantic_quads, semantic_object_ids
    )

    mask = np.asarray(
        _load_np_ref(provenance.experiment_root, observation["processed_mask_ref"]),
        dtype=bool,
    )
    depth_image = np.asarray(imageio.imread(depth_path), dtype=np.int64)
    rgb_u8 = np.asarray(imageio.imread(rgb_path), dtype=float).astype(np.uint8)
    # Mirror GradSLAMDataset.__getitem__ exactly. JPEG decoders can disagree by
    # several integer levels, so cv2.imread is not an admissible provenance
    # reconstruction for the stored color payload.
    # The original CUDA division used reciprocal multiplication in float32. CPU
    # division differs by one ULP for some values; emulate that arithmetic while
    # retaining the original utility's normalization step.
    rgb = (rgb_u8.astype(np.float32) * np.float32(1.0 / 255.0)) * np.float32(255.0)
    depth_scale = float(
        config.get("png_depth_scale") or config.get("depth_scale") or 6553.5
    )
    depth = (depth_image.astype(float) / depth_scale).astype(np.float32)
    intrinsics = np.asarray(frame["intrinsics"], dtype=np.float32)[:3, :3]
    pose = np.asarray(frame["pose"], dtype=np.float64)
    objects_3d = detections_to_obj_pcd_and_bbox(
        depth,
        np.asarray([mask]),
        intrinsics,
        rgb,
        pose,
        obj_pcd_max_points=int(config["obj_pcd_max_points"]),
        device="cpu",
    )
    if len(objects_3d) != 1 or objects_3d[0] is None:
        raise ValueError("failed to reconstruct source observation")
    raw_pcd = objects_3d[0]["pcd"]
    raw_payload = {
        "points": np.asarray(raw_pcd.points, dtype=np.float64).copy(),
        "colors": np.asarray(raw_pcd.colors, dtype=np.float64).copy(),
    }
    reconstructed, reconstruction_stats = init_process_pcd(
        raw_pcd,
        float(config["downsample_voxel_size"]),
        bool(config["dbscan_remove_noise"]),
        float(config["dbscan_eps"]),
        int(config["dbscan_min_points"]),
        return_stats=True,
    )
    reconstructed_points = np.asarray(reconstructed.points, dtype=np.float64)
    reconstructed_colors = np.asarray(reconstructed.colors, dtype=np.float64)
    with np.load(pcd_path, allow_pickle=False) as archive:
        stored_points = np.asarray(archive["points"], dtype=np.float64)
        stored_colors = np.asarray(archive["colors"], dtype=np.float64)
    points_exact = bool(np.array_equal(reconstructed_points, stored_points))
    colors_exact = bool(np.array_equal(reconstructed_colors, stored_colors))
    reconstruction_exact = points_exact and colors_exact
    if not reconstruction_exact:
        point_max_abs = (
            float(np.max(np.abs(reconstructed_points - stored_points)))
            if reconstructed_points.shape == stored_points.shape
            else None
        )
        color_max_abs = (
            float(np.max(np.abs(reconstructed_colors - stored_colors)))
            if reconstructed_colors.shape == stored_colors.shape
            else None
        )
        raise ValueError(
            "source preprocessing did not exactly reproduce stored PCD: "
            f"points_exact={points_exact} point_max_abs={point_max_abs} "
            f"colors_exact={colors_exact} color_max_abs={color_max_abs}"
        )

    semantic_first = _semantic_query(scene, triangle_object_ids, raw_payload["points"])
    semantic_second = _semantic_query(scene, triangle_object_ids, raw_payload["points"])
    query_deterministic = all(
        np.array_equal(semantic_first[name], semantic_second[name])
        for name in ("primitive_ids", "object_ids", "closest_points", "distance")
    )
    assigned_object_ids = semantic_first["object_ids"]
    unknown_object_ids = sorted(
        set(int(item) for item in assigned_object_ids) - set(objects)
    )
    if unknown_object_ids:
        raise ValueError(
            f"semantic mesh object IDs absent from metadata: {unknown_object_ids}"
        )
    max_distance = float(semantic_first["distance"].max(initial=0.0))
    if max_distance > args.max_surface_distance:
        raise ValueError(
            f"semantic surface distance exceeds gate: {max_distance} > "
            f"{args.max_surface_distance}"
        )

    unique_object_ids = sorted(set(int(item) for item in assigned_object_ids))
    part_index_by_object = {
        object_id: index for index, object_id in enumerate(unique_object_ids)
    }
    assignment = np.asarray(
        [part_index_by_object[int(item)] for item in assigned_object_ids],
        dtype=np.uint16,
    )
    assignment_hash = partition_assignment_sha256(assignment)
    args.output_root.mkdir(parents=True, exist_ok=True)
    assignment_path = args.output_root / "prevoxel_partition_assignment.npy"
    np.save(assignment_path, assignment, allow_pickle=False)
    assignment_file_hash = file_sha256(assignment_path)
    if (
        partition_assignment_sha256(np.load(assignment_path, allow_pickle=False))
        != assignment_hash
    ):
        raise ValueError("written assignment artifact failed content hash verification")

    parts = []
    retained_class = args.retain_class.strip().lower()
    for object_id in unique_object_ids:
        metadata = objects[object_id]
        class_name = str(metadata["class_name"]).lower()
        disposition = (
            "EMIT_OBSERVATION"
            if class_name == retained_class
            else "EXCLUDE_AS_CONTAMINATION"
        )
        parts.append(
            ObservationPartitionPart(
                part_index=part_index_by_object[object_id],
                part_uid=f"replica_object_{object_id:05d}_{class_name}",
                identity_uid=f"replica-instance:room_0:object:{object_id}",
                label=f"{class_name}#{object_id}",
                disposition=disposition,
            )
        )

    contract = ObservationPartitionContract(
        obs_uid=args.obs_uid,
        source_point_count=len(raw_payload["points"]),
        source_payload_sha256=observation_payload_sha256(raw_payload),
        assignment_sha256=assignment_hash,
        parts=tuple(parts),
        evidence_refs=(
            f"human-confirmed:{args.obs_uid}:floor_contamination",
            f"replica-semantic-mesh-sha256:{source_artifact_hashes_before['semantic_mesh']}",
            f"replica-semantic-info-sha256:{source_artifact_hashes_before['semantic_info']}",
        ),
        source_stage="PRE_VOXEL_SAMPLED_PAYLOAD",
        assignment_ref={
            "path": str(assignment_path.resolve()),
            "format": "npy",
            "sha256": assignment_file_hash,
            "assignment_sha256": assignment_hash,
        },
    )
    execution = apply_observation_partition(
        contract,
        payload=raw_payload,
        assignment=assignment,
    )
    constraint = SparseRepairConstraint.from_mapping(
        {
            "type": "PARTITION_OBSERVATION",
            "obs_uid": args.obs_uid,
            "partition_contract": contract.as_dict(),
            "applies_at_event_uid": association["event_uid"],
            "active_from_sequence": provenance.sequence(association),
            "source": "replica_semantic_mesh_development_oracle",
            "reason": "human-confirmed undersegmentation translated through official semantic mesh",
            "evidence_refs": list(contract.evidence_refs),
        }
    )
    association_stage_decision = ConstraintEngine([constraint]).resolve_for_observation(
        obs_uid=args.obs_uid,
        event_uid=str(association["event_uid"]),
        event_sequence=provenance.sequence(association),
        natural_match=None,
        natural_candidates=[],
    )
    voxel_audit = _voxel_mixing_audit(
        raw_pcd,
        assigned_object_ids,
        stored_points,
        voxel_size=float(config["downsample_voxel_size"]),
        dbscan_eps=float(config["dbscan_eps"]),
        dbscan_min_points=int(config["dbscan_min_points"]),
        scene=scene,
        triangle_object_ids=triangle_object_ids,
    )

    source_hashes_after = provenance.source_hashes()
    source_artifact_hashes_after = {
        role: file_sha256(path) for role, path in source_paths.items()
    }
    emitted_parts = [part for part in parts if part.disposition == "EMIT_OBSERVATION"]
    excluded_parts = [
        part for part in parts if part.disposition == "EXCLUDE_AS_CONTAMINATION"
    ]
    checks = {
        "mapping_semantic_mesh_geometry_exact": geometry_exact,
        "source_preprocessing_reconstruction_exact": reconstruction_exact,
        "source_count_trace_exact": (
            int(reconstruction_stats["input_points"])
            == int(observation["pcd_before_downsample_points"])
            and int(reconstruction_stats["after_downsample_points"])
            == int(observation["pcd_after_downsample_points"])
            and int(reconstruction_stats["after_dbscan_points"])
            == int(observation["pcd_after_dbscan_points"])
        ),
        "semantic_query_deterministic": query_deterministic,
        "semantic_surface_distance_within_gate": max_distance
        <= args.max_surface_distance,
        "semantic_object_metadata_complete": not unknown_object_ids,
        "assignment_roundtrip_exact": partition_assignment_sha256(
            np.load(assignment_path, allow_pickle=False)
        )
        == assignment_hash,
        "partition_executor_pass": bool(execution.validation["pass"]),
        "partition_exhaustive": bool(execution.validation["exhaustive"]),
        "partition_disjoint": bool(execution.validation["disjoint"]),
        "two_table_instances_emitted": len(emitted_parts) == 2,
        "three_contamination_instances_excluded": len(excluded_parts) == 3,
        "association_stage_fail_closed_until_integrated": (
            association_stage_decision.action.value == "DEFER"
            and association_stage_decision.reason
            == "partition_observation_requires_pre_association_payload_stage"
        ),
        "provenance_hashes_unchanged": source_hashes_before == source_hashes_after,
        "source_artifact_hashes_unchanged": source_artifact_hashes_before
        == source_artifact_hashes_after,
    }
    result = {
        "schema_version": "3.0.0",
        "evaluation_role": (
            "DEVELOPMENT_ORACLE_POINT_GOLD_AND_PRE_ASSOCIATION_PURE_EXECUTOR"
        ),
        "production_commit_permitted": False,
        "oracle_only": True,
        "case_uid": "human6_room0_false_merge_f74cb76c",
        "obs_uid": args.obs_uid,
        "event_uid": association["event_uid"],
        "constraint_uid": constraint.constraint_uid,
        "partition_uid": contract.partition_uid,
        "pass": all(checks.values()),
        "checks": checks,
        "design_decision": {
            "stored_point_v1": "REJECT_AS_LOSSY",
            "prevoxel_v2": "ADMIT_AS_DEVELOPMENT_ORACLE_POINT_GOLD",
            "full_sparse_replay": "DEFER_UNTIL_ONE_TO_MANY_PRE_ASSOCIATION_INTEGRATION",
            "reason": (
                "Voxelization mixes semantic instances in some stored points; "
                "the exact deterministic partition boundary is the sampled "
                "pre-voxel payload, with non-table parts excluded as contamination."
            ),
        },
        "mesh_alignment": {
            "geometry_exact": geometry_exact,
            "mapping": mapping_mesh_audit,
            "semantic": semantic_mesh_audit,
        },
        "source_reconstruction": {
            "exact": reconstruction_exact,
            "point_reconstruction_exact": points_exact,
            "color_reconstruction_exact": colors_exact,
            "rgb_decoder": "imageio.v2.imread",
            "color_normalization_replay": (
                "CUDA_FLOAT32_RECIPROCAL_MULTIPLICATION_EMULATED_ON_CPU"
            ),
            "raw_mask_valid_depth_point_count": int(mask.sum()),
            "prevoxel_sampled_point_count": int(len(raw_payload["points"])),
            "stored_point_count": int(len(stored_points)),
            "stats": reconstruction_stats,
            "source_payload_sha256": contract.source_payload_sha256,
        },
        "point_gold": {
            "source_stage": contract.source_stage,
            "assignment_path": str(assignment_path.resolve()),
            "assignment_file_sha256": assignment_file_hash,
            "assignment_sha256": assignment_hash,
            "semantic_surface_distance": _quantiles(semantic_first["distance"]),
            "semantic_instance_counts": _counter_rows(assigned_object_ids, objects),
            "part_index_by_replica_object_id": {
                str(key): value for key, value in part_index_by_object.items()
            },
        },
        "voxel_information_loss": voxel_audit,
        "partition_contract": contract.as_dict(),
        "constraint_candidate": constraint.as_dict(),
        "pure_execution": {
            "validation": execution.validation,
            "emitted_parts": [
                {
                    "part_uid": item.part_uid,
                    "identity_uid": item.identity_uid,
                    "point_count": item.point_count,
                    "disposition": item.disposition,
                }
                for item in execution.parts
            ],
            "excluded_parts": [
                {
                    "part_uid": item.part_uid,
                    "identity_uid": item.identity_uid,
                    "point_count": item.point_count,
                    "disposition": item.disposition,
                }
                for item in execution.excluded_parts
            ],
        },
        "association_stage_decision": association_stage_decision.as_dict(),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_artifact_hashes_before": source_artifact_hashes_before,
        "source_artifact_hashes_after": source_artifact_hashes_after,
    }
    _write_json(args.output_root / "partition_oracle_manifest.json", result)
    if args.audit_output is not None:
        _write_json(args.audit_output.resolve(), result)
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "partition_uid": contract.partition_uid,
                "prevoxel_point_count": len(raw_payload["points"]),
                "semantic_instance_counts": result["point_gold"][
                    "semantic_instance_counts"
                ],
                "mixed_stored_voxel_count": voxel_audit[
                    "mixed_voxel_count_in_stored_payload"
                ],
                "output": str(
                    (args.output_root / "partition_oracle_manifest.json").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
