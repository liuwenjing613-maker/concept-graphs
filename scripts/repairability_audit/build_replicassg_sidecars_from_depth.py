#!/usr/bin/env python3
"""Build frame-aligned ReplicaSSG instance sidecars from RGB-D geometry.

This avoids Habitat semantic-camera rendering.  Each valid depth pixel is
unprojected with the frozen Replica intrinsics and raw c2w pose, then assigned
the nearest annotated ReplicaSSG face-centre instance ID.  A strict distance
gate keeps geometry outside the semantic mesh unlabeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


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
UNLABELED_ID = np.iinfo(np.uint16).max


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_labels(objects_path: Path, source_scene: str) -> dict[int, str]:
    payload = json.loads(objects_path.read_text(encoding="utf-8"))
    scans = [item for item in payload["scans"] if item["scan"] == source_scene]
    if len(scans) != 1:
        raise ValueError(f"expected one objects entry for {source_scene}")
    return {int(item["id"]): str(item["label"]) for item in scans[0]["objects"]}


def load_semantic_face_centres(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = PlyData.read(str(path))["vertex"].data
    required = {"x", "y", "z", "objectId"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"{path} lacks required properties {sorted(required)}")
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(
        np.float32
    )
    object_ids = np.asarray(vertices["objectId"], dtype=np.uint16)
    if len(xyz) != len(object_ids) or not len(xyz):
        raise ValueError(f"invalid semantic face-centre mesh: {path}")
    return xyz, object_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=sorted(SCENE_NAMES), required=True)
    parser.add_argument("--replica-root", type=Path, required=True)
    parser.add_argument("--replica-ssg-root", type=Path, required=True)
    parser.add_argument("--objects-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=680)
    parser.add_argument("--fx", type=float, default=600.0)
    parser.add_argument("--fy", type=float, default=600.0)
    parser.add_argument("--cx", type=float, default=599.5)
    parser.add_argument("--cy", type=float, default=339.5)
    parser.add_argument("--depth-scale", type=float, default=6553.5)
    parser.add_argument("--max-distance-m", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride < 1 or args.start < 0 or args.max_distance_m <= 0:
        raise ValueError("invalid frame range or distance gate")
    sequence_root = args.replica_root.resolve() / args.sequence
    trajectory_path = sequence_root / "traj.txt"
    results_root = sequence_root / "results"
    source_scene = SCENE_NAMES[args.sequence]
    semantic_mesh = (
        args.replica_ssg_root.resolve()
        / "Replica"
        / "data"
        / source_scene
        / "labels.instances.annotated.v2.ply"
    )
    objects_path = args.objects_json.resolve()
    for required in (trajectory_path, results_root, semantic_mesh, objects_path):
        if not required.exists():
            raise FileNotFoundError(required)

    poses = np.loadtxt(trajectory_path).reshape(-1, 4, 4)
    stop = len(poses) if args.end < 0 else min(args.end, len(poses))
    frames = list(range(args.start, stop, args.stride))
    if not frames:
        raise ValueError("requested frame range is empty")
    labels = load_labels(objects_path, source_scene)
    semantic_xyz, semantic_object_ids = load_semantic_face_centres(semantic_mesh)
    if not set(np.unique(semantic_object_ids)).issubset(labels):
        missing = sorted(set(map(int, np.unique(semantic_object_ids))) - set(labels))
        raise ValueError(f"semantic mesh IDs missing from objects.json: {missing}")
    tree = cKDTree(semantic_xyz)

    output = args.output_root.resolve() / args.sequence
    ready = output / "READY"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ready.unlink(missing_ok=True)
    (output / "INCOMPLETE").write_text(
        f"started_at_unix={time.time()}\n", encoding="utf-8"
    )

    yy, xx = np.mgrid[0 : args.height, 0 : args.width]
    diagnostics: list[dict[str, object]] = []
    visible_counts: Counter[int] = Counter()
    started = time.perf_counter()
    for ordinal, raw_frame in enumerate(frames):
        frame_started = time.perf_counter()
        depth_path = results_root / f"depth{raw_frame:06d}.png"
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / args.depth_scale
        if depth.shape != (args.height, args.width):
            raise ValueError(f"frame {raw_frame}: unexpected depth shape {depth.shape}")
        valid = depth > 0
        z = depth[valid]
        camera_xyz = np.column_stack(
            [
                (xx[valid] - args.cx) * z / args.fx,
                (yy[valid] - args.cy) * z / args.fy,
                z,
            ]
        ).astype(np.float32, copy=False)
        pose = poses[raw_frame]
        world_xyz = camera_xyz @ pose[:3, :3].T + pose[:3, 3]

        nearest = np.empty(len(world_xyz), dtype=np.int64)
        distances = np.empty(len(world_xyz), dtype=np.float32)
        for begin in range(0, len(world_xyz), args.query_chunk):
            end = min(begin + args.query_chunk, len(world_xyz))
            chunk_distance, chunk_nearest = tree.query(
                world_xyz[begin:end], k=1, workers=args.workers
            )
            distances[begin:end] = chunk_distance
            nearest[begin:end] = chunk_nearest

        accepted = distances <= args.max_distance_m
        assigned = np.full(len(z), UNLABELED_ID, dtype=np.uint16)
        assigned[accepted] = semantic_object_ids[nearest[accepted]]
        semantic = np.full(depth.shape, UNLABELED_ID, dtype=np.uint16)
        semantic[valid] = assigned
        target = output / f"frame{raw_frame:06d}.npz"
        temporary = output / f"frame{raw_frame:06d}.npz.incomplete"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, semantic=semantic)
        temporary.replace(target)

        ids, counts = np.unique(assigned[accepted], return_counts=True)
        visible_counts.update(
            {int(instance_id): int(count) for instance_id, count in zip(ids, counts)}
        )
        diagnostics.append(
            {
                "ordinal": ordinal,
                "raw_frame": raw_frame,
                "valid_depth_pixels": int(valid.sum()),
                "accepted_pixels": int(accepted.sum()),
                "accepted_fraction": float(accepted.mean()),
                "distance_mean_m": float(distances.mean()),
                "distance_p50_m": float(np.quantile(distances, 0.50)),
                "distance_p90_m": float(np.quantile(distances, 0.90)),
                "distance_p95_m": float(np.quantile(distances, 0.95)),
                "distance_p99_m": float(np.quantile(distances, 0.99)),
                "within_5cm": float(np.mean(distances <= 0.05)),
                "elapsed_seconds": time.perf_counter() - frame_started,
            }
        )
        if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(frames):
            print(f"built {ordinal + 1}/{len(frames)}", flush=True)

    request = {
        "schema_version": "2.0.0",
        "method": "current-frame depth unprojection to nearest ReplicaSSG annotated face centre",
        "sequence": args.sequence,
        "source_scene": source_scene,
        "frames": frames,
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "width": args.width,
        "height": args.height,
        "depth_scale": args.depth_scale,
        "trajectory_sha256": sha256_file(trajectory_path),
        "semantic_mesh_sha256": sha256_file(semantic_mesh),
        "objects_sha256": sha256_file(objects_path),
    }
    manifest = {
        **request,
        "frame_count": len(frames),
        "intrinsics": {
            "width": args.width,
            "height": args.height,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "depth_scale": args.depth_scale,
        },
        "max_distance_m": args.max_distance_m,
        "unlabeled_id": int(UNLABELED_ID),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "semantic_mesh": str(semantic_mesh),
        "semantic_mesh_sha256": sha256_file(semantic_mesh),
        "objects_json": str(objects_path),
        "objects_json_sha256": sha256_file(objects_path),
        "alignment": diagnostics,
        "alignment_summary": {
            "max_median_abs_depth_m": max(
                row["distance_p50_m"] for row in diagnostics
            ),
            "max_p90_abs_depth_m": max(
                row["distance_p90_m"] for row in diagnostics
            ),
            "max_p99_abs_depth_m": max(
                row["distance_p99_m"] for row in diagnostics
            ),
            "min_within_5cm": min(row["within_5cm"] for row in diagnostics),
            "min_accepted_fraction": min(row["accepted_fraction"] for row in diagnostics),
            "max_distance_p95_m": max(row["distance_p95_m"] for row in diagnostics),
            "max_distance_p99_m": max(row["distance_p99_m"] for row in diagnostics),
        },
        "visible_instances": [
            {
                "id": instance_id,
                "label": labels[instance_id],
                "accumulated_visible_pixels": count,
            }
            for instance_id, count in sorted(visible_counts.items())
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "request.json", request)
    atomic_json(output / "manifest.json", manifest)
    (output / "INCOMPLETE").unlink(missing_ok=True)
    ready.write_text("ready\n", encoding="utf-8")
    print(json.dumps(manifest["alignment_summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
