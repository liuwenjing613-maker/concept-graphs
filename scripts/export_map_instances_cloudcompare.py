#!/usr/bin/env python3
"""Export a ConceptGraphs object map as instance-colored CloudCompare PLY files.

The combined PLY stores RGB plus two scalar fields (``instance_id`` and
``is_background``).  A deterministic high-contrast color is assigned to each
map object.  Optional per-instance PLY files make it easy to toggle individual
objects in CloudCompare's DB Tree.

Only trusted ConceptGraphs pickle files should be used: Python pickle loading is
not safe for untrusted input.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("instance_id", "<u4"),
        ("is_background", "u1"),
    ]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a ConceptGraphs pcd_*.pkl.gz map to instance-colored binary "
            "PLY files for CloudCompare."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A pcd_*.pkl.gz file, an uncompressed .pkl, or a run directory containing exactly one pcd_*.pkl.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory. Default: <run-dir>/cloudcompare_instances. It must not already exist.",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also write one PLY per instance for independent DB Tree visibility control.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Skip map objects with fewer finite points than this value (default: 1).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional visualization-only voxel size in meters; 0 preserves all stored points.",
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Omit objects whose is_background flag is true.",
    )
    parser.add_argument(
        "--color-offset",
        type=int,
        default=0,
        help="Integer offset for the deterministic color sequence (default: 0).",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if not (path.name.endswith(".pkl.gz") or path.suffix == ".pkl"):
            raise ValueError(f"Expected .pkl.gz or .pkl input, got: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    matches = sorted(path.glob("pcd_*.pkl.gz"))
    if len(matches) != 1:
        names = ", ".join(item.name for item in matches) or "none"
        raise ValueError(
            f"Run directory must contain exactly one pcd_*.pkl.gz; found {len(matches)}: {names}. "
            "Pass the desired file explicitly."
        )
    return matches[0].resolve()


def load_objects(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if "objects" not in payload:
            raise ValueError("Dictionary payload does not contain an 'objects' key")
        objects = payload["objects"]
        context = {
            "payload_keys": sorted(str(key) for key in payload),
            "class_names_count": len(payload.get("class_names", [])),
        }
    elif isinstance(payload, list):
        objects = payload
        context = {"payload_keys": None, "class_names_count": None}
    else:
        raise TypeError(f"Unsupported payload type: {type(payload).__name__}")
    if not isinstance(objects, list):
        objects = list(objects)
    if any(not isinstance(obj, dict) for obj in objects):
        raise TypeError("Every map object must be a dictionary")
    return objects, context


def object_points(obj: dict[str, Any]) -> np.ndarray:
    if "pcd_np" in obj:
        points = np.asarray(obj["pcd_np"])
    elif "pcd" in obj and hasattr(obj["pcd"], "points"):
        points = np.asarray(obj["pcd"].points)
    else:
        raise KeyError("Map object has neither pcd_np nor an Open3D-style pcd.points")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point array must have shape (N, 3), got {points.shape}")
    return np.asarray(points, dtype=np.float64)


def finite_and_voxelized(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, int]:
    finite = np.isfinite(points).all(axis=1)
    invalid_count = int(len(points) - finite.sum())
    points = points[finite]
    if voxel_size > 0 and len(points):
        cells = np.floor(points / voxel_size).astype(np.int64)
        _, indices = np.unique(cells, axis=0, return_index=True)
        points = points[np.sort(indices)]
    return points, invalid_count


def instance_rgb(index: int, offset: int = 0) -> tuple[int, int, int]:
    """Stable, non-sequential hues with controlled saturation/value variation."""
    step = 0.6180339887498949
    color_index = index + offset
    hue = (0.07 + color_index * step) % 1.0
    saturation = (0.82, 0.96, 0.70)[color_index % 3]
    value = (0.98, 0.82)[(color_index // 3) % 2]
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(round(channel * 255)) for channel in rgb)


def display_class_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        labels = [str(item) for item in value if item is not None]
        return Counter(labels).most_common(1)[0][0] if labels else "unknown"
    return "unknown" if value is None else str(value)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        array = np.asarray(value)
        if array.size == 1:
            return int(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        pass
    return default


def safe_name(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
    return (normalized or "unknown")[:60]


def ply_header(vertex_count: int) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by export_map_instances_cloudcompare.py\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uint instance_id\n"
        "property uchar is_background\n"
        "end_header\n"
    ).encode("ascii")


def make_vertex_rows(
    points: np.ndarray,
    color: tuple[int, int, int],
    instance_id: int,
    is_background: bool,
) -> np.ndarray:
    rows = np.empty(len(points), dtype=PLY_DTYPE)
    rows["x"] = points[:, 0]
    rows["y"] = points[:, 1]
    rows["z"] = points[:, 2]
    rows["red"], rows["green"], rows["blue"] = color
    rows["instance_id"] = instance_id
    rows["is_background"] = int(is_background)
    return rows


def write_binary_ply(path: Path, chunks: Iterable[np.ndarray], vertex_count: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(ply_header(vertex_count))
            written = 0
            for rows in chunks:
                if rows.dtype != PLY_DTYPE:
                    raise TypeError(f"Unexpected PLY dtype: {rows.dtype}")
                rows.tofile(handle)
                written += len(rows)
        if written != vertex_count:
            raise RuntimeError(f"Expected {vertex_count} vertices but wrote {written}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def export(args: argparse.Namespace) -> Path:
    if args.min_points < 1:
        raise ValueError("--min-points must be at least 1")
    if args.voxel_size < 0:
        raise ValueError("--voxel-size must be non-negative")
    source = resolve_input(args.input)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source.parent / "cloudcompare_instances"
    )
    if output.exists():
        raise FileExistsError(
            f"Output directory already exists: {output}. Choose a new --output-dir; no files were overwritten."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    objects, payload_context = load_objects(source)

    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    invalid_points_total = 0
    for map_index, obj in enumerate(objects):
        background = bool(obj.get("is_background", False))
        if args.exclude_background and background:
            skipped.append({"map_object_index": map_index, "reason": "background"})
            continue
        points, invalid_count = finite_and_voxelized(object_points(obj), args.voxel_size)
        invalid_points_total += invalid_count
        if len(points) < args.min_points:
            skipped.append(
                {
                    "map_object_index": map_index,
                    "reason": "below_min_points",
                    "finite_points": len(points),
                }
            )
            continue
        class_name = display_class_name(obj.get("class_name"))
        color = instance_rgb(map_index, args.color_offset)
        prepared.append(
            {
                "map_object_index": map_index,
                "object_uid": str(obj.get("id", "")),
                "class_name": class_name,
                "num_detections": safe_int(obj.get("num_detections")),
                "is_background": background,
                "points": points,
                "point_count": len(points),
                "rgb": color,
                "hex_color": "#" + "".join(f"{channel:02X}" for channel in color),
                "bbox_min": points.min(axis=0).tolist(),
                "bbox_max": points.max(axis=0).tolist(),
            }
        )
    if not prepared:
        raise ValueError("No objects remain after filtering")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        total_points = sum(item["point_count"] for item in prepared)
        combined_path = stage / "instances_colored.ply"
        chunks = (
            make_vertex_rows(
                item["points"],
                item["rgb"],
                item["map_object_index"],
                item["is_background"],
            )
            for item in prepared
        )
        write_binary_ply(combined_path, chunks, total_points)

        generated: list[Path] = [combined_path]
        if args.separate:
            instance_dir = stage / "instances"
            instance_dir.mkdir()
            list_lines = []
            for item in prepared:
                filename = (
                    f"instance_{item['map_object_index']:04d}_"
                    f"{safe_name(item['class_name'])}.ply"
                )
                path = instance_dir / filename
                rows = make_vertex_rows(
                    item["points"],
                    item["rgb"],
                    item["map_object_index"],
                    item["is_background"],
                )
                write_binary_ply(path, [rows], item["point_count"])
                generated.append(path)
                list_lines.append(f"instances/{filename}")
            (stage / "cloudcompare_file_list.txt").write_text(
                "\n".join(list_lines) + "\n", encoding="utf-8"
            )
            generated.append(stage / "cloudcompare_file_list.txt")

        public_rows = [
            {key: json_ready(value) for key, value in item.items() if key != "points"}
            for item in prepared
        ]
        csv_fields = [
            "map_object_index",
            "object_uid",
            "class_name",
            "num_detections",
            "is_background",
            "point_count",
            "hex_color",
            "red",
            "green",
            "blue",
            "bbox_min_x",
            "bbox_min_y",
            "bbox_min_z",
            "bbox_max_x",
            "bbox_max_y",
            "bbox_max_z",
        ]
        with (stage / "instances.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            for row in public_rows:
                writer.writerow(
                    {
                        "map_object_index": row["map_object_index"],
                        "object_uid": row["object_uid"],
                        "class_name": row["class_name"],
                        "num_detections": row["num_detections"],
                        "is_background": row["is_background"],
                        "point_count": row["point_count"],
                        "hex_color": row["hex_color"],
                        "red": row["rgb"][0],
                        "green": row["rgb"][1],
                        "blue": row["rgb"][2],
                        "bbox_min_x": row["bbox_min"][0],
                        "bbox_min_y": row["bbox_min"][1],
                        "bbox_min_z": row["bbox_min"][2],
                        "bbox_max_x": row["bbox_max"][0],
                        "bbox_max_y": row["bbox_max"][1],
                        "bbox_max_z": row["bbox_max"][2],
                    }
                )
        (stage / "instances.json").write_text(
            json.dumps(public_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        readme = """CloudCompare usage
==================

Quick overview
1. File > Open > instances_colored.ply.
2. Keep the imported RGB colors enabled. Every map object has one deterministic color.
3. Pick a point to inspect the instance_id scalar field; join it to instances.csv/json.

Toggle individual instances
1. Select all PLY files under the instances/ directory and open them together.
2. Each DB Tree entity is named instance_NNNN_<class>.ply and can be hidden independently.
3. cloudcompare_file_list.txt contains the relative file list.

Notes
- map_object_index and PLY instance_id are the same zero-based index in the source map.
- Colors encode instances, not semantic classes.
- Coordinates are preserved in the source world frame and stored as float32 for visualization.
- The source pickle was read only; this directory is a derived visualization artifact.
"""
        (stage / "README.txt").write_text(readme, encoding="utf-8")
        generated.extend([stage / "instances.csv", stage / "instances.json", stage / "README.txt"])

        header_size = len(ply_header(total_points))
        expected_size = header_size + total_points * PLY_DTYPE.itemsize
        actual_size = combined_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Combined PLY integrity failure: expected {expected_size} bytes, got {actual_size}"
            )
        unique_colors = len({tuple(item["rgb"]) for item in prepared})
        manifest = {
            "schema_version": "conceptgraphs-cloudcompare-instance-export-v1",
            "source": str(source),
            "source_sha256": sha256_file(source),
            "output_directory": str(output),
            "source_object_count": len(objects),
            "exported_object_count": len(prepared),
            "skipped_objects": skipped,
            "total_exported_points": total_points,
            "invalid_source_points_removed": invalid_points_total,
            "voxel_size_m": args.voxel_size,
            "min_points": args.min_points,
            "exclude_background": args.exclude_background,
            "color_offset": args.color_offset,
            "unique_instance_colors": unique_colors,
            "combined_ply": {
                "path": "instances_colored.ply",
                "format": "binary_little_endian PLY",
                "vertex_count": total_points,
                "vertex_stride_bytes": PLY_DTYPE.itemsize,
                "expected_bytes": expected_size,
                "actual_bytes": actual_size,
                "sha256": sha256_file(combined_path),
                "properties": list(PLY_DTYPE.names or ()),
            },
            "separate_instance_files": args.separate,
            "payload_context": payload_context,
        }
        if unique_colors != len(prepared):
            raise RuntimeError(
                f"Color collision: {len(prepared)} instances but only {unique_colors} colors"
            )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stage.replace(output)
    except Exception:
        # Do not remove a partially generated staging directory: retain it for diagnosis.
        raise RuntimeError(f"Export failed; staging files retained at: {stage}")

    print(f"Source: {source}")
    print(f"Objects: {len(prepared)}/{len(objects)} exported")
    print(f"Points: {total_points:,}")
    print(f"Distinct colors: {unique_colors}")
    print(f"Combined PLY: {output / 'instances_colored.ply'}")
    if args.separate:
        print(f"Per-instance PLY directory: {output / 'instances'}")
    print(f"Instance legend: {output / 'instances.csv'}")
    print(f"Manifest: {output / 'manifest.json'}")
    return output


def main() -> None:
    args = build_parser().parse_args()
    export(args)


if __name__ == "__main__":
    main()
