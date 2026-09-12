from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from plyfile import PlyData


PROTOCOL = "replica_official_habitat_semantic_mesh_v1"
OFFICIAL_SOURCE = (
    "https://github.com/facebookresearch/Replica-Dataset/releases/tag/v1.0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_class_name(value: Any) -> str:
    name = " ".join(str(value or "").strip().split())
    if not name:
        raise ValueError("official object has an empty class_name")
    return name


def load_official_objects(info_path: Path) -> dict[int, str]:
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("info_semantic.json must contain an objects list")
    result: dict[int, str] = {}
    for row in objects:
        if not isinstance(row, dict) or "id" not in row or "class_name" not in row:
            raise ValueError("every official object must contain id and class_name")
        object_id = int(row["id"])
        class_name = _normalise_class_name(row["class_name"])
        previous = result.setdefault(object_id, class_name)
        if previous != class_name:
            raise ValueError(
                f"official object id {object_id} maps to both {previous!r} "
                f"and {class_name!r}"
            )
    if not result:
        raise ValueError("info_semantic.json contains no objects")
    return result


def _property_name(names: Iterable[str], choices: tuple[str, ...], kind: str) -> str:
    available = tuple(str(item) for item in names)
    for choice in choices:
        if choice in available:
            return choice
    raise ValueError(f"cannot find {kind} property; available={available}")


def _face_arrays(mesh_path: Path) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    mesh = PlyData.read(str(mesh_path))
    if "vertex" not in mesh or "face" not in mesh:
        raise ValueError("mesh_semantic.ply must contain vertex and face elements")
    vertex = mesh["vertex"].data
    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32, copy=False
    )
    if not np.isfinite(vertices).all():
        raise ValueError("official semantic mesh contains non-finite vertices")
    face = mesh["face"].data
    vertex_property = _property_name(
        face.dtype.names or (), ("vertex_indices", "vertex_index"), "face vertices"
    )
    object_property = _property_name(
        face.dtype.names or (), ("object_id", "objectid", "segment_id"), "object id"
    )
    polygons = [np.asarray(item, dtype=np.int64) for item in face[vertex_property]]
    object_ids = np.asarray(face[object_property], dtype=np.int64)
    if len(polygons) != len(object_ids):
        raise ValueError("face polygons and object ids do not align")
    for face_index, polygon in enumerate(polygons):
        if len(polygon) < 3:
            raise ValueError(f"face {face_index} has fewer than three vertices")
        if ((polygon < 0) | (polygon >= len(vertices))).any():
            raise ValueError(f"face {face_index} contains an invalid vertex index")
    return np.ascontiguousarray(vertices), polygons, object_ids


def assign_vertex_instances(
    vertex_count: int,
    polygons: Iterable[np.ndarray],
    face_object_ids: np.ndarray,
    valid_object_ids: frozenset[int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign each mesh vertex the majority object id among adjacent faces."""

    instance = np.full(vertex_count, -1, dtype=np.int64)
    polygons = list(polygons)
    valid_face = np.isin(face_object_ids, np.fromiter(valid_object_ids, dtype=np.int64))
    unknown_face_count = int((~valid_face).sum())
    valid_polygons = [polygon for polygon, keep in zip(polygons, valid_face) if keep]
    valid_face_ids = face_object_ids[valid_face]
    if valid_polygons:
        lengths = np.fromiter((len(item) for item in valid_polygons), dtype=np.int64)
        corner_vertex = np.concatenate(valid_polygons)
        corner_object = np.repeat(valid_face_ids, lengths)

        order = np.lexsort((corner_object, corner_vertex))
        sorted_vertex = corner_vertex[order]
        sorted_object = corner_object[order]
        new_pair = np.ones(len(order), dtype=bool)
        new_pair[1:] = (sorted_vertex[1:] != sorted_vertex[:-1]) | (
            sorted_object[1:] != sorted_object[:-1]
        )
        pair_start = np.flatnonzero(new_pair)
        pair_count = np.diff(np.append(pair_start, len(order)))
        pair_vertex = sorted_vertex[pair_start]
        pair_object = sorted_object[pair_start]

        ranking = np.lexsort((pair_object, -pair_count, pair_vertex))
        ranked_vertex = pair_vertex[ranking]
        ranked_object = pair_object[ranking]
        ranked_count = pair_count[ranking]
        first = np.ones(len(ranking), dtype=bool)
        first[1:] = ranked_vertex[1:] != ranked_vertex[:-1]
        instance[ranked_vertex[first]] = ranked_object[first]

        top_positions = np.flatnonzero(first)
        has_second = (top_positions + 1 < len(ranking)) & (
            ranked_vertex[np.minimum(top_positions + 1, len(ranking) - 1)]
            == ranked_vertex[top_positions]
        )
        top_with_second = top_positions[has_second]
        tied_vertex_count = int(
            np.sum(
                ranked_count[top_with_second]
                == ranked_count[top_with_second + 1]
            )
        )
    else:
        tied_vertex_count = 0
    return instance, {
        "face_count": int(len(face_object_ids)),
        "face_with_unknown_object_id_count": int(unknown_face_count),
        "labeled_vertex_count": int((instance >= 0).sum()),
        "unlabeled_vertex_count": int((instance < 0).sum()),
        "boundary_tie_vertex_count": int(tied_vertex_count),
    }


def build_reference(scene_dir: Path, scene: str, output: Path) -> Path:
    scene_dir = scene_dir.resolve()
    mesh_path = scene_dir / "habitat" / "mesh_semantic.ply"
    info_path = scene_dir / "habitat" / "info_semantic.json"
    for path in (mesh_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    object_to_class = load_official_objects(info_path)
    vertices, polygons, face_object_ids = _face_arrays(mesh_path)
    used_object_ids = frozenset(int(item) for item in np.unique(face_object_ids))
    unknown_object_ids = sorted(used_object_ids - object_to_class.keys())
    instance, diagnostics = assign_vertex_instances(
        len(vertices), polygons, face_object_ids, frozenset(object_to_class)
    )

    class_names = sorted(set(object_to_class.values()), key=lambda item: item.casefold())
    class_to_semantic = {name: index + 1 for index, name in enumerate(class_names)}
    instance_to_semantic = {
        object_id: class_to_semantic[class_name]
        for object_id, class_name in sorted(object_to_class.items())
    }
    semantic = np.zeros(len(vertices), dtype=np.int64)
    for object_id, semantic_id in instance_to_semantic.items():
        semantic[instance == object_id] = semantic_id

    staging = output.with_name(output.name + f".building-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    try:
        reference_path = staging / "reference.npz"
        np.savez_compressed(
            reference_path,
            xyz=vertices,
            semantic=semantic,
            instance=instance,
        )
        manifest = {
            "scene": scene,
            "protocol": PROTOCOL,
            "source": "Replica Dataset v1 official Habitat export",
            "source_release": OFFICIAL_SOURCE,
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "source_scene_dir": str(scene_dir),
            "mesh_semantic_path": str(mesh_path),
            "mesh_semantic_sha256": sha256_file(mesh_path),
            "info_semantic_path": str(info_path),
            "info_semantic_sha256": sha256_file(info_path),
            "point_assignment": (
                "official mesh vertices; majority object_id over adjacent faces; "
                "ties choose smaller native object_id"
            ),
            "instance_id_encoding": "replica_native_object_id",
            "unlabeled_instance_id": -1,
            "undeclared_face_object_id_policy": (
                "object_id values absent from info_semantic.json are unlabeled, "
                "regardless of sign"
            ),
            "semantic_classes": class_names,
            "instance_id_to_semantic_id": {
                str(key): value for key, value in instance_to_semantic.items()
            },
            "instance_id_to_class_name": {
                str(key): value for key, value in sorted(object_to_class.items())
            },
            "point_count": int(len(vertices)),
            "semantic_class_count": len(class_names),
            "declared_instance_count": len(object_to_class),
            "used_instance_count": int(len(np.unique(instance[instance >= 0]))),
            "unlabeled_face_object_ids": unknown_object_ids,
            "diagnostics": diagnostics,
            "reference_sha256": sha256_file(reference_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
        return output
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build observation-GT reference from official Replica Habitat assets"
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--scene", required=True, help="evaluation scene name, e.g. office2")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = build_reference(args.scene_dir, args.scene, args.out.resolve())
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
