#!/usr/bin/env python3
"""Traceable post-map relationship inference and ReplicaSSG evaluation.

The pipeline never mutates a source map.  It builds an unordered candidate-pair
manifest from map geometry and co-visibility, renders evidence bound by SHA-256,
runs an OpenAI-compatible VLM with resumable per-pair outputs, exports native
ali-dev edge JSON sidecars, and evaluates directed predicates with the same
ReplicaSSG geometry matcher used by the repository evaluator.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCENE_ALIASES = {
    "room0": "room_0",
    "room1": "room_1",
    "room2": "room_2",
    "office0": "office_0",
    "office1": "office_1",
    "office2": "office_2",
    "office3": "office_3",
    "office4": "office_4",
}

PAPER_PREDICATES = (
    "on",
    "in",
    "near",
    "above",
    "under",
    "attached to",
    "with",
    "none",
)
POSITIVE_PREDICATES = frozenset(PAPER_PREDICATES) - {"none"}
COMPAT_RELATIONSHIPS = frozenset({"on top of", "under"})

SYSTEM_PROMPT = """You are a conservative indoor 3D scene-graph relationship classifier.
You receive evidence for exactly two final 3D map objects, A and B. Images use a
red overlay for A and a cyan overlay for B. A geometry diagram and numeric 3D
metadata may also be present. Infer only relationships grounded in the evidence.

Return exactly one JSON object and no prose outside it. The JSON schema is:
{
  "paper": {
    "a_to_b": [{"predicate": "...", "confidence": 0.0}, ... exactly 3],
    "b_to_a": [{"predicate": "...", "confidence": 0.0}, ... exactly 3]
  },
  "ali_dev_compatible": [
    {"source": "A|B", "relationship": "on top of|under", "target": "A|B", "confidence": 0.0}
  ],
  "same_physical_object": false,
  "evidence_quality": "good|partial|poor",
  "reason": "brief evidence-based reason"
}

For each directed paper ranking, use three distinct predicates in descending
confidence. Allowed predicates are exactly: on, in, near, above, under,
attached to, with, none.

Definitions:
- on: the source is physically supported by and in contact with the top surface
  of the target.
- in: the source is physically contained inside the target.
- near: the two objects are spatially close, without a stronger relation.
- above: the source is vertically higher than the target without requiring contact.
- under: the source is vertically lower than the target without requiring contact.
- attached to: the source is physically fixed or connected to the target.
- with: the pair forms a visually grounded semantic grouping not captured above.
- none: no listed relation is sufficiently supported.

Near and with are normally symmetric. On/in/above/under are directional. Prefer
on over above only when support/contact is visible or strongly supported by 3D
geometry. Do not invent a relation just because object names make it plausible.

The ali_dev_compatible field is a second, restricted head matching the original
ali-dev label space. Include only visually supported on-top/under statements;
otherwise return an empty list. Do not emit both inverse phrasings of the same
support fact. Never relate an object to itself."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be SCENE=/absolute/map.pkl.gz")
    scene, raw_path = value.split("=", 1)
    if scene not in SCENE_ALIASES:
        raise argparse.ArgumentTypeError(f"unsupported scene: {scene}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"map does not exist: {path}")
    return scene, path


def parse_suffix(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise argparse.ArgumentTypeError("result suffix must contain only letters, digits, '_' or '-'")
    return value


def load_map(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        raise ValueError(f"unexpected map format: {path}")
    return payload["objects"], payload


def clean_float(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def object_geometry(obj: dict[str, Any]) -> dict[str, list[float] | float]:
    bbox = np.asarray(obj["bbox_np"], dtype=np.float64)
    minimum = bbox.min(axis=0)
    maximum = bbox.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = maximum - minimum
    return {
        "minimum_xyz": [clean_float(x) for x in minimum],
        "maximum_xyz": [clean_float(x) for x in maximum],
        "center_xyz": [clean_float(x) for x in center],
        "extent_xyz": [clean_float(x) for x in extent],
        "volume": clean_float(np.prod(np.maximum(extent, 0.0))),
    }


def pair_geometry(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    amin = np.asarray(a["minimum_xyz"], dtype=np.float64)
    amax = np.asarray(a["maximum_xyz"], dtype=np.float64)
    bmin = np.asarray(b["minimum_xyz"], dtype=np.float64)
    bmax = np.asarray(b["maximum_xyz"], dtype=np.float64)
    ac = np.asarray(a["center_xyz"], dtype=np.float64)
    bc = np.asarray(b["center_xyz"], dtype=np.float64)
    gaps = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    intersections = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    union_span = np.maximum(amax, bmax) - np.minimum(amin, bmin)
    return {
        "center_delta_a_to_b_xyz": [clean_float(x) for x in bc - ac],
        "center_distance": clean_float(np.linalg.norm(bc - ac)),
        "surface_gap_xyz": [clean_float(x) for x in gaps],
        "surface_gap": clean_float(np.linalg.norm(gaps)),
        "aabb_intersection_xyz": [clean_float(x) for x in intersections],
        "aabb_intersection_volume": clean_float(np.prod(intersections)),
        "aabb_union_span_xyz": [clean_float(x) for x in union_span],
        "vertical_axis": "y",
        "vertical_center_delta_a_to_b": clean_float(bc[1] - ac[1]),
    }


def observation_index(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    paths = obj.get("color_path") or []
    boxes = obj.get("xyxy") or []
    masks = obj.get("mask") or []
    confidences = obj.get("conf") or []
    for index, path_value in enumerate(paths):
        path = str(path_value)
        if index >= len(boxes):
            continue
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        box = np.asarray(boxes[index], dtype=np.float64)
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        record = {
            "path": path,
            "box": box,
            "mask": masks[index] if index < len(masks) else None,
            "confidence": confidence,
            "area": area,
            "observation_index": index,
        }
        previous = result.get(path)
        if previous is None or (confidence, area) > (previous["confidence"], previous["area"]):
            result[path] = record
    return result


def font(size: int = 24) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, maximum: int = 1280) -> Image.Image:
    if max(image.size) <= maximum:
        return image
    scale = maximum / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


def mask_overlay(image: Image.Image, mask: Any, color: tuple[int, int, int], alpha: int = 80) -> Image.Image:
    if mask is None:
        return image
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != (image.height, image.width):
        return image
    mask_image = Image.fromarray((array.astype(bool) * alpha).astype(np.uint8), mode="L")
    layer = Image.new("RGBA", image.size, color + (0,))
    layer.putalpha(mask_image)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def crop_bounds(box_a: np.ndarray, box_b: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = float(min(box_a[0], box_b[0]))
    y1 = float(min(box_a[1], box_b[1]))
    x2 = float(max(box_a[2], box_b[2]))
    y2 = float(max(box_a[3], box_b[3]))
    padding = max(24.0, 0.18 * max(x2 - x1, y2 - y1))
    return (
        max(0, math.floor(x1 - padding)),
        max(0, math.floor(y1 - padding)),
        min(width, math.ceil(x2 + padding)),
        min(height, math.ceil(y2 + padding)),
    )


def draw_observation(
    output: Path,
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    tag_a: str,
    tag_b: str,
) -> None:
    image = Image.open(record_a["path"]).convert("RGB")
    image = mask_overlay(image, record_a["mask"], (255, 30, 30))
    image = mask_overlay(image, record_b["mask"], (0, 190, 230))
    draw = ImageDraw.Draw(image)
    box_a = record_a["box"]
    box_b = record_b["box"]
    draw.rectangle(tuple(float(x) for x in box_a), outline=(255, 0, 0), width=6)
    draw.rectangle(tuple(float(x) for x in box_b), outline=(0, 180, 220), width=6)
    label_font = font(26)
    draw.text((max(0, box_a[0]), max(0, box_a[1] - 32)), f"A: {tag_a}", fill=(255, 0, 0), font=label_font, stroke_width=2, stroke_fill="white")
    draw.text((max(0, box_b[0]), max(0, box_b[1] - 32)), f"B: {tag_b}", fill=(0, 150, 190), font=label_font, stroke_width=2, stroke_fill="white")
    bounds = crop_bounds(box_a, box_b, image.width, image.height)
    image = fit_image(image.crop(bounds))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="JPEG", quality=90, optimize=True)


def draw_single_crop(output: Path, record: dict[str, Any], label: str, color: tuple[int, int, int]) -> None:
    image = Image.open(record["path"]).convert("RGB")
    image = mask_overlay(image, record["mask"], color)
    box = record["box"]
    padding = max(24.0, 0.2 * max(box[2] - box[0], box[3] - box[1]))
    bounds = (
        max(0, math.floor(box[0] - padding)),
        max(0, math.floor(box[1] - padding)),
        min(image.width, math.ceil(box[2] + padding)),
        min(image.height, math.ceil(box[3] + padding)),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(float(x) for x in box), outline=color, width=6)
    draw.text((max(0, box[0]), max(0, box[1] - 32)), label, fill=color, font=font(26), stroke_width=2, stroke_fill="white")
    fit_image(image.crop(bounds)).save(output, format="JPEG", quality=90, optimize=True)


def draw_geometry(output: Path, a: dict[str, Any], b: dict[str, Any], pair: dict[str, Any]) -> None:
    canvas = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(30)
    body_font = font(20)
    draw.text((30, 18), "3D geometry evidence (A=red, B=cyan; y is vertical)", fill="black", font=title_font)

    def panel(origin: tuple[int, int], axes: tuple[int, int], label: str) -> None:
        ox, oy = origin
        width, height = 470, 400
        draw.rectangle((ox, oy, ox + width, oy + height), outline=(80, 80, 80), width=2)
        draw.text((ox + 10, oy + 8), label, fill="black", font=body_font)
        amin = np.asarray(a["minimum_xyz"])[list(axes)]
        amax = np.asarray(a["maximum_xyz"])[list(axes)]
        bmin = np.asarray(b["minimum_xyz"])[list(axes)]
        bmax = np.asarray(b["maximum_xyz"])[list(axes)]
        minimum = np.minimum(amin, bmin)
        maximum = np.maximum(amax, bmax)
        span = np.maximum(maximum - minimum, 1e-3)

        def rectangle(vmin: np.ndarray, vmax: np.ndarray, color: tuple[int, int, int], text: str) -> None:
            x1 = ox + 35 + (vmin[0] - minimum[0]) / span[0] * (width - 70)
            x2 = ox + 35 + (vmax[0] - minimum[0]) / span[0] * (width - 70)
            y1 = oy + 55 + (vmin[1] - minimum[1]) / span[1] * (height - 90)
            y2 = oy + 55 + (vmax[1] - minimum[1]) / span[1] * (height - 90)
            top, bottom = sorted((y1, y2))
            draw.rectangle((x1, top, x2, bottom), outline=color, width=6)
            draw.text((x1 + 6, top + 6), text, fill=color, font=body_font)

        rectangle(amin, amax, (230, 20, 20), "A")
        rectangle(bmin, bmax, (0, 150, 190), "B")

    panel((30, 80), (0, 2), "Top view: x-z")
    panel((590, 80), (0, 1), "Side view: x-y")
    lines = [
        f"A tag: {pair['object_a']['tag']}  center={a['center_xyz']}  extent={a['extent_xyz']}",
        f"B tag: {pair['object_b']['tag']}  center={b['center_xyz']}  extent={b['extent_xyz']}",
        f"center delta A->B={pair['geometry']['center_delta_a_to_b_xyz']}  surface gap={pair['geometry']['surface_gap']:.3f} m",
    ]
    for index, line in enumerate(lines):
        draw.text((35, 510 + index * 38), line, fill="black", font=body_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=92, optimize=True)


def best_single_observation(index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not index:
        return None
    return max(index.values(), key=lambda record: (record["confidence"], record["area"]))


def evidence_for_pair(
    case_dir: Path,
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    pair: dict[str, Any],
    max_observation_images: int,
) -> list[dict[str, Any]]:
    index_a = observation_index(obj_a)
    index_b = observation_index(obj_b)
    common = sorted(
        set(index_a) & set(index_b),
        key=lambda path: (
            min(index_a[path]["confidence"], index_b[path]["confidence"]),
            min(index_a[path]["area"], index_b[path]["area"]),
        ),
        reverse=True,
    )
    evidence: list[dict[str, Any]] = []
    geometry_path = case_dir / "geometry.jpg"
    draw_geometry(geometry_path, pair["object_a"]["geometry"], pair["object_b"]["geometry"], pair)
    evidence.append({"kind": "geometry", "path": str(geometry_path), "sha256": sha256_file(geometry_path)})

    for rank, path in enumerate(common[:max_observation_images], 1):
        output = case_dir / f"covisible_{rank:02d}.jpg"
        draw_observation(output, index_a[path], index_b[path], pair["object_a"]["tag"], pair["object_b"]["tag"])
        evidence.append(
            {
                "kind": "covisible",
                "path": str(output),
                "source_image": path,
                "source_sha256": sha256_file(Path(path)),
                "sha256": sha256_file(output),
                "confidence_a": clean_float(index_a[path]["confidence"]),
                "confidence_b": clean_float(index_b[path]["confidence"]),
            }
        )
    if not common:
        for label, index, color in (("A", index_a, (230, 20, 20)), ("B", index_b, (0, 150, 190))):
            record = best_single_observation(index)
            if record is None:
                continue
            output = case_dir / f"separate_{label.lower()}.jpg"
            draw_single_crop(output, record, f"{label}: {pair['object_a' if label == 'A' else 'object_b']['tag']}", color)
            evidence.append(
                {
                    "kind": f"separate_{label.lower()}",
                    "path": str(output),
                    "source_image": record["path"],
                    "source_sha256": sha256_file(Path(record["path"])),
                    "sha256": sha256_file(output),
                    "confidence": clean_float(record["confidence"]),
                }
            )
    return evidence


def node_record(index: int, obj: dict[str, Any]) -> dict[str, Any]:
    point_count = int(obj["n_points"]) if "n_points" in obj else len(np.asarray(obj.get("pcd_np", [])))
    return {
        "index": index,
        "uuid": str(obj.get("id", "")),
        "curr_obj_num": int(obj.get("curr_obj_num", index)),
        "tag": str(obj.get("class_name") or "unknown"),
        "is_background": bool(obj.get("is_background", False)),
        "num_detections": int(obj.get("num_detections", len(obj.get("color_path") or []))),
        "n_points": point_count,
        "geometry": object_geometry(obj),
    }


def build_candidates_for_scene(
    scene: str,
    map_path: Path,
    output_root: Path,
    k_nearest: int,
    max_surface_gap: float,
    max_covisible_center_distance: float,
    max_observation_images: int,
) -> dict[str, Any]:
    scene_root = output_root / scene
    scene_root.mkdir(parents=True, exist_ok=True)
    objects, _payload = load_map(map_path)
    map_sha256 = sha256_file(map_path)
    nodes = [node_record(index, obj) for index, obj in enumerate(objects)]
    active = [node["index"] for node in nodes if not node["is_background"]]
    obs_paths = [set(str(path) for path in (obj.get("color_path") or [])) for obj in objects]
    pair_info: dict[tuple[int, int], dict[str, Any]] = {}
    distances: dict[tuple[int, int], float] = {}

    for position, a_index in enumerate(active):
        for b_index in active[position + 1 :]:
            geometry = pair_geometry(nodes[a_index]["geometry"], nodes[b_index]["geometry"])
            common = obs_paths[a_index] & obs_paths[b_index]
            key = (a_index, b_index)
            distances[key] = float(geometry["surface_gap"])
            reasons: list[str] = []
            if geometry["aabb_intersection_volume"] > 0:
                reasons.append("aabb_intersection")
            if geometry["surface_gap"] <= max_surface_gap:
                reasons.append("surface_gap")
            if common and geometry["center_distance"] <= max_covisible_center_distance:
                reasons.append("covisible_proximity")
            if reasons:
                pair_info[key] = {
                    "geometry": geometry,
                    "covisible_frame_count": len(common),
                    "candidate_reasons": reasons,
                }

    for a_index in active:
        neighbors: list[tuple[float, int]] = []
        for b_index in active:
            if a_index == b_index:
                continue
            key = tuple(sorted((a_index, b_index)))
            neighbors.append((distances[key], b_index))
        for distance, b_index in sorted(neighbors)[:k_nearest]:
            if distance > max_surface_gap * 1.5:
                continue
            key = tuple(sorted((a_index, b_index)))
            if key not in pair_info:
                geometry = pair_geometry(nodes[key[0]]["geometry"], nodes[key[1]]["geometry"])
                pair_info[key] = {
                    "geometry": geometry,
                    "covisible_frame_count": len(obs_paths[key[0]] & obs_paths[key[1]]),
                    "candidate_reasons": [],
                }
            if "knn" not in pair_info[key]["candidate_reasons"]:
                pair_info[key]["candidate_reasons"].append("knn")

    candidates: list[dict[str, Any]] = []
    cases_root = scene_root / "cases"
    for serial, ((a_index, b_index), info) in enumerate(sorted(pair_info.items()), 1):
        case_id = hashlib.sha1(f"{scene}:{a_index}:{b_index}".encode("utf-8")).hexdigest()[:20]
        pair = {
            "schema_version": "1.0.0",
            "case_id": case_id,
            "scene": scene,
            "replicassg_scene": SCENE_ALIASES[scene],
            "source_map": str(map_path),
            "source_map_sha256": map_sha256,
            "object_a": nodes[a_index],
            "object_b": nodes[b_index],
            **info,
        }
        pair["evidence"] = evidence_for_pair(
            cases_root / case_id,
            objects[a_index],
            objects[b_index],
            pair,
            max_observation_images,
        )
        atomic_json(cases_root / case_id / "case.json", pair)
        candidates.append(pair)
        if serial % 50 == 0 or serial == len(pair_info):
            print(f"BUILD scene={scene} {serial}/{len(pair_info)}", flush=True)

    candidates_path = scene_root / "candidates.jsonl"
    temporary = candidates_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, candidates_path)
    summary = {
        "schema_version": "1.0.0",
        "scene": scene,
        "source_map": str(map_path),
        "source_map_sha256": map_sha256,
        "objects_total": len(objects),
        "objects_excluding_background": len(active),
        "background_objects_excluded": len(objects) - len(active),
        "candidate_pairs": len(candidates),
        "candidate_reason_counts": dict(sorted(Counter(reason for pair in candidates for reason in pair["candidate_reasons"]).items())),
        "candidates_with_covisible_evidence": sum(pair["covisible_frame_count"] > 0 for pair in candidates),
        "evidence_images": sum(len(pair["evidence"]) for pair in candidates),
    }
    atomic_json(scene_root / "build_summary.json", summary)
    return summary


def command_build(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene, path in args.input:
        summaries.append(
            build_candidates_for_scene(
                scene,
                path,
                output_root,
                args.k_nearest,
                args.max_surface_gap,
                args.max_covisible_center_distance,
                args.max_observation_images,
            )
        )
    manifest_path = output_root / "build_manifest.json"
    existing_summaries: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            existing_summaries = {
                item["scene"]: item for item in read_json(manifest_path).get("scenes", [])
            }
        except Exception:
            existing_summaries = {}
    existing_summaries.update({item["scene"]: item for item in summaries})
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "method": "ali-dev-postmap-relations-v1",
        "candidate_protocol": {
            "uses_ground_truth": False,
            "exclude_background": True,
            "k_nearest": args.k_nearest,
            "max_surface_gap": args.max_surface_gap,
            "max_covisible_center_distance": args.max_covisible_center_distance,
            "max_observation_images": args.max_observation_images,
            "candidate_union": ["aabb_intersection", "surface_gap", "covisible_proximity", "knn"],
        },
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "paper_predicates": list(PAPER_PREDICATES),
        "scenes": [existing_summaries[scene] for scene in sorted(existing_summaries)],
    }
    atomic_json(manifest_path, manifest)
    print("BUILD_SUMMARY " + json.dumps(manifest, ensure_ascii=False), flush=True)


def load_candidates(root: Path, scenes: set[str] | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/candidates.jsonl")):
        if scenes is not None and path.parent.name not in scenes:
            continue
        with path.open("r", encoding="utf-8") as stream:
            candidates.extend(json.loads(line) for line in stream if line.strip())
    return candidates


def normalize_ranking(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("ranking is not a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        predicate = str(item.get("predicate", "")).strip().lower()
        if predicate not in PAPER_PREDICATES or predicate in seen:
            continue
        confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        result.append({"predicate": predicate, "confidence": confidence})
        seen.add(predicate)
        if len(result) == 3:
            break
    for predicate in ("none", "near", "with", "above", "under", "on", "in", "attached to"):
        if len(result) == 3:
            break
        if predicate not in seen:
            result.append({"predicate": predicate, "confidence": 0.0})
            seen.add(predicate)
    return result


def normalize_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    paper = payload.get("paper")
    if not isinstance(paper, dict):
        raise ValueError("paper head missing")
    compatible: list[dict[str, Any]] = []
    raw_compatible = payload.get("ali_dev_compatible")
    if isinstance(raw_compatible, list):
        seen = set()
        for edge in raw_compatible:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip().upper()
            target = str(edge.get("target", "")).strip().upper()
            relationship = str(edge.get("relationship", "")).strip().lower()
            if source not in {"A", "B"} or target not in {"A", "B"} or source == target:
                continue
            if relationship not in COMPAT_RELATIONSHIPS:
                continue
            key = (source, relationship, target)
            if key in seen:
                continue
            seen.add(key)
            compatible.append(
                {
                    "source": source,
                    "relationship": relationship,
                    "target": target,
                    "confidence": min(1.0, max(0.0, float(edge.get("confidence", 0.0)))),
                }
            )
    quality = str(payload.get("evidence_quality", "poor")).strip().lower()
    if quality not in {"good", "partial", "poor"}:
        quality = "poor"
    return {
        "paper": {
            "a_to_b": normalize_ranking(paper.get("a_to_b")),
            "b_to_a": normalize_ranking(paper.get("b_to_a")),
        },
        "ali_dev_compatible": compatible,
        "same_physical_object": bool(payload.get("same_physical_object", False)),
        "evidence_quality": quality,
        "reason": str(payload.get("reason", ""))[:2000],
    }


def first_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    parsed, _end = decoder.raw_decode(value)
    return parsed


def data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def candidate_user_prompt(candidate: dict[str, Any]) -> str:
    compact = {
        "scene": candidate["scene"],
        "object_A": {
            "map_index": candidate["object_a"]["index"],
            "detector_tag": candidate["object_a"]["tag"],
            "num_detections": candidate["object_a"]["num_detections"],
            "geometry": candidate["object_a"]["geometry"],
        },
        "object_B": {
            "map_index": candidate["object_b"]["index"],
            "detector_tag": candidate["object_b"]["tag"],
            "num_detections": candidate["object_b"]["num_detections"],
            "geometry": candidate["object_b"]["geometry"],
        },
        "pair_geometry": candidate["geometry"],
        "covisible_frame_count": candidate["covisible_frame_count"],
        "candidate_reasons": candidate["candidate_reasons"],
        "evidence_sha256": [item["sha256"] for item in candidate["evidence"]],
    }
    return "Classify both directed relations for this pair. Detector tags are noisy hints, not truth.\n" + json.dumps(compact, ensure_ascii=False)


def infer_one(
    candidate: dict[str, Any],
    output_root: Path,
    base_url: str,
    model: str,
    api_key: str,
    key_slot: int,
    timeout: int,
    retries: int,
) -> tuple[str, str, str | None]:
    scene = candidate["scene"]
    output = output_root / scene / "predictions" / f"{candidate['case_id']}.json"
    if output.is_file():
        try:
            existing = read_json(output)
            normalize_response(existing["response"])
            return scene, "cached", None
        except Exception:
            pass
    content: list[dict[str, Any]] = [{"type": "text", "text": candidate_user_prompt(candidate)}]
    for item in candidate["evidence"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            return scene, "failed", f"evidence hash mismatch: {path.name}"
        content.append({"type": "image_url", "image_url": {"url": data_uri(path), "detail": "high"}})
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1600,
        "stream": False,
        "store": False,
    }
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    last_error: str | None = None
    started_all = time.monotonic()
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ali-dev-relations/1.0",
            },
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                envelope = json.loads(response.read())
            choices = envelope.get("choices") or []
            raw_text = choices[0].get("message", {}).get("content") if choices else None
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError("empty response content")
            normalized = normalize_response(first_json(raw_text))
            record = {
                "schema_version": "1.0.0",
                "case_id": candidate["case_id"],
                "scene": scene,
                "source_map": candidate["source_map"],
                "source_map_sha256": candidate["source_map_sha256"],
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "evidence_sha256": [item["sha256"] for item in candidate["evidence"]],
                "model_requested": model,
                "model_returned": str(envelope.get("model") or model),
                "response_id": envelope.get("id"),
                "key_slot": key_slot,
                "attempt": attempt,
                "elapsed_seconds": clean_float(time.monotonic() - started, 3),
                "total_elapsed_seconds": clean_float(time.monotonic() - started_all, 3),
                "usage": envelope.get("usage") or {},
                "response": normalized,
                "completed_at": utc_now(),
            }
            atomic_json(output, record)
            return scene, "completed", None
        except urllib.error.HTTPError as exc:
            safe_body = exc.read(800).decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {safe_body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    return scene, "failed", (last_error or "unknown inference failure")[:1200]


def key_names() -> list[tuple[str, str]]:
    keys = []
    for index in range(1, 33):
        name = f"REL_API_KEY_{index}"
        value = os.environ.get(name)
        if value:
            keys.append((name, value))
    if not keys and os.environ.get("VLM_API_KEY"):
        keys.append(("VLM_API_KEY", os.environ["VLM_API_KEY"]))
    return keys


def command_infer(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    scenes = set(args.scene) if args.scene else None
    candidates = load_candidates(root, scenes)
    keys = key_names()
    if not keys:
        raise SystemExit("no API key environment variables; set REL_API_KEY_1..N")
    workers = min(args.workers or len(keys), len(keys))
    keys = keys[:workers]
    counts: Counter[str] = Counter()
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, str]] = []
    lock = threading.Lock()
    started = time.monotonic()

    def run_shard(slot: int, api_key: str, shard: list[dict[str, Any]]) -> None:
        for candidate in shard:
            scene, status, error = infer_one(
                candidate,
                root,
                args.base_url,
                args.model,
                api_key,
                slot,
                args.timeout,
                args.retries,
            )
            with lock:
                counts[status] += 1
                scene_counts[scene][status] += 1
                done = sum(counts.values())
                if error:
                    errors.append({"scene": scene, "case_id": candidate["case_id"], "error": error})
                detail = f" error={error[:300]}" if error else ""
                print(
                    f"INFER {done}/{len(candidates)} scene={scene} status={status} slot={slot}{detail}",
                    flush=True,
                )

    shards = [candidates[index::workers] for index in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_shard, slot + 1, keys[slot][1], shards[slot]) for slot in range(workers)]
        for future in as_completed(futures):
            future.result()
    manifest = {
        "schema_version": "1.0.0",
        "completed_at": utc_now(),
        "base_url": args.base_url,
        "model_requested": args.model,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "key_slots": len(keys),
        "key_environment_names": [name for name, _key in keys],
        "api_keys_serialized": False,
        "candidates": len(candidates),
        "outcomes": dict(counts),
        "outcomes_by_scene": {scene: dict(values) for scene, values in sorted(scene_counts.items())},
        "errors": errors,
        "elapsed_seconds": clean_float(time.monotonic() - started, 3),
    }
    atomic_json(root / "inference_manifest.json", manifest)
    print("INFER_SUMMARY " + json.dumps(manifest, ensure_ascii=False), flush=True)
    if counts["failed"]:
        raise SystemExit(1)


def prediction_records(root: Path, scene: str) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((root / scene / "predictions").glob("*.json")):
        record = read_json(path)
        record["response"] = normalize_response(record["response"])
        result[record["case_id"]] = record
    return result


def export_edges_for_scene(root: Path, scene: str, minimum_confidence: float) -> dict[str, Any]:
    candidates = [item for item in load_candidates(root, {scene})]
    predictions = prediction_records(root, scene)
    paper_edges: dict[tuple[int, int, str], dict[str, Any]] = {}
    compat_edges: dict[tuple[int, int, str], dict[str, Any]] = {}
    ranked = []

    def add_edge(
        target: dict[tuple[int, int, str], dict[str, Any]],
        source_node: dict[str, Any],
        object_node: dict[str, Any],
        relationship: str,
        confidence: float,
        case_id: str,
        head: str,
    ) -> None:
        if source_node["index"] == object_node["index"]:
            return
        key = (source_node["index"], object_node["index"], relationship)
        record = {
            "edge_description": f"{source_node['tag']} {relationship} {object_node['tag']}",
            "num_detections": 1,
            "object_1_id": source_node["curr_obj_num"],
            "object_1_index": source_node["index"],
            "object_1_uuid": source_node["uuid"],
            "object_1_tag": source_node["tag"],
            "object_2_id": object_node["curr_obj_num"],
            "object_2_index": object_node["index"],
            "object_2_uuid": object_node["uuid"],
            "object_2_tag": object_node["tag"],
            "relationship": relationship,
            "confidence": confidence,
            "source_case_ids": [case_id],
            "head": head,
        }
        previous = target.get(key)
        if previous is None or confidence > previous["confidence"]:
            target[key] = record
        elif case_id not in previous["source_case_ids"]:
            previous["source_case_ids"].append(case_id)
            previous["num_detections"] += 1

    for candidate in candidates:
        prediction = predictions.get(candidate["case_id"])
        if prediction is None:
            continue
        response = prediction["response"]
        a = candidate["object_a"]
        b = candidate["object_b"]
        ranked.append(
            {
                "case_id": candidate["case_id"],
                "object_a_index": a["index"],
                "object_b_index": b["index"],
                "paper": response["paper"],
                "ali_dev_compatible": response["ali_dev_compatible"],
                "evidence_quality": response["evidence_quality"],
                "same_physical_object": response["same_physical_object"],
            }
        )
        for direction, source, target in (("a_to_b", a, b), ("b_to_a", b, a)):
            top = response["paper"][direction][0]
            if top["predicate"] in POSITIVE_PREDICATES and top["confidence"] >= minimum_confidence:
                add_edge(paper_edges, source, target, top["predicate"], top["confidence"], candidate["case_id"], "paper")
        for edge in response["ali_dev_compatible"]:
            if edge["confidence"] < minimum_confidence:
                continue
            source = a if edge["source"] == "A" else b
            target = a if edge["target"] == "A" else b
            add_edge(compat_edges, source, target, edge["relationship"], edge["confidence"], candidate["case_id"], "ali_dev_compatible")

    export_root = root / scene / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    paper_payload = {}
    for index, record in enumerate(sorted(paper_edges.values(), key=lambda item: (item["object_1_index"], item["object_2_index"], item["relationship"]))):
        record = dict(record)
        record["edge_id"] = index
        paper_payload[f"edge_{index}"] = record
    compat_payload = {}
    for index, record in enumerate(sorted(compat_edges.values(), key=lambda item: (item["object_1_index"], item["object_2_index"], item["relationship"]))):
        record = dict(record)
        record["edge_id"] = index
        compat_payload[f"edge_{index}"] = record
    atomic_json(export_root / "edge_json_paper_aligned.json", paper_payload)
    atomic_json(export_root / "edge_json_ali_dev_compatible.json", compat_payload)
    atomic_json(export_root / "ranked_relations.json", ranked)
    manifest = {
        "scene": scene,
        "minimum_confidence": minimum_confidence,
        "candidate_pairs": len(candidates),
        "completed_predictions": len(predictions),
        "paper_positive_directed_edges": len(paper_payload),
        "ali_dev_compatible_directed_edges": len(compat_payload),
        "files": {},
    }
    for path in sorted(export_root.glob("*.json")):
        if path.name != "export_manifest.json":
            manifest["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    atomic_json(export_root / "export_manifest.json", manifest)
    return manifest


def command_export(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    scenes = args.scene or sorted(path.name for path in root.iterdir() if (path / "candidates.jsonl").is_file())
    manifests = [export_edges_for_scene(root, scene, args.minimum_confidence) for scene in scenes]
    atomic_json(root / "export_summary.json", {"created_at": utc_now(), "scenes": manifests})
    print("EXPORT_SUMMARY " + json.dumps(manifests, ensure_ascii=False), flush=True)


def load_ground_truth(scene: str, annotations_dir: Path) -> tuple[list[int], list[tuple[int, int, str]]]:
    mapping = read_json(annotations_dir / "replica_to_visual_genome.json")
    object_scans = read_json(annotations_dir / "objects.json")["scans"]
    relation_scans = read_json(annotations_dir / "relationships.json")["scans"]
    object_scan = next(item for item in object_scans if item["scan"] == scene)
    relation_scan = next(item for item in relation_scans if item["scan"] == scene)
    valid_classes = set(mapping["VisualGenome_list"])
    replica_to_vg = mapping["Replica2VisualGenome"]
    valid_ids = {
        int(obj["id"])
        for obj in object_scan["objects"]
        if obj["label"] in replica_to_vg and replica_to_vg[obj["label"]] in valid_classes
    }
    relations = [
        (int(source), int(target), str(predicate))
        for source, target, _predicate_id, predicate in relation_scan["relationships"]
        if int(source) in valid_ids and int(target) in valid_ids
    ]
    return sorted(valid_ids), relations


def match_instances(
    gt_ply: Path,
    gt_ids: list[int],
    pred_objects: list[dict[str, Any]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from plyfile import PlyData
    from scipy.spatial import KDTree

    mesh = PlyData.read(str(gt_ply))
    vertices = mesh["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    point_object_ids = np.asarray(vertices["objectId"])
    valid_mask = np.isin(point_object_ids, np.asarray(gt_ids))
    points = points[valid_mask]
    point_object_ids = point_object_ids[valid_mask]
    object_id_to_index = {object_id: index for index, object_id in enumerate(gt_ids)}
    overlap_count = np.zeros((len(gt_ids), len(pred_objects)), dtype=np.int64)
    tree = KDTree(points)
    for pred_index, pred_object in enumerate(pred_objects):
        segment = np.asarray(pred_object["pcd_np"])
        if not len(segment):
            continue
        _distances, indices = tree.query(segment, distance_upper_bound=threshold)
        matched_indices = indices[indices != tree.n]
        matched_gt = np.fromiter(
            (object_id_to_index[int(point_object_ids[index])] for index in matched_indices),
            dtype=np.int64,
        )
        if matched_gt.size:
            overlap_count[:, pred_index] = np.bincount(matched_gt, minlength=len(gt_ids))
        sorted_gt = np.flip(np.argsort(overlap_count[:, pred_index], kind="stable"))
        best = int(sorted_gt[0])
        best_fraction = overlap_count[best, pred_index] / len(segment)
        second_fraction = 0.0
        if len(sorted_gt) > 1:
            second = int(sorted_gt[1])
            second_fraction = overlap_count[second, pred_index] / len(segment)
        ambiguity = second_fraction / best_fraction if best_fraction > 0 else np.inf
        if best_fraction < 0.5 or ambiguity > 0.75:
            overlap_count[:, pred_index] = 0
        else:
            overlap_count[np.arange(len(gt_ids)) != best, pred_index] = 0
    gt_to_pred = np.full(len(gt_ids), -1, dtype=np.int64)
    matched_points = np.zeros(len(gt_ids), dtype=np.int64)
    for gt_index in range(len(gt_ids)):
        pred_index = int(np.argmax(overlap_count[gt_index]))
        if overlap_count[gt_index, pred_index] > 0:
            gt_to_pred[gt_index] = pred_index
            matched_points[gt_index] = overlap_count[gt_index, pred_index]
    return gt_to_pred, matched_points, overlap_count


@dataclass
class HeadPredictions:
    top1: dict[tuple[int, int], str]
    top3: dict[tuple[int, int], list[str]]
    positive: set[tuple[int, int, str]]


def collect_head_predictions(
    candidates: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    head: str,
    minimum_confidence: float,
) -> HeadPredictions:
    top1: dict[tuple[int, int], str] = {}
    top3: dict[tuple[int, int], list[str]] = {}
    positive: set[tuple[int, int, str]] = set()
    for candidate in candidates:
        record = predictions.get(candidate["case_id"])
        if record is None:
            continue
        response = record["response"]
        a = int(candidate["object_a"]["index"])
        b = int(candidate["object_b"]["index"])
        if head == "paper":
            for direction, source, target in (("a_to_b", a, b), ("b_to_a", b, a)):
                ranking = response["paper"][direction]
                top1[(source, target)] = ranking[0]["predicate"]
                top3[(source, target)] = [item["predicate"] for item in ranking]
                if ranking[0]["predicate"] in POSITIVE_PREDICATES and ranking[0]["confidence"] >= minimum_confidence:
                    positive.add((source, target, ranking[0]["predicate"]))
        elif head == "ali_dev_compatible":
            for edge in response["ali_dev_compatible"]:
                if edge["confidence"] < minimum_confidence:
                    continue
                source = a if edge["source"] == "A" else b
                target = a if edge["target"] == "A" else b
                predicate = "on" if edge["relationship"] == "on top of" else "under"
                top1[(source, target)] = predicate
                top3[(source, target)] = [predicate]
                positive.add((source, target, predicate))
        else:
            raise ValueError(head)
    return HeadPredictions(top1, top3, positive)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def threshold_tag(value: float) -> str:
    return (f"{value:.3f}".rstrip("0").rstrip(".") or "0").replace(".", "p")


def evaluate_scene(
    root: Path,
    scene: str,
    dataset_root: Path,
    annotations_dir: Path,
    minimum_confidence: float,
    overlap_threshold: float,
    map_override: Path | None = None,
    result_suffix: str = "",
) -> dict[str, Any]:
    candidates = load_candidates(root, {scene})
    if not candidates:
        raise ValueError(f"no candidates for {scene}")
    inference_map_path = Path(candidates[0]["source_map"])
    map_path = map_override.resolve() if map_override is not None else inference_map_path
    objects, _payload = load_map(map_path)
    gt_scene = SCENE_ALIASES[scene]
    gt_ids, gt_relations = load_ground_truth(gt_scene, annotations_dir)
    gt_ply = dataset_root / "data" / gt_scene / "labels.instances.annotated.v2.ply"
    gt_to_pred, matched_points, _overlap = match_instances(gt_ply, gt_ids, objects, overlap_threshold)
    gt_index = {object_id: index for index, object_id in enumerate(gt_ids)}
    predictions = prediction_records(root, scene)
    candidate_pairs = {
        tuple(sorted((int(item["object_a"]["index"]), int(item["object_b"]["index"]))))
        for item in candidates
    }
    heads = {
        name: collect_head_predictions(candidates, predictions, name, minimum_confidence)
        for name in ("paper", "ali_dev_compatible")
    }
    head_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    head_summary: dict[str, dict[str, Any]] = {}

    mapped_gt_triples = set()
    endpoint_matched = candidate_covered = 0
    for source_id, target_id, predicate in gt_relations:
        source_pred = int(gt_to_pred[gt_index[source_id]])
        target_pred = int(gt_to_pred[gt_index[target_id]])
        endpoints = source_pred >= 0 and target_pred >= 0
        covered = endpoints and tuple(sorted((source_pred, target_pred))) in candidate_pairs
        endpoint_matched += int(endpoints)
        candidate_covered += int(covered)
        if endpoints:
            mapped_gt_triples.add((source_pred, target_pred, predicate))
        for head_name, head in heads.items():
            ranking = head.top3.get((source_pred, target_pred), []) if endpoints else []
            row = {
                "gt_source_id": source_id,
                "gt_target_id": target_id,
                "predicate": predicate,
                "source_pred_index": source_pred,
                "target_pred_index": target_pred,
                "endpoint_matched": endpoints,
                "candidate_covered": covered,
                "predicted_top1": ranking[0] if ranking else None,
                "predicted_top3": ranking,
                "hit_at_1": bool(ranking) and ranking[0] == predicate,
                "hit_at_3": predicate in ranking[:3],
                "export_hit": (source_pred, target_pred, predicate) in head.positive,
            }
            head_rows[head_name].append(row)

    for head_name, rows in head_rows.items():
        head = heads[head_name]
        per_class: dict[str, dict[str, Any]] = {}
        for predicate in sorted(set(row["predicate"] for row in rows)):
            subset = [row for row in rows if row["predicate"] == predicate]
            per_class[predicate] = {
                "count": len(subset),
                "endpoint_matched": sum(row["endpoint_matched"] for row in subset),
                "candidate_covered": sum(row["candidate_covered"] for row in subset),
                "hits_at_1": sum(row["hit_at_1"] for row in subset),
                "hits_at_3": sum(row["hit_at_3"] for row in subset),
                "export_hits": sum(row["export_hit"] for row in subset),
                "recall_at_1": safe_ratio(sum(row["hit_at_1"] for row in subset), len(subset)),
                "recall_at_3": safe_ratio(sum(row["hit_at_3"] for row in subset), len(subset)),
                "export_recall": safe_ratio(sum(row["export_hit"] for row in subset), len(subset)),
            }
        tp = len(head.positive & mapped_gt_triples)
        fp = len(head.positive - mapped_gt_triples)
        fn = len(mapped_gt_triples - head.positive)
        precision = safe_ratio(tp, tp + fp)
        recall_mapped = safe_ratio(tp, tp + fn)
        f1 = None if precision is None or recall_mapped is None or precision + recall_mapped == 0 else 2 * precision * recall_mapped / (precision + recall_mapped)
        head_summary[head_name] = {
            "gt_relations": len(rows),
            "hits_at_1": sum(row["hit_at_1"] for row in rows),
            "hits_at_3": sum(row["hit_at_3"] for row in rows),
            "recall_at_1": safe_ratio(sum(row["hit_at_1"] for row in rows), len(rows)),
            "recall_at_3": safe_ratio(sum(row["hit_at_3"] for row in rows), len(rows)),
            "mean_recall_at_1": float(np.mean([value["recall_at_1"] for value in per_class.values()])),
            "mean_recall_at_3": float(np.mean([value["recall_at_3"] for value in per_class.values()])),
            "conditional_recall_at_1_given_endpoint_match": safe_ratio(sum(row["hit_at_1"] for row in rows), endpoint_matched),
            "conditional_recall_at_1_given_candidate": safe_ratio(sum(row["hit_at_1"] for row in rows), candidate_covered),
            "export_hits": sum(row["export_hit"] for row in rows),
            "export_recall_all_gt": safe_ratio(sum(row["export_hit"] for row in rows), len(rows)),
            "export_conditional_recall_given_endpoint_match": safe_ratio(sum(row["export_hit"] for row in rows), endpoint_matched),
            "export_conditional_recall_given_candidate": safe_ratio(sum(row["export_hit"] for row in rows), candidate_covered),
            "closed_world_precision": precision,
            "closed_world_recall_on_mapped_gt": recall_mapped,
            "closed_world_f1": f1,
            "true_positive_edges": tp,
            "false_positive_edges": fp,
            "false_negative_mapped_gt_edges": fn,
            "positive_predicted_directed_edges": len(head.positive),
            "per_class": per_class,
        }

    result = {
        "schema_version": "1.0.0",
        "scene": scene,
        "replicassg_scene": gt_scene,
        "source_map": str(map_path),
        "source_map_sha256": sha256_file(map_path),
        "relationship_inference_source_map": str(inference_map_path),
        "relationship_inference_source_map_sha256": candidates[0]["source_map_sha256"],
        "protocol": {
            "geometry_match": "FROSS ReplicaSSG one-way KD-tree; 0.1m, >=50% best overlap, second/best <=0.75",
            "overlap_threshold": overlap_threshold,
            "predicate_directional": True,
            "predicate_ignores_object_class": True,
            "minimum_confidence": minimum_confidence,
            "map_override_used": map_override is not None,
        },
        "integrity": {
            "gt_objects": len(gt_ids),
            "predicted_objects": len(objects),
            "geometry_matched_gt_objects": int(np.sum(gt_to_pred >= 0)),
            "gt_relations": len(gt_relations),
            "relations_with_both_endpoints_matched": endpoint_matched,
            "relations_with_candidate_pair": candidate_covered,
            "candidate_oracle_recall": safe_ratio(candidate_covered, len(gt_relations)),
            "candidate_recall_given_endpoint_match": safe_ratio(candidate_covered, endpoint_matched),
            "candidate_pairs": len(candidate_pairs),
            "completed_predictions": len(predictions),
        },
        "heads": head_summary,
    }
    eval_root = root / scene / "evaluation"
    tag = threshold_tag(minimum_confidence) + (f"_{result_suffix}" if result_suffix else "")
    atomic_json(eval_root / f"results_threshold_{tag}.json", result)
    for head_name, rows in head_rows.items():
        path = eval_root / f"predicate_matches_{head_name}_threshold_{tag}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows({**row, "predicted_top3": "|".join(row["predicted_top3"])} for row in rows)
    return result


def pool_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    pooled: dict[str, Any] = {
        "scenes": [result["scene"] for result in results],
        "gt_relations": sum(result["integrity"]["gt_relations"] for result in results),
        "relations_with_both_endpoints_matched": sum(result["integrity"]["relations_with_both_endpoints_matched"] for result in results),
        "relations_with_candidate_pair": sum(result["integrity"]["relations_with_candidate_pair"] for result in results),
        "candidate_pairs": sum(result["integrity"]["candidate_pairs"] for result in results),
        "heads": {},
    }
    pooled["candidate_oracle_recall"] = safe_ratio(pooled["relations_with_candidate_pair"], pooled["gt_relations"])
    pooled["candidate_recall_given_endpoint_match"] = safe_ratio(pooled["relations_with_candidate_pair"], pooled["relations_with_both_endpoints_matched"])
    for head_name in ("paper", "ali_dev_compatible"):
        class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for result in results:
            for predicate, metrics in result["heads"][head_name]["per_class"].items():
                for key in ("count", "endpoint_matched", "candidate_covered", "hits_at_1", "hits_at_3", "export_hits"):
                    class_counts[predicate][key] += int(metrics[key])
        per_class = {}
        for predicate, metrics in sorted(class_counts.items()):
            per_class[predicate] = {
                **metrics,
                "recall_at_1": safe_ratio(metrics["hits_at_1"], metrics["count"]),
                "recall_at_3": safe_ratio(metrics["hits_at_3"], metrics["count"]),
                "export_recall": safe_ratio(metrics["export_hits"], metrics["count"]),
            }
        totals = {
            key: sum(result["heads"][head_name][key] for result in results)
            for key in (
                "hits_at_1",
                "hits_at_3",
                "export_hits",
                "true_positive_edges",
                "false_positive_edges",
                "false_negative_mapped_gt_edges",
                "positive_predicted_directed_edges",
            )
        }
        precision = safe_ratio(totals["true_positive_edges"], totals["true_positive_edges"] + totals["false_positive_edges"])
        recall_mapped = safe_ratio(totals["true_positive_edges"], totals["true_positive_edges"] + totals["false_negative_mapped_gt_edges"])
        f1 = None if precision is None or recall_mapped is None or precision + recall_mapped == 0 else 2 * precision * recall_mapped / (precision + recall_mapped)
        pooled["heads"][head_name] = {
            **totals,
            "recall_at_1": safe_ratio(totals["hits_at_1"], pooled["gt_relations"]),
            "recall_at_3": safe_ratio(totals["hits_at_3"], pooled["gt_relations"]),
            "mean_recall_at_1": float(np.mean([value["recall_at_1"] for value in per_class.values()])),
            "mean_recall_at_3": float(np.mean([value["recall_at_3"] for value in per_class.values()])),
            "conditional_recall_at_1_given_endpoint_match": safe_ratio(totals["hits_at_1"], pooled["relations_with_both_endpoints_matched"]),
            "conditional_recall_at_1_given_candidate": safe_ratio(totals["hits_at_1"], pooled["relations_with_candidate_pair"]),
            "export_recall_all_gt": safe_ratio(totals["export_hits"], pooled["gt_relations"]),
            "export_conditional_recall_given_endpoint_match": safe_ratio(totals["export_hits"], pooled["relations_with_both_endpoints_matched"]),
            "export_conditional_recall_given_candidate": safe_ratio(totals["export_hits"], pooled["relations_with_candidate_pair"]),
            "closed_world_precision": precision,
            "closed_world_recall_on_mapped_gt": recall_mapped,
            "closed_world_f1": f1,
            "per_class": per_class,
        }
    return pooled


def command_evaluate(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    scenes = args.scene or sorted(path.name for path in root.iterdir() if (path / "candidates.jsonl").is_file())
    map_overrides = dict(args.map_override or [])
    results = []
    for scene in scenes:
        print(f"EVALUATE scene={scene}", flush=True)
        results.append(
            evaluate_scene(
                root,
                scene,
                args.dataset_root.resolve(),
                args.annotations_dir.resolve(),
                args.minimum_confidence,
                args.overlap_threshold,
                map_overrides.get(scene),
                args.result_suffix,
            )
        )
    summary = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "per_scene": results,
        "pooled_all": pool_results(results),
    }
    dev = [result for result in results if result["scene"] in {"room0", "office0"}]
    held_out = [result for result in results if result["scene"] not in {"room0", "office0"}]
    if dev:
        summary["pooled_development_room0_office0"] = pool_results(dev)
    if held_out:
        summary["pooled_held_out_six_scenes"] = pool_results(held_out)
    tag = threshold_tag(args.minimum_confidence) + (f"_{args.result_suffix}" if args.result_suffix else "")
    atomic_json(root / f"evaluation_summary_threshold_{tag}.json", summary)
    print("EVALUATION_SUMMARY " + json.dumps(summary["pooled_all"], ensure_ascii=False), flush=True)


def command_audit(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    candidates = load_candidates(root)
    issues: list[str] = []
    usage = Counter()
    elapsed = []
    model_returned = Counter()
    quality = Counter()
    key_slots = Counter()
    candidates_by_scene = Counter()
    predictions_by_scene = Counter()
    same_physical_object = 0
    predictions_total = 0
    map_hash_cache: dict[str, str] = {}
    seen_cases: set[tuple[str, str]] = set()
    seen_pairs: set[tuple[str, int, int]] = set()
    expected_prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    for candidate in candidates:
        scene = str(candidate["scene"])
        case_id = str(candidate["case_id"])
        a_index = int(candidate["object_a"]["index"])
        b_index = int(candidate["object_b"]["index"])
        candidates_by_scene[scene] += 1
        case_key = (scene, case_id)
        pair_key = (scene, *sorted((a_index, b_index)))
        if case_key in seen_cases:
            issues.append(f"duplicate case id: {scene}/{case_id}")
        if pair_key in seen_pairs:
            issues.append(f"duplicate candidate pair: {scene}/{pair_key[1]}/{pair_key[2]}")
        if a_index == b_index:
            issues.append(f"self candidate pair: {scene}/{case_id}")
        seen_cases.add(case_key)
        seen_pairs.add(pair_key)
        case_path = root / candidate["scene"] / "cases" / candidate["case_id"] / "case.json"
        source_map = candidate["source_map"]
        if source_map not in map_hash_cache:
            map_hash_cache[source_map] = sha256_file(Path(source_map))
        if not case_path.is_file() or map_hash_cache[source_map] != candidate["source_map_sha256"]:
            issues.append(f"case/map integrity failed: {candidate['scene']}/{candidate['case_id']}")
        for item in candidate["evidence"]:
            if not Path(item["path"]).is_file() or sha256_file(Path(item["path"])) != item["sha256"]:
                issues.append(f"evidence integrity failed: {candidate['scene']}/{candidate['case_id']}/{Path(item['path']).name}")
        prediction_path = root / candidate["scene"] / "predictions" / f"{candidate['case_id']}.json"
        if not prediction_path.is_file():
            issues.append(f"missing prediction: {candidate['scene']}/{candidate['case_id']}")
            continue
        try:
            record = read_json(prediction_path)
            response = normalize_response(record["response"])
            if record.get("case_id") != case_id or record.get("scene") != scene:
                issues.append(f"prediction identity mismatch: {scene}/{case_id}")
            if record.get("source_map_sha256") != candidate["source_map_sha256"]:
                issues.append(f"prediction map binding mismatch: {scene}/{case_id}")
            if record.get("prompt_sha256") != expected_prompt_hash:
                issues.append(f"prediction prompt binding mismatch: {scene}/{case_id}")
            if record["evidence_sha256"] != [item["sha256"] for item in candidate["evidence"]]:
                issues.append(f"prediction evidence binding mismatch: {candidate['scene']}/{candidate['case_id']}")
            predictions_total += 1
            predictions_by_scene[scene] += 1
            same_physical_object += int(response["same_physical_object"])
            model_returned[str(record.get("model_returned"))] += 1
            key_slots[int(record.get("key_slot", 0))] += 1
            elapsed.append(float(record.get("elapsed_seconds", 0.0)))
            quality[response["evidence_quality"]] += 1
            for key, value in (record.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value
        except Exception as exc:
            issues.append(f"invalid prediction: {candidate['scene']}/{candidate['case_id']}: {exc}")
    report = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "PASS" if not issues else "FAIL",
        "candidates": len(candidates),
        "valid_predictions": predictions_total,
        "candidates_by_scene": dict(sorted(candidates_by_scene.items())),
        "valid_predictions_by_scene": dict(sorted(predictions_by_scene.items())),
        "unique_case_ids": len(seen_cases),
        "unique_unordered_pairs": len(seen_pairs),
        "self_candidate_pairs": sum(1 for scene, a, b in seen_pairs if a == b),
        "issues": issues,
        "model_returned_counts": dict(model_returned),
        "key_slot_counts": {str(key): value for key, value in sorted(key_slots.items())},
        "evidence_quality_counts": dict(quality),
        "same_physical_object_count": same_physical_object,
        "usage": dict(usage),
        "latency_seconds": {
            "mean": clean_float(np.mean(elapsed), 3) if elapsed else None,
            "median": clean_float(np.median(elapsed), 3) if elapsed else None,
            "p95": clean_float(np.percentile(elapsed, 95), 3) if elapsed else None,
            "max": clean_float(max(elapsed), 3) if elapsed else None,
        },
    }
    atomic_json(root / "integrity_audit.json", report)
    print("AUDIT_SUMMARY " + json.dumps(report, ensure_ascii=False), flush=True)
    if issues:
        raise SystemExit(1)


def command_sync(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    destinations = [path.resolve() for path in args.destination]
    copied = []
    for scene_root in sorted(path for path in root.iterdir() if (path / "exports").is_dir()):
        scene = scene_root.name
        source_map = read_json(scene_root / "build_summary.json")["source_map"]
        source_hash = read_json(scene_root / "build_summary.json")["source_map_sha256"]
        for destination in destinations:
            target = destination / scene
            target.mkdir(parents=True, exist_ok=True)
            for source in sorted((scene_root / "exports").glob("*.json")):
                target_file = target / source.name
                shutil.copy2(source, target_file)
                copied.append({"source": str(source), "target": str(target_file), "sha256": sha256_file(target_file)})
            manifest = {
                "schema_version": "1.0.0",
                "method": "ali-dev-postmap-relations-v1",
                "scene": scene,
                "source_map": source_map,
                "source_map_sha256": source_hash,
                "relationship_result_root": str(root),
                "source_maps_mutated": False,
                "synced_at": utc_now(),
            }
            atomic_json(target / "sync_manifest.json", manifest)
    summary = {"created_at": utc_now(), "destinations": [str(path) for path in destinations], "files_copied": copied}
    atomic_json(root / "sync_summary.json", summary)
    print("SYNC_SUMMARY " + json.dumps({"destinations": summary["destinations"], "file_count": len(copied)}, ensure_ascii=False), flush=True)


def candidate_index_set(
    objects: list[dict[str, Any]],
    k_nearest: int,
    max_surface_gap: float,
    max_covisible_center_distance: float,
) -> set[tuple[int, int]]:
    nodes = [node_record(index, obj) for index, obj in enumerate(objects)]
    active = [node["index"] for node in nodes if not node["is_background"]]
    paths = [set(str(path) for path in (obj.get("color_path") or [])) for obj in objects]
    selected: set[tuple[int, int]] = set()
    distances: dict[tuple[int, int], float] = {}
    for position, a_index in enumerate(active):
        for b_index in active[position + 1 :]:
            key = (a_index, b_index)
            geometry = pair_geometry(nodes[a_index]["geometry"], nodes[b_index]["geometry"])
            distances[key] = float(geometry["surface_gap"])
            if (
                geometry["aabb_intersection_volume"] > 0
                or geometry["surface_gap"] <= max_surface_gap
                or (paths[a_index] & paths[b_index] and geometry["center_distance"] <= max_covisible_center_distance)
            ):
                selected.add(key)
    for a_index in active:
        neighbors = sorted(
            (distances[tuple(sorted((a_index, b_index)))], b_index)
            for b_index in active
            if b_index != a_index
        )
        for distance, b_index in neighbors[:k_nearest]:
            if distance <= max_surface_gap * 1.5:
                selected.add(tuple(sorted((a_index, b_index))))
    return selected


def remap_edge_payload(payload: dict[str, Any], target_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, raw in payload.items():
        edge = dict(raw)
        first = target_nodes[int(edge["object_1_index"])]
        second = target_nodes[int(edge["object_2_index"])]
        edge.update(
            {
                "edge_description": f"{first['tag']} {edge['relationship']} {second['tag']}",
                "object_1_id": first["curr_obj_num"],
                "object_1_uuid": first["uuid"],
                "object_1_tag": first["tag"],
                "object_2_id": second["curr_obj_num"],
                "object_2_uuid": second["uuid"],
                "object_2_tag": second["tag"],
            }
        )
        result[key] = edge
    return result


def command_remap(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    scene = args.scene
    destination = args.destination.resolve()
    candidates = load_candidates(root, {scene})
    if not candidates:
        raise ValueError(f"no source candidates for {scene}")
    source_objects, _source_payload = load_map(Path(candidates[0]["source_map"]))
    target_objects, _target_payload = load_map(args.target_map.resolve())
    source_nodes = [node_record(index, obj) for index, obj in enumerate(source_objects)]
    target_nodes = [node_record(index, obj) for index, obj in enumerate(target_objects)]
    count_equal = len(source_nodes) == len(target_nodes)
    comparable = min(len(source_nodes), len(target_nodes))
    class_matches = sum(source_nodes[index]["tag"] == target_nodes[index]["tag"] for index in range(comparable))
    background_matches = sum(source_nodes[index]["is_background"] == target_nodes[index]["is_background"] for index in range(comparable))
    bbox_deltas = []
    clip_cosines = []
    observation_jaccards = []
    observation_frame_jaccards = []
    for index in range(comparable):
        source_bbox = np.asarray(source_objects[index]["bbox_np"], dtype=np.float64)
        target_bbox = np.asarray(target_objects[index]["bbox_np"], dtype=np.float64)
        if source_bbox.shape == target_bbox.shape:
            bbox_deltas.append(float(np.max(np.abs(source_bbox - target_bbox))))
        source_clip = np.asarray(source_objects[index]["clip_ft"], dtype=np.float64)
        target_clip = np.asarray(target_objects[index]["clip_ft"], dtype=np.float64)
        denominator = np.linalg.norm(source_clip) * np.linalg.norm(target_clip)
        clip_cosines.append(float(np.dot(source_clip, target_clip) / denominator) if denominator else 0.0)
        source_paths = set(str(path) for path in (source_objects[index].get("color_path") or []))
        target_paths = set(str(path) for path in (target_objects[index].get("color_path") or []))
        union = source_paths | target_paths
        observation_jaccards.append(len(source_paths & target_paths) / len(union) if union else 1.0)
        source_frames = {Path(path).name for path in source_paths}
        target_frames = {Path(path).name for path in target_paths}
        frame_union = source_frames | target_frames
        observation_frame_jaccards.append(
            len(source_frames & target_frames) / len(frame_union) if frame_union else 1.0
        )

    protocol = read_json(root / "build_manifest.json")["candidate_protocol"]
    source_pairs = {
        tuple(sorted((int(item["object_a"]["index"]), int(item["object_b"]["index"]))))
        for item in candidates
    }
    target_pairs = candidate_index_set(
        target_objects,
        int(protocol["k_nearest"]),
        float(protocol["max_surface_gap"]),
        float(protocol["max_covisible_center_distance"]),
    )
    pair_union = source_pairs | target_pairs
    pair_jaccard = len(source_pairs & target_pairs) / len(pair_union) if pair_union else 1.0
    parity = {
        "schema_version": "1.0.0",
        "scene": scene,
        "source_map": candidates[0]["source_map"],
        "source_map_sha256": candidates[0]["source_map_sha256"],
        "target_map": str(args.target_map.resolve()),
        "target_map_sha256": sha256_file(args.target_map.resolve()),
        "object_count_source": len(source_nodes),
        "object_count_target": len(target_nodes),
        "object_count_equal": count_equal,
        "class_matches_by_index": class_matches,
        "background_flag_matches_by_index": background_matches,
        "bbox_max_abs_delta": max(bbox_deltas) if bbox_deltas else None,
        "bbox_mean_of_index_max_abs_delta": float(np.mean(bbox_deltas)) if bbox_deltas else None,
        "clip_cosine_min": min(clip_cosines) if clip_cosines else None,
        "clip_cosine_mean": float(np.mean(clip_cosines)) if clip_cosines else None,
        "observation_path_jaccard_min": min(observation_jaccards) if observation_jaccards else None,
        "observation_path_jaccard_mean": float(np.mean(observation_jaccards)) if observation_jaccards else None,
        "observation_frame_name_jaccard_min": min(observation_frame_jaccards) if observation_frame_jaccards else None,
        "observation_frame_name_jaccard_mean": float(np.mean(observation_frame_jaccards)) if observation_frame_jaccards else None,
        "candidate_pairs_source": len(source_pairs),
        "candidate_pairs_target": len(target_pairs),
        "candidate_pair_jaccard": pair_jaccard,
    }
    parity["decision"] = "PASS" if (
        count_equal
        and class_matches == comparable
        and background_matches == comparable
        and (parity["bbox_max_abs_delta"] or 0.0) <= args.max_bbox_delta
        and (parity["clip_cosine_min"] or 0.0) >= args.min_clip_cosine
        and (parity["observation_frame_name_jaccard_min"] or 0.0) >= args.min_observation_frame_jaccard
        and pair_jaccard >= args.min_candidate_jaccard
    ) else "FAIL"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "map_parity_report.json", parity)
    if parity["decision"] != "PASS" and not args.allow_failed_parity:
        raise SystemExit("target map parity failed; refusing to remap edges")
    source_export = root / scene / "exports"
    for name in ("edge_json_paper_aligned.json", "edge_json_ali_dev_compatible.json"):
        atomic_json(destination / name, remap_edge_payload(read_json(source_export / name), target_nodes))
    shutil.copy2(source_export / "ranked_relations.json", destination / "ranked_relations.json")
    files = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(destination.glob("*.json"))
        if path.name != "remap_manifest.json"
    }
    manifest = {
        "schema_version": "1.0.0",
        "method": "ali-dev-postmap-relations-v1",
        "scene": scene,
        "relationship_predictions_reused": True,
        "node_fields_remapped_to_target_map": True,
        "source_map_mutated": False,
        "target_map_mutated": False,
        "parity_decision": parity["decision"],
        "files": files,
        "created_at": utc_now(),
    }
    atomic_json(destination / "remap_manifest.json", manifest)
    print("REMAP_SUMMARY " + json.dumps({"parity": parity, "destination": str(destination), "files": files}, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--input", type=parse_input, action="append", required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--k-nearest", type=int, default=8)
    build.add_argument("--max-surface-gap", type=float, default=1.75)
    build.add_argument("--max-covisible-center-distance", type=float, default=3.0)
    build.add_argument("--max-observation-images", type=int, default=3)
    build.set_defaults(func=command_build)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--scene", action="append", choices=sorted(SCENE_ALIASES))
    infer.add_argument("--base-url", default="https://api.pinaic.com/v1")
    infer.add_argument("--model", default="gpt-5.6-sol")
    infer.add_argument("--workers", type=int, default=0)
    infer.add_argument("--timeout", type=int, default=240)
    infer.add_argument("--retries", type=int, default=4)
    infer.set_defaults(func=command_infer)

    export = subparsers.add_parser("export")
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--scene", action="append", choices=sorted(SCENE_ALIASES))
    export.add_argument("--minimum-confidence", type=float, default=0.5)
    export.set_defaults(func=command_export)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--scene", action="append", choices=sorted(SCENE_ALIASES))
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--annotations-dir", type=Path, required=True)
    evaluate.add_argument("--minimum-confidence", type=float, default=0.5)
    evaluate.add_argument("--overlap-threshold", type=float, default=0.1)
    evaluate.add_argument("--map-override", type=parse_input, action="append")
    evaluate.add_argument("--result-suffix", type=parse_suffix, default="")
    evaluate.set_defaults(func=command_evaluate)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output-root", type=Path, required=True)
    audit.set_defaults(func=command_audit)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--output-root", type=Path, required=True)
    sync.add_argument("--destination", type=Path, action="append", required=True)
    sync.set_defaults(func=command_sync)

    remap = subparsers.add_parser("remap")
    remap.add_argument("--output-root", type=Path, required=True)
    remap.add_argument("--scene", required=True, choices=sorted(SCENE_ALIASES))
    remap.add_argument("--target-map", type=Path, required=True)
    remap.add_argument("--destination", type=Path, required=True)
    remap.add_argument("--max-bbox-delta", type=float, default=0.02)
    remap.add_argument("--min-clip-cosine", type=float, default=0.99)
    remap.add_argument("--min-observation-frame-jaccard", type=float, default=0.99)
    remap.add_argument("--min-candidate-jaccard", type=float, default=0.99)
    remap.add_argument("--allow-failed-parity", action="store_true")
    remap.set_defaults(func=command_remap)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
