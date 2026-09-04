"""Blocking, pre-fusion association arbitration for online mapping.

The gate receives the exact pre-fusion map state H, writes an auditable evidence
bundle, and returns one match index per detection. ``off`` and ``audit`` never
alter baseline decisions.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import shutil
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import httpx
import numpy as np
import openai
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


SCHEMA_VERSION = "blocking-association-gate-v1.3"
VALID_MODES = {"off", "audit", "oracle", "vlm", "human"}
VALID_SCOPES = {"create_only", "both"}
ALIASES = ("A", "B", "C")
DISCARD_MATCH_INDEX = -1


class HumanInputUnavailableError(RuntimeError):
    """Raised when blocking human mode cannot read an interactive decision."""

TARGET_READING_POLICY = """
MANDATORY TARGET-LOCK RULE

In every RGB image, the RED mask defines the physical target to reason about.

The target is the physical content actually covered by the red-masked pixels.
It is NOT:
- the largest object in the crop,
- the most visually salient object,
- the object that occupies most of the surrounding scene,
- or an object that merely contains, supports, overlaps, or lies next to the red mask.

A physical object is eligible to be treated as the target only if the red mask actually covers that object.

Examples of the rule:
- If a small red mask marks a pillow on a large sofa, the target is the pillow, not the sofa.
- If a red mask marks one chair beside a large table, the target is the masked chair, not the table.
- If the red mask shows only part of an object, reason from that masked part plus the other provided evidence; never replace it with a larger nearby object because it appears more complete.

Use unmasked pixels only to understand spatial context such as:
"next to", "on", "behind", or "surrounded by".
Do NOT use unmasked objects as evidence for the target's category, shape, texture, material, color, or identity.

The red color itself is only an annotation overlay and is not the object's real color.

Before comparing CURRENT with any candidate, first internally lock onto the physical entity covered by CURRENT's red mask.
Do the same independently for every red-masked historical candidate view.

Once a target is locked, never switch attention to a larger neighboring object later in the reasoning.

If the red mask itself covers multiple physical objects or is too ambiguous to determine what physical entity it represents, choose UNCERTAIN rather than substituting a nearby salient object.
"""


EVIDENCE_FORMAT_POLICY = """
Evidence format:

CURRENT CONTEXT:
The yellow CURRENT box only helps locate the observation.
Inside it, the RED mask defines the CURRENT target.
Everything outside the red mask is context only.

CURRENT CROP:
A closer view of the same CURRENT target.
The crop boundary does not define the object.
Only the red mask defines the target; unmasked pixels inside the crop remain context.

Each CANDIDATE card contains two sections.

TOP — historical RGB evidence:
H1/H2/H3 are frozen historical observations of that candidate.
In every history tile, only the RED-masked physical entity is the candidate target.
Unmasked nearby objects are context only.

BOTTOM — 3D evidence:
A CURRENT-versus-candidate point-cloud comparison.

Dark padding, borders, labels, panel layout, and annotation colors are presentation aids rather than physical scene evidence.
"""


POINT_CLOUD_READING_POLICY = """
How to read the 3D evidence:

XY, XZ, and YZ are three orthographic projections of the same 3D scene geometry.

MAGENTA = CURRENT observation point cloud.
CYAN = the candidate object's historical point cloud before CURRENT is fused.

CURRENT and every candidate already use the same online world coordinate system.
Do not mentally translate, recenter, rotate, or align one cloud to make it fit another.

All candidate cards in the same event use the same world bounds and metric scale, so visible offsets can be compared directly.

Use 3D evidence to answer whether the two observations occupy a physically compatible location and geometry.

Evidence supporting SAME identity:
- substantial spatial agreement in multiple informative projections,
- compatible geometry,
- and no stable physical separation that requires moving one object to align it with the other.

Evidence supporting DIFFERENT identity:
- persistent spatial separation,
- incompatible geometry,
- or nearby/parallel surfaces that remain distinct across multiple projections.

Do not dismiss a stable spatial offset as merely a viewpoint change: viewpoint changes do not move an object in the shared world coordinate system.

However:
- partial non-overlap alone does not prove DIFFERENT because observations may be incomplete due to viewpoint, occlusion, or partial masks;
- overlap alone does not prove SAME because nearby objects, mask leakage, or polluted object history may overlap.

Never decide from only one projection, a few stray points, centroid proximity alone, or overall size alone.

Interpret 3D jointly with the red-masked RGB evidence.
If reliable 2D and 3D evidence materially disagree, choose UNCERTAIN rather than forcing a match.
"""


INSTANCE_IDENTITY_POLICY = """
The task is physical INSTANCE matching, not category recognition.

The question is:
"Are these observations of the same individual physical object in the world?"

Two objects can have the same category, color, shape, and generic appearance while still being different physical instances.

Category agreement is weak evidence.
Category disagreement is also not decisive because detector labels may vary across observations.

For 2D identity evidence, compare only the red-masked targets and look for a consistent combination of:
- object geometry and proportions,
- arrangement of visible parts,
- distinctive texture or pattern,
- material or appearance details,
- distinctive marks,
- and compatible scene placement.

Generic features shared by the whole category are not sufficient.

Repeated or adjacent objects require stronger evidence.
For visually similar instances, 3D world position is especially important.

Select a candidate only when the combined red-mask RGB evidence and 3D evidence support that CURRENT and that candidate are the same individual physical object.

Choose NEW only when CURRENT is a coherent physical object and the evidence supports that none of the listed candidates is the same instance.

Choose UNCERTAIN when the available evidence cannot reliably distinguish among repeated or adjacent instances, when important evidence conflicts, or when CURRENT itself cannot be grounded reliably.
"""


BASE_SYSTEM_PROMPT = """
You are an object-identity adjudicator for an online 3D map.

All evidence is frozen before CURRENT is fused into the map.

Your single task is to determine whether CURRENT is the same individual physical object as one listed candidate, or whether it is a NEW object.

{target_reading_policy}

{evidence_format_policy}

{point_cloud_reading_policy}

{instance_identity_policy}

MANDATORY DECISION PROCEDURE

Step 1 — Lock CURRENT target.
Identify only the physical entity actually covered by CURRENT's red mask.
Do not reason about candidate identity before this target is fixed.

Step 2 — Lock candidate targets.
For each candidate, identify only the physical entity covered by the red masks in its historical RGB views.

Step 3 — Test every candidate independently.
For each candidate ask:

"Does the red-masked CURRENT target and this red-masked candidate history represent the same individual physical object?"

Use:
(a) masked 2D appearance and structure,
and
(b) shared-world 3D position and geometry.

Do not compare the surrounding large objects as if they were the targets.

Step 4 — Compare hypotheses.
A candidate should win only if the combined evidence supports SAME identity more strongly than the alternatives.

Step 5 — Final decision.
Choose:
- a candidate code when the evidence supports the same physical instance;
- NEW when CURRENT is different from every candidate;
- UNCERTAIN when the evidence is insufficient, ambiguous, or conflicting.

Important prohibitions:
- Never redefine a target based on visual saliency or object size.
- Never treat the whole crop as the target.
- Never use a nearby larger object as a substitute for the masked target.
- Never select a candidate merely because it shows the same category or the same surrounding room.
- Never use candidate order or the mapper's likely original decision as evidence.

Briefly assess every candidate, then return only the required JSON object.
"""

BASE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT.format(
    target_reading_policy=TARGET_READING_POLICY,
    evidence_format_policy=EVIDENCE_FORMAT_POLICY,
    point_cloud_reading_policy=POINT_CLOUD_READING_POLICY,
    instance_identity_policy=INSTANCE_IDENTITY_POLICY,
)

ASSOCIATION_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT
CREATE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return str(value)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl_append(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_plain(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _mask_array(value: Any, shape: Tuple[int, int]) -> np.ndarray:
    mask = _as_numpy(value).astype(bool, copy=False)
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def _bbox_from_mask(mask: np.ndarray, fallback: Optional[Sequence[float]] = None) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs):
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if fallback is not None and len(fallback) >= 4:
        return tuple(int(round(float(item))) for item in fallback[:4])  # type: ignore[return-value]
    return 0, 0, mask.shape[1], mask.shape[0]


def _annotated_crop(
    image_rgb: np.ndarray,
    mask_value: Any,
    bbox_value: Optional[Sequence[float]],
    label: str,
) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask = _mask_array(mask_value, image.shape[:2])
    overlay = image.copy()
    overlay[mask] = np.array([255, 54, 54], dtype=np.uint8)
    image = np.where(mask[..., None], (0.60 * image + 0.40 * overlay).astype(np.uint8), image)
    x1, y1, x2, y2 = _bbox_from_mask(mask, bbox_value)
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    pad_x, pad_y = int(round(width * 0.18)), int(round(height * 0.18))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = image.copy()
    border = max(2, int(round(min(crop.shape[:2]) * 0.01)))
    header_height = 30 if label else 0
    crop = cv2.copyMakeBorder(crop, header_height, border, border, border, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    if label:
        cv2.putText(crop, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    return crop


def _annotated_context(image_rgb: np.ndarray, mask_value: Any, bbox_value: Optional[Sequence[float]]) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask = _mask_array(mask_value, image.shape[:2])
    overlay = image.copy()
    overlay[mask] = np.array([255, 54, 54], dtype=np.uint8)
    image = np.where(mask[..., None], (0.62 * image + 0.38 * overlay).astype(np.uint8), image)
    x1, y1, x2, y2 = _bbox_from_mask(mask, bbox_value)
    cv2.rectangle(image, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (255, 255, 0), 3)
    cv2.putText(image, "CURRENT", (max(4, x1), max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
    return image


def _point_array(value: Any) -> np.ndarray:
    """Extract finite XYZ points from a live or serialized map object."""
    if isinstance(value, Mapping):
        if value.get("pcd") is not None:
            value = value.get("pcd")
        elif value.get("pcd_np") is not None:
            value = value.get("pcd_np")
        else:
            return np.empty((0, 3), dtype=np.float64)
    if hasattr(value, "points"):
        value = value.points
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    points = points[:, :3]
    return points[np.isfinite(points).all(axis=1)]


def _sample_points(points: np.ndarray, limit: int, seed_text: str) -> np.ndarray:
    if len(points) <= int(limit):
        return points
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return points[np.sort(rng.choice(len(points), size=int(limit), replace=False))]


def _fit_rgb_panel(image_rgb: np.ndarray, width: int, height: int, background: int = 20) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    out_width = max(1, int(round(image.shape[1] * scale)))
    out_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (out_width, out_height), interpolation=interpolation)
    x0, y0 = (width - out_width) // 2, (height - out_height) // 2
    canvas[y0:y0 + out_height, x0:x0 + out_width] = resized
    return canvas


def _shared_projection_ranges(
    point_sets: Sequence[np.ndarray],
) -> List[Tuple[float, float, float]]:
    """Return event-level (center_a, center_b, square_extent) per XY/XZ/YZ view."""
    finite = [np.asarray(points, dtype=np.float64) for points in point_sets if len(points)]
    all_points = np.concatenate(finite, axis=0) if finite else np.empty((0, 3), dtype=np.float64)
    ranges: List[Tuple[float, float, float]] = []
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        if not len(all_points):
            ranges.append((0.0, 0.0, 1.0))
            continue
        low_a, high_a = np.quantile(all_points[:, axis_a], [0.005, 0.995])
        low_b, high_b = np.quantile(all_points[:, axis_b], [0.005, 0.995])
        center_a, center_b = (low_a + high_a) / 2.0, (low_b + high_b) / 2.0
        extent = max(float(high_a - low_a), float(high_b - low_b), 1e-3) * 1.12
        ranges.append((float(center_a), float(center_b), float(extent)))
    return ranges


def _render_point_cloud_comparison(
    current_points: np.ndarray,
    candidate_points: np.ndarray,
    alias: str,
    projection_ranges: Optional[Sequence[Tuple[float, float, float]]] = None,
    width: int = 1024,
    height: int = 456,
) -> np.ndarray:
    """Render XY/XZ/YZ without recentering; supplied ranges are shared by A/B/C."""
    canvas = np.full((height, width, 3), (248, 249, 252), dtype=np.uint8)
    dark, muted, border = (30, 38, 57), (70, 78, 96), (194, 203, 218)
    current_color, candidate_color = (220, 62, 108), (20, 162, 184)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"3D  CURRENT vs CANDIDATE {alias}", (12, 25), font, 0.55, dark, 1, cv2.LINE_AA)
    legend_x = max(430, width - 420)
    cv2.circle(canvas, (legend_x, 20), 5, current_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "CURRENT", (legend_x + 12, 25), font, 0.43, muted, 1, cv2.LINE_AA)
    cv2.circle(canvas, (legend_x + 145, 20), 5, candidate_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"CANDIDATE {alias} @ H", (legend_x + 157, 25), font, 0.43, muted, 1, cv2.LINE_AA)

    projections = ((0, 1, "XY (top)"), (0, 2, "XZ (front)"), (1, 2, "YZ (side)"))
    if projection_ranges is None:
        projection_ranges = _shared_projection_ranges((current_points, candidate_points))
    if len(projection_ranges) != len(projections):
        raise ValueError("projection_ranges must contain XY, XZ, and YZ ranges")
    gap, margin, top, bottom = 10, 10, 38, height - 10
    panel_width = (width - 2 * margin - 2 * gap) // 3

    for panel_index, (axis_a, axis_b, name) in enumerate(projections):
        left = margin + panel_index * (panel_width + gap)
        right = left + panel_width
        cv2.rectangle(canvas, (left, top), (right, bottom), border, 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (left + 8, top + 20), font, 0.46, muted, 1, cv2.LINE_AA)
        if not len(current_points) and not len(candidate_points):
            cv2.putText(canvas, "3D UNAVAILABLE", (left + 58, (top + bottom) // 2), font, 0.48, muted, 1, cv2.LINE_AA)
            continue

        center_a, center_b, metric_extent = projection_ranges[panel_index]
        inner_left, inner_right = left + 14, right - 14
        inner_top, inner_bottom = top + 32, bottom - 14
        pixel_scale = min((inner_right - inner_left) / metric_extent, (inner_bottom - inner_top) / metric_extent)
        center_x, center_y = (inner_left + inner_right) / 2.0, (inner_top + inner_bottom) / 2.0

        def project(points: np.ndarray) -> np.ndarray:
            if not len(points):
                return np.empty((0, 2), dtype=np.int32)
            x = np.rint(center_x + (points[:, axis_a] - center_a) * pixel_scale)
            y = np.rint(center_y - (points[:, axis_b] - center_b) * pixel_scale)
            pixels = np.stack((x, y), axis=1).astype(np.int32)
            inside = (
                (pixels[:, 0] >= inner_left) & (pixels[:, 0] <= inner_right)
                & (pixels[:, 1] >= inner_top) & (pixels[:, 1] <= inner_bottom)
            )
            return pixels[inside]

        for x, y in project(candidate_points):
            cv2.circle(canvas, (int(x), int(y)), 1, candidate_color, -1, cv2.LINE_AA)
        for x, y in project(current_points):
            cv2.circle(canvas, (int(x), int(y)), 2, current_color, -1, cv2.LINE_AA)
        for points, color in ((candidate_points, candidate_color), (current_points, current_color)):
            if len(points):
                center_pixel = project(np.median(points, axis=0, keepdims=True))
                if len(center_pixel):
                    cv2.drawMarker(canvas, tuple(center_pixel[0]), color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    return canvas


def _candidate_evidence_composite(
    history_rgbs: Sequence[np.ndarray],
    history_labels: Sequence[str],
    current_points: np.ndarray,
    candidate_points: np.ndarray,
    alias: str,
    projection_ranges: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """Build one 1024px card: three 2D histories above, shared-scale 3D below."""
    width, height, top_height, card_header = 1024, 1024, 563, 44
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(
        canvas, f"CANDIDATE {alias}   |   2D HISTORY (RED MASK)", (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA,
    )
    margin, gap = 12, 8
    tile_width = (width - 2 * margin - 2 * gap) // 3
    tile_top, tile_bottom = card_header + 4, top_height - 8
    for index in range(3):
        left = margin + index * (tile_width + gap)
        right = left + tile_width
        cv2.rectangle(canvas, (left, tile_top), (right, tile_bottom), (92, 101, 118), 1, cv2.LINE_AA)
        label = history_labels[index] if index < len(history_labels) else f"H{index + 1}  NO HISTORY"
        cv2.putText(canvas, label, (left + 8, tile_top + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 247, 252), 1, cv2.LINE_AA)
        if index < len(history_rgbs):
            panel = _fit_rgb_panel(history_rgbs[index], tile_width - 4, tile_bottom - tile_top - 32)
            canvas[tile_top + 29:tile_bottom - 3, left + 2:right - 2] = panel
    cv2.line(canvas, (0, top_height), (width - 1, top_height), (255, 255, 255), 3, cv2.LINE_AA)
    canvas[top_height + 5:] = _render_point_cloud_comparison(
        current_points, candidate_points, alias, projection_ranges,
        width=width, height=height - top_height - 5,
    )
    return canvas


def _letterbox_api_raster(image_rgb: np.ndarray, canvas_size: int = 1024) -> np.ndarray:
    """Render real RGB evidence onto a square raster without distorting it."""
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError(f"invalid RGB evidence shape: {image.shape}")
    height, width = image.shape[:2]
    usable = canvas_size - 32
    scale = min(usable / width, usable / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((canvas_size, canvas_size, 3), 20, dtype=np.uint8)
    offset_x = (canvas_size - resized_width) // 2
    offset_y = (canvas_size - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    # Every file retained in an event directory is the exact, fully rendered
    # raster submitted to the VLM.  A 1024px letterbox keeps narrow masks
    # readable and avoids tiny-image rejection upstream.
    rendered = _letterbox_api_raster(image_rgb)
    bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"failed to write image: {path}")


def _image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    # The upstream VLM accepts raster images only.  Evidence is always rendered
    # by OpenCV as JPEG; validate both the container signature and decodability
    # before constructing the request so SVG (or a mislabeled file) can never
    # reach the API as ``data:image/jpeg``.
    if not raw.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"VLM evidence is not a JPEG bitstream: {path}")
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"VLM evidence JPEG cannot be decoded: {path}")
    if min(decoded.shape[:2]) < 512:
        raise ValueError(f"VLM evidence raster is smaller than 512px: {path} {decoded.shape[1]}x{decoded.shape[0]}")
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _image_media_descriptor(path: Path) -> dict:
    raw = path.read_bytes()
    data_url = _image_data_url(path)
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return {
        "path": path.name,
        "mime_type": "image/jpeg",
        "data_url_prefix": "data:image/jpeg;base64,",
        "jpeg_magic_hex": raw[:3].hex(),
        "encoded_bytes": len(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
        "svg_forbidden": True,
        "validated_data_url_length": len(data_url),
    }


def _extract_raw_index(obs_uid: Optional[str]) -> Optional[int]:
    match = re.search(r"_r(\d+)$", str(obs_uid)) if obs_uid else None
    return int(match.group(1)) if match else None


def compute_trigger(
    scores: Sequence[float],
    baseline_match: Optional[int],
    sim_threshold: float,
    margin_threshold: float,
    threshold_distance: float,
    threshold_scope: str = "create_only",
) -> Optional[Dict[str, float]]:
    """Return trigger metadata or None. This is independent of mapping classes."""
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    ordered = np.sort(finite)[::-1]
    top1 = float(ordered[0])
    distance = abs(top1 - float(sim_threshold))
    if baseline_match is None:
        if distance < float(threshold_distance):
            return {"kind": "create", "threshold_distance": distance, "top1": top1}
        return None
    if len(ordered) >= 2:
        margin = float(ordered[0] - ordered[1])
        if margin < float(margin_threshold):
            return {"kind": "association", "margin": margin, "top1": top1, "top2": float(ordered[1])}
    if threshold_scope == "both" and distance < float(threshold_distance):
        return {"kind": "association", "threshold_distance": distance, "top1": top1}
    return None


def compute_support_drop(
    current: float, history: Sequence[float], *, min_history: int = 3,
    reference_min: float = 0.75, drop_threshold: float = 0.20,
) -> dict:
    """Use only already-committed, non-triggered ATTACH support samples."""
    reference = float(np.median(history)) if len(history) else None
    drop = reference - float(current) if reference is not None else None
    return {
        "current": float(current), "reference": reference, "drop": drop,
        "history_count": len(history), "history_supports": list(history),
        "triggered": bool(
            len(history) >= min_history and reference >= reference_min
            and drop >= drop_threshold - 1e-7
        ),
    }


def route_choice(
    choice: str,
    aliases_to_indices: Mapping[str, int],
    baseline_match: Optional[int],
) -> Tuple[Optional[int], str]:
    normalized = str(choice).strip().upper()
    if normalized in aliases_to_indices:
        return int(aliases_to_indices[normalized]), "model_candidate"
    if normalized == "NEW":
        return None, "model_new"
    if normalized == "DISCARD":
        return DISCARD_MATCH_INDEX, "model_discard_observation"
    return baseline_match, "fallback_baseline"


def same_frame_mask_iou(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Tuple[float, Optional[dict]]:
    """Return the maximum visible-mask IoU on a frame shared by two map objects.

    Masks from different camera frames are not comparable.  Requiring a shared
    source frame makes this a conservative duplicate-hypothesis test rather
    than a general object-similarity heuristic.
    """
    left_masks, right_masks = list(left.get("mask", [])), list(right.get("mask", []))
    left_frames, right_frames = list(left.get("image_idx", [])), list(right.get("image_idx", []))
    left_paths, right_paths = list(left.get("color_path", [])), list(right.get("color_path", []))
    left_obs, right_obs = list(left.get("obs_uids", [])), list(right.get("obs_uids", []))
    best_iou, best = 0.0, None
    for left_idx, left_mask_value in enumerate(left_masks):
        for right_idx, right_mask_value in enumerate(right_masks):
            same_path = (
                left_idx < len(left_paths) and right_idx < len(right_paths)
                and str(left_paths[left_idx]) == str(right_paths[right_idx])
            )
            same_frame = (
                left_idx < len(left_frames) and right_idx < len(right_frames)
                and int(left_frames[left_idx]) == int(right_frames[right_idx])
            )
            if not (same_path or same_frame):
                continue
            left_mask = _as_numpy(left_mask_value).astype(bool, copy=False)
            right_mask = _as_numpy(right_mask_value).astype(bool, copy=False)
            if left_mask.shape != right_mask.shape:
                right_mask = cv2.resize(
                    right_mask.astype(np.uint8),
                    (left_mask.shape[1], left_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            union = int(np.logical_or(left_mask, right_mask).sum())
            iou = float(np.logical_and(left_mask, right_mask).sum() / union) if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best = {
                    "left_member_index": left_idx,
                    "right_member_index": right_idx,
                    "shared_frame_idx": int(left_frames[left_idx]) if left_idx < len(left_frames) else None,
                    "shared_rgb_path": str(left_paths[left_idx]) if left_idx < len(left_paths) else None,
                    "left_obs_uid": str(left_obs[left_idx]) if left_idx < len(left_obs) else None,
                    "right_obs_uid": str(right_obs[right_idx]) if right_idx < len(right_obs) else None,
                }
    return best_iou, best


def deduplicate_ranked_candidates(
    scores: Sequence[float],
    objects: Sequence[Mapping[str, Any]],
    iou_threshold: float,
    max_keep: int,
) -> Tuple[List[int], dict]:
    """Greedy score-ordered NMS over map candidates using shared-frame mask IoU."""
    row = np.asarray(scores, dtype=float)
    ranked = [int(idx) for idx in np.argsort(-row, kind="stable") if np.isfinite(row[idx])]
    kept: List[int] = []
    dropped: List[dict] = []
    comparisons = 0
    exhausted = True
    for obj_idx in ranked:
        duplicate = None
        for representative_idx in kept:
            iou, evidence = same_frame_mask_iou(objects[obj_idx], objects[representative_idx])
            comparisons += 1
            if iou > float(iou_threshold) and (duplicate is None or iou > duplicate[0]):
                duplicate = (iou, representative_idx, evidence)
        if duplicate is not None:
            iou, representative_idx, evidence = duplicate
            dropped.append({
                "object_index": obj_idx,
                "score": float(row[obj_idx]),
                "representative_object_index": representative_idx,
                "representative_score": float(row[representative_idx]),
                "same_frame_mask_iou": float(iou),
                "shared_frame_evidence": evidence,
            })
            continue
        kept.append(obj_idx)
        if len(kept) >= int(max_keep):
            exhausted = False
            break
    return kept, {
        "method": "score_ordered_same_frame_mask_nms",
        "iou_operator": ">",
        "iou_threshold": float(iou_threshold),
        "ranked_before_top": ranked[: max(6, int(max_keep))],
        "kept_object_indices": kept,
        "dropped": dropped,
        "comparisons": comparisons,
        "search_exhausted": exhausted,
    }


class BlockingAssociationGate:
    def __init__(self, *, cfg: Any, output_dir: Path, rerun: Any = None):
        gate_cfg = cfg.get("association_gate") or {}
        self.mode = str(gate_cfg.get("mode", "off")).lower()
        if self.mode not in VALID_MODES:
            raise ValueError(f"unsupported association gate mode: {self.mode}")
        self.threshold_scope = str(gate_cfg.get("threshold_scope", "create_only"))
        if self.threshold_scope not in VALID_SCOPES:
            raise ValueError(f"unsupported threshold_scope: {self.threshold_scope}")
        self.margin_threshold = float(gate_cfg.get("margin_threshold", 0.20))
        self.threshold_distance = float(gate_cfg.get("threshold_distance", 0.30))
        self.association_top_k = int(gate_cfg.get("association_top_k", 2))
        self.create_top_k = int(gate_cfg.get("create_top_k", 3))
        self.candidate_iou_filter_enabled = bool(gate_cfg.get("candidate_iou_filter_enabled", True))
        self.candidate_iou_threshold = float(gate_cfg.get("candidate_iou_threshold", 0.85))
        self.sim_threshold = float(cfg.get("sim_threshold"))
        self.review_all_new = bool(gate_cfg.get("review_all_new", True))
        self.mask_change_enabled = bool(gate_cfg.get("mask_change_enabled", True))
        self.support_window = int(gate_cfg.get("support_window", 5))
        self.support_min_history = int(gate_cfg.get("support_min_history", 3))
        self.support_reference_min = float(gate_cfg.get("support_reference_min", 0.75))
        self.support_drop_threshold = float(gate_cfg.get("support_drop_threshold", 0.20))
        if not 1 <= self.support_min_history <= self.support_window:
            raise ValueError("support history must satisfy 1 <= min_history <= window")
        if not (0 <= self.support_reference_min <= 1 and 0 < self.support_drop_threshold <= 1):
            raise ValueError("support thresholds must be in [0,1], with drop > 0")
        if self.mode != "off" and self.mask_change_enabled and cfg.get("spatial_sim_type", "overlap") != "overlap":
            raise ValueError("mask_change requires spatial_sim_type=overlap; disable it for other metrics")
        self._support_history: Dict[str, deque] = {}
        self._last_support_frame: Optional[int] = None
        self.model = str(gate_cfg.get("model", "gpt-5.6-terra"))
        reasoning_effort = gate_cfg.get("reasoning_effort", "high")
        self.reasoning_effort = None if reasoning_effort is None or str(reasoning_effort).lower() in {"", "none", "null"} else str(reasoning_effort)
        self.base_url = str(gate_cfg.get("base_url", "https://api.codelink.chat/v1")).rstrip("/")
        self.timeout_seconds = float(gate_cfg.get("timeout_seconds", 300))
        self.max_retries = int(gate_cfg.get("max_retries", 1))
        self.api_key_env = str(gate_cfg.get("api_key_env", "GATE_API_KEY"))
        self.api_key_required = bool(gate_cfg.get("api_key_required", True))
        self.max_events = int(gate_cfg.get("max_events", 0))
        self.oracle_min_purity = float(gate_cfg.get("oracle_min_purity", 0.80))
        self.oracle_gt_path = gate_cfg.get("oracle_gt_path")
        self.rerun = rerun
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.iou_prefilter_path = self.output_dir / "iou_prefilter.jsonl"
        self.support_path = self.output_dir / "spatial_support.jsonl"
        self.annotation_dir = self.output_dir / "human_annotation_blind"
        self.annotation_cases_dir = self.annotation_dir / "cases"
        self.annotation_cases_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_map_path = self.output_dir / "human_annotation_case_map.jsonl"
        self.started_at = _utc_now()
        self.stats = Counter()
        self.events: List[dict] = []
        self.annotation_cases: List[dict] = []
        self._human_input = input
        self.gt = self._load_oracle_gt(self.oracle_gt_path) if self.mode == "oracle" else {}
        if self.mode == "vlm" and self.api_key_required and not os.environ.get(self.api_key_env):
            raise RuntimeError(f"{self.api_key_env} must be set for association_gate.mode=vlm")
        self.config = {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "sim_threshold": self.sim_threshold,
            "margin_threshold": self.margin_threshold,
            "threshold_distance": self.threshold_distance,
            "threshold_scope": self.threshold_scope,
            "review_all_new": self.review_all_new,
            "mask_change_enabled": self.mask_change_enabled,
            "support_window": self.support_window,
            "support_min_history": self.support_min_history,
            "support_reference_min": self.support_reference_min,
            "support_drop_threshold": self.support_drop_threshold,
            "support_metric": "overlap: current points within 0.025m of pre-fusion object / current points",
            "support_history_policy": "stable object UID; previous frames only; non-triggered baseline ATTACH only; no history transfer between merged UIDs",
            "association_top_k": self.association_top_k,
            "create_top_k": self.create_top_k,
            "candidate_iou_filter_enabled": self.candidate_iou_filter_enabled,
            "candidate_iou_threshold": self.candidate_iou_threshold,
            "candidate_iou_metric": "maximum visible-mask IoU on a shared historical frame",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_required": self.api_key_required,
            "oracle_gt_path": str(self.oracle_gt_path) if self.oracle_gt_path else None,
            "max_events": self.max_events,
            "formal_vlm_actions": list(ALIASES) + ["NEW", "UNCERTAIN"],
            "formal_human_actions": list(ALIASES) + ["NEW", "UNCERTAIN", "DISCARD"],
            "human_interaction": "blocking terminal choice only; evidence saved before prompt",
            "discard_route_implemented_but_prompt_disabled": True,
            "candidate_history_views": 3,
            "candidate_point_cloud_scale": "shared_across_all_candidates_per_event",
        }
        _json_dump(self.output_dir / "config.json", self.config)
        (self.annotation_dir / "README.md").write_text(
            "# Blind human annotation inputs\n\n"
            "Open `index.html` and choose one Candidate, `NEW`, or `UNCERTAIN` with the option buttons. "
            "You may skip any case; only explicitly saved cases are included by **Export labeled JSONL**. "
            "Progress is stored in this browser's local storage. `labels_template.jsonl` remains a manual fallback.\n\n"
            "Each `case_XXXX` directory is a copy of the exact images and prompts sent for one online gate event. "
            "The packet deliberately omits frame identity, trigger type, baseline action, scores, object IDs, "
            "VLM responses, parsed choices, and final mapping decisions.\n",
            encoding="utf-8",
        )
        self._write_summary(status="running")

    @staticmethod
    def _load_oracle_gt(path_value: Any) -> Dict[Tuple[int, int], dict]:
        if not path_value:
            raise ValueError("association_gate.oracle_gt_path is required in oracle mode")
        path = Path(str(path_value))
        if not path.is_file():
            raise FileNotFoundError(path)
        records: Dict[Tuple[int, int], dict] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                raw_index = _extract_raw_index(row.get("obs_uid"))
                if raw_index is not None:
                    # Map objects store the processed dataset index in image_idx;
                    # raw_frame is the original source id (e.g. 0, 5, 10 at stride 5).
                    records[(int(row["frame_idx"]), raw_index)] = row
        return records

    def _member_gt(self, frame_idx: int, obs_uid: Optional[str]) -> Optional[int]:
        raw_index = _extract_raw_index(obs_uid)
        row = self.gt.get((int(frame_idx), raw_index)) if raw_index is not None else None
        if not row or not row.get("gt_assignment_eligible"):
            return None
        if float(row.get("gt_purity") or 0.0) < self.oracle_min_purity:
            return None
        return int(row["gt_top_id"])

    def _object_gt(self, obj: Mapping[str, Any]) -> Tuple[Optional[int], dict]:
        image_indices, obs_uids = list(obj.get("image_idx", [])), list(obj.get("obs_uids", []))
        votes = []
        for idx, raw_frame in enumerate(image_indices):
            obs_uid = obs_uids[idx] if idx < len(obs_uids) else None
            gt_id = self._member_gt(int(raw_frame), obs_uid)
            if gt_id is not None:
                votes.append(gt_id)
        if not votes:
            return None, {"eligible_members": 0, "dominant_fraction": None}
        gt_id, count = Counter(votes).most_common(1)[0]
        fraction = count / len(votes)
        return (int(gt_id) if fraction >= self.oracle_min_purity else None), {
            "eligible_members": len(votes),
            "dominant_fraction": fraction,
            "dominant_gt_id": int(gt_id),
        }

    @staticmethod
    def _representative_members(obj: Mapping[str, Any], limit: int = 3) -> List[Tuple[str, int]]:
        """Migrate the annotation selector: best mask, recent good, then diverse."""
        paths, masks = list(obj.get("color_path", [])), list(obj.get("mask", []))
        image_indices = list(obj.get("image_idx", []))
        confidences = list(obj.get("conf", []))
        records = []
        for index in range(min(len(paths), len(masks))):
            try:
                area = float(_as_numpy(masks[index]).astype(bool).sum())
            except Exception:
                continue
            path = Path(str(paths[index]))
            if area <= 0 or not path.is_file():
                continue
            frame = int(image_indices[index]) if index < len(image_indices) else index
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            records.append({"index": index, "area": area, "frame": frame, "confidence": confidence})
        if not records or limit <= 0:
            return []

        best = max(records, key=lambda row: (row["area"], row["confidence"], row["frame"], -row["index"]))
        chosen: List[Tuple[str, int]] = [("H1 BEST MASK", int(best["index"]))]
        remaining = [row for row in records if row["index"] != best["index"]]
        if remaining and len(chosen) < limit:
            good_area = max(25.0, 0.35 * float(best["area"]))
            good = [row for row in remaining if row["area"] >= good_area] or remaining
            recent = max(good, key=lambda row: (row["frame"], row["confidence"], row["area"], -row["index"]))
            chosen.append(("H2 RECENT", int(recent["index"])))
            remaining = [row for row in remaining if row["index"] != recent["index"]]
        if remaining and len(chosen) < limit:
            diverse = max(
                remaining,
                key=lambda row: (
                    abs(int(row["frame"]) - int(best["frame"])),
                    row["area"], row["confidence"], -row["index"],
                ),
            )
            chosen.append(("H3 DIVERSE", int(diverse["index"])))
        return chosen[:limit]

    def _save_candidate_image(
        self,
        event_dir: Path,
        alias: str,
        obj: Mapping[str, Any],
        current_points: np.ndarray,
        candidate_points: np.ndarray,
        projection_ranges: Sequence[Tuple[float, float, float]],
    ) -> dict:
        paths, masks = list(obj.get("color_path", [])), list(obj.get("mask", []))
        boxes, obs_uids = list(obj.get("xyxy", [])), list(obj.get("obs_uids", []))
        image_indices = list(obj.get("image_idx", []))
        selected = self._representative_members(obj, limit=3)
        history_rgbs: List[np.ndarray] = []
        history_labels: List[str] = []
        history_metadata: List[dict] = []
        for role, member_idx in selected:
            source_path = Path(str(paths[member_idx]))
            image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            bbox = boxes[member_idx] if member_idx < len(boxes) else None
            history_rgbs.append(_annotated_crop(image_rgb, masks[member_idx], bbox, ""))
            history_labels.append(role)
            history_metadata.append({
                "display_role": role.split()[0],
                "selection_reason": " ".join(role.split()[1:]).lower().replace(" ", "_"),
                "member_index": member_idx,
                "frame_idx": int(image_indices[member_idx]) if member_idx < len(image_indices) else None,
                "obs_uid": str(obs_uids[member_idx]) if member_idx < len(obs_uids) else None,
                "source_rgb_path": str(source_path),
                "mask_area": int(_as_numpy(masks[member_idx]).astype(bool).sum()),
            })
        if not history_rgbs:
            raise ValueError(f"candidate {alias} has no readable historical RGB/mask member")
        output_path = event_dir / f"candidate_{alias}.jpg"
        _write_rgb(
            output_path,
            _candidate_evidence_composite(
                history_rgbs, history_labels, current_points, candidate_points,
                alias, projection_ranges,
            ),
        )
        range_payload = [
            {"projection": name, "center": [center_a, center_b], "square_extent_m": extent}
            for name, (center_a, center_b, extent) in zip(("XY", "XZ", "YZ"), projection_ranges)
        ]
        return {
            "alias": alias,
            "object_uid": str(obj.get("id")),
            "class_name": str(obj.get("class_name", "")),
            "num_detections": int(obj.get("num_detections", len(paths))),
            "selected_history": history_metadata,
            "image_path": output_path.name,
            "image_sha256": _sha256_file(output_path),
            "image_layout": "top_55pct=three_historical_rgb_red_masks;bottom_45pct=shared_world_xyz_projections",
            "point_cloud_role": "auxiliary_identity_evidence",
            "point_cloud_sources": {"current": "detection['pcd']", "candidate": "obj['pcd']"},
            "point_cloud_projections": ["XY", "XZ", "YZ"],
            "point_cloud_event_shared_ranges": range_payload,
            "point_cloud_event_shared_ranges_uid": hashlib.sha256(
                json.dumps(range_payload, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "current_point_count_rendered": int(len(current_points)),
            "candidate_point_count_rendered": int(len(candidate_points)),
            "current_point_color": "magenta",
            "candidate_point_color": "cyan",
        }

    @staticmethod
    def _prompts(kind: str, aliases: Sequence[str]) -> Tuple[str, str]:
        options = ", ".join(list(aliases) + ["NEW", "UNCERTAIN"])
        system = ASSOCIATION_SYSTEM_PROMPT if kind == "association" else CREATE_SYSTEM_PROMPT
        return system, (
            "Review this association event.\n\n"
            "Image order:\n"
            "I1: CURRENT scene context.\n"
            "I2: CURRENT cropped observation.\n"
            "Then: one evidence card for each listed candidate. Each card contains up to three historical red-mask views above and the shared-scale XY/XZ/YZ point-cloud comparison below.\n\n"
            "Assess every candidate independently using both its 2D history and 3D evidence. "
            "Decide which candidate, if any, is the same individual physical object as CURRENT.\n\n"
            f"Allowed final choices: {options}."
        )

    def _request_payload(self, system_prompt: str, user_prompt: str, images: Sequence[Tuple[str, Path]], aliases: Sequence[str]) -> dict:
        allowed = list(aliases) + ["NEW", "UNCERTAIN"]
        content: List[dict] = [{"type": "text", "text": user_prompt}]
        for label, path in images:
            content.extend([
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": _image_data_url(path), "detail": "high"}},
            ])
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "association_gate_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "candidate_assessments": {
                                "type": "array",
                                "minItems": len(aliases),
                                "maxItems": len(aliases),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "code": {"type": "string", **({"enum": list(aliases)} if aliases else {})},
                                        "relation": {"type": "string", "enum": ["SAME", "DIFFERENT", "UNCERTAIN"]},
                                        "evidence": {"type": "string"},
                                    },
                                    "required": ["code", "relation", "evidence"],
                                    "additionalProperties": False,
                                },
                            },
                            "choice": {"type": "string", "enum": allowed},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reason": {"type": "string"},
                        },
                        "required": ["candidate_assessments", "choice", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_completion_tokens": 1400,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    @staticmethod
    def _redact_payload(payload: dict, images: Sequence[Tuple[str, Path]]) -> dict:
        redacted = json.loads(json.dumps(payload))
        image_iter = iter(images)
        for part in redacted["messages"][1]["content"]:
            if part.get("type") == "image_url":
                label, path = next(image_iter)
                part["image_url"]["url"] = (
                    "data:image/jpeg;base64,"
                    f"<redacted label={label!r} sha256={_sha256_file(path)}>"
                )
        return redacted

    def _write_human_annotation_case(
        self,
        *,
        event_id: str,
        snapshot_uid: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[Tuple[str, Path]],
        aliases: Sequence[str],
    ) -> str:
        """Copy exact VLM inputs into a neutral packet with every answer hidden."""
        case_id = f"case_{len(self.annotation_cases) + 1:04d}"
        case_dir = self.annotation_cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        public_images = []
        for order, (label, source) in enumerate(images, 1):
            destination = case_dir / source.name
            shutil.copy2(source, destination)
            descriptor = _image_media_descriptor(destination)
            public_images.append({
                "order": order,
                "label": label,
                "path": destination.name,
                "sha256": descriptor["payload_sha256"],
                "width": descriptor["width"],
                "height": descriptor["height"],
                "mime_type": descriptor["mime_type"],
            })
        (case_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
        (case_dir / "user_prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
        public_case = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "blind": True,
            "images": public_images,
            "prompt_files": ["system_prompt.txt", "user_prompt.txt"],
            "allowed_choices": list(aliases) + ["NEW", "UNCERTAIN"] + (["DISCARD"] if self.mode == "human" else []),
            "annotation_fields": {
                "choice": None,
                "confidence": None,
                "candidate_notes": {alias: "" for alias in aliases},
                "reason": "",
            },
        }
        _json_dump(case_dir / "case.json", public_case)
        self.annotation_cases.append(public_case)
        _jsonl_append(self.annotation_map_path, {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "event_id": event_id,
            "h_snapshot_uid": snapshot_uid,
        })
        self.stats["human_annotation_cases"] += 1
        self._write_human_annotation_index()
        return case_id

    def _write_human_annotation_index(self) -> None:
        (self.annotation_dir / "README.md").write_text(
            "# Blind human annotation inputs\n\n"
            "Open `index.html` and choose one Candidate, `NEW`, or `UNCERTAIN` with the option buttons. "
            "You may skip any case; only explicitly saved cases are included by **Export labeled JSONL**. "
            "Progress is stored in this browser's local storage. `labels_template.jsonl` remains a manual fallback.\n\n"
            "Each `case_XXXX` directory is a copy of the exact images and prompts sent for one online gate event. "
            "The packet deliberately omits frame identity, trigger type, baseline action, scores, object IDs, "
            "VLM responses, parsed choices, and final mapping decisions.\n",
            encoding="utf-8",
        )
        _json_dump(self.annotation_dir / "manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "blind": True,
            "case_count": len(self.annotation_cases),
            "cases": [
                {"case_id": case["case_id"], "path": f"cases/{case['case_id']}/case.json"}
                for case in self.annotation_cases
            ],
        })
        with (self.annotation_dir / "labels_template.jsonl").open("w", encoding="utf-8") as handle:
            for case in self.annotation_cases:
                handle.write(json.dumps({
                    "case_id": case["case_id"],
                    "choice": None,
                    "confidence": None,
                    "candidate_notes": case["annotation_fields"]["candidate_notes"],
                    "reason": "",
                }, ensure_ascii=False, sort_keys=True) + "\n")
        public_cases_json = json.dumps(
            self.annotation_cases, ensure_ascii=False, separators=(",", ":")
        ).replace("<", "\\u003c")
        document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>门控事件人工盲标</title>
<style>
:root{--bg:#0b1018;--panel:#131b27;--panel2:#182334;--line:#334155;--text:#e5edf7;--muted:#9cabc0;--blue:#2563eb;--green:#15803d;--amber:#d97706;--red:#dc2626}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,"Microsoft YaHei",sans-serif}button,select,textarea{font:inherit}
header{position:sticky;top:0;z-index:5;background:#0f172a;border-bottom:1px solid var(--line);padding:12px 18px}.top{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.top h1{font-size:21px;margin:0 12px 0 0}.status{color:#bfdbfe}.help{color:var(--muted);margin:7px 0 0;line-height:1.5}.layout{display:grid;grid-template-columns:210px minmax(640px,1fr) 320px;min-height:calc(100vh - 92px)}
nav{border-right:1px solid var(--line);padding:12px;overflow:auto;background:#0f172a}.case-item{display:block;width:100%;text-align:left;color:var(--text);background:var(--panel);border:1px solid var(--line);border-left:4px solid #64748b;border-radius:7px;padding:9px;margin:0 0 7px;cursor:pointer}.case-item.done{border-left-color:#22c55e}.case-item.current{background:#1d4ed8}.case-item small{display:block;color:#cbd5e1;margin-top:3px}
main{padding:16px;overflow:auto}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:14px}h2{font-size:18px;margin:0 0 10px}.current-grid{display:grid;grid-template-columns:1.35fr .65fr;gap:12px}.candidate-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.evidence{margin:0;background:#080c12;border:1px solid var(--line);border-radius:8px;padding:8px}.evidence img{display:block;width:100%;max-height:720px;object-fit:contain;cursor:zoom-in}.current-grid .evidence img{max-height:480px}.evidence figcaption{color:#cbd5e1;padding:7px 3px 1px;font-weight:650}
aside{border-left:1px solid var(--line);padding:14px;background:#0f172a}.sticky{position:sticky;top:105px}.choice{display:block;width:100%;text-align:left;color:var(--text);background:var(--panel2);border:1px solid #475569;border-radius:8px;padding:11px;margin:8px 0;cursor:pointer;font-weight:700}.choice.selected{background:#1d4ed8;outline:3px solid #60a5fa}.field{margin:14px 0}.field label{display:block;color:#cbd5e1;margin-bottom:6px}.field select,.field textarea{width:100%;color:var(--text);background:var(--panel2);border:1px solid #475569;border-radius:7px;padding:8px}.field textarea{min-height:90px;resize:vertical}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.wide{grid-column:1/3}.btn{border:0;border-radius:8px;padding:9px 10px;color:white;background:#334155;cursor:pointer}.primary{background:var(--blue)}.ok{background:var(--green)}.warn{background:var(--amber)}.danger{background:#7f1d1d}.message{min-height:24px;color:#93c5fd;margin-top:10px}.links{line-height:1.8}.links a{color:#7dd3fc}.export{border-top:1px solid var(--line);margin-top:16px;padding-top:13px}.export .btn{width:100%;margin:4px 0}
#zoom{display:none;position:fixed;inset:0;z-index:20;background:#000e;padding:24px;align-items:center;justify-content:center}#zoom.open{display:flex}#zoom img{max-width:98vw;max-height:96vh;object-fit:contain}.badge{display:inline-block;background:#334155;border-radius:999px;padding:2px 8px;color:#dbeafe}
@media(max-width:1150px){.layout{grid-template-columns:170px 1fr}.layout aside{grid-column:1/3;border-left:0;border-top:1px solid var(--line)}.sticky{position:static}.candidate-grid{grid-template-columns:1fr}}@media(max-width:760px){.layout{display:block}nav{display:flex;gap:6px;overflow:auto}.case-item{min-width:125px}.current-grid{grid-template-columns:1fr}main{padding:9px}}
</style></head><body>
<header><div class="top"><h1>门控事件人工盲标</h1><span id="counter" class="badge"></span><span id="progress" class="status"></span></div><p class="help">单选 Candidate / NEW / UNCERTAIN。无需逐例标注：可以直接上一例、下一例或跳过；只有点击“只保存”或“保存并下一例”的案例才进入导出文件。数据仅保存在当前浏览器，不包含 mapper/VLM 答案。</p></header>
<div class="layout"><nav id="caseList"></nav><main><section class="panel"><h2>当前观测</h2><div id="currentImages" class="current-grid"></div></section><section class="panel"><h2>候选证据</h2><div id="candidateImages" class="candidate-grid"></div></section></main>
<aside><div class="sticky"><h2>本例选择</h2><div id="choices"></div><div class="field"><label for="confidence">置信度（可不填）</label><select id="confidence"><option value="">不填写</option><option value="1">高</option><option value="0.67">中</option><option value="0.33">低</option></select></div><div class="field"><label for="reason">备注（可不填）</label><textarea id="reason" placeholder="只写有助于之后复核的简短依据"></textarea></div><div class="actions"><button id="save" class="btn primary">只保存</button><button id="saveNext" class="btn ok">保存并下一例</button><button id="prev" class="btn">上一例</button><button id="next" class="btn">下一例/跳过</button><button id="nextBlank" class="btn warn wide">下一未标注</button><button id="clear" class="btn danger wide">清除本例标签</button></div><div id="message" class="message"></div><div class="links"><a id="caseJson">查看 case.json</a> · <a id="systemPrompt">system prompt</a> · <a id="userPrompt">user prompt</a></div><div class="export"><button id="exportLabeled" class="btn ok">导出已标注 JSONL</button><button id="exportAll" class="btn">导出全部状态 JSON</button><button id="importLabels" class="btn">导入已有 JSONL</button><input id="importFile" type="file" accept=".jsonl,.json" hidden><a href="labels_template.jsonl">空标签模板</a></div></div></aside></div>
<div id="zoom"><img alt="放大证据"></div>
<script>
const CASES=__CASES_JSON__;
const STORAGE_KEY='blocking-gate-blind-labels:'+location.pathname;
let index=0,labels={},draftChoice=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
try{labels=JSON.parse(localStorage.getItem(STORAGE_KEY)||'{}')}catch(_){labels={}}
function persist(){localStorage.setItem(STORAGE_KEY,JSON.stringify(labels));renderNav()}
function showMessage(text,error=false){$('message').textContent=text;$('message').style.color=error?'#fca5a5':'#93c5fd'}
function choiceText(value){return value==='NEW'?'NEW（新建对象）':value==='UNCERTAIN'?'UNCERTAIN（无法可靠判断）':value==='DISCARD'?'DISCARD（舍弃当前观测）':`Candidate ${value}`}
function renderNav(){const done=CASES.filter(c=>labels[c.case_id]?.choice).length;$('progress').textContent=`已保存 ${done} / ${CASES.length}；未标 ${CASES.length-done}`;$('caseList').innerHTML=CASES.map((c,i)=>`<button class="case-item ${labels[c.case_id]?.choice?'done':''} ${i===index?'current':''}" data-i="${i}">${esc(c.case_id)}<small>${labels[c.case_id]?.choice?'已标：'+esc(labels[c.case_id].choice):'未标，可跳过'}</small></button>`).join('');document.querySelectorAll('.case-item').forEach(b=>b.onclick=()=>load(Number(b.dataset.i)))}
function figure(c,item){return `<figure class="evidence"><img src="cases/${encodeURIComponent(c.case_id)}/${encodeURIComponent(item.path)}" alt="${esc(item.label)}"><figcaption>${esc(item.label)}</figcaption></figure>`}
function bindZoom(){document.querySelectorAll('.evidence img').forEach(img=>img.onclick=()=>{$('#zoom img').src=img.src;$('zoom').classList.add('open')})}
function load(i){if(!CASES.length)return;index=Math.max(0,Math.min(i,CASES.length-1));const c=CASES[index],saved=labels[c.case_id]||{};draftChoice=saved.choice||null;$('counter').textContent=`${index+1} / ${CASES.length} · ${c.case_id}`;const current=c.images.filter(x=>x.path.startsWith('current_'));const candidates=c.images.filter(x=>x.path.startsWith('candidate_'));$('currentImages').innerHTML=current.map(x=>figure(c,x)).join('');$('candidateImages').innerHTML=candidates.map(x=>figure(c,x)).join('');$('choices').innerHTML=c.allowed_choices.map(v=>`<button class="choice ${draftChoice===v?'selected':''}" data-value="${esc(v)}">${esc(choiceText(v))}</button>`).join('');document.querySelectorAll('.choice').forEach(b=>b.onclick=()=>selectChoice(b.dataset.value));$('confidence').value=saved.confidence==null?'':String(saved.confidence);$('reason').value=saved.reason||'';$('caseJson').href=`cases/${c.case_id}/case.json`;$('systemPrompt').href=`cases/${c.case_id}/system_prompt.txt`;$('userPrompt').href=`cases/${c.case_id}/user_prompt.txt`;$('prev').disabled=index===0;$('next').disabled=index===CASES.length-1;showMessage(saved.choice?'已加载本浏览器保存的标签':'本例未标，可直接跳过');renderNav();bindZoom();window.scrollTo({top:0,behavior:'smooth'})}
function selectChoice(value){draftChoice=value;document.querySelectorAll('.choice').forEach(b=>b.classList.toggle('selected',b.dataset.value===value));showMessage(`已选择 ${value}，点击保存后才会写入`) }
function save(goNext){const c=CASES[index];if(!draftChoice){showMessage('请先选择一个选项；若不想标本例，直接点击“下一例/跳过”',true);return}labels[c.case_id]={case_id:c.case_id,choice:draftChoice,confidence:$('confidence').value===''?null:Number($('confidence').value),candidate_notes:Object.fromEntries(c.allowed_choices.filter(x=>/^[A-Z]$/.test(x)).map(x=>[x,''])),reason:$('reason').value.trim()};persist();showMessage(`已保存 ${c.case_id}`);if(goNext&&index<CASES.length-1)load(index+1)}
function clearCurrent(){const id=CASES[index].case_id;delete labels[id];persist();load(index);showMessage(`已清除 ${id}`)}
function nextBlank(){for(let offset=1;offset<=CASES.length;offset++){const i=(index+offset)%CASES.length;if(!labels[CASES[i].case_id]?.choice){load(i);return}}showMessage('当前所有案例均已标注')}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function exportLabeled(){const rows=CASES.map(c=>labels[c.case_id]).filter(x=>x?.choice);if(!rows.length){showMessage('目前没有已保存标签',true);return}download('blocking_gate_manual_labels.jsonl',rows.map(x=>JSON.stringify(x)).join('\\n')+'\\n','application/x-ndjson');showMessage(`已导出 ${rows.length} 条已标注结果`)}
function exportAll(){download('blocking_gate_annotation_state.json',JSON.stringify({schema_version:'blocking-gate-human-labels-v1',saved_at:new Date().toISOString(),labels},null,2),'application/json');showMessage('已导出全部浏览器状态')}
async function importFile(file){try{const text=await file.text();let rows;try{const parsed=JSON.parse(text);rows=parsed.labels?Object.values(parsed.labels):(Array.isArray(parsed)?parsed:[parsed])}catch(_){rows=text.split(/\\r?\\n/).filter(Boolean).map(JSON.parse)}const known=new Map(CASES.map(c=>[c.case_id,c]));let count=0;for(const row of rows){const c=known.get(row.case_id);if(c&&c.allowed_choices.includes(row.choice)){labels[row.case_id]=row;count++}}persist();load(index);showMessage(`已导入 ${count} 条有效标签`)}catch(e){showMessage(`导入失败：${e.message}`,true)}}
$('save').onclick=()=>save(false);$('saveNext').onclick=()=>save(true);$('prev').onclick=()=>load(index-1);$('next').onclick=()=>load(index+1);$('nextBlank').onclick=nextBlank;$('clear').onclick=clearCurrent;$('exportLabeled').onclick=exportLabeled;$('exportAll').onclick=exportAll;$('importLabels').onclick=()=>$('importFile').click();$('importFile').onchange=e=>e.target.files[0]&&importFile(e.target.files[0]);$('zoom').onclick=()=>$('zoom').classList.remove('open');
document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')load(index-1);else if(e.key==='ArrowRight')load(index+1);else{const c=CASES[index],key=e.key.toUpperCase(),value=key==='N'?'NEW':key==='U'?'UNCERTAIN':key;if(c.allowed_choices.includes(value))selectChoice(value)}};
renderNav();load(0);
</script></body></html>""".replace("__CASES_JSON__", public_cases_json)
        (self.annotation_dir / "index.html").write_text(document, encoding="utf-8")

    def _call_vlm(self, payload: dict, failure_log_path: Optional[Path] = None) -> Tuple[dict, dict, float]:
        """Call the OpenAI-compatible endpoint through the official SDK.

        ``trust_env=False`` is deliberate: this server has a SOCKS proxy setting
        without socksio installed, while the gateway is reachable directly.  It
        also avoids inheriting an unintended proxy/client identity.
        """
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for VLM requests")
        request_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        transport = httpx.Client(timeout=self.timeout_seconds, trust_env=False)
        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
            http_client=transport,
        )
        last_error: Optional[Exception] = None
        diagnostics: List[dict] = []
        started = time.perf_counter()
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.chat.completions.create(**payload)
                    raw = response.model_dump(mode="json")
                    raw["_request_id"] = getattr(response, "_request_id", None)
                    content = response.choices[0].message.content
                    if isinstance(content, list):
                        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                    text = str(content).strip()
                    if text.startswith("```"):
                        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
                    return raw, json.loads(text), time.perf_counter() - started
                except APIStatusError as exc:
                    detail = exc.response.text
                    request_id = getattr(exc, "request_id", None) or exc.response.headers.get("x-request-id")
                    diagnostics.append({
                        "attempt": attempt + 1,
                        "http_status": exc.status_code,
                        "request_id": request_id,
                        "response_headers": {
                            key: exc.response.headers.get(key)
                            for key in ("content-type", "x-request-id", "retry-after")
                            if exc.response.headers.get(key) is not None
                        },
                        "error_type": type(exc).__name__,
                        "error_body": detail,
                    })
                    last_error = RuntimeError(f"HTTP {exc.status_code}: {detail}")
                except (APIConnectionError, APITimeoutError) as exc:
                    diagnostics.append({
                        "attempt": attempt + 1,
                        "http_status": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    last_error = exc
                except Exception as exc:
                    diagnostics.append({"attempt": attempt + 1, "http_status": None, "error_type": type(exc).__name__, "error": str(exc)})
                    last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 4.0))
        finally:
            client.close()
        if failure_log_path is not None:
            _json_dump(failure_log_path, {
                "client": "openai-python",
                "openai_sdk_version": openai.__version__,
                "transport": "httpx.Client(trust_env=False)",
                "endpoint": f"{self.base_url}/chat/completions",
                "request_body_utf8_bytes": request_bytes,
                "attempts": diagnostics,
            })
        raise RuntimeError(f"VLM request failed after {self.max_retries + 1} attempts: {last_error}")

    def _write_live_human_review(
        self,
        event_dir: Path,
        event_id: str,
        aliases: Sequence[str],
    ) -> Path:
        """Replace the single live review page with the currently blocking event."""
        allowed = list(aliases) + ["NEW", "UNCERTAIN", "DISCARD"]
        event_prefix = f"events/{event_dir.name}"
        figures = [
            f'<figure><img src="{event_prefix}/current_context.jpg" alt="CURRENT context">'
            '<figcaption>CURRENT context</figcaption></figure>',
            f'<figure><img src="{event_prefix}/current_crop.jpg" alt="CURRENT crop">'
            '<figcaption>CURRENT crop</figcaption></figure>',
        ]
        figures.extend(
            f'<figure><img src="{event_prefix}/candidate_{html.escape(alias)}.jpg" '
            f'alt="Candidate {html.escape(alias)}"><figcaption>Candidate '
            f'{html.escape(alias)}</figcaption></figure>'
            for alias in aliases
        )
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>Human gate · {html.escape(event_id)}</title>
<style>body{{font-family:system-ui;margin:20px;background:#111827;color:#f8fafc}}
.notice{{padding:12px;background:#1e293b;border:1px solid #475569;border-radius:8px}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}
figure{{margin:0;padding:10px;background:#0f172a;border:1px solid #334155}}
img{{width:100%;max-height:78vh;object-fit:contain;background:#020617}}
figcaption{{margin-top:8px;font-weight:700}}code{{color:#7dd3fc}}</style></head>
<body><h1>Blocking human review</h1>
<p class="notice">Event <code>{html.escape(event_id)}</code> is paused. Inspect the RED-masked
CURRENT target and each candidate card, then return to the terminal and enter exactly one option:
<strong>{html.escape(' / '.join(allowed))}</strong>.<br>
DISCARD = skip this CURRENT observation (no association or new object).
UNCERTAIN = retain the mapper's original decision.</p>
<div class="images">{''.join(figures)}</div></body></html>"""
        review_path = self.output_dir / "human_review.html"
        temporary_path = self.output_dir / ".human_review.html.tmp"
        temporary_path.write_text(page, encoding="utf-8")
        temporary_path.replace(review_path)
        return review_path

    def _call_human(
        self,
        *,
        event_id: str,
        event_dir: Path,
        aliases: Sequence[str],
    ) -> dict:
        """Block until the operator supplies one valid final action."""
        allowed = list(aliases) + ["NEW", "UNCERTAIN", "DISCARD"]
        allowed_set = set(allowed)
        review_path = self._write_live_human_review(event_dir, event_id, aliases)
        print("\n[human-gate] Mapping is paused for a human decision.", flush=True)
        print(f"[human-gate] Review page: {review_path}", flush=True)
        print(f"[human-gate] Allowed choices: {' / '.join(allowed)}", flush=True)
        while True:
            try:
                raw_choice = self._human_input("[human-gate] Choice: ")
            except EOFError as exc:
                raise HumanInputUnavailableError(
                    "human mode requires interactive stdin; mapping remains stopped "
                    f"with evidence saved at {review_path}"
                ) from exc
            choice = str(raw_choice).strip().upper()
            if choice in allowed_set:
                return {"choice": choice}
            self.stats["human_invalid_inputs"] += 1
            print(
                f"[human-gate] Invalid choice {raw_choice!r}; enter one of: "
                f"{' / '.join(allowed)}",
                flush=True,
            )

    def _oracle_choice(
        self,
        frame_idx: int,
        detection: Mapping[str, Any],
        candidates: Sequence[Tuple[str, int, Mapping[str, Any]]],
    ) -> Tuple[str, dict]:
        obs_uids = list(detection.get("obs_uids", []))
        current_gt = self._member_gt(frame_idx, obs_uids[0] if obs_uids else None)
        candidate_gt, matches = {}, []
        for alias, obj_idx, obj in candidates:
            gt_id, diagnostics = self._object_gt(obj)
            candidate_gt[alias] = {"object_index": obj_idx, "gt_id": gt_id, **diagnostics}
            if current_gt is not None and gt_id == current_gt:
                matches.append(alias)
        diagnostics = {"current_gt_id": current_gt, "candidates": candidate_gt}
        if current_gt is None:
            return "UNCERTAIN", diagnostics
        if matches:
            return matches[0], diagnostics
        if all(item["gt_id"] is not None for item in candidate_gt.values()):
            return "NEW", diagnostics
        return "UNCERTAIN", diagnostics

    def _event_evidence(
        self,
        event_dir: Path,
        image_rgb: np.ndarray,
        detection: Mapping[str, Any],
        candidates: Sequence[Tuple[str, int, Mapping[str, Any]]],
    ) -> Tuple[List[dict], List[Tuple[str, Path]]]:
        masks, boxes = list(detection.get("mask", [])), list(detection.get("xyxy", []))
        if not masks:
            raise ValueError("current detection has no mask")
        mask, bbox = masks[-1], boxes[-1] if boxes else None
        current_points = _sample_points(
            _point_array(detection), 900, f"current:{event_dir.name}"
        )
        prepared_candidates = [
            (
                alias,
                obj,
                _sample_points(
                    _point_array(obj), 1600, f"candidate:{obj.get('id')}:{event_dir.name}"
                ),
            )
            for alias, _, obj in candidates
        ]
        projection_ranges = _shared_projection_ranges(
            [current_points] + [candidate_points for _, _, candidate_points in prepared_candidates]
        )
        context_path, crop_path = event_dir / "current_context.jpg", event_dir / "current_crop.jpg"
        _write_rgb(context_path, _annotated_context(image_rgb, mask, bbox))
        _write_rgb(crop_path, _annotated_crop(image_rgb, mask, bbox, "CURRENT OBSERVATION"))
        manifest = [
            {"role": "I1", "path": context_path.name, "sha256": _sha256_file(context_path)},
            {"role": "I1-crop", "path": crop_path.name, "sha256": _sha256_file(crop_path)},
        ]
        request_images: List[Tuple[str, Path]] = [
            ("I1 current context", context_path), ("I1-crop current observation", crop_path)
        ]
        for alias, obj, candidate_points in prepared_candidates:
            item = self._save_candidate_image(
                event_dir, alias, obj, current_points, candidate_points, projection_ranges,
            )
            manifest.append({"role": f"candidate-{alias}", **item})
            request_images.append((
                f"Candidate {alias}: top three historical RGB/red-mask views; "
                "bottom event-shared-scale CURRENT-vs-candidate 3D",
                event_dir / item["image_path"],
            ))
        return manifest, request_images

    def route_frame(
        self,
        *,
        frame_idx: int,
        source_frame_id: str,
        image_rgb: np.ndarray,
        detection_list: Sequence[Mapping[str, Any]],
        objects: Sequence[Mapping[str, Any]],
        aggregate_sim: Any,
        baseline_match_indices: Sequence[Optional[int]],
        spatial_sim: Any = None,
    ) -> List[Optional[int]]:
        final_matches = list(baseline_match_indices)
        if self.mode == "off":
            return final_matches
        scores = _as_numpy(aggregate_sim).astype(float, copy=False)
        if scores.shape != (len(detection_list), len(objects)):
            raise ValueError(f"aggregate similarity shape {scores.shape} != {(len(detection_list), len(objects))}")
        if len(final_matches) != len(detection_list):
            raise ValueError("baseline match count must equal detection count")
        support_checks = {}
        excluded_from_history = set()
        if self.mask_change_enabled:
            if self._last_support_frame is not None and frame_idx <= self._last_support_frame:
                raise ValueError("support history requires strictly increasing online frame indices")
            if spatial_sim is None:
                raise ValueError("mask_change requires the pre-fusion spatial_sim matrix")
            support = _as_numpy(spatial_sim).astype(float, copy=False)
            if support.shape != scores.shape:
                raise ValueError("spatial similarity shape must equal aggregate similarity shape")
            if not np.isfinite(support).all() or np.any((support < 0) | (support > 1)):
                raise ValueError("overlap support values must be finite and in [0,1]")
            object_uids = [str(obj["id"]) for obj in objects]
            active_uids = set(object_uids)
            if len(active_uids) != len(object_uids):
                raise ValueError("support history requires distinct stable object UIDs")
            self._support_history = {
                uid: history for uid, history in self._support_history.items() if uid in active_uids
            }
            # Freeze every check before processing any observation in this frame.
            for index, match in enumerate(baseline_match_indices):
                if match is None:
                    continue
                if not 0 <= match < len(objects):
                    raise ValueError(f"invalid baseline association index: {match}")
                uid = object_uids[match]
                history = list(self._support_history.get(uid, ()))
                support_checks[index] = {
                    "object_uid": uid, "baseline_match_index": int(match),
                    "history_frames": [sample[0] for sample in history],
                    **compute_support_drop(
                        support[index, match], [sample[1] for sample in history],
                        min_history=self.support_min_history,
                        reference_min=self.support_reference_min,
                        drop_threshold=self.support_drop_threshold,
                    ),
                }
        for detected_idx, detection in enumerate(detection_list):
            baseline_match = baseline_match_indices[detected_idx]
            raw_trigger = compute_trigger(
                scores[detected_idx], baseline_match, self.sim_threshold,
                self.margin_threshold, self.threshold_distance, self.threshold_scope,
            )
            support_check = support_checks.get(detected_idx)
            mask_change = bool(support_check and support_check["triggered"])
            review_new = self.review_all_new and baseline_match is None
            if raw_trigger is None and not mask_change and not review_new:
                continue
            self.stats["raw_triggered_before_iou"] += int(raw_trigger is not None)
            max_ranked = min(max(self.association_top_k, self.create_top_k), len(ALIASES))
            if self.candidate_iou_filter_enabled:
                ranked, iou_filter = deduplicate_ranked_candidates(
                    scores[detected_idx], objects, self.candidate_iou_threshold, max_ranked,
                )
            else:
                ranked = [
                    int(item) for item in np.argsort(-scores[detected_idx], kind="stable")
                    if np.isfinite(scores[detected_idx, item])
                ][:max_ranked]
                iou_filter = {
                    "method": "disabled", "iou_threshold": self.candidate_iou_threshold,
                    "ranked_before_top": ranked, "kept_object_indices": ranked,
                    "dropped": [], "comparisons": 0, "search_exhausted": False,
                }
            for item in iou_filter["dropped"]:
                item["object_uid"] = str(objects[item["object_index"]].get("id"))
                item["representative_object_uid"] = str(objects[item["representative_object_index"]].get("id"))
            filtered_scores = [float(scores[detected_idx, obj_idx]) for obj_idx in ranked]
            trigger = compute_trigger(
                filtered_scores, baseline_match, self.sim_threshold,
                self.margin_threshold, self.threshold_distance, self.threshold_scope,
            )
            score_suppressed = raw_trigger is not None and trigger is None
            if score_suppressed:
                self.stats["suppressed_by_iou_prefilter"] += 1
                self.stats["iou_candidates_dropped"] += len(iou_filter["dropped"])
                obs_uids = list(detection.get("obs_uids", []))
                _jsonl_append(self.iou_prefilter_path, {
                    "schema_version": SCHEMA_VERSION,
                    "frame_idx": frame_idx,
                    "source_frame_id": source_frame_id,
                    "detected_obj_idx": detected_idx,
                    "current_observation_uid": str(obs_uids[0]) if obs_uids else None,
                    "baseline_match_index": baseline_match,
                    "raw_trigger_before_iou": raw_trigger,
                    "filtered_trigger": None,
                    "candidate_iou_prefilter_hidden_from_vlm": iou_filter,
                    "outcome": "score_trigger_suppressed_event_retained" if mask_change or review_new else "trigger_suppressed",
                })
            reasons = []
            if trigger is not None:
                reasons.append("association_margin" if "margin" in trigger else "score_threshold_distance")
            if mask_change:
                reasons.append("mask_change")
            # Supplemental NEW review only when the existing score gate did not fire.
            if trigger is None and review_new:
                reasons.append("all_new")
            if not reasons:
                continue
            trigger = dict(trigger or {"kind": "create" if baseline_match is None else "association"})
            trigger["reasons"] = reasons
            excluded_from_history.add(detected_idx)
            for reason in reasons:
                self.stats[f"trigger_reason_{reason}"] += 1
            self.stats["triggered"] += 1
            self.stats[f"triggered_{trigger['kind']}"] += 1
            self.stats["events_with_iou_drops"] += int(bool(iou_filter["dropped"]))
            if not score_suppressed:
                self.stats["iou_candidates_dropped"] += len(iou_filter["dropped"])
            if self.max_events > 0 and self.stats["processed"] >= self.max_events:
                self.stats["suppressed_by_max_events"] += 1
                continue
            top_k = self.association_top_k if trigger["kind"] == "association" else self.create_top_k
            ranked = ranked[: min(top_k, len(ALIASES))]
            if trigger["kind"] == "association" and len(ranked) < 2 and not mask_change:
                self.stats["skipped_insufficient_candidates"] += 1
                continue
            candidates = [(ALIASES[pos], obj_idx, objects[obj_idx]) for pos, obj_idx in enumerate(ranked)]
            aliases_to_indices = {alias: obj_idx for alias, obj_idx, _ in candidates}
            event_id = f"f{frame_idx:06d}_d{detected_idx:03d}_{trigger['kind']}"
            event_dir = self.output_dir / "events" / event_id
            event_dir.mkdir(parents=True, exist_ok=False)
            h_time = _utc_now()
            evidence_manifest, request_images = self._event_evidence(event_dir, image_rgb, detection, candidates)
            system_prompt, user_prompt = self._prompts(trigger["kind"], list(aliases_to_indices))
            (event_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
            (event_dir / "user_prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
            snapshot_payload = {
                "frame_idx": frame_idx,
                "objects": [
                    {"alias": alias, "index": obj_idx, "uid": str(obj.get("id")), "num_detections": int(obj.get("num_detections", 0))}
                    for alias, obj_idx, obj in candidates
                ],
            }
            snapshot_uid = hashlib.sha256(json.dumps(snapshot_payload, sort_keys=True).encode("utf-8")).hexdigest()
            human_case_id = self._write_human_annotation_case(
                event_id=event_id,
                snapshot_uid=snapshot_uid,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=request_images,
                aliases=list(aliases_to_indices),
            )
            neutral_assessments = [
                {"code": alias, "relation": "UNCERTAIN", "evidence": "not adjudicated"}
                for alias in aliases_to_indices
            ]
            output = {
                "candidate_assessments": neutral_assessments,
                "choice": "UNCERTAIN",
                "confidence": 0.0,
                "reason": "audit mode: no adjudication requested",
            }
            decision_source, raw_response, error, latency_seconds, oracle_diagnostics = self.mode, None, None, 0.0, None
            adjudication_started = time.perf_counter()
            try:
                if self.mode == "vlm":
                    payload = self._request_payload(system_prompt, user_prompt, request_images, list(aliases_to_indices))
                    _json_dump(event_dir / "request_media_manifest.json", [
                        {"label": label, **_image_media_descriptor(path)} for label, path in request_images
                    ])
                    _json_dump(event_dir / "actual_request_redacted.json", self._redact_payload(payload, request_images))
                    _json_dump(event_dir / "request_transport.json", {
                        "client": "openai-python",
                        "transport": "httpx.Client(trust_env=False)",
                        "endpoint": f"{self.base_url}/chat/completions",
                        "request_body_utf8_bytes": len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
                        "retry_policy": {"same_key_serial_attempts": self.max_retries + 1, "multi_key_failover": False},
                    })
                    raw_response, output, latency_seconds = self._call_vlm(payload, failure_log_path=event_dir / "vlm_error.json")
                    _json_dump(event_dir / "vlm_raw_response.json", raw_response)
                    decision_source = "vlm"
                elif self.mode == "human":
                    _json_dump(event_dir / "actual_request_redacted.json", {
                        "not_sent": True,
                        "mode": "human",
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "allowed_choices": list(aliases_to_indices) + ["NEW", "UNCERTAIN", "DISCARD"],
                        "images": [
                            {"label": label, "path": path.name, "sha256": _sha256_file(path)}
                            for label, path in request_images
                        ],
                    })
                    output = self._call_human(
                        event_id=event_id,
                        event_dir=event_dir,
                        aliases=list(aliases_to_indices),
                    )
                    latency_seconds = time.perf_counter() - adjudication_started
                    decision_source = "human"
                elif self.mode == "oracle":
                    choice, oracle_diagnostics = self._oracle_choice(frame_idx, detection, candidates)
                    output = {
                        "candidate_assessments": [
                            {
                                "code": alias,
                                "relation": "SAME" if alias == choice else "DIFFERENT",
                                "evidence": "GT sidecar oracle",
                            }
                            for alias in aliases_to_indices
                        ],
                        "choice": choice,
                        "confidence": 1.0 if choice != "UNCERTAIN" else 0.0,
                        "reason": "GT sidecar oracle",
                    }
                    decision_source = "oracle"
                    _json_dump(event_dir / "actual_request_redacted.json", {
                        "not_sent": True,
                        "mode": "oracle",
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "images": [{"label": label, "path": path.name, "sha256": _sha256_file(path)} for label, path in request_images],
                    })
                else:
                    _json_dump(event_dir / "actual_request_redacted.json", {
                        "not_sent": True,
                        "mode": "audit",
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "images": [{"label": label, "path": path.name, "sha256": _sha256_file(path)} for label, path in request_images],
                    })
                allowed = set(aliases_to_indices) | {"NEW", "UNCERTAIN"}
                if self.mode == "human":
                    allowed.add("DISCARD")
                if str(output.get("choice", "")).upper() not in allowed:
                    raise ValueError(f"invalid choice: {output.get('choice')}")
                if self.mode == "vlm":
                    assessments = list(output.get("candidate_assessments") or [])
                    assessment_codes = [str(item.get("code", "")).upper() for item in assessments]
                    if len(assessment_codes) != len(aliases_to_indices) or set(assessment_codes) != set(aliases_to_indices):
                        raise ValueError(f"candidate assessments do not cover aliases exactly once: {assessment_codes}")
                    invalid_relations = [
                        item.get("relation") for item in assessments
                        if str(item.get("relation", "")).upper() not in {"SAME", "DIFFERENT", "UNCERTAIN"}
                    ]
                    if invalid_relations:
                        raise ValueError(f"invalid candidate assessment relations: {invalid_relations}")
                confidence = float(output.get("confidence", 0.0))
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"invalid confidence: {confidence}")
            except HumanInputUnavailableError:
                self._write_summary(status="waiting_for_human_input")
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                latency_seconds = time.perf_counter() - adjudication_started
                output = {
                    "candidate_assessments": neutral_assessments,
                    "choice": "UNCERTAIN",
                    "confidence": 0.0,
                    "reason": "invalid response or API failure; baseline fallback",
                }
                decision_source = f"{decision_source}_failure"
                self.stats["failures"] += 1
            output_file = "human_output.json" if self.mode == "human" else "vlm_output.json"
            _json_dump(event_dir / output_file, output)
            if oracle_diagnostics is not None:
                _json_dump(event_dir / "oracle_diagnostics.json", oracle_diagnostics)
            if self.mode in {"oracle", "vlm", "human"}:
                final_match, route_reason = route_choice(output["choice"], aliases_to_indices, baseline_match)
            else:
                final_match, route_reason = baseline_match, "audit_keeps_baseline"
            final_matches[detected_idx] = final_match
            changed = final_match != baseline_match
            obs_uids = list(detection.get("obs_uids", []))
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "mode": self.mode,
                "trigger": trigger,
                "timeline": {
                    "s_frame": frame_idx, "d_frame": frame_idx, "h_frame": frame_idx, "c_frame": frame_idx,
                    "h_utc": h_time, "c_utc": _utc_now(),
                    "online_main_graph_latest_frame_at_h": frame_idx, "ordering_valid": True,
                },
                "source_frame_id": source_frame_id,
                "detected_obj_idx": detected_idx,
                "current_observation_uid": str(obs_uids[0]) if obs_uids else None,
                "h_snapshot_uid": snapshot_uid,
                "c_bound_h_snapshot_uid": snapshot_uid,
                "candidate_alias_to_object_index": aliases_to_indices,
                "candidate_iou_prefilter_hidden_from_vlm": iou_filter,
                "raw_trigger_before_iou": raw_trigger,
                "spatial_support_hidden_from_reviewer": support_check,
                "audit_scores_hidden_from_vlm": {
                    "sim_threshold": self.sim_threshold,
                    "margin_threshold": self.margin_threshold,
                    "threshold_distance": self.threshold_distance,
                    "candidate_scores": {
                        alias: float(scores[detected_idx, obj_idx])
                        for alias, obj_idx in aliases_to_indices.items()
                    },
                },
                "candidate_object_uids_distinct": len({str(obj.get("id")) for _, _, obj in candidates}) == len(candidates),
                "evidence": evidence_manifest,
                "prompt_files": {"system": "system_prompt.txt", "user": "user_prompt.txt"},
                "human_annotation_case_id": human_case_id,
                "baseline_match_index": baseline_match,
                "decision_source": decision_source,
                "output_file": output_file,
                "model_output": output,
                "final_match_index": final_match,
                "route_reason": route_reason,
                "changed": changed,
                "latency_seconds": latency_seconds,
                "error": error,
            }
            _json_dump(event_dir / "decision.json", event)
            _jsonl_append(self.events_path, event)
            self.events.append(event)
            self.stats["processed"] += 1
            self.stats["changed"] += int(changed)
            self.stats[f"choice_{str(output['choice']).upper()}"] += 1
            print(
                f"[association-gate] {event_id} mode={self.mode} trigger={trigger['kind']} "
                f"choice={output['choice']} baseline={baseline_match} final={final_match} "
                f"changed={changed} latency={latency_seconds:.2f}s",
                flush=True,
            )
            self._log_rerun(event_dir, event)
            self._write_summary(status="running")
            self._write_index()
        # Only untriggered baseline associations enter the next frame's reference.
        # Triggered/limited/discarded/overridden observations never train this window.
        for index, check in support_checks.items():
            recorded = index not in excluded_from_history
            if recorded:
                self._support_history.setdefault(check["object_uid"], deque(maxlen=self.support_window)).append(
                    (int(frame_idx), check["current"])
                )
                self.stats["support_history_samples"] += 1
            obs_uids = list(detection_list[index].get("obs_uids", []))
            _jsonl_append(self.support_path, {
                "frame_idx": int(frame_idx), "source_frame_id": source_frame_id,
                "current_observation_uid": str(obs_uids[0]) if obs_uids else None,
                "detected_obj_idx": index, **check, "history_updated": recorded,
                "excluded_by_gate_trigger": index in excluded_from_history,
            })
        if self.mask_change_enabled:
            self._last_support_frame = int(frame_idx)
        return final_matches

    def _log_rerun(self, event_dir: Path, event: dict) -> None:
        if self.rerun is None:
            return
        try:
            self.rerun.log("association_gate/current_context", self.rerun.ImageEncoded(path=str(event_dir / "current_context.jpg")))
            self.rerun.log("association_gate/current_crop", self.rerun.ImageEncoded(path=str(event_dir / "current_crop.jpg")))
            for alias in event["candidate_alias_to_object_index"]:
                self.rerun.log(f"association_gate/candidate_{alias}", self.rerun.ImageEncoded(path=str(event_dir / f"candidate_{alias}.jpg")))
            text = json.dumps(event["model_output"], ensure_ascii=False)
            self.rerun.log("association_gate/decision", self.rerun.TextDocument(text, media_type="text/markdown"))
        except Exception as exc:
            self.stats["rerun_log_failures"] += 1
            print(f"[association-gate] rerun logging failed: {exc}", flush=True)

    def _write_summary(self, status: str) -> None:
        _json_dump(self.output_dir / "summary.json", {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "config": self.config,
            "stats": dict(self.stats),
            "events_jsonl": self.events_path.name,
            "iou_prefilter_jsonl": self.iou_prefilter_path.name,
            "dashboard": "index.html",
            "human_annotation_blind": "human_annotation_blind/index.html",
            "human_annotation_private_map": self.annotation_map_path.name,
        })

    def _write_index(self) -> None:
        cards = []
        for event in reversed(self.events):
            event_id = event["event_id"]
            images = [
                f'<img src="events/{event_id}/current_context.jpg" alt="current context">',
                f'<img src="events/{event_id}/current_crop.jpg" alt="current crop">',
            ]
            images.extend(
                f'<img src="events/{event_id}/candidate_{alias}.jpg" alt="candidate {alias}">'
                for alias in event["candidate_alias_to_object_index"]
            )
            cards.append(
                f'<section><h2>{html.escape(event_id)}</h2><p>mode={html.escape(event["mode"])} | '
                f'trigger={html.escape(event["trigger"]["kind"])} | baseline={event["baseline_match_index"]} | '
                f'choice={html.escape(str(event["model_output"]["choice"]))} | final={event["final_match_index"]} | '
                f'changed={event["changed"]} | latency={event["latency_seconds"]:.2f}s</p>'
                f'<div class="images">{"".join(images)}</div>'
                f'<pre>{html.escape(json.dumps(event["model_output"], ensure_ascii=False, indent=2))}</pre>'
                f'<p><a href="events/{event_id}/actual_request_redacted.json">redacted input</a> · '
                f'<a href="events/{event_id}/{html.escape(event.get("output_file", "vlm_output.json"))}">parsed output</a> · '
                f'<a href="events/{event_id}/decision.json">decision</a></p></section>'
            )
        document = f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>Blocking association gate</title><style>body{{font-family:system-ui;margin:20px;background:#111;color:#eee}}section{{border:1px solid #555;padding:14px;margin:14px 0}}.images{{display:flex;gap:8px;flex-wrap:wrap}}img{{max-width:300px;max-height:240px;object-fit:contain;background:#222}}a{{color:#7dd3fc}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>Blocking association gate: {html.escape(self.mode)}</h1><p>Auto-refresh: 5 s · processed={self.stats['processed']} · changed={self.stats['changed']} · failures={self.stats['failures']}</p>{''.join(cards)}</body></html>"""
        (self.output_dir / "index.html").write_text(document, encoding="utf-8")

    def close(self, *, status: str = "completed") -> None:
        self._write_summary(status=status)
        self._write_index()
        self._write_human_annotation_index()
