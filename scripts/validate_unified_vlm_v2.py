#!/usr/bin/env python3
"""Validate declarative ``object_state_v2`` on frozen online evidence."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import html
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scripts.validate_unified_vlm_v1 import (
    FrozenRun,
    PreflightDefer,
    bbox_from_mask,
    clamp,
    event_sequence,
    frame_index,
    get_font,
    normalize_label,
    safe_error,
    sha256_file,
    sharpness_score,
    write_json,
)


PROMPT_VERSION = "ali_my_object_state_v2_v3_9_timeline_20260829"
ALIAS_COLORS = {
    "A": (235, 52, 64),
    "E0": (42, 111, 230),
    "E1": (38, 166, 91),
}
MISSING_EVIDENCE = (
    "NONE",
    "EVENT_EVIDENCE_UNCLEAR",
    "CURRENT_OBJECT_UNCLEAR",
    "ALTERNATIVE_OBJECT_UNCLEAR",
    "WIDER_CONTEXT_NEEDED",
    "LABEL_NOT_LISTED",
    "COMPOUND_STATE_REQUIRED",
)

SYSTEM_PROMPT = """You are reviewing one object association in an online 3D map.

A is the observation being reviewed.
E0 is the current map entity that owns A, shown again in its latest available
state.
E1 is the strongest alternative map entity, when one is available. E0 and E1
are map identities, so they can be duplicate representations of one physical
object; do not assume they are physically distinct merely because both exist.

I1 is tied to this review issue's S event. Red marks A and blue marks E0. It
prefers exact S; otherwise it may use the nearest causally available real frame
not later than D only when A and E0 both have real masks in one physical
frame. It never combines different frames. The case facts state the exact I1
frame and its offset from S. If no valid shared frame exists, I1 is A-only at S.

I2 shows the best available view of the latest E0.

I3 shows E1 in green. If no valid E1 exists, I3 shows wider scene context around
A or E0 instead.

Mask colors are annotations only. They are not the physical colors of objects
and are not ground truth.

In a same-frame I1, compare the red A and blue E0 contours directly. If they
trace the same visible object surface or one contour is simply the visible
subregion of the other, choose E0. Do not infer two objects from two annotation
colors, a small boundary offset, or a partial crop.

Observation masks can cover only a visible fragment or subregion of one object,
while E0 can cover more of that same object in the same or a later view. Mask
containment, partial overlap, or different mask extent alone does not make A a
separate physical object. Choose SEPARATE only when the evidence clearly shows
two distinct physical instances.

Choose the map entity that should own A. Choose E1 when A matches the recorded
alternative and E0 is a duplicate/fragment map identity that should be
consolidated into E1. Choose E0 when A belongs with E0 and E1 is a different
physical instance. Choose SEPARATE only when A is distinct from all displayed
entities; choose UNRESOLVED when the supplied evidence cannot distinguish the
valid owners.
The current A→E0 assignment is review context, not supporting evidence. Never
choose E0 merely to preserve that assignment. In particular, when I1 is A-only
and I2/I3 leave E0 and E1 as indistinguishable duplicate views, choose
UNRESOLVED with EVENT_EVIDENCE_UNCLEAR.
Only when A belongs to E0, choose the best semantic label for E0.
For every non-E0 identity_target (E1, SEPARATE, or UNRESOLVED), semantic_target
MUST be NOT_APPLICABLE. Never assign an L-label to A, E1, or a new/separate
entity; the L-label candidates are exclusively possible replacement labels for
the existing E0.

If identity_target is E0, E1, or SEPARATE, missing_evidence must be NONE. If
required evidence is missing, use identity_target UNRESOLVED instead.

Write reason as one complete sentence of at most 180 characters and end it with
sentence punctuation. Return only JSON that matches the schema supplied by the
caller."""


@dataclass(frozen=True)
class PreparedObjectStateCase:
    ticket_uid: str
    case_dir: Path
    schema: dict[str, Any]
    available_identity_targets: tuple[str, ...]
    available_semantic_targets: tuple[str, ...]
    message_sequence: tuple[dict[str, str], ...]


def _accepted_rows(
    run: FrozenRun, version: Mapping[str, Any], freeze_frame: int
) -> list[dict[str, Any]]:
    rows = []
    for obs_uid in version.get("member_observation_uids") or ():
        row = run.observations.get(str(obs_uid))
        if (
            row
            and row.get("status") == "kept"
            and row.get("processed_mask_ref")
            and frame_index(row.get("frame_uid") or obs_uid) <= freeze_frame
        ):
            rows.append(row)
    return rows


def _expanded_bbox(mask: np.ndarray, margin: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_from_mask(mask)
    width, height = x1 - x0, y1 - y0
    mx = max(8, round(width * margin))
    my = max(8, round(height * margin))
    left = max(0, x0 - mx)
    top = max(0, y0 - my)
    right = min(mask.shape[1], x1 + mx)
    bottom = min(mask.shape[0], y1 + my)

    # Extremely thin masks otherwise produce ribbon-like crops that hide the
    # object and scene context. Keep at least 42% of each source dimension;
    # this changes only the crop, never the real mask or evidence frame.
    min_width = min(mask.shape[1], max(1, round(mask.shape[1] * 0.42)))
    min_height = min(mask.shape[0], max(1, round(mask.shape[0] * 0.42)))

    def expand_axis(low: int, high: int, limit: int, minimum: int) -> tuple[int, int]:
        if high - low >= minimum:
            return low, high
        center = (low + high) / 2.0
        low = max(0, round(center - minimum / 2.0))
        high = min(limit, low + minimum)
        low = max(0, high - minimum)
        return low, high

    left, right = expand_axis(left, right, mask.shape[1], min_width)
    top, bottom = expand_axis(top, bottom, mask.shape[0], min_height)
    return left, top, right, bottom


def _view(
    run: FrozenRun,
    frame: int,
    alias_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    margin: float,
) -> dict[str, Any]:
    rgb = run.load_rgb(frame)
    masks: dict[str, np.ndarray] = {}
    normalized_rows: dict[str, list[dict[str, Any]]] = {}
    for alias in ("A", "E0", "E1"):
        rows = [dict(row) for row in alias_rows.get(alias, ())]
        if not rows:
            continue
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for row in rows:
            value = run.load_mask(row)
            if value.shape != mask.shape:
                raise PreflightDefer(
                    "DEFER_MISSING_DECISION_PROVENANCE",
                    f"mask shape mismatch for {row.get('obs_uid')}",
                )
            mask |= value
        if mask.any():
            masks[alias] = mask
            normalized_rows[alias] = rows
    if not masks:
        raise PreflightDefer(
            "DEFER_MISSING_DECISION_PROVENANCE", "view has no accepted processed mask"
        )
    total = np.zeros(rgb.shape[:2], dtype=bool)
    for mask in masks.values():
        total |= mask
    crop_box = _expanded_bbox(total, margin)
    x0, y0, x1, y1 = crop_box
    crop = rgb[y0:y1, x0:x1]
    gray = np.dot(crop[..., :3], np.array([0.299, 0.587, 0.114]))
    area_score = clamp(math.sqrt(float(total.sum()) / max(1.0, total.size * 0.12)))
    sharpness = sharpness_score(gray)
    mean, std = float(gray.mean()) / 255.0, float(gray.std()) / 255.0
    exposure = 0.70 * clamp(1.0 - abs(mean - 0.5) / 0.5) + 0.30 * clamp(std / 0.18)
    boundaries = [
        float(row.get("boundary_touch_ratio") or 0.0)
        for rows in normalized_rows.values()
        for row in rows
    ]
    non_occlusion = clamp(1.0 - sum(boundaries) / max(1, len(boundaries)))
    bx0, by0, bx1, by1 = bbox_from_mask(total)
    boundary_margin = clamp(
        4.0
        * min(
            bx0 / max(1, rgb.shape[1]),
            by0 / max(1, rgb.shape[0]),
            (rgb.shape[1] - bx1) / max(1, rgb.shape[1]),
            (rgb.shape[0] - by1) / max(1, rgb.shape[0]),
        )
    )
    quality = (
        0.35 * area_score
        + 0.25 * sharpness
        + 0.20 * non_occlusion
        + 0.10 * exposure
        + 0.10 * boundary_margin
    )
    scale = min(1.0, 1280.0 / max(1, x1 - x0, y1 - y0))
    alias_short_sides = {
        alias: min(
            bbox_from_mask(mask)[2] - bbox_from_mask(mask)[0],
            bbox_from_mask(mask)[3] - bbox_from_mask(mask)[1],
        ) * scale
        for alias, mask in masks.items()
    }
    target_alias = "A" if "A" in alias_short_sides else next(iter(alias_short_sides))
    short_side = alias_short_sides[target_alias]
    return {
        "frame": int(frame),
        "alias_rows": normalized_rows,
        "masks": masks,
        "crop_box": crop_box,
        "margin": float(margin),
        "visible_aliases": tuple(alias for alias in ("A", "E0", "E1") if alias in masks),
        "quality": float(quality),
        "quality_terms": {
            "mask_area_score": area_score,
            "sharpness": sharpness,
            "non_occlusion": non_occlusion,
            "exposure_quality": exposure,
            "boundary_margin": boundary_margin,
        },
        "target_mask_short_side_px": float(short_side),
        "alias_mask_short_sides_px": alias_short_sides,
        "eligible": bool(short_side >= 96 and exposure >= 0.15 and non_occlusion >= 0.20),
    }


def _render_image(run: FrozenRun, view: Mapping[str, Any], role: str) -> Image.Image:
    rgb = run.load_rgb(int(view["frame"]))
    array = rgb.copy()
    # Preserve the real texture/color for VLM identity comparison.  The thick,
    # distinct outlines carry alias identity; fills are only a light aid for
    # locating the mask and must not make a white object look physically red.
    fill_alpha = {"A": 0.10, "E0": 0.08, "E1": 0.08}
    for alias in ("E1", "E0", "A"):
        mask = view["masks"].get(alias)
        if mask is None:
            continue
        color = np.asarray(ALIAS_COLORS[alias], dtype=np.float32)
        alpha = fill_alpha[alias]
        array[mask] = np.clip(
            (1.0 - alpha) * array[mask].astype(np.float32) + alpha * color, 0, 255
        ).astype(np.uint8)
    canvas = Image.fromarray(array)
    for alias in ("E1", "E0", "A"):
        mask = view["masks"].get(alias)
        if mask is None:
            continue
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        if alias == "E0":
            # An exterior cyan halo remains visible even where A overlaps E0.
            outline = (
                np.asarray(mask_image.filter(ImageFilter.MaxFilter(11))) > 0
            ) & ~mask
        else:
            outline = (
                np.asarray(mask_image.filter(ImageFilter.MaxFilter(7))) > 0
            ) & ~(
                np.asarray(mask_image.filter(ImageFilter.MinFilter(7))) > 0
            )
        outlined = np.asarray(canvas).copy()
        outlined[outline] = np.asarray(ALIAS_COLORS[alias], dtype=np.uint8)
        canvas = Image.fromarray(outlined)
    canvas = canvas.crop(tuple(int(value) for value in view["crop_box"]))
    if max(canvas.size) > 1280:
        scale = 1280.0 / max(canvas.size)
        canvas = canvas.resize(
            (max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale))),
            Image.Resampling.LANCZOS,
        )
    # Keep every text label outside the RGB crop so it cannot hide evidence.
    font_size = max(9, min(14, round(min(canvas.size) * 0.045)))
    font = get_font(font_size)
    pad = max(3, font_size // 3)
    probe = ImageDraw.Draw(canvas)
    role_box = probe.textbbox((0, 0), role, font=font)
    token_boxes = {
        alias: probe.textbbox((0, 0), alias, font=font)
        for alias in view["visible_aliases"]
    }
    header_height = max(
        role_box[3] - role_box[1],
        *(box[3] - box[1] for box in token_boxes.values()),
    ) + 2 * pad
    header_width = (
        pad
        + role_box[2] - role_box[0]
        + 2 * pad
        + sum(box[2] - box[0] + 3 * pad for box in token_boxes.values())
    )
    framed_width = max(canvas.width, header_width)
    framed = Image.new("RGB", (framed_width, canvas.height + header_height), (18, 18, 18))
    framed.paste(canvas, ((framed_width - canvas.width) // 2, header_height))
    draw = ImageDraw.Draw(framed)
    x = pad
    draw.text((x, pad), role, fill=(255, 255, 255), font=font)
    x += role_box[2] - role_box[0] + 2 * pad
    for alias in view["visible_aliases"]:
        box = token_boxes[alias]
        width = box[2] - box[0]
        draw.rounded_rectangle(
            (x, pad - 1, x + width + 2 * pad, header_height - pad + 1),
            radius=max(2, pad),
            fill=ALIAS_COLORS[alias],
        )
        draw.text((x + pad, pad), alias, fill=(0, 0, 0), font=font)
        x += width + 3 * pad
    return framed


def _shared_event_frame(
    anchor_rows: Iterable[Mapping[str, Any]],
    core_rows: Iterable[Mapping[str, Any]],
    s_frame: int,
    d_frame: int,
) -> int | None:
    """Prefer exact S, else the nearest same-frame A/E0 evidence no later than D."""

    anchor_frames = {
        frame_index(row.get("frame_uid") or row["obs_uid"]) for row in anchor_rows
    }
    core_frames = {
        frame_index(row.get("frame_uid") or row["obs_uid"]) for row in core_rows
    }
    shared = {
        frame for frame in anchor_frames & core_frames
        if frame >= 0 and frame <= int(d_frame)
    }
    return min(shared, key=lambda frame: (frame != int(s_frame), abs(frame - int(s_frame)), frame)) if shared else None


def _h_snapshot_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": snapshot.get("schema_version"),
        "cutoff_frame": int(snapshot.get("cutoff_frame", -1)),
        "cutoff_sequence": int(snapshot.get("cutoff_sequence", -1)),
        "active_object_version_uids": dict(
            sorted((snapshot.get("active_object_version_uids") or {}).items())
        ),
        "merge_redirects": dict(sorted((snapshot.get("merge_redirects") or {}).items())),
    }


def _validate_h_snapshot(packet: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = packet.get("h_snapshot")
    if not isinstance(snapshot, Mapping):
        raise PreflightDefer("DEFER_H_SNAPSHOT_MISSING", "packet has no immutable H snapshot")
    payload = _h_snapshot_payload(snapshot)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if snapshot.get("snapshot_sha256") != digest:
        raise PreflightDefer("DEFER_H_SNAPSHOT_HASH_MISMATCH", "H snapshot digest is invalid")
    if snapshot.get("snapshot_uid") != "hsnap_" + digest[:16]:
        raise PreflightDefer("DEFER_H_SNAPSHOT_UID_MISMATCH", "H snapshot UID is invalid")
    return dict(snapshot)


def _candidate_event_context(
    packet: Mapping[str, Any], observations: Mapping[str, Mapping[str, Any]]
) -> list[tuple[str, dict[str, Any]]]:
    """Use only the resolver-bound distinct E1; never enumerate raw candidates."""

    available = set((packet.get("alias_version_uids") or {}).keys())
    alias_refs = packet.get("candidate_alias_observation_uids") or {}
    sources: list[tuple[str, dict[str, Any]]] = []
    for alias in ("E1",):
        if alias not in available:
            continue
        for uid in alias_refs.get(alias) or ():
            row = observations.get(str(uid))
            if row and row.get("status") == "kept" and row.get("processed_mask_ref"):
                sources.append((alias, dict(row)))
    return sources


def _save_image(image: Image.Image, path: Path) -> dict[str, Any]:
    original_size = image.size
    if max(image.size) < 384:
        scale = 384.0 / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    image.save(path, quality=90, subsampling=0)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "width": image.width,
        "height": image.height,
        "source_width": original_size[0],
        "source_height": original_size[1],
        "display_upscaled": image.size != original_size,
    }


def _best_row_view(
    run: FrozenRun,
    rows: Iterable[Mapping[str, Any]],
    alias: str,
    *,
    margin: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = []
    for row in rows:
        try:
            view = _view(
                run,
                frame_index(row.get("frame_uid") or row.get("obs_uid")),
                {alias: [row]},
                margin=margin,
            )
        except (FileNotFoundError, ValueError, PreflightDefer):
            continue
        candidates.append((view, dict(row)))
    return max(
        candidates,
        key=lambda item: (
            item[0]["quality"],
            item[0]["target_mask_short_side_px"],
            -item[0]["frame"],
        ),
        default=None,
    )


def _semantic_labels(
    rows: Iterable[Mapping[str, Any]], current_label: str
) -> list[dict[str, Any]]:
    counts = Counter(
        normalize_label(row.get("class_name")) for row in rows
        if normalize_label(row.get("class_name"))
        and normalize_label(row.get("class_name")) != current_label
        and normalize_label(row.get("class_name")) not in {"wall", "floor", "ceiling"}
    )
    return [
        {
            "id": f"L{index}",
            "text": label,
            "source": "accepted U/E0 observation label; machine hypothesis",
        }
        for index, (label, _) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3], 1
        )
    ]


def _review_question(
    issue_family: str,
    issue: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    has_e1: bool,
) -> str:
    """Describe the suspected event without presenting it as ground truth."""

    predicate = str(contract.get("repair_predicate") or "")
    decision = str((issue.get("raw_signals") or {}).get("decision") or "")
    if predicate == "JOIN_CANDIDATE" and has_e1:
        event = "created/kept E0" if decision == "CREATE_OBJECT" else "assigned A to E0"
        return (
            f"The online mapper {event} while E1 was a recorded existing candidate. "
            "This is only a suspicion, not ground truth. Use the images to decide "
            "whether A should remain with E0 or be consolidated into E1. If A, E0, "
            "and E1 depict the same physical instance, choose the pre-existing E1 "
            "instead of retaining duplicate E0; if E1 is a different instance, keep E0."
        )
    if predicate == "SEPARATE_FROM_CURRENT":
        return (
            "The coarse prefilter selected A→E0 for neutral identity review; this does "
            "not imply that the assignment is wrong. Decide from the images whether A "
            "belongs to E0, E1 when available, or is separate. In a same-frame I1, "
            "overlapping red/blue contours on the same physical surface support E0."
        )
    if predicate == "ADOPT_LABEL" or "SEMANTIC" in issue_family:
        return (
            "E0's current semantic label is under review. First verify that A belongs "
            "to E0; only then select the best listed label for E0."
        )
    return (
        "The current assignment A→E0 was selected by the coarse prefilter for review. "
        "Treat that suspicion as context rather than ground truth and decide from the "
        "supplied evidence."
    )


def output_schema(
    identity_targets: Iterable[str], semantic_targets: Iterable[str]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "identity_target",
            "semantic_target",
            "evidence_ids",
            "reason",
            "missing_evidence",
        ],
        "properties": {
            "identity_target": {"type": "string", "enum": list(identity_targets)},
            "semantic_target": {"type": "string", "enum": list(semantic_targets)},
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"type": "string", "enum": ["I1", "I2", "I3"]},
            },
            "reason": {"type": "string", "minLength": 1, "maxLength": 320},
            "missing_evidence": {"type": "string", "enum": list(MISSING_EVIDENCE)},
        },
    }


def validate_output(
    value: Any,
    available_identity_targets: Iterable[str],
    available_semantic_targets: Iterable[str],
) -> list[str]:
    required = {
        "identity_target",
        "semantic_target",
        "evidence_ids",
        "reason",
        "missing_evidence",
    }
    if not isinstance(value, dict):
        return ["output is not a JSON object"]
    errors = []
    if set(value) != required:
        errors.append(f"keys must equal {sorted(required)}")
    identity = value.get("identity_target")
    semantic = value.get("semantic_target")
    if identity not in set(available_identity_targets):
        errors.append("identity_target is unavailable in this frozen case")
    if semantic not in set(available_semantic_targets):
        errors.append("semantic_target is unavailable in this frozen case")
    if identity == "E0":
        if semantic == "NOT_APPLICABLE":
            errors.append("identity_target E0 requires semantic evaluation")
    elif semantic != "NOT_APPLICABLE":
        errors.append("non-E0 identity requires semantic_target NOT_APPLICABLE")
    evidence = value.get("evidence_ids")
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 3
        or len(evidence) != len(set(evidence))
        or any(item not in {"I1", "I2", "I3"} for item in evidence)
    ):
        errors.append("evidence_ids must contain 1-3 unique IDs from I1/I2/I3")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason or len(reason) > 320:
        errors.append("reason must be a nonempty string of at most 320 characters")
    elif reason.rstrip()[-1:] not in ".!?。！？":
        errors.append("reason must be a complete sentence ending in punctuation")
    missing = value.get("missing_evidence")
    if missing not in MISSING_EVIDENCE:
        errors.append("invalid missing_evidence")
    unresolved = identity == "UNRESOLVED" or semantic == "UNRESOLVED"
    if (missing != "NONE") != unresolved:
        errors.append("missing_evidence must be non-NONE iff one target is UNRESOLVED")
    return errors


def _spec(image_id: str, role: str, metadata: Mapping[str, Any]) -> str:
    return "IMAGE_SPEC\n" + json.dumps(
        {"image_id": image_id, "role": role, **dict(metadata)},
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + f"\nThe next image is {image_id} and only {image_id}."


def prepare_case(
    run: FrozenRun,
    ticket_uid: str,
    output_root: Path,
    *,
    packet_root: Path | None = None,
) -> PreparedObjectStateCase:
    packet_base = packet_root.resolve() if packet_root is not None else run.online_root
    packet_path = packet_base / "vlm" / ticket_uid / "evidence" / "packet_manifest.json"
    if not packet_path.is_file():
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "V2 packet manifest missing")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("output_contract_version") != "object_state_v2":
        raise PreflightDefer("DEFER_WRONG_CONTRACT", "packet is not object_state_v2")
    freeze_frame = int(packet["freeze_frame"])
    freeze_sequence = int(packet["freeze_sequence"])
    issue = packet["issue"]
    issue_family = str(issue.get("issue_type") or issue.get("family") or "UNKNOWN")
    contract = packet["repair_contract"]
    timeline = packet.get("timeline")
    if not isinstance(timeline, Mapping):
        raise PreflightDefer("DEFER_TIMELINE_MISSING", "packet has no S/D/H timeline")
    s_frame = int(timeline.get("s_frame", -1))
    d_frame = int(timeline.get("d_frame", -1))
    h_frame = int(timeline.get("h_frame", -1))
    if not (0 <= s_frame <= d_frame <= h_frame):
        raise PreflightDefer("DEFER_TIMELINE_ORDER_INVALID", "S <= D <= H is not satisfied")
    if h_frame != freeze_frame or int(timeline.get("h_sequence", -1)) != freeze_sequence:
        raise PreflightDefer("DEFER_TIMELINE_H_MISMATCH", "timeline H differs from packet freeze")
    if str(timeline.get("s_event_uid") or "") != str(issue.get("anchor_event_uid") or ""):
        raise PreflightDefer("DEFER_TIMELINE_S_ISSUE_MISMATCH", "S is not bound to review_issue")
    h_snapshot = _validate_h_snapshot(packet)
    if (
        int(h_snapshot.get("cutoff_frame", -1)) != h_frame
        or int(h_snapshot.get("cutoff_sequence", -1)) != freeze_sequence
        or h_snapshot.get("snapshot_uid") != timeline.get("h_snapshot_uid")
        or h_snapshot.get("snapshot_sha256") != timeline.get("h_snapshot_sha256")
    ):
        raise PreflightDefer("DEFER_TIMELINE_H_SNAPSHOT_MISMATCH", "timeline and H snapshot differ")
    review_uids = [str(uid) for uid in contract.get("review_unit_obs_uids") or ()]
    alias_version_uids = {
        str(alias): str(uid)
        for alias, uid in (packet.get("alias_version_uids") or {}).items()
    }
    alias_owner_uids = {
        str(alias): str(uid)
        for alias, uid in (packet.get("alias_owner_uids") or {}).items()
    }
    if set(alias_owner_uids) != set(alias_version_uids) or "E0" not in alias_owner_uids:
        raise PreflightDefer("DEFER_H_ALIAS_BINDING_INCOMPLETE", "H alias owner/version binding is incomplete")
    if alias_owner_uids.get("E1") == alias_owner_uids.get("E0"):
        raise PreflightDefer("DEFER_E0_E1_NOT_DISTINCT", "E0 and E1 resolve to the same object")
    active_at_h = h_snapshot.get("active_object_version_uids") or {}
    if any(
        active_at_h.get(owner_uid) != alias_version_uids.get(alias)
        for alias, owner_uid in alias_owner_uids.items()
    ):
        raise PreflightDefer("DEFER_H_ALIAS_NOT_LATEST", "I2/I3 alias is not latest in H snapshot")
    versions = {
        alias: run.versions.get(uid) for alias, uid in alias_version_uids.items()
    }
    if not versions.get("E0") or versions["E0"].get("status") != "active":
        raise PreflightDefer("DEFER_CURRENT_OWNER_UNBOUND", "E0 active version is unavailable")
    e0_rows = _accepted_rows(run, versions["E0"], freeze_frame)
    anchor_rows = [
        run.observations[uid] for uid in review_uids
        if uid in run.observations
        and run.observations[uid].get("status") == "kept"
        and run.observations[uid].get("processed_mask_ref")
    ]
    if not anchor_rows:
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "review unit has no accepted mask")
    core_uid_set = {
        str(uid) for uid in contract.get("event_owner_core_obs_uids") or ()
    }
    event_e0_rows = [
        run.observations[uid]
        for uid in core_uid_set
        if uid in run.observations
        and run.observations[uid].get("status") == "kept"
        and run.observations[uid].get("processed_mask_ref")
    ]
    i1_frame = _shared_event_frame(anchor_rows, event_e0_rows, s_frame, d_frame)
    if i1_frame is not None:
        frame_anchor_rows = [
            row for row in anchor_rows
            if frame_index(row.get("frame_uid") or row["obs_uid"]) == i1_frame
        ]
        frame_e0_rows = [
            row for row in event_e0_rows
            if frame_index(row.get("frame_uid") or row["obs_uid"]) == i1_frame
        ]
        anchor = min(frame_anchor_rows, key=lambda row: str(row["obs_uid"]))
        i1 = _view(
            run,
            i1_frame,
            {"A": frame_anchor_rows, "E0": frame_e0_rows},
            margin=0.35,
        )
        i1_source_relation = "EXACT_S" if i1_frame == s_frame else "NEAREST_SHARED_CAUSAL"
        primary_image = _render_image(
            run,
            i1,
            "I1 - Exact S event" if i1_frame == s_frame else "I1 - Nearest same-frame A + E0 evidence",
        )
        i1_layout = "same_frame"
        i1_frames = [i1_frame]
        i1_visible_aliases = list(i1["visible_aliases"])
    else:
        exact_s_anchor_rows = [
            row for row in anchor_rows
            if frame_index(row.get("frame_uid") or row["obs_uid"]) == s_frame
        ]
        selected_anchor = _best_row_view(run, exact_s_anchor_rows, "A", margin=0.35)
        if selected_anchor is None:
            raise PreflightDefer(
                "DEFER_I1_SOURCE_AT_S_MISSING",
                "no same-frame A/E0 evidence and exact-S A cannot be recovered",
            )
        i1, anchor = selected_anchor
        i1_frame = int(i1["frame"])
        primary_image = _render_image(
            run, i1, "I1 - Exact S | A-only; E0-at-S unavailable"
        )
        i1_source_relation = "EXACT_S_A_ONLY"
        i1_layout = "a_only"
        i1_frames = [i1_frame]
        i1_visible_aliases = ["A"]
    i1_small_view = bool(i1["target_mask_short_side_px"] < 96)
    i1_quality_status = "IMAGE_DEGRADED" if i1_small_view else "PASS"

    anchor_uid = str(anchor["obs_uid"])
    i2_pool = [row for row in e0_rows if str(row["obs_uid"]) != anchor_uid] or e0_rows
    selected_i2 = _best_row_view(run, i2_pool, "E0", margin=0.25)
    i2_e0_visible = selected_i2 is not None
    if selected_i2 is None:
        raise PreflightDefer(
            "DEFER_H_E0_RENDER_MISSING",
            "latest E0 in H snapshot has no renderable accepted mask",
        )
    i2, i2_row = selected_i2
    i2_small_view = bool(i2["target_mask_short_side_px"] < 96)

    i3_mode = "H_WIDER_CONTEXT"
    i3 = None
    i3_row = None
    if versions.get("E1"):
        selected_e1 = _best_row_view(
            run, _accepted_rows(run, versions["E1"], freeze_frame), "E1", margin=0.25
        )
        if selected_e1:
            i3, i3_row = selected_e1
            i3_mode = "LIVE_E1"
    if i3 is None:
        wide_frame = int(i2["frame"])
        wide_e0_rows = [
            row for row in e0_rows
            if frame_index(row.get("frame_uid") or row["obs_uid"]) == wide_frame
        ]
        aliases: dict[str, list[dict[str, Any]]] = {"E0": wide_e0_rows or [i2_row]}
        wide_anchor_rows = [
            row for row in anchor_rows
            if frame_index(row.get("frame_uid") or row["obs_uid"]) == wide_frame
        ]
        if wide_anchor_rows:
            aliases["A"] = wide_anchor_rows
        i3_row = i2_row
        i3 = _view(
            run,
            wide_frame,
            aliases,
            margin=0.25,
        )

    case_dir = output_root / ticket_uid
    case_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "I1": _save_image(primary_image, case_dir / "I1_EVENT.jpg"),
        "I2": _save_image(
            _render_image(
                run,
                i2,
                "I2 - Current E0" if i2_e0_visible else "I2 - Current E0 unavailable | A context",
            ),
            case_dir / "I2_CURRENT.jpg",
        ),
        "I3": _save_image(
            _render_image(
                run,
                i3,
                "I3 - Alternative E1" if i3_mode == "LIVE_E1" else "I3 - Wider scene context",
            ),
            case_dir / "I3_DIAGNOSTIC.jpg",
        ),
    }

    current_label = normalize_label(versions["E0"].get("class_name")) or "unknown"
    alternatives = _semantic_labels([anchor, *e0_rows], current_label)
    identity_targets = tuple(
        ["E0"]
        + (["E1"] if versions.get("E1") and i3_mode == "LIVE_E1" else [])
        + ["SEPARATE", "UNRESOLVED"]
    )
    semantic_targets = tuple(
        ["L0"] + [row["id"] for row in alternatives] + ["UNRESOLVED", "NOT_APPLICABLE"]
    )
    schema = output_schema(identity_targets, semantic_targets)
    label_candidates = {"L0": current_label}
    label_candidates.update({str(row["id"]): str(row["text"]) for row in alternatives})
    image_descriptions = {
        "I1": (
            (
                "A and E0 are shown in exact S in one saved frame."
                if i1_source_relation == "EXACT_S"
                else "A and E0 are shown in one real shared frame not later than D; the offset from S is explicit."
            )
            if i1_layout == "same_frame"
            else "Only exact-S A is shown; no real same-frame E0 source mask was recovered."
        ),
        "I2": (
            "Best available real view of the latest E0."
            if i2_e0_visible
            else "No renderable E0 view was recovered; A context is shown as an explicit fallback."
        ),
        "I3": (
            "Strongest alternative map entity E1 is shown in green."
            if i3_mode == "LIVE_E1"
            else "No valid distinct E1 is available; wider context is built from the H-bound E0 view."
        ),
    }
    degraded_images = []
    if i1_small_view:
        degraded_images.append("I1")
    if i2_small_view or not i2_e0_visible:
        degraded_images.append("I2")
    if i3_mode == "LIVE_E1" and i3["target_mask_short_side_px"] < 96:
        degraded_images.append("I3")
    review_question = _review_question(
        issue_family,
        issue,
        contract,
        has_e1=bool(versions.get("E1") and i3_mode == "LIVE_E1"),
    )
    input_summary = {
        "issue_family": issue_family,
        "review_question": review_question,
        "current_assignment": str(packet.get("current_assignment") or "UNKNOWN"),
        "newer_state_available": bool(packet.get("newer_state_available")),
        "timeline": {
            "S": s_frame,
            "D": d_frame,
            "H": h_frame,
            "I1_frame": i1_frame,
            "I1_offset_from_S": int(i1_frame) - s_frame,
            "I1_source_relation": i1_source_relation,
            "H_snapshot_uid": h_snapshot.get("snapshot_uid"),
        },
        "allowed_identity_targets": list(identity_targets),
        "allowed_semantic_targets": list(semantic_targets),
        "label_candidates": label_candidates,
        "images": image_descriptions,
        "degraded_images": degraded_images,
    }
    incident_text = "CASE_FACTS\n" + json.dumps(
        input_summary, indent=2, ensure_ascii=False, sort_keys=True
    )
    specs = {
        "I1": _spec(
            "I1",
            "ASSIGNMENT_EVENT",
            {
                "description": image_descriptions["I1"],
                "layout": i1_layout,
                "source_relation": i1_source_relation,
                "s_frame": s_frame,
                "d_frame": d_frame,
                "i1_frame": i1_frame,
                "offset_from_s": int(i1_frame) - s_frame,
                "same_physical_frame_for_A_and_E0": i1_layout == "same_frame",
                "image_warning": i1_quality_status if i1_small_view else "NONE",
            },
        ),
        "I2": _spec(
            "I2",
            "LATEST_E0",
            {
                "description": image_descriptions["I2"],
                "h_frame": h_frame,
                "h_snapshot_uid": h_snapshot.get("snapshot_uid"),
                "e0_owner_uid": alias_owner_uids["E0"],
                "e0_version_uid": alias_version_uids["E0"],
                "newer_state_available": bool(packet.get("newer_state_available")),
                "image_warning": "IMAGE_DEGRADED" if i2_small_view else "NONE",
            },
        ),
        "I3": _spec(
            "I3",
            "ALTERNATIVE_E1" if i3_mode == "LIVE_E1" else "WIDER_CONTEXT",
            {
                "description": image_descriptions["I3"],
                "h_frame": h_frame,
                "h_snapshot_uid": h_snapshot.get("snapshot_uid"),
                "e1_owner_uid": alias_owner_uids.get("E1"),
                "e1_version_uid": alias_version_uids.get("E1"),
                "image_warning": (
                    "IMAGE_DEGRADED"
                    if i3_mode == "LIVE_E1" and i3["target_mask_short_side_px"] < 96
                    else "NONE"
                ),
            },
        ),
    }
    final_text = "Return only object_state_v2 JSON."
    sequence = (
        {"type": "text", "text": incident_text},
        {"type": "text", "text": specs["I1"]},
        {"type": "image", "path": str(case_dir / images["I1"]["file"]), "image_id": "I1"},
        {"type": "text", "text": specs["I2"]},
        {"type": "image", "path": str(case_dir / images["I2"]["file"]), "image_id": "I2"},
        {"type": "text", "text": specs["I3"]},
        {"type": "image", "path": str(case_dir / images["I3"]["file"]), "image_id": "I3"},
        {"type": "text", "text": final_text},
    )
    audit_content = []
    for item in sequence:
        if item["type"] == "text":
            audit_content.append({"type": "text", "text": item["text"]})
        else:
            record = images[item["image_id"]]
            audit_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"[REDACTED_BASE64] local_file={record['file']} sha256={record['sha256']}",
                        "detail": "high",
                    },
                }
            )
    write_json(
        case_dir / "actual_request_redacted.json",
        {
            "model": "set at runtime",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": audit_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "object_state_v2", "strict": True, "schema": schema},
            },
            "credential_storage": "memory only; absent from artifact",
        },
    )
    write_json(case_dir / "strict_output_schema.json", schema)
    write_json(case_dir / "input_summary.json", input_summary)
    manifest = {
        "schema_version": "ali_my_object_state_validation/2.0",
        "prompt_version": PROMPT_VERSION,
        "ticket_uid": ticket_uid,
        "source_online_experiment": str(run.root),
        "source_packet": str(packet_path),
        "freeze_frame": freeze_frame,
        "freeze_sequence": freeze_sequence,
        "timeline": dict(timeline),
        "h_snapshot": h_snapshot,
        "alias_owner_uids": alias_owner_uids,
        "alias_version_uids": alias_version_uids,
        "issue_uid": issue["issue_uid"],
        "issue_family": issue_family,
        "resolution": packet.get("resolution"),
        "routing": packet.get("routing"),
        "ranking": packet.get("ranking"),
        "available_identity_targets": list(identity_targets),
        "available_semantic_targets": list(semantic_targets),
        "current_assignment": str(packet.get("current_assignment") or "UNKNOWN"),
        "newer_state_available": bool(packet.get("newer_state_available")),
        "i1_quality_status": i1_quality_status,
        "degraded_images": degraded_images,
        "semantic_label_hypotheses": [{"id": "L0", "text": current_label}, *alternatives],
        "images": {
            "I1": {
                **images["I1"],
                "frames": i1_frames,
                "layout": i1_layout,
                "source_relation": i1_source_relation,
                "s_frame": s_frame,
                "d_frame": d_frame,
                "offset_from_s": int(i1_frame) - s_frame,
                "same_physical_frame_for_A_and_E0": i1_layout == "same_frame",
                "visible_aliases": i1_visible_aliases,
                "image_degraded": i1_small_view,
            },
            "I2": {
                **images["I2"],
                "frame": i2["frame"],
                "member_obs_uid": i2_row["obs_uid"],
                "e0_visible": i2_e0_visible,
                "image_degraded": i2_small_view,
                "quality": i2["quality"],
            },
            "I3": {
                **images["I3"],
                "frame": i3["frame"],
                "member_obs_uid": i3_row["obs_uid"],
                "routing_mode": i3_mode,
                "visible_aliases": list(i3["visible_aliases"]),
            },
        },
        "cutoff_audit": {
            "maximum_observation_frame_used": max([*i1_frames, i2["frame"], i3["frame"]]),
            "all_observation_frames_lte_freeze_frame": max([*i1_frames, i2["frame"], i3["frame"]]) <= freeze_frame,
            "all_active_versions_lte_freeze_sequence": all(
                event_sequence(row["trigger_event_uid"]) <= freeze_sequence
                for row in versions.values() if row
            ),
            "s_lte_d_lte_h": s_frame <= d_frame <= h_frame,
            "i1_not_after_d": int(i1_frame) <= d_frame,
            "i1_single_physical_frame": len(set(i1_frames)) == 1,
            "i1_uses_review_issue_event_core_only": True,
            "i2_e0_is_latest_in_h_snapshot": active_at_h.get(alias_owner_uids["E0"])
            == alias_version_uids["E0"],
            "i3_e1_is_latest_in_h_snapshot": (
                active_at_h.get(alias_owner_uids["E1"]) == alias_version_uids["E1"]
                if "E1" in alias_owner_uids
                else True
            ),
            "e0_e1_distinct_objects": (
                alias_owner_uids.get("E0") != alias_owner_uids.get("E1")
                if "E1" in alias_owner_uids
                else True
            ),
            "h_snapshot_hash_valid": True,
            "active_e0_status": versions["E0"].get("status"),
            "mask_source_required": "processed_mask_ref",
            "final_membership_read": False,
            "ground_truth_read": False,
            "old_vlm_response_read": False,
            "labels_from_e1": False,
        },
    }
    write_json(case_dir / "case_manifest.json", manifest)
    (case_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    return PreparedObjectStateCase(
        ticket_uid,
        case_dir,
        schema,
        identity_targets,
        semantic_targets,
        sequence,
    )


def request_content(case: PreparedObjectStateCase) -> list[dict[str, Any]]:
    content = []
    for item in case.message_sequence:
        if item["type"] == "text":
            content.append({"type": "text", "text": item["text"]})
        else:
            encoded = base64.b64encode(Path(item["path"]).read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
                }
            )
    return content


def call_vlm(
    case: PreparedObjectStateCase,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request_content(case)},
        ],
        "max_completion_tokens": 800,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "object_state_v2", "strict": True, "schema": case.schema},
        },
        "stream": False,
        "store": False,
    }
    audit_path = case.case_dir / "actual_request_redacted.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["model"] = model
    audit["endpoint"] = base_url.rstrip("/") + "/chat/completions"
    write_json(audit_path, audit)
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ali-my-object-state-v2/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read())
        elapsed = time.monotonic() - started
        write_json(case.case_dir / "vlm_raw_response.json", raw)
        choices = raw.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(content, str):
            raise ValueError("VLM response has no string message content")
        output = json.loads(content)
        write_json(case.case_dir / "vlm_output.json", output)
        errors = validate_output(
            output, case.available_identity_targets, case.available_semantic_targets
        )
        validation = {
            "status": "VALID" if not errors else "DEFER_INVALID_OUTPUT",
            "errors": errors,
            "elapsed_seconds": elapsed,
            "model": raw.get("model") or model,
            "response_id": raw.get("id"),
            "usage": raw.get("usage") or {},
            "single_vlm_call": True,
            "schema_valid": not errors,
            "cross_field_valid": not errors,
        }
        write_json(case.case_dir / "validation.json", validation)
        write_case_html(case.case_dir)
        return {
            "ticket_uid": case.ticket_uid,
            "status": validation["status"],
            "output": output,
            "available_identity_targets": list(case.available_identity_targets),
            "available_semantic_targets": list(case.available_semantic_targets),
            "elapsed_seconds": elapsed,
            "model": validation["model"],
            "usage": validation["usage"],
            "case_dir": str(case.case_dir),
        }
    except urllib.error.HTTPError as exc:
        error = safe_error(RuntimeError(f"HTTP {exc.code}: {exc.read(4000).decode('utf-8', 'replace')}"))
    except Exception as exc:
        error = safe_error(exc)
    elapsed = time.monotonic() - started
    failure = {
        "ticket_uid": case.ticket_uid,
        "status": "API_OR_PARSE_ERROR",
        "error": error,
        "elapsed_seconds": elapsed,
        "single_vlm_call": True,
        "retry_count": 0,
        "case_dir": str(case.case_dir),
    }
    write_json(case.case_dir / "validation.json", failure)
    write_case_html(case.case_dir)
    return failure


def _read_optional(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _semantic_label_text(summary: Mapping[str, Any]) -> str:
    labels = summary.get("label_candidates") or {}
    if not isinstance(labels, Mapping) or not labels:
        return "无"
    parts = []
    for label_id, label_text in labels.items():
        current = "（当前 E0 标签）" if str(label_id) == "L0" else ""
        parts.append(f"{label_id}={label_text}{current}")
    return " · ".join(parts)


def _pool_route_text(manifest: Mapping[str, Any]) -> str:
    routing = manifest.get("routing") or {}
    current = str(routing.get("pool_location") or "UNKNOWN")
    destination = str(routing.get("destination") or current)
    return current if destination == current else f"{current} → {destination}"


def write_case_html(case_dir: Path) -> None:
    manifest = _read_optional(case_dir / "case_manifest.json") or {}
    summary = _read_optional(case_dir / "input_summary.json") or {}
    output = _read_optional(case_dir / "vlm_output.json")
    validation = _read_optional(case_dir / "validation.json") or {}
    images = manifest.get("images") or {}
    image_titles = {
        "I1": "I1 · S 事件/因果邻帧（红 A、蓝 E0 必须同帧）",
        "I2": "I2 · H 快照中的最新 E0（蓝色）",
        "I3": "I3 · H 快照中的独立 E1（绿色）或 H 宽场景",
    }
    cards = "".join(
        '<figure class="image-card '
        + ("wide" if image_id == "I1" else "")
        + '"><div class="image-title">'
        + html.escape(image_titles[image_id])
        + '</div><img src="'
        + html.escape(str(record.get("file") or ""))
        + '"><figcaption>'
        + html.escape(str((summary.get("images") or {}).get(image_id) or ""))
        + (" <b>仅质量提醒，不拦截。</b>" if image_id in set(summary.get("degraded_images") or ()) else "")
        + "</figcaption></figure>"
        for image_id, record in ((key, images.get(key) or {}) for key in ("I1", "I2", "I3"))
    )
    status = str(validation.get("status") or "PREPARED_ONLY")
    output_text = (
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True)
        if output is not None
        else "尚未调用 VLM：本页只验证输入构造与输出约束。"
    )
    pool = _pool_route_text(manifest)
    i1_layout = str((images.get("I1") or {}).get("layout") or "UNKNOWN")
    assignment = str(summary.get("current_assignment") or "UNKNOWN")
    warnings = ", ".join(summary.get("degraded_images") or ()) or "无"
    semantic_labels = _semantic_label_text(summary)
    review_question = str(summary.get("review_question") or "未提供")
    timeline = manifest.get("timeline") or {}
    c_frame = timeline.get("c_frame")
    timeline_text = (
        f"S{timeline.get('s_frame')} ≤ D{timeline.get('d_frame')} ≤ "
        f"H{timeline.get('h_frame')} ≤ C{c_frame if c_frame is not None else '待保存'}"
    )
    i1_record = images.get("I1") or {}
    i1_relation = str(i1_record.get("source_relation") or "UNKNOWN")
    i1_offset = i1_record.get("offset_from_s")
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(case_dir.name)}</title>
<style>
*{{box-sizing:border-box}}body{{font-family:"Segoe UI",Arial,sans-serif;margin:0;background:#eef2f6;color:#182230}}main{{max-width:1480px;margin:auto;padding:24px}}h1{{margin:0 0 8px;font-size:25px}}h2{{font-size:18px;margin:26px 0 10px}}.sub{{color:#526173;margin-bottom:18px}}.facts{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:10px;margin:16px 0}}.fact,.panel,.image-card{{background:#fff;border:1px solid #d8e0e8;border-radius:12px;box-shadow:0 2px 8px #26384a12}}.fact{{padding:12px}}.fact small{{display:block;color:#68788b;margin-bottom:5px}}.fact strong{{font-size:16px}}.question,.semantic{{margin:0 0 10px;padding:11px 14px;border-radius:10px;line-height:1.55}}.question{{background:#eaf4ff;border:1px solid #8dbdea}}.question b{{color:#124e7c;margin-right:8px}}.semantic{{background:#fff7df;border:1px solid #edcf78}}.semantic b{{color:#704c00;margin-right:8px}}.legend{{display:flex;gap:12px;flex-wrap:wrap;background:#fff;padding:10px 14px;border-radius:10px;border:1px solid #d8e0e8}}.tag{{display:inline-flex;align-items:center;gap:6px;font-weight:700}}.dot{{width:13px;height:13px;border-radius:50%;display:inline-block}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:14px}}.image-card{{margin:0;padding:12px;min-width:0}}.image-card.wide{{grid-column:1/-1}}.image-title{{font-weight:750;margin:0 0 8px}}img{{display:block;width:100%;max-height:620px;object-fit:contain;background:#111;border-radius:8px}}figcaption{{line-height:1.5;color:#4f5f70;padding-top:8px}}figcaption b{{color:#a64b00}}.panel{{padding:14px}}pre{{white-space:pre-wrap;word-break:break-word;margin:0;font:13px/1.5 Consolas,monospace}}.ok{{color:#087830}}.warn{{color:#a34b00}}@media(max-width:850px){{.facts{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}.image-card.wide{{grid-column:auto}}}}
</style><main>
<h1>VLM 输入检查 · {html.escape(case_dir.name)}</h1>
<div class="sub">此页展示真正送入 VLM 的三张图与精简事实；标签都在图像外部，不遮挡 mask。</div>
<div class="facts"><div class="fact"><small>准备状态</small><strong class="{'ok' if status == 'VALID' else 'warn'}">{html.escape(status)}</strong></div><div class="fact"><small>统一时间线</small><strong>{html.escape(timeline_text)}</strong></div><div class="fact"><small>I1 相对 S</small><strong>{html.escape(i1_relation)} · Δ{html.escape(str(i1_offset))}</strong></div><div class="fact"><small>当前池 → Shadow 投影</small><strong>{html.escape(pool)}</strong></div><div class="fact"><small>当前 A 归属</small><strong>{html.escape(assignment)}</strong></div><div class="fact"><small>质量提醒</small><strong>{html.escape(warnings)}</strong></div></div>
<div class="question"><b>本例为什么送审（实际随请求发送，属于怀疑而非答案）</b>{html.escape(review_question)}</div>
<div class="semantic"><b>语义候选（实际随请求发送）</b>{html.escape(semantic_labels)}</div>
<div class="legend"><span class="tag"><i class="dot" style="background:rgb{ALIAS_COLORS['A']}"></i>A：待审核观测</span><span class="tag"><i class="dot" style="background:rgb{ALIAS_COLORS['E0']}"></i>E0：建票时主对象</span><span class="tag"><i class="dot" style="background:rgb{ALIAS_COLORS['E1']}"></i>E1：最强独立备选</span></div>
<div class="grid">{cards}</div>
<h2>VLM 输出</h2><div class="panel"><pre>{html.escape(output_text)}</pre></div>
<h2>实际送入的精简事实</h2><div class="panel"><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))}</pre></div>
<h2>统一时间线、H 快照与防泄露检查</h2><div class="panel"><pre>{html.escape(json.dumps({'timeline': timeline, 'h_snapshot_binding': {key: (manifest.get('h_snapshot') or {}).get(key) for key in ('snapshot_uid','snapshot_sha256','cutoff_frame','cutoff_sequence','watermark_source')}, 'cutoff_audit': manifest.get('cutoff_audit')}, indent=2, ensure_ascii=False, sort_keys=True))}</pre></div>
</main>"""
    (case_dir / "index.html").write_text(page, encoding="utf-8")


def write_root_html(root: Path) -> None:
    cards = []
    vlm_attempted = False
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = _read_optional(case_dir / "case_manifest.json") or {}
        summary = _read_optional(case_dir / "input_summary.json") or {}
        validation = _read_optional(case_dir / "validation.json") or {}
        output = _read_optional(case_dir / "vlm_output.json") or {}
        vlm_attempted = vlm_attempted or str(validation.get("status") or "") not in {
            "",
            "PREPARED_ONLY",
        }
        image_manifest = manifest.get("images") or {}
        pool = _pool_route_text(manifest)
        assignment = str(summary.get("current_assignment") or "UNKNOWN")
        layout = str((image_manifest.get("I1") or {}).get("layout") or "UNKNOWN")
        warning = ", ".join(summary.get("degraded_images") or ()) or "无"
        semantic_labels = _semantic_label_text(summary)
        review_question = str(summary.get("review_question") or "未提供")
        timeline = manifest.get("timeline") or {}
        c_frame = timeline.get("c_frame")
        timeline_text = (
            f"S{timeline.get('s_frame')} ≤ D{timeline.get('d_frame')} ≤ "
            f"H{timeline.get('h_frame')} ≤ C{c_frame if c_frame is not None else '待保存'}"
        )
        i1_record = image_manifest.get("I1") or {}
        i1_relation = str(i1_record.get("source_relation") or "UNKNOWN")
        i1_offset = i1_record.get("offset_from_s")
        thumbs = "".join(
            f'<div><b>{image_id}</b><img src="{html.escape(case_dir.name)}/{html.escape(str((image_manifest.get(image_id) or {}).get("file") or ""))}"></div>'
            for image_id in ("I1", "I2", "I3")
        )
        cards.append(
            '<article><div class="top"><div><span class="pool">'
            + html.escape(pool)
            + '</span><h2><a href="'
            + html.escape(case_dir.name)
            + '/index.html">'
            + html.escape(case_dir.name)
            + '</a></h2></div><strong>'
            + html.escape(str(validation.get("status") or "PREPARED_ONLY"))
            + '</strong></div><div class="meta"><span>触发：'
            + html.escape(str(manifest.get("issue_family") or "UNKNOWN"))
            + '</span><span>A 当前归属：'
            + html.escape(assignment)
            + '</span><span>I1：'
             + html.escape(layout)
            + '</span><span>时间线：'
            + html.escape(timeline_text)
            + '</span><span>I1 相对 S：'
            + html.escape(f"{i1_relation} · Δ{i1_offset}")
            + '</span><span>质量提醒：'
            + html.escape(warning)
            + '</span></div><div class="question"><b>送审原因（怀疑，不是答案）</b>'
            + html.escape(review_question)
            + '</div><div class="semantic"><b>语义候选（实际输入 VLM）</b>'
            + html.escape(semantic_labels)
            + '</div><div class="thumbs">'
            + thumbs
            + '</div><div class="result">VLM：'
            + html.escape(
                f"{output.get('identity_target')} / {output.get('semantic_target')}"
                if output else "尚未调用（仅检查输入）"
            )
            + '</div></article>'
        )
    case_count = len(cards)
    stage_title = (
        f"候选池 → VLM：{case_count} 例真实审核结果"
        if vlm_attempted
        else f"候选池 → VLM 输入包：{case_count} 例准备结果"
    )
    stage_lead = (
        "本页展示真实 VLM 调用的三图输入、语义候选与结构化输出；只做诊断，"
        "没有执行地图修复。所有图像均受各自冻结水位限制；没有读取最终建图成员关系、"
        "GT 或旧 VLM 答案。点击案例编号可查看大图、完整输出与防泄露检查。"
        if vlm_attempted
        else "本阶段只检查筛选后案例的 E0/E1 绑定、三张真实输入图和严格输出范围，"
        "尚未调用 VLM，也未执行修复。所有图像均受各自冻结水位限制；没有读取最终建图"
        "成员关系、GT 或旧 VLM 答案。点击案例编号可查看大图与完整解释。"
    )
    page = f"""<!doctype html><meta charset="utf-8"><title>object_state_v2 可视化</title>
<style>*{{box-sizing:border-box}}body{{font-family:"Segoe UI",Arial,sans-serif;margin:0;background:#eef2f6;color:#182230}}main{{max-width:1580px;margin:auto;padding:24px}}h1{{margin:0 0 8px}}.lead{{line-height:1.6;color:#516173;max-width:1100px}}.legend{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0;padding:12px;background:#fff;border:1px solid #d9e1e9;border-radius:10px}}.legend b{{display:inline-flex;align-items:center;gap:6px}}.dot{{width:13px;height:13px;border-radius:50%}}.cases{{display:grid;gap:18px}}article{{background:#fff;border:1px solid #d7e0e8;border-radius:14px;padding:15px;box-shadow:0 2px 9px #23384d12}}.top{{display:flex;justify-content:space-between;gap:12px;align-items:start}}h2{{font-size:17px;margin:6px 0}}a{{color:#1557a0}}.pool{{background:#e8f1ff;color:#164d85;font-weight:700;border-radius:999px;padding:3px 9px;font-size:12px}}.meta{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 10px}}.meta span{{background:#f1f4f7;border-radius:7px;padding:5px 8px;font-size:13px}}.question,.semantic{{margin:0 0 10px;padding:9px 11px;border-radius:8px;line-height:1.5}}.question{{background:#eaf4ff;border:1px solid #8dbdea}}.question b{{color:#124e7c;margin-right:8px}}.semantic{{background:#fff7df;border:1px solid #edcf78}}.semantic b{{color:#704c00;margin-right:8px}}.thumbs{{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:10px}}.thumbs div{{font-size:13px}}.thumbs img{{display:block;width:100%;height:260px;object-fit:contain;background:#111;border-radius:8px;margin-top:5px}}.result{{margin-top:10px;color:#526173}}@media(max-width:900px){{.thumbs{{grid-template-columns:1fr}}.thumbs img{{height:auto;max-height:480px}}}}</style><main>
<h1>{html.escape(stage_title)}</h1>
<p class="lead">{html.escape(stage_lead)}</p>
<div class="legend"><b><i class="dot" style="background:rgb{ALIAS_COLORS['A']}"></i>红 A：待审核观测</b><b><i class="dot" style="background:rgb{ALIAS_COLORS['E0']}"></i>蓝 E0：建票时主对象</b><b><i class="dot" style="background:rgb{ALIAS_COLORS['E1']}"></i>绿 E1：最强独立备选</b></div>
<section class="cases">{''.join(cards)}</section></main>"""
    (root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--online-subdir", default="online_mvp")
    parser.add_argument(
        "--packet-subdir",
        help="Optional packet-only subdirectory; online events still come from --online-subdir.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ticket", action="append", required=True, dest="tickets")
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    run = FrozenRun(args.experiment_root, online_subdir=args.online_subdir)
    packet_root = (
        args.experiment_root / args.packet_subdir
        if args.packet_subdir
        else run.online_root
    )
    prepared = []
    results = []
    for ticket in args.tickets:
        try:
            prepared.append(
                prepare_case(
                    run,
                    ticket,
                    args.output_root,
                    packet_root=packet_root,
                )
            )
        except PreflightDefer as exc:
            results.append({"ticket_uid": ticket, "status": exc.code, "detail": exc.detail})
    if args.prepare_only:
        for case in prepared:
            write_json(case.case_dir / "validation.json", {"status": "PREPARED_ONLY"})
            write_case_html(case.case_dir)
    else:
        keys = [
            getpass.getpass(f"API key {index + 1}/{len(prepared)} (memory only): ").strip()
            for index in range(len(prepared))
        ]
        with ThreadPoolExecutor(max_workers=max(1, len(prepared))) as executor:
            futures = {
                executor.submit(call_vlm, case, keys[index], args.base_url, args.model, args.timeout_seconds): case
                for index, case in enumerate(prepared)
            }
            for future in as_completed(futures):
                results.append(future.result())
    write_root_html(args.output_root)
    write_json(
        args.output_root / "run_summary.json",
        {
            "prompt_version": PROMPT_VERSION,
            "output_contract_version": "object_state_v2",
            "prepared": len(prepared),
            "results": sorted(results, key=lambda row: row["ticket_uid"]),
            "api_keys_persisted": False,
            "repair_or_map_mutation": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
