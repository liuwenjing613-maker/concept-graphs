#!/usr/bin/env python3
"""Validate declarative ``object_state_v2`` on frozen online evidence."""

from __future__ import annotations

import argparse
import base64
import getpass
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
    ALIAS_COLORS,
    FrozenRun,
    PreflightDefer,
    bbox_from_mask,
    clamp,
    event_sequence,
    frame_index,
    get_font,
    normalize_label,
    pose_difference,
    safe_error,
    sha256_file,
    sharpness_score,
    write_json,
)


PROMPT_VERSION = "ali_my_object_state_v2_20260828"
MISSING_EVIDENCE = (
    "NONE",
    "EVENT_EVIDENCE_UNCLEAR",
    "CURRENT_OBJECT_UNCLEAR",
    "ALTERNATIVE_OBJECT_UNCLEAR",
    "WIDER_CONTEXT_NEEDED",
    "LABEL_NOT_LISTED",
    "COMPOUND_STATE_REQUIRED",
)

SYSTEM_PROMPT = """You are reviewing the current object state of one online 3D mapping case.
The trigger is only a machine hypothesis, not proof of an error.

Use exactly I1_EVENT, I2_CURRENT, and I3_DIAGNOSTIC with their IMAGE_SPEC.
Overlay colors are artificial pointers and are never physical object colors.
Every overlay is an accepted processed mask from the stated RGB frame, not
ground truth, a raw proposal, or future end-of-run membership.

First decide which physical entity review unit A should belong to now: E0, an
available E1/E2, SEPARATE, or UNRESOLVED. Only if identity_target is E0,
decide whether current label L0 or one listed alternative best describes E0.
If identity changes, output semantic_target=NOT_EVALUATED.

I1 shows A and E0 in the same RGB frame. A may be a partial observation of
the larger E0 entity. Never choose SEPARATE merely because A covers only one
part of E0 or because the two masks have different apparent sizes.
If I1 is marked LOW_RESOLUTION, it remains diagnostic evidence, but prefer
UNRESOLVED over a structural identity change unless I2/I3 independently make
that change clear.

Describe a target state only. Do not propose an action, merge command,
constraint, candidate ID, or confidence score. Use UNRESOLVED when the three
images do not establish the state. Return only JSON matching the schema."""


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
    return (
        max(0, x0 - mx),
        max(0, y0 - my),
        min(mask.shape[1], x1 + mx),
        min(mask.shape[0], y1 + my),
    )


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
    for alias in ("A", "E0", "E1", "E2"):
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
        "visible_aliases": tuple(alias for alias in ("A", "E0", "E1", "E2") if alias in masks),
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
    fill_alpha = {"A": 0.30, "E0": 0.12, "E1": 0.14, "E2": 0.14}
    for alias in ("E2", "E1", "E0", "A"):
        mask = view["masks"].get(alias)
        if mask is None:
            continue
        color = np.asarray(ALIAS_COLORS[alias], dtype=np.float32)
        alpha = fill_alpha[alias]
        array[mask] = np.clip(
            (1.0 - alpha) * array[mask].astype(np.float32) + alpha * color, 0, 255
        ).astype(np.uint8)
    canvas = Image.fromarray(array)
    for alias in ("E2", "E1", "E0", "A"):
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
    trigger_frame: int,
) -> int | None:
    """Select the closest trigger-time frame where both A and E0 are visible."""

    anchor_frames = {
        frame_index(row.get("frame_uid") or row["obs_uid"]) for row in anchor_rows
    }
    core_frames = {
        frame_index(row.get("frame_uid") or row["obs_uid"]) for row in core_rows
    }
    shared = anchor_frames & core_frames
    return min(shared, key=lambda frame: (abs(frame - trigger_frame), frame)) if shared else None


def _candidate_event_context(
    packet: Mapping[str, Any], observations: Mapping[str, Mapping[str, Any]]
) -> list[tuple[str, dict[str, Any]]]:
    """Use only resolver-bound E1/E2 aliases; never enumerate raw candidates."""

    available = set((packet.get("alias_version_uids") or {}).keys())
    alias_refs = packet.get("candidate_alias_observation_uids") or {}
    sources: list[tuple[str, dict[str, Any]]] = []
    for alias in ("E1", "E2"):
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
            int(item[0]["eligible"]),
            item[0]["quality"],
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
            "reason": {"type": "string", "minLength": 1, "maxLength": 240},
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
        if semantic == "NOT_EVALUATED":
            errors.append("identity_target E0 requires semantic evaluation")
    elif semantic != "NOT_EVALUATED":
        errors.append("non-E0 identity requires semantic_target NOT_EVALUATED")
    evidence = value.get("evidence_ids")
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 3
        or len(evidence) != len(set(evidence))
        or any(item not in {"I1", "I2", "I3"} for item in evidence)
    ):
        errors.append("evidence_ids must contain 1-3 unique IDs from I1/I2/I3")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason or len(reason) > 240:
        errors.append("reason must be a nonempty string of at most 240 characters")
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
    run: FrozenRun, ticket_uid: str, output_root: Path
) -> PreparedObjectStateCase:
    packet_path = run.online_root / "vlm" / ticket_uid / "evidence" / "packet_manifest.json"
    if not packet_path.is_file():
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "V2 packet manifest missing")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("output_contract_version") != "object_state_v2":
        raise PreflightDefer("DEFER_WRONG_CONTRACT", "packet is not object_state_v2")
    freeze_frame = int(packet["freeze_frame"])
    freeze_sequence = int(packet["freeze_sequence"])
    issue = packet["issue"]
    contract = packet["repair_contract"]
    trigger_frame = int(issue["detected_frame"])
    review_uids = [str(uid) for uid in contract.get("review_unit_obs_uids") or ()]
    versions = {
        alias: run.versions.get(str(uid))
        for alias, uid in (packet.get("alias_version_uids") or {}).items()
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
    review_uid_set = set(review_uids)
    e0_context_rows = [
        row for row in e0_rows if str(row["obs_uid"]) not in review_uid_set
    ]
    i1_frame = _shared_event_frame(anchor_rows, e0_context_rows, trigger_frame)
    if i1_frame is None:
        raise PreflightDefer(
            "DEFER_EVENT_PAIR_UNAVAILABLE",
            "I1 requires accepted A and E0 masks in the same event frame",
        )
    frame_anchor_rows = [
        row for row in anchor_rows
        if frame_index(row.get("frame_uid") or row["obs_uid"]) == i1_frame
    ]
    frame_e0_rows = [
        row for row in e0_rows
        if frame_index(row.get("frame_uid") or row["obs_uid"]) == i1_frame
    ]
    anchor = min(frame_anchor_rows, key=lambda row: str(row["obs_uid"]))
    event_alias_rows: dict[str, list[dict[str, Any]]] = {
        "A": frame_anchor_rows,
        "E0": frame_e0_rows,
    }
    context_sources = _candidate_event_context(packet, run.observations)
    for alias, row in context_sources:
        if frame_index(row.get("frame_uid") or row["obs_uid"]) == i1_frame:
            event_alias_rows.setdefault(alias, []).append(row)
    i1 = _view(run, i1_frame, event_alias_rows, margin=0.35)
    if not {"A", "E0"}.issubset(i1["visible_aliases"]):
        raise PreflightDefer(
            "DEFER_EVENT_PAIR_UNAVAILABLE",
            "I1 could not render both A and E0 from accepted same-frame masks",
        )
    i1_quality_status = "PASS" if i1["eligible"] else "LOW_RESOLUTION"
    primary_image = _render_image(
        run,
        i1,
        "I1" if i1["eligible"] else "I1 LOW-RES",
    )
    dual_panel = False

    anchor_uid = str(anchor["obs_uid"])
    post_non_anchor = [
        row for row in e0_rows
        if str(row["obs_uid"]) != anchor_uid
        and frame_index(row.get("frame_uid") or row["obs_uid"]) > trigger_frame
    ]
    historical_non_anchor = [row for row in e0_rows if str(row["obs_uid"]) != anchor_uid]
    i2_pool = post_non_anchor or historical_non_anchor or [anchor]
    selected_i2 = _best_row_view(run, i2_pool, "E0", margin=0.25)
    if not selected_i2:
        raise PreflightDefer("DEFER_CURRENT_OBJECT_UNCLEAR", "no renderable E0 member view")
    i2, i2_row = selected_i2
    if not i2["eligible"]:
        raise PreflightDefer(
            "DEFER_CURRENT_OBJECT_UNCLEAR",
            "best active E0 member view is below the 96px/visibility gate",
        )

    issue_family = str(issue.get("family") or "")
    i3_mode = "WIDER_CONTEXT"
    i3 = None
    i3_row = None
    if versions.get("E1"):
        selected_e1 = _best_row_view(
            run, _accepted_rows(run, versions["E1"], freeze_frame), "E1", margin=0.25
        )
        if selected_e1 and selected_e1[0]["eligible"]:
            i3, i3_row = selected_e1
            i3_mode = "LIVE_E1"
    if i3 is None and issue_family in {"SEMANTIC_ASSOCIATION_CONFLICT", "SEMANTIC_DRIFT"}:
        diverse = []
        for row in e0_rows:
            if str(row["obs_uid"]) == str(i2_row["obs_uid"]):
                continue
            selected = _best_row_view(run, [row], "E0", margin=0.25)
            if not selected:
                continue
            _, _, pose_score = pose_difference(
                run.frames[selected[0]["frame"]], run.frames[i2["frame"]]
            )
            diverse.append((0.65 * selected[0]["quality"] + 0.35 * pose_score, selected))
        if diverse:
            _, (i3, i3_row) = max(diverse, key=lambda item: item[0])
            i3_mode = "SEMANTIC_DIVERSE_E0"
    if i3 is None:
        i3_row = anchor if i1["target_mask_short_side_px"] >= i2["target_mask_short_side_px"] else i2_row
        alias = "A" if str(i3_row["obs_uid"]) == anchor_uid else "E0"
        i3 = _view(
            run,
            frame_index(i3_row.get("frame_uid") or i3_row["obs_uid"]),
            {alias: [i3_row]},
            margin=0.45,
        )

    case_dir = output_root / ticket_uid
    case_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "I1": _save_image(primary_image, case_dir / "I1_EVENT.jpg"),
        "I2": _save_image(_render_image(run, i2, "I2"), case_dir / "I2_CURRENT.jpg"),
        "I3": _save_image(_render_image(run, i3, "I3"), case_dir / "I3_DIAGNOSTIC.jpg"),
    }

    current_label = normalize_label(versions["E0"].get("class_name")) or "unknown"
    alternatives = _semantic_labels([anchor, *e0_rows], current_label)
    identity_targets = tuple(
        ["E0"]
        + [alias for alias in ("E1", "E2") if versions.get(alias)]
        + ["SEPARATE", "UNRESOLVED"]
    )
    semantic_targets = tuple(
        ["L0"] + [row["id"] for row in alternatives] + ["UNRESOLVED", "NOT_EVALUATED"]
    )
    schema = output_schema(identity_targets, semantic_targets)
    input_summary = {
        "trigger_family": issue_family,
        "trigger_is_machine_hypothesis": True,
        "trigger_frame": trigger_frame,
        "review_frame": freeze_frame,
        "review_unit": "A",
        "current_owner": "E0",
        "available_identity_targets": list(identity_targets),
        "has_post_event_update": bool(packet.get("resolution", {}).get("has_post_event_update")),
        "i1_quality_status": i1_quality_status,
        "constraint_policy": (
            "ELIGIBLE_AFTER_VALIDATION"
            if i1_quality_status == "PASS"
            else "DIAGNOSTIC_ONLY_NO_STRUCTURAL_CONSTRAINT"
        ),
        "current_label": {"id": "L0", "text": current_label},
        "alternative_labels": alternatives,
        "labels_use_only_review_unit_and_e0": True,
    }
    incident_text = "CURRENT REVIEW STATE\n" + json.dumps(
        input_summary, indent=2, ensure_ascii=False, sort_keys=True
    )
    specs = {
        "I1": _spec(
            "I1",
            "EVENT_EVIDENCE",
            {
                "frame": i1_frame,
                "visible_aliases": list(i1["visible_aliases"]),
                "mask_source": "processed_mask_ref",
                "dual_panel": dual_panel,
                "quality_status": i1_quality_status,
                "target_mask_short_side_px": round(i1["target_mask_short_side_px"], 3),
                "structural_constraint_eligible": i1_quality_status == "PASS",
                "selection_reason": "same-frame accepted A and event-time E0 masks in one union crop",
            },
        ),
        "I2": _spec(
            "I2",
            "CURRENT_OBJECT",
            {
                "frame": i2["frame"],
                "object_version_uid": versions["E0"]["object_version_uid"],
                "member_obs_uid": i2_row["obs_uid"],
                "visible_aliases": ["E0"],
                "mask_source": "processed_mask_ref",
                "post_event": i2["frame"] > trigger_frame,
                "reuses_anchor": str(i2_row["obs_uid"]) == anchor_uid,
                "quality_score": round(i2["quality"], 6),
            },
        ),
        "I3": _spec(
            "I3",
            "DIAGNOSTIC",
            {
                "frame": i3["frame"],
                "routing_mode": i3_mode,
                "member_obs_uid": i3_row["obs_uid"],
                "visible_aliases": list(i3["visible_aliases"]),
                "mask_source": "processed_mask_ref",
                "crop_margin": i3["margin"],
            },
        ),
    }
    final_text = (
        "Return only object_state_v2 JSON. Available identity targets: "
        + ", ".join(identity_targets)
        + ". Available semantic targets: "
        + ", ".join(semantic_targets)
        + ".\nOUTPUT_SCHEMA\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )
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
        "issue_uid": issue["issue_uid"],
        "issue_family": issue_family,
        "resolution": packet.get("resolution"),
        "ranking": packet.get("ranking"),
        "available_identity_targets": list(identity_targets),
        "available_semantic_targets": list(semantic_targets),
        "i1_quality_status": i1_quality_status,
        "structural_constraint_eligible": i1_quality_status == "PASS",
        "semantic_label_hypotheses": [{"id": "L0", "text": current_label}, *alternatives],
        "images": {
            "I1": {**images["I1"], "frame": i1_frame, "dual_panel": dual_panel},
            "I2": {**images["I2"], "frame": i2["frame"], "member_obs_uid": i2_row["obs_uid"], "quality": i2["quality"]},
            "I3": {**images["I3"], "frame": i3["frame"], "member_obs_uid": i3_row["obs_uid"], "routing_mode": i3_mode},
        },
        "cutoff_audit": {
            "maximum_observation_frame_used": max(i1_frame, i2["frame"], i3["frame"]),
            "all_observation_frames_lte_freeze_frame": max(i1_frame, i2["frame"], i3["frame"]) <= freeze_frame,
            "all_active_versions_lte_freeze_sequence": all(
                event_sequence(row["trigger_event_uid"]) <= freeze_sequence
                for row in versions.values() if row
            ),
            "active_e0_status": versions["E0"].get("status"),
            "mask_source_required": "processed_mask_ref",
            "final_membership_read": False,
            "ground_truth_read": False,
            "old_vlm_response_read": False,
            "labels_from_e1_or_e2": False,
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


def write_case_html(case_dir: Path) -> None:
    manifest = _read_optional(case_dir / "case_manifest.json") or {}
    summary = _read_optional(case_dir / "input_summary.json") or {}
    output = _read_optional(case_dir / "vlm_output.json")
    validation = _read_optional(case_dir / "validation.json") or {}
    cards = "".join(
        f'<figure><img src="{name}_{role}.jpg"><figcaption>{name} {role}</figcaption></figure>'
        for name, role in (("I1", "EVENT"), ("I2", "CURRENT"), ("I3", "DIAGNOSTIC"))
    )
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(case_dir.name)}</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f5f7fa;color:#17202a}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}figure{{margin:0;background:white;padding:10px;border-radius:10px}}img{{width:100%;height:360px;object-fit:contain;background:#111}}pre{{white-space:pre-wrap;background:white;padding:14px;border-radius:10px;overflow:auto}}.ok{{color:#087830;font-weight:bold}}.warn{{color:#a34b00;font-weight:bold}}</style>
<h1>object_state_v2 · {html.escape(case_dir.name)}</h1>
<p class="{'ok' if validation.get('status') == 'VALID' else 'warn'}">{html.escape(str(validation.get('status')))}</p>
<div class="grid">{cards}</div>
<h2>VLM 实际输出</h2><pre>{html.escape(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))}</pre>
<h2>实际输入摘要</h2><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))}</pre>
<h2>水位与防泄露审计</h2><pre>{html.escape(json.dumps(manifest.get('cutoff_audit'), indent=2, ensure_ascii=False, sort_keys=True))}</pre>"""
    (case_dir / "index.html").write_text(page, encoding="utf-8")


def write_root_html(root: Path) -> None:
    rows = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = _read_optional(case_dir / "case_manifest.json") or {}
        validation = _read_optional(case_dir / "validation.json") or {}
        output = _read_optional(case_dir / "vlm_output.json") or {}
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(case_dir.name)}/index.html">{html.escape(case_dir.name)}</a></td>'
            f"<td>{html.escape(str(manifest.get('issue_family')))}</td>"
            f"<td>{html.escape(str(manifest.get('resolution', {}).get('state')))}</td>"
            f"<td>{html.escape(str(manifest.get('ranking', {}).get('error_tier')))}</td>"
            f"<td>{html.escape(str(validation.get('status')))}</td>"
            f"<td>{html.escape(str(output.get('identity_target')))}</td>"
            f"<td>{html.escape(str(output.get('semantic_target')))}</td>"
            f"<td>{html.escape(str(output.get('missing_evidence')))}</td>"
            "</tr>"
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>object_state_v2 可视化</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f5f7fa;color:#17202a}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;border:1px solid #dfe6ee;text-align:left}}th{{background:#e9f1fb}}</style>
<h1>候选池 + VLM：object_state_v2 可视化</h1>
<p>每个案例均使用冻结 review 水位，三图只读取 accepted processed masks；未读取 final membership、GT 或旧 VLM 输出。</p>
<table><thead><tr><th>Ticket</th><th>触发族</th><th>池状态</th><th>错误档</th><th>JSON校验</th><th>Identity</th><th>Semantic</th><th>缺失证据</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"""
    (root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--online-subdir", default="online_mvp")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ticket", action="append", required=True, dest="tickets")
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    run = FrozenRun(args.experiment_root, online_subdir=args.online_subdir)
    prepared = []
    results = []
    for ticket in args.tickets:
        try:
            prepared.append(prepare_case(run, ticket, args.output_root))
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
