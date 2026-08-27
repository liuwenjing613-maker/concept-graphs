#!/usr/bin/env python3
"""Minimal online-snapshot validation for the ali-my unified three-view VLM plan.

This script deliberately stops at the VLM output.  It reconstructs evidence only
from a ticket's frozen online snapshot, renders exactly three accepted-mask
views, sends one interleaved multimodal request, and saves everything required
for human review.  It never reads final_membership.json or old VLM responses.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROMPT_VERSION = "ali_my_unified_v1"
ALIAS_ORDER = ("A", "E0", "E1", "E2")
ALIAS_COLORS = {
    "A": (255, 212, 0),
    "E0": (0, 213, 255),
    "E1": (255, 61, 187),
    "E2": (56, 214, 107),
}
ALIAS_COLOR_NAMES = {"A": "yellow", "E0": "cyan", "E1": "magenta", "E2": "green"}
NEEDED_EVIDENCE = (
    "clearer_detail_view",
    "more_diverse_view",
    "candidate_entity_not_visible",
    "mask_boundary_ambiguous",
    "correct_label_not_listed",
    "compound_repair_required",
    "insufficient_provenance",
)

SYSTEM_PROMPT = """You are the conservative visual diagnosis module of an open-vocabulary 3D mapping system.

You will receive exactly three separately identified images: I1_DECISION, I2_DETAIL, and I3_DIVERSE. Each image is immediately preceded by its own IMAGE_SPEC. Bind every image only to that IMAGE_SPEC and do not infer missing metadata.

The translucent colored overlays are the post-processed 2D masks actually accepted by the mapper at the frozen decision snapshot. They are visual pointers, not ground truth. They are not raw proposals and not future end-of-run membership. Inspect both the original RGB pixels inside the overlay and the surrounding scene.

Aliases are fixed: A/yellow is the incident anchor observation; E0/cyan is its current owner; E1/magenta and E2/green are competing entities. Colors and aliases do not indicate correctness or confidence. If an alias is not listed as visible in an IMAGE_SPEC, treat it as unobserved in that frame, not absent from the world.

All labels, captions, checker findings, and candidate labels are machine hypotheses, not facts. Repeated labels from the same pipeline are not independent evidence.

Judge two axes separately and in this order:
1) IDENTITY: do the accepted masks form coherent physical instances, and is A assigned to the correct entity?
2) SEMANTIC_LABEL: only if the instance is coherent, does the stable label match the visible base object category?

If mixed or split physical instances explain the label inconsistency, choose an identity action rather than RELABEL. Do not relabel for harmless synonyms, reasonable category granularity, color, material, room, or position. Relabel only when at least two complementary views show positive discriminative features that contradict the current label and support one listed alternative.

Select exactly one listed candidate, H0, or DEFER. H0 means the evidence supports the current state. DEFER means evidence is insufficient, the correct action/label is not listed, or a compound repair is required. Never invent an executable label, entity, mask, constraint, or map mutation.

Return only one JSON object matching the supplied schema. The selected candidate ID is the only actionable part of your response."""


class PreflightDefer(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class PreparedCase:
    ticket_uid: str
    case_dir: Path
    schema: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    message_sequence: tuple[dict[str, str], ...]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_index(value: Any) -> int:
    match = re.search(r"(?:^|_)f(\d+)(?:_|$)", str(value))
    if not match:
        raise ValueError(f"cannot parse frame from {value!r}")
    return int(match.group(1))


def event_sequence(value: Any) -> int:
    match = re.search(r"(?:^|_)e(\d+)(?:_|$)", str(value))
    if not match:
        raise ValueError(f"cannot parse event sequence from {value!r}")
    return int(match.group(1))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_label(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = re.sub(r"\s+\d+$", "", text)
    return text[:80]


def safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED_API_KEY]", text)[:2000]


class FrozenRun:
    """Read-only index over online provenance.  Forbidden final artifacts are never opened."""

    def __init__(self, experiment_root: Path) -> None:
        self.root = experiment_root.resolve()
        evidence = self.root / "evidence"
        self.observations = {row["obs_uid"]: row for row in read_jsonl(evidence / "observations.jsonl")}
        self.frames = {int(row["frame_idx"]): row for row in read_jsonl(evidence / "frames.jsonl")}
        self.associations = {row["event_uid"]: row for row in read_jsonl(evidence / "associations.jsonl")}
        self.versions = {row["object_version_uid"]: row for row in read_jsonl(evidence / "object_versions.jsonl")}
        self.versions_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.versions.values():
            self.versions_by_object[str(row["object_uid"])].append(row)
        self.issue_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(self.root / "online_mvp" / "online_events.jsonl"):
            if row.get("type") == "ISSUE_UPSERT" and row.get("ticket_uid") and row.get("issue"):
                self.issue_events[str(row["ticket_uid"])].append(row)
        self._mask_cache: dict[str, np.ndarray] = {}
        self._rgb_cache: dict[int, np.ndarray] = {}

    def resolve_ref(self, ref: Mapping[str, Any]) -> Path:
        path = Path(str(ref.get("path") or ""))
        resolved = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def load_mask(self, row: Mapping[str, Any]) -> np.ndarray:
        ref = row.get("processed_mask_ref")
        if not isinstance(ref, Mapping):
            raise FileNotFoundError(f"accepted observation {row.get('obs_uid')} has no processed_mask_ref")
        path = self.resolve_ref(ref)
        cache_key = f"{path}|{ref.get('key')}|{ref.get('index')}"
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached
        with np.load(path, allow_pickle=False) as payload:
            key = str(ref.get("key") or next(iter(payload.keys())))
            value = np.asarray(payload[key])
        index = ref.get("index")
        if index is not None:
            value = value[int(index)]
        mask = np.asarray(value, dtype=bool)
        self._mask_cache[cache_key] = mask
        return mask

    def load_rgb(self, index: int) -> np.ndarray:
        cached = self._rgb_cache.get(index)
        if cached is not None:
            return cached
        row = self.frames[index]
        ref = row.get("rgb_ref") or {"path": row.get("rgb_path")}
        path = self.resolve_ref(ref)
        value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        self._rgb_cache[index] = value
        return value

    def issue_at_cutoff(self, ticket_uid: str, issue_uid: str, freeze_frame: int, freeze_sequence: int) -> dict[str, Any]:
        matches = []
        for event in self.issue_events.get(ticket_uid, ()):
            issue = event.get("issue") or {}
            if str(issue.get("issue_uid")) != issue_uid:
                continue
            if int(issue.get("detected_frame", 10**12)) <= freeze_frame and int(issue.get("detected_sequence", 10**12)) <= freeze_sequence:
                matches.append(issue)
        if not matches:
            raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "no cutoff-valid ISSUE_UPSERT record")
        return min(matches, key=lambda item: int(item["detected_sequence"]))

    @staticmethod
    def version_is_at_cutoff(row: Mapping[str, Any], freeze_frame: int, freeze_sequence: int) -> bool:
        try:
            return frame_index(row.get("frame_uid")) <= freeze_frame and event_sequence(row.get("trigger_event_uid")) <= freeze_sequence
        except ValueError:
            return False

    def latest_object_version(self, object_uid: str, freeze_frame: int, freeze_sequence: int) -> dict[str, Any] | None:
        rows = [
            row for row in self.versions_by_object.get(object_uid, ())
            if self.version_is_at_cutoff(row, freeze_frame, freeze_sequence)
        ]
        return max(rows, key=lambda row: (int(row.get("version", 0)), event_sequence(row["trigger_event_uid"])), default=None)


def union_masks(run: FrozenRun, rows: Iterable[Mapping[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for row in rows:
        mask = run.load_mask(row)
        if mask.shape != shape:
            raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", f"mask shape {mask.shape} != RGB shape {shape}")
        result |= mask
    return result


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("empty mask")
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def expanded_bbox(mask: np.ndarray, margin: float = 0.15) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_from_mask(mask)
    width, height = x1 - x0, y1 - y0
    mx, my = max(8, int(round(width * margin))), max(8, int(round(height * margin)))
    return max(0, x0 - mx), max(0, y0 - my), min(mask.shape[1], x1 + mx), min(mask.shape[0], y1 + my)


def sharpness_score(gray: np.ndarray) -> float:
    if min(gray.shape[:2]) < 3:
        return 0.0
    value = gray.astype(np.float32) / 255.0
    lap = -4.0 * value[1:-1, 1:-1] + value[:-2, 1:-1] + value[2:, 1:-1] + value[1:-1, :-2] + value[1:-1, 2:]
    return clamp(math.log1p(float(np.var(lap)) * 100000.0) / math.log1p(4000.0))


def pose_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[float, float, float]:
    pa, pb = np.asarray(a["pose"], dtype=float), np.asarray(b["pose"], dtype=float)
    baseline = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3]))
    rotation = pa[:3, :3].T @ pb[:3, :3]
    angle = math.degrees(math.acos(clamp((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)))
    score = 0.6 * clamp(baseline / 2.0) + 0.4 * clamp(angle / 90.0)
    return baseline, angle, score


def build_view_record(
    run: FrozenRun,
    frame: int,
    alias_rows: Mapping[str, list[dict[str, Any]]],
    relevant_entity_aliases: tuple[str, ...],
) -> dict[str, Any]:
    rgb = run.load_rgb(frame)
    shape = rgb.shape[:2]
    masks: dict[str, np.ndarray] = {}
    for alias in ALIAS_ORDER:
        rows = alias_rows.get(alias) or []
        if rows:
            mask = union_masks(run, rows, shape)
            if mask.any():
                masks[alias] = mask
    if not masks:
        raise ValueError("no visible accepted masks")
    total = np.zeros(shape, dtype=bool)
    for mask in masks.values():
        total |= mask
    crop_box = expanded_bbox(total)
    x0, y0, x1, y1 = crop_box
    crop_rgb = rgb[y0:y1, x0:x1]
    resize_scale = min(1.0, 1280.0 / max(x1 - x0, y1 - y0))
    alias_short_sides: dict[str, float] = {}
    alias_frame_edge_counts: dict[str, int] = {}
    recognizable_aliases = []
    for alias, mask in masks.items():
        ax0, ay0, ax1, ay1 = bbox_from_mask(mask)
        short_side = min(ax1 - ax0, ay1 - ay0) * resize_scale
        edge_count = sum((ax0 <= 2, ay0 <= 2, ax1 >= shape[1] - 2, ay1 >= shape[0] - 2))
        alias_short_sides[alias] = float(short_side)
        alias_frame_edge_counts[alias] = int(edge_count)
        if short_side >= 128.0 and edge_count <= 1:
            recognizable_aliases.append(alias)
    target_short_side = max(alias_short_sides.values(), default=0.0)
    area_score = clamp(math.sqrt(float(total.sum()) / max(1.0, float(total.size) * 0.12)))
    gray = np.dot(crop_rgb[..., :3], np.array([0.299, 0.587, 0.114]))
    sharp = sharpness_score(gray)
    mean, std = float(gray.mean()) / 255.0, float(gray.std()) / 255.0
    exposure = 0.7 * clamp(1.0 - abs(mean - 0.5) / 0.5) + 0.3 * clamp(std / 0.18)
    boundary_values = []
    for rows in alias_rows.values():
        for row in rows:
            boundary_values.append(float(row.get("boundary_touch_ratio") or 0.0))
    ledger_non_occlusion = clamp(1.0 - (sum(boundary_values) / max(1, len(boundary_values))))
    edge_non_occlusion = 1.0 - sum(min(1.0, value / 2.0) for value in alias_frame_edge_counts.values()) / max(1, len(alias_frame_edge_counts))
    non_occlusion = clamp(0.70 * ledger_non_occlusion + 0.30 * edge_non_occlusion)
    visible = tuple(alias for alias in ALIAS_ORDER if alias in masks)
    entity_visible = set(recognizable_aliases).intersection(relevant_entity_aliases)
    coverage = len(entity_visible) / max(1, len(relevant_entity_aliases))
    quality = 0.30 * area_score + 0.20 * sharp + 0.15 * non_occlusion + 0.10 * exposure + 0.25 * coverage
    return {
        "frame": frame,
        "visible_aliases": visible,
        "alias_rows": {alias: alias_rows[alias] for alias in visible},
        "masks": masks,
        "total_mask": total,
        "crop_box": crop_box,
        "quality": float(quality),
        "quality_terms": {
            "relevant_mask_area": area_score,
            "sharpness": sharp,
            "non_occlusion": non_occlusion,
            "exposure_quality": exposure,
            "relevant_alias_coverage": coverage,
        },
        "target_mask_short_side_px": float(target_short_side),
        "alias_mask_short_side_px": alias_short_sides,
        "alias_frame_edge_counts": alias_frame_edge_counts,
        "recognizable_aliases": tuple(alias for alias in ALIAS_ORDER if alias in recognizable_aliases),
        "eligible": bool(recognizable_aliases and exposure >= 0.15 and non_occlusion >= 0.20),
    }


def get_font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_view(run: FrozenRun, view: Mapping[str, Any], output: Path, role: str, full_frame: bool) -> dict[str, Any]:
    rgb = run.load_rgb(int(view["frame"]))
    canvas = Image.fromarray(rgb).convert("RGB")
    array = np.asarray(canvas, dtype=np.uint8).copy()
    # Paint entity masks first and A last, so the incident mask remains visible.
    for alias in ("E2", "E1", "E0", "A"):
        mask = view["masks"].get(alias)
        if mask is None:
            continue
        color = np.asarray(ALIAS_COLORS[alias], dtype=np.float32)
        array[mask] = np.clip(0.80 * array[mask].astype(np.float32) + 0.20 * color, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(array)
    draw = ImageDraw.Draw(canvas)
    label_font = get_font(22)
    for alias in ("E2", "E1", "E0", "A"):
        mask = view["masks"].get(alias)
        if mask is None:
            continue
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(7))) > 0
        eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(7))) > 0
        outline = dilated & ~eroded
        outline_array = np.asarray(canvas).copy()
        outline_array[outline] = np.asarray(ALIAS_COLORS[alias], dtype=np.uint8)
        canvas = Image.fromarray(outline_array)
        draw = ImageDraw.Draw(canvas)
        x0, y0, _, _ = bbox_from_mask(mask)
        box = draw.textbbox((x0, y0), alias, font=label_font, stroke_width=1)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=ALIAS_COLORS[alias])
        draw.text((x0, y0), alias, fill=(0, 0, 0), font=label_font, stroke_width=0)
    if not full_frame:
        canvas = canvas.crop(tuple(int(v) for v in view["crop_box"]))
    if max(canvas.size) > 1280:
        scale = 1280.0 / max(canvas.size)
        canvas = canvas.resize((max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    role_font = get_font(24)
    role_text = role
    role_box = draw.textbbox((12, 10), role_text, font=role_font)
    draw.rectangle((6, 5, role_box[2] + 18, role_box[3] + 13), fill=(0, 0, 0))
    draw.text((12, 10), role_text, fill=(255, 255, 255), font=role_font)
    canvas.save(output, quality=90, subsampling=0)
    return {"file": output.name, "sha256": sha256_file(output), "width": canvas.width, "height": canvas.height}


def accepted_rows_for_version(run: FrozenRun, version: Mapping[str, Any], freeze_frame: int) -> list[dict[str, Any]]:
    result = []
    for obs_uid in version.get("member_observation_uids") or ():
        row = run.observations.get(str(obs_uid))
        if not row or row.get("status") != "kept" or not row.get("processed_mask_ref"):
            continue
        if frame_index(row.get("frame_uid") or obs_uid) <= freeze_frame:
            result.append(row)
    return result


def semantic_hypotheses(alias_rows: Mapping[str, list[dict[str, Any]]], current_label: str, incident_hash: str) -> list[dict[str, Any]]:
    aliases_per_label: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for alias, rows in alias_rows.items():
        for row in rows:
            label = normalize_label(row.get("class_name"))
            if not label or label == current_label or label in {"wall", "floor", "ceiling"}:
                continue
            aliases_per_label[label].add(alias)
            counts[label] += 1
    labels = sorted(
        counts,
        key=lambda label: (-len(aliases_per_label[label]), -counts[label], hashlib.sha256(f"{incident_hash}:{label}".encode()).hexdigest()),
    )[:3]
    return [
        {"id": f"L{index}", "text": label, "source": "normalized accepted-view label evidence; machine hypothesis"}
        for index, label in enumerate(labels, 1)
    ]


def action_candidates(decision: str, aliases: Mapping[str, Mapping[str, Any]], alternatives: list[dict[str, Any]], current_label: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = [
        {"id": "H0", "axis": "NONE", "action": "NO_OP", "meaning": "current instance assignment and stable label are acceptable"}
    ]
    next_id = 1
    if decision == "CREATE_OBJECT":
        for alias in ("E1", "E2"):
            if alias in aliases:
                values.append({
                    "id": f"H{next_id}", "axis": "IDENTITY", "action": "SAME_INSTANCE",
                    "parameters": {"entities": ["A", alias]}, "executable": True,
                })
                next_id += 1
    else:
        for alias in ("E1", "E2"):
            if alias in aliases and aliases[alias].get("object_uid") != aliases.get("E0", {}).get("object_uid"):
                values.append({
                    "id": f"H{next_id}", "axis": "IDENTITY", "action": "MOVE_OBSERVATION",
                    "parameters": {"observation": "A", "from": "E0", "to": alias}, "executable": True,
                })
                next_id += 1
        if "E0" in aliases:
            values.append({
                "id": f"H{next_id}", "axis": "IDENTITY", "action": "SEPARATE_MEMBER_GROUPS",
                "parameters": {"groups": [["A"], ["E0"]]}, "executable": True,
            })
            next_id += 1
    for item in alternatives:
        values.append({
            "id": f"H{next_id}", "axis": "SEMANTIC_LABEL", "action": "RELABEL_ENTITY",
            "parameters": {"entity": "E0", "from_label": "L0", "to_label": item["id"]},
            "label_text": {"from": current_label, "to": item["text"]},
            "executable": False, "mode": "SHADOW_ONLY",
        })
        next_id += 1
    values.append({"id": "DEFER", "axis": "NONE", "action": "REQUEST_MORE_EVIDENCE"})
    return values


def output_schema(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "selected_candidate", "decision_axis", "confidence_diagnostic", "evidence_ids",
            "reason", "counterevidence", "needed_evidence", "suggested_label_for_logging",
        ],
        "properties": {
            "selected_candidate": {"type": "string", "enum": candidate_ids},
            "decision_axis": {"type": "string", "enum": ["IDENTITY", "SEMANTIC_LABEL", "NONE"]},
            "confidence_diagnostic": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence_ids": {
                "type": "array", "minItems": 1, "maxItems": 3,
                "items": {"type": "string", "enum": ["I1", "I2", "I3"]},
            },
            "reason": {"type": "string", "minLength": 1, "maxLength": 320},
            "counterevidence": {"type": "string", "minLength": 1, "maxLength": 240},
            "needed_evidence": {
                "type": "array",
                "items": {"type": "string", "enum": list(NEEDED_EVIDENCE)},
            },
            "suggested_label_for_logging": {"type": ["string", "null"], "maxLength": 80},
        },
    }


def finding_text(issue: Mapping[str, Any]) -> tuple[str, str, str]:
    family = str(issue.get("family") or "UNKNOWN")
    signals = issue.get("raw_signals") or {}
    source = "semantic_checker" if "SEMANTIC" in family else "identity_checker"
    axis = "UNKNOWN" if "SEMANTIC_ASSOCIATION_CONFLICT" in family else "IDENTITY"
    compact = json.dumps(signals, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    message = f"{family} opened this review from online association signals {compact}. This is a trigger, not a diagnosis."
    return source, axis, message[:500]


def overlay_details(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    details = []
    for alias in ALIAS_ORDER:
        rows = view["alias_rows"].get(alias) or []
        if not rows:
            continue
        details.append({
            "alias": alias,
            "color": ALIAS_COLOR_NAMES[alias],
            "accepted_observation_ids": [str(row["obs_uid"]) for row in rows],
            "source": "accepted processed_mask_ref at or before the frozen snapshot",
        })
    return details


def image_spec_text(image_id: str, view: Mapping[str, Any], snapshot_seq: int, extra: Mapping[str, Any]) -> str:
    role = {"I1": "DECISION_CONTEXT", "I2": "BEST_DETAIL", "I3": "DIVERSE_VIEW"}[image_id]
    if image_id == "I1":
        reason = "exact frame in which the incident assignment/decision occurred"
        scope = "full RGB frame with global scene context"
        intended = "inspect global layout, co-visibility, occlusion, mask boundaries, and the decision-time assignment context"
    elif image_id == "I2":
        reason = "highest-quality accepted support view after mask area, sharpness, non-occlusion, exposure, and relevant-alias coverage scoring"
        scope = "context crop with 15% margin around relevant accepted masks; it is not a mask-only crop"
        intended = "inspect visible parts, shape, material cues, instance continuity, and base-category discriminative features"
    else:
        reason = "highest-quality accepted view that maximizes camera-pose diversity and uncovered relevant-alias gain relative to I1/I2"
        scope = "context crop with 15% margin"
        intended = "test cross-view physical consistency, reject single-view occlusion/perspective mistakes, and verify semantic discriminative features"
    payload: dict[str, Any] = {
        "image_id": image_id,
        "role": role,
        "frame_id": int(view["frame"]),
        "snapshot_seq": snapshot_seq,
        "selection_reason": reason,
        "view_scope": scope,
        "visible_aliases": list(view["visible_aliases"]),
        "overlay_details": overlay_details(view),
        "mask_semantics": "every shown overlay is an accepted post-processed mask at this frozen snapshot; it is a visual pointer, not ground truth, and is neither a raw proposal nor future membership",
        "absence_rule": "aliases not listed above are unobserved in this frame, not proven absent",
        "intended_use": intended,
    }
    payload.update(extra)
    if image_id == "I2":
        payload["neutrality_rule"] = "being selected as the clearest view does not mean this image supports the current state or any repair candidate"
    return f"IMAGE_SPEC {image_id}\n" + json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + f"\nThe next image is {image_id} and only {image_id}."


def build_incident_text(
    ticket_uid: str,
    snapshot_id: str,
    snapshot_seq: int,
    issue: Mapping[str, Any],
    anchor_uid: str,
    aliases: Mapping[str, Mapping[str, Any]],
    current_label: str,
    alternatives: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    source, axis, message = finding_text(issue)
    entities = {alias: row.get("object_uid") for alias, row in aliases.items()}
    semantic = [{"id": "L0", "text": current_label, "source": "current machine-maintained stable label; not ground truth"}] + alternatives
    return "\n".join([
        "INCIDENT",
        f"incident_id: {ticket_uid}",
        f"prompt_version: {PROMPT_VERSION}",
        f"snapshot_id: {snapshot_id}",
        f"snapshot_seq: {snapshot_seq}",
        f"finding_source: {source}",
        f"suspected_axis: {axis}",
        f"finding_message: {message}",
        "Important: the finding only explains why this case was opened; it is not a correct diagnosis.",
        "",
        "CURRENT STATE",
        f"A is observation {anchor_uid}.",
        f"A is currently assigned to E0={entities.get('E0', 'none')}.",
        f"E1={entities.get('E1', 'none')}; E2={entities.get('E2', 'none')}.",
        f"E0 current stable label is L0=\"{current_label}\". L0 is a machine-maintained hypothesis, not ground truth.",
        "",
        "SEMANTIC LABEL HYPOTHESES",
        json.dumps(semantic, indent=2, ensure_ascii=False, sort_keys=True),
        "These are finite machine hypotheses. If none fits, choose DEFER; any new label may only be written to suggested_label_for_logging.",
        "",
        "ACTION CANDIDATES",
        json.dumps(candidates, indent=2, ensure_ascii=False, sort_keys=True),
    ])


def final_rules_text(schema: Mapping[str, Any]) -> str:
    return """FINAL DECISION RULES
- First test identity and mask purity. Only then test the stable semantic label.
- Prefer an identity candidate when mixed/split instances explain the apparent label problem.
- RELABEL requires a coherent entity and visible positive support from at least two complementary images.
- H0 means sufficient evidence supports no change; DEFER means insufficient evidence or no valid listed action.
- Select exactly one candidate and return JSON only.

OUTPUT SCHEMA
""" + json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True)


def prepare_case(run: FrozenRun, ticket_uid: str, output_root: Path) -> PreparedCase:
    source_dir = run.root / "online_mvp" / "vlm" / ticket_uid
    manifest_path = source_dir / "evidence" / "packet_manifest.json"
    if not manifest_path.is_file():
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", f"missing {manifest_path}")
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze_frame = int(old_manifest["freeze_frame"])
    freeze_sequence = int(old_manifest["freeze_sequence"])
    issue_uid = str(old_manifest["issue_uid"])
    association = run.associations.get(str(old_manifest["anchor_event_uid"]))
    if not association or int(association.get("event_sequence", 10**12)) > freeze_sequence:
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "anchor association is missing or after cutoff")
    anchor_uid = str(association.get("obs_uid") or "")
    anchor = run.observations.get(anchor_uid)
    if not anchor or anchor.get("status") != "kept" or not anchor.get("processed_mask_ref"):
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "anchor accepted processed mask is unavailable")
    incident_frame = frame_index(anchor.get("frame_uid") or anchor_uid)
    if incident_frame > freeze_frame or incident_frame not in run.frames:
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "anchor frame is outside the frozen snapshot")
    issue = run.issue_at_cutoff(ticket_uid, issue_uid, freeze_frame, freeze_sequence)

    aliases: dict[str, dict[str, Any]] = {}
    old_to_new = {"CURRENT_ENTITY_CONTEXT": "E0", "CANDIDATE_1_CONTEXT": "E1", "CANDIDATE_2_CONTEXT": "E2"}
    for old_alias, new_alias in old_to_new.items():
        version_uid = (old_manifest.get("alias_version_uids") or {}).get(old_alias)
        version = run.versions.get(str(version_uid)) if version_uid else None
        if version and run.version_is_at_cutoff(version, freeze_frame, freeze_sequence):
            aliases[new_alias] = version
    target_uid = str(association.get("target_object_uid") or "")
    if "E0" not in aliases and target_uid:
        target_version = run.latest_object_version(target_uid, freeze_frame, freeze_sequence)
        if target_version:
            aliases["E0"] = target_version
    if "E0" not in aliases:
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "current owner E0 cannot be reconstructed at cutoff")

    alias_rows: dict[str, list[dict[str, Any]]] = {"A": [anchor]}
    for alias in ("E0", "E1", "E2"):
        if alias in aliases:
            rows = accepted_rows_for_version(run, aliases[alias], freeze_frame)
            alias_rows[alias] = [row for row in rows if str(row["obs_uid"]) != anchor_uid]
    # Detect incompatible alias snapshots instead of silently painting one accepted mask twice.
    owners: dict[str, list[str]] = defaultdict(list)
    for alias in ("E0", "E1", "E2"):
        for row in alias_rows.get(alias, ()):
            owners[str(row["obs_uid"])].append(alias)
    overlaps = {uid: values for uid, values in owners.items() if len(values) > 1}
    if overlaps:
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", f"alias snapshots overlap on {len(overlaps)} observations")

    by_frame: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for alias, rows in alias_rows.items():
        for row in rows:
            frame = frame_index(row.get("frame_uid") or row["obs_uid"])
            if frame <= freeze_frame and frame in run.frames and run.frames[frame].get("pose") is not None:
                by_frame[frame][alias].append(row)
    if incident_frame not in by_frame or not by_frame[incident_frame].get("A"):
        raise PreflightDefer("DEFER_MISSING_DECISION_PROVENANCE", "incident view cannot bind A to processed mask")
    relevant_entities = tuple(alias for alias in ("E0", "E1", "E2") if alias in aliases)
    i1 = build_view_record(run, incident_frame, by_frame[incident_frame], relevant_entities)
    pool = []
    for frame, rows in by_frame.items():
        if frame == incident_frame:
            continue
        try:
            view = build_view_record(run, frame, rows, relevant_entities)
        except (FileNotFoundError, ValueError, PreflightDefer):
            continue
        if view["eligible"]:
            pool.append(view)
    if len(pool) < 2:
        raise PreflightDefer("DEFER_INSUFFICIENT_VIEWS", f"only {len(pool) + 1} unique qualified accepted views")
    i2 = max(pool, key=lambda item: (float(item["quality"]), -int(item["frame"])))
    candidates_i3 = []
    covered = set(i1["visible_aliases"]) | set(i2["visible_aliases"])
    for view in pool:
        if int(view["frame"]) == int(i2["frame"]):
            continue
        b1, a1, p1 = pose_difference(run.frames[int(view["frame"])], run.frames[int(i1["frame"])])
        b2, a2, p2 = pose_difference(run.frames[int(view["frame"])], run.frames[int(i2["frame"])])
        diversity = min(p1, p2)
        # Different frame IDs are not enough: reject nearly identical camera views.
        if diversity < 0.08:
            continue
        uncovered = len(set(view["visible_aliases"]) - covered) / max(1, len(relevant_entities))
        score = 0.55 * float(view["quality"]) + 0.30 * diversity + 0.15 * uncovered
        enriched = dict(view)
        enriched.update({
            "diverse_score": score,
            "pose_diversity_min_to_i1_i2": diversity,
            "baseline_to_i1_m": b1,
            "view_angle_to_i1_deg": a1,
            "baseline_to_i2_m": b2,
            "view_angle_to_i2_deg": a2,
            "newly_covered_aliases": tuple(alias for alias in ALIAS_ORDER if alias in set(view["visible_aliases"]) - covered),
        })
        candidates_i3.append(enriched)
    if not candidates_i3:
        raise PreflightDefer("DEFER_INSUFFICIENT_VIEWS", "no third distinct qualified accepted view")
    i3 = max(candidates_i3, key=lambda item: (float(item["diverse_score"]), -int(item["frame"])))
    if len({int(i1["frame"]), int(i2["frame"]), int(i3["frame"])}) != 3:
        raise PreflightDefer("DEFER_INSUFFICIENT_VIEWS", "view selector did not produce three distinct frames")

    case_dir = output_root / ticket_uid
    case_dir.mkdir(parents=True, exist_ok=True)
    image_records = {
        "I1": render_view(run, i1, case_dir / "I1_DECISION.jpg", "I1 DECISION", True),
        "I2": render_view(run, i2, case_dir / "I2_DETAIL.jpg", "I2 DETAIL", False),
        "I3": render_view(run, i3, case_dir / "I3_DIVERSE.jpg", "I3 DIVERSE", False),
    }

    current_label = normalize_label(aliases["E0"].get("class_name")) or normalize_label(anchor.get("class_name")) or "unknown"
    alternatives = semantic_hypotheses(alias_rows, current_label, ticket_uid)
    candidates = action_candidates(str(association.get("decision") or ""), aliases, alternatives, current_label)
    schema = output_schema([str(item["id"]) for item in candidates])
    snapshot_id = f"{ticket_uid}:f{freeze_frame}:s{freeze_sequence}"
    incident = build_incident_text(ticket_uid, snapshot_id, freeze_sequence, issue, anchor_uid, aliases, current_label, alternatives, candidates)
    spec1 = image_spec_text("I1", i1, freeze_sequence, {})
    spec2 = image_spec_text("I2", i2, freeze_sequence, {"quality_score": round(float(i2["quality"]), 6), "quality_terms": i2["quality_terms"]})
    spec3 = image_spec_text("I3", i3, freeze_sequence, {
        "view_difference": {
            "baseline_to_I1_m": round(float(i3["baseline_to_i1_m"]), 4),
            "orientation_delta_to_I1_deg": round(float(i3["view_angle_to_i1_deg"]), 2),
            "minimum_pose_diversity_to_I1_I2": round(float(i3["pose_diversity_min_to_i1_i2"]), 6),
        },
        "newly_covered_aliases": list(i3["newly_covered_aliases"]) or ["none"],
        "quality_score": round(float(i3["quality"]), 6),
        "diverse_score": round(float(i3["diverse_score"]), 6),
    })
    final = final_rules_text(schema)
    sequence = (
        {"type": "text", "text": incident},
        {"type": "text", "text": spec1},
        {"type": "image", "path": str(case_dir / image_records["I1"]["file"]), "image_id": "I1"},
        {"type": "text", "text": spec2},
        {"type": "image", "path": str(case_dir / image_records["I2"]["file"]), "image_id": "I2"},
        {"type": "text", "text": spec3},
        {"type": "image", "path": str(case_dir / image_records["I3"]["file"]), "image_id": "I3"},
        {"type": "text", "text": final},
    )
    audit_content = []
    for item in sequence:
        if item["type"] == "text":
            audit_content.append({"type": "text", "text": item["text"]})
        else:
            record = image_records[item["image_id"]]
            audit_content.append({
                "type": "image_url",
                "image_url": {"url": f"[REDACTED_BASE64] local_file={record['file']} sha256={record['sha256']}", "detail": "high"},
            })
    request_audit = {
        "model": "set at runtime",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": audit_content}],
        "max_completion_tokens": 1200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "ali_my_unified_v1", "strict": True, "schema": schema}},
        "stream": False,
        "store": False,
        "credential_storage": "API key was read from a process environment variable and is absent from this artifact",
    }
    write_json(case_dir / "actual_request_redacted.json", request_audit)
    write_json(case_dir / "action_candidates.json", candidates)
    write_json(case_dir / "strict_output_schema.json", schema)
    (case_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    user_review = []
    for item in sequence:
        if item["type"] == "text":
            user_review.append(item["text"])
        else:
            record = image_records[item["image_id"]]
            user_review.append(f"[{item['image_id']} IMAGE: {record['file']} sha256={record['sha256']}]")
    (case_dir / "user_prompt_interleaved.txt").write_text("\n\n".join(user_review) + "\n", encoding="utf-8")

    used_rows = {str(row["obs_uid"]): row for rows in alias_rows.values() for row in rows}
    used_versions = {alias: str(row["object_version_uid"]) for alias, row in aliases.items()}
    case_manifest = {
        "schema_version": "ali_my_unified_validation/1.0",
        "ticket_uid": ticket_uid,
        "source_online_experiment": str(run.root),
        "source_frozen_packet": str(manifest_path),
        "prompt_version": PROMPT_VERSION,
        "freeze_frame": freeze_frame,
        "freeze_sequence": freeze_sequence,
        "anchor_event_uid": association["event_uid"],
        "anchor_observation_uid": anchor_uid,
        "issue_uid": issue_uid,
        "issue_family": issue.get("family"),
        "association_decision": association.get("decision"),
        "aliases": {
            alias: {"object_uid": row.get("object_uid"), "object_version_uid": row.get("object_version_uid"), "stable_label_hypothesis": row.get("class_name")}
            for alias, row in aliases.items()
        },
        "semantic_label_hypotheses": [{"id": "L0", "text": current_label}] + alternatives,
        "images": {
            "I1": {**image_records["I1"], "frame": i1["frame"], "visible_aliases": list(i1["visible_aliases"])},
            "I2": {**image_records["I2"], "frame": i2["frame"], "visible_aliases": list(i2["visible_aliases"]), "quality": i2["quality"], "quality_terms": i2["quality_terms"]},
            "I3": {**image_records["I3"], "frame": i3["frame"], "visible_aliases": list(i3["visible_aliases"]), "quality": i3["quality"], "quality_terms": i3["quality_terms"], "diverse_score": i3["diverse_score"]},
        },
        "cutoff_audit": {
            "maximum_observation_frame_used": max(frame_index(row.get("frame_uid") or uid) for uid, row in used_rows.items()),
            "maximum_object_version_trigger_sequence_used": max(event_sequence(row["trigger_event_uid"]) for row in aliases.values()),
            "all_observation_frames_lte_freeze_frame": all(frame_index(row.get("frame_uid") or uid) <= freeze_frame for uid, row in used_rows.items()),
            "all_version_events_lte_freeze_sequence": all(event_sequence(row["trigger_event_uid"]) <= freeze_sequence for row in aliases.values()),
            "used_object_versions": used_versions,
            "mask_source_required": "processed_mask_ref",
            "forbidden_sources_used": [],
            "final_membership_read": False,
            "old_vlm_response_read": False,
            "old_compilation_read": False,
        },
    }
    write_json(case_dir / "case_manifest.json", case_manifest)
    return PreparedCase(ticket_uid, case_dir, schema, tuple(candidates), sequence)


def request_content(case: PreparedCase) -> list[dict[str, Any]]:
    result = []
    for item in case.message_sequence:
        if item["type"] == "text":
            result.append({"type": "text", "text": item["text"]})
        else:
            path = Path(item["path"])
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            result.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}})
    return result


def validate_output(value: Any, candidates: tuple[dict[str, Any], ...]) -> list[str]:
    errors = []
    required = {
        "selected_candidate", "decision_axis", "confidence_diagnostic", "evidence_ids",
        "reason", "counterevidence", "needed_evidence", "suggested_label_for_logging",
    }
    if not isinstance(value, dict):
        return ["output is not a JSON object"]
    if set(value) != required:
        errors.append(f"keys must equal {sorted(required)}; got {sorted(value)}")
    by_id = {str(item["id"]): item for item in candidates}
    selected = value.get("selected_candidate")
    if selected not in by_id:
        errors.append("selected_candidate is not in this case's finite candidate table")
    axis = value.get("decision_axis")
    if axis not in {"IDENTITY", "SEMANTIC_LABEL", "NONE"}:
        errors.append("invalid decision_axis")
    if selected in by_id and axis != by_id[selected]["axis"]:
        errors.append(f"decision_axis {axis!r} does not match candidate axis {by_id[selected]['axis']!r}")
    confidence = value.get("confidence_diagnostic")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence_diagnostic must be a number in [0,1]")
    evidence = value.get("evidence_ids")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3 or len(evidence) != len(set(evidence)) or any(item not in {"I1", "I2", "I3"} for item in evidence):
        errors.append("evidence_ids must contain 1-3 unique IDs from I1/I2/I3")
    for key, limit in (("reason", 320), ("counterevidence", 240)):
        item = value.get(key)
        if not isinstance(item, str) or not item or len(item) > limit:
            errors.append(f"{key} must be a nonempty string of at most {limit} characters")
    needed = value.get("needed_evidence")
    if not isinstance(needed, list) or len(needed) != len(set(needed)) or any(item not in NEEDED_EVIDENCE for item in needed):
        errors.append("needed_evidence contains an invalid or duplicate controlled value")
    suggestion = value.get("suggested_label_for_logging")
    if suggestion is not None and (not isinstance(suggestion, str) or len(suggestion) > 80):
        errors.append("suggested_label_for_logging must be null or a string up to 80 characters")
    if selected != "DEFER" and suggestion is not None:
        errors.append("suggested_label_for_logging must be null unless selected_candidate is DEFER")
    return errors


def call_vlm(case: PreparedCase, api_key: str, base_url: str, model: str, timeout: float) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request_content(case)}],
        "max_completion_tokens": 1200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "ali_my_unified_v1", "strict": True, "schema": case.schema}},
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
            data=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ali-my-unified-vlm-validation/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_bytes = response.read()
        elapsed = time.monotonic() - started
        raw = json.loads(raw_bytes)
        write_json(case.case_dir / "vlm_raw_response.json", raw)
        choices = raw.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(content, str):
            raise ValueError("VLM response has no string message content")
        # Structured output must be a bare JSON object; do not repair fences or prose.
        output = json.loads(content)
        write_json(case.case_dir / "vlm_output.json", output)
        errors = validate_output(output, case.candidates)
        validation = {
            "status": "VALID" if not errors else "DEFER_INVALID_OUTPUT",
            "errors": errors,
            "elapsed_seconds": elapsed,
            "model": raw.get("model") or model,
            "response_id": raw.get("id"),
            "usage": raw.get("usage") or {},
            "single_vlm_call": True,
            "schema_mode": "json_schema_strict",
        }
        write_json(case.case_dir / "validation.json", validation)
        return {"ticket_uid": case.ticket_uid, "status": validation["status"], "output": output, **{key: validation[key] for key in ("elapsed_seconds", "model", "usage")}}
    except urllib.error.HTTPError as exc:
        error_body = exc.read(4000).decode("utf-8", "replace")
        error = safe_error(RuntimeError(f"HTTP {exc.code}: {error_body}"))
    except Exception as exc:  # Each case must retain its own failure artifact.
        error = safe_error(exc)
    elapsed = time.monotonic() - started
    failure = {"status": "API_OR_PARSE_ERROR", "error": error, "elapsed_seconds": elapsed, "single_vlm_call": True, "retry_count": 0}
    write_json(case.case_dir / "validation.json", failure)
    return {"ticket_uid": case.ticket_uid, **failure}


def case_review_html(case_dir: Path) -> None:
    manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    request = json.loads((case_dir / "actual_request_redacted.json").read_text(encoding="utf-8"))
    validation = json.loads((case_dir / "validation.json").read_text(encoding="utf-8")) if (case_dir / "validation.json").is_file() else {"status": "PREPARED_ONLY"}
    output = json.loads((case_dir / "vlm_output.json").read_text(encoding="utf-8")) if (case_dir / "vlm_output.json").is_file() else None
    candidates = json.loads((case_dir / "action_candidates.json").read_text(encoding="utf-8"))
    def block(value: Any) -> str:
        return html.escape(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
    images = "".join(
        f'<figure><img src="{name}" alt="{image_id}"><figcaption>{image_id}: frame {manifest["images"][image_id]["frame"]}; visible {html.escape(str(manifest["images"][image_id]["visible_aliases"]))}</figcaption></figure>'
        for image_id, name in (("I1", "I1_DECISION.jpg"), ("I2", "I2_DETAIL.jpg"), ("I3", "I3_DIVERSE.jpg"))
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(case_dir.name)}</title>
<style>body{{font:15px/1.55 system-ui;margin:28px;max-width:1500px}}.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}figure{{margin:0}}img{{width:100%;height:auto;border:1px solid #bbb}}pre{{white-space:pre-wrap;word-break:break-word;background:#f5f5f5;padding:14px;border-radius:7px}}.valid{{color:#087a29}}@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}</style></head><body>
<h1>{html.escape(case_dir.name)}</h1><p>状态：<strong class="valid">{html.escape(str(validation.get("status")))}</strong>。三张图均来自冻结在线快照的 accepted processed mask。</p>
<div class="images">{images}</div>
<h2>VLM 实际输出</h2><pre>{block(output)}</pre>
<h2>解析校验与延迟</h2><pre>{block(validation)}</pre>
<h2>候选动作（VLM 只能选择 ID）</h2><pre>{block(candidates)}</pre>
<h2>冻结与防未来泄露审计</h2><pre>{block(manifest["cutoff_audit"])}</pre>
<details open><summary><strong>实际发送请求（图片 base64 已替换为文件名和 hash）</strong></summary><pre>{block(request)}</pre></details>
<p><a href="user_prompt_interleaved.txt">逐图交错 User Prompt</a> · <a href="system_prompt.txt">System Prompt</a> · <a href="vlm_raw_response.json">API 原始响应</a></p>
</body></html>"""
    (case_dir / "review.html").write_text(page, encoding="utf-8")


def root_review_html(output_root: Path, results: list[dict[str, Any]], elapsed: float) -> None:
    rows = []
    for item in sorted(results, key=lambda value: value["ticket_uid"]):
        output = item.get("output") or {}
        rows.append(
            "<tr>" + "".join([
                f'<td><a href="{html.escape(item["ticket_uid"])}/review.html">{html.escape(item["ticket_uid"])}</a></td>',
                f'<td>{html.escape(str(item.get("status")))}</td>',
                f'<td>{html.escape(str(output.get("selected_candidate", "-")))}</td>',
                f'<td>{html.escape(str(output.get("decision_axis", "-")))}</td>',
                f'<td>{html.escape(str(output.get("reason", item.get("detail", item.get("error", "-")))))}</td>',
                f'<td>{float(item.get("elapsed_seconds", 0.0)):.2f}s</td>',
            ]) + "</tr>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>ali-my unified VLM V1 validation</title>
<style>body{{font:15px/1.5 system-ui;margin:30px;max-width:1500px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:8px;text-align:left;vertical-align:top}}th{{background:#eee}}code{{background:#eee;padding:2px 4px}}</style></head><body>
<h1>ali-my unified VLM V1：在线冻结证据验证</h1><p>本轮只验证 VLM 输入与输出，不做 replay 或地图修改。总墙钟时间 {elapsed:.2f}s。</p>
<table><thead><tr><th>Case</th><th>校验状态</th><th>选择</th><th>轴</th><th>简短理由/错误</th><th>API 延迟</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>点击每个 case 可查看三张实际输入图、完整交错提示词、有限候选、VLM 原始输出和 cutoff 审计。</p>
</body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ticket", action="append", required=True, dest="tickets")
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--api-key-env-prefix", default="ALI_VLM_API_KEY_")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.output_root.mkdir(parents=True, exist_ok=False)
    run = FrozenRun(args.experiment_root)
    prepared: list[PreparedCase] = []
    results: list[dict[str, Any]] = []
    for ticket in args.tickets:
        try:
            case = prepare_case(run, ticket, args.output_root)
            prepared.append(case)
        except PreflightDefer as exc:
            case_dir = args.output_root / ticket
            case_dir.mkdir(parents=True, exist_ok=True)
            value = {"ticket_uid": ticket, "status": exc.code, "detail": exc.detail, "vlm_called": False}
            write_json(case_dir / "preflight_defer.json", value)
            results.append(value)
    if args.prepare_only:
        for case in prepared:
            write_json(case.case_dir / "validation.json", {"status": "PREPARED_ONLY", "vlm_called": False})
            case_review_html(case.case_dir)
            results.append({"ticket_uid": case.ticket_uid, "status": "PREPARED_ONLY"})
    else:
        keys = [os.environ.get(f"{args.api_key_env_prefix}{index}", "") for index in range(1, len(prepared) + 1)]
        if any(not key for key in keys):
            raise RuntimeError(f"need one nonempty in-memory API key per prepared case using {args.api_key_env_prefix}1..N")
        with ThreadPoolExecutor(max_workers=max(1, len(prepared))) as executor:
            futures = {
                executor.submit(call_vlm, case, keys[index], args.base_url, args.model, args.timeout_seconds): case
                for index, case in enumerate(prepared)
            }
            for future in as_completed(futures):
                results.append(future.result())
        for case in prepared:
            case_review_html(case.case_dir)
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": "ali_my_unified_validation_run/1.0",
        "experiment_root": str(args.experiment_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "tickets_requested": args.tickets,
        "prepared_case_count": len(prepared),
        "vlm_call_count": 0 if args.prepare_only else len(prepared),
        "parallel_api_count": 0 if args.prepare_only else len(prepared),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "total_wall_seconds": elapsed,
        "results": sorted(results, key=lambda value: value["ticket_uid"]),
        "scope": "VLM evidence and judgment only; no replay, constraint execution, or map mutation",
    }
    write_json(args.output_root / "run_summary.json", summary)
    root_review_html(args.output_root, results, elapsed)
    print(json.dumps({"output_root": str(args.output_root), "prepared": len(prepared), "results": results, "wall_seconds": elapsed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
