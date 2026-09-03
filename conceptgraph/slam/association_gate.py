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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import httpx
import numpy as np
import openai
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


SCHEMA_VERSION = "blocking-association-gate-v1.0"
VALID_MODES = {"off", "audit", "oracle", "vlm"}
VALID_SCOPES = {"create_only", "both"}
ALIASES = ("A", "B", "C")
DISCARD_MATCH_INDEX = -1

IMAGE_READING_POLICY = """How to read the evidence images:
- The semi-transparent RED overlay marks the object mask to judge. The red-highlighted pixels are the target; unhighlighted pixels are background/context only.
- I1 is the full CURRENT scene. Its red mask, additionally localized by a yellow rectangle, is the current observation under review.
- I1-crop is a closer view of that same current red-masked observation; surrounding unmasked pixels provide local context only.
- Each Candidate image is one frozen historical view of that candidate map object. Its red mask marks the candidate object to compare against the current red mask.
- Dark letterbox padding and the rendered CURRENT/CANDIDATE labels are presentation aids, not scene evidence.
Judge physical-object identity primarily from the red-masked regions. Use unmasked background only as secondary spatial/context evidence. Never select a candidate merely because the images show the same room, nearby furniture, or a similar background. Allow normal changes in viewpoint, scale, lighting, partial occlusion, and mask boundary accuracy."""

INSTANCE_IDENTITY_POLICY = """This is physical INSTANCE re-identification, not category classification.
Two objects may have the same class name, semantic category, color, and general appearance while still being different physical instances. Category agreement is only a weak contextual cue and is never sufficient for association. Conversely, category or label disagreement alone is not decisive because detector naming can vary across views.
Select a candidate only when the current red-masked object and the candidate red-masked object are supported as the very same individual physical object observed at different times or viewpoints. Look for a consistent combination of instance-level cues: exact geometry and proportions, arrangement of parts, distinctive texture or pattern, material and color details, wear, damage or other unique marks, plus compatible placement in the scene. Do not rely on one generic cue shared by all objects of that category.
For repeated or near-identical objects, be conservative. If visible instance-level cues contradict a candidate, do not associate with it. If the current object is usable but the views lack enough instance-specific evidence to distinguish the same instance from another object of the same category, choose UNCERTAIN rather than guessing. Choose NEW only when the usable current observation is supported as a different instance from every listed candidate.
If a candidate is selected, the reason must cite instance-specific cross-view evidence. A reason based only on a matching class, label, or object type is invalid."""

QUALITY_POLICY = """Apply a mandatory CURRENT-observation quality gate before any identity comparison.
Stage 1 must use only I1 and I1-crop. Temporarily ignore every candidate image, category label, similarity score, and likely identity. First assign exactly one observation_quality value:
- USABLE: the red mask represents one coherent physical instance; sufficient surface information is visible for it to serve as reliable map evidence.
- MIXED_INSTANCES: the red mask materially covers parts of two or more distinct physical instances, excluding masks that are fragmented merely due to occlusion of the same physical object instance.
- SEVERE_FRAGMENT: the red mask is only an isolated part, surface patch, thin strip, corner, or disconnected fragments rather than a coherent observation of the object.
- BORDERLINE: it is genuinely unclear whether the red mask is a coherent single-object observation or one of the two defects above.
The quality gate has veto priority over identity matching:
- MIXED_INSTANCES or SEVERE_FRAGMENT => choose DISCARD and stop.
- BORDERLINE => choose UNCERTAIN and stop. Do not guess a candidate, NEW, or DISCARD.
- Only USABLE permits candidate comparison and an A/B/C or NEW identity decision.
Ordinary occlusion, image-boundary truncation, viewpoint change, or small boundary leakage can still be USABLE when the red mask preserves a coherent substantial portion of exactly one physical instance. Candidate-image weakness is not a reason to discard a USABLE current observation.
For DISCARD, the reason must name MIXED_INSTANCES or SEVERE_FRAGMENT and describe the visible mask defect. For quality-driven UNCERTAIN, state why the mask is borderline. For every other UNCERTAIN, state which instance-specific cross-view evidence is missing."""

ASSOCIATION_SYSTEM_PROMPT = """You are an instance-association adjudicator for an online 3D map.
The current observation and candidate history images are all frozen before the current observation is fused.
{image_reading_policy}
{quality_policy}
{instance_identity_policy}
Follow the quality-gate decision tree exactly. Do not inspect or use candidates until observation_quality is USABLE. Then compare the current red-masked object with candidates A and B and make the identity decision.
For identity, compare target-level category, shape, parts, color/material/texture, and spatial context while allowing cross-view appearance changes.
When observation_quality is USABLE, choose A or B only when the visual evidence supports the same physical instance. Choose NEW only when the usable observation is supported as different from both candidates. Choose UNCERTAIN when identity evidence is insufficient.
The reason must briefly cite visible target-level evidence; for NEW, state the decisive mismatch, and for UNCERTAIN, state what evidence is missing. Do not infer the mapper's original decision. Return only the required JSON object."""

ASSOCIATION_SYSTEM_PROMPT = ASSOCIATION_SYSTEM_PROMPT.format(
    image_reading_policy=IMAGE_READING_POLICY,
    quality_policy=QUALITY_POLICY,
    instance_identity_policy=INSTANCE_IDENTITY_POLICY,
)

CREATE_SYSTEM_PROMPT = """You are a new-object adjudicator for an online 3D map.
The current observation and all candidate history images are frozen before the current observation is fused.
{image_reading_policy}
{quality_policy}
{instance_identity_policy}
Follow the quality-gate decision tree exactly. Do not inspect or use candidates until observation_quality is USABLE. Then compare the current red-masked object with every listed candidate red mask and decide whether it belongs to one candidate or is a new object.
For identity, compare target-level category, shape, parts, color/material/texture, and spatial context while allowing cross-view appearance changes.
When observation_quality is USABLE, choose a candidate only when the visual evidence supports the same physical instance. Choose NEW only when the usable observation is supported as different from every candidate. Choose UNCERTAIN when identity evidence is insufficient.
The reason must briefly cite visible target-level evidence; for NEW, state the decisive mismatch from the candidates, and for UNCERTAIN, state what evidence is missing. Do not infer the mapper's original decision. Return only the required JSON object."""

CREATE_SYSTEM_PROMPT = CREATE_SYSTEM_PROMPT.format(
    image_reading_policy=IMAGE_READING_POLICY,
    quality_policy=QUALITY_POLICY,
    instance_identity_policy=INSTANCE_IDENTITY_POLICY,
)


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
    crop = cv2.copyMakeBorder(crop, 30, border, border, border, cv2.BORDER_CONSTANT, value=(20, 20, 20))
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
        self.started_at = _utc_now()
        self.stats = Counter()
        self.events: List[dict] = []
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
        }
        _json_dump(self.output_dir / "config.json", self.config)
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
    def _best_member(obj: Mapping[str, Any]) -> int:
        masks = list(obj.get("mask", []))
        areas = []
        for mask in masks:
            try:
                areas.append(float(_as_numpy(mask).astype(bool).sum()))
            except Exception:
                areas.append(-1.0)
        return int(np.argmax(areas)) if areas else 0

    def _save_candidate_image(self, event_dir: Path, alias: str, obj: Mapping[str, Any]) -> dict:
        member_idx = self._best_member(obj)
        paths, masks = list(obj.get("color_path", [])), list(obj.get("mask", []))
        boxes, obs_uids = list(obj.get("xyxy", [])), list(obj.get("obs_uids", []))
        image_indices = list(obj.get("image_idx", []))
        if not paths or member_idx >= len(paths) or member_idx >= len(masks):
            raise ValueError(f"candidate {alias} has no historical RGB/mask member")
        source_path = Path(str(paths[member_idx]))
        image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(source_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bbox = boxes[member_idx] if member_idx < len(boxes) else None
        output_path = event_dir / f"candidate_{alias}.jpg"
        _write_rgb(output_path, _annotated_crop(image_rgb, masks[member_idx], bbox, f"CANDIDATE {alias}"))
        return {
            "alias": alias,
            "object_uid": str(obj.get("id")),
            "class_name": str(obj.get("class_name", "")),
            "num_detections": int(obj.get("num_detections", len(paths))),
            "selected_member_index": member_idx,
            "selected_member_frame_idx": int(image_indices[member_idx]) if member_idx < len(image_indices) else None,
            "selected_member_obs_uid": str(obs_uids[member_idx]) if member_idx < len(obs_uids) else None,
            "source_rgb_path": str(source_path),
            "image_path": output_path.name,
            "image_sha256": _sha256_file(output_path),
        }

    @staticmethod
    def _prompts(kind: str, aliases: Sequence[str]) -> Tuple[str, str]:
        if kind == "association":
            return ASSOCIATION_SYSTEM_PROMPT, (
                "I1 is the current scene context; I1-crop is the current masked observation. "
                "I2 and I3 are frozen historical views for candidates A and B. "
                "In every image, judge the red-masked target; use unmasked background only as supporting context. "
                "Apply the mandatory quality gate using only I1 and I1-crop before looking at candidates. "
                "Select exactly one of A, B, NEW, DISCARD, or UNCERTAIN and report observation_quality. "
                "MIXED_INSTANCES/SEVERE_FRAGMENT require DISCARD; BORDERLINE requires UNCERTAIN; only USABLE permits A, B, or NEW."
            )
        options = ", ".join(list(aliases) + ["NEW", "DISCARD", "UNCERTAIN"])
        return CREATE_SYSTEM_PROMPT, (
            "I1 is the current scene context; I1-crop is the current masked observation. "
            "The remaining images are frozen historical candidate views. In every image, judge the red-masked target; "
            "use unmasked background only as supporting context. Apply the mandatory quality gate using only I1 and I1-crop before looking at candidates. "
            f"Select exactly one of {options} and report observation_quality. MIXED_INSTANCES/SEVERE_FRAGMENT require DISCARD; "
            "BORDERLINE requires UNCERTAIN; only USABLE permits a candidate or NEW."
        )

    def _request_payload(self, system_prompt: str, user_prompt: str, images: Sequence[Tuple[str, Path]], aliases: Sequence[str]) -> dict:
        allowed = list(aliases) + ["NEW", "DISCARD", "UNCERTAIN"]
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
                            "observation_quality": {
                                "type": "string",
                                "enum": ["USABLE", "BORDERLINE", "MIXED_INSTANCES", "SEVERE_FRAGMENT"],
                            },
                            "choice": {"type": "string", "enum": allowed},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reason": {"type": "string"},
                        },
                        "required": ["observation_quality", "choice", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_completion_tokens": 1600,
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
        for alias, _, obj in candidates:
            item = self._save_candidate_image(event_dir, alias, obj)
            manifest.append({"role": f"candidate-{alias}", **item})
            request_images.append((f"Candidate {alias} frozen history", event_dir / item["image_path"]))
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
    ) -> List[Optional[int]]:
        final_matches = list(baseline_match_indices)
        if self.mode == "off" or not objects:
            return final_matches
        scores = _as_numpy(aggregate_sim).astype(float, copy=False)
        if scores.shape != (len(detection_list), len(objects)):
            raise ValueError(f"aggregate similarity shape {scores.shape} != {(len(detection_list), len(objects))}")
        for detected_idx, detection in enumerate(detection_list):
            baseline_match = baseline_match_indices[detected_idx]
            raw_trigger = compute_trigger(
                scores[detected_idx], baseline_match, self.sim_threshold,
                self.margin_threshold, self.threshold_distance, self.threshold_scope,
            )
            if raw_trigger is None:
                continue
            self.stats["raw_triggered_before_iou"] += 1
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
            if trigger is None:
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
                    "outcome": "trigger_suppressed",
                })
                continue
            self.stats["triggered"] += 1
            self.stats[f"triggered_{trigger['kind']}"] += 1
            self.stats["events_with_iou_drops"] += int(bool(iou_filter["dropped"]))
            self.stats["iou_candidates_dropped"] += len(iou_filter["dropped"])
            if self.max_events > 0 and self.stats["processed"] >= self.max_events:
                self.stats["suppressed_by_max_events"] += 1
                continue
            top_k = self.association_top_k if trigger["kind"] == "association" else self.create_top_k
            ranked = ranked[: min(top_k, len(ALIASES))]
            if trigger["kind"] == "association" and len(ranked) < 2:
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
            output = {"choice": "UNCERTAIN", "confidence": 0.0, "reason": "audit mode: no adjudication requested"}
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
                elif self.mode == "oracle":
                    choice, oracle_diagnostics = self._oracle_choice(frame_idx, detection, candidates)
                    output = {"choice": choice, "confidence": 1.0 if choice != "UNCERTAIN" else 0.0, "reason": "GT sidecar oracle"}
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
                allowed = set(aliases_to_indices) | {"NEW", "DISCARD", "UNCERTAIN"}
                if str(output.get("choice", "")).upper() not in allowed:
                    raise ValueError(f"invalid choice: {output.get('choice')}")
                confidence = float(output.get("confidence", 0.0))
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"invalid confidence: {confidence}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                latency_seconds = time.perf_counter() - adjudication_started
                output = {"choice": "UNCERTAIN", "confidence": 0.0, "reason": "invalid response or API failure; baseline fallback"}
                decision_source = f"{decision_source}_failure"
                self.stats["failures"] += 1
            _json_dump(event_dir / "vlm_output.json", output)
            if oracle_diagnostics is not None:
                _json_dump(event_dir / "oracle_diagnostics.json", oracle_diagnostics)
            if self.mode in {"oracle", "vlm"}:
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
                "baseline_match_index": baseline_match,
                "decision_source": decision_source,
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
                f'<a href="events/{event_id}/vlm_output.json">parsed output</a> · '
                f'<a href="events/{event_id}/decision.json">decision</a></p></section>'
            )
        document = f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>Blocking association gate</title><style>body{{font-family:system-ui;margin:20px;background:#111;color:#eee}}section{{border:1px solid #555;padding:14px;margin:14px 0}}.images{{display:flex;gap:8px;flex-wrap:wrap}}img{{max-width:300px;max-height:240px;object-fit:contain;background:#222}}a{{color:#7dd3fc}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>Blocking association gate: {html.escape(self.mode)}</h1><p>Auto-refresh: 5 s · processed={self.stats['processed']} · changed={self.stats['changed']} · failures={self.stats['failures']}</p>{''.join(cards)}</body></html>"""
        (self.output_dir / "index.html").write_text(document, encoding="utf-8")

    def close(self, *, status: str = "completed") -> None:
        self._write_summary(status=status)
        self._write_index()
