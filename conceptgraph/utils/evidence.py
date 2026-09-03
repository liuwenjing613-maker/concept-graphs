"""Append-only evidence records for realtime ConceptGraphs mapping.

The recorder is deliberately a sidecar: failures are captured in the evidence
summary and never allowed to change the mapping decision path.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


SCHEMA_VERSION = "0.2.0"
DEFAULT_AUDIT_POLICY = {
    "observation_ownership": "exclusive_single_target",
    "same_frame_many_to_one": "allowed",
    "relation_cardinality": "many_to_many",
    "environment_mode": "static",
    "object_granularity": "instance_with_part_whole_ambiguity",
    "association_rule": {
        "type": "independent_greedy_argmax",
        "threshold_comparison": "strict_greater_than",
        "max_score_equal_threshold": "create_object",
    },
    "postprocess_merge": {
        "source_single_consumption": True,
        "target_must_be_active": True,
        "source_must_be_active": True,
    },
    "missing_evidence_policy": "unknown_not_pass",
}
JSONL_FILES = (
    "frames.jsonl",
    "observations.jsonl",
    "associations.jsonl",
    "mapping_events.jsonl",
    "vlm_events.jsonl",
    "filter_trace.jsonl",
    "object_versions.jsonl",
    "object_pair_decisions.jsonl",
)


def _default_value(value):
    return value() if callable(value) else value


def evidence_safe(default=None):
    """Keep development evidence sidecars transparent; fail formal strict runs."""

    def decorator(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            if not self.enabled:
                return _default_value(default)
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive boundary
                self._record_error(func.__name__, exc)
                if getattr(self, "evidence_mode", "best_effort") == "strict":
                    raise
                return _default_value(default)

        return wrapped

    return decorator


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return str(value)


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_similarity_matrix(
    name: str, value: Any, expected_shape: tuple[int, int]
) -> tuple[np.ndarray, dict]:
    """Return a safe matrix plus an explicit validation result.

    Invalid matrices are preserved as NaN-filled evidence so the evidence gate
    can fail deterministically. They must never be ranked as association
    candidates.
    """
    arr = _array(value).astype(np.float32, copy=False)
    if arr.shape != expected_shape:
        return (
            np.full(expected_shape, np.nan, dtype=np.float32),
            {
                "valid": False,
                "error": "SHAPE_MISMATCH",
                "name": name,
                "actual_shape": list(arr.shape),
                "expected_shape": list(expected_shape),
            },
        )
    return arr, {"valid": True, "name": name, "shape": list(arr.shape)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _artifact_ref(
    path: Optional[Path],
    *,
    root: Optional[Path] = None,
    fmt: Optional[str] = None,
    key: Optional[str] = None,
    index: Optional[int] = None,
    shape: Optional[Iterable[int]] = None,
    dtype: Optional[Any] = None,
) -> Optional[dict]:
    """Create a portable, verifiable artifact reference.

    References remain useful when the producer has not materialised an artifact
    yet: in that case ``exists`` is false and the audit gate reports the exact
    missing artifact instead of silently accepting a string path.
    """
    if path is None:
        return None
    path = Path(path)
    payload = {
        "path": os.path.relpath(str(path.resolve()), str(root.resolve()))
        if root is not None
        else str(path),
        "format": fmt or path.suffix.lstrip(".") or "file",
        "key": key,
        "index": int(index) if index is not None else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
        "shape": list(shape) if shape is not None else None,
        "dtype": str(np.dtype(dtype)) if dtype is not None else None,
    }
    return payload


def _object_uid(obj: dict) -> str:
    value = obj.get("id")
    if value is None:
        value = uuid.uuid4()
        obj["id"] = value
    return str(value)


def _bbox_center_extent(bbox) -> tuple:
    if bbox is None:
        return None, None
    center_value = getattr(bbox, "get_center", None)
    center_value = center_value() if callable(center_value) else getattr(bbox, "center", None)
    extent_value = getattr(bbox, "get_extent", None)
    extent_value = extent_value() if callable(extent_value) else getattr(bbox, "extent", None)
    center = np.asarray(center_value, dtype=float).tolist() if center_value is not None else None
    extent = np.asarray(extent_value, dtype=float).tolist() if extent_value is not None else None
    return center, extent


def _bbox_summary(obj: dict) -> Dict[str, Any]:
    try:
        center, extent = _bbox_center_extent(obj.get("bbox"))
        return {"center": center, "extent": extent}
    except Exception:
        return {"center": None, "extent": None}


def _point_count(obj: dict) -> int:
    try:
        return int(len(obj["pcd"].points))
    except Exception:
        return int(obj.get("n_points", 0) or 0)


class EvidenceRecorder:
    """Write a stable evidence graph alongside the original experiment files."""

    def __init__(
        self,
        exp_out_path: Path,
        cfg: Any,
        detection_cfg: Any,
        enabled: bool = True,
        model_versions: Optional[dict] = None,
        prompt_versions: Optional[dict] = None,
    ):
        self.enabled = bool(enabled)
        self.exp_out_path = Path(exp_out_path).resolve()
        self.cfg = _plain(cfg) or {}
        self.detection_cfg = _plain(detection_cfg) or {}
        self.model_versions = model_versions or {}
        self.prompt_versions = prompt_versions or {}
        self.evidence_mode = str(self.cfg.get("evidence_mode", "best_effort"))
        if self.evidence_mode not in {"best_effort", "strict"}:
            raise ValueError(f"unsupported evidence_mode: {self.evidence_mode}")
        self._closed = False
        self._files: Dict[str, Any] = {}
        self._errors = []
        self._event_counter = 0
        self._frames = set()
        self._observation_frames = {}
        self._association_observations = []
        self._final_member_observations = []
        self._merged_parents: Dict[str, list] = {}
        self._vlm_context = None
        self._vlm_pending = []
        self._last_association_events = []
        self._merge_transactions: Dict[tuple, dict] = {}
        self._merge_candidate_ranks = Counter()
        self._object_versions: Dict[str, int] = {}
        self._current_object_versions: Dict[str, str] = {}
        self._last_event_by_object: Dict[str, str] = {}
        self.counters = Counter()

        scene_id = str(self.cfg.get("scene_id", "scene"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{_safe_name(scene_id)}_{stamp}_{uuid.uuid4().hex[:8]}"
        self.evidence_dir = self.exp_out_path / "evidence"
        self.similarity_dir = self.evidence_dir / "similarities"
        self.observation_pcd_dir = self.evidence_dir / "observation_pcd"
        self.processed_mask_dir = self.evidence_dir / "processed_masks"
        self.object_feature_dir = self.evidence_dir / "object_features"
        self.source_snapshot_dir = self.evidence_dir / "source_snapshot"

        if not self.enabled:
            return

        try:
            self._initialize_storage(scene_id)
        except Exception as exc:  # pragma: no cover - filesystem boundary
            for handle in self._files.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._files.clear()
            self._record_error("initialize", exc)
            self.enabled = False

    def _initialize_storage(self, scene_id: str) -> None:
        for directory in (
            self.similarity_dir,
            self.observation_pcd_dir,
            self.processed_mask_dir,
            self.object_feature_dir,
            self.source_snapshot_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            # JSONL files are reset for every run, so binary sidecars must be
            # reset as well or repeated use of an exp_suffix mixes run IDs.
            for stale_npz in directory.glob("*.npz"):
                if stale_npz.is_file():
                    stale_npz.unlink()

        for filename in JSONL_FILES:
            path = self.evidence_dir / filename
            path.write_text("", encoding="utf-8")
            self._files[filename] = path.open("a", encoding="utf-8")

        git_info = self._git_info()
        git_patch_ref = self._write_runtime_patch() if git_info["dirty"] else None
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "scene_id": scene_id,
            "dataset": self.cfg.get("dataset_config"),
            "branch": git_info["branch"],
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "git_diff_sha256": git_info.get("diff_sha256"),
            "git_patch_ref": git_patch_ref,
            "start_time": _utc_now(),
            "end_time": None,
            "status": "running",
            "evidence_mode": self.evidence_mode,
            "runtime": self._runtime_info(),
            "mapping_config_ref": _artifact_ref(
                self.exp_out_path / "config_params.json", root=self.exp_out_path
            ),
            "detection_config_ref": _artifact_ref(
                self.exp_out_path / "config_params_detections.json",
                root=self.exp_out_path,
            ),
            "detection_exp_suffix": self.cfg.get("detections_exp_suffix"),
            "mapping_exp_suffix": self.cfg.get("exp_suffix"),
            "model_versions": _plain(self.model_versions),
            "prompt_versions": _plain(self.prompt_versions),
            "make_edges": bool(self.cfg.get("make_edges", False)),
            "branch_id": "baseline",
            "audit_policy": {
                **_plain(DEFAULT_AUDIT_POLICY),
                "environment_mode": self.cfg.get(
                    "audit_environment_mode", "static"
                ),
            },
            "random_seeds": self._random_seed_info(),
            "runtime_source_snapshot": self._source_snapshot(),
        }
        self._write_json("manifest.json", self.manifest)
        atexit.register(self._finalize_abandoned_run)

    def frame_uid(self, frame_idx: int) -> str:
        return f"{self.run_id}_f{int(frame_idx):06d}"

    def observation_uid(self, frame_idx: int, raw_det_idx: int) -> str:
        return f"{self.frame_uid(frame_idx)}_r{int(raw_det_idx):04d}"

    def _next_event_uid(self) -> str:
        self._event_counter += 1
        return f"{self.run_id}_e{self._event_counter:08d}"

    def _record_error(self, operation: str, exc: Exception) -> None:
        message = f"{operation}: {type(exc).__name__}: {exc}"
        self._errors.append(message)
        print(f"[evidence] {message}")

    def _git_info(self) -> dict:
        repo = Path(__file__).resolve().parents[2]

        def run(*args):
            return subprocess.check_output(
                ["git", "-C", str(repo), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

        try:
            diff = subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            return {
                "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain")),
                "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
            }
        except Exception:
            return {
                "branch": "unknown",
                "commit": "unknown",
                "dirty": None,
                "diff_sha256": None,
            }

    def _runtime_info(self) -> dict:
        info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        }
        for module_name, key in (("numpy", "numpy_version"), ("torch", "torch_version"), ("open3d", "open3d_version")):
            try:
                module = __import__(module_name)
                info[key] = str(getattr(module, "__version__", "unknown"))
                if module_name == "torch":
                    info["cuda_version"] = str(getattr(module.version, "cuda", None))
            except Exception:
                info[key] = None
        return info

    def _random_seed_info(self) -> dict:
        seeds = {
            "python": self.cfg.get("seed"),
            "numpy": None,
            "torch": None,
        }
        try:
            seeds["numpy"] = int(np.random.get_state()[1][0])
        except Exception:
            pass
        try:
            import torch

            seeds["torch"] = int(torch.initial_seed())
        except Exception:
            pass
        return seeds

    def _write_runtime_patch(self) -> Optional[dict]:
        try:
            repo = Path(__file__).resolve().parents[2]
            patch = subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            if not patch:
                return None
            path = self.evidence_dir / "git_runtime.patch"
            path.write_bytes(patch)
            return _artifact_ref(path, root=self.exp_out_path)
        except Exception:
            return None

    def _source_snapshot(self) -> list[dict]:
        repo = Path(__file__).resolve().parents[2]
        relative_paths = (
            "conceptgraph/utils/evidence.py",
            "conceptgraph/audit/evidence_audit.py",
            "conceptgraph/slam/rerun_realtime_mapping.py",
            "conceptgraph/slam/mapping.py",
            "conceptgraph/slam/utils.py",
            "conceptgraph/hydra_configs/rerun_realtime_mapping.yaml",
        )
        snapshots = []
        for relative in relative_paths:
            source = repo / relative
            if not source.is_file():
                continue
            target = self.source_snapshot_dir / relative.replace("/", "__")
            shutil.copy2(source, target)
            snapshots.append(
                {
                    "source_path": relative,
                    "artifact_ref": _artifact_ref(target, root=self.exp_out_path),
                }
            )
        return snapshots

    def _relative(self, path: Any) -> Optional[str]:
        if path is None:
            return None
        try:
            return os.path.relpath(str(Path(path).resolve()), str(self.exp_out_path))
        except Exception:
            return str(path)

    def _write_json(self, filename: str, payload: Any) -> None:
        path = self.evidence_dir / filename
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(_plain(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)

    def _append(self, filename: str, payload: dict) -> None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            **_plain(payload),
        }
        handle = self._files[filename]
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    @evidence_safe(list)
    def prepare_observations(
        self,
        raw_gobs: dict,
        frame_idx: int,
        detection_path: Optional[Path] = None,
    ) -> list:
        count = len(raw_gobs.get("xyxy", []))
        raw_indices = np.arange(count, dtype=np.int64)
        obs_uids = np.asarray(
            [self.observation_uid(frame_idx, index) for index in raw_indices],
            dtype=object,
        )
        raw_gobs["raw_det_idx"] = raw_indices
        raw_gobs["obs_uid"] = obs_uids

        captions = raw_gobs.get("captions") or []
        captions_by_id = {
            str(item.get("id")): item for item in captions if isinstance(item, dict)
        }
        labels = raw_gobs.get("detection_class_labels") or []
        classes = raw_gobs.get("classes") or []
        detection_ref = self._relative(detection_path) if detection_path else None
        detection_path_abs = str(Path(detection_path).resolve()) if detection_path else None
        text_feats = raw_gobs.get("text_feats")
        image_feats = raw_gobs.get("image_feats")
        snapshots = []
        for index in range(count):
            class_id = int(raw_gobs["class_id"][index])
            label = labels[index] if index < len(labels) else None
            label_id = str(label).split()[-1] if label else str(index)
            mask = raw_gobs["mask"][index]
            mask_array = np.asarray(mask)
            image_feature = (
                np.asarray(image_feats[index])
                if image_feats is not None and len(image_feats) > index
                else None
            )
            snapshots.append(
                {
                    "obs_uid": str(obs_uids[index]),
                    "raw_det_idx": index,
                    "bbox_2d": _plain(raw_gobs["xyxy"][index]),
                    "mask_area": int(mask_array.sum()),
                    "raw_mask_shape": list(mask_array.shape),
                    "raw_mask_dtype": str(mask_array.dtype),
                    "image_feature_shape": list(image_feature.shape)
                    if image_feature is not None
                    else None,
                    "image_feature_dtype": str(image_feature.dtype)
                    if image_feature is not None
                    else None,
                    "confidence": float(raw_gobs["confidence"][index])
                    if raw_gobs.get("confidence") is not None
                    else None,
                    "class_id": class_id,
                    "class_name": str(classes[class_id]) if class_id < len(classes) else None,
                    "detection_label": label,
                    "raw_caption": captions_by_id.get(label_id),
                    "detection_ref": detection_ref,
                    "detection_path": detection_path_abs,
                    "text_feature_available": bool(
                        text_feats is not None
                        and len(text_feats) > index
                        and np.asarray(text_feats[index]).size > 0
                    ),
                }
            )
        return snapshots

    def _filter_reason(self, obs: dict, image_shape, bg_classes: Iterable[str]) -> Optional[str]:
        mask_threshold = max(float(self.cfg.get("mask_area_threshold", 10)), 10.0)
        if obs["mask_area"] < mask_threshold:
            return "mask_area_below_threshold"
        bg_classes = set(bg_classes or [])
        if self.cfg.get("skip_bg") and obs["class_name"] in bg_classes:
            return "background_class"
        if obs["class_name"] not in bg_classes:
            x1, y1, x2, y2 = obs["bbox_2d"]
            bbox_area = max(0.0, float(x2 - x1) * float(y2 - y1))
            image_area = float(image_shape[0] * image_shape[1])
            max_ratio = self.cfg.get("max_bbox_area_ratio")
            if max_ratio is not None and bbox_area > float(max_ratio) * image_area:
                return "bbox_area_above_ratio"
        confidence_threshold = self.cfg.get("mask_conf_threshold")
        if (
            confidence_threshold is not None
            and obs["confidence"] is not None
            and obs["confidence"] < float(confidence_threshold)
        ):
            return "confidence_below_threshold"
        return None

    @evidence_safe(list)
    def record_observations(
        self,
        frame_idx: int,
        snapshots: list,
        filtered_gobs: dict,
        obj_pcds_and_bboxes: list,
        image_shape,
        bg_classes: Iterable[str],
        filter_trace: Optional[list] = None,
        pre_subtract_masks: Optional[Any] = None,
        depth_array: Optional[Any] = None,
    ) -> list:
        frame_uid = self.frame_uid(frame_idx)
        filtered_raw_indices = [int(index) for index in filtered_gobs.get("raw_det_idx", [])]
        filtered_positions = {
            raw_index: filtered_index
            for filtered_index, raw_index in enumerate(filtered_raw_indices)
        }
        detection_obs_uids = []
        save_pcd = bool(self.cfg.get("evidence_save_observation_pcd", True))
        # Full post-init PCDs are the default because they are the exact geometry
        # fused into the map.  A positive cap remains available for development.
        max_points = int(self.cfg.get("evidence_observation_pcd_max_points", 0))
        pre_masks = (
            np.asarray(pre_subtract_masks, dtype=bool)
            if pre_subtract_masks is not None
            else None
        )

        for obs in snapshots:
            raw_index = obs["raw_det_idx"]
            filtered_index = filtered_positions.get(raw_index)
            trace = None
            if filter_trace:
                trace = next(
                    (
                        item
                        for item in filter_trace
                        if int(item.get("raw_det_idx", -1)) == raw_index
                    ),
                    None,
                )
            pcd_info = None
            if filtered_index is not None and filtered_index < len(obj_pcds_and_bboxes):
                pcd_info = obj_pcds_and_bboxes[filtered_index]
                if trace is not None:
                    trace.setdefault("evaluated_gates", []).append(
                        {
                            "gate": "valid_3d_observation",
                            "value": pcd_info is not None,
                            "operator": "is",
                            "threshold": True,
                            "passed": pcd_info is not None,
                        }
                    )
                    if pcd_info is None:
                        trace["decision"] = "REJECT"
                        trace["first_failed_gate"] = "insufficient_3d_points"
            reason = (
                (trace or {}).get("first_failed_gate")
                if trace is not None
                else self._filter_reason(obs, image_shape, bg_classes)
            )
            if filtered_index is not None and pcd_info is None and reason is None:
                reason = "insufficient_3d_points"

            kept = filtered_index is not None and pcd_info is not None
            pcd_ref = None
            n_points = 0
            pcd_stored_points = 0
            pcd_is_sampled = False
            points_sha256 = None
            bbox_center = None
            bbox_extent = None
            processed_mask_ref = None
            processed_mask_area = None
            pre_subtract_mask_area = None
            subtract_source_obs_uids = []
            valid_depth_ratio = None
            raw_valid_depth_points = None
            depth_quantiles = None
            boundary_touch_ratio = None
            pcd_stats = {}
            if kept:
                detection_obs_uids.append(obs["obs_uid"])
                pcd = pcd_info["pcd"]
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                n_points = int(len(points))
                bbox_center, bbox_extent = _bbox_center_extent(
                    pcd_info.get("bbox")
                )
                if save_pcd:
                    if max_points > 0 and len(points) > max_points:
                        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
                        points = points[indices]
                        if len(colors) == n_points:
                            colors = colors[indices]
                        pcd_is_sampled = True
                    pcd_path = self.observation_pcd_dir / f"{obs['obs_uid']}.npz"
                    np.savez_compressed(pcd_path, points=points, colors=colors)
                    pcd_stored_points = int(len(points))
                    points_sha256 = hashlib.sha256(
                        np.ascontiguousarray(points).tobytes()
                    ).hexdigest()
                    pcd_ref = _artifact_ref(
                        pcd_path,
                        root=self.exp_out_path,
                        fmt="npz",
                        key="points",
                        shape=points.shape,
                        dtype=points.dtype,
                    )

                if filtered_index < len(filtered_gobs.get("mask", [])):
                    processed_mask = np.asarray(
                        filtered_gobs["mask"][filtered_index], dtype=bool
                    )
                    if pre_masks is not None and filtered_index < len(pre_masks):
                        pre_mask = pre_masks[filtered_index]
                        pre_subtract_mask_area = int(pre_mask.sum())
                        removed = np.logical_and(pre_mask, ~processed_mask)
                        if removed.any():
                            for other_index, other_mask in enumerate(pre_masks):
                                if other_index == filtered_index:
                                    continue
                                if np.logical_and(removed, other_mask).any():
                                    source_raw_index = filtered_raw_indices[other_index]
                                    subtract_source_obs_uids.append(
                                        self.observation_uid(frame_idx, source_raw_index)
                                    )
                    processed_mask_path = self.processed_mask_dir / f"{obs['obs_uid']}.npz"
                    np.savez_compressed(processed_mask_path, mask=processed_mask)
                    processed_mask_area = int(processed_mask.sum())
                    if processed_mask_area:
                        border = np.zeros_like(processed_mask, dtype=bool)
                        border[:2, :] = True
                        border[-2:, :] = True
                        border[:, :2] = True
                        border[:, -2:] = True
                        boundary_touch_ratio = float(
                            np.logical_and(processed_mask, border).sum()
                            / processed_mask_area
                        )
                        if depth_array is not None:
                            depth_values = np.asarray(depth_array)[processed_mask]
                            valid_depth = depth_values[
                                np.isfinite(depth_values) & (depth_values > 0)
                            ]
                            raw_valid_depth_points = int(len(valid_depth))
                            valid_depth_ratio = float(
                                len(valid_depth) / processed_mask_area
                            )
                            if len(valid_depth):
                                q05, q25, q50, q75, q95 = np.quantile(
                                    valid_depth, [0.05, 0.25, 0.50, 0.75, 0.95]
                                )
                                depth_quantiles = {
                                    "q05": float(q05),
                                    "q25": float(q25),
                                    "q50": float(q50),
                                    "q75": float(q75),
                                    "q95": float(q95),
                                }
                    if filtered_index < len(obj_pcds_and_bboxes):
                        obj_info = obj_pcds_and_bboxes[filtered_index]
                        if obj_info:
                            pcd_stats = _plain(
                                obj_info.get("evidence_pcd_stats", {})
                            )
                    processed_mask_ref = _artifact_ref(
                        processed_mask_path,
                        root=self.exp_out_path,
                        fmt="npz",
                        key="mask",
                        shape=processed_mask.shape,
                        dtype=processed_mask.dtype,
                    )

            detection_ref = obs.get("detection_ref")
            detection_base = (
                Path(obs["detection_path"])
                if obs.get("detection_path")
                else None
            )
            raw_mask_path = detection_base / "mask.npz" if detection_base else None
            raw_mask_ref = _artifact_ref(
                raw_mask_path,
                root=self.exp_out_path,
                fmt="npz",
                key="arr_0",
                index=raw_index,
                shape=obs.get("raw_mask_shape"),
                dtype=obs.get("raw_mask_dtype"),
            )
            crop_path = detection_base / "image_crops.pkl.gz" if detection_base else None
            image_feat_path = detection_base / "image_feats.npz" if detection_base else None
            text_feat_path = detection_base / "text_feats.pkl.gz" if detection_base else None
            text_feature_status = (
                "AVAILABLE"
                if obs.get("text_feature_available")
                and text_feat_path is not None
                and text_feat_path.is_file()
                else "NOT_COMPUTED"
            )
            record = {
                "frame_uid": frame_uid,
                "obs_uid": obs["obs_uid"],
                "raw_det_idx": raw_index,
                "filtered_det_idx": filtered_index,
                "status": "kept" if kept else "rejected",
                "filter_reason": None if kept else (reason or "not_retained"),
                "filter_trace": trace,
                "bbox_2d": obs["bbox_2d"],
                "mask_area": obs["mask_area"],
                "confidence": obs["confidence"],
                "class_id": obs["class_id"],
                "class_name": obs["class_name"],
                "mask_ref": raw_mask_ref,
                "raw_mask_ref": raw_mask_ref,
                "processed_mask_ref": processed_mask_ref,
                "raw_mask_area": obs["mask_area"],
                "pre_subtract_mask_area": pre_subtract_mask_area,
                "processed_mask_area": processed_mask_area,
                "removed_pixel_count": (
                    int(pre_subtract_mask_area - processed_mask_area)
                    if processed_mask_area is not None
                    and pre_subtract_mask_area is not None
                    else None
                ),
                "mask_operations": [
                    "resize_nearest",
                    "filter",
                    "subtract_contained",
                ],
                "subtract_source_obs_uids": subtract_source_obs_uids,
                "crop_ref": _artifact_ref(
                    crop_path,
                    root=self.exp_out_path,
                    fmt="pickle.gz",
                    index=raw_index,
                ) if crop_path is not None else None,
                "image_feat_ref": _artifact_ref(
                    image_feat_path,
                    root=self.exp_out_path,
                    fmt="npz",
                    key="arr_0",
                    index=raw_index,
                    shape=obs.get("image_feature_shape"),
                    dtype=obs.get("image_feature_dtype"),
                ) if image_feat_path is not None else None,
                "text_feat_ref": (
                    _artifact_ref(
                        text_feat_path,
                        root=self.exp_out_path,
                        fmt="pickle.gz",
                        index=raw_index,
                    )
                    if text_feature_status == "AVAILABLE"
                    else None
                ),
                "text_feature_status": text_feature_status,
                "pcd_ref": pcd_ref,
                "n_points": n_points,
                "pcd_stage": "after_init_process" if kept else None,
                "pcd_is_sampled": pcd_is_sampled if kept else None,
                "pcd_raw_valid_depth_points": raw_valid_depth_points,
                "pcd_before_downsample_points": pcd_stats.get("input_points"),
                "pcd_after_downsample_points": pcd_stats.get(
                    "after_downsample_points"
                ),
                "pcd_after_dbscan_points": n_points if kept else 0,
                "pcd_stored_points": pcd_stored_points if kept else 0,
                "pcd_sample_indices_ref": None,
                "pcd_random_seed": None,
                "voxel_size": self.cfg.get("downsample_voxel_size"),
                "dbscan_eps": self.cfg.get("dbscan_eps"),
                "dbscan_min_points": self.cfg.get("dbscan_min_points"),
                "valid_depth_ratio": valid_depth_ratio,
                "depth_quantiles": depth_quantiles,
                "boundary_touch_ratio": boundary_touch_ratio,
                "pre_dbscan": {
                    "cluster_count": pcd_stats.get("cluster_count"),
                    "largest_cluster_ratio": pcd_stats.get(
                        "largest_cluster_ratio"
                    ),
                    "second_cluster_ratio": pcd_stats.get(
                        "second_cluster_ratio"
                    ),
                    "largest_centers_distance": pcd_stats.get(
                        "largest_centers_distance"
                    ),
                    "n_points": pcd_stats.get("after_downsample_points"),
                }
                if kept
                else None,
                "post_dbscan": {"n_points": n_points} if kept else None,
                "points_sha256": points_sha256,
                "bbox_3d_center": bbox_center,
                "bbox_3d_extent": bbox_extent,
                "raw_caption": obs["raw_caption"],
                "detection_label": obs["detection_label"],
            }
            self._append("observations.jsonl", record)
            self._observation_frames[obs["obs_uid"]] = frame_uid
            self.counters["num_raw_detections"] += 1
            self.counters[
                "num_kept_observations" if kept else "num_rejected_observations"
            ] += 1
        return detection_obs_uids

    @evidence_safe(None)
    def record_filter_trace(self, frame_idx: int, filter_trace: Optional[list]) -> None:
        """Persist the gates evaluated by the real filtering function.

        This is intentionally a separate append-only stream so the validator can
        distinguish an execution trace from a reason reconstructed later.
        """
        for trace in filter_trace or []:
            self._append(
                "filter_trace.jsonl",
                {"frame_uid": self.frame_uid(frame_idx), **_plain(trace)},
            )

    @evidence_safe(None)
    def record_frame(
        self,
        frame_idx: int,
        source_frame_id: str,
        rgb_path: Optional[Path],
        depth_path: Optional[Path],
        pose: Any,
        intrinsics: Any,
        processed: bool,
        skip_reason: Optional[str],
        num_raw_detections: int,
        num_kept_observations: int,
    ) -> None:
        frame_uid = self.frame_uid(frame_idx)
        self._append(
            "frames.jsonl",
            {
                "frame_uid": frame_uid,
                "frame_idx": int(frame_idx),
                "source_frame_id": str(source_frame_id),
                "rgb_path": self._relative(rgb_path),
                "depth_path": self._relative(depth_path),
                "rgb_ref": _artifact_ref(rgb_path, root=self.exp_out_path),
                "depth_ref": _artifact_ref(depth_path, root=self.exp_out_path),
                "pose": _plain(pose),
                "intrinsics": _plain(intrinsics),
                "processed": bool(processed),
                "skip_reason": skip_reason,
                "num_raw_detections": int(num_raw_detections),
                "num_kept_observations": int(num_kept_observations),
            },
        )
        self._frames.add(frame_uid)
        self.counters["num_frames"] += 1

    @evidence_safe(None)
    def attach_observation_membership(self, detection_list, observation_uids: list) -> None:
        if len(detection_list) != len(observation_uids):
            raise ValueError(
                f"detection/observation mismatch: {len(detection_list)} != {len(observation_uids)}"
            )
        for detection, obs_uid in zip(detection_list, observation_uids):
            detection["obs_uids"] = [str(obs_uid)]

    @evidence_safe(list)
    def record_associations(
        self,
        frame_idx: int,
        detection_list,
        objects_before,
        spatial_sim: Any,
        visual_sim: Any,
        aggregate_sim: Any,
        match_indices: list,
    ) -> list:
        frame_uid = self.frame_uid(frame_idx)
        observation_uids = [str(det["obs_uids"][0]) for det in detection_list]
        object_uids = [_object_uid(obj) for obj in objects_before]
        expected_shape = (len(detection_list), len(objects_before))
        spatial, spatial_validation = _validate_similarity_matrix(
            "spatial_sim", spatial_sim, expected_shape
        )
        visual, visual_validation = _validate_similarity_matrix(
            "visual_sim", visual_sim, expected_shape
        )
        aggregate, aggregate_validation = _validate_similarity_matrix(
            "aggregate_sim", aggregate_sim, expected_shape
        )
        similarity_validation = {
            "valid": all(
                item["valid"]
                for item in (
                    spatial_validation,
                    visual_validation,
                    aggregate_validation,
                )
            ),
            "matrices": {
                "spatial_sim": spatial_validation,
                "visual_sim": visual_validation,
                "aggregate_sim": aggregate_validation,
            },
        }

        similarity_path = self.similarity_dir / f"frame_{int(frame_idx):06d}.npz"
        np.savez_compressed(
            similarity_path,
            observation_uids=np.asarray(observation_uids),
            object_uids=np.asarray(object_uids),
            spatial_sim=spatial,
            visual_sim=visual,
            aggregate_sim=aggregate,
        )
        similarity_ref = _artifact_ref(
            similarity_path,
            root=self.exp_out_path,
            fmt="npz",
        )
        top_k = max(1, int(self.cfg.get("evidence_top_k", 3)))
        targets = []
        self._last_association_events = []
        pending_versions = Counter()
        for det_index, (detection, match_index) in enumerate(
            zip(detection_list, match_indices)
        ):
            discarded = match_index == -1
            row = (
                aggregate[det_index]
                if similarity_validation["valid"] and aggregate.shape[1]
                else np.asarray([])
            )
            order = np.argsort(row)[::-1][:top_k] if row.size else []
            candidates = [
                {
                    "object_uid": object_uids[int(index)],
                    "spatial_score": float(spatial[det_index, index]),
                    "visual_score": float(visual[det_index, index]),
                    "aggregate_score": float(aggregate[det_index, index]),
                }
                for index in order
            ]
            top1 = float(row[order[0]]) if len(order) > 0 else None
            top2 = float(row[order[1]]) if len(order) > 1 else None
            target_uid = None if discarded else (
                _object_uid(detection)
                if match_index is None
                else object_uids[int(match_index)]
            )
            targets.append(target_uid)
            event_uid = self._next_event_uid()
            mapping_event_uid = self._next_event_uid()
            decision = (
                "DISCARD_OBSERVATION" if discarded
                else "CREATE_OBJECT" if match_index is None
                else "MERGE_TO_OBJECT"
            )
            if discarded:
                target_version_before = None
                target_version_after = None
            else:
                pending_count = int(pending_versions[target_uid])
                base_version = int(self._object_versions.get(target_uid, 0))
                target_version_before = (
                    self._current_object_versions.get(target_uid)
                    if pending_count == 0
                    else f"{target_uid}@v{base_version + pending_count:06d}"
                )
                target_version_after = f"{target_uid}@v{base_version + pending_count + 1:06d}"
                pending_versions[target_uid] += 1
            self._append(
                "associations.jsonl",
                {
                    "event_uid": event_uid,
                    "frame_uid": frame_uid,
                    "obs_uid": observation_uids[det_index],
                    "object_uids_before": object_uids,
                    "spatial_sim_ref": {**similarity_ref, "key": "spatial_sim"},
                    "visual_sim_ref": {**similarity_ref, "key": "visual_sim"},
                    "aggregate_sim_ref": {**similarity_ref, "key": "aggregate_sim"},
                    "similarity_evidence_valid": similarity_validation["valid"],
                    "similarity_validation": similarity_validation,
                    "top_candidates": candidates,
                    "top1_score": top1,
                    "top2_score": top2,
                    "margin": (top1 - top2) if top1 is not None and top2 is not None else None,
                    "sim_threshold": self.cfg.get("sim_threshold"),
                    "match_method": self.cfg.get("match_method"),
                    "phys_bias": self.cfg.get("phys_bias"),
                    "decision": decision,
                    "decision_override": "blocking_gate_quality_discard" if discarded else None,
                    "target_object_uid": target_uid,
                    "target_object_version_before": target_version_before,
                    "target_object_version_after": target_version_after,
                    "candidate_object_version_uids": [
                        self._current_object_versions.get(uid) for uid in object_uids
                    ],
                    "mapping_event_uid": mapping_event_uid,
                    "transaction_uid": f"{self.run_id}_assoc_f{int(frame_idx):06d}",
                    "branch_id": "baseline",
                    "event_sequence": int(event_uid.rsplit("e", 1)[-1]),
                },
            )
            self._append(
                "mapping_events.jsonl",
                {
                    "event_uid": mapping_event_uid,
                    "frame_uid": frame_uid,
                    "event_type": (
                        "OBS_DISCARD"
                        if discarded
                        else "OBJECT_CREATE"
                        if match_index is None
                        else "OBS_ASSOCIATE"
                    ),
                    "object_uid": target_uid,
                    "obs_uid": observation_uids[det_index],
                    "association_event_uid": event_uid,
                    "reason": decision,
                    "transaction_uid": f"{self.run_id}_assoc_f{int(frame_idx):06d}",
                    "parent_event_uids": [event_uid],
                    "input_object_version_uids": [target_version_before]
                    if target_version_before
                    else [],
                    "output_object_version_uids": [target_version_after] if target_version_after else [],
                    "branch_id": "baseline",
                    "event_sequence": int(mapping_event_uid.rsplit("e", 1)[-1]),
                },
            )
            self._last_association_events.append(
                {
                    "event_uid": mapping_event_uid,
                    "association_event_uid": event_uid,
                    "object_uid": target_uid,
                    "obs_uid": observation_uids[det_index],
                    "event_type": decision,
                    "target_object_version_before": target_version_before,
                    "target_object_version_after": target_version_after,
                    "detected_obj_idx": det_index,
                }
            )
            self._association_observations.append(observation_uids[det_index])
            counter = (
                "num_discard_decisions" if discarded
                else "num_create_decisions" if match_index is None
                else "num_associate_decisions"
            )
            self.counters[counter] += 1
        return targets

    @evidence_safe(dict)
    def snapshot_objects(self, objects) -> dict:
        snapshot = {}
        for obj in objects:
            uid = _object_uid(obj)
            class_values = [str(value) for value in obj.get("class_id", [])]
            class_histogram = Counter(class_values)
            dominant_class = None
            dominant_ratio = 0.0
            if class_histogram:
                dominant_class, dominant_count = class_histogram.most_common(1)[0]
                dominant_ratio = float(dominant_count / sum(class_histogram.values()))
            snapshot[uid] = {
                "object_uid": uid,
                "member_observation_uids": list(obj.get("obs_uids", [])),
                "n_points": _point_count(obj),
                "num_detections": int(obj.get("num_detections", 0)),
                "bbox": _bbox_summary(obj),
                "class_name": obj.get("class_name"),
                "class_histogram": dict(class_histogram),
                "dominant_class": dominant_class,
                "dominant_class_ratio": dominant_ratio,
            }
        return snapshot

    def _next_object_version_uid(self, object_uid: str) -> str:
        return f"{object_uid}@v{int(self._object_versions.get(object_uid, 0)) + 1:06d}"

    def _append_object_version(
        self,
        *,
        frame_idx: int,
        obj: Optional[dict],
        summary: dict,
        operation: str,
        trigger_event_uid: str,
        status: str = "active",
        parent_version_uids: Optional[list] = None,
    ) -> str:
        uid = str(summary["object_uid"])
        version = int(self._object_versions.get(uid, 0)) + 1
        version_uid = f"{uid}@v{version:06d}"
        previous = self._current_object_versions.get(uid)
        parents = list(parent_version_uids or ([previous] if previous else []))
        member_uids = list(dict.fromkeys(summary.get("member_observation_uids", [])))
        bbox = summary.get("bbox") or {}
        extent = bbox.get("extent")
        bbox_volume = None
        if extent is not None:
            try:
                bbox_volume = float(np.prod(np.asarray(extent, dtype=float)))
            except Exception:
                pass
        feature_ref = None
        feature_hash = None
        if obj is not None and obj.get("clip_ft") is not None:
            feature = _array(obj.get("clip_ft"))
            if feature.size:
                feature_path = self.object_feature_dir / f"{_safe_name(version_uid)}.npz"
                np.savez_compressed(feature_path, clip_feature=feature)
                feature_hash = hashlib.sha256(
                    np.ascontiguousarray(feature).tobytes()
                ).hexdigest()
                feature_ref = _artifact_ref(
                    feature_path,
                    root=self.exp_out_path,
                    fmt="npz",
                    key="clip_feature",
                    shape=feature.shape,
                    dtype=feature.dtype,
                )
        origin = member_uids[0] if member_uids else None
        record = {
            "object_version_uid": version_uid,
            "object_uid": uid,
            "version": version,
            "frame_uid": self.frame_uid(frame_idx),
            "trigger_event_uid": trigger_event_uid,
            "operation": operation,
            "status": status,
            "member_observation_uids": member_uids,
            "num_unique_observations": len(member_uids),
            "num_detections": int(summary.get("num_detections", len(member_uids))),
            "unique_frame_count": len(
                {self._observation_frames.get(obs_uid) for obs_uid in member_uids}
                - {None}
            ),
            "n_points": int(summary.get("n_points", 0)),
            "bbox_center": bbox.get("center"),
            "bbox_extent": extent,
            "bbox_volume": bbox_volume,
            "class_histogram": summary.get("class_histogram", {}),
            "dominant_class": summary.get("dominant_class"),
            "dominant_class_ratio": summary.get("dominant_class_ratio", 0.0),
            "class_name": summary.get("class_name"),
            "clip_feature_ref": feature_ref,
            "clip_feature_sha256": feature_hash,
            "parent_version_uids": parents,
            "lineage_uid": f"origin_{origin}" if origin else f"origin_{uid}",
            "origin_observation_uid": origin,
            "branch_id": "baseline",
        }
        self._append("object_versions.jsonl", record)
        self._object_versions[uid] = version
        self._current_object_versions[uid] = version_uid
        self._last_event_by_object[uid] = trigger_event_uid
        return version_uid

    @evidence_safe(None)
    def record_association_object_version(
        self,
        frame_idx: int,
        detected_obj_idx: int,
        existing_obj_match_idx: Optional[int],
        before_object: Optional[dict],
        after_object: dict,
    ) -> None:
        event = self._last_association_events[int(detected_obj_idx)]
        uid = _object_uid(after_object)
        if uid != event["object_uid"]:
            raise ValueError(f"association target changed: {uid} != {event['object_uid']}")
        summary = self.snapshot_objects([after_object])[uid]
        version_uid = self._append_object_version(
            frame_idx=frame_idx,
            obj=after_object,
            summary=summary,
            operation="OBJECT_CREATE"
            if existing_obj_match_idx is None
            else "OBS_ASSOCIATE",
            trigger_event_uid=event["event_uid"],
        )
        if version_uid != event["target_object_version_after"]:
            raise ValueError(
                f"object version drift: {version_uid} != {event['target_object_version_after']}"
            )

    @evidence_safe(None)
    def record_denoise(self, frame_idx: int, before: dict, objects_after) -> None:
        after = self.snapshot_objects(objects_after)
        objects_by_uid = {_object_uid(obj): obj for obj in objects_after}
        for uid, current in after.items():
            previous = before.get(uid)
            if previous is None:
                continue
            if previous["n_points"] == current["n_points"] and previous["bbox"] == current["bbox"]:
                continue
            event_uid = self._next_event_uid()
            output_version_uid = self._next_object_version_uid(uid)
            input_version_uid = self._current_object_versions.get(uid)
            self._append(
                "mapping_events.jsonl",
                {
                    "event_uid": event_uid,
                    "frame_uid": self.frame_uid(frame_idx),
                    "event_type": "OBJECT_DENOISE",
                    "object_uid": uid,
                    "before_summary": previous,
                    "after_summary": current,
                    "reason": "scheduled_denoise",
                    "transaction_uid": f"{self.run_id}_denoise_f{int(frame_idx):06d}",
                    "parent_event_uids": [self._last_event_by_object[uid]]
                    if uid in self._last_event_by_object
                    else [],
                    "input_object_version_uids": [input_version_uid]
                    if input_version_uid
                    else [],
                    "output_object_version_uids": [output_version_uid],
                    "branch_id": "baseline",
                    "event_sequence": int(event_uid.rsplit("e", 1)[-1]),
                },
            )
            self._append_object_version(
                frame_idx=frame_idx,
                obj=objects_by_uid[uid],
                summary=current,
                operation="OBJECT_DENOISE",
                trigger_event_uid=event_uid,
            )
            self.counters["num_object_denoise"] += 1

    @evidence_safe(None)
    def record_filter(self, frame_idx: int, before: dict, objects_after) -> None:
        after = self.snapshot_objects(objects_after)
        for uid in sorted(set(before) - set(after)):
            event_uid = self._next_event_uid()
            output_version_uid = self._next_object_version_uid(uid)
            input_version_uid = self._current_object_versions.get(uid)
            self._append(
                "mapping_events.jsonl",
                {
                    "event_uid": event_uid,
                    "frame_uid": self.frame_uid(frame_idx),
                    "event_type": "OBJECT_FILTER",
                    "object_uid": uid,
                    "before_summary": before[uid],
                    "after_summary": None,
                    "reason": {
                        "type": "minimum_support_filter",
                        "obj_min_points": self.cfg.get("obj_min_points"),
                        "obj_min_detections": self.cfg.get("obj_min_detections"),
                    },
                    "transaction_uid": f"{self.run_id}_filter_f{int(frame_idx):06d}",
                    "parent_event_uids": [self._last_event_by_object[uid]]
                    if uid in self._last_event_by_object
                    else [],
                    "input_object_version_uids": [input_version_uid]
                    if input_version_uid
                    else [],
                    "output_object_version_uids": [output_version_uid],
                    "branch_id": "baseline",
                    "event_sequence": int(event_uid.rsplit("e", 1)[-1]),
                },
            )
            self._append_object_version(
                frame_idx=frame_idx,
                obj=None,
                summary=before[uid],
                operation="OBJECT_FILTER",
                trigger_event_uid=event_uid,
                status="filtered",
            )
            self.counters["num_filtered_objects"] += 1

    @evidence_safe(None)
    def record_merge_candidate(
        self,
        frame_idx: int,
        source_object: dict,
        target_object: dict,
        overlap_ratio: Any,
        visual_similarity: Any,
        text_similarity: Any,
        decision: str,
        reject_reason: Optional[list] = None,
        source_active_before: bool = True,
        target_active_before: bool = True,
    ) -> str:
        source_uid = _object_uid(source_object)
        target_uid = _object_uid(target_object)
        tx_uid = f"{self.run_id}_merge_f{int(frame_idx):06d}"
        source_before = self.snapshot_objects([source_object]).get(source_uid)
        target_before = self.snapshot_objects([target_object]).get(target_uid)
        payload = {
            "merge_transaction_uid": tx_uid,
            "candidate_rank": int(self._merge_candidate_ranks[tx_uid] + 1),
            "frame_uid": self.frame_uid(frame_idx),
            "source_object_uid": source_uid,
            "target_object_uid": target_uid,
            "source_object_version_uid": self._current_object_versions.get(source_uid),
            "target_object_version_uid": self._current_object_versions.get(target_uid),
            "overlap_a_to_b": float(_plain(overlap_ratio)),
            "overlap_b_to_a": None,
            "visual_similarity": float(_plain(visual_similarity)),
            "text_similarity": float(_plain(text_similarity)),
            "text_similarity_source": "VISUAL_PROXY",
            "independent_evidence_group": "image_clip",
            "thresholds": {
                "overlap": self.cfg.get("merge_overlap_thresh"),
                "visual": self.cfg.get("merge_visual_sim_thresh"),
                "text": self.cfg.get("merge_text_sim_thresh"),
            },
            "decision": str(decision),
            "reject_reason": reject_reason,
            "source_active_before": bool(source_active_before),
            "target_active_before": bool(target_active_before),
            "source_consumed_after": bool(decision == "ACCEPT"),
            "source_member_set_before": (source_before or {}).get("member_observation_uids", []),
            "target_member_set_before": (target_before or {}).get("member_observation_uids", []),
            "member_intersection_before": sorted(
                set((source_before or {}).get("member_observation_uids", []))
                & set((target_before or {}).get("member_observation_uids", []))
            ),
        }
        self._append("object_pair_decisions.jsonl", payload)
        self._merge_candidate_ranks[tx_uid] += 1
        if decision == "ACCEPT":
            self._merge_transactions[(int(frame_idx), source_uid, target_uid)] = {
                "merge_transaction_uid": tx_uid,
                "target_before": target_before,
                "source_before": source_before,
            }
        return tx_uid

    @evidence_safe(None)
    def record_object_merge(
        self,
        frame_idx: int,
        source_object: dict,
        target_object: dict,
        overlap_ratio: Any,
        visual_similarity: Any,
        text_similarity: Any,
    ) -> None:
        source_uid = _object_uid(source_object)
        target_uid = _object_uid(target_object)
        parents = self._merged_parents.setdefault(target_uid, [])
        if source_uid not in parents:
            parents.append(source_uid)
        source_summary = self.snapshot_objects([source_object])[source_uid]
        target_summary = self.snapshot_objects([target_object])[target_uid]
        tx = self._merge_transactions.get((int(frame_idx), source_uid, target_uid), {})
        target_before = tx.get("target_before") or {}
        tx_uid = tx.get("merge_transaction_uid")
        event_uid = self._next_event_uid()
        source_version_before = self._current_object_versions.get(source_uid)
        target_version_before = self._current_object_versions.get(target_uid)
        source_version_after = self._next_object_version_uid(source_uid)
        target_version_after = self._next_object_version_uid(target_uid)
        self._append(
            "mapping_events.jsonl",
            {
                "event_uid": event_uid,
                "merge_transaction_uid": tx_uid,
                "frame_uid": self.frame_uid(frame_idx),
                "event_type": "OBJECT_MERGE",
                "source_object_uid": source_uid,
                "target_object_uid": target_uid,
                "before_summary": source_summary,
                "source_before": source_summary,
                "target_before": target_before,
                "after_summary": target_summary,
                "target_after": target_summary,
                "source_consumed_in_transaction": True,
                "source_member_set_before": source_summary.get("member_observation_uids", []),
                "target_member_set_before": target_before.get("member_observation_uids", []),
                "member_intersection_before": sorted(
                    set(source_summary.get("member_observation_uids", []))
                    & set(target_before.get("member_observation_uids", []))
                ),
                "member_union_after": target_summary.get("member_observation_uids", []),
                "reason": {
                    "type": "postprocess_overlap_merge",
                    "overlap_ratio": float(_plain(overlap_ratio)),
                    "visual_similarity": float(_plain(visual_similarity)),
                    "text_similarity": float(_plain(text_similarity)),
                    "text_similarity_source": "VISUAL_PROXY",
                    "merge_overlap_thresh": self.cfg.get("merge_overlap_thresh"),
                    "merge_visual_sim_thresh": self.cfg.get("merge_visual_sim_thresh"),
                    "merge_text_sim_thresh": self.cfg.get("merge_text_sim_thresh"),
                },
                "transaction_uid": tx_uid,
                "parent_event_uids": sorted(
                    {
                        value
                        for value in (
                            self._last_event_by_object.get(source_uid),
                            self._last_event_by_object.get(target_uid),
                        )
                        if value
                    }
                ),
                "input_object_version_uids": [
                    value
                    for value in (source_version_before, target_version_before)
                    if value
                ],
                "output_object_version_uids": [source_version_after, target_version_after],
                "branch_id": "baseline",
                "event_sequence": int(event_uid.rsplit("e", 1)[-1]),
            },
        )
        self._append_object_version(
            frame_idx=frame_idx,
            obj=None,
            summary=source_summary,
            operation="OBJECT_MERGE",
            trigger_event_uid=event_uid,
            status="merged",
        )
        self._append_object_version(
            frame_idx=frame_idx,
            obj=target_object,
            summary=target_summary,
            operation="OBJECT_MERGE",
            trigger_event_uid=event_uid,
            status="active",
            parent_version_uids=[
                value
                for value in (target_version_before, source_version_before)
                if value
            ],
        )
        self.counters["num_object_merges"] += 1

    @evidence_safe(None)
    def record_merge(self, frame_idx: int, before: dict, objects_after) -> None:
        after = self.snapshot_objects(objects_after)
        disappeared = sorted(set(before) - set(after))
        for source_uid in disappeared:
            source_members = set(before[source_uid]["member_observation_uids"])
            candidates = []
            for target_uid, target in after.items():
                overlap = len(source_members.intersection(target["member_observation_uids"]))
                if overlap:
                    candidates.append((overlap, target_uid))
            target_uid = max(candidates)[1] if candidates else None
            if target_uid is None:
                self._record_error(
                    "record_merge",
                    RuntimeError(f"unable to infer target for merged object {source_uid}"),
                )
                continue
            self._merged_parents.setdefault(target_uid, []).append(source_uid)
            self._append(
                "mapping_events.jsonl",
                {
                    "event_uid": self._next_event_uid(),
                    "frame_uid": self.frame_uid(frame_idx),
                    "event_type": "OBJECT_MERGE",
                    "source_object_uid": source_uid,
                    "target_object_uid": target_uid,
                    "before_summary": before[source_uid],
                    "after_summary": after[target_uid],
                    "reason": {
                        "type": "postprocess_overlap_merge",
                        "merge_overlap_thresh": self.cfg.get("merge_overlap_thresh"),
                        "merge_visual_sim_thresh": self.cfg.get("merge_visual_sim_thresh"),
                        "merge_text_sim_thresh": self.cfg.get("merge_text_sim_thresh"),
                        "score_capture": "not_exposed_by_upstream_merge_api",
                    },
                },
            )
            self.counters["num_object_merges"] += 1

    @evidence_safe(dict)
    def snapshot_edges(self, map_edges, objects) -> dict:
        snapshot = {}
        for (obj1_index, obj2_index), edge in map_edges.edges_by_index.items():
            if obj1_index >= len(objects) or obj2_index >= len(objects):
                continue
            obj1_uid = _object_uid(objects[obj1_index])
            obj2_uid = _object_uid(objects[obj2_index])
            key = (obj1_uid, obj2_uid, str(edge.rel_type))
            snapshot[key] = {
                "source_object_uid": obj1_uid,
                "target_object_uid": obj2_uid,
                "relation": str(edge.rel_type),
                "num_detections": int(edge.num_detections),
                "first_detected": edge.first_detected,
            }
        return snapshot

    @evidence_safe(None)
    def record_edge_diff(
        self,
        frame_idx: int,
        before: dict,
        map_edges,
        objects,
        reason: str,
        source_observation_uids: Optional[list] = None,
    ) -> None:
        after = self.snapshot_edges(map_edges, objects)
        all_keys = set(before) | set(after)
        for key in sorted(all_keys):
            previous = before.get(key)
            current = after.get(key)
            if previous == current:
                continue
            if previous is None:
                event_type = "EDGE_ADD"
            elif current is None:
                event_type = "EDGE_DELETE"
            else:
                event_type = "EDGE_UPDATE"
            payload = current or previous
            edge_uid = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:20]
            event_uid = self._next_event_uid()
            source_uid = payload.get("source_object_uid")
            target_uid = payload.get("target_object_uid")
            self._append(
                "mapping_events.jsonl",
                {
                    "event_uid": event_uid,
                    "frame_uid": self.frame_uid(frame_idx),
                    "event_type": event_type,
                    "edge_uid": edge_uid,
                    **payload,
                    "source_observation_uids": source_observation_uids or [],
                    "before_summary": previous,
                    "after_summary": current,
                    "reason": reason,
                    "transaction_uid": f"{self.run_id}_edge_f{int(frame_idx):06d}",
                    "parent_event_uids": sorted(
                        {
                            value
                            for value in (
                                self._last_event_by_object.get(source_uid),
                                self._last_event_by_object.get(target_uid),
                            )
                            if value
                        }
                    ),
                    "input_object_version_uids": [
                        value
                        for value in (
                            self._current_object_versions.get(source_uid),
                            self._current_object_versions.get(target_uid),
                        )
                        if value
                    ],
                    "output_object_version_uids": [],
                    "branch_id": "baseline",
                    "event_sequence": int(event_uid.rsplit("e", 1)[-1]),
                },
            )
            self.counters[f"num_{event_type.lower()}"] += 1

    @evidence_safe(None)
    def begin_vlm_context(self, **context) -> None:
        context = _plain(context)
        image_path = context.pop("input_image_ref", None)
        if image_path:
            context["image_inputs"] = [
                {
                    "artifact_ref": _artifact_ref(
                        Path(image_path), root=self.exp_out_path
                    ),
                    "linked_obs_uids": context.get("input_observation_uids", []),
                }
            ]
        self._vlm_context = context
        self._vlm_pending = []

    def _prompt_fingerprint(self, messages: Any) -> str:
        fragments = []

        def visit(value):
            if isinstance(value, str):
                if not value.startswith("data:image/"):
                    fragments.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(messages)
        return hashlib.sha256("\n".join(fragments).encode("utf-8")).hexdigest()

    def _infer_vlm_call_type(self, messages: Any) -> str:
        text = json.dumps(_plain(messages), ensure_ascii=False).lower()
        if "consolidating multiple captions" in text:
            return "OBJECT_CAPTION_CONSOLIDATION"
        if "accurate captioning objects" in text:
            return "FRAME_CAPTION"
        return "FRAME_EDGE"

    def _normalise_prompt(self, messages: Any) -> tuple[Any, list[dict]]:
        embedded_images = []

        def visit(value):
            if isinstance(value, str) and value.startswith("data:image/"):
                digest = None
                try:
                    encoded = value.split(",", 1)[1]
                    digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
                except Exception:
                    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                embedded_images.append({"sha256": digest})
                return {"embedded_image_sha256": digest}
            if isinstance(value, dict):
                return {str(key): visit(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [visit(item) for item in value]
            return _plain(value)

        return visit(messages), embedded_images

    def _capture_vlm_call(
        self,
        model_name: str,
        messages: Any,
        raw_response: Optional[str],
        latency_ms: float,
        status: str,
        error: Optional[str],
        generation_params: Optional[dict] = None,
        request_id: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            call_type = self._infer_vlm_call_type(messages)
            context = dict(self._vlm_context or {})
            normalised_prompt, embedded_images = self._normalise_prompt(messages)
            image_inputs = list(context.pop("image_inputs", []))
            for index, embedded in enumerate(embedded_images):
                if index < len(image_inputs):
                    image_inputs[index].setdefault("sha256", embedded["sha256"])
                else:
                    image_inputs.append(embedded)
            event_uid = self._next_event_uid()
            self._vlm_pending.append(
                {
                    "event_uid": event_uid,
                    "call_type": call_type,
                    "model_name": model_name,
                    "model_version": model_name,
                    "prompt_version": self.prompt_versions.get(call_type),
                    "prompt_text": json.dumps(
                        normalised_prompt, ensure_ascii=False, sort_keys=True
                    ),
                    "prompt_fingerprint": self._prompt_fingerprint(messages),
                    "image_inputs": image_inputs,
                    "generation_params": _plain(generation_params or {}),
                    "request_id": request_id,
                    "raw_response": raw_response,
                    "parser_version": "conceptgraphs-vlm-parser-v1",
                    "transaction_uid": f"{self.run_id}_vlm",
                    "parent_event_uids": [],
                    "input_object_version_uids": [],
                    "output_object_version_uids": [],
                    "branch_id": "baseline",
                    "event_sequence": int(event_uid.rsplit("e", 1)[-1]),
                    "latency_ms": latency_ms,
                    "status": status,
                    "error": error,
                    **context,
                }
            )
        except Exception as exc:
            self._record_error("capture_vlm_call", exc)

    @evidence_safe(None)
    def finish_vlm_context(self, parsed_outputs: Optional[dict] = None) -> None:
        parsed_outputs = parsed_outputs or {}
        for record in self._vlm_pending:
            record["parsed_output"] = parsed_outputs.get(record["call_type"])
            self._append("vlm_events.jsonl", record)
            self.counters["num_vlm_calls"] += 1
        self._vlm_context = None
        self._vlm_pending = []

    def wrap_openai_client(self, client):
        if not self.enabled or client is None:
            return client
        try:
            return _OpenAIProxy(client, self)
        except Exception as exc:
            self._record_error("wrap_openai_client", exc)
            return client

    def _edge_uids_for_object(self, object_uid: str, edge_snapshot: dict):
        outgoing = []
        incoming = []
        for key, edge in edge_snapshot.items():
            edge_uid = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:20]
            if edge["source_object_uid"] == object_uid:
                outgoing.append(edge_uid)
            if edge["target_object_uid"] == object_uid:
                incoming.append(edge_uid)
        return sorted(outgoing), sorted(incoming)

    def _final_membership(self, objects, map_edges) -> list:
        edge_snapshot = self.snapshot_edges(map_edges, objects) if map_edges is not None else {}
        membership = []
        for index, obj in enumerate(objects or []):
            uid = _object_uid(obj)
            class_histogram = Counter(str(item) for item in obj.get("class_id", []))
            outgoing, incoming = self._edge_uids_for_object(uid, edge_snapshot)
            member_occurrences = [str(item) for item in obj.get("obs_uids", [])]
            member_counts = Counter(member_occurrences)
            members = list(dict.fromkeys(member_occurrences))
            duplicate_members = {
                member_uid: count
                for member_uid, count in member_counts.items()
                if count > 1
            }
            self.counters["duplicate_membership_occurrences"] += sum(
                count - 1 for count in duplicate_members.values()
            )
            self._final_member_observations.extend(members)
            bbox = _bbox_summary(obj)
            membership.append(
                {
                    "object_uid": uid,
                    "current_object_index": index,
                    "curr_obj_num": obj.get("curr_obj_num"),
                    "status": "active",
                    "class_name": obj.get("class_name"),
                    "class_histogram": dict(class_histogram),
                    "member_observation_uids": members,
                    "duplicate_member_observation_uids": duplicate_members,
                    "num_detections": int(obj.get("num_detections", len(members))),
                    "bbox_center": bbox["center"],
                    "bbox_extent": bbox["extent"],
                    "n_points": _point_count(obj),
                    "consolidated_caption": obj.get("consolidated_caption"),
                    "parent_or_merged_from_object_uids": sorted(
                        set(self._merged_parents.get(uid, []))
                    ),
                    "outgoing_edge_uids": outgoing,
                    "incoming_edge_uids": incoming,
                }
            )
        return membership

    def _missing_reference_count(self) -> int:
        missing = 0
        for frame_uid in self._observation_frames.values():
            missing += int(frame_uid not in self._frames)
        for obs_uid in self._association_observations:
            missing += int(obs_uid not in self._observation_frames)
        for obs_uid in self._final_member_observations:
            missing += int(obs_uid not in self._observation_frames)
        return missing

    @evidence_safe(None)
    def close(
        self,
        status: str,
        objects=None,
        map_edges=None,
        failure_reason: Optional[str] = None,
    ) -> None:
        if self._closed:
            return
        membership = self._final_membership(objects, map_edges) if objects is not None else []
        self._write_json("final_membership.json", membership)
        edge_count = len(self.snapshot_edges(map_edges, objects)) if map_edges is not None else 0
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "num_frames": self.counters["num_frames"],
            "num_raw_detections": self.counters["num_raw_detections"],
            "num_kept_observations": self.counters["num_kept_observations"],
            "num_rejected_observations": self.counters["num_rejected_observations"],
            "num_create_decisions": self.counters["num_create_decisions"],
            "num_associate_decisions": self.counters["num_associate_decisions"],
            "num_discard_decisions": self.counters["num_discard_decisions"],
            "num_object_merges": self.counters["num_object_merges"],
            "num_filtered_objects": self.counters["num_filtered_objects"],
            "num_final_objects": len(membership),
            "num_vlm_calls": self.counters["num_vlm_calls"],
            "num_edges": edge_count,
            "missing_reference_count": self._missing_reference_count(),
            "duplicate_membership_occurrences": self.counters[
                "duplicate_membership_occurrences"
            ],
            "logging_error_count": len(self._errors),
            "logging_errors": list(self._errors),
            "evidence_mode": self.manifest.get("evidence_mode", "best_effort"),
            "evidence_schema_version": SCHEMA_VERSION,
            "object_version_count": sum(self._object_versions.values()),
            "filter_trace_count": sum(1 for _ in self._iter_jsonl("filter_trace.jsonl")),
        }
        self._write_json("evidence_summary.json", summary)
        self.manifest.update(
            {
                "status": status,
                "end_time": _utc_now(),
                "failure_reason": failure_reason,
                "evidence_summary_ref": _artifact_ref(
                    self.evidence_dir / "evidence_summary.json",
                    root=self.exp_out_path,
                ),
                "final_outputs": self._final_output_info(),
            }
        )
        self._write_json("manifest.json", self.manifest)
        for handle in self._files.values():
            handle.close()
        self._closed = True
        audit_result = None
        if status in {"completed", "early_exit"}:
            try:
                from conceptgraph.audit.evidence_audit import audit_evidence

                audit_result = audit_evidence(
                    self.evidence_dir,
                    strict=self.evidence_mode == "strict",
                    write=True,
                )
                evidence_valid = audit_result["summary"]["gate_status"] == "PASS"
                self.manifest["status"] = (
                    "MAP_COMPLETED_EVIDENCE_VALID"
                    if evidence_valid
                    else "MAP_COMPLETED_EVIDENCE_INVALID"
                )
                self.manifest["audit_summary_ref"] = _artifact_ref(
                    self.exp_out_path / "audit" / "audit_summary.json",
                    root=self.exp_out_path,
                )
                self._write_json("manifest.json", self.manifest)
                if self.evidence_mode == "strict" and not evidence_valid:
                    raise RuntimeError("strict evidence validation failed")
            except Exception as exc:
                if audit_result is None:
                    self.manifest["status"] = "MAP_COMPLETED_EVIDENCE_INVALID"
                    self.manifest["audit_error"] = f"{type(exc).__name__}: {exc}"
                    self._write_json("manifest.json", self.manifest)
                if self.evidence_mode == "strict":
                    raise
        elif status == "failed":
            self.manifest["status"] = "MAP_FAILED"
            self._write_json("manifest.json", self.manifest)
        try:
            atexit.unregister(self._finalize_abandoned_run)
        except Exception:
            pass

    def _final_output_info(self) -> dict:
        outputs = {}
        try:
            for path in sorted(self.exp_out_path.iterdir()):
                if not path.is_file() or path.name.startswith("config_params"):
                    continue
                if path.suffix.lower() not in {".json", ".gz"}:
                    continue
                digest = _sha256_file(path)
                if digest:
                    outputs[path.name] = {
                        "artifact_ref": _artifact_ref(path, root=self.exp_out_path),
                        "sha256": digest,
                    }
        except Exception:
            pass
        return outputs

    def _iter_jsonl(self, filename: str):
        path = self.evidence_dir / filename
        if not path.exists():
            return iter(())
        def _records():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        return _records()

    def _finalize_abandoned_run(self) -> None:
        if self.enabled and not self._closed:
            self.close(
                "failed",
                failure_reason="process exited before explicit evidence finalization",
            )


class _CompletionsProxy:
    def __init__(self, delegate, recorder: EvidenceRecorder):
        self._delegate = delegate
        self._recorder = recorder

    def create(self, *args, **kwargs):
        started = time.perf_counter()
        model = str(kwargs.get("model", "unknown"))
        messages = kwargs.get("messages", [])
        generation_params = {
            key: kwargs[key]
            for key in ("temperature", "max_tokens", "seed", "top_p")
            if key in kwargs
        }
        try:
            response = self._delegate.create(*args, **kwargs)
        except Exception as exc:
            self._recorder._capture_vlm_call(
                model,
                messages,
                None,
                (time.perf_counter() - started) * 1000.0,
                "error",
                f"{type(exc).__name__}: {exc}",
                generation_params,
            )
            raise
        raw_response = None
        try:
            raw_response = response.choices[0].message.content
        except Exception:
            raw_response = str(response)
        self._recorder._capture_vlm_call(
            model,
            messages,
            raw_response,
            (time.perf_counter() - started) * 1000.0,
            "ok",
            None,
            generation_params,
            str(getattr(response, "id", "")) or None,
        )
        return response

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class _ChatProxy:
    def __init__(self, delegate, recorder: EvidenceRecorder):
        self._delegate = delegate
        self.completions = _CompletionsProxy(delegate.completions, recorder)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class _OpenAIProxy:
    def __init__(self, delegate, recorder: EvidenceRecorder):
        self._delegate = delegate
        self.chat = _ChatProxy(delegate.chat, recorder)

    def __getattr__(self, name):
        return getattr(self._delegate, name)
