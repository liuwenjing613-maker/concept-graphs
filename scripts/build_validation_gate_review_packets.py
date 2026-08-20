#!/usr/bin/env python3
"""Build traceable, human-readable evidence for Audit Validity Gate R1.

The original evidence packets are useful diagnostic exports, but their flat
image gallery does not make three distinctions explicit:

1. the exact observation and numeric records consumed by the checker;
2. representative views selected to help a person interpret those records;
3. the exact final map objects needed to judge downstream harm.

This builder adds ``review_evidence.json`` and deterministic review images to
every frozen R1 case without changing the finding, worklist, map, or label.  It
also writes a top-level manifest that the browser service verifies before it
accepts labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "1.0.0"
REVIEW_FILENAME = "review_evidence.json"
MANIFEST_FILENAME = "review_evidence_manifest.json"

# These are not missing ledger records: they are missing *visual state
# snapshots* that would be required to visually verify the numeric trigger.
# Keeping them explicit is preferable to silently presenting a post-event point
# cloud as though it were the pre-event state used by the checker.
CHECKER_VISUAL_GAPS: dict[str, list[dict[str, Any]]] = {
    "SEG-002": [
        {
            "code": "PRE_DBSCAN_POINT_SNAPSHOT_NOT_RETAINED",
            "critical": True,
            "message": (
                "规则使用了精确保存的聚类数量与比例，但本次运行没有保存 DBSCAN 前的点坐标；"
                "页面只能展示原始 RGB、深度、mask、数值统计和 DBSCAN 后点云，不能把后者冒充触发时的多簇几何。"
            ),
        }
    ],
    "GEO-003": [
        {
            "code": "PRE_DBSCAN_POINT_SNAPSHOT_NOT_RETAINED",
            "critical": True,
            "message": (
                "规则使用了 DBSCAN 前的多簇统计，但 DBSCAN 前点坐标未留存；"
                "现有 3D 点是仅保留最大簇后的结果，无法直接目视复核原始多簇形状。"
            ),
        }
    ],
    "GEO-005": [
        {
            "code": "DENOISE_BEFORE_AFTER_PCD_NOT_RETAINED",
            "critical": True,
            "message": (
                "对象版本保存了去噪前后的点数、bbox 与成员关系，但没有保存两个时刻的完整对象点云；"
                "因此可以核对数值突降，不能目视确认被删掉的是主体还是噪声。"
            ),
        }
    ],
    "FUSE-007": [
        {
            "code": "FUSION_VERSION_PCD_NOT_RETAINED",
            "critical": True,
            "message": (
                "对象版本保存了融合前后的中心、尺度、点数和成员关系，但没有保存两个版本的完整点云快照；"
                "页面会展示精确数值变化和最终对象，不能把成员点云并集冒充当时的融合状态。"
            ),
        }
    ],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def sha256_file_cached(path_text: str) -> str:
    return sha256_file(Path(path_text))


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    scene = str(row.get("scene_id") or "")
    uid = str(row.get("case_uid") or row.get("finding_uid") or "")
    if not scene or not uid:
        raise ValueError("worklist row needs scene_id and case_uid/finding_uid")
    return scene, uid


def uid_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        value = [value]
    return list(dict.fromkeys(str(item) for item in value if item))


def frame_number(value: Any) -> int | None:
    text = str(value or "")
    marker = "_f"
    if marker not in text:
        return None
    suffix = text.rsplit(marker, 1)[-1]
    digits = "".join(character for character in suffix if character.isdigit())
    return int(digits[:6]) if digits else None


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(item) for item in value]
    return value


def resolve_path(run_root: Path, evidence_root: Path, value: Any) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    if value is None:
        raise FileNotFoundError("empty artifact path")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        run_root / path,
        evidence_root / path,
        evidence_root.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_array(run_root: Path, evidence_root: Path, ref: Any) -> np.ndarray | None:
    if not ref:
        return None
    parsed = ref if isinstance(ref, dict) else {"path": ref}
    path = resolve_path(run_root, evidence_root, parsed)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            key = parsed.get("key")
            if key is None:
                key = archive.files[0]
            array = np.asarray(archive[key])
    elif suffixes.endswith(".npy"):
        array = np.asarray(np.load(path, allow_pickle=False))
    else:
        return None
    index = parsed.get("index")
    if index is not None:
        array = array[int(index)]
    return array


def artifact_summary(run_root: Path, evidence_root: Path, ref: Any) -> dict[str, Any] | None:
    if not ref:
        return None
    parsed = dict(ref) if isinstance(ref, dict) else {"path": str(ref)}
    path = resolve_path(run_root, evidence_root, parsed)
    actual_sha = sha256_file_cached(str(path)) if path.is_file() else None
    expected_sha = parsed.get("sha256")
    return {
        "path": str(parsed.get("path")),
        "format": parsed.get("format"),
        "key": parsed.get("key"),
        "index": parsed.get("index"),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "hash_matches": expected_sha is None or expected_sha == actual_sha,
        "shape": parsed.get("shape"),
        "dtype": parsed.get("dtype"),
    }


def compact_histogram(values: Iterable[Any], limit: int = 6) -> dict[str, int]:
    counts = Counter(str(value or "unknown") for value in values)
    return dict(counts.most_common(limit))


def version_summary(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "object_version_uid": item.get("object_version_uid"),
        "object_uid": item.get("object_uid"),
        "version": item.get("version"),
        "operation": item.get("operation"),
        "status": item.get("status"),
        "frame_uid": item.get("frame_uid"),
        "n_points": item.get("n_points"),
        "num_detections": item.get("num_detections"),
        "unique_frame_count": item.get("unique_frame_count"),
        "member_count": len(item.get("member_observation_uids") or []),
        "class_name": item.get("class_name"),
        "class_histogram": item.get("class_histogram"),
        "dominant_class": item.get("dominant_class"),
        "dominant_class_ratio": item.get("dominant_class_ratio"),
        "bbox_center": item.get("bbox_center"),
        "bbox_extent": item.get("bbox_extent"),
        "bbox_volume": item.get("bbox_volume"),
        "parent_version_uids": item.get("parent_version_uids") or [],
        "trigger_event_uid": item.get("trigger_event_uid"),
    }


class SceneEvidence:
    def __init__(self, validation_root: Path, scene_id: str):
        self.scene_id = scene_id
        self.run_root = (validation_root / "runs" / scene_id / "formal").resolve()
        self.evidence_root = self.run_root / "evidence"
        self.manifest_path = self.evidence_root / "manifest.json"
        self.manifest = read_json(self.manifest_path)
        self.frames = read_jsonl(self.evidence_root / "frames.jsonl")
        self.observations = read_jsonl(self.evidence_root / "observations.jsonl")
        self.associations = read_jsonl(self.evidence_root / "associations.jsonl")
        self.events = read_jsonl(self.evidence_root / "mapping_events.jsonl")
        self.versions = read_jsonl(self.evidence_root / "object_versions.jsonl")
        self.final_membership = read_json(self.evidence_root / "final_membership.json")
        self.frame_by_uid = {str(item.get("frame_uid")): item for item in self.frames}
        self.obs_by_uid = {str(item.get("obs_uid")): item for item in self.observations}
        self.assoc_by_obs = {str(item.get("obs_uid")): item for item in self.associations}
        self.event_by_uid = {
            str(item.get("event_uid")): item for item in self.events if item.get("event_uid")
        }
        self.version_by_uid = {
            str(item.get("object_version_uid")): item
            for item in self.versions
            if item.get("object_version_uid")
        }
        self.versions_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.versions:
            self.versions_by_object[str(item.get("object_uid"))].append(item)
        for items in self.versions_by_object.values():
            items.sort(key=lambda item: int(item.get("version") or 0))
        self.final_by_uid = {
            str(item.get("object_uid")): item for item in self.final_membership
        }
        self.owners: dict[str, list[str]] = defaultdict(list)
        for item in self.final_membership:
            object_uid = str(item.get("object_uid"))
            for obs_uid in item.get("member_observation_uids") or []:
                self.owners[str(obs_uid)].append(object_uid)

        pcd_candidates = sorted(self.run_root.glob("pcd_*.pkl.gz"))
        if len(pcd_candidates) != 1:
            raise ValueError(f"expected one final pcd pickle in {self.run_root}, found {len(pcd_candidates)}")
        self.final_pickle_path = pcd_candidates[0]
        expected = (
            self.manifest.get("final_outputs", {})
            .get(self.final_pickle_path.name, {})
            .get("sha256")
        )
        self.final_pickle_sha256 = sha256_file(self.final_pickle_path)
        if expected and expected != self.final_pickle_sha256:
            raise ValueError(f"final pickle hash mismatch for {scene_id}")
        with gzip.open(self.final_pickle_path, "rb") as handle:
            payload = pickle.load(handle)
        objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
        self.output_object_by_uid: dict[str, dict[str, Any]] = {}
        for item in objects:
            uid = str(item.get("id") or item.get("object_uid") or item.get("uuid") or "")
            if uid:
                self.output_object_by_uid[uid] = item
        self._validate_final_output()

    def _validate_final_output(self) -> None:
        missing = sorted(set(self.final_by_uid) - set(self.output_object_by_uid))
        if missing:
            raise ValueError(f"{self.scene_id}: {len(missing)} final objects missing from output pickle")
        problems = []
        for uid, final in self.final_by_uid.items():
            output = self.output_object_by_uid[uid]
            expected_members = list(final.get("member_observation_uids") or [])
            actual_members = list(output.get("obs_uids") or [])
            points = np.asarray(output.get("pcd_np"))
            if Counter(expected_members) != Counter(actual_members):
                problems.append(f"{uid}:membership")
            if points.ndim != 2 or points.shape[1] != 3 or int(final.get("n_points") or 0) != len(points):
                problems.append(f"{uid}:points")
        if problems:
            raise ValueError(f"{self.scene_id}: final output mismatch: {problems[:5]}")

    def resolve_final(self, object_uid: str) -> list[str]:
        if object_uid in self.final_by_uid:
            return [object_uid]
        direct = [
            uid
            for uid, item in self.final_by_uid.items()
            if object_uid in set(map(str, item.get("parent_or_merged_from_object_uids") or []))
        ]
        if direct:
            return direct
        # Fall back to merge events if an old run did not flatten merge ancestry
        # into final_membership.
        frontier = [object_uid]
        visited = set(frontier)
        while frontier:
            source = frontier.pop(0)
            targets = []
            for event in self.events:
                if event.get("event_type") != "OBJECT_MERGE":
                    continue
                candidate_source = str(
                    event.get("source_object_uid")
                    or event.get("consumed_object_uid")
                    or ""
                )
                if candidate_source != source:
                    continue
                target = str(event.get("target_object_uid") or event.get("object_uid") or "")
                if target:
                    targets.append(target)
            for target in targets:
                if target in self.final_by_uid:
                    direct.append(target)
                elif target not in visited:
                    visited.add(target)
                    frontier.append(target)
        return list(dict.fromkeys(direct))

    def final_summary(self, uid: str) -> dict[str, Any]:
        item = self.final_by_uid[uid]
        members = [
            self.obs_by_uid[value]
            for value in item.get("member_observation_uids") or []
            if value in self.obs_by_uid
        ]
        frames = [frame_number(member.get("frame_uid")) for member in members]
        frames = [value for value in frames if value is not None]
        output = self.output_object_by_uid[uid]
        points = np.asarray(output.get("pcd_np"), dtype=float)
        return {
            "object_uid": uid,
            "status": item.get("status", "active"),
            "class_name": item.get("class_name"),
            "observed_class_histogram": compact_histogram(
                member.get("class_name") for member in members
            ),
            "recorded_class_histogram": item.get("class_histogram"),
            "member_count": len(item.get("member_observation_uids") or []),
            "unique_frame_count": len({member.get("frame_uid") for member in members}),
            "first_frame": min(frames) if frames else None,
            "last_frame": max(frames) if frames else None,
            "n_points": item.get("n_points"),
            "bbox_center": item.get("bbox_center"),
            "bbox_extent": item.get("bbox_extent"),
            "parent_or_merged_from_object_uids": item.get("parent_or_merged_from_object_uids") or [],
            "pcd_sha256": sha256_array(points),
            "pcd_shape": list(points.shape),
            "source_pickle": self.final_pickle_path.name,
            "source_pickle_sha256": self.final_pickle_sha256,
            "membership_matches_final_output": Counter(item.get("member_observation_uids") or [])
            == Counter(output.get("obs_uids") or []),
            "point_count_matches_final_output": int(item.get("n_points") or 0) == len(points),
        }


def trigger_observation_uids(case: dict[str, Any], scene: SceneEvidence) -> list[str]:
    scope = case.get("scope") or {}
    values = uid_list(scope.get("obs_uid")) + uid_list(scope.get("obs_uids"))
    for event_uid in uid_list(scope.get("event_uid")):
        event = scene.event_by_uid.get(event_uid, {})
        values.extend(uid_list(event.get("obs_uid")))
        values.extend(uid_list(event.get("observation_uid")))
    version_uid = scope.get("object_version_uid")
    version = scene.version_by_uid.get(str(version_uid), {}) if version_uid else {}
    if version:
        parents = [scene.version_by_uid.get(str(uid), {}) for uid in version.get("parent_version_uids") or []]
        before = set(value for parent in parents for value in parent.get("member_observation_uids") or [])
        after = set(version.get("member_observation_uids") or [])
        values.extend(sorted(after - before))
        event = scene.event_by_uid.get(str(version.get("trigger_event_uid") or ""), {})
        values.extend(uid_list(event.get("obs_uid")))
        values.extend(uid_list(event.get("observation_uid")))
    return [value for value in dict.fromkeys(values) if value in scene.obs_by_uid]


def relevant_object_roles(
    case: dict[str, Any], trigger_uids: list[str], scene: SceneEvidence
) -> dict[str, list[str]]:
    scope = case.get("scope") or {}
    roles: dict[str, list[str]] = defaultdict(list)

    def add(uid: Any, role: str) -> None:
        if uid is None or uid == "":
            return
        key = str(uid)
        if role not in roles[key]:
            roles[key].append(role)

    add(scope.get("object_uid"), "primary_object")
    for uid in uid_list(scope.get("object_uids")):
        add(uid, "compared_object")
    for field, role in (
        ("chosen_target_object_uid", "chosen_target"),
        ("spatial_top_object_uid", "spatial_top_candidate"),
        ("visual_top_object_uid", "visual_top_candidate"),
        ("counterfactual_alternate_object_uid", "counterfactual_alternate"),
    ):
        add(scope.get(field), role)
    aggregate = uid_list(scope.get("aggregate_top_object_uids"))
    if aggregate:
        add(aggregate[0], "aggregate_top1")
    if len(aggregate) > 1:
        add(aggregate[1], "aggregate_top2")
    for uid in uid_list(scope.get("alternate_object_uids")):
        add(uid, "alternate_candidate")
    for role, uid in (scope.get("association_candidate_roles") or {}).items():
        add(uid, str(role))
    for role_map in (scope.get("association_candidate_roles_by_observation") or {}).values():
        for role, uid in (role_map or {}).items():
            add(uid, str(role))

    # Rebuild roles from the exact association rows as a guard against a stale
    # or incomplete case projection.
    for obs_uid in trigger_uids:
        association = scene.assoc_by_obs.get(obs_uid, {})
        add(association.get("target_object_uid"), "association_target")
        candidates = association.get("top_candidates") or []
        for index, item in enumerate(candidates[:3], 1):
            add(item.get("object_uid"), f"association_top{index}")
        for owner in scene.owners.get(obs_uid, []):
            add(owner, "final_owner_of_trigger")
    return dict(roles)


def aliases_for(uids: Iterable[str]) -> dict[str, str]:
    return {uid: f"O{index + 1}" for index, uid in enumerate(dict.fromkeys(uids))}


def crop_bounds(shape: tuple[int, ...], bbox: Any, margin_ratio: float = 0.45) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except Exception:
        return 0, 0, width, height
    margin = max(20, int(max(x2 - x1, y2 - y1) * margin_ratio))
    return (
        max(0, int(math.floor(x1)) - margin),
        max(0, int(math.floor(y1)) - margin),
        min(width, int(math.ceil(x2)) + margin),
        min(height, int(math.ceil(y2)) + margin),
    )


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = rgb.astype(float).copy()
    valid = np.asarray(mask, dtype=bool)
    if valid.shape == output.shape[:2]:
        output[valid] = 0.52 * output[valid] + 0.48 * np.asarray(color, dtype=float)
    return np.clip(output, 0, 255).astype(np.uint8)


def equal_axis(axis: Any, x: np.ndarray, y: np.ndarray) -> None:
    if not len(x):
        return
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
    axis.set_xlim(xmid - span / 2, xmid + span / 2)
    axis.set_ylim(ymid - span / 2, ymid + span / 2)
    axis.set_aspect("equal", adjustable="box")


def render_observation_panel(
    case_dir: Path,
    alias: str,
    obs: dict[str, Any],
    scene: SceneEvidence,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    frame = scene.frame_by_uid.get(str(obs.get("frame_uid")), {})
    rgb_path = resolve_path(scene.run_root, scene.evidence_root, frame.get("rgb_ref") or frame.get("rgb_path"))
    depth_path = resolve_path(scene.run_root, scene.evidence_root, frame.get("depth_ref") or frame.get("depth_path"))
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.asarray(Image.open(depth_path), dtype=float)
    raw = load_array(scene.run_root, scene.evidence_root, obs.get("raw_mask_ref") or obs.get("mask_ref"))
    processed = load_array(scene.run_root, scene.evidence_root, obs.get("processed_mask_ref"))
    raw = np.asarray(raw, dtype=bool) if raw is not None else np.zeros(rgb.shape[:2], dtype=bool)
    processed = (
        np.asarray(processed, dtype=bool)
        if processed is not None
        else np.zeros(rgb.shape[:2], dtype=bool)
    )
    bounds = crop_bounds(rgb.shape, obs.get("bbox_2d"))
    x1, y1, x2, y2 = bounds
    rgb_crop = rgb[y1:y2, x1:x2]
    raw_crop = raw[y1:y2, x1:x2]
    processed_crop = processed[y1:y2, x1:x2]
    depth_crop = depth[y1:y2, x1:x2]
    removed = np.logical_and(raw_crop, ~processed_crop)
    added = np.logical_and(processed_crop, ~raw_crop)
    changes = rgb_crop.astype(float).copy()
    changes[removed] = 0.35 * changes[removed] + 0.65 * np.asarray((255, 133, 27))
    changes[added] = 0.35 * changes[added] + 0.65 * np.asarray((172, 61, 255))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(rgb_crop)
    axes[0, 0].set_title("Exact RGB context")
    axes[0, 1].imshow(mask_overlay(rgb_crop, raw_crop, (255, 65, 85)))
    axes[0, 1].set_title(f"Raw mask | area={int(raw.sum())}")
    axes[0, 2].imshow(mask_overlay(rgb_crop, processed_crop, (0, 184, 217)))
    axes[0, 2].set_title(f"Processed mask | area={int(processed.sum())}")
    axes[1, 0].imshow(np.clip(changes, 0, 255).astype(np.uint8))
    axes[1, 0].set_title(f"Mask change | removed={int(np.logical_and(raw, ~processed).sum())}")
    masked_depth = np.where(processed_crop & np.isfinite(depth_crop) & (depth_crop > 0), depth_crop, np.nan)
    axes[1, 1].imshow(masked_depth, cmap="viridis")
    axes[1, 1].set_title("Exact depth inside processed mask")
    points = load_array(scene.run_root, scene.evidence_root, obs.get("pcd_ref"))
    points = np.asarray(points, dtype=float) if points is not None else np.empty((0, 3))
    if points.ndim == 2 and points.shape[1] == 3 and len(points):
        step = max(1, len(points) // 3000)
        shown = points[::step]
        axes[1, 2].scatter(shown[:, 0], shown[:, 2], s=4, alpha=0.65, color="#2457d6")
        equal_axis(axes[1, 2], shown[:, 0], shown[:, 2])
    axes[1, 2].set_title(f"Stored post-DBSCAN PCD | n={len(points)}")
    axes[1, 2].set_xlabel("world x")
    axes[1, 2].set_ylabel("world z")
    for axis in axes.flat[:5]:
        axis.axis("off")
    fig.suptitle(f"{alias} | {obs.get('class_name') or 'unknown'} | {obs.get('obs_uid')}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    name = f"review_observation_{alias}.png"
    fig.savefig(case_dir / name, dpi=130)
    plt.close(fig)
    return name


PALETTE = (
    "#e43f5a",
    "#1976d2",
    "#00a878",
    "#f59e0b",
    "#7c3aed",
    "#d946ef",
    "#0f766e",
    "#6b7280",
)


def render_final_objects(
    case_dir: Path,
    final_uids: list[str],
    aliases: dict[str, str],
    scene: SceneEvidence,
) -> list[str]:
    if not final_uids:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = []
    for index, uid in enumerate(final_uids):
        item = scene.output_object_by_uid[uid]
        points = np.asarray(item.get("pcd_np"), dtype=float)
        step = max(1, len(points) // 6000)
        values.append((uid, points, points[::step], PALETTE[index % len(PALETTE)]))

    # Shared world axes preserve whether objects occupy the same place or are
    # physically separated.  This is essential for split/merge judgments.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    projections = ((0, 1, "world x", "world y"), (0, 2, "world x", "world z"), (1, 2, "world y", "world z"))
    for axis, (left, right, xlabel, ylabel) in zip(axes, projections):
        all_x, all_y = [], []
        for uid, _, shown, color in values:
            axis.scatter(
                shown[:, left],
                shown[:, right],
                s=1.4,
                alpha=0.55,
                color=color,
                label=f"{aliases[uid]} {scene.final_by_uid[uid].get('class_name') or ''}",
            )
            all_x.append(shown[:, left])
            all_y.append(shown[:, right])
        equal_axis(axis, np.concatenate(all_x), np.concatenate(all_y))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.15)
    axes[0].legend(fontsize=8, markerscale=4)
    fig.suptitle("Exact final-map point coordinates | shared world axes | alias colors")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    relative_name = "review_final_objects_relative.png"
    fig.savefig(case_dir / relative_name, dpi=140)
    plt.close(fig)

    # Per-object normalized views make small or thin objects inspectable without
    # losing the shared-coordinate comparison above.
    fig, axes = plt.subplots(len(values), 3, figsize=(14, max(4, 3.6 * len(values))), squeeze=False)
    for row, (uid, points, shown, color) in enumerate(values):
        for column, (left, right, xlabel, ylabel) in enumerate(projections):
            axis = axes[row, column]
            axis.scatter(shown[:, left], shown[:, right], s=1.5, alpha=0.6, color=color)
            equal_axis(axis, shown[:, left], shown[:, right])
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.15)
            if column == 0:
                final = scene.final_by_uid[uid]
                axis.set_title(
                    f"{aliases[uid]} | {final.get('class_name') or 'unknown'} | "
                    f"obs={len(final.get('member_observation_uids') or [])} | points={len(points)}",
                    loc="left",
                    fontsize=9,
                )
    fig.suptitle("Exact final-map objects | per-object scaled views")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    detail_name = "review_final_objects_detail.png"
    fig.savefig(case_dir / detail_name, dpi=140)
    plt.close(fig)
    return [relative_name, detail_name]


def render_representative_view_assets(
    case_dir: Path,
    obs: dict[str, Any],
    scene: SceneEvidence,
) -> tuple[dict[str, str], dict[str, dict[str, Any] | None]]:
    """Regenerate reviewer crops from the exact ledger-referenced artifacts."""

    from PIL import Image

    frame = scene.frame_by_uid.get(str(obs.get("frame_uid")), {})
    rgb_ref = frame.get("rgb_ref") or frame.get("rgb_path")
    rgb_path = resolve_path(scene.run_root, scene.evidence_root, rgb_ref)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    processed_ref = obs.get("processed_mask_ref") or obs.get("raw_mask_ref") or obs.get("mask_ref")
    processed = load_array(scene.run_root, scene.evidence_root, processed_ref)
    if processed is None:
        raise ValueError(f"representative view has no retained mask: {obs.get('obs_uid')}")
    processed = np.asarray(processed, dtype=bool)
    if processed.shape != rgb.shape[:2]:
        raise ValueError(
            f"representative mask shape mismatch for {obs.get('obs_uid')}: {processed.shape} vs {rgb.shape[:2]}"
        )
    x1, y1, x2, y2 = crop_bounds(rgb.shape, obs.get("bbox_2d"))
    rgb_crop = rgb[y1:y2, x1:x2]
    mask_crop = processed[y1:y2, x1:x2]
    masked = np.full_like(rgb_crop, 238)
    masked[mask_crop] = rgb_crop[mask_crop]
    token = hashlib.sha256(str(obs.get("obs_uid")).encode("utf-8")).hexdigest()[:16]
    context_name = f"review_view_context_{token}.png"
    masked_name = f"review_view_processed_mask_{token}.png"
    Image.fromarray(rgb_crop).save(case_dir / context_name, format="PNG", compress_level=6)
    Image.fromarray(masked).save(case_dir / masked_name, format="PNG", compress_level=6)
    source_artifacts = {
        "rgb_ref": artifact_summary(scene.run_root, scene.evidence_root, rgb_ref),
        "processed_mask_ref": artifact_summary(
            scene.run_root, scene.evidence_root, processed_ref
        ),
    }
    return {
        "context_crop": context_name,
        "masked_crop": masked_name,
    }, source_artifacts


def selected_view_records(
    case_dir: Path,
    scene: SceneEvidence,
    aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], bool]:
    path = case_dir / "view_selection.json"
    selection = (read_json(path).get("observations") or []) if path.exists() else []
    records = []
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"selected": 0, "total": 0})
    artifact_hashes_ok = True
    for item in selection:
        uid = str(item.get("obs_uid") or "")
        if uid not in scene.obs_by_uid:
            raise ValueError(f"view_selection references unknown observation: {uid}")
        obs = scene.obs_by_uid[uid]
        object_uids = list(map(str, item.get("object_uids") or []))
        for object_uid in object_uids:
            coverage[object_uid]["selected"] += 1
        assets, source_artifacts = render_representative_view_assets(case_dir, obs, scene)
        artifact_hashes_ok = artifact_hashes_ok and all(
            value is None or value.get("hash_matches")
            for value in source_artifacts.values()
        )
        records.append(
            {
                "obs_uid": uid,
                "frame_uid": item.get("frame_uid"),
                "frame_number": frame_number(item.get("frame_uid")),
                "class_name": obs.get("class_name"),
                "confidence": obs.get("confidence"),
                "n_points": obs.get("n_points"),
                "object_uids": object_uids,
                "object_aliases": [aliases.get(value, value[-8:]) for value in object_uids],
                "object_roles": item.get("object_roles") or [],
                "selection_reasons": item.get("selection_reasons") or [],
                "assets": assets,
                "source_artifacts": source_artifacts,
            }
        )
    for object_uid in aliases:
        value = coverage[object_uid]
        version_uid = None
        # view_selection was built from a decision version for association
        # objects; its exact version is recorded in association_candidates.json.
        association_path = case_dir / "association_candidates.json"
        if association_path.exists():
            roles = read_json(association_path).get("roles") or {}
            for role in roles.values():
                if str(role.get("object_uid")) == object_uid:
                    version_uid = role.get("object_version_uid")
                    break
        version = scene.version_by_uid.get(str(version_uid), {}) if version_uid else {}
        if version:
            value["total"] = len(version.get("member_observation_uids") or [])
        elif object_uid in scene.final_by_uid:
            value["total"] = len(scene.final_by_uid[object_uid].get("member_observation_uids") or [])
        else:
            versions = scene.versions_by_object.get(object_uid, [])
            value["total"] = len(versions[-1].get("member_observation_uids") or []) if versions else 0
    return records, dict(coverage), artifact_hashes_ok


def association_records(
    trigger_uids: list[str], scene: SceneEvidence, aliases: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for obs_uid in trigger_uids:
        item = scene.assoc_by_obs.get(obs_uid)
        if not item:
            continue
        candidates = []
        version_map = {
            str(uid): version
            for uid, version in zip(
                item.get("object_uids_before") or [],
                item.get("candidate_object_version_uids") or [],
            )
        }
        for rank, candidate in enumerate(item.get("top_candidates") or [], 1):
            uid = str(candidate.get("object_uid") or "")
            candidates.append(
                {
                    "rank": rank,
                    "object_uid": uid,
                    "object_alias": aliases.get(uid, uid[-8:]),
                    "object_version_uid": version_map.get(uid),
                    "spatial_score": candidate.get("spatial_score"),
                    "visual_score": candidate.get("visual_score"),
                    "aggregate_score": candidate.get("aggregate_score"),
                }
            )
        target = str(item.get("target_object_uid") or "")
        output.append(
            {
                "obs_uid": obs_uid,
                "event_uid": item.get("event_uid"),
                "decision": item.get("decision"),
                "target_object_uid": target,
                "target_object_alias": aliases.get(target, target[-8:]),
                "target_object_version_before": item.get("target_object_version_before"),
                "target_object_version_after": item.get("target_object_version_after"),
                "top1_score": item.get("top1_score"),
                "top2_score": item.get("top2_score"),
                "margin": item.get("margin"),
                "sim_threshold": item.get("sim_threshold"),
                "match_method": item.get("match_method"),
                "phys_bias": item.get("phys_bias"),
                "similarity_evidence_valid": item.get("similarity_evidence_valid"),
                "candidates": candidates,
                "source": "evidence/associations.jsonl + referenced similarity matrix",
            }
        )
    return output


def build_case(
    validation_root: Path,
    row: dict[str, Any],
    scene: SceneEvidence,
) -> dict[str, Any]:
    scene_id, case_uid = case_key(row)
    case_dir = Path(str(row.get("case_dir"))).resolve()
    expected_root = (validation_root / "cases" / scene_id).resolve()
    if expected_root != case_dir and expected_root not in case_dir.parents:
        raise ValueError(f"case directory escaped validation root: {case_dir}")
    case_path = case_dir / "case.json"
    case = read_json(case_path)
    if str(case.get("finding_uid")) != case_uid:
        raise ValueError(f"case UID mismatch at {case_path}")

    trigger_uids = trigger_observation_uids(case, scene)
    roles = relevant_object_roles(case, trigger_uids, scene)
    resolved_final_uids = []
    for object_uid in roles:
        resolved = scene.resolve_final(object_uid)
        resolved_final_uids.extend(resolved)
        if not resolved:
            versions = scene.versions_by_object.get(object_uid, [])
            for member_uid in (versions[-1].get("member_observation_uids") or []) if versions else []:
                resolved_final_uids.extend(scene.owners.get(str(member_uid), []))
    for obs_uid in trigger_uids:
        resolved_final_uids.extend(scene.owners.get(obs_uid, []))
    resolved_final_uids = list(dict.fromkeys(resolved_final_uids))
    all_object_uids = list(dict.fromkeys([*roles, *resolved_final_uids]))
    aliases = aliases_for(all_object_uids)

    observation_records = []
    observation_assets = []
    artifact_hashes_ok = True
    dynamic_gaps: list[dict[str, Any]] = []
    for index, obs_uid in enumerate(trigger_uids, 1):
        obs = scene.obs_by_uid[obs_uid]
        frame = scene.frame_by_uid.get(str(obs.get("frame_uid")), {})
        alias = f"Q{index}"
        panel = render_observation_panel(case_dir, alias, obs, scene)
        observation_assets.append(panel)
        refs = {
            "rgb_ref": artifact_summary(
                scene.run_root,
                scene.evidence_root,
                frame.get("rgb_ref") or frame.get("rgb_path"),
            ),
            "depth_ref": artifact_summary(
                scene.run_root,
                scene.evidence_root,
                frame.get("depth_ref") or frame.get("depth_path"),
            ),
            **{
                name: artifact_summary(scene.run_root, scene.evidence_root, obs.get(name))
                for name in ("raw_mask_ref", "processed_mask_ref", "pcd_ref", "image_feat_ref")
            },
        }
        artifact_hashes_ok = artifact_hashes_ok and all(
            value is None or value.get("hash_matches") for value in refs.values()
        )
        observation_records.append(
            {
                "observation_alias": alias,
                "obs_uid": obs_uid,
                "frame_uid": obs.get("frame_uid"),
                "frame_number": frame_number(obs.get("frame_uid")),
                "status": obs.get("status"),
                "class_name": obs.get("class_name"),
                "confidence": obs.get("confidence"),
                "bbox_2d": obs.get("bbox_2d"),
                "raw_mask_area": obs.get("raw_mask_area"),
                "pre_subtract_mask_area": obs.get("pre_subtract_mask_area"),
                "processed_mask_area": obs.get("processed_mask_area"),
                "removed_pixel_count": obs.get("removed_pixel_count"),
                "valid_depth_ratio": obs.get("valid_depth_ratio"),
                "depth_quantiles": obs.get("depth_quantiles"),
                "n_points": obs.get("n_points"),
                "pre_dbscan": obs.get("pre_dbscan"),
                "post_dbscan": obs.get("post_dbscan"),
                "bbox_3d_center": obs.get("bbox_3d_center"),
                "bbox_3d_extent": obs.get("bbox_3d_extent"),
                "final_owner_uids": scene.owners.get(obs_uid, []),
                "final_owner_aliases": [aliases[value] for value in scene.owners.get(obs_uid, [])],
                "source_artifacts": refs,
                "missing_source_artifacts": [name for name, value in refs.items() if value is None],
                "panel_asset": panel,
                "panel_semantics": {
                    "rgb": "exact source RGB",
                    "raw_mask": "exact raw detector mask",
                    "processed_mask": "exact mask consumed downstream",
                    "depth": "exact source depth inside the processed mask",
                    "pcd": "stored post-init/post-DBSCAN observation PCD",
                },
            }
        )

        missing_core = [name for name in ("rgb_ref", "depth_ref", "raw_mask_ref", "processed_mask_ref") if refs[name] is None]
        if missing_core:
            dynamic_gaps.append(
                {
                    "code": "TRIGGER_SOURCE_ARTIFACT_NOT_RETAINED",
                    "critical": True,
                    "message": f"{alias} 缺少系统输入 artifact：{', '.join(missing_core)}。",
                }
            )
        elif refs["pcd_ref"] is None:
            dynamic_gaps.append(
                {
                    "code": "TRIGGER_PCD_NOT_RETAINED",
                    "critical": False,
                    "message": f"{alias} 没有保存 observation PCD；这可能是无有效深度/点数不足导致的精确系统结果。",
                }
            )

    selected_views, coverage, representative_hashes_ok = selected_view_records(case_dir, scene, aliases)
    artifact_hashes_ok = artifact_hashes_ok and representative_hashes_ok
    object_records = []
    scope = case.get("scope") or {}
    scoped_versions = scope.get("association_candidate_object_versions") or {}
    explicit_version_uid = scope.get("object_version_uid")
    for object_uid in roles:
        version_uid = scoped_versions.get(object_uid)
        if explicit_version_uid and str(scene.version_by_uid.get(str(explicit_version_uid), {}).get("object_uid")) == object_uid:
            version_uid = explicit_version_uid
        if not version_uid:
            association_versions = []
            for obs_uid in trigger_uids:
                association = scene.assoc_by_obs.get(obs_uid, {})
                mapping = {
                    str(uid): value
                    for uid, value in zip(
                        association.get("object_uids_before") or [],
                        association.get("candidate_object_version_uids") or [],
                    )
                }
                if object_uid in mapping:
                    association_versions.append(mapping[object_uid])
                if str(association.get("target_object_uid")) == object_uid:
                    association_versions.append(
                        association.get("target_object_version_before")
                        or association.get("target_object_version_after")
                    )
            version_uid = next((value for value in association_versions if value), None)
        version = scene.version_by_uid.get(str(version_uid), {}) if version_uid else {}
        parents = [
            version_summary(scene.version_by_uid.get(str(value)))
            for value in version.get("parent_version_uids") or []
            if value in scene.version_by_uid
        ]
        final_uids = scene.resolve_final(object_uid)
        object_records.append(
            {
                "object_uid": object_uid,
                "object_alias": aliases[object_uid],
                "roles": roles[object_uid],
                "trigger_or_decision_version": version_summary(version),
                "parent_versions": parents,
                "final_status": "ACTIVE_FINAL" if object_uid in scene.final_by_uid else (
                    "ABSORBED_OR_REIDENTIFIED" if final_uids else "NOT_IN_ACTIVE_FINAL_MAP"
                ),
                "resolved_final_uids": final_uids,
                "resolved_final_aliases": [aliases[value] for value in final_uids],
                "representative_view_coverage": coverage.get(object_uid, {"selected": 0, "total": 0}),
            }
        )

    final_assets = render_final_objects(case_dir, resolved_final_uids, aliases, scene)
    final_records = []
    for uid in resolved_final_uids:
        summary = scene.final_summary(uid)
        summary["object_alias"] = aliases[uid]
        final_records.append(summary)

    gaps = [*CHECKER_VISUAL_GAPS.get(str(case.get("checker_id")), []), *dynamic_gaps]
    exact_final_ok = all(
        item["membership_matches_final_output"] and item["point_count_matches_final_output"]
        for item in final_records
    )
    final_outcome = {
        "status": "LINKED_ACTIVE_FINAL_OBJECTS" if final_records else "ABSENT_FROM_ACTIVE_FINAL_MAP",
        "message": (
            f"本例相关材料精确追溯到 {len(final_records)} 个 active final object。"
            if final_records
            else "已核对完整 final_membership：触发 observation 与相关对象都未形成或进入 active final object；这是最终结果，不是页面漏图。"
        ),
        "trigger_observation_final_owners": {
            obs_uid: scene.owners.get(obs_uid, []) for obs_uid in trigger_uids
        },
    }
    fidelity_status = "TRACEABLE_WITH_CRITICAL_GAP" if any(item.get("critical") for item in gaps) else "TRACEABLE"

    displayed_asset_sha256 = {
        path.relative_to(case_dir).as_posix(): sha256_file(path)
        for path in sorted(case_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }

    review = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "case_uid": case_uid,
        "finding_uid": case.get("finding_uid"),
        "checker_id": case.get("checker_id"),
        "stage": case.get("stage"),
        "subtype": case.get("subtype"),
        "source_case_json_sha256": sha256_file(case_path),
        "source_manifest_sha256": sha256_file(scene.manifest_path),
        "source_run_id": scene.manifest.get("run_id"),
        "source_git_commit": scene.manifest.get("git_commit"),
        "evidence_contract": {
            "principle": (
                "页面中的数值直接来自 checker 使用的 ledger 记录；图片是这些同一记录所引用的 RGB、depth、mask、PCD 的确定性投影；"
                "最终对象图来自 manifest 哈希锁定的最终 map pickle。代表性视图会明确写出抽样覆盖率，不冒充完整成员集合。"
            ),
            "fidelity_status": fidelity_status,
            "artifact_hashes_match": artifact_hashes_ok,
            "exact_final_map_linkage": exact_final_ok,
            "critical_gaps": gaps,
            "displayed_view_policy": "representative views with explicit selected/total coverage",
            "raw_packet_pcd_warning": (
                "原 packet 的 pcd_overlay.png 只是所选 observation 的叠加，不是最终 object；"
                "本页将它放在原始材料区，并另行展示 exact final-map object。"
            ),
        },
        "trigger_observations": observation_records,
        "association_decisions": association_records(trigger_uids, scene, aliases),
        "objects": object_records,
        "final_outcome": final_outcome,
        "final_objects": final_records,
        "representative_views": selected_views,
        "assets": {
            "trigger_observation_panels": observation_assets,
            "final_object_geometry": final_assets,
            "raw_selected_observation_overlay": "pcd_overlay.png" if (case_dir / "pcd_overlay.png").is_file() else None,
            "timeline": "timeline.jpg" if (case_dir / "timeline.jpg").is_file() else None,
        },
        "displayed_asset_sha256": displayed_asset_sha256,
    }
    write_json(case_dir / REVIEW_FILENAME, review)
    return {
        "scene_id": scene_id,
        "case_uid": case_uid,
        "checker_id": case.get("checker_id"),
        "fidelity_status": fidelity_status,
        "critical_gap_count": sum(1 for item in gaps if item.get("critical")),
        "trigger_observation_count": len(observation_records),
        "relevant_object_count": len(object_records),
        "final_object_count": len(final_records),
        "artifact_hashes_match": artifact_hashes_ok,
        "exact_final_map_linkage": exact_final_ok,
        "displayed_asset_count": len(displayed_asset_sha256),
        "review_evidence_path": (
            Path("cases") / scene_id / case_dir.name / REVIEW_FILENAME
        ).as_posix(),
        "review_evidence_sha256": sha256_file(case_dir / REVIEW_FILENAME),
    }


def build(validation_root: Path) -> dict[str, Any]:
    validation_root = validation_root.resolve()
    worklist_path = validation_root / "labels" / "r1_worklist.jsonl"
    worklist = read_jsonl(worklist_path)
    scenes: dict[str, SceneEvidence] = {}
    cases = []
    for row in worklist:
        scene_id, _ = case_key(row)
        if scene_id not in scenes:
            scenes[scene_id] = SceneEvidence(validation_root, scene_id)
        cases.append(build_case(validation_root, row, scenes[scene_id]))
    counts = Counter(item["fidelity_status"] for item in cases)
    checker_gaps = Counter(
        item["checker_id"] for item in cases if item["critical_gap_count"] > 0
    )
    status = "READY_WITH_DECLARED_LIMITATIONS" if checker_gaps else "READY"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "worklist_sha256": sha256_file(worklist_path),
        "case_count": len(cases),
        "scene_count": len(scenes),
        "fidelity_status_counts": dict(counts),
        "critical_gap_cases_by_checker": dict(checker_gaps),
        "all_artifact_hashes_match": all(item["artifact_hashes_match"] for item in cases),
        "all_available_final_objects_link_exactly": all(
            item["exact_final_map_linkage"] for item in cases
        ),
        "cases": cases,
    }
    write_json(validation_root / MANIFEST_FILENAME, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.validation_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "case_count": result["case_count"],
                "fidelity_status_counts": result["fidelity_status_counts"],
                "critical_gap_cases_by_checker": result["critical_gap_cases_by_checker"],
                "manifest": str(args.validation_root.resolve() / MANIFEST_FILENAME),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
