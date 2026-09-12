from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


QUALITY_CLEAN = "CLEAN"
QUALITY_MIXED = "MIXED"
QUALITY_UNSCORABLE = "UNSCORABLE"


@dataclass(frozen=True)
class ObservationLabelConfig:
    gt_match_distance_m: float = 0.05
    min_depth_m: float = 0.05
    max_depth_m: float = 10.0
    min_valid_depth_ratio: float = 0.70
    min_gt_support_ratio: float = 0.80
    min_matched_gt_points: int = 50
    min_core_matched_gt_points: int = 20
    min_top1_purity: float = 0.80
    max_top2_purity: float = 0.15
    min_purity_margin: float = 0.60
    mask_erosion_radius: int = 2
    max_sampled_points_per_mask: int = 2048
    query_workers: int = 4

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationLabelConfig":
        fields = set(cls.__dataclass_fields__)
        payload = {key: item for key, item in value.items() if key in fields}
        result = cls(**payload)
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
            "gt_match_distance_m",
            "min_depth_m",
            "max_depth_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")
        for name in (
            "min_valid_depth_ratio",
            "min_gt_support_ratio",
            "min_top1_purity",
            "max_top2_purity",
            "min_purity_margin",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_matched_gt_points < 1 or self.min_core_matched_gt_points < 1:
            raise ValueError("minimum point counts must be positive")
        if self.mask_erosion_radius < 0:
            raise ValueError("mask_erosion_radius cannot be negative")
        if self.max_sampled_points_per_mask < self.min_matched_gt_points:
            raise ValueError(
                "max_sampled_points_per_mask must be >= min_matched_gt_points"
            )
        if self.query_workers == 0:
            raise ValueError("query_workers cannot be zero")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GTReference:
    xyz: np.ndarray
    semantic: np.ndarray
    instance: np.ndarray
    semantic_classes: tuple[str, ...]
    instance_to_semantic: Mapping[int, int]

    @classmethod
    def load(cls, reference_npz: Path, manifest: Mapping[str, Any]) -> "GTReference":
        with np.load(reference_npz, allow_pickle=False) as archive:
            xyz = np.asarray(archive["xyz"], dtype=np.float64)
            semantic = np.asarray(archive["semantic"], dtype=np.int64)
            instance = np.asarray(archive["instance"], dtype=np.int64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("GT xyz must have shape [N, 3]")
        if len(xyz) != len(semantic) or len(xyz) != len(instance):
            raise ValueError("GT xyz, semantic and instance arrays must align")
        if not np.isfinite(xyz).all():
            raise ValueError("GT xyz contains non-finite values")
        names = tuple(str(item) for item in manifest["semantic_classes"])
        if ((semantic < 0) | (semantic > len(names))).any():
            raise ValueError("GT semantic ids are outside manifest class range")
        raw_mapping = manifest.get("instance_id_to_semantic_id")
        if raw_mapping is None:
            # Backward-compatible loading of the old unified reference.  Do not
            # consult instance_classes: structural GT instances are valid too.
            instance_to_semantic: dict[int, int] = {}
            for instance_id in np.unique(instance[(semantic > 0) & (instance >= 1000)]):
                semantic_ids = np.unique(semantic[instance == instance_id])
                if len(semantic_ids) != 1:
                    raise ValueError(
                        f"GT instance {int(instance_id)} spans semantic ids "
                        f"{semantic_ids.tolist()}"
                    )
                instance_to_semantic[int(instance_id)] = int(semantic_ids[0])
        else:
            instance_to_semantic = {
                int(instance_id): int(semantic_id)
                for instance_id, semantic_id in dict(raw_mapping).items()
            }
        for instance_id, semantic_id in instance_to_semantic.items():
            if semantic_id <= 0 or semantic_id > len(names):
                raise ValueError(
                    f"GT instance {instance_id} maps to invalid semantic id {semantic_id}"
                )
        for instance_id in np.unique(instance):
            expected_semantic = instance_to_semantic.get(int(instance_id))
            if expected_semantic is None:
                continue
            observed_semantics = np.unique(semantic[instance == instance_id])
            if not np.array_equal(
                observed_semantics, np.asarray([expected_semantic], dtype=np.int64)
            ):
                raise ValueError(
                    f"GT instance {int(instance_id)} has semantic ids "
                    f"{observed_semantics.tolist()}, expected {expected_semantic}"
                )
        return cls(
            xyz=np.ascontiguousarray(xyz),
            semantic=np.ascontiguousarray(semantic),
            instance=np.ascontiguousarray(instance),
            semantic_classes=names,
            instance_to_semantic=instance_to_semantic,
        )

    def class_name(self, semantic_id: int) -> str | None:
        if semantic_id <= 0 or semantic_id > len(self.semantic_classes):
            return None
        return self.semantic_classes[semantic_id - 1]

    def class_name_for_instance(self, instance_id: int) -> str | None:
        semantic_id = self.instance_to_semantic.get(int(instance_id))
        return self.class_name(semantic_id) if semantic_id is not None else None


def backproject_mask_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    pose_camera_to_world: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(pose_camera_to_world, dtype=np.float64)
    if mask.shape != depth.shape:
        raise ValueError("mask and depth shape mismatch")
    if intrinsics.shape not in {(3, 3), (4, 4)}:
        raise ValueError("intrinsics must have shape [3,3] or [4,4]")
    if pose.shape != (4, 4):
        raise ValueError("pose must have shape [4,4]")
    mask_area = int(mask.sum())
    valid = mask & np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    flat_indices = np.flatnonzero(valid)
    valid_count = int(len(flat_indices))
    if valid_count > max_points:
        selected = np.linspace(0, valid_count - 1, max_points, dtype=np.int64)
        flat_indices = flat_indices[selected]
    rows, cols = np.unravel_index(flat_indices, depth.shape)
    z = depth[rows, cols]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("camera focal lengths must be non-zero")
    camera = np.column_stack(
        ((cols.astype(np.float64) - cx) * z / fx, (rows.astype(np.float64) - cy) * z / fy, z)
    )
    if len(camera):
        world = camera @ pose[:3, :3].T + pose[:3, 3]
    else:
        world = np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray(world), {
        "mask_area": mask_area,
        "valid_depth_pixels": valid_count,
        "valid_depth_ratio": valid_count / mask_area if mask_area else 0.0,
        "sampled_valid_depth_points": int(len(world)),
        "sampling_applied": valid_count > max_points,
    }


class ObservationGTLabeler:
    def __init__(self, reference: GTReference, config: ObservationLabelConfig) -> None:
        config.validate()
        self.reference = reference
        self.config = config
        self.tree = cKDTree(reference.xyz)

    def _query(self, points: np.ndarray) -> dict[str, Any]:
        if not len(points):
            return {
                "sampled_points": 0,
                "matched_any_points": 0,
                "matched_gt_points": 0,
                "unlabeled_gt_points": 0,
                "gt_support_ratio": 0.0,
                "instance_support_ratio": 0.0,
                "unlabeled_gt_ratio": 0.0,
                "top1_gt_instance": None,
                "top1_gt_class": None,
                "top1_purity": 0.0,
                "top2_gt_instance": None,
                "top2_purity": 0.0,
                "purity_margin": 0.0,
                "median_gt_distance_m": None,
                "p95_gt_distance_m": None,
                "gt_instance_counts": {},
            }
        distance, index = self.tree.query(
            points, k=1, workers=self.config.query_workers
        )
        within = distance < self.config.gt_match_distance_m
        matched_semantic = self.reference.semantic[index]
        matched_instance = self.reference.instance[index]
        # Validity is declared by the reference mapping, not by an ID threshold.
        # This supports native Replica object IDs including 0 and preserves
        # wall/ceiling/floor/other as ordinary instances.
        valid_instance_ids = np.fromiter(
            self.reference.instance_to_semantic, dtype=np.int64
        )
        eligible = within & np.isin(matched_instance, valid_instance_ids)
        unlabeled = within & ~eligible
        eligible_instances = matched_instance[eligible]
        unique, counts = np.unique(eligible_instances, return_counts=True)
        ranking = sorted(
            ((int(count), int(gt_id)) for gt_id, count in zip(unique, counts)),
            key=lambda item: (-item[0], item[1]),
        )
        matched_count = int(eligible.sum())
        unlabeled_count = int(unlabeled.sum())
        matched_any = int(within.sum())
        top1_count, top1_id = ranking[0] if ranking else (0, None)
        top2_count, top2_id = ranking[1] if len(ranking) > 1 else (0, None)
        top1_purity = top1_count / matched_any if matched_any else 0.0
        top2_purity = top2_count / matched_any if matched_any else 0.0
        accepted_distances = distance[within]
        top1_class = (
            self.reference.class_name_for_instance(int(top1_id))
            if top1_id is not None
            else None
        )
        return {
            "sampled_points": int(len(points)),
            "matched_any_points": matched_any,
            "matched_gt_points": matched_count,
            "unlabeled_gt_points": unlabeled_count,
            "gt_support_ratio": matched_any / len(points),
            "instance_support_ratio": matched_count / len(points),
            "unlabeled_gt_ratio": (
                unlabeled_count / matched_any if matched_any else 0.0
            ),
            "top1_gt_instance": top1_id,
            "top1_gt_class": top1_class,
            "top1_purity": top1_purity,
            "top2_gt_instance": top2_id,
            "top2_purity": top2_purity,
            "purity_margin": top1_purity - top2_purity,
            "median_gt_distance_m": (
                float(np.median(accepted_distances)) if len(accepted_distances) else None
            ),
            "p95_gt_distance_m": (
                float(np.quantile(accepted_distances, 0.95))
                if len(accepted_distances)
                else None
            ),
            "gt_instance_counts": {
                str(gt_id): count for count, gt_id in ranking
            },
        }

    def label_mask(
        self,
        *,
        obs_uid: str,
        frame_uid: str,
        class_name: str | None,
        mask: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        pose_camera_to_world: np.ndarray,
    ) -> dict[str, Any]:
        points, geometry = backproject_mask_points(
            mask,
            depth_m,
            intrinsics,
            pose_camera_to_world,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
            max_points=self.config.max_sampled_points_per_mask,
        )
        full = self._query(points)
        radius = self.config.mask_erosion_radius
        core_mask = (
            binary_erosion(np.asarray(mask, dtype=bool), iterations=radius)
            if radius > 0
            else np.asarray(mask, dtype=bool)
        )
        core_points, core_geometry = backproject_mask_points(
            core_mask,
            depth_m,
            intrinsics,
            pose_camera_to_world,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
            max_points=self.config.max_sampled_points_per_mask,
        )
        core = self._query(core_points)
        core_agrees = (
            full["top1_gt_instance"] is not None
            and full["top1_gt_instance"] == core["top1_gt_instance"]
        )
        status, reason = self._classify(
            geometry=geometry,
            full=full,
            core=core,
            core_agrees=core_agrees,
        )
        assigned_gt = full["top1_gt_instance"] if status == QUALITY_CLEAN else None
        result = {
            "obs_uid": str(obs_uid),
            "frame_uid": str(frame_uid),
            "class_name": class_name,
            **geometry,
            **full,
            "core_mask_area": core_geometry["mask_area"],
            "core_valid_depth_pixels": core_geometry["valid_depth_pixels"],
            "core_matched_gt_points": core["matched_gt_points"],
            "core_top1_gt_instance": core["top1_gt_instance"],
            "core_top1_purity": core["top1_purity"],
            "core_agrees_with_full_mask": bool(core_agrees),
            "assigned_gt_instance": assigned_gt,
            "quality_status": status,
            "quality_reason": reason,
        }
        return result

    def _classify(
        self,
        *,
        geometry: Mapping[str, Any],
        full: Mapping[str, Any],
        core: Mapping[str, Any],
        core_agrees: bool,
    ) -> tuple[str, str]:
        if geometry["mask_area"] == 0:
            return QUALITY_UNSCORABLE, "empty_mask"
        if geometry["valid_depth_ratio"] < self.config.min_valid_depth_ratio:
            return QUALITY_UNSCORABLE, "insufficient_valid_depth_ratio"
        if (
            full["matched_gt_points"] < self.config.min_matched_gt_points
            and full["unlabeled_gt_points"] >= self.config.min_matched_gt_points
        ):
            return QUALITY_UNSCORABLE, "unlabeled_without_instance_identity"
        if full["matched_gt_points"] < self.config.min_matched_gt_points:
            return QUALITY_UNSCORABLE, "insufficient_matched_gt_points"
        if full["gt_support_ratio"] < self.config.min_gt_support_ratio:
            return QUALITY_UNSCORABLE, "insufficient_gt_support_ratio"
        clean_conditions = (
            full["top1_purity"] >= self.config.min_top1_purity
            and full["top2_purity"] <= self.config.max_top2_purity
            and full["purity_margin"] >= self.config.min_purity_margin
            and core["matched_gt_points"] >= self.config.min_core_matched_gt_points
            and core_agrees
        )
        if clean_conditions:
            return QUALITY_CLEAN, "dominant_gt_stable"
        if core["matched_gt_points"] < self.config.min_core_matched_gt_points:
            return QUALITY_MIXED, "insufficient_core_support"
        if not core_agrees:
            return QUALITY_MIXED, "full_core_gt_disagreement"
        return QUALITY_MIXED, "insufficient_instance_purity"


def aggregate_projection_diagnostics(labels: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(labels)
    distances = [
        float(row["median_gt_distance_m"])
        for row in rows
        if row.get("median_gt_distance_m") is not None
    ]
    matched = sum(int(row.get("matched_any_points", 0)) for row in rows)
    sampled = sum(int(row.get("sampled_points", 0)) for row in rows)
    return {
        "observation_count": len(rows),
        "sampled_point_count": sampled,
        "matched_within_threshold_count": matched,
        "matched_within_threshold_ratio": matched / sampled if sampled else None,
        "observation_median_distance_m_median": (
            float(np.median(distances)) if distances else None
        ),
        "observation_median_distance_m_p90": (
            float(np.quantile(distances, 0.90)) if distances else None
        ),
        "observation_median_distance_m_p95": (
            float(np.quantile(distances, 0.95)) if distances else None
        ),
    }
