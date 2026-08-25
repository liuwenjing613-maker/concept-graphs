"""Leakage-resistant multi-view projection evidence for identity repair.

CMVIC scores causal state geometry available before each held-out frame. It
never regenerates masks and canonicalizes partitions without mapper entity UUIDs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment


CMVIC_STATISTIC_NAME = "COUNTERFACTUAL_MULTI_VIEW_INSTANCE_CONSISTENCY"
CAUSAL_GEOMETRY_POLICY = (
    "LEAVE_ONE_VERIFICATION_FRAME_OUT_REPLAY_THROUGH_PREVIOUS_FRAME"
)
OBSERVED_MASK_POLICY = "ALL_STATE_UNION_ROI_INTERSECTION_WITH_FROZEN_PROCESSED_MASKS"
POSE_CONVENTION = "FRAME_POSE_IS_CAMERA_TO_WORLD_USE_INVERSE_FOR_WORLD_TO_CAMERA"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _uid(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return prefix + digest[:20]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_index_from_uid(frame_uid: str) -> int:
    try:
        return int(str(frame_uid).rsplit("_f", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot derive frame index from {frame_uid!r}") from exc


def canonical_partition_uid(member_obs_uids: Iterable[str]) -> str:
    members = tuple(sorted(set(str(item) for item in member_obs_uids)))
    if not members:
        raise ValueError("an instance partition must contain an observation")
    return _uid("instance_partition_", members)


@dataclass(frozen=True)
class InstanceGeometry:
    canonical_partition_uid: str
    member_obs_uids: tuple[str, ...]
    points: np.ndarray
    colors: np.ndarray | None
    source_state_hash: str

    @classmethod
    def build(
        cls,
        *,
        member_obs_uids: Iterable[str],
        points: Any,
        colors: Any | None = None,
        source_state_hash: str,
    ) -> "InstanceGeometry":
        members = tuple(sorted(set(str(item) for item in member_obs_uids)))
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim != 2 or point_array.shape[1] != 3:
            raise ValueError("instance points must have shape [N, 3]")
        if not np.isfinite(point_array).all():
            raise ValueError("instance points must be finite")
        color_array = None
        if colors is not None:
            color_array = np.asarray(colors, dtype=np.float64)
            if color_array.shape != point_array.shape:
                raise ValueError("instance colors must align with points")
        return cls(
            canonical_partition_uid=canonical_partition_uid(members),
            member_obs_uids=members,
            points=np.ascontiguousarray(point_array),
            colors=(
                np.ascontiguousarray(color_array) if color_array is not None else None
            ),
            source_state_hash=str(source_state_hash),
        )

    def audit_dict(self) -> dict[str, Any]:
        points32 = np.ascontiguousarray(self.points, dtype=np.float32)
        return {
            "canonical_partition_uid": self.canonical_partition_uid,
            "member_obs_uids": list(self.member_obs_uids),
            "point_count": int(len(self.points)),
            "points_sha256": hashlib.sha256(points32.tobytes()).hexdigest(),
            "source_state_hash": self.source_state_hash,
        }


def extract_affected_instance_geometries(
    *,
    raw_objects: Sequence[Mapping[str, Any]],
    affected_obs_uids: Iterable[str],
    source_state_hash: str,
) -> tuple[InstanceGeometry, ...]:
    affected = set(str(item) for item in affected_obs_uids)
    geometries = []
    seen = set()
    for obj in raw_objects:
        members = tuple(sorted(set(str(item) for item in obj.get("obs_uids", ()))))
        if not members or not (set(members) & affected):
            continue
        pcd = obj.get("pcd")
        if pcd is None or not hasattr(pcd, "points"):
            raise ValueError("raw replay object is missing point-cloud geometry")
        colors = (
            np.asarray(pcd.colors, dtype=np.float64)
            if hasattr(pcd, "colors") and len(pcd.colors)
            else None
        )
        geometry = InstanceGeometry.build(
            member_obs_uids=members,
            points=np.asarray(pcd.points, dtype=np.float64),
            colors=colors,
            source_state_hash=source_state_hash,
        )
        if geometry.canonical_partition_uid in seen:
            raise ValueError("duplicate canonical instance partition")
        seen.add(geometry.canonical_partition_uid)
        geometries.append(geometry)
    return tuple(sorted(geometries, key=lambda item: item.canonical_partition_uid))


def freeze_instance_geometries(
    *,
    output_dir: Path,
    state_uid: str,
    geometries: Sequence[InstanceGeometry],
) -> tuple[dict[str, Any], ...]:
    state_dir = output_dir.resolve() / str(state_uid)
    state_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for geometry in sorted(geometries, key=lambda item: item.canonical_partition_uid):
        path = state_dir / f"{geometry.canonical_partition_uid}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite geometry artifact: {path}")
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            payload = {
                "points": np.asarray(geometry.points, dtype=np.float32),
                "member_obs_uids": np.asarray(geometry.member_obs_uids, dtype=np.str_),
                "source_state_hash": np.asarray(geometry.source_state_hash),
                "canonical_partition_uid": np.asarray(geometry.canonical_partition_uid),
            }
            if geometry.colors is not None:
                payload["colors"] = np.asarray(geometry.colors, dtype=np.float32)
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
        row = geometry.audit_dict()
        row.update(
            {
                "artifact_path": str(path),
                "artifact_sha256": sha256_file(path),
                "state_uid": str(state_uid),
            }
        )
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class ProjectionFrameEvidence:
    frame_uid: str
    frame_index: int
    rgb_ref: dict[str, Any]
    depth_ref: dict[str, Any]
    pose: np.ndarray
    intrinsics: np.ndarray
    processed_mask_refs: tuple[dict[str, Any], ...]
    evidence_hash: str
    depth_scale_to_meters: float
    rgb_path: Path
    depth_path: Path
    depth_m: np.ndarray
    observed_masks: tuple[np.ndarray, ...]
    observed_mask_uids: tuple[str, ...]

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.depth_m.shape)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "frame_uid": self.frame_uid,
            "frame_index": self.frame_index,
            "rgb_ref": dict(self.rgb_ref),
            "depth_ref": dict(self.depth_ref),
            "pose": self.pose.tolist(),
            "intrinsics": self.intrinsics.tolist(),
            "processed_mask_refs": [dict(item) for item in self.processed_mask_refs],
            "observed_mask_uids": list(self.observed_mask_uids),
            "observed_mask_count": len(self.observed_masks),
            "image_shape": list(self.image_shape),
            "depth_scale_to_meters": self.depth_scale_to_meters,
            "pose_convention": POSE_CONVENTION,
            "evidence_hash": self.evidence_hash,
        }


class ProjectionEvidenceLoader:
    """Load immutable RGB-D frames and mapper-processed masks."""

    def __init__(
        self,
        base_run: Path,
        *,
        depth_scale_to_meters: float | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.base_run = base_run.resolve()
        self.verify_hashes = bool(verify_hashes)
        with (self.base_run / "config_params.json").open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.depth_scale_to_meters = float(
            depth_scale_to_meters or self._load_depth_scale()
        )
        if not math.isfinite(self.depth_scale_to_meters):
            raise ValueError("depth scale must be finite")
        if self.depth_scale_to_meters <= 0.0:
            raise ValueError("depth scale must be positive")
        self.frames = {}
        with (self.base_run / "evidence/frames.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self.frames[str(row["frame_uid"])] = row
        self.observations_by_frame: dict[str, list[dict[str, Any]]] = {}
        with (self.base_run / "evidence/observations.jsonl").open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("processed_mask_ref"):
                    continue
                self.observations_by_frame.setdefault(str(row["frame_uid"]), []).append(
                    row
                )
        for rows in self.observations_by_frame.values():
            rows.sort(
                key=lambda row: (
                    int(row.get("filtered_det_idx", 0)),
                    str(row["obs_uid"]),
                )
            )

    def _load_depth_scale(self) -> float:
        import yaml

        configured = Path(str(self.config.get("dataset_config") or ""))
        candidates = [configured]
        if not configured.is_file():
            dataconfig_root = (
                Path(__file__).resolve().parents[1] / "dataset/dataconfigs"
            )
            candidates.extend(dataconfig_root.rglob(configured.name))
        config_path = next((path for path in candidates if path.is_file()), None)
        if config_path is None:
            raise FileNotFoundError("cannot resolve dataset config for depth scale")
        with config_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        return float(value["camera_params"]["png_depth_scale"])

    def _resolve(self, ref: Mapping[str, Any]) -> Path:
        path = (self.base_run / str(ref["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = ref.get("sha256")
        if self.verify_hashes and expected and sha256_file(path) != str(expected):
            raise ValueError(f"evidence hash drift: {path}")
        return path

    def load_frame(self, frame_uid: str) -> ProjectionFrameEvidence:
        row = self.frames[str(frame_uid)]
        rgb_ref = dict(row["rgb_ref"])
        depth_ref = dict(row["depth_ref"])
        rgb_path = self._resolve(rgb_ref)
        depth_path = self._resolve(depth_ref)
        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float64)
        depth_m = np.ascontiguousarray(
            depth_raw / self.depth_scale_to_meters,
            dtype=np.float64,
        )
        mask_refs = []
        mask_uids = []
        masks = []
        for observation in self.observations_by_frame.get(str(frame_uid), ()):
            ref = dict(observation["processed_mask_ref"])
            path = self._resolve(ref)
            with np.load(path, allow_pickle=False) as archive:
                key = str(ref.get("key") or "mask")
                if key not in archive:
                    raise KeyError(f"{path} is missing processed mask key {key}")
                mask = np.asarray(archive[key], dtype=bool)
            if mask.shape != depth_m.shape:
                raise ValueError("processed mask and depth shape mismatch")
            mask_refs.append(ref)
            mask_uids.append(str(observation["obs_uid"]))
            masks.append(np.ascontiguousarray(mask))
        pose = np.asarray(row["pose"], dtype=np.float64)
        intrinsics = np.asarray(row["intrinsics"], dtype=np.float64)
        if pose.shape != (4, 4) or intrinsics.shape not in {(3, 3), (4, 4)}:
            raise ValueError("pose and intrinsics must be square camera matrices")
        evidence_payload = {
            "frame_uid": str(frame_uid),
            "rgb_sha256": rgb_ref.get("sha256") or sha256_file(rgb_path),
            "depth_sha256": depth_ref.get("sha256") or sha256_file(depth_path),
            "processed_masks": [
                {
                    "obs_uid": uid,
                    "sha256": ref.get("sha256") or sha256_file(self._resolve(ref)),
                }
                for uid, ref in zip(mask_uids, mask_refs)
            ],
            "pose": pose.tolist(),
            "intrinsics": intrinsics.tolist(),
            "depth_scale_to_meters": self.depth_scale_to_meters,
            "pose_convention": POSE_CONVENTION,
        }
        return ProjectionFrameEvidence(
            frame_uid=str(frame_uid),
            frame_index=frame_index_from_uid(str(frame_uid)),
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
            pose=pose,
            intrinsics=intrinsics,
            processed_mask_refs=tuple(mask_refs),
            evidence_hash=_uid("projection_frame_evidence_", evidence_payload),
            depth_scale_to_meters=self.depth_scale_to_meters,
            rgb_path=rgb_path,
            depth_path=depth_path,
            depth_m=depth_m,
            observed_masks=tuple(masks),
            observed_mask_uids=tuple(mask_uids),
        )


@dataclass(frozen=True)
class ProjectedInstance:
    state_uid: str
    canonical_partition_uid: str
    member_obs_uids: tuple[str, ...]
    mask: np.ndarray
    visible_point_count: int
    total_projected_point_count: int
    depth_compatible_pixel_count: int
    source_state_hash: str

    def audit_dict(self) -> dict[str, Any]:
        return {
            "state_uid": self.state_uid,
            "canonical_partition_uid": self.canonical_partition_uid,
            "member_obs_uids": list(self.member_obs_uids),
            "visible_point_count": self.visible_point_count,
            "total_projected_point_count": self.total_projected_point_count,
            "depth_compatible_pixel_count": self.depth_compatible_pixel_count,
            "projected_area": int(self.mask.sum()),
            "mask_sha256": hashlib.sha256(
                np.ascontiguousarray(self.mask, dtype=np.uint8).tobytes()
            ).hexdigest(),
            "source_state_hash": self.source_state_hash,
        }


@dataclass(frozen=True)
class FrameConsistencyResult:
    frame_uid: str
    state_uid: str
    score: float
    matching: tuple[dict[str, Any], ...]
    projected_instance_count: int
    observed_mask_count: int
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_uid": self.frame_uid,
            "state_uid": self.state_uid,
            "score": self.score,
            "matching": [dict(item) for item in self.matching],
            "projected_instance_count": self.projected_instance_count,
            "observed_mask_count": self.observed_mask_count,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class CMVICResult:
    state_uid: str
    score: float
    frame_results: tuple[FrameConsistencyResult, ...]
    observable: bool
    projected_difference_pixel_count: int
    score_uid: str
    evidence_policy_uid: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_uid": self.state_uid,
            "score": self.score,
            "frame_results": [item.as_dict() for item in self.frame_results],
            "observable": self.observable,
            "projected_difference_pixel_count": (self.projected_difference_pixel_count),
            "score_uid": self.score_uid,
            "evidence_policy_uid": self.evidence_policy_uid,
        }


@dataclass(frozen=True)
class CMVICComparison:
    noop: CMVICResult
    candidate: CMVICResult
    advantage_over_noop: float
    observable: bool
    disposition: str
    evidence_selection_audit: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_statistic": CMVIC_STATISTIC_NAME,
            "evidence_policy_uid": self.noop.evidence_policy_uid,
            "noop": self.noop.as_dict(),
            "candidate": self.candidate.as_dict(),
            "advantage_over_noop": self.advantage_over_noop,
            "observable": self.observable,
            "disposition": self.disposition,
            "evidence_selection_audit": [
                dict(item) for item in self.evidence_selection_audit
            ],
        }


class CounterfactualProjectionVerifier:
    """Project causal instance geometry and compute one continuous CMVIC score."""

    statistic_name = CMVIC_STATISTIC_NAME

    def __init__(
        self,
        *,
        voxel_size: float,
        depth_tolerance: float | None = None,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.depth_tolerance = float(
            2.0 * self.voxel_size if depth_tolerance is None else depth_tolerance
        )
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("voxel size must be finite and positive")
        if not math.isfinite(self.depth_tolerance) or self.depth_tolerance <= 0.0:
            raise ValueError("depth tolerance must be finite and positive")
        self._disk_offsets: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _offsets(self, radius: int) -> tuple[np.ndarray, np.ndarray]:
        if radius not in self._disk_offsets:
            values = np.arange(-radius, radius + 1)
            dy, dx = np.meshgrid(values, values, indexing="ij")
            keep = dy * dy + dx * dx <= radius * radius
            self._disk_offsets[radius] = (dy[keep], dx[keep])
        return self._disk_offsets[radius]

    def project_state(
        self,
        *,
        state_uid: str,
        instances: Sequence[InstanceGeometry],
        frame: ProjectionFrameEvidence,
    ) -> tuple[ProjectedInstance, ...]:
        height, width = frame.image_shape
        world_to_camera = np.linalg.inv(frame.pose)
        fx = float(frame.intrinsics[0, 0])
        fy = float(frame.intrinsics[1, 1])
        cx = float(frame.intrinsics[0, 2])
        cy = float(frame.intrinsics[1, 2])
        results = []
        for instance in sorted(
            instances, key=lambda item: item.canonical_partition_uid
        ):
            points = instance.points
            homogeneous = np.concatenate(
                [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
            )
            camera = homogeneous @ world_to_camera.T
            z = camera[:, 2]
            safe_z = np.where(z == 0.0, 1.0, z)
            u = np.rint(fx * camera[:, 0] / safe_z + cx).astype(np.int64)
            v = np.rint(fy * camera[:, 1] / safe_z + cy).astype(np.int64)
            projected = (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            z_buffer = np.full((height, width), np.inf, dtype=np.float64)
            visible_points = 0
            write_count = 0
            for point_index in np.flatnonzero(projected):
                radius = max(
                    1,
                    int(math.ceil(fx * self.voxel_size / float(z[point_index]))),
                )
                dy, dx = self._offsets(radius)
                yy = v[point_index] + dy
                xx = u[point_index] + dx
                in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
                if not in_bounds.any():
                    continue
                yy = yy[in_bounds]
                xx = xx[in_bounds]
                observed_depth = frame.depth_m[yy, xx]
                compatible = (
                    np.isfinite(observed_depth)
                    & (observed_depth > 0.0)
                    & (np.abs(observed_depth - z[point_index]) <= self.depth_tolerance)
                )
                if not compatible.any():
                    continue
                yy = yy[compatible]
                xx = xx[compatible]
                nearer = z[point_index] < z_buffer[yy, xx]
                if nearer.any():
                    z_buffer[yy[nearer], xx[nearer]] = z[point_index]
                    write_count += int(nearer.sum())
                visible_points += 1
            mask = np.isfinite(z_buffer)
            results.append(
                ProjectedInstance(
                    state_uid=str(state_uid),
                    canonical_partition_uid=instance.canonical_partition_uid,
                    member_obs_uids=instance.member_obs_uids,
                    mask=np.ascontiguousarray(mask),
                    visible_point_count=visible_points,
                    total_projected_point_count=int(projected.sum()),
                    depth_compatible_pixel_count=write_count,
                    source_state_hash=instance.source_state_hash,
                )
            )
        return tuple(results)

    @staticmethod
    def select_common_observed_masks(
        *,
        frame: ProjectionFrameEvidence,
        projected_by_state: Mapping[str, Sequence[ProjectedInstance]],
    ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], np.ndarray]:
        roi = np.zeros(frame.image_shape, dtype=bool)
        for projected in projected_by_state.values():
            for instance in projected:
                roi |= instance.mask
        selected_masks = []
        selected_uids = []
        if roi.any():
            for uid, mask in zip(frame.observed_mask_uids, frame.observed_masks):
                if np.logical_and(mask, roi).any():
                    selected_masks.append(mask)
                    selected_uids.append(uid)
        return tuple(selected_masks), tuple(selected_uids), roi

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> float:
        intersection = int(np.logical_and(first, second).sum())
        union = int(np.logical_or(first, second).sum())
        return float(intersection / union) if union else 0.0

    def score_frame(
        self,
        *,
        frame: ProjectionFrameEvidence,
        state_uid: str,
        projected: Sequence[ProjectedInstance],
        observed_masks: Sequence[np.ndarray],
        observed_mask_uids: Sequence[str],
    ) -> FrameConsistencyResult:
        visible = tuple(item for item in projected if item.mask.any())
        matrix = np.zeros((len(visible), len(observed_masks)), dtype=np.float64)
        for row_index, instance in enumerate(visible):
            for column_index, mask in enumerate(observed_masks):
                matrix[row_index, column_index] = self._iou(instance.mask, mask)
        matching = []
        matched_sum = 0.0
        if matrix.size:
            row_indices, column_indices = linear_sum_assignment(-matrix)
            for row_index, column_index in zip(row_indices, column_indices):
                iou = float(matrix[row_index, column_index])
                matched_sum += iou
                matching.append(
                    {
                        "canonical_partition_uid": visible[
                            int(row_index)
                        ].canonical_partition_uid,
                        "observed_mask_uid": str(observed_mask_uids[int(column_index)]),
                        "iou": iou,
                    }
                )
        denominator = max(len(visible), len(observed_masks))
        score = matched_sum / denominator if denominator else 0.0
        diagnostics = {
            "hungarian_iou_sum": matched_sum,
            "normalization_denominator": denominator,
            "unmatched_projected_instance_count": (len(visible) - len(matching)),
            "unmatched_observed_mask_count": (len(observed_masks) - len(matching)),
            "visible_projected_area": int(
                sum(int(item.mask.sum()) for item in visible)
            ),
            "visible_point_count": int(
                sum(item.visible_point_count for item in visible)
            ),
            "semantic_threshold_count": 0,
        }
        return FrameConsistencyResult(
            frame_uid=frame.frame_uid,
            state_uid=str(state_uid),
            score=float(score),
            matching=tuple(matching),
            projected_instance_count=len(visible),
            observed_mask_count=len(observed_masks),
            diagnostics=diagnostics,
        )

    @staticmethod
    def partition_difference_pixel_count(
        first: Sequence[ProjectedInstance],
        second: Sequence[ProjectedInstance],
    ) -> int:
        first_masks = tuple(item.mask for item in first if item.mask.any())
        second_masks = tuple(item.mask for item in second if item.mask.any())
        if not first_masks and not second_masks:
            return 0
        if not first_masks:
            return int(sum(mask.sum() for mask in second_masks))
        if not second_masks:
            return int(sum(mask.sum() for mask in first_masks))
        cost = np.zeros((len(first_masks), len(second_masks)), dtype=np.float64)
        for row_index, first_mask in enumerate(first_masks):
            for column_index, second_mask in enumerate(second_masks):
                cost[row_index, column_index] = np.logical_xor(
                    first_mask, second_mask
                ).sum()
        rows, columns = linear_sum_assignment(cost)
        difference = int(cost[rows, columns].sum())
        matched_rows = set(int(item) for item in rows)
        matched_columns = set(int(item) for item in columns)
        difference += sum(
            int(mask.sum())
            for index, mask in enumerate(first_masks)
            if index not in matched_rows
        )
        difference += sum(
            int(mask.sum())
            for index, mask in enumerate(second_masks)
            if index not in matched_columns
        )
        return int(difference)

    def compare(
        self,
        *,
        noop_state_uid: str,
        candidate_state_uid: str,
        frames: Sequence[ProjectionFrameEvidence],
        projected_by_frame: Mapping[str, Mapping[str, Sequence[ProjectedInstance]]],
    ) -> CMVICComparison:
        ordered_frames = tuple(
            sorted(frames, key=lambda item: (item.frame_index, item.frame_uid))
        )
        noop_results = []
        candidate_results = []
        difference_pixels = 0
        selection_audit = []
        for frame in ordered_frames:
            states = projected_by_frame[frame.frame_uid]
            noop_projected = tuple(states[str(noop_state_uid)])
            candidate_projected = tuple(states[str(candidate_state_uid)])
            observed_masks, observed_uids, roi = self.select_common_observed_masks(
                frame=frame,
                projected_by_state={
                    str(noop_state_uid): noop_projected,
                    str(candidate_state_uid): candidate_projected,
                },
            )
            noop_results.append(
                self.score_frame(
                    frame=frame,
                    state_uid=noop_state_uid,
                    projected=noop_projected,
                    observed_masks=observed_masks,
                    observed_mask_uids=observed_uids,
                )
            )
            candidate_results.append(
                self.score_frame(
                    frame=frame,
                    state_uid=candidate_state_uid,
                    projected=candidate_projected,
                    observed_masks=observed_masks,
                    observed_mask_uids=observed_uids,
                )
            )
            difference_pixels += self.partition_difference_pixel_count(
                noop_projected, candidate_projected
            )
            selection_audit.append(
                {
                    "frame_uid": frame.frame_uid,
                    "frame_evidence_hash": frame.evidence_hash,
                    "shared_roi_area": int(roi.sum()),
                    "selected_observed_mask_uids": list(observed_uids),
                }
            )
        policy_payload = {
            "statistic": self.statistic_name,
            "causal_geometry_policy": CAUSAL_GEOMETRY_POLICY,
            "observed_mask_policy": OBSERVED_MASK_POLICY,
            "pose_convention": POSE_CONVENTION,
            "voxel_size": self.voxel_size,
            "depth_tolerance": self.depth_tolerance,
            "normalization": ("HUNGARIAN_IOU_SUM_DIVIDED_BY_MAX_INSTANCE_COUNTS"),
            "observability": ("EXACT_ZERO_UNORDERED_VISIBLE_PARTITION_DIFFERENCE"),
        }
        evidence_policy_uid = _uid("cmvic_evidence_policy_", policy_payload)
        noop_score = (
            sum(item.score for item in noop_results) / len(noop_results)
            if noop_results
            else 0.0
        )
        candidate_score = (
            sum(item.score for item in candidate_results) / len(candidate_results)
            if candidate_results
            else 0.0
        )
        observable = difference_pixels != 0

        def result(
            state_uid: str,
            score: float,
            frame_results: Sequence[FrameConsistencyResult],
        ) -> CMVICResult:
            score_payload = {
                "score": score,
                "frame_results": [item.as_dict() for item in frame_results],
                "evidence_policy_uid": evidence_policy_uid,
                "projected_difference_pixel_count": difference_pixels,
                "observable": observable,
                "evidence_selection_audit": selection_audit,
            }
            return CMVICResult(
                state_uid=str(state_uid),
                score=float(score),
                frame_results=tuple(frame_results),
                observable=observable,
                projected_difference_pixel_count=difference_pixels,
                score_uid=_uid("cmvic_score_", score_payload),
                evidence_policy_uid=evidence_policy_uid,
            )

        noop = result(noop_state_uid, noop_score, noop_results)
        candidate = result(candidate_state_uid, candidate_score, candidate_results)
        return CMVICComparison(
            noop=noop,
            candidate=candidate,
            advantage_over_noop=float(candidate.score - noop.score),
            observable=observable,
            disposition=(
                "COUNTERFACTUAL_OBSERVABLE"
                if observable
                else "COUNTERFACTUAL_UNOBSERVABLE"
            ),
            evidence_selection_audit=tuple(selection_audit),
        )


def _contour(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    return np.logical_and(mask, np.logical_not(binary_erosion(mask)))


def _deterministic_color(seed: str, partition_uid: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(f"{seed}:{partition_uid}".encode("utf-8")).digest()
    return tuple(64 + int(value) % 160 for value in digest[:3])


def render_projection_overlay(
    *,
    frame: ProjectionFrameEvidence,
    projected: Sequence[ProjectedInstance],
    observed_masks: Sequence[np.ndarray],
    output_path: Path,
    color_seed: str,
) -> dict[str, Any]:
    """Render an action-blind state overlay with deterministic group colors."""

    image = Image.open(frame.rgb_path).convert("RGB")
    canvas = np.asarray(image, dtype=np.uint8).copy()
    for mask in observed_masks:
        edge = _contour(mask)
        canvas[edge] = np.asarray([255, 255, 255], dtype=np.uint8)
    labels = []
    visible = sorted(
        (item for item in projected if item.mask.any()),
        key=lambda item: item.canonical_partition_uid,
    )
    for index, instance in enumerate(visible, 1):
        color = np.asarray(
            _deterministic_color(color_seed, instance.canonical_partition_uid),
            dtype=np.uint8,
        )
        edge = _contour(instance.mask)
        canvas[edge] = color
        yy, xx = np.nonzero(instance.mask)
        if len(xx):
            labels.append(
                (
                    int(np.median(xx)),
                    int(np.median(yy)),
                    f"GROUP_{index:02d}",
                    tuple(int(value) for value in color),
                )
            )
    rendered = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    for x, y, label, color in labels:
        draw.rectangle((x - 2, y - 11, x + 70, y + 3), fill=(0, 0, 0))
        draw.text((x, y - 10), label, fill=color)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite overlay: {output_path}")
    rendered.save(output_path, format="PNG")
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "frame_uid": frame.frame_uid,
        "group_count": len(visible),
        "action_semantics_present": False,
        "color_seed": str(color_seed),
    }
