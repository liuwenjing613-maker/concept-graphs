from __future__ import annotations

import copy
import gzip
import hashlib
import pickle
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .geometry import ObservationGeometryContract
from .index import ProvenanceIndex


class MaterializationError(RuntimeError):
    """Raised rather than silently approximating a missing replay payload."""


def _frame_index(frame_uid: str) -> int:
    try:
        return int(frame_uid.rsplit("_f", 1)[-1])
    except ValueError as exc:
        raise MaterializationError(
            f"cannot parse frame index from {frame_uid}"
        ) from exc


def _uuid(value: str | uuid.UUID | None, obs_uid: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return uuid.uuid5(uuid.NAMESPACE_URL, value)
    return uuid.uuid5(uuid.NAMESPACE_URL, "conceptgraphs-observation:" + obs_uid)


class ObservationMaterializer:
    def __init__(
        self, provenance: ProvenanceIndex, cfg: dict[str, Any] | None = None
    ) -> None:
        self.provenance = provenance
        self.cfg = dict(cfg or {})
        self.frames = {}
        frames_path = provenance.evidence_root / "frames.jsonl"
        if frames_path.is_file():
            import json

            with frames_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.frames[row["frame_uid"]] = row

    def resolve_ref(self, ref: dict[str, Any] | None) -> Path:
        if not ref or not isinstance(ref.get("path"), str):
            raise MaterializationError("required artifact reference is absent")
        path = Path(ref["path"])
        if not path.is_absolute():
            path = self.provenance.experiment_root / path
        path = path.resolve()
        if not path.is_file():
            raise MaterializationError(f"referenced artifact does not exist: {path}")
        return path

    def load_ref(self, ref: dict[str, Any]) -> Any:
        path = self.resolve_ref(ref)
        fmt = str(ref.get("format") or path.suffix.lstrip("."))
        if fmt == "npz" or path.suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            key = ref.get("key") or archive.files[0]
            if key not in archive.files:
                raise MaterializationError(f"missing key {key} in {path}")
            value = archive[key]
            index = ref.get("index")
            return value if index is None else value[int(index)]
        if fmt == "pickle.gz" or path.name.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as handle:
                value = pickle.load(handle)
            index = ref.get("index")
            return value if index is None else value[int(index)]
        raise MaterializationError(f"unsupported replay artifact format: {fmt}")

    def materialize(
        self,
        obs_uid: str,
        *,
        preferred_uid: str | uuid.UUID | None = None,
        geometry_contract: ObservationGeometryContract
        | Mapping[str, Any]
        | None = None,
        geometry_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            row = self.provenance.get_observation(obs_uid)
        except KeyError as exc:
            raise MaterializationError(f"unknown observation: {obs_uid}") from exc
        if row.get("status") != "kept":
            raise MaterializationError(f"observation is not replayable: {obs_uid}")
        if not row.get("pcd_ref") or not row.get("image_feat_ref"):
            raise MaterializationError(
                f"observation lacks exact PCD/CLIP payload: {obs_uid}"
            )

        import open3d as o3d
        import torch

        pcd_path = self.resolve_ref(row["pcd_ref"])
        pcd_archive = np.load(pcd_path, allow_pickle=False)
        if "points" not in pcd_archive.files or "colors" not in pcd_archive.files:
            raise MaterializationError(
                f"observation PCD lacks points/colors: {pcd_path}"
            )
        points = np.asarray(pcd_archive["points"], dtype=np.float64)
        colors = np.asarray(pcd_archive["colors"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise MaterializationError(f"invalid observation PCD shape: {pcd_path}")
        if (
            not len(points)
            or not np.isfinite(points).all()
            or not np.isfinite(colors).all()
        ):
            raise MaterializationError(f"invalid observation PCD values: {pcd_path}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        from conceptgraph.slam.utils import get_bounding_box

        bbox = get_bounding_box(self.cfg.get("spatial_sim_type", "overlap"), pcd)
        bbox.color = [0, 1, 0]
        feature = np.asarray(self.load_ref(row["image_feat_ref"]), dtype=np.float32)
        if feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
            raise MaterializationError(f"invalid CLIP feature for {obs_uid}")
        mask = np.asarray(self.load_ref(row["processed_mask_ref"]), dtype=bool)
        if mask.ndim != 2:
            raise MaterializationError(f"invalid processed mask for {obs_uid}")

        frame_idx = _frame_index(str(row["frame_uid"]))
        raw_idx = int(row["raw_det_idx"])
        frame = self.frames.get(row["frame_uid"], {})
        rgb_path = frame.get("rgb_path") or row.get("crop_ref", {}).get("path", "")
        digest = hashlib.sha256(obs_uid.encode()).digest()
        instance_color = (
            np.asarray([digest[0], digest[1], digest[2]], dtype=np.float64) / 255.0
        )
        class_id = int(row["class_id"])
        class_name = str(row["class_name"])
        detection = {
            "id": _uuid(preferred_uid, obs_uid),
            "image_idx": [frame_idx],
            "mask_idx": [int(row.get("filtered_det_idx", raw_idx))],
            "color_path": [Path(str(rgb_path))],
            "class_name": class_name,
            "class_id": [class_id],
            "captions": [
                {
                    "id": str(row.get("filtered_det_idx", raw_idx)),
                    "name": class_name,
                    "caption": row.get("raw_caption"),
                }
            ],
            "num_detections": 1,
            "mask": [mask],
            "xyxy": [np.asarray(row["bbox_2d"], dtype=np.float32)],
            "conf": [np.float32(row.get("confidence") or 0.0)],
            "n_points": int(len(points)),
            "contain_number": [None],
            "inst_color": instance_color,
            "is_background": class_name in set(self.cfg.get("bg_classes", ())),
            "clip_ft": torch.from_numpy(feature.copy()),
            "num_obj_in_class": 1,
            "curr_obj_num": frame_idx * 1000 + raw_idx,
            "new_counter": frame_idx * 1000 + raw_idx,
            "obs_uids": [obs_uid],
            "pcd": pcd,
            "bbox": bbox,
        }
        if geometry_contract is None:
            return detection

        contract = (
            geometry_contract
            if isinstance(geometry_contract, ObservationGeometryContract)
            else ObservationGeometryContract.from_mapping(geometry_contract)
        )
        if contract.obs_uid != obs_uid:
            raise MaterializationError(
                f"geometry contract scope mismatch: {contract.obs_uid} != {obs_uid}"
            )
        try:
            source_verification = contract.verify_source_bindings(
                row, base_root=self.provenance.experiment_root
            )
            payload = contract.load_payload(base_root=self.provenance.experiment_root)
        except ValueError as exc:
            raise MaterializationError(
                f"geometry contract failed closed for {obs_uid}: {exc}"
            ) from exc
        restored_pcd = o3d.geometry.PointCloud()
        restored_pcd.points = o3d.utility.Vector3dVector(payload["points"])
        restored_pcd.colors = o3d.utility.Vector3dVector(payload["colors"])
        restored_bbox = get_bounding_box(
            self.cfg.get("spatial_sim_type", "overlap"), restored_pcd
        )
        restored_bbox.color = [0, 1, 0]
        detection.update(
            {
                "pcd": restored_pcd,
                "bbox": restored_bbox,
                "mask": [payload["mask"]],
                "n_points": int(len(payload["points"])),
            }
        )
        if geometry_audit is not None:
            geometry_audit.clear()
            geometry_audit.update(
                {
                    "applied": True,
                    "obs_uid": obs_uid,
                    "payload_uid": contract.payload_uid,
                    "source_binding_pass": source_verification["pass"],
                    "source_observation_sha256": source_verification[
                        "source_observation_sha256"
                    ],
                    "replacement_pcd_sha256": contract.replacement_pcd_ref["sha256"],
                    "replacement_mask_sha256": contract.replacement_mask_ref["sha256"],
                    "replacement_points_sha256": payload["replacement_points_sha256"],
                    "replacement_colors_sha256": payload["replacement_colors_sha256"],
                    "replacement_mask_array_sha256": payload[
                        "replacement_mask_array_sha256"
                    ],
                    "original_point_count": int(len(points)),
                    "restored_point_count": int(len(payload["points"])),
                    "original_mask_area": int(mask.sum()),
                    "restored_mask_area": int(payload["mask"].sum()),
                }
            )
        return detection

    def fidelity(self, obs_uid: str) -> dict[str, Any]:
        row = self.provenance.get_observation(obs_uid)
        detection = self.materialize(obs_uid)
        points = np.asarray(detection["pcd"].points)
        archive = np.load(self.resolve_ref(row["pcd_ref"]), allow_pickle=False)
        expected = np.asarray(archive["points"])
        center = np.asarray(detection["bbox"].get_center(), dtype=float)
        extent = np.asarray(detection["bbox"].extent, dtype=float)
        expected_center = np.asarray(row.get("bbox_3d_center"), dtype=float)
        expected_extent = np.asarray(row.get("bbox_3d_extent"), dtype=float)
        clip_expected = np.asarray(
            self.load_ref(row["image_feat_ref"]), dtype=np.float32
        )
        clip_actual = detection["clip_ft"].detach().cpu().numpy()
        checks = {
            "point_count_equal": len(points) == int(row.get("n_points", -1)),
            "points_exact": np.array_equal(points, expected),
            "bbox_center_allclose": bool(
                expected_center.shape == (3,)
                and np.allclose(center, expected_center, atol=2e-4)
            ),
            "bbox_extent_allclose": bool(
                expected_extent.shape == (3,)
                and np.allclose(extent, expected_extent, atol=2e-4)
            ),
            "clip_exact": np.array_equal(clip_actual, clip_expected),
            "class_equal": detection["class_id"] == [int(row["class_id"])]
            and detection["class_name"] == row["class_name"],
            "obs_uid_equal": detection["obs_uids"] == [obs_uid],
        }
        return {"pass": all(checks.values()), "obs_uid": obs_uid, "checks": checks}

    def rebuild_object_from_members(
        self,
        obs_uids: list[str] | tuple[str, ...],
        *,
        preferred_uid: str | uuid.UUID | None = None,
        run_final_dbscan: bool = False,
    ) -> dict[str, Any]:
        if not obs_uids:
            raise MaterializationError("cannot rebuild an object without observations")
        from conceptgraph.slam.utils import (
            merge_obj2_into_obj1,
            process_pcd,
            get_bounding_box,
        )

        result = copy.deepcopy(
            self.materialize(obs_uids[0], preferred_uid=preferred_uid)
        )
        for obs_uid in obs_uids[1:]:
            result = merge_obj2_into_obj1(
                obj1=result,
                obj2=self.materialize(obs_uid),
                downsample_voxel_size=float(
                    self.cfg.get("downsample_voxel_size", 0.01)
                ),
                dbscan_remove_noise=bool(self.cfg.get("dbscan_remove_noise", True)),
                dbscan_eps=float(self.cfg.get("dbscan_eps", 0.1)),
                dbscan_min_points=int(self.cfg.get("dbscan_min_points", 10)),
                spatial_sim_type=str(self.cfg.get("spatial_sim_type", "overlap")),
                device=str(self.cfg.get("device", "cpu")),
                run_dbscan=False,
            )
        if run_final_dbscan:
            result["pcd"] = process_pcd(
                result["pcd"],
                float(self.cfg.get("downsample_voxel_size", 0.01)),
                bool(self.cfg.get("dbscan_remove_noise", True)),
                float(self.cfg.get("dbscan_eps", 0.1)),
                int(self.cfg.get("dbscan_min_points", 10)),
                run_dbscan=True,
            )
            result["n_points"] = len(np.asarray(result["pcd"].points))
            result["bbox"] = get_bounding_box(
                str(self.cfg.get("spatial_sim_type", "overlap")), result["pcd"]
            )
        return result
