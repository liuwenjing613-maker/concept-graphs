#!/usr/bin/env python3
"""Audit real room0 observation projection without outcome-dependent selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from conceptgraph.revision.counterfactual_projection import (
    POSE_CONVENTION,
    CounterfactualProjectionVerifier,
    InstanceGeometry,
    ProjectionEvidenceLoader,
    sha256_file,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _frame_index(row: dict[str, Any]) -> int:
    return int(str(row["frame_uid"]).rsplit("_f", 1)[1])


def _select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows if row.get("processed_mask_ref") and row.get("pcd_ref")
    ]
    if count != 5:
        raise ValueError("the frozen v0 roundtrip audit requires exactly 5 rows")
    frame_values = np.asarray([_frame_index(row) for row in eligible])
    areas = np.asarray(
        [
            int(row.get("processed_mask_area") or row.get("mask_area") or 0)
            for row in eligible
        ]
    )
    frame_targets = np.linspace(frame_values.min(), frame_values.max(), count)
    area_quantiles = (0.10, 0.50, 0.90, 0.25, 0.75)
    selected = []
    used = set()
    for frame_target, quantile in zip(frame_targets, area_quantiles):
        area_target = float(np.quantile(areas, quantile))
        candidates = sorted(
            eligible,
            key=lambda row: (
                abs(_frame_index(row) - frame_target),
                str(row["obs_uid"]),
            ),
        )[:100]
        chosen = min(
            (row for row in candidates if str(row["obs_uid"]) not in used),
            key=lambda row: (
                abs(
                    int(row.get("processed_mask_area") or row.get("mask_area") or 0)
                    - area_target
                ),
                abs(_frame_index(row) - frame_target),
                str(row["obs_uid"]),
            ),
        )
        used.add(str(chosen["obs_uid"]))
        selected.append(chosen)
    return selected


def _transform_diagnostic(
    *,
    points: np.ndarray,
    frame: Any,
    world_to_camera: np.ndarray,
) -> dict[str, Any]:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=float)], axis=1
    )
    camera = homogeneous @ world_to_camera.T
    z = camera[:, 2]
    safe_z = np.where(z == 0.0, 1.0, z)
    intrinsics = frame.intrinsics
    u = np.rint(intrinsics[0, 0] * camera[:, 0] / safe_z + intrinsics[0, 2]).astype(int)
    v = np.rint(intrinsics[1, 1] * camera[:, 1] / safe_z + intrinsics[1, 2]).astype(int)
    height, width = frame.image_shape
    valid = (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth_errors = np.abs(z[valid] - frame.depth_m[v[valid], u[valid]])
    return {
        "positive_in_frame_point_count": int(valid.sum()),
        "median_depth_error_m": (
            float(np.median(depth_errors)) if len(depth_errors) else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    base_run = args.base_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    config = json.loads((base_run / "config_params.json").read_text(encoding="utf-8"))
    voxel_size = float(config["downsample_voxel_size"])
    verifier = CounterfactualProjectionVerifier(voxel_size=voxel_size)
    loader = ProjectionEvidenceLoader(base_run)
    selected = _select(
        _read_jsonl(base_run / "evidence/observations.jsonl"), args.count
    )

    audits = []
    for row in selected:
        obs_uid = str(row["obs_uid"])
        frame = loader.load_frame(str(row["frame_uid"]))
        pcd_path = (base_run / str(row["pcd_ref"]["path"])).resolve()
        mask_path = (base_run / str(row["processed_mask_ref"]["path"])).resolve()
        with np.load(pcd_path, allow_pickle=False) as archive:
            points = np.asarray(archive["points"], dtype=np.float64)
        with np.load(mask_path, allow_pickle=False) as archive:
            mask = np.asarray(
                archive[str(row["processed_mask_ref"].get("key") or "mask")],
                dtype=bool,
            )
        geometry = InstanceGeometry.build(
            member_obs_uids=(obs_uid,),
            points=points,
            source_state_hash="OBSERVATION_ROUNDTRIP",
        )
        projected = verifier.project_state(
            state_uid="ROUNDTRIP",
            instances=(geometry,),
            frame=frame,
        )[0]
        intersection = int(np.logical_and(projected.mask, mask).sum())
        union = int(np.logical_or(projected.mask, mask).sum())
        inverse = _transform_diagnostic(
            points=points,
            frame=frame,
            world_to_camera=np.linalg.inv(frame.pose),
        )
        direct = _transform_diagnostic(
            points=points,
            frame=frame,
            world_to_camera=frame.pose,
        )
        audits.append(
            {
                "obs_uid": obs_uid,
                "frame_uid": frame.frame_uid,
                "class_name": str(row.get("class_name") or "unknown"),
                "mask_area": int(mask.sum()),
                "point_count": int(len(points)),
                "projected_area": int(projected.mask.sum()),
                "visible_point_count": projected.visible_point_count,
                "projected_mask_iou": (float(intersection / union) if union else 0.0),
                "source_mask_coverage": float(intersection / max(1, int(mask.sum()))),
                "projected_mask_precision": float(
                    intersection / max(1, int(projected.mask.sum()))
                ),
                "inverse_pose_diagnostic": inverse,
                "direct_pose_diagnostic": direct,
                "frame_evidence_hash": frame.evidence_hash,
                "pcd_sha256": sha256_file(pcd_path),
                "processed_mask_sha256": sha256_file(mask_path),
            }
        )

    structural_pass = all(
        row["visible_point_count"] > 0
        and row["inverse_pose_diagnostic"]["positive_in_frame_point_count"]
        > row["direct_pose_diagnostic"]["positive_in_frame_point_count"]
        for row in audits
    )
    result = {
        "schema_version": "1.0.0",
        "audit_uid": "cmvic_roundtrip_"
        + hashlib.sha256(
            json.dumps(audits, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20],
        "status": (
            "PASS_TRANSFORM_PROVEN" if structural_pass else "STOP_AND_FIX_TRANSFORM"
        ),
        "selection_policy": (
            "FIVE_TEMPORAL_TARGETS_WITH_FIXED_MASK_AREA_QUANTILE_SCHEDULE"
        ),
        "selection_outcome_blind": True,
        "human_or_gold_loaded": False,
        "pose_convention": POSE_CONVENTION,
        "depth_unit": "meters",
        "depth_scale_from_dataset_config": loader.depth_scale_to_meters,
        "voxel_size_m": voxel_size,
        "depth_tolerance_m": verifier.depth_tolerance,
        "semantic_threshold_count": 0,
        "observation_count": len(audits),
        "observations": audits,
        "summary": {
            "minimum_iou": min(row["projected_mask_iou"] for row in audits),
            "mean_iou": float(np.mean([row["projected_mask_iou"] for row in audits])),
            "minimum_source_mask_coverage": min(
                row["source_mask_coverage"] for row in audits
            ),
            "maximum_inverse_median_depth_error_m": max(
                row["inverse_pose_diagnostic"]["median_depth_error_m"] for row in audits
            ),
        },
    }
    _write(output, result)
    print(json.dumps(result["summary"], indent=2))
    print(result["status"])
    return 0 if structural_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
