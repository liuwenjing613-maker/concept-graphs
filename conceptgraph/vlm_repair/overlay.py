from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from .evidence import EndpointEvidence, sha256_file


class OverlayError(RuntimeError):
    """Raised when a derived-map mutation cannot be proven isolated and coherent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bbox_corners(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise OverlayError("cannot compute bbox for an empty/non-3D point array")
    minimum = np.nanmin(points, axis=0)
    maximum = np.nanmax(points, axis=0)
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def _merge_value(first: Any, second: Any) -> Any:
    if isinstance(first, list) and isinstance(second, list):
        return first + second
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        if first.ndim == second.ndim and first.ndim >= 1 and first.shape[1:] == second.shape[1:]:
            return np.concatenate([first, second], axis=0)
    return first


def _merge_objects(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(primary)
    primary_count = int(primary.get("num_detections") or 0)
    secondary_count = int(secondary.get("num_detections") or 0)
    detection_fields = {
        "image_idx",
        "mask_idx",
        "color_path",
        "class_id",
        "captions",
        "mask",
        "xyxy",
        "conf",
        "obs_uids",
    }
    for key in detection_fields:
        if key in primary and key in secondary:
            merged[key] = _merge_value(primary[key], secondary[key])
    for key in ("pcd_np", "pcd_color_np"):
        if key not in primary or key not in secondary:
            raise OverlayError(f"MERGE_WITH requires {key} in both serialized objects")
        merged[key] = np.concatenate(
            [np.asarray(primary[key]), np.asarray(secondary[key])], axis=0
        )
    merged["num_detections"] = primary_count + secondary_count
    if "clip_ft" not in primary or "clip_ft" not in secondary:
        raise OverlayError("MERGE_WITH requires clip_ft in both serialized objects")
    primary_feature = np.asarray(primary["clip_ft"])
    secondary_feature = np.asarray(secondary["clip_ft"])
    if primary_feature.shape != secondary_feature.shape or primary_feature.size == 0:
        raise OverlayError("MERGE_WITH requires aligned non-empty clip_ft arrays")
    total_count = primary_count + secondary_count
    if total_count <= 0:
        raise OverlayError("MERGE_WITH requires a positive combined detection count")
    feature_dtype = np.result_type(
        primary_feature.dtype, secondary_feature.dtype, np.float32
    )
    merged_feature = (
        primary_feature.astype(np.float64) * primary_count
        + secondary_feature.astype(np.float64) * secondary_count
    ) / total_count
    feature_norm = float(np.linalg.norm(merged_feature))
    if not np.isfinite(feature_norm) or feature_norm <= 0.0:
        raise OverlayError("MERGE_WITH produced an invalid clip_ft vector")
    merged["clip_ft"] = (merged_feature / feature_norm).astype(feature_dtype)
    merged["n_points"] = int(len(merged["pcd_np"]))
    merged["bbox_np"] = _bbox_corners(np.asarray(merged["pcd_np"]))
    return merged


def _membership_bbox(points: np.ndarray) -> tuple[list[float], list[float]]:
    minimum = np.nanmin(points, axis=0)
    maximum = np.nanmax(points, axis=0)
    return ((minimum + maximum) / 2.0).tolist(), (maximum - minimum).tolist()


def _refresh_membership_indices(
    objects: list[dict[str, Any]], membership: list[dict[str, Any]]
) -> None:
    if len(objects) != len(membership):
        raise OverlayError("object and membership lengths diverged")
    for index, (obj, member) in enumerate(zip(objects, membership)):
        member["current_object_index"] = index
        member["class_name"] = obj.get("class_name")
        member["num_detections"] = int(obj.get("num_detections") or 0)
        member["n_points"] = int(len(np.asarray(obj.get("pcd_np"))))
        center, extent = _membership_bbox(np.asarray(obj["pcd_np"]))
        member["bbox_center"] = center
        member["bbox_extent"] = extent


def apply_repairs_to_bundle(
    bundle: dict[str, Any],
    membership: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    derived = copy.deepcopy(bundle)
    derived_membership = copy.deepcopy(membership)
    objects = derived.get("objects")
    if not isinstance(objects, list) or not isinstance(derived_membership, list):
        raise OverlayError("unexpected map bundle/membership schema")
    if len(objects) != len(derived_membership):
        raise OverlayError("final membership is not aligned with serialized objects")
    edge_bundle = derived.get("edges")
    if isinstance(edge_bundle, dict) and "objects" in edge_bundle:
        edge_objects = edge_bundle["objects"]
        if not isinstance(edge_objects, list) or len(edge_objects) != len(objects):
            raise OverlayError("edge-bundle object snapshot is not aligned with objects")
        object_ids = [str(obj.get("id", index)) for index, obj in enumerate(objects)]
        edge_object_ids = [
            str(obj.get("id", index)) for index, obj in enumerate(edge_objects)
        ]
        if edge_object_ids != object_ids:
            raise OverlayError("edge-bundle object IDs are not aligned with objects")
        if edge_bundle.get("edges") and any(
            repair.get("action") in {"DELETE", "MERGE_WITH"} for repair in repairs
        ):
            raise OverlayError(
                "structural repair is unsafe while serialized graph edges are non-empty"
            )
    for index, member in enumerate(derived_membership):
        if member.get("current_object_index") != index:
            raise OverlayError("membership current_object_index is not exact")

    reports: list[dict[str, Any]] = []
    touched: set[str] = set()
    for repair in repairs:
        action = str(repair["action"])
        target_uid = str(repair["target_uid"])
        other_uid = str(repair.get("other_uid") or "") or None
        resources = {target_uid} | ({other_uid} if other_uid else set())
        if touched.intersection(resources):
            reports.append(
                {
                    **repair,
                    "apply_status": "SKIPPED_CONFLICT",
                    "apply_reason": "another approved repair already touches this object",
                }
            )
            continue
        uid_to_index = {
            str(member["object_uid"]): index
            for index, member in enumerate(derived_membership)
        }
        if target_uid not in uid_to_index:
            reports.append(
                {
                    **repair,
                    "apply_status": "SKIPPED_MISSING_TARGET",
                    "apply_reason": "target UID is not active in the derived map",
                }
            )
            continue
        target_index = uid_to_index[target_uid]
        if action == "RELABEL":
            old_label = objects[target_index].get("class_name")
            new_label = str(repair["new_label"]).strip()
            objects[target_index]["class_name"] = new_label
            derived_membership[target_index]["class_name"] = new_label
            reports.append(
                {
                    **repair,
                    "apply_status": "APPLIED",
                    "old_label": old_label,
                    "new_label": new_label,
                }
            )
        elif action == "DELETE":
            objects.pop(target_index)
            derived_membership.pop(target_index)
            reports.append({**repair, "apply_status": "APPLIED"})
        elif action == "MERGE_WITH":
            if not other_uid or other_uid not in uid_to_index:
                reports.append(
                    {
                        **repair,
                        "apply_status": "SKIPPED_MISSING_OTHER",
                        "apply_reason": "context object UID is not active",
                    }
                )
                continue
            other_index = uid_to_index[other_uid]
            # Keep the better-supported UID as primary so downstream references remain as
            # stable as possible; provenance records both source UIDs.
            if int(objects[other_index].get("num_detections") or 0) > int(
                objects[target_index].get("num_detections") or 0
            ):
                primary_index, secondary_index = other_index, target_index
            else:
                primary_index, secondary_index = target_index, other_index
            primary_uid = str(derived_membership[primary_index]["object_uid"])
            secondary_uid = str(derived_membership[secondary_index]["object_uid"])
            merged = _merge_objects(objects[primary_index], objects[secondary_index])
            primary_member = derived_membership[primary_index]
            secondary_member = derived_membership[secondary_index]
            combined_obs = list(
                dict.fromkeys(
                    list(primary_member.get("member_observation_uids") or [])
                    + list(secondary_member.get("member_observation_uids") or [])
                )
            )
            primary_member["member_observation_uids"] = combined_obs
            primary_member["parent_or_merged_from_object_uids"] = list(
                dict.fromkeys(
                    list(primary_member.get("parent_or_merged_from_object_uids") or [])
                    + [secondary_uid]
                )
            )
            objects[primary_index] = merged
            objects.pop(secondary_index)
            derived_membership.pop(secondary_index)
            reports.append(
                {
                    **repair,
                    "apply_status": "APPLIED",
                    "surviving_uid": primary_uid,
                    "redirected_uid": secondary_uid,
                }
            )
        else:
            reports.append(
                {
                    **repair,
                    "apply_status": "SKIPPED_UNREGISTERED",
                    "apply_reason": f"overlay executor does not implement {action}",
                }
            )
            continue
        touched.update(resources)
        _refresh_membership_indices(objects, derived_membership)
    derived["objects"] = objects
    if isinstance(edge_bundle, dict) and "objects" in edge_bundle:
        edge_bundle["objects"] = copy.deepcopy(objects)
    return derived, derived_membership, reports


def _object_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, obj in enumerate(objects, start=1):
        points = np.asarray(obj["pcd_np"])
        center, extent = _membership_bbox(points)
        object_id = obj.get("id", index - 1)
        if isinstance(object_id, UUID):
            object_id = str(object_id)
        elif isinstance(object_id, np.generic):
            object_id = object_id.item()
        result[f"object_{index}"] = {
            "id": object_id,
            "object_tag": obj.get("class_name"),
            "object_caption": obj.get("consolidated_caption", ""),
            "bbox_extent": [round(float(value), 2) for value in extent],
            "bbox_center": [round(float(value), 2) for value in center],
            "bbox_volume": round(float(np.prod(extent)), 2),
        }
    return result


def _load_approved_repairs(run_root: Path, scene_id: str) -> list[dict[str, Any]]:
    repairs = []
    for path in sorted((run_root / "cases" / scene_id).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        execution = result.get("execution") or {}
        if execution.get("executable") is not True:
            continue
        diagnosis = result["diagnosis"]
        repair = diagnosis["repair"]
        evidence = EndpointEvidence.load(result["case_dir"])
        other_alias = repair.get("other_alias")
        repairs.append(
            {
                "case_uid": result["case_uid"],
                "result_sha256": sha256_file(path),
                "action": repair["action"],
                "target_uid": result["target_object_uid"],
                "new_label": repair.get("new_label"),
                "other_uid": evidence.alias_to_uid.get(str(other_alias).upper())
                if other_alias
                else None,
                "diagnosis_confidence": diagnosis["confidence"],
                "verification_confidence": result["verification"]["parsed"]["confidence"],
            }
        )
    return repairs


def _scene_inputs(validation_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (validation_root / "incident_worklist_manifest.json").read_text(encoding="utf-8")
    )
    return list(manifest.get("scenes") or [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply approved VLM repairs only to new derived map files."
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_root = args.validation_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root in {validation_root, run_root} or validation_root in output_root.parents:
        raise SystemExit("output-root must be separate from source evidence and inference results")
    requested = set(args.scene)
    overall: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": "ali-my-VLM-only-repair-v1-derived-map",
        "created_at": _utc_now(),
        "source_validation_root": str(validation_root),
        "source_run_root": str(run_root),
        "in_place_mutation": False,
        "scenes": [],
    }
    for scene in _scene_inputs(validation_root):
        scene_id = str(scene["scene_id"])
        if requested and scene_id not in requested:
            continue
        experiment_dir = Path(scene["experiment_dir"])
        pcd_files = sorted(experiment_dir.glob("pcd_*.pkl.gz"))
        if len(pcd_files) != 1:
            raise OverlayError(f"expected one source PCD pickle in {experiment_dir}")
        source_pickle = pcd_files[0]
        membership_path = experiment_dir / "evidence" / "final_membership.json"
        with gzip.open(source_pickle, "rb") as stream:
            bundle = pickle.load(stream)  # trusted, manifest-bound project artifact
        membership = json.loads(membership_path.read_text(encoding="utf-8"))
        repairs = _load_approved_repairs(run_root, scene_id)
        derived, derived_membership, reports = apply_repairs_to_bundle(
            bundle, membership, repairs
        )
        scene_root = output_root / scene_id
        scene_root.mkdir(parents=True, exist_ok=True)
        output_pickle = scene_root / f"pcd_{scene_id}_ali_my_vlm_repaired_v1.pkl.gz"
        temporary = output_pickle.with_suffix(output_pickle.suffix + ".tmp")
        with gzip.open(temporary, "wb") as stream:
            pickle.dump(derived, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output_pickle)
        membership_output = scene_root / "final_membership_vlm_repaired_v1.json"
        objects_output = scene_root / "obj_json_vlm_repaired_v1.json"
        _write_json(membership_output, derived_membership)
        _write_json(objects_output, _object_summary(derived["objects"]))
        scene_manifest = {
            "scene_id": scene_id,
            "source_pickle": str(source_pickle),
            "source_pickle_sha256": sha256_file(source_pickle),
            "source_membership": str(membership_path),
            "source_membership_sha256": sha256_file(membership_path),
            "approved_repair_count": len(repairs),
            "reports": reports,
            "outputs": {
                str(output_pickle): sha256_file(output_pickle),
                str(membership_output): sha256_file(membership_output),
                str(objects_output): sha256_file(objects_output),
            },
        }
        _write_json(scene_root / "repair_manifest.json", scene_manifest)
        overall["scenes"].append(scene_manifest)
    _write_json(output_root / "derived_map_manifest.json", overall)
    print(json.dumps({"scene_count": len(overall["scenes"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
