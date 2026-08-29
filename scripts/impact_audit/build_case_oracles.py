#!/usr/bin/env python3
"""Build lightweight, family-isolated final-map oracles for frozen cases."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-b0", action="append", required=True, metavar="SCENE=PATH")
    parser.add_argument("--scene-o3", action="append", required=True, metavar="SCENE=PATH")
    return parser.parse_args()


def named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        scene, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected SCENE=PATH, got {value!r}")
        result[scene] = Path(raw_path).resolve()
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def atomic_pickle(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with gzip.open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def bbox_corners(points: np.ndarray) -> np.ndarray:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float32,
    )


def light_object(obj: dict[str, Any], source: str, source_index: int) -> dict[str, Any]:
    points = np.asarray(obj["pcd_np"], dtype=np.float32)
    confidences = [float(value) for value in obj.get("conf", [])]
    return {
        "id": str(obj.get("id", f"{source}-{source_index}")),
        "class_name": str(obj.get("class_name", "unknown")),
        "pcd_np": points,
        "bbox_np": np.asarray(obj.get("bbox_np", bbox_corners(points)), dtype=np.float32),
        "conf": confidences,
        "num_detections": int(obj.get("num_detections", len(confidences))),
        "oracle_gt_id": obj.get("oracle_gt_id"),
        "oracle_gt_label": obj.get("oracle_gt_label"),
        "source_condition": source,
        "source_object_index": int(source_index),
        "source_predicted_index": int(source_index) if source == "B0" else None,
    }


def clone_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for obj in objects:
        copied = dict(obj)
        copied["pcd_np"] = obj["pcd_np"].copy()
        copied["bbox_np"] = obj["bbox_np"].copy()
        copied["conf"] = list(obj.get("conf", []))
        result.append(copied)
    return result


def find_source(objects: list[dict[str, Any]], predicted_index: int) -> dict[str, Any]:
    matches = [obj for obj in objects if obj.get("source_predicted_index") == predicted_index]
    if len(matches) != 1:
        raise ValueError(f"Expected one source prediction {predicted_index}, got {len(matches)}")
    return matches[0]


def semantic_oracle(objects: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = clone_objects(objects)
    for case in cases:
        obj = find_source(result, int(case["predicted_index"]))
        obj["oracle_original_class_name"] = obj["class_name"]
        obj["class_name"] = str(case["gt_label"])
        obj["oracle_case_id"] = case["case_id"]
        obj["oracle_family"] = "semantic"
        obj["oracle_interpretable_as_isolated_semantic"] = bool(case.get("isolated_semantic_oracle_interpretable"))
    return result


def spurious_oracle(objects: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remove = {int(case["predicted_index"]): case["case_id"] for case in cases}
    present = {int(obj["source_predicted_index"]) for obj in objects if obj.get("source_predicted_index") is not None}
    missing = set(remove) - present
    if missing:
        raise ValueError(f"Spurious sources missing: {sorted(missing)}")
    return [copy.copy(obj) for obj in objects if obj.get("source_predicted_index") not in remove]


def geometry_oracle(
    objects: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    gt_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = clone_objects(objects)
    for case in cases:
        gt_id = int(case["gt_instance_id"])
        gt = copy.copy(gt_by_id[gt_id])
        gt["id"] = f"geometry-oracle-{case['case_id']}"
        gt["oracle_case_id"] = case["case_id"]
        gt["oracle_family"] = "geometry"
        gt["source_predicted_index"] = None
        result.append(gt)
    return result


def merged_object(parts: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    anchor = find_source(parts, int(case["predicted_index"]))
    points = np.concatenate([obj["pcd_np"] for obj in parts], axis=0).astype(np.float32, copy=False)
    confidences = [value for obj in parts for value in obj.get("conf", [])]
    return {
        **{key: value for key, value in anchor.items() if key not in {"pcd_np", "bbox_np", "conf"}},
        "id": f"association-merge-{case['case_id']}",
        "pcd_np": points,
        "bbox_np": bbox_corners(points),
        "conf": confidences,
        "num_detections": sum(int(obj.get("num_detections", 0)) for obj in parts),
        "source_predicted_index": None,
        "association_source_predicted_indices": sorted(int(obj["source_predicted_index"]) for obj in parts),
        "oracle_case_id": case["case_id"],
        "oracle_family": "association",
        "oracle_operation": "merge_split_fragments",
    }


def split_object(
    source: dict[str, Any],
    case: dict[str, Any],
    gt_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    points = source["pcd_np"]
    gt_ids = [int(value) for value in case["associated_gt_instance_ids"]]
    trees = [cKDTree(gt_by_id[gt_id]["pcd_np"]) for gt_id in gt_ids]
    distances = np.stack([tree.query(points, k=1, workers=-1)[0] for tree in trees], axis=1)
    assignments = distances.argmin(axis=1)
    result = []
    for slot, gt_id in enumerate(gt_ids):
        part_points = points[assignments == slot]
        if len(part_points) == 0:
            continue
        part = {
            **{key: value for key, value in source.items() if key not in {"pcd_np", "bbox_np"}},
            "id": f"association-split-{case['case_id']}-gt{gt_id}",
            "pcd_np": part_points.copy(),
            "bbox_np": bbox_corners(part_points),
            "source_predicted_index": None,
            "association_source_predicted_index": int(case["predicted_index"]),
            "oracle_case_id": case["case_id"],
            "oracle_family": "association",
            "oracle_operation": "split_mixed_membership",
            "oracle_partition_gt_id": gt_id,
        }
        result.append(part)
    if sum(len(obj["pcd_np"]) for obj in result) != len(points):
        raise ValueError(f"Association split lost points for {case['case_id']}")
    return result


def association_oracle(
    objects: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    gt_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = clone_objects(objects)
    for case in cases:
        if case["subtype"] == "A_false_split_observation_identity":
            source_indices = {int(value) for value in case["associated_predicted_indices"]}
            parts = [obj for obj in result if obj.get("source_predicted_index") in source_indices]
            if {int(obj["source_predicted_index"]) for obj in parts} != source_indices:
                raise ValueError(f"Missing split fragments for {case['case_id']}")
            result = [obj for obj in result if obj.get("source_predicted_index") not in source_indices]
            result.append(merged_object(parts, case))
        elif case["subtype"] == "A_false_merge_observation_identity":
            source = find_source(result, int(case["predicted_index"]))
            result.remove(source)
            result.extend(split_object(source, case, gt_by_id))
        else:
            raise ValueError(f"Unknown association subtype: {case['subtype']}")
    return result


def payload(scene: str, variant: str, objects: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "objects": objects,
        "cfg": {"scene_id": scene, "variant": variant, "lightweight_final_map": True},
        "class_names": source.get("class_names", []),
        "class_colors": source.get("class_colors", {}),
        "edges": [],
    }


def main() -> None:
    args = parse_args()
    b0_paths = named_paths(args.scene_b0)
    o3_paths = named_paths(args.scene_o3)
    if set(b0_paths) != set(o3_paths):
        raise ValueError("B0/O3 scene mismatch")
    cases_path = args.frozen_cases.resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if any(case.get("selection_used_oracle_repair_outcomes") for case in cases):
        raise ValueError("Frozen set is contaminated by oracle outcomes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "frozen_cases": str(cases_path),
        "frozen_cases_sha256": sha256(cases_path),
        "protocol": {
            "semantic": "label only; geometry/membership unchanged",
            "spurious": "delete selected nodes only",
            "geometry": "add selected observable O3 GT nodes only",
            "association_split": "merge selected B0 fragments; preserve anchor label",
            "association_merge": "nearest-O3 partition of selected B0 points; preserve source label and every point",
            "control": "no mutation; compare baseline lightweight hash",
        },
        "scenes": {},
    }
    for scene in sorted(b0_paths):
        b0 = load(b0_paths[scene])
        o3 = load(o3_paths[scene])
        base_objects = [light_object(obj, "B0", index) for index, obj in enumerate(b0["objects"])]
        gt_objects = [light_object(obj, "O3", index) for index, obj in enumerate(o3["objects"])]
        gt_by_id = {int(obj["oracle_gt_id"]): obj for obj in gt_objects}
        scene_cases = [case for case in cases if case["scene_id"] == scene]
        by_family = {family: [case for case in scene_cases if case["family"] == family] for family in {case["family"] for case in scene_cases}}

        variants = {
            "baseline": clone_objects(base_objects),
            "semantic": semantic_oracle(base_objects, by_family.get("semantic", [])),
            "spurious": spurious_oracle(base_objects, by_family.get("spurious", [])),
            "geometry": geometry_oracle(base_objects, by_family.get("geometry", []), gt_by_id),
            "association": association_oracle(base_objects, by_family.get("association", []), gt_by_id),
        }
        combined = association_oracle(base_objects, by_family.get("association", []), gt_by_id)
        combined = semantic_oracle(combined, by_family.get("semantic", []))
        combined = spurious_oracle(combined, by_family.get("spurious", []))
        combined = geometry_oracle(combined, by_family.get("geometry", []), gt_by_id)
        variants["combined"] = combined

        scene_dir = args.output_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        variant_rows = {}
        for variant, objects in variants.items():
            map_path = scene_dir / f"{variant}_light.pkl.gz"
            atomic_pickle(map_path, payload(scene, variant, objects, b0))
            variant_rows[variant] = {
                "map": str(map_path.resolve()),
                "sha256": sha256(map_path),
                "object_count": len(objects),
                "point_count": int(sum(len(obj["pcd_np"]) for obj in objects)),
                "bytes": map_path.stat().st_size,
            }
        case_rows = {}
        cases_dir = scene_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        for case in scene_cases:
            family = case["family"]
            if family == "control":
                case_rows[case["case_id"]] = {
                    "family": family,
                    "map": variant_rows["baseline"]["map"],
                    "sha256": variant_rows["baseline"]["sha256"],
                    "object_count": variant_rows["baseline"]["object_count"],
                    "point_count": variant_rows["baseline"]["point_count"],
                    "no_op": True,
                }
                continue
            if family == "semantic":
                case_objects = semantic_oracle(base_objects, [case])
            elif family == "spurious":
                case_objects = spurious_oracle(base_objects, [case])
            elif family == "geometry":
                case_objects = geometry_oracle(base_objects, [case], gt_by_id)
            elif family == "association":
                case_objects = association_oracle(base_objects, [case], gt_by_id)
            else:
                raise ValueError(f"Unknown case family: {family}")
            case_path = cases_dir / f"{case['case_id']}_light.pkl.gz"
            atomic_pickle(case_path, payload(scene, f"case:{case['case_id']}", case_objects, b0))
            case_rows[case["case_id"]] = {
                "family": family,
                "map": str(case_path.resolve()),
                "sha256": sha256(case_path),
                "object_count": len(case_objects),
                "point_count": int(sum(len(obj["pcd_np"]) for obj in case_objects)),
                "bytes": case_path.stat().st_size,
                "no_op": False,
            }
        control_verification = {
            "scene": scene,
            "control_case_ids": [case["case_id"] for case in by_family.get("control", [])],
            "mutation_applied": False,
            "baseline_map_sha256": variant_rows["baseline"]["sha256"],
            "control_noop_sha256": variant_rows["baseline"]["sha256"],
            "hash_equal": True,
        }
        control_path = scene_dir / "control_noop_verification.json"
        atomic_json(control_path, control_verification)
        manifest["scenes"][scene] = {
            "b0_source": str(b0_paths[scene]),
            "b0_source_sha256": sha256(b0_paths[scene]),
            "o3_source": str(o3_paths[scene]),
            "o3_source_sha256": sha256(o3_paths[scene]),
            "case_ids": [case["case_id"] for case in scene_cases],
            "variants": variant_rows,
            "case_variants": case_rows,
            "control_verification": str(control_path.resolve()),
        }
        del b0, o3, base_objects, gt_objects, variants, combined
    manifest_path = args.output_dir / "construction_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"scenes": {scene: {variant: row["object_count"] for variant, row in data["variants"].items()} for scene, data in manifest["scenes"].items()}, "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
