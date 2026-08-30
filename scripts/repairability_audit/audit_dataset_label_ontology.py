#!/usr/bin/env python3
"""Audit Replica/ReplicaSSG pairing and semantic-ontology sensitivity.

This audit deliberately separates three label policies:

* ``official_only``: the checked-in ReplicaSSG -> Visual Genome mapping only;
* ``current_aliases``: the policy used by the frozen evaluator;
* ``reviewed_lamp_aliases``: current policy plus an explicit, narrow lamp-family
  sensitivity set (for example ``desk lamp`` -> ``lamp``).

The expanded aliases are a sensitivity analysis, not a replacement ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.oracle.evaluate_geometry_semantics import (  # noqa: E402
    load_map,
    prepare_objects,
)
from scripts.repairability_audit.evaluate_class_conditioned_localization import (  # noqa: E402
    evaluate_one,
)


SCENES = {
    "room0": {
        "replicassg_scene": "room_0",
        "b0_exp": "b0_room0_fresh",
        "b0_json": "obj_json_b0_room0_fresh.json",
    },
    "office0": {
        "replicassg_scene": "office_0",
        "b0_exp": "b0_office0_fresh_final",
        "b0_json": "obj_json_b0_office0_fresh_final.json",
    },
}

CURRENT_ALIASES = {
    "arm chair": "chair",
    "armchair": "chair",
    "blinds": "curtain",
    "closet door": "door",
    "couch": "chair",
    "end table": "table",
    "coffee table": "table",
    "dining table": "table",
    "paper bag": "bag",
    "potted plant": "plant",
    "sofa": "chair",
    "sofa chair": "chair",
    "stool": "chair",
    "television": "screen",
    "tv": "screen",
}

# Narrow, explicit sensitivity set requested after noticing compound-name drift.
# These entries are never silently folded into the primary score.
REVIEWED_LAMP_ALIASES = {
    "bedside lamp": "lamp",
    "ceiling lamp": "lamp",
    "ceiling light": "lamp",
    "desk lamp": "lamp",
    "floor lamp": "lamp",
    "table lamp": "lamp",
}

SENSITIVITY_CONDITIONS = ("B0", "OM_all", "OG")


def normalize(label: object) -> str:
    return " ".join(
        str(label).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class PolicyCanonicalizer:
    def __init__(self, mapping_path: Path, mode: str):
        payload = load_json(mapping_path)
        self.replica = {
            normalize(source): normalize(target)
            for source, target in payload["Replica2VisualGenome"].items()
        }
        self.visual_genome = {
            normalize(item) for item in payload["VisualGenome_list"]
        }
        self.mode = mode
        aliases: dict[str, str] = {}
        if mode in {"current_aliases", "reviewed_lamp_aliases"}:
            aliases.update(CURRENT_ALIASES)
        if mode == "reviewed_lamp_aliases":
            aliases.update(REVIEWED_LAMP_ALIASES)
        if mode not in {"official_only", "current_aliases", "reviewed_lamp_aliases"}:
            raise ValueError(f"unsupported label mode: {mode}")
        self.aliases = aliases

    def __call__(self, label: object) -> str:
        value = normalize(label)
        # Match the frozen evaluator's precedence for reproducibility.
        if value in self.aliases:
            return self.aliases[value]
        if value in self.replica:
            return self.replica[value]
        if value in self.visual_genome:
            return value
        return "unknown"


def numbered_files(root: Path, pattern: str, prefix: str, suffix: str) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    malformed: list[str] = []
    expression = re.compile(rf"^{re.escape(prefix)}(\d{{6}}){re.escape(suffix)}$")
    for path in sorted(root.glob(pattern)):
        match = expression.match(path.name)
        if match:
            indices.append(int(match.group(1)))
        else:
            malformed.append(path.name)
    return indices, malformed


def exact_range(values: list[int], count: int) -> bool:
    return values == list(range(count))


def dataset_scene_audit(
    *,
    scene: str,
    dataset_root: Path,
    gt_root: Path,
    objects_payload: dict,
    objects_path: Path,
) -> dict:
    spec = SCENES[scene]
    scene_root = dataset_root / scene
    results_link = scene_root / "results"
    results_root = results_link.resolve()
    trajectory = (scene_root / "traj.txt").resolve()
    gt_manifest_path = gt_root / scene / "manifest.json"
    gt_manifest = load_json(gt_manifest_path)
    exp_root = scene_root / "exps" / spec["b0_exp"]
    config_path = exp_root / "config_params.json"
    config_detection_path = exp_root / "config_params_detections.json"
    config = load_json(config_path)
    config_detection = load_json(config_detection_path)

    rgb_indices, rgb_malformed = numbered_files(results_root, "frame*.jpg", "frame", ".jpg")
    depth_indices, depth_malformed = numbered_files(results_root, "depth*.png", "depth", ".png")
    trajectory_rows = [
        line for line in trajectory.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    gt_frames = [int(value) for value in gt_manifest["frames"]]
    sidecar_indices, sidecar_malformed = numbered_files(
        gt_root / scene, "frame*.npz", "frame", ".npz"
    )

    scan_entries = [
        entry
        for entry in objects_payload["scans"]
        if entry.get("scan") == spec["replicassg_scene"]
    ]
    scan_objects = scan_entries[0]["objects"] if len(scan_entries) == 1 else []
    scan_ids = [int(item["id"]) for item in scan_objects]
    visible_ids = [int(item["id"]) for item in gt_manifest["visible_instances"]]
    expected_frames = list(range(config["start"], len(rgb_indices), config["stride"]))
    b0_json_path = exp_root / spec["b0_json"]

    checks = {
        "dataset_scene_directory_exists": scene_root.is_dir(),
        "dataset_results_is_symlink": results_link.is_symlink(),
        "dataset_results_resolves_to_requested_scene": results_root.name == "results"
        and results_root.parent.name == scene,
        "rgb_count_equals_depth_count": len(rgb_indices) == len(depth_indices),
        "rgb_indices_are_contiguous": exact_range(rgb_indices, len(rgb_indices)),
        "depth_indices_are_contiguous": exact_range(depth_indices, len(depth_indices)),
        "rgb_names_well_formed": not rgb_malformed,
        "depth_names_well_formed": not depth_malformed,
        "trajectory_rows_equal_rgb_count": len(trajectory_rows) == len(rgb_indices),
        "trajectory_hash_matches_gt_render": sha256_file(trajectory)
        == gt_manifest["trajectory_sha256"],
        "sequence_matches_scene": gt_manifest["sequence"] == scene,
        "replicassg_scene_alias_matches": gt_manifest["source_scene"]
        == spec["replicassg_scene"],
        "b0_scene_matches": config["scene_id"] == scene,
        "b0_and_detection_configs_identical": config == config_detection,
        "b0_stride_matches_gt": config["stride"] == gt_manifest["stride"],
        "b0_starts_at_zero": config["start"] == 0,
        "b0_runs_to_end": config["end"] == -1,
        "b0_uses_cuda": config["device"] == "cuda",
        "b0_visual_render_disabled": config["vis_render"] is False,
        "gt_frames_equal_b0_online_frame_schedule": gt_frames == expected_frames,
        "gt_sidecars_equal_manifest_frames": sidecar_indices == gt_frames,
        "gt_sidecar_names_well_formed": not sidecar_malformed,
        "objects_file_hash_matches_gt_render": sha256_file(objects_path)
        == gt_manifest["objects_sha256"],
        "exactly_one_replicassg_scan_entry": len(scan_entries) == 1,
        "replicassg_object_ids_unique": len(scan_ids) == len(set(scan_ids)),
        "visible_instance_ids_exist_in_objects_file": set(visible_ids).issubset(scan_ids),
        "b0_json_exists": b0_json_path.is_file(),
        "b0_map_exists": any(exp_root.glob("pcd_*.pkl.gz")),
    }

    critical_failures = [key for key, passed in checks.items() if not passed]
    return {
        "pass": not critical_failures,
        "critical_failures": critical_failures,
        "checks": checks,
        "paths": {
            "dataset_scene_root": str(scene_root.resolve()),
            "dataset_results_resolved": str(results_root),
            "trajectory": str(trajectory),
            "gt_manifest": str(gt_manifest_path.resolve()),
            "b0_config": str(config_path.resolve()),
            "b0_json": str(b0_json_path.resolve()),
        },
        "counts": {
            "rgb_frames": len(rgb_indices),
            "depth_frames": len(depth_indices),
            "trajectory_rows": len(trajectory_rows),
            "online_frames": len(gt_frames),
            "replicassg_objects": len(scan_objects),
            "visible_replicassg_objects": len(visible_ids),
        },
        "frame_schedule": {
            "start": gt_frames[0] if gt_frames else None,
            "last": gt_frames[-1] if gt_frames else None,
            "stride": gt_manifest["stride"],
        },
        "gt_alignment_summary": gt_manifest["alignment_summary"],
        "noncritical_metadata_note": {
            "render_camera_path": config.get("render_camera_path"),
            "interpretation": (
                "The frozen office config inherited a room0 visualization-camera path, "
                "but vis_render=false and mapping selects RGB/depth/poses from dataset_root+scene_id."
            ),
        },
        "source_hashes": {
            "trajectory": sha256_file(trajectory),
            "gt_manifest": sha256_file(gt_manifest_path),
            "b0_config": sha256_file(config_path),
            "objects_json": sha256_file(objects_path),
        },
    }


def label_inventory(path: Path) -> dict:
    payload = load_json(path)
    counts = Counter(normalize(item.get("object_tag", "unknown")) for item in payload.values())
    return {"object_count": sum(counts.values()), "labels": dict(sorted(counts.items()))}


def semantic_from_frozen_matches(row: dict, canonicalize: PolicyCanonicalizer) -> dict:
    correct = 0
    denominator = 0
    predicted_unknown = 0
    for match in row["matches"]:
        gt_label = canonicalize(match["gt_label"])
        if gt_label == "unknown":
            continue
        denominator += 1
        pred_label = canonicalize(match["predicted_label"])
        predicted_unknown += int(pred_label == "unknown")
        correct += int(pred_label == gt_label)
    return {
        "correct": correct,
        "denominator": denominator,
        "accuracy": correct / denominator if denominator else None,
        "predicted_unknown_with_evaluable_gt": predicted_unknown,
    }


def compact_proxy(row: dict) -> dict:
    keys = (
        "class_conditioned_f1",
        "semantic_evaluable_predicted_nodes",
        "semantic_evaluable_gt_nodes",
        "unique_class_query_count",
        "unique_class_top1_success_count",
        "unique_class_top1_success_rate",
        "unique_class_top3_success_count",
        "unique_class_top3_success_rate",
        "unique_class_top1_within_1m_count",
        "unique_class_top1_within_1m_rate",
        "semantic_class_count_mae",
        "unique_class_top1_candidate_coverage",
    )
    return {key: row[key] for key in keys}


def close(a: object, b: object, tolerance: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tolerance
    return a == b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--objects-json", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    old_root = args.old_root.resolve()
    dataset_root = args.dataset_root.resolve()
    gt_root = (old_root / "pilot" / "gt_full").resolve()
    objects_path = args.objects_json.resolve()
    mapping_path = args.label_mapping.resolve()
    objects_payload = load_json(objects_path)
    mapping_payload = load_json(mapping_path)
    modes = ("official_only", "current_aliases", "reviewed_lamp_aliases")
    canonicalizers = {
        mode: PolicyCanonicalizer(mapping_path, mode) for mode in modes
    }

    normalized_sources: dict[str, set[str]] = defaultdict(set)
    for source, target in mapping_payload["Replica2VisualGenome"].items():
        normalized_sources[normalize(source)].add(normalize(target))
    normalized_source_conflicts = {
        source: sorted(targets)
        for source, targets in normalized_sources.items()
        if len(targets) > 1
    }

    dataset_audit = {}
    inventories = {}
    fixed_match_semantics = {}
    proxy_sensitivity = {}
    current_reproduction = {"semantic": {}, "downstream_proxy": {}}
    changed_pairs: Counter[tuple[str, str, str, str, str, str]] = Counter()

    for scene, spec in SCENES.items():
        dataset_audit[scene] = dataset_scene_audit(
            scene=scene,
            dataset_root=dataset_root,
            gt_root=gt_root,
            objects_payload=objects_payload,
            objects_path=objects_path,
        )
        b0_json = (
            dataset_root
            / scene
            / "exps"
            / spec["b0_exp"]
            / spec["b0_json"]
        )
        inventories[scene] = {"b0": label_inventory(b0_json)}
        observed = set(inventories[scene]["b0"]["labels"])
        inventories[scene]["reviewed_aliases_observed"] = sorted(
            observed & set(REVIEWED_LAMP_ALIASES)
        )

        metrics_path = root / "evaluation" / scene / "voxel0p05" / "metrics.json"
        metrics = load_json(metrics_path)
        rows = {row["name"]: row for row in metrics["results"]}
        fixed_match_semantics[scene] = {}
        semantic_reproduction_pass = True
        for condition, row in rows.items():
            fixed_match_semantics[scene][condition] = {
                mode: semantic_from_frozen_matches(row, canonicalizers[mode])
                for mode in modes
            }
            frozen_current = fixed_match_semantics[scene][condition]["current_aliases"]
            semantic_reproduction_pass &= (
                frozen_current["correct"] == row["semantic_correct"]
                and frozen_current["denominator"] == row["semantic_denominator"]
                and close(frozen_current["accuracy"], row["semantic_accuracy"])
            )
            for match in row["matches"]:
                pred_current = canonicalizers["current_aliases"](match["predicted_label"])
                pred_reviewed = canonicalizers["reviewed_lamp_aliases"](
                    match["predicted_label"]
                )
                gt_current = canonicalizers["current_aliases"](match["gt_label"])
                gt_reviewed = canonicalizers["reviewed_lamp_aliases"](match["gt_label"])
                if pred_current != pred_reviewed or gt_current != gt_reviewed:
                    changed_pairs[
                        (
                            scene,
                            condition,
                            normalize(match["predicted_label"]),
                            normalize(match["gt_label"]),
                            f"{pred_current}->{pred_reviewed}",
                            f"{gt_current}->{gt_reviewed}",
                        )
                    ] += 1
        current_reproduction["semantic"][scene] = semantic_reproduction_pass

        gt_payload = load_map(Path(metrics["gt_map"]).resolve())
        ground_truth = prepare_objects(gt_payload["objects"], 0.05)
        proxy_sensitivity[scene] = {}
        original_proxy_path = root / "downstream_proxy_v2" / scene / "voxel0p05.json"
        original_proxy = {
            row["name"]: row for row in load_json(original_proxy_path)["results"]
        }
        proxy_reproduction_pass = True
        for mode in modes:
            mode_rows = {}
            for condition in SENSITIVITY_CONDITIONS:
                source = rows[condition]
                evaluated = evaluate_one(
                    name=condition,
                    path=Path(source["map"]),
                    ground_truth=ground_truth,
                    voxel_size=0.05,
                    threshold=0.25,
                    canonicalize=canonicalizers[mode],
                )
                mode_rows[condition] = compact_proxy(evaluated)
                if mode == "current_aliases":
                    frozen = compact_proxy(original_proxy[condition])
                    proxy_reproduction_pass &= all(
                        close(mode_rows[condition][key], frozen[key]) for key in frozen
                    )
            proxy_sensitivity[scene][mode] = mode_rows
        current_reproduction["downstream_proxy"][scene] = proxy_reproduction_pass

    changed_rows = [
        {
            "scene": key[0],
            "condition": key[1],
            "predicted_raw": key[2],
            "gt_raw": key[3],
            "predicted_canonical_change": key[4],
            "gt_canonical_change": key[5],
            "count": count,
        }
        for key, count in sorted(changed_pairs.items())
    ]

    mode_conclusions = {}
    for mode in modes:
        scene_deltas = {
            scene: (
                proxy_sensitivity[scene][mode]["OM_all"]["class_conditioned_f1"]
                - proxy_sensitivity[scene][mode]["B0"]["class_conditioned_f1"]
            )
            for scene in SCENES
        }
        mode_conclusions[mode] = {
            "om_all_minus_b0_class_conditioned_f1": scene_deltas,
            "same_sign_across_scenes": len(
                {0 if value == 0 else (1 if value > 0 else -1) for value in scene_deltas.values()}
            )
            == 1,
        }

    payload = {
        "schema_version": "1.0.0",
        "evaluation_role": (
            "dataset/ontology validity and sensitivity audit; expanded aliases are not ground truth"
        ),
        "dataset_audit": dataset_audit,
        "dataset_pairing_pass": all(row["pass"] for row in dataset_audit.values()),
        "ontology": {
            "mapping_path": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "objects_path": str(objects_path),
            "objects_sha256": sha256_file(objects_path),
            "official_source_label_count": len(mapping_payload["Replica2VisualGenome"]),
            "visual_genome_label_count": len(mapping_payload["VisualGenome_list"]),
            "normalized_source_conflicts": normalized_source_conflicts,
            "current_aliases": CURRENT_ALIASES,
            "reviewed_lamp_aliases": REVIEWED_LAMP_ALIASES,
            "reviewed_alias_policy": (
                "narrow compound-lamp sensitivity only; never selected post-hoc per sample"
            ),
        },
        "label_inventory": inventories,
        "fixed_geometry_match_semantic_sensitivity": fixed_match_semantics,
        "downstream_proxy_5cm_sensitivity": proxy_sensitivity,
        "current_policy_exact_reproduction": current_reproduction,
        "changed_matched_pairs_current_to_reviewed": changed_rows,
        "mode_conclusions": mode_conclusions,
        "interpretation_rules": [
            "Geometry/AP/F1 conclusions do not depend on semantic aliases.",
            "The official-only policy is the strict cross-dataset baseline.",
            "Current aliases reproduce prior scores and are retained for comparability.",
            "Reviewed lamp aliases are reported only as sensitivity, not as a new primary metric.",
            "A direction claim is called ontology-robust only if it is stable in all three modes.",
        ],
    }
    payload["pass"] = bool(
        payload["dataset_pairing_pass"]
        and not normalized_source_conflicts
        and all(current_reproduction["semantic"].values())
        and all(current_reproduction["downstream_proxy"].values())
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "dataset_pairing_pass": payload["dataset_pairing_pass"],
                "current_policy_exact_reproduction": current_reproduction,
                "reviewed_aliases_observed": {
                    scene: inventories[scene]["reviewed_aliases_observed"]
                    for scene in SCENES
                },
                "mode_conclusions": mode_conclusions,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
