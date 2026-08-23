from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.benchmark.experiment_v1 import (
    aligned_relation_metrics,
    write_json,
)
from conceptgraph.revision.constraints import ReplayMode
from conceptgraph.revision.evaluate import geometry_metrics, membership_metrics
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.relations import (
    AliDevBaselineRelationBackend,
    load_baseline_frame_records,
)
from conceptgraph.revision.replay import CounterfactualReplayEngine
from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine


def _object_parity(
    reference: Mapping[str, Any],
    replayed: Mapping[str, Any],
    *,
    bbox_atol: float = 2e-3,
) -> dict[str, Any]:
    def keyed(state: Mapping[str, Any]) -> dict[tuple[str, ...], Mapping[str, Any]]:
        return {
            tuple(sorted(str(item) for item in row["member_observation_uids"])): row
            for row in state.get("objects") or ()
        }

    left = keyed(reference)
    right = keyed(replayed)
    missing = sorted(set(left) - set(right))
    unexpected = sorted(set(right) - set(left))
    rows = []
    for members in sorted(set(left) & set(right)):
        expected = left[members]
        actual = right[members]
        center_error = float(
            np.linalg.norm(
                np.asarray(expected["bbox_center"], dtype=float)
                - np.asarray(actual["bbox_center"], dtype=float)
            )
        )
        extent_error = float(
            np.linalg.norm(
                np.asarray(expected["bbox_extent"], dtype=float)
                - np.asarray(actual["bbox_extent"], dtype=float)
            )
        )
        checks = {
            "point_count_exact": int(expected["n_points"]) == int(actual["n_points"]),
            "point_digest_exact": expected.get("point_digest")
            == actual.get("point_digest"),
            "clip_feature_digest_exact": expected.get("clip_feature_digest")
            == actual.get("clip_feature_digest"),
            "class_histogram_exact": expected.get("class_histogram")
            == actual.get("class_histogram"),
            "class_name_exact": expected.get("class_name") == actual.get("class_name"),
            "bbox_center_within_tolerance": center_error <= bbox_atol,
            "bbox_extent_within_tolerance": extent_error <= bbox_atol,
        }
        if not all(checks.values()):
            rows.append(
                {
                    "member_observation_uids": list(members),
                    "reference_entity_uid": expected["entity_uid"],
                    "replayed_entity_uid": actual["entity_uid"],
                    "checks": checks,
                    "center_error": center_error,
                    "extent_error": extent_error,
                    "reference_n_points": expected["n_points"],
                    "replayed_n_points": actual["n_points"],
                }
            )
    return {
        "pass": not missing and not unexpected and not rows,
        "bbox_atol": bbox_atol,
        "matched_object_count": len(set(left) & set(right)),
        "missing_member_partitions": [list(item) for item in missing],
        "unexpected_member_partitions": [list(item) for item in unexpected],
        "mismatch_count": len(rows),
        "mismatches": rows,
    }


def _point_digest(points: np.ndarray, *, quantized: bool) -> str:
    value = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if quantized:
        value = np.round(value, decimals=6)
        if len(value):
            order = np.lexsort((value[:, 2], value[:, 1], value[:, 0]))
            value = value[order]
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _bbox_corner_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    """Permutation-invariant distance between two serialized box corner sets."""

    left = np.asarray(first, dtype=float).reshape(-1, 3)
    right = np.asarray(second, dtype=float).reshape(-1, 3)
    if len(left) != 8 or len(right) != 8:
        return float("inf")
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _raw_object_parity(
    frozen_objects: list[Mapping[str, Any]],
    replayed_objects: Any,
    *,
    bbox_atol: float = 2e-3,
    feature_cosine_tolerance: float = 1e-5,
    member_normalizer: Callable[[Any], str] = str,
) -> dict[str, Any]:
    frozen = {
        tuple(sorted(member_normalizer(item) for item in row.get("obs_uids", ()))): row
        for row in frozen_objects
    }
    replayed = {
        tuple(sorted(member_normalizer(item) for item in row.get("obs_uids", ()))): row
        for row in replayed_objects
    }
    missing = sorted(set(frozen) - set(replayed))
    unexpected = sorted(set(replayed) - set(frozen))
    mismatches = []
    exact_point_digest_count = 0
    for members in sorted(set(frozen) & set(replayed)):
        expected = frozen[members]
        actual = replayed[members]
        expected_points = np.asarray(expected["pcd_np"], dtype=np.float32)
        actual_points = np.asarray(actual["pcd"].points, dtype=np.float32)
        expected_bbox = np.asarray(expected["bbox_np"], dtype=float)
        expected_center = expected_bbox.mean(axis=0)
        actual_bbox_points = np.asarray(actual["bbox"].get_box_points(), dtype=float)
        actual_center = np.asarray(actual["bbox"].get_center(), dtype=float)
        center_error = float(np.linalg.norm(expected_center - actual_center))
        bbox_corner_hausdorff = _bbox_corner_hausdorff(
            expected_bbox, actual_bbox_points
        )
        expected_clip = np.asarray(expected.get("clip_ft", ()), dtype=float).reshape(-1)
        actual_clip_value = actual.get("clip_ft")
        if hasattr(actual_clip_value, "detach"):
            actual_clip_value = actual_clip_value.detach().cpu().numpy()
        actual_clip = np.asarray(actual_clip_value, dtype=float).reshape(-1)
        denominator = float(np.linalg.norm(expected_clip) * np.linalg.norm(actual_clip))
        cosine = (
            float(np.dot(expected_clip, actual_clip) / denominator)
            if denominator and expected_clip.shape == actual_clip.shape
            else 0.0
        )
        expected_histogram = dict(
            Counter(str(int(value)) for value in expected.get("class_id", ()))
        )
        actual_histogram = dict(
            Counter(str(int(value)) for value in actual.get("class_id", ()))
        )
        exact_point_digest = _point_digest(
            expected_points, quantized=False
        ) == _point_digest(actual_points, quantized=False)
        quantized_point_digest = _point_digest(
            expected_points, quantized=True
        ) == _point_digest(actual_points, quantized=True)
        exact_point_digest_count += int(exact_point_digest)
        checks = {
            "point_count_exact": len(expected_points) == len(actual_points),
            "point_set_digest_1e_6_exact": quantized_point_digest,
            "clip_feature_cosine_within_tolerance": cosine
            >= 1.0 - feature_cosine_tolerance,
            "class_histogram_exact": expected_histogram == actual_histogram,
            "bbox_center_within_tolerance": center_error <= bbox_atol,
            "bbox_corner_hausdorff_within_tolerance": bbox_corner_hausdorff
            <= bbox_atol,
        }
        if not all(checks.values()):
            mismatches.append(
                {
                    "member_observation_uids": list(members),
                    "checks": checks,
                    "center_error": center_error,
                    "bbox_corner_hausdorff": bbox_corner_hausdorff,
                    "clip_feature_cosine": cosine,
                    "raw_point_digest_exact": exact_point_digest,
                    "expected_n_points": len(expected_points),
                    "replayed_n_points": len(actual_points),
                }
            )
    return {
        "pass": not missing and not unexpected and not mismatches,
        "bbox_atol": bbox_atol,
        "feature_cosine_tolerance": feature_cosine_tolerance,
        "bbox_comparison": "PERMUTATION_INVARIANT_CORNER_HAUSDORFF",
        "matched_object_count": len(set(frozen) & set(replayed)),
        "raw_exact_point_digest_count": exact_point_digest_count,
        "missing_member_partitions": [list(item) for item in missing],
        "unexpected_member_partitions": [list(item) for item in unexpected],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _expected_postprocess_counts(engine: SparseCounterfactualReplayEngine) -> dict[str, int]:
    from conceptgraph.slam.utils import processing_needed

    counts = {}
    for name in ("denoise", "filter", "merge"):
        interval = int(engine.cfg.get(f"{name}_interval", 0))
        final_enabled = bool(engine.cfg.get(f"run_{name}_final_frame", False))
        counts[name] = sum(
            processing_needed(
                interval,
                final_enabled,
                frame,
                frame == engine.final_frame,
            )
            for frame in range(engine.final_frame + 1)
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate natural full replay parity")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--edge-stream")
    args = parser.parse_args()
    provenance = ProvenanceIndex(args.base_run)
    reference = CounterfactualReplayEngine(provenance).clean_state()
    engine = SparseCounterfactualReplayEngine(provenance)
    replayed, replayed_objects = engine.replay_global_with_objects(
        mode=ReplayMode.NATURAL_REPLAY
    )
    frozen_paths = sorted(provenance.experiment_root.glob("pcd_*.pkl.gz"))
    if len(frozen_paths) != 1:
        raise FileNotFoundError(f"expected one frozen map, found {len(frozen_paths)}")
    with gzip.open(frozen_paths[0], "rb") as handle:
        frozen_payload = pickle.load(handle)
    relation_details = {}
    for name, state in (("reference", reference), ("replayed", replayed)):
        objects, records = load_baseline_frame_records(
            provenance,
            state["membership"],
            edge_stream_root=args.edge_stream,
        )
        rebuilt = AliDevBaselineRelationBackend().rebuild(
            objects=objects, frame_records=records
        )
        state["edges"] = rebuilt["output_edges"]
        relation_details[name] = rebuilt

    all_observations = {
        str(obs_uid)
        for members in reference["membership"].values()
        for obs_uid in members
    }
    membership = membership_metrics(reference["membership"], replayed["membership"])
    geometry = geometry_metrics(
        reference, replayed, observation_scope=all_observations
    )
    relation = aligned_relation_metrics(reference, replayed)
    objects = _raw_object_parity(frozen_payload["objects"], replayed_objects)
    expected_postprocess_counts = _expected_postprocess_counts(engine)
    first_decision_divergence = None
    for row in replayed["decision_trace"]:
        association = provenance.get_association_for_obs(str(row["obs_uid"]))
        expected_create = str(association.get("decision")) == "CREATE_OBJECT"
        replayed_create = row.get("natural_match") is None
        if expected_create != replayed_create:
            first_decision_divergence = {
                "frame_idx": row.get("frame_idx"),
                "obs_uid": row.get("obs_uid"),
                "event_uid": row.get("event_uid"),
                "expected_decision": association.get("decision"),
                "replayed_natural_match": row.get("natural_match"),
                "natural_candidates": row.get("natural_candidates"),
            }
            break
    checks = {
        "membership_partition_exact": membership["member_f1"] >= 1.0 - 1e-12,
        "object_count_equal": len(reference["membership"]) == len(replayed["membership"]),
        "bbox_iou_ge_0_999": geometry["bbox_iou_to_clean"] >= 0.999,
        "object_payload_parity": objects["pass"],
        "postprocess_schedule_exact": replayed["postprocess_counts"]
        == expected_postprocess_counts,
        "decision_create_merge_parity": first_decision_divergence is None,
        "relation_state_exact_after_entity_alignment": relation["edge_state_match"],
        "source_hashes_equal": reference["source_hashes"] == replayed["source_hashes"],
    }
    result = {
        "schema_version": "1.0.0",
        "pass": all(checks.values()),
        "checks": checks,
        "membership": membership,
        "geometry": geometry,
        "object_payload_parity": objects,
        "relation": relation,
        "reference_object_count": len(reference["membership"]),
        "replayed_object_count": len(replayed["membership"]),
        "first_decision_divergence": first_decision_divergence,
        "runtime_ms": replayed["runtime_ms"],
        "postprocess_counts": replayed["postprocess_counts"],
        "expected_postprocess_counts": expected_postprocess_counts,
        "relation_rebuild": relation_details,
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "checks": result["checks"],
                "reference_object_count": result["reference_object_count"],
                "replayed_object_count": result["replayed_object_count"],
                "payload_mismatch_count": objects["mismatch_count"],
                "first_decision_divergence": first_decision_divergence,
                "runtime_ms": result["runtime_ms"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
