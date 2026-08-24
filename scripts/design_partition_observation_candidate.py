#!/usr/bin/env python3
"""Design and audit one real PARTITION_OBSERVATION candidate without fabricating gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.partition import observation_payload_sha256


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_ref(provenance: ProvenanceIndex, ref: Mapping[str, Any]) -> Path:
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = provenance.experiment_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--case-uid", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = _read(args.identity_manifest)
    case = next(
        row for row in manifest.get("cases") or () if row["case_uid"] == args.case_uid
    )
    if case.get("causal_disposition") != "DEFER_NON_ASSOCIATION_ROOT":
        raise ValueError(
            "partition design case must be a deferred non-association root"
        )
    roots = list(case.get("root_observation_uids") or ())
    if len(roots) != 1:
        raise ValueError("partition design requires exactly one source observation")
    obs_uid = str(roots[0])

    provenance = ProvenanceIndex(args.base_run)
    observation = provenance.get_observation(obs_uid)
    pcd_ref = observation.get("pcd_ref")
    if not isinstance(pcd_ref, Mapping):
        raise ValueError("source observation has no stored point payload")
    pcd_path = _resolve_ref(provenance, pcd_ref)
    pcd_file_sha = _sha256(pcd_path)
    pcd_file_hash_matches = pcd_file_sha == pcd_ref.get("sha256")
    if not pcd_file_hash_matches:
        raise ValueError("stored point payload file hash mismatch")
    with np.load(pcd_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    point_count = int(np.asarray(payload["points"]).shape[0])
    aligned = {
        name: bool(np.asarray(array).ndim >= 1 and len(array) == point_count)
        for name, array in payload.items()
    }
    if not aligned or not all(aligned.values()):
        raise ValueError("stored observation payload fields are not point aligned")
    payload_hash = observation_payload_sha256(payload)
    association = provenance.get_association_for_obs(obs_uid)
    threshold = float(association["sim_threshold"])
    top_candidates = list(association.get("top_candidates") or ())
    top_score = float(top_candidates[0]["aggregate_score"]) if top_candidates else None
    recorded_decision = str(association.get("decision"))

    missing_requirements = [
        "uint16 point assignment with one label for every stored point",
        "independent point-to-physical-part evidence for table versus floor",
        "assignment artifact SHA-256",
        "pre-association payload executor integration",
        "independent partition endpoint evaluator",
    ]
    result = {
        "schema_version": "2.0.0",
        "design_uid": "partition_observation_f74cb_real_candidate_20260824",
        "case_uid": case["case_uid"],
        "incident_uid": case["incident_uid"],
        "evaluation_role": (
            "REAL_PAYLOAD_EXECUTABILITY_DESIGN; NOT_A_REPAIR CLAIM; "
            "NO SYNTHETIC OR HEURISTIC POINT ASSIGNMENT"
        ),
        "posthoc_human_interpretation": {
            "endpoint_error_type": case["endpoint_error_type"],
            "notes": case.get("human_label", {}).get("notes"),
            "earliest_causal_stage": case["earliest_causal_stage"],
        },
        "source_observation": {
            "obs_uid": obs_uid,
            "frame_uid": observation.get("frame_uid"),
            "class_name": observation.get("class_name"),
            "point_count": point_count,
            "point_payload_path": str(pcd_path),
            "point_payload_file_sha256": pcd_file_sha,
            "point_payload_file_hash_matches_ledger": pcd_file_hash_matches,
            "point_payload_fields": {
                name: {
                    "shape": list(np.asarray(array).shape),
                    "dtype": str(np.asarray(array).dtype),
                    "point_aligned": aligned[name],
                }
                for name, array in sorted(payload.items())
            },
            "observation_payload_sha256": payload_hash,
            "ledger_points_sha256": observation.get("points_sha256"),
            "pcd_stage": observation.get("pcd_stage"),
            "pcd_is_sampled": observation.get("pcd_is_sampled"),
            "pre_dbscan_cluster_count": (observation.get("pre_dbscan") or {}).get(
                "cluster_count"
            ),
            "pre_dbscan_cluster_labels_stored": False,
        },
        "native_association": {
            "event_uid": association["event_uid"],
            "event_sequence": provenance.sequence(association),
            "recorded_decision": recorded_decision,
            "top1_score": top_score,
            "sim_threshold": threshold,
            "strict_greater_than_merge": bool(
                top_score is not None and top_score > threshold
            ),
            "native_create_confirmed": recorded_decision == "CREATE_OBJECT",
        },
        "non_executable_contract_template": {
            "type": "PARTITION_OBSERVATION",
            "obs_uid": obs_uid,
            "source_point_count": point_count,
            "source_payload_sha256": payload_hash,
            "assignment_dtype": "uint16",
            "assignment_sha256": None,
            "parts": [
                {
                    "part_index": 0,
                    "semantic_role": "table_component",
                    "effective_identity_uid": None,
                },
                {
                    "part_index": 1,
                    "semantic_role": "floor_contamination",
                    "effective_identity_uid": None,
                },
            ],
            "identity_uid_allocation_status": (
                "WITHHELD_UNTIL_HASH_BOUND_POINT_ASSIGNMENT_EXISTS"
            ),
            "exhaustive_partition_proven": False,
            "disjoint_partition_proven": False,
            "nonempty_parts_proven": False,
        },
        "gate": {
            "candidate_family": "PARTITION_OBSERVATION",
            "decision": "DEFER",
            "constraint_emitted": False,
            "atomic_apply_attempted": False,
            "reason": (
                "The real 1,990-point payload is available and hash-bound, but no "
                "point-level table/floor assignment or independent split endpoint "
                "gold exists. Cluster geometry is not a substitute for physical-part "
                "truth, so emitting a partition would fabricate repair evidence."
            ),
            "missing_requirements": missing_requirements,
        },
        "future_admission_rule": {
            "all_required": True,
            "checks": [
                "payload hash equals the frozen observation payload hash",
                "assignment length equals source point count",
                "assignment dtype is canonical uint16",
                "part indices are contiguous and every part is nonempty",
                "every point has exactly one part",
                "each part receives one distinct effective identity",
                "pre-association executor applies every point-aligned field atomically",
                "shadow replay corrects an independently labeled endpoint",
                "collateral, invariant, negative-control, and parity gates pass",
            ],
        },
    }
    _write(args.output, result)
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": "DEFER",
                "point_count": point_count,
                "payload_sha256": payload_hash,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
