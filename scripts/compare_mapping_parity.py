"""Compare two mapping runs by canonical membership and numeric content."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_FIELDS = ("bbox_np", "pcd_np", "pcd_color_np", "clip_ft")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {root}, found {len(matches)}")
    return matches[0]


def _membership(item: dict) -> tuple[tuple[int, int], ...]:
    image_indices = list(item.get("image_idx", []))
    mask_indices = list(item.get("mask_idx", []))
    if len(image_indices) != len(mask_indices):
        raise ValueError("image_idx and mask_idx length mismatch")
    return tuple(sorted((int(image), int(mask)) for image, mask in zip(image_indices, mask_indices)))


def _membership_label(value: tuple[tuple[int, int], ...]) -> str:
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:12]
    return f"membership-{digest}-n{len(value)}"


def _load_run(root: Path) -> dict:
    pcd_path = _single(root, "pcd_*.pkl.gz")
    object_json_path = _single(root, "obj_json_*.json")
    edge_json_path = _single(root, "edge_json_*.json")
    trace_path = root / "parity_trace.json"
    with gzip.open(pcd_path, "rb") as handle:
        payload = pickle.load(handle)
    objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
    by_membership = {}
    for item in objects:
        key = _membership(item)
        if key in by_membership:
            raise ValueError(f"duplicate canonical membership in {root}: {_membership_label(key)}")
        by_membership[key] = item
    return {
        "root": root,
        "pcd_path": pcd_path,
        "object_json_path": object_json_path,
        "edge_json_path": edge_json_path,
        "trace_path": trace_path,
        "objects": objects,
        "by_membership": by_membership,
        "object_json": json.loads(object_json_path.read_text(encoding="utf-8")),
        "edge_json": json.loads(edge_json_path.read_text(encoding="utf-8")),
        "trace": json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else None,
    }


def _array_result(left: Any, right: Any, atol: float, rtol: float) -> dict:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    same_shape = left_array.shape == right_array.shape
    equal = bool(
        same_shape
        and np.allclose(left_array, right_array, atol=atol, rtol=rtol, equal_nan=True)
    )
    max_abs_diff = None
    if same_shape and left_array.size:
        difference = np.abs(left_array.astype(float) - right_array.astype(float))
        finite = difference[np.isfinite(difference)]
        max_abs_diff = float(finite.max()) if finite.size else 0.0
    return {
        "equal_within_tolerance": equal,
        "left_shape": list(left_array.shape),
        "right_shape": list(right_array.shape),
        "max_abs_diff": max_abs_diff,
    }


def compare_runs(off_dir: Path, on_dir: Path, *, atol: float, rtol: float) -> dict:
    off = _load_run(off_dir.resolve())
    on = _load_run(on_dir.resolve())
    off_keys = set(off["by_membership"])
    on_keys = set(on["by_membership"])
    common = sorted(off_keys & on_keys)
    missing_from_on = [_membership_label(key) for key in sorted(off_keys - on_keys)]
    missing_from_off = [_membership_label(key) for key in sorted(on_keys - off_keys)]
    object_mismatches = []
    max_differences = {field: 0.0 for field in ARRAY_FIELDS}
    for key in common:
        left = off["by_membership"][key]
        right = on["by_membership"][key]
        mismatch = {"membership": _membership_label(key), "fields": {}}
        scalar_checks = {
            "class_id_histogram": Counter(map(str, left.get("class_id", [])))
            == Counter(map(str, right.get("class_id", []))),
            "num_detections": int(left.get("num_detections", 0))
            == int(right.get("num_detections", 0)),
            "active_status": bool(left.get("is_active", True))
            == bool(right.get("is_active", True)),
        }
        for name, equal in scalar_checks.items():
            if not equal:
                mismatch["fields"][name] = {"equal": False}
        for field in ARRAY_FIELDS:
            result = _array_result(left.get(field, []), right.get(field, []), atol, rtol)
            if result["max_abs_diff"] is not None:
                max_differences[field] = max(max_differences[field], result["max_abs_diff"])
            if not result["equal_within_tolerance"]:
                mismatch["fields"][field] = result
        if mismatch["fields"]:
            object_mismatches.append(mismatch)

    trace_available = off["trace"] is not None and on["trace"] is not None
    checks = {
        "object_count_equal": len(off["objects"]) == len(on["objects"]),
        "canonical_membership_equal": off_keys == on_keys,
        "object_fields_within_tolerance": not object_mismatches,
        "object_json_equal": off["object_json"] == on["object_json"],
        "edge_topology_equal": off["edge_json"] == on["edge_json"],
        "parity_trace_present": trace_available,
        "per_frame_counts_equal": trace_available and off["trace"] == on["trace"],
    }
    passed = all(checks.values())
    return {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if passed else "FAIL",
        "tolerances": {"absolute": atol, "relative": rtol},
        "inputs": {
            "evidence_off": {
                "directory": str(off["root"]),
                "pcd_sha256": _sha256(off["pcd_path"]),
                "object_json_sha256": _sha256(off["object_json_path"]),
                "edge_json_sha256": _sha256(off["edge_json_path"]),
            },
            "evidence_on": {
                "directory": str(on["root"]),
                "pcd_sha256": _sha256(on["pcd_path"]),
                "object_json_sha256": _sha256(on["object_json_path"]),
                "edge_json_sha256": _sha256(on["edge_json_path"]),
            },
        },
        "checks": checks,
        "counts": {
            "evidence_off_objects": len(off["objects"]),
            "evidence_on_objects": len(on["objects"]),
            "common_memberships": len(common),
            "evidence_off_edges": len(off["edge_json"]),
            "evidence_on_edges": len(on["edge_json"]),
            "evidence_off_trace_frames": len(off["trace"] or []),
            "evidence_on_trace_frames": len(on["trace"] or []),
        },
        "max_abs_differences": max_differences,
        "missing_from_evidence_on": missing_from_on,
        "missing_from_evidence_off": missing_from_off,
        "object_mismatches": object_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-off", type=Path, required=True)
    parser.add_argument("--evidence-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    report = compare_runs(args.evidence_off, args.evidence_on, atol=args.atol, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
