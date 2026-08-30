#!/usr/bin/env python3
"""Objectively verify prior geometry-replay cases against rendered GT masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(path: Path) -> dict[str, dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        scan["scan"]: {int(obj["id"]): str(obj["label"]) for obj in scan["objects"]}
        for scan in payload["scans"]
    }


def source_by_role(contract: dict, role: str) -> dict:
    matches = [item for item in contract["source_artifacts"] if item.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected one {role}, got {len(matches)}")
    return matches[0]


def load_mask(source: dict) -> np.ndarray:
    path = Path(source["path"]).resolve()
    with np.load(path) as archive:
        key = source.get("key")
        if key is None:
            if len(archive.files) != 1:
                raise ValueError(f"ambiguous npz keys for {path}: {archive.files}")
            key = archive.files[0]
        values = np.asarray(archive[key])
    if values.ndim == 3:
        index = source.get("index")
        if index is None:
            raise ValueError(f"mask stack needs an index: {path}")
        values = values[int(index)]
    if values.ndim != 2:
        raise ValueError(f"expected 2D mask, got {values.shape}: {path}")
    return values.astype(bool)


def frame_number(contract: dict) -> int:
    for role in ("raw_mask", "depth", "rgb"):
        text = source_by_role(contract, role)["path"]
        match = re.search(r"(?:frame|depth)(\d{6})", text)
        if match:
            return int(match.group(1))
    raise ValueError("could not recover raw frame number")


def overlap(mask: np.ndarray, semantic: np.ndarray, labels: dict[int, str]) -> list[dict]:
    area = int(mask.sum())
    visible_ids, visible_counts = np.unique(semantic, return_counts=True)
    visible = {int(i): int(c) for i, c in zip(visible_ids, visible_counts)}
    ids, counts = np.unique(semantic[mask], return_counts=True)
    rows = []
    for instance_id, intersection in zip(ids.tolist(), counts.tolist()):
        instance_id = int(instance_id)
        if instance_id not in labels:
            continue
        intersection = int(intersection)
        gt_area = visible.get(instance_id, 0)
        union = area + gt_area - intersection
        rows.append(
            {
                "gt_id": instance_id,
                "gt_label": labels[instance_id],
                "intersection_pixels": intersection,
                "mask_pixels": area,
                "gt_visible_pixels": gt_area,
                "purity": intersection / area if area else 0.0,
                "recall": intersection / gt_area if gt_area else 0.0,
                "iou": intersection / union if union else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (row["intersection_pixels"], row["gt_id"]), reverse=True)


def audit_case(name: str, scene: str, build_path: Path, gt_root: Path, labels: dict) -> dict:
    build = json.loads(build_path.read_text(encoding="utf-8"))
    contract = build.get("contract") or build["constraint"]["geometry_contract"]
    raw_source = source_by_role(contract, "raw_mask")
    processed_source = source_by_role(contract, "processed_mask")
    raw = load_mask(raw_source)
    processed = load_mask(processed_source)
    frame = frame_number(contract)
    gt_path = (gt_root / scene / f"frame{frame:06d}.npz").resolve()
    with np.load(gt_path) as archive:
        semantic = np.asarray(archive["semantic"], dtype=np.uint16)
    if raw.shape != processed.shape or raw.shape != semantic.shape:
        raise ValueError(f"shape mismatch: raw={raw.shape}, processed={processed.shape}, gt={semantic.shape}")
    raw_rows = overlap(raw, semantic, labels)
    processed_rows = overlap(processed, semantic, labels)
    if not raw_rows:
        raise ValueError("raw mask has no labeled GT overlap")
    owner = raw_rows[0]
    processed_same = next((row for row in processed_rows if row["gt_id"] == owner["gt_id"]), None)
    metrics = build.get("geometry_metrics", {})
    source_checks = []
    for source in contract["source_artifacts"]:
        path = Path(source["path"]).resolve()
        expected = source.get("sha256")
        actual = sha256_file(path)
        source_checks.append(
            {
                "role": source.get("role"),
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_match": expected is None or expected == actual,
            }
        )
    return {
        "case": name,
        "scene": scene,
        "evaluation_role": build.get("evaluation_role"),
        "obs_uid": contract["obs_uid"],
        "raw_frame": frame,
        "build_manifest": str(build_path.resolve()),
        "build_manifest_sha256": sha256_file(build_path.resolve()),
        "gt_sidecar": str(gt_path),
        "gt_sidecar_sha256": sha256_file(gt_path),
        "all_source_hashes_match": all(row["hash_match"] for row in source_checks),
        "source_hash_checks": source_checks,
        "raw_best_owner": owner,
        "raw_second_owner": raw_rows[1] if len(raw_rows) > 1 else None,
        "raw_strict_usable_0p5": owner["purity"] >= 0.5 and owner["recall"] >= 0.5,
        "processed_same_owner": processed_same,
        "processed_strict_usable_0p5": bool(
            processed_same
            and processed_same["purity"] >= 0.5
            and processed_same["recall"] >= 0.5
        ),
        "raw_mask_pixels": int(raw.sum()),
        "processed_mask_pixels": int(processed.sum()),
        "processed_to_raw_area_ratio": float(processed.sum() / raw.sum()) if raw.any() else None,
        "manifest_geometry_metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-build", type=Path, required=True)
    parser.add_argument("--office-build", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--objects-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_labels = load_labels(args.objects_json.resolve())
    cases = [
        audit_case("room0_restore_dev", "room0", args.room_build.resolve(), args.gt_root.resolve(), all_labels["room_0"]),
        audit_case("office0_restore_holdout", "office0", args.office_build.resolve(), args.gt_root.resolve(), all_labels["office_0"]),
    ]
    report = {
        "schema_version": "1.0.0",
        "evaluation_role": "posthoc objective GT audit of already executed real replay cases",
        "criteria": {"raw_usable": "purity >= 0.5 and visible-instance recall >= 0.5"},
        "cases": cases,
        "aggregate": {
            "case_count": len(cases),
            "raw_strict_usable_count": sum(row["raw_strict_usable_0p5"] for row in cases),
            "processed_strict_usable_count": sum(row["processed_strict_usable_0p5"] for row in cases),
            "all_source_hashes_match": all(row["all_source_hashes_match"] for row in cases),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    for row in cases:
        print(
            row["case"],
            "raw", row["raw_best_owner"],
            "processed", row["processed_same_owner"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
