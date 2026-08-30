#!/usr/bin/env python3
"""Mine objective raw-good/processed-bad geometry-restoration candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


BACKGROUND = {
    "ceiling",
    "floor",
    "rug",
    "wall",
    "window",
    "door",
    "curtain",
    "blinds",
    "undefined",
    "unknown",
}
PATTERN = re.compile(r"_f(?P<frame>\d{6})_r(?P<proposal>\d{4})$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_for(path: Path, scan_name: str) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [scan for scan in payload["scans"] if scan["scan"] == scan_name]
    if len(matches) != 1:
        raise ValueError(f"expected one scan {scan_name}, got {len(matches)}")
    return {int(obj["id"]): str(obj["label"]) for obj in matches[0]["objects"]}


def npz_array(path: Path, preferred: str | None = None) -> np.ndarray:
    with np.load(path) as archive:
        if preferred in archive.files:
            return np.asarray(archive[preferred])
        if len(archive.files) != 1:
            raise ValueError(f"ambiguous npz keys {archive.files}: {path}")
        return np.asarray(archive[archive.files[0]])


def metric(mask: np.ndarray, semantic: np.ndarray, labels: dict[int, str]) -> dict | None:
    area = int(mask.sum())
    if not area:
        return None
    visible_ids, visible_counts = np.unique(semantic, return_counts=True)
    visible = {int(i): int(c) for i, c in zip(visible_ids.tolist(), visible_counts.tolist())}
    ids, counts = np.unique(semantic[mask], return_counts=True)
    candidates = []
    for instance_id, intersection in zip(ids.tolist(), counts.tolist()):
        instance_id = int(instance_id)
        if instance_id not in labels:
            continue
        intersection = int(intersection)
        gt_area = visible.get(instance_id, 0)
        union = area + gt_area - intersection
        candidates.append(
            {
                "gt_id": instance_id,
                "gt_label": labels[instance_id],
                "intersection_pixels": intersection,
                "mask_pixels": area,
                "gt_visible_pixels": gt_area,
                "purity": intersection / area,
                "recall": intersection / gt_area if gt_area else 0.0,
                "iou": intersection / union if union else 0.0,
            }
        )
    return max(candidates, key=lambda row: (row["intersection_pixels"], row["gt_id"])) if candidates else None


def same_owner_metric(mask: np.ndarray, semantic: np.ndarray, owner: dict) -> dict:
    area = int(mask.sum())
    intersection = int(np.count_nonzero(mask & (semantic == owner["gt_id"])))
    gt_area = int(np.count_nonzero(semantic == owner["gt_id"]))
    union = area + gt_area - intersection
    return {
        "gt_id": owner["gt_id"],
        "gt_label": owner["gt_label"],
        "intersection_pixels": intersection,
        "mask_pixels": area,
        "gt_visible_pixels": gt_area,
        "purity": intersection / area if area else 0.0,
        "recall": intersection / gt_area if gt_area else 0.0,
        "iou": intersection / union if union else 0.0,
    }


def usable(row: dict | None, threshold: float) -> bool:
    return bool(row and row["purity"] >= threshold and row["recall"] >= threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["room0", "office0"], required=True)
    parser.add_argument("--scan-name", required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--objects-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_run = args.base_run.resolve()
    processed_root = base_run / "evidence" / "processed_masks"
    detections_root = args.detections_root.resolve()
    gt_scene_root = args.gt_root.resolve() / args.scene
    labels = labels_for(args.objects_json.resolve(), args.scan_name)
    detection_dirs = sorted(path for path in detections_root.glob("frame[0-9][0-9][0-9][0-9][0-9][0-9]") if path.is_dir())
    if not detection_dirs:
        raise ValueError(f"no detection frame directories: {detections_root}")

    by_frame: dict[int, list[tuple[int, Path, str]]] = defaultdict(list)
    for path in sorted(processed_root.glob("*.npz")):
        match = PATTERN.search(path.stem)
        if not match:
            raise ValueError(f"cannot parse processed-mask name: {path.name}")
        frame_index = int(match.group("frame"))
        proposal_index = int(match.group("proposal"))
        by_frame[frame_index].append((proposal_index, path, path.stem))

    rows = []
    for ordinal, frame_index in enumerate(sorted(by_frame)):
        if frame_index >= len(detection_dirs):
            raise ValueError(f"frame index {frame_index} exceeds {len(detection_dirs)} detection frames")
        frame_dir = detection_dirs[frame_index]
        raw_frame = int(frame_dir.name.removeprefix("frame"))
        raw_path = frame_dir / "mask.npz"
        gt_path = gt_scene_root / f"frame{raw_frame:06d}.npz"
        raw_stack = npz_array(raw_path, "arr_0").astype(bool)
        semantic = npz_array(gt_path, "semantic").astype(np.uint16)
        if raw_stack.ndim != 3 or raw_stack.shape[1:] != semantic.shape:
            raise ValueError(f"raw/GT shape mismatch at {raw_frame}: {raw_stack.shape}, {semantic.shape}")
        for proposal_index, processed_path, obs_uid in by_frame[frame_index]:
            if proposal_index >= len(raw_stack):
                raise ValueError(f"proposal {proposal_index} out of range {len(raw_stack)}: {raw_path}")
            raw_mask = raw_stack[proposal_index]
            processed_mask = npz_array(processed_path, "mask").astype(bool)
            if processed_mask.shape != semantic.shape:
                raise ValueError(f"processed/GT shape mismatch: {processed_path}")
            raw_owner = metric(raw_mask, semantic, labels)
            if raw_owner is None:
                continue
            processed_owner = same_owner_metric(processed_mask, semantic, raw_owner)
            threshold_flags = {
                f"{threshold:.1f}": {
                    "raw_usable": usable(raw_owner, threshold),
                    "processed_usable": usable(processed_owner, threshold),
                    "restore_candidate": usable(raw_owner, threshold) and not usable(processed_owner, threshold),
                }
                for threshold in (0.3, 0.5, 0.7)
            }
            if not any(item["restore_candidate"] for item in threshold_flags.values()):
                continue
            rows.append(
                {
                    "scene": args.scene,
                    "obs_uid": obs_uid,
                    "frame_index": frame_index,
                    "raw_frame": raw_frame,
                    "proposal_index": proposal_index,
                    "gt_id": raw_owner["gt_id"],
                    "gt_label": raw_owner["gt_label"],
                    "is_background": raw_owner["gt_label"].lower() in BACKGROUND,
                    "raw": raw_owner,
                    "processed": processed_owner,
                    "recall_loss": raw_owner["recall"] - processed_owner["recall"],
                    "purity_loss": raw_owner["purity"] - processed_owner["purity"],
                    "thresholds": threshold_flags,
                    "raw_mask_path": str(raw_path.resolve()),
                    "processed_mask_path": str(processed_path.resolve()),
                    "gt_sidecar_path": str(gt_path.resolve()),
                }
            )
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(by_frame):
            print(f"{args.scene}: processed frames {ordinal + 1}/{len(by_frame)}, candidates={len(rows)}", flush=True)

    rows.sort(
        key=lambda row: (
            row["thresholds"]["0.5"]["restore_candidate"],
            not row["is_background"],
            row["recall_loss"],
            row["raw"]["recall"],
            row["raw"]["purity"],
        ),
        reverse=True,
    )
    for row in rows:
        row["hashes"] = {
            "raw_mask_stack_sha256": sha256_file(Path(row["raw_mask_path"])),
            "processed_mask_sha256": sha256_file(Path(row["processed_mask_path"])),
            "gt_sidecar_sha256": sha256_file(Path(row["gt_sidecar_path"])),
        }
    summary = {}
    for threshold in (0.3, 0.5, 0.7):
        key = f"{threshold:.1f}"
        selected = [row for row in rows if row["thresholds"][key]["restore_candidate"]]
        summary[key] = {
            "candidate_observations": len(selected),
            "candidate_unique_gt_objects": len({row["gt_id"] for row in selected}),
            "candidate_unique_frames": len({row["raw_frame"] for row in selected}),
            "foreground_observations": sum(not row["is_background"] for row in selected),
            "foreground_unique_gt_objects": len({row["gt_id"] for row in selected if not row["is_background"]}),
        }
    report = {
        "schema_version": "1.0.0",
        "evaluation_role": "objective GT mining for executable observation-level restoration",
        "scene": args.scene,
        "base_run": str(base_run),
        "base_run_evidence_manifest_sha256": sha256_file(base_run / "evidence" / "manifest.json"),
        "detections_root": str(detections_root),
        "processed_observations_scanned": sum(len(items) for items in by_frame.values()),
        "unique_frames_scanned": len(by_frame),
        "criteria": "raw purity and visible-instance recall both >= threshold; processed same-owner mask fails",
        "summary": summary,
        "candidates": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
