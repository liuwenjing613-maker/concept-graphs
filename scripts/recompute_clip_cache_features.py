#!/usr/bin/env python3
"""Create a detection cache with recomputed CLIP features only.

YOLO boxes, SAM masks, classes, confidences, and provenance are reused from the
source cache. The source cache is never modified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import open_clip
import supervision as sv
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conceptgraph.utils.model_utils import compute_clip_features_batched


def load_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 1:
            raise ValueError(f"expected one array in {path}, found {archive.files}")
        return np.asarray(archive[archive.files[0]])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def resolve_rgb_path(stem: str, row: dict, rgb_root: Path | None) -> Path:
    recorded = Path(row.get("source", ""))
    if recorded.is_file():
        return recorded
    if rgb_root is not None:
        for suffix in (".jpg", ".png", ".jpeg"):
            candidate = rgb_root / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"RGB source for {stem} is unavailable: {recorded}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--destination-cache", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--bbox-padding", type=int, default=20)
    parser.add_argument("--masked-weight", type=float, default=0.5)
    parser.add_argument("--masked-background-factor", type=float, default=0.1)
    parser.add_argument("--masked-blur-radius", type=float, default=3.0)
    args = parser.parse_args()

    source_cache = args.source_cache.resolve()
    destination_cache = args.destination_cache.resolve()
    rgb_root = args.rgb_root.resolve() if args.rgb_root else None
    source_manifest_path = source_cache / "image_detection_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    if destination_cache.exists():
        raise FileExistsError(f"refusing to overwrite destination: {destination_cache}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise ValueError("source detection cache is not complete")
    frames = source_manifest.get("frames", {})
    if not frames:
        raise ValueError("source detection cache contains no frames")

    destination_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_cache.name}.tmp.",
            dir=destination_cache.parent,
        )
    )
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        shutil.rmtree(temporary_root)
        shutil.copytree(source_cache, temporary_root, copy_function=hardlink_or_copy)

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            args.clip_model,
            args.clip_pretrained,
            cache_dir=str(args.clip_cache_dir.resolve()) if args.clip_cache_dir else None,
        )
        clip_model = clip_model.to(args.device).eval()
        clip_tokenizer = open_clip.get_tokenizer(args.clip_model)

        output_rows = {}
        for stem, row in tqdm(sorted(frames.items()), desc="CLIP feature cache"):
            source_frame = source_cache / "detections" / stem
            destination_frame = temporary_root / "detections" / stem
            rgb_path = resolve_rgb_path(stem, row, rgb_root)
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"failed to read RGB image: {rgb_path}")
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            xyxy = load_npz(source_frame / "xyxy.npz")
            masks = load_npz(source_frame / "mask.npz")
            class_ids = load_npz(source_frame / "class_id.npz")
            confidence = load_npz(source_frame / "confidence.npz")
            with gzip.open(source_frame / "classes.pkl.gz", "rb") as stream:
                classes = np.asarray(pickle.load(stream))
            detections = sv.Detections(
                xyxy=xyxy,
                mask=masks,
                class_id=class_ids,
                confidence=confidence,
            )

            _, image_feats, _ = compute_clip_features_batched(
                image_rgb,
                detections,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
                classes,
                args.device,
                bbox_padding=args.bbox_padding,
                masked_weight=args.masked_weight,
                masked_background_factor=args.masked_background_factor,
                masked_blur_radius=args.masked_blur_radius,
            )
            feature_path = destination_frame / "image_feats.npz"
            feature_path.unlink()
            np.savez_compressed(feature_path, image_feats)
            norms = np.linalg.norm(image_feats.astype(np.float32), axis=-1)
            output_rows[stem] = {
                "count": int(len(image_feats)),
                "shape": list(image_feats.shape),
                "norm_min": float(norms.min()) if len(norms) else None,
                "norm_max": float(norms.max()) if len(norms) else None,
                "sha256": sha256_file(feature_path),
            }

        feature_manifest = {
            "schema": "clip-feature-cache-v1",
            "status": "complete",
            "source_cache": str(source_cache),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "recipe": {
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "clip_cache_dir": (
                    str(args.clip_cache_dir.resolve()) if args.clip_cache_dir else None
                ),
                "bbox_padding": args.bbox_padding,
                "bbox_weight": 1.0 - args.masked_weight,
                "masked_weight": args.masked_weight,
                "masked_background_factor": args.masked_background_factor,
                "masked_blur_radius": args.masked_blur_radius,
                "mask_stage": "raw_detection_before_filter_gobs",
            },
            "frames": output_rows,
        }
        (temporary_root / "clip_feature_manifest.json").write_text(
            json.dumps(feature_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_root.replace(destination_cache)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(destination_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
