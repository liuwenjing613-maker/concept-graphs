#!/usr/bin/env python3
"""Build hash-bound, two-stage annotation packets from a causal evidence ledger.

The builder is a read-only sidecar.  In --follow mode it waits one mapper frame
before exposing an event, so all event/object-version rows from that frame have
already been written.  It never writes into the evidence root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = "experiment0-association-packet/1.0"
DEFAULT_SEED = "experiment0-prevalence-v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--worklist", type=Path)
    parser.add_argument("--sample-probability", type=float, default=0.20)
    parser.add_argument("--sample-seed", default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--history-views", type=int, default=6)
    parser.add_argument("--min-history-observations", type=int, default=2)
    parser.add_argument("--min-history-frames", type=int, default=2)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                # A live writer can leave one final incomplete line between polls.
                if line_no == sum(1 for _ in path.open("r", encoding="utf-8")):
                    break
                raise
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(seed: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stable_case_uid(scene: str, event_uid: str) -> str:
    short = hashlib.sha256(f"{scene}:{event_uid}".encode()).hexdigest()[:16]
    return f"{scene}_event_{short}"


def resolve_artifact(exp_root: Path, ref: dict[str, Any] | None) -> Path | None:
    if not ref or not ref.get("path"):
        return None
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = exp_root / path
    path = path.resolve()
    return path if path.is_file() else None


def frame_index_from_uid(uid: str) -> int:
    marker = "_f"
    try:
        return int(uid.rsplit(marker, 1)[1][:6])
    except (IndexError, ValueError):
        return -1


def load_rgb(exp_root: Path, frame: dict[str, Any]) -> Image.Image:
    candidates: list[Path] = []
    if frame.get("rgb_path"):
        candidates.append(Path(str(frame["rgb_path"])))
    ref_path = resolve_artifact(exp_root, frame.get("rgb_ref"))
    if ref_path is not None:
        candidates.append(ref_path)
    for path in candidates:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"RGB missing for frame {frame.get('frame_uid')}")


def load_mask(exp_root: Path, observation: dict[str, Any]) -> np.ndarray:
    path = resolve_artifact(exp_root, observation.get("processed_mask_ref"))
    if path is None:
        raise FileNotFoundError(f"processed mask missing for {observation.get('obs_uid')}")
    key = str((observation.get("processed_mask_ref") or {}).get("key") or "mask")
    with np.load(path, allow_pickle=False) as bundle:
        mask = np.asarray(bundle[key], dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask is not HxW for {observation.get('obs_uid')}")
    return mask


def load_points(exp_root: Path, observation: dict[str, Any]) -> np.ndarray:
    path = resolve_artifact(exp_root, observation.get("pcd_ref"))
    if path is None:
        return np.empty((0, 3), dtype=float)
    key = str((observation.get("pcd_ref") or {}).get("key") or "points")
    with np.load(path, allow_pickle=False) as bundle:
        points = np.asarray(bundle[key], dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    points = points[:, :3]
    return points[np.isfinite(points).all(axis=1)]


def mask_edge(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    array = np.asarray(image, dtype=np.uint8).copy()
    if array.shape[:2] != mask.shape:
        raise ValueError(f"RGB/mask shape mismatch: {array.shape[:2]} vs {mask.shape}")
    fill = mask
    array[fill] = np.round(array[fill] * 0.55 + np.asarray(color) * 0.45).astype(np.uint8)
    edge = mask_edge(mask)
    array[edge] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def crop_bbox(image: Image.Image, bbox: list[float] | None, margin: float = 0.18) -> Image.Image:
    if not bbox or len(bbox) != 4:
        return image.copy()
    x1, y1, x2, y2 = map(float, bbox)
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    x1 = max(0, int(math.floor(x1 - width * margin)))
    y1 = max(0, int(math.floor(y1 - height * margin)))
    x2 = min(image.width, int(math.ceil(x2 + width * margin)))
    y2 = min(image.height, int(math.ceil(y2 + height * margin)))
    return image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))


def fit_image(image: Image.Image, size: tuple[int, int], background=(245, 247, 252)) -> Image.Image:
    canvas = Image.new("RGB", size, background)
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def labeled_tile(image: Image.Image, label: str, size=(300, 235)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(205, 211, 224), width=2)
    fitted = fit_image(image, (size[0] - 12, size[1] - 42))
    canvas.paste(fitted, (6, 34))
    draw.text((10, 9), label, fill=(28, 37, 55), font=ImageFont.load_default())
    return canvas


def contact_sheet(tiles: list[Image.Image], columns: int = 2, gap: int = 10) -> Image.Image:
    if not tiles:
        empty = Image.new("RGB", (620, 180), "white")
        ImageDraw.Draw(empty).text((18, 18), "No usable history view", fill=(155, 40, 40))
        return empty
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * width + (columns - 1) * gap, rows * height + (rows - 1) * gap), (241, 244, 250))
    for index, tile in enumerate(tiles):
        x = (index % columns) * (width + gap)
        y = (index // columns) * (height + gap)
        sheet.paste(tile, (x, y))
    return sheet


def select_history_observations(
    version: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    members = [observations[str(uid)] for uid in version.get("member_observation_uids") or [] if str(uid) in observations]
    members.sort(key=lambda row: (frame_index_from_uid(str(row.get("frame_uid") or "")), str(row.get("obs_uid"))))
    if len(members) <= limit:
        return members
    indices = {0, len(members) - 1}
    indices.add(max(range(len(members)), key=lambda idx: int(members[idx].get("processed_mask_area") or 0)))
    indices.add(max(range(len(members)), key=lambda idx: float(members[idx].get("confidence") or 0)))
    if limit > len(indices):
        for value in np.linspace(0, len(members) - 1, num=limit, dtype=int):
            indices.add(int(value))
            if len(indices) >= limit:
                break
    return [members[index] for index in sorted(indices)[:limit]]


def render_observation_assets(
    exp_root: Path,
    observation: dict[str, Any],
    frames: dict[str, dict[str, Any]],
    output_dir: Path,
    prefix: str,
    color: tuple[int, int, int],
) -> tuple[str, str]:
    frame = frames[str(observation["frame_uid"])]
    rgb = load_rgb(exp_root, frame)
    mask = load_mask(exp_root, observation)
    overlay = overlay_mask(rgb, mask, color)
    context_name = f"{prefix}_context.jpg"
    crop_name = f"{prefix}_crop.jpg"
    overlay.save(output_dir / context_name, quality=90)
    crop_bbox(overlay, observation.get("bbox_2d")).save(output_dir / crop_name, quality=92)
    return context_name, crop_name


def render_history_sheet(
    exp_root: Path,
    version: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    frames: dict[str, dict[str, Any]],
    output_path: Path,
    limit: int,
) -> list[dict[str, Any]]:
    selected = select_history_observations(version, observations, limit)
    tiles: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []
    for index, observation in enumerate(selected, 1):
        try:
            rgb = load_rgb(exp_root, frames[str(observation["frame_uid"])])
            mask = load_mask(exp_root, observation)
            overlay = overlay_mask(rgb, mask, (25, 173, 197))
            crop = crop_bbox(overlay, observation.get("bbox_2d"))
        except (FileNotFoundError, KeyError, ValueError):
            continue
        frame_idx = frame_index_from_uid(str(observation.get("frame_uid") or ""))
        tiles.append(labeled_tile(crop, f"H{index}  frame={frame_idx}"))
        metadata.append({
            "display_index": index,
            "obs_uid": observation.get("obs_uid"),
            "frame_idx": frame_idx,
        })
    contact_sheet(tiles).save(output_path, quality=91)
    return metadata


def sampled_points(points: np.ndarray, limit: int, seed_text: str) -> np.ndarray:
    if len(points) <= limit:
        return points
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return points[np.sort(rng.choice(len(points), size=limit, replace=False))]


def render_3d_comparison(
    current: np.ndarray,
    history: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    width, height = 1080, 360
    image = Image.new("RGB", (width, height), (250, 251, 254))
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), title, fill=(30, 38, 57), font=ImageFont.load_default())
    panels = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    all_points = np.concatenate([array for array in (current, history) if len(array)], axis=0) if len(current) or len(history) else np.empty((0, 3))
    for panel_index, (axis_a, axis_b, name) in enumerate(panels):
        left = panel_index * 360 + 10
        top = 35
        right = left + 340
        bottom = 345
        draw.rectangle((left, top, right, bottom), outline=(200, 207, 220), width=1)
        draw.text((left + 8, top + 7), name, fill=(70, 78, 96), font=ImageFont.load_default())
        if not len(all_points):
            continue
        values_a = all_points[:, axis_a]
        values_b = all_points[:, axis_b]
        low_a, high_a = np.quantile(values_a, [0.01, 0.99])
        low_b, high_b = np.quantile(values_b, [0.01, 0.99])
        if high_a <= low_a:
            high_a = low_a + 1e-3
        if high_b <= low_b:
            high_b = low_b + 1e-3
        for points, color in ((history, (20, 162, 184)), (current, (220, 62, 108))):
            for point in points:
                x = left + 14 + int(np.clip((point[axis_a] - low_a) / (high_a - low_a), 0, 1) * (right - left - 28))
                y = bottom - 14 - int(np.clip((point[axis_b] - low_b) / (high_b - low_b), 0, 1) * (bottom - top - 40))
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    draw.rectangle((810, 9, 823, 21), fill=(220, 62, 108))
    draw.text((828, 8), "current", fill=(70, 78, 96), font=ImageFont.load_default())
    draw.rectangle((905, 9, 918, 21), fill=(20, 162, 184))
    draw.text((923, 8), "candidate history", fill=(70, 78, 96), font=ImageFont.load_default())
    image.save(output_path, quality=92)


def load_similarity_row(exp_root: Path, association: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve_artifact(exp_root, association.get("aggregate_sim_ref"))
    if path is None:
        return []
    with np.load(path, allow_pickle=False) as bundle:
        obs_uids = [str(value) for value in bundle["observation_uids"].tolist()]
        object_uids = [str(value) for value in bundle["object_uids"].tolist()]
        spatial = np.asarray(bundle["spatial_sim"], dtype=float)
        visual = np.asarray(bundle["visual_sim"], dtype=float)
        aggregate = np.asarray(bundle["aggregate_sim"], dtype=float)
    obs_uid = str(association["obs_uid"])
    if obs_uid not in obs_uids:
        return []
    row = obs_uids.index(obs_uid)
    version_uids = association.get("candidate_object_version_uids") or []
    if len(version_uids) != len(object_uids):
        raise ValueError(f"candidate/version mismatch for {association.get('event_uid')}")
    return [
        {
            "object_uid": object_uid,
            "object_version_uid": version_uids[index],
            "spatial_score": float(spatial[row, index]),
            "visual_score": float(visual[row, index]),
            "aggregate_score": float(aggregate[row, index]),
        }
        for index, object_uid in enumerate(object_uids)
    ]


def candidate_cards(
    association: dict[str, Any],
    similarity: list[dict[str, Any]],
    versions: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(similarity, key=lambda row: (-row["aggregate_score"], row["object_uid"]))
    chosen = ranked[:top_k]
    target_uid = str(association.get("target_object_uid") or "")
    if target_uid and not any(row["object_uid"] == target_uid for row in chosen):
        target = next((row for row in ranked if row["object_uid"] == target_uid), None)
        if target is not None:
            chosen.append(target)
    chosen = [row for row in chosen if row.get("object_version_uid") in versions]
    seed = int(hashlib.sha256(str(association["event_uid"]).encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(chosen)
    for index, row in enumerate(chosen):
        row["code"] = chr(ord("A") + index)
    return chosen


def history_frame_count(version: dict[str, Any], observations: dict[str, dict[str, Any]]) -> int:
    frames = {
        str(observations[str(uid)].get("frame_uid"))
        for uid in version.get("member_observation_uids") or []
        if str(uid) in observations
    }
    return len(frames)


def mapper_complete(evidence_root: Path) -> bool:
    path = evidence_root / "manifest.json"
    if not path.is_file():
        return False
    try:
        status = str(read_json(path).get("status") or "")
    except (json.JSONDecodeError, OSError):
        return False
    return "COMPLETED" in status or status in {"completed", "early_exit"}


def event_frame(association: dict[str, Any], observations: dict[str, dict[str, Any]]) -> int:
    observation = observations.get(str(association.get("obs_uid")))
    if observation is not None:
        return frame_index_from_uid(str(observation.get("frame_uid") or ""))
    return frame_index_from_uid(str(association.get("frame_uid") or ""))


def load_exp_state(evidence_root: Path) -> dict[str, Any]:
    exp_root = evidence_root.parent.resolve()
    frames_rows = read_jsonl(evidence_root / "frames.jsonl")
    observations_rows = read_jsonl(evidence_root / "observations.jsonl")
    associations = read_jsonl(evidence_root / "associations.jsonl")
    versions_rows = read_jsonl(evidence_root / "object_versions.jsonl")
    frames = {str(row["frame_uid"]): row for row in frames_rows}
    observations = {str(row["obs_uid"]): row for row in observations_rows if row.get("status") == "kept"}
    versions = {str(row["object_version_uid"]): row for row in versions_rows}
    latest_frame = max((frame_index_from_uid(uid) for uid in frames), default=-1)
    complete = mapper_complete(evidence_root)
    ready_frame = latest_frame if complete else latest_frame - 1
    return {
        "exp_root": exp_root,
        "frames": frames,
        "observations": observations,
        "associations": associations,
        "versions": versions,
        "latest_frame": latest_frame,
        "ready_frame": ready_frame,
        "complete": complete,
    }


def explicit_worklist(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    return read_jsonl(path)


def selected_cases(args: argparse.Namespace, state: dict[str, Any]) -> list[dict[str, Any]]:
    by_event = {str(row.get("event_uid")): row for row in state["associations"]}
    explicit = explicit_worklist(args.worklist)
    selected: list[dict[str, Any]] = []
    if explicit is not None:
        for item in explicit:
            event_uid = str(item.get("event_uid") or "")
            association = by_event.get(event_uid)
            if association is None:
                continue
            if event_frame(association, state["observations"]) > state["ready_frame"]:
                continue
            selected.append({
                "case_uid": str(item.get("case_uid") or stable_case_uid(args.scene, event_uid)),
                "association": association,
                "sampling": {key: value for key, value in item.items() if key not in {"case_uid", "event_uid"}},
            })
    else:
        if not 0 < args.sample_probability <= 1:
            raise ValueError("--sample-probability must be in (0, 1]")
        for association in state["associations"]:
            if association.get("decision") != "MERGE_TO_OBJECT":
                continue
            event_uid = str(association.get("event_uid") or "")
            if event_frame(association, state["observations"]) > state["ready_frame"]:
                continue
            observation = state["observations"].get(str(association.get("obs_uid")))
            version = state["versions"].get(str(association.get("target_object_version_before")))
            if observation is None or version is None:
                continue
            history_obs = [state["observations"].get(str(uid)) for uid in version.get("member_observation_uids") or []]
            history_obs = [row for row in history_obs if row is not None]
            history_frames = {str(row.get("frame_uid")) for row in history_obs}
            if len(history_obs) < args.min_history_observations or len(history_frames) < args.min_history_frames:
                continue
            if stable_fraction(args.sample_seed, event_uid) >= args.sample_probability:
                continue
            selected.append({
                "case_uid": stable_case_uid(args.scene, event_uid),
                "association": association,
                "sampling": {
                    "sample_kind": "PREVALENCE_BERNOULLI",
                    "sample_probability": args.sample_probability,
                    "sample_seed": args.sample_seed,
                },
            })
    selected.sort(key=lambda item: item["case_uid"])
    return selected[: args.max_cases] if args.max_cases else selected


def build_case(args: argparse.Namespace, state: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    association = item["association"]
    case_uid = item["case_uid"]
    case_dir = args.output_root / "cases" / case_uid
    case_dir.mkdir(parents=True, exist_ok=True)
    public_path = case_dir / "case_public.json"
    private_path = case_dir / "case_private.json"
    if public_path.is_file() and private_path.is_file():
        public = read_json(public_path)
        return {
            "case_uid": case_uid,
            "event_uid": association.get("event_uid"),
            "case_dir": str(case_dir.resolve()),
            "source_frame": public.get("source_frame"),
            "repeat_of": item["sampling"].get("repeat_of"),
            "sample_kind": item["sampling"].get("sample_kind"),
        }

    exp_root: Path = state["exp_root"]
    frames = state["frames"]
    observations = state["observations"]
    versions = state["versions"]
    observation = observations[str(association["obs_uid"])]
    frame = frames[str(observation["frame_uid"])]
    source_frame = frame.get("source_frame_id")
    current_context, current_crop = render_observation_assets(
        exp_root, observation, frames, case_dir, "current", (230, 66, 112)
    )
    current_points = sampled_points(load_points(exp_root, observation), 800, str(observation["obs_uid"]))

    similarity = load_similarity_row(exp_root, association)
    candidates = candidate_cards(association, similarity, versions, args.top_k)
    if not candidates:
        raise ValueError(f"no renderable candidates for {association.get('event_uid')}")
    target_uid = str(association.get("target_object_uid") or "")
    selected_code = next((row["code"] for row in candidates if row["object_uid"] == target_uid), None)
    if selected_code is None:
        raise ValueError(f"target missing from candidate cards for {association.get('event_uid')}")

    public_candidates = []
    private_candidates = []
    for candidate in candidates:
        code = candidate["code"]
        version = versions[str(candidate["object_version_uid"])]
        history_name = f"candidate_{code}_history.jpg"
        history_meta = render_history_sheet(
            exp_root,
            version,
            observations,
            frames,
            case_dir / history_name,
            args.history_views,
        )
        selected_history = select_history_observations(version, observations, args.history_views)
        history_points_parts = [load_points(exp_root, row) for row in selected_history]
        history_points = np.concatenate([part for part in history_points_parts if len(part)], axis=0) if any(len(part) for part in history_points_parts) else np.empty((0, 3), dtype=float)
        history_points = sampled_points(history_points, 1200, str(candidate["object_version_uid"]))
        pcd_name = f"candidate_{code}_3d.jpg"
        render_3d_comparison(current_points, history_points, case_dir / pcd_name, f"Candidate {code}")
        public_candidates.append({
            "code": code,
            "history_asset": history_name,
            "pcd_asset": pcd_name,
            "history_observation_count": len(version.get("member_observation_uids") or []),
            "history_frame_count": history_frame_count(version, observations),
            "displayed_history_count": len(history_meta),
        })
        private_candidates.append({
            **candidate,
            "history_displayed": history_meta,
            "history_member_observation_uids": list(version.get("member_observation_uids") or []),
        })

    asset_names = [current_context, current_crop]
    for candidate in public_candidates:
        asset_names.extend([candidate["history_asset"], candidate["pcd_asset"]])
    asset_hashes = {name: sha256_file(case_dir / name) for name in sorted(asset_names)}
    public = {
        "schema_version": SCHEMA_VERSION,
        "scene": args.scene,
        "case_uid": case_uid,
        "event_uid": association.get("event_uid"),
        "event_frame_idx": event_frame(association, observations),
        "source_frame": source_frame,
        "captured_when_mapper_latest_frame": state["latest_frame"],
        "current": {
            "context_asset": current_context,
            "crop_asset": current_crop,
            "mask_area": observation.get("processed_mask_area"),
            "valid_depth_ratio": observation.get("valid_depth_ratio"),
            "stored_point_count": observation.get("pcd_stored_points"),
        },
        "candidates": public_candidates,
        "displayed_asset_sha256": asset_hashes,
        "annotation_notice": "Mapper choice, UID, rank, score, auto GT and sample stratum are hidden until the blind identity judgement is saved.",
    }
    private = {
        "schema_version": SCHEMA_VERSION,
        "scene": args.scene,
        "case_uid": case_uid,
        "event_uid": association.get("event_uid"),
        "obs_uid": association.get("obs_uid"),
        "association_event": association,
        "selected_target_code": selected_code,
        "selected_target_uid": target_uid,
        "selected_target_version_before": association.get("target_object_version_before"),
        "candidates": private_candidates,
        "sampling": item["sampling"],
        "source_public_sha256": None,
    }
    write_json_atomic(public_path, public)
    private["source_public_sha256"] = sha256_file(public_path)
    write_json_atomic(private_path, private)
    return {
        "case_uid": case_uid,
        "event_uid": association.get("event_uid"),
        "case_dir": str(case_dir.resolve()),
        "source_frame": source_frame,
        "repeat_of": item["sampling"].get("repeat_of"),
        "sample_kind": item["sampling"].get("sample_kind"),
    }


def build_once(args: argparse.Namespace) -> dict[str, Any]:
    evidence_root = args.evidence_root.resolve()
    args.output_root = args.output_root.resolve()
    state = load_exp_state(evidence_root)
    cases = selected_cases(args, state)
    manifest_rows = []
    failures = []
    for item in cases:
        try:
            manifest_rows.append(build_case(args, state, item))
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            failures.append({
                "case_uid": item["case_uid"],
                "event_uid": item["association"].get("event_uid"),
                "error": f"{type(exc).__name__}: {exc}",
            })
    manifest_rows.sort(key=lambda row: row["case_uid"])
    write_jsonl_atomic(args.output_root / "worklist.jsonl", manifest_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if not failures else "READY_WITH_FAILURES",
        "scene": args.scene,
        "evidence_root": str(evidence_root),
        "evidence_manifest_sha256": sha256_file(evidence_root / "manifest.json") if (evidence_root / "manifest.json").is_file() else None,
        "mapper_complete": state["complete"],
        "mapper_latest_frame": state["latest_frame"],
        "ready_through_frame": state["ready_frame"],
        "case_count": len(manifest_rows),
        "failure_count": len(failures),
        "worklist_sha256": sha256_file(args.output_root / "worklist.jsonl"),
        "failures": failures,
    }
    write_json_atomic(args.output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    args.evidence_root = args.evidence_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    while True:
        manifest = build_once(args)
        print(json.dumps({key: manifest[key] for key in (
            "status", "scene", "mapper_complete", "mapper_latest_frame", "ready_through_frame", "case_count", "failure_count"
        )}, ensure_ascii=False), flush=True)
        if not args.follow or manifest["mapper_complete"]:
            return 0 if manifest["failure_count"] == 0 else 2
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())

