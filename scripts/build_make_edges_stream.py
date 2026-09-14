from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import getpass
import gzip
import hashlib
import io
import json
import pickle
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import supervision as sv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.utils.general_utils import (  # noqa: E402
    ObjectClasses,
    annotate_for_vlm,
    filter_detections,
)
from conceptgraph.utils.vlm import system_prompt_only_top  # noqa: E402


_RELATIONS = {"on top of", "under"}
_LABEL_ID = re.compile(r"(?:object\s*)?(\d+)\s*$", re.IGNORECASE)
_PREPROCESS_STDOUT_LOCK = threading.Lock()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _load_frame_ids(path: Path | None, detection_root: Path) -> list[str]:
    if path is None:
        return sorted(item.name for item in detection_root.iterdir() if item.is_dir())
    result = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame_id = str(row["source_frame_id"])
            if frame_id in seen:
                raise ValueError(f"duplicate source frame id: {frame_id}")
            seen.add(frame_id)
            result.append(frame_id)
    return result


def _semantic_detection_digest(gobs: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("xyxy", "confidence", "class_id", "mask"):
        value = np.ascontiguousarray(gobs[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.tobytes())
    digest.update(_canonical_json(list(gobs["detection_class_labels"])))
    digest.update(_canonical_json(list(gobs["classes"])))
    return digest.hexdigest()


def _load_detection(root: Path, frame_id: str) -> dict[str, Any]:
    path = root / frame_id
    if not path.is_dir():
        raise FileNotFoundError(f"missing detection frame: {path}")
    result = {}
    for key in ("xyxy", "confidence", "class_id", "mask"):
        with np.load(path / f"{key}.npz", allow_pickle=False) as data:
            result[key] = data["arr_0"]
    for key in ("detection_class_labels", "classes"):
        with gzip.open(path / f"{key}.pkl.gz", "rb") as handle:
            result[key] = pickle.load(handle)
    return result


def _parse_label_id(value: Any) -> str:
    match = _LABEL_ID.search(str(value).strip())
    if not match:
        raise ValueError(f"invalid edge endpoint label: {value!r}")
    return str(int(match.group(1)))


def _parse_edges_strict(text: str, allowed_ids: set[str]) -> list[list[str]]:
    flattened = text.replace("\n", " ").strip()
    start = flattened.find("[")
    end = flattened.rfind("]")
    if start < 0 or end < start:
        raise ValueError("response does not contain a list")
    parsed = ast.literal_eval(flattened[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("response payload is not a list")
    result = []
    seen = set()
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"malformed relation tuple: {item!r}")
        source = _parse_label_id(item[0])
        relation = str(item[1]).strip().lower()
        target = _parse_label_id(item[2])
        if relation not in _RELATIONS:
            raise ValueError(f"unsupported relation type: {relation!r}")
        if source not in allowed_ids or target not in allowed_ids:
            raise ValueError(f"relation endpoint not in input labels: {item!r}")
        if source == target:
            continue
        key = (source, relation, target)
        if key not in seen:
            seen.add(key)
            result.append(list(key))
    return result


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _request_edges(
    *,
    client: OpenAI,
    model: str,
    image_path: Path,
    label_list: list[str],
    max_attempts: int,
) -> dict[str, Any]:
    image_b64 = _encode_image(image_path)
    user_query = (
        "Here is the list of labels for the annotations of the objects in the "
        f"image: {label_list}. Please describe the spatial relationships between "
        "the objects in the image."
    )
    messages = [
        {"role": "system", "content": system_prompt_only_top},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            ],
        },
        {"role": "user", "content": user_query},
    ]
    last_error = None
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            latency_ms = (time.perf_counter() - started) * 1000.0
            raw = str(response.choices[0].message.content or "")
            allowed = {item.split(":", 1)[0].strip() for item in label_list}
            edges = _parse_edges_strict(raw, allowed)
            usage = getattr(response, "usage", None)
            return {
                "edges": edges,
                "raw_response": raw,
                "actual_model": str(getattr(response, "model", "") or model),
                "latency_ms": latency_ms,
                "attempts": attempt,
                "usage": {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                },
            }
        except Exception as exc:  # API and strict schema failures are retriable.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                time.sleep(min(8.0, float(2 ** (attempt - 1))))
    raise RuntimeError(last_error or "edge request failed")


def _prepare_frame(
    *,
    frame_id: str,
    detection_root: Path,
    image_root: Path,
    image_suffix: str,
    object_classes: ObjectClasses,
    image_output: Path,
) -> dict[str, Any]:
    gobs = _load_detection(detection_root, frame_id)
    image_path = image_root / f"{frame_id}{image_suffix}"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot load RGB frame: {image_path}")
    raw_labels = list(gobs["detection_class_labels"])
    detections = sv.Detections(
        xyxy=np.asarray(gobs["xyxy"]),
        confidence=np.asarray(gobs["confidence"]),
        class_id=np.asarray(gobs["class_id"]),
        mask=np.asarray(gobs["mask"]),
    )
    image_output.parent.mkdir(parents=True, exist_ok=True)
    # redirect_stdout is process-global, so serialize the short preprocessing
    # section while the API requests remain fully parallel.
    with _PREPROCESS_STDOUT_LOCK, contextlib.redirect_stdout(io.StringIO()):
        filtered, labels = filter_detections(
            image=image,
            detections=detections,
            classes=object_classes,
            top_x_detections=150000,
            confidence_threshold=0.00001,
            given_labels=raw_labels,
        )
        annotated, _ = annotate_for_vlm(
            image, filtered, object_classes, labels, save_path=str(image_output)
        )
    if not image_output.is_file():
        if not cv2.imwrite(str(image_output), annotated):
            raise RuntimeError(f"failed to save annotated image: {image_output}")
    label_list = []
    for label in labels:
        label_num = str(label).split(" ")[-1]
        label_name = re.sub(r"\s*\d+$", "", str(label)).strip()
        label_list.append(f"{label_num}: {label_name}")
    return {
        "frame_id": frame_id,
        "source_image": str(image_path),
        "annotated_image": str(image_output),
        "raw_detection_count": len(raw_labels),
        "vlm_detection_count": len(labels),
        "input_labels": label_list,
        "detection_digest": _semantic_detection_digest(gobs),
        "image_sha256": _sha256_file(image_output),
    }


def _cached_prediction(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = (
        value.get("status") == "PASS"
        and value.get("detection_digest") == expected["detection_digest"]
        and value.get("image_sha256") == expected["image_sha256"]
        and value.get("input_labels") == expected["input_labels"]
    )
    return value if required else None


def _worker(
    *,
    slot: int,
    api_key: str,
    tasks: list[str],
    args: argparse.Namespace,
    object_classes: ObjectClasses,
    lock: threading.Lock,
) -> list[dict[str, Any]]:
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    rows = []
    for frame_id in tasks:
        prediction_path = args.output_root / "predictions" / f"{frame_id}.json"
        image_path = args.output_root / "images" / f"{frame_id}_annotated_for_vlm.jpg"
        prepared = _prepare_frame(
            frame_id=frame_id,
            detection_root=args.detection_root,
            image_root=args.image_root,
            image_suffix=args.image_suffix,
            object_classes=object_classes,
            image_output=image_path,
        )
        cached = _cached_prediction(prediction_path, prepared)
        if cached is not None:
            rows.append(cached)
            with lock:
                print(f"CACHE slot={slot} frame={frame_id}", flush=True)
            continue
        started_at = time.time()
        response = _request_edges(
            client=client,
            model=args.model,
            image_path=image_path,
            label_list=prepared["input_labels"],
            max_attempts=args.max_attempts,
        )
        row = {
            "schema_version": "0.1.0",
            "status": "PASS",
            "source_frame_id": frame_id,
            "source_frame_number": int(re.search(r"(\d+)$", frame_id).group(1)),
            "key_slot": slot,
            "requested_model": args.model,
            "started_at_unix": started_at,
            **prepared,
            **response,
        }
        row["response_sha256"] = _sha256_bytes(
            str(row["raw_response"]).encode("utf-8")
        )
        _write_json_atomic(prediction_path, row)
        rows.append(row)
        with lock:
            print(
                f"DONE slot={slot} frame={frame_id} edges={len(row['edges'])} "
                f"attempts={row['attempts']} latency_ms={row['latency_ms']:.1f}",
                flush=True,
            )
    return rows


def _parity_audit(
    source_root: Path, parity_root: Path | None, frame_ids: Iterable[str]
) -> dict[str, Any]:
    if parity_root is None:
        return {"performed": False, "reason": "no parity detection root supplied"}
    compared = equal = missing = 0
    mismatches = []
    for frame_id in frame_ids:
        compared += 1
        try:
            first = _semantic_detection_digest(_load_detection(source_root, frame_id))
            second = _semantic_detection_digest(_load_detection(parity_root, frame_id))
        except FileNotFoundError:
            missing += 1
            if len(mismatches) < 20:
                mismatches.append({"frame": frame_id, "reason": "missing"})
            continue
        if first == second:
            equal += 1
        elif len(mismatches) < 20:
            mismatches.append(
                {"frame": frame_id, "source_digest": first, "parity_digest": second}
            )
    return {
        "performed": True,
        "source_root": str(source_root),
        "parity_root": str(parity_root),
        "frames_compared": compared,
        "frames_equal": equal,
        "missing_frames": missing,
        "pass": equal == compared and missing == 0,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an immutable ali-dev make_edges-compatible frame stream"
    )
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--parity-detection-root", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-suffix", default=".jpg")
    parser.add_argument("--frames-jsonl", type=Path)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--key-count", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    args.detection_root = args.detection_root.resolve()
    args.image_root = args.image_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.key_count <= 0:
        raise ValueError("key-count must be positive")

    frame_ids = _load_frame_ids(args.frames_jsonl, args.detection_root)
    if not frame_ids:
        raise RuntimeError("no frames selected")
    keys = [
        getpass.getpass(f"API key slot {index + 1}/{args.key_count}: ")
        for index in range(args.key_count)
    ]
    if any(not item.strip() for item in keys) or len(set(keys)) != len(keys):
        raise ValueError("API keys must be non-empty and unique")

    object_classes = ObjectClasses(
        classes_file_path=args.classes_file,
        bg_classes=["wall", "floor", "ceiling"],
        skip_bg=False,
    )
    assignments = [frame_ids[index :: len(keys)] for index in range(len(keys))]
    lock = threading.Lock()
    rows = []
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        futures = [
            pool.submit(
                _worker,
                slot=slot,
                api_key=keys[slot],
                tasks=assignments[slot],
                args=args,
                object_classes=object_classes,
                lock=lock,
            )
            for slot in range(len(keys))
        ]
        for future in futures:
            rows.extend(future.result())
    rows.sort(key=lambda item: int(item["source_frame_number"]))

    frame_path = args.output_root / "frames.jsonl"
    temporary = frame_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            compact = {
                "schema_version": row["schema_version"],
                "source_frame_id": row["source_frame_id"],
                "source_frame_number": row["source_frame_number"],
                "input_labels": row["input_labels"],
                "edges": row["edges"],
                "detection_digest": row["detection_digest"],
                "image_sha256": row["image_sha256"],
                "response_sha256": row["response_sha256"],
                "key_slot": row["key_slot"],
                "actual_model": row["actual_model"],
            }
            handle.write(json.dumps(compact, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(frame_path)

    relation_counts = Counter(
        edge[1] for row in rows for edge in row.get("edges") or ()
    )
    parity = _parity_audit(
        args.detection_root, args.parity_detection_root, frame_ids
    )
    prompt_hash = _sha256_bytes(system_prompt_only_top.encode("utf-8"))
    input_digest = _sha256_bytes(
        "\n".join(row["detection_digest"] for row in rows).encode("ascii")
    )
    manifest = {
        "schema_version": "0.1.0",
        "status": "PASS" if len(rows) == len(frame_ids) and parity.get("pass", True) else "FAIL",
        "method": "ali-dev make_edges-compatible FRAME_EDGE stream",
        "semantics": (
            "Original ali-dev on-top-of/under prompt and frame-level label semantics; "
            "strict schema validation and resumable five-slot execution added."
        ),
        "detection_root": str(args.detection_root),
        "image_root": str(args.image_root),
        "requested_model": args.model,
        "base_url": args.base_url,
        "prompt_sha256": prompt_hash,
        "classes_file": str(args.classes_file.resolve()),
        "classes_file_sha256": _sha256_file(args.classes_file.resolve()),
        "input_detection_digest": input_digest,
        "frames_sha256": _sha256_file(frame_path),
        "frame_count": len(rows),
        "nonempty_frame_count": sum(bool(row.get("edges")) for row in rows),
        "input_edge_observations": sum(len(row.get("edges") or ()) for row in rows),
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "actual_model_counts": dict(
            sorted(Counter(row["actual_model"] for row in rows).items())
        ),
        "key_slot_counts": dict(
            sorted(Counter(str(row["key_slot"]) for row in rows).items())
        ),
        "usage": {
            key: sum(int(row.get("usage", {}).get(key, 0)) for row in rows)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "latency_ms": {
            "mean": float(np.mean([row["latency_ms"] for row in rows])),
            "median": float(np.median([row["latency_ms"] for row in rows])),
            "p95": float(np.percentile([row["latency_ms"] for row in rows], 95)),
            "max": float(max(row["latency_ms"] for row in rows)),
        },
        "parity": parity,
        "api_keys_persisted": False,
    }
    _write_json_atomic(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
