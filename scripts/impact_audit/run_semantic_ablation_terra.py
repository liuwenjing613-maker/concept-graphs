#!/usr/bin/env python3
"""Metrics-only Terra comparison of LLaVA captions and identical GPT views."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

BRANCHES = ("llava_caption_gpt_text", "gpt_vision")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def normalize_label(value: object) -> str:
    return " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def first_json(text: str) -> dict:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("response contains no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/png;base64," + encoded


def api_keys() -> list[str]:
    values = []
    for index in range(1, 33):
        value = os.environ.get(f"REL_API_KEY_{index}")
        if value:
            values.append(value)
    if not values:
        raise SystemExit("no temporary REL_API_KEY_1..N variables are set")
    return values


def system_prompt(labels: list[str]) -> str:
    return (
        "You classify one physical object from multiple observations. "
        "Choose exactly one canonical label from the allowed list. "
        "All views refer to the same graph node, but masks and crops may be noisy. "
        "Return JSON only with keys label, confidence, and reason. "
        "confidence must be between 0 and 1. Allowed labels: "
        + json.dumps(labels, ensure_ascii=False)
    )


def normalize_response(value: dict, allowed: set[str]) -> dict:
    label = normalize_label(value.get("label"))
    if label not in allowed:
        raise ValueError(f"label is outside frozen vocabulary: {label!r}")
    confidence = float(value.get("confidence"))
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    reason = str(value.get("reason") or "").strip()
    return {"label": label, "confidence": confidence, "reason": reason}


def request_one(
    *,
    branch: str,
    node: dict,
    captions: list[str],
    prompt: str,
    output: Path,
    base_url: str,
    model: str,
    api_key: str,
    key_slot: int,
    timeout: int,
    retries: int,
    max_completion_tokens: int,
    allowed: set[str],
    source_map_sha256: str,
    view_manifest_sha256: str,
) -> tuple[str, int, str, str | None]:
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            normalize_response(existing["response"], allowed)
            return branch, node["node_index"], "cached", None
        except Exception:
            pass
    if branch == "llava_caption_gpt_text":
        content: Any = (
            "Classify the object using only these LLaVA view captions:\n"
            + json.dumps(captions, ensure_ascii=False)
        )
    elif branch == "gpt_vision":
        content = [
            {
                "type": "text",
                "text": (
                    "Classify the central object using these exact ranked views. "
                    "Do not assume the existing detector tag is correct."
                ),
            }
        ]
        for view in node["views"]:
            path = Path(view["crop_path"])
            if sha256_file(path) != view["crop_sha256"]:
                return branch, node["node_index"], "failed", "crop hash mismatch"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri(path), "detail": "high"},
                }
            )
    else:
        raise ValueError(branch)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_completion_tokens,
        "stream": False,
        "store": False,
    }
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    last_error = None
    started_all = time.monotonic()
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=encoded,
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                envelope = json.loads(response.read())
            choices = envelope.get("choices") or []
            raw = choices[0].get("message", {}).get("content") if choices else None
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("empty response")
            normalized = normalize_response(first_json(raw), allowed)
            record = {
                "schema_version": "1.0.0",
                "branch": branch,
                "node_index": node["node_index"],
                "oracle_gt_id": node.get("oracle_gt_id"),
                "source_map_sha256": source_map_sha256,
                "view_manifest_sha256": view_manifest_sha256,
                "view_crop_sha256": [
                    item["crop_sha256"] for item in node["views"]
                ],
                "llava_captions_sha256": (
                    hashlib.sha256(
                        json.dumps(captions, ensure_ascii=False).encode("utf-8")
                    ).hexdigest()
                    if branch == "llava_caption_gpt_text"
                    else None
                ),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "model_requested": model,
                "model_returned": str(envelope.get("model") or model),
                "key_slot": key_slot,
                "attempt": attempt,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "total_elapsed_seconds": round(time.monotonic() - started_all, 3),
                "usage": envelope.get("usage") or {},
                "response": normalized,
            }
            atomic_json(output, record)
            return branch, node["node_index"], "completed", None
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    return branch, node["node_index"], "failed", last_error


def canonical_gt(label: object, mapping: dict) -> str:
    normalized = normalize_label(label)
    replica = {
        normalize_label(source): normalize_label(target)
        for source, target in mapping["Replica2VisualGenome"].items()
    }
    vocabulary = {normalize_label(item) for item in mapping["VisualGenome_list"]}
    if normalized in replica:
        return replica[normalized]
    if normalized in vocabulary:
        return normalized
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--llava-captions", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.codelink.chat/v1")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=100)
    args = parser.parse_args()

    started_wall = time.monotonic()
    source_map = args.source_map.resolve()
    manifest_path = args.view_manifest.resolve()
    captions_path = args.llava_captions.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_map_sha256"] != sha256_file(source_map):
        raise ValueError("view manifest does not bind to source map")
    caption_rows = json.loads(captions_path.read_text(encoding="utf-8"))
    captions = {int(row["id"]): list(row["captions"]) for row in caption_rows}
    for node in manifest["nodes"]:
        if node["observation_count"] >= manifest["protocol"]["min_views_per_node"]:
            if len(captions.get(node["node_index"], [])) != len(node["views"]):
                raise ValueError(
                    f"LLaVA/view count mismatch for node {node['node_index']}"
                )

    mapping = json.loads(args.label_mapping.resolve().read_text(encoding="utf-8"))
    labels = sorted({normalize_label(item) for item in mapping["VisualGenome_list"]})
    allowed = set(labels)
    prompt = system_prompt(labels)
    source_map_sha256 = sha256_file(source_map)
    view_manifest_sha256 = sha256_file(manifest_path)
    selected_nodes = [node for node in manifest["nodes"] if node["views"]]
    if args.max_nodes > 0:
        selected_nodes = selected_nodes[: args.max_nodes]
    jobs = [
        (branch, node)
        for branch in BRANCHES
        for node in selected_nodes
    ]
    keys = api_keys()
    workers = min(args.workers, len(keys))
    shards = [[] for _ in range(workers)]
    for index, job in enumerate(jobs):
        shards[index % workers].append(job)
    counts = Counter()
    errors = []
    lock = threading.Lock()

    def shard_runner(slot: int, key: str, shard: list[tuple[str, dict]]) -> None:
        for branch, node in shard:
            result = request_one(
                branch=branch,
                node=node,
                captions=captions.get(node["node_index"], []),
                prompt=prompt,
                output=output_root / "predictions" / branch
                / f"node{node['node_index']:04d}.json",
                base_url=args.base_url,
                model=args.model,
                api_key=key,
                key_slot=slot,
                timeout=args.timeout,
                retries=args.retries,
                max_completion_tokens=args.max_completion_tokens,
                allowed=allowed,
                source_map_sha256=source_map_sha256,
                view_manifest_sha256=view_manifest_sha256,
            )
            with lock:
                _, node_index, status, error = result
                counts[status] += 1
                done = sum(counts.values())
                print(
                    f"SEMANTIC {done}/{len(jobs)} branch={branch} "
                    f"node={node_index} status={status} slot={slot} "
                    f"error={error or '-'}",
                    flush=True,
                )
                if error:
                    errors.append(
                        {"branch": branch, "node_index": node_index, "error": error}
                    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(shard_runner, index + 1, keys[index], shards[index])
            for index in range(workers)
        ]
        for future in futures:
            future.result()
    if errors:
        # Preserve failures explicitly. Map construction below deterministically
        # falls back to the frozen detector label for an absent prediction.
        atomic_json(output_root / "errors.json", {"errors": errors})

    predictions = {}
    prediction_records = {}
    for branch in BRANCHES:
        predictions[branch] = {}
        prediction_records[branch] = {}
        for path in sorted((output_root / "predictions" / branch).glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            predictions[branch][int(record["node_index"])] = record["response"]
            prediction_records[branch][int(record["node_index"])] = record
    branch_rows = {}
    for branch in BRANCHES:
        correct_model = denominator_model = 0
        correct_applied = denominator_applied = 0
        fallback_nodes = 0
        for node in manifest["nodes"]:
            gt_label = canonical_gt(node.get("oracle_gt_label"), mapping)
            prediction = predictions[branch].get(node["node_index"])
            if not prediction:
                fallback_nodes += 1
            applied_label = normalize_label(
                prediction["label"] if prediction else node.get("detector_class_name")
            )
            if gt_label != "unknown" and node.get("oracle_gt_id") is not None:
                denominator_applied += 1
                correct_applied += int(applied_label == gt_label)
                if prediction:
                    denominator_model += 1
                    correct_model += int(prediction["label"] == gt_label)
        usage_totals = Counter()
        for record in prediction_records[branch].values():
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = record.get("usage", {}).get(key)
                if isinstance(value, (int, float)):
                    usage_totals[key] += value
        branch_rows[branch] = {
            "model_covered_accuracy": (
                correct_model / denominator_model if denominator_model else None
            ),
            "model_covered_correct": correct_model,
            "model_covered_denominator": denominator_model,
            "applied_map_accuracy_with_detector_fallback": (
                correct_applied / denominator_applied if denominator_applied else None
            ),
            "applied_map_correct": correct_applied,
            "applied_map_denominator": denominator_applied,
            "fallback_nodes": fallback_nodes,
            "prediction_count": len(predictions[branch]),
            "usage": dict(usage_totals),
        }
    summary = {
        "schema_version": "1.0.0",
        "source_map": str(source_map),
        "source_map_sha256": source_map_sha256,
        "view_manifest": str(manifest_path),
        "view_manifest_sha256": view_manifest_sha256,
        "llava_captions": str(captions_path),
        "llava_captions_sha256": sha256_file(captions_path),
        "model": args.model,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "jobs": len(jobs),
        "metrics_only": True,
        "max_completion_tokens": args.max_completion_tokens,
        "wall_seconds": round(time.monotonic() - started_wall, 3),
        "counts": dict(counts),
        "inference_failures": len(errors),
        "errors": errors,
        "branches": branch_rows,
    }
    summary_path = output_root / "semantic_ablation_summary.json"
    atomic_json(summary_path, summary)
    (output_root / "READY").write_text(
        sha256_file(summary_path) + "\n", encoding="utf-8"
    )
    print(json.dumps(branch_rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
