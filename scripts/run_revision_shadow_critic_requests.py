#!/usr/bin/env python3
"""Run frozen action-blind shadow-critic requests with ephemeral API keys."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from conceptgraph.revision.evidence_split import sha256_file
from conceptgraph.revision.shadow_critic import (
    OpenAICompatibleShadowCriticClient,
    ShadowCriticEvidence,
)


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


def _validate_request(
    manifest_row: Mapping[str, Any],
) -> tuple[dict[str, Any], ShadowCriticEvidence]:
    path = Path(str(manifest_row["path"])).resolve()
    if sha256_file(path) != str(manifest_row["sha256"]):
        raise ValueError(f"frozen request hash drift: {path}")
    request = _read(path)
    if str(request["request_uid"]) != str(manifest_row["request_uid"]):
        raise ValueError(f"request UID mismatch: {path}")
    image_paths = []
    image_manifest = []
    for image in request.get("images") or ():
        image_path = Path(str(image["path"])).resolve()
        if sha256_file(image_path) != str(image["sha256"]):
            raise ValueError(f"critic image hash drift: {image_path}")
        image_paths.append(image_path)
        image_manifest.append(
            {
                "evidence_id": str(image["evidence_id"]),
                "frame_index": int(image["frame_index"]),
                "class_name": str(image.get("class_name") or "unknown"),
                "sha256": str(image["sha256"]),
            }
        )
    evidence = ShadowCriticEvidence(
        incident_uid=str(request["case_uid"]),
        prompt=str(request["prompt"]),
        image_paths=tuple(image_paths),
        image_manifest=tuple(image_manifest),
        allowed_state_ids=tuple(str(item) for item in request["allowed_state_ids"]),
    )
    return request, evidence


def _run_one(
    *,
    manifest_row: Mapping[str, Any],
    api_key: str,
    credential_slot: int,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    request, evidence = _validate_request(manifest_row)
    try:
        response = OpenAICompatibleShadowCriticClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
        ).complete(evidence)
        return {
            "request_uid": str(request["request_uid"]),
            "case_uid": str(request["case_uid"]),
            "pair_uid": str(request["pair_uid"]),
            "order_index": int(request["order_index"]),
            "request_sha256": str(manifest_row["sha256"]),
            "credential_slot": credential_slot,
            "status": "PASS",
            "response": response,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
        }
    except Exception as exc:
        return {
            "request_uid": str(request["request_uid"]),
            "case_uid": str(request["case_uid"]),
            "pair_uid": str(request["pair_uid"]),
            "order_index": int(request["order_index"]),
            "request_sha256": str(manifest_row["sha256"]),
            "credential_slot": credential_slot,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_ms": (time.perf_counter() - started) * 1000.0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--credential-slots", type=int, default=4)
    parser.add_argument("--only-case", action="append", default=[])
    args = parser.parse_args()

    protocol_path = args.freeze_protocol.resolve()
    protocol = _read(protocol_path)
    if protocol.get("runtime_human_or_gold_loaded") is not False:
        raise ValueError("freeze protocol did not pass runtime oracle isolation")
    requests = list(protocol.get("critic_requests") or ())
    selected = set(str(item) for item in args.only_case)
    if selected:
        requests = [row for row in requests if str(row.get("case_uid")) in selected]
    if not requests:
        raise ValueError("no critic requests selected")
    output_root = args.output_root.resolve()
    result_path = output_root / "critic_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    slot_count = min(max(1, args.credential_slots), len(requests))
    api_keys = [
        getpass.getpass(f"API key for ephemeral credential slot {index}: ")
        for index in range(slot_count)
    ]
    if any(not key for key in api_keys):
        raise ValueError("API keys must be non-empty")

    started = time.perf_counter()
    results = []
    for batch_start in range(0, len(requests), slot_count):
        batch = requests[batch_start : batch_start + slot_count]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    manifest_row=row,
                    api_key=api_keys[index],
                    credential_slot=index,
                    base_url=args.base_url,
                    model=args.model,
                ): str(row["request_uid"])
                for index, row in enumerate(batch)
            }
            for future in as_completed(futures):
                results.append(future.result())
    del api_keys
    results.sort(key=lambda row: str(row["request_uid"]))
    aggregate = {
        "schema_version": "1.0.0",
        "freeze_protocol_path": str(protocol_path),
        "freeze_protocol_sha256": sha256_file(protocol_path),
        "base_url_sha256": hashlib.sha256(args.base_url.encode()).hexdigest(),
        "requested_model": args.model,
        "request_count": len(requests),
        "pass_count": sum(row["status"] == "PASS" for row in results),
        "error_count": sum(row["status"] != "PASS" for row in results),
        "ephemeral_credential_slot_count": slot_count,
        "credentials_persisted": False,
        "total_wall_ms": (time.perf_counter() - started) * 1000.0,
        "results": results,
    }
    aggregate["status"] = (
        "PASS" if aggregate["pass_count"] == aggregate["request_count"] else "ERROR"
    )
    _write(result_path, aggregate)
    print(
        json.dumps(
            {
                key: aggregate[key]
                for key in (
                    "status",
                    "request_count",
                    "pass_count",
                    "error_count",
                    "total_wall_ms",
                )
            },
            indent=2,
        )
    )
    return 0 if aggregate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
