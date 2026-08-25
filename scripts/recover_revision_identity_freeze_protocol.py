#!/usr/bin/env python3
"""Recover hash-validated complete cases from an interrupted freeze batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from conceptgraph.revision.auto_constraints import forbidden_inference_paths
from conceptgraph.revision.evidence_split import sha256_file


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


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-root", required=True, type=Path)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only-case", action="append", default=[])
    args = parser.parse_args()

    partial_root = args.partial_root.resolve()
    manifest_path = args.case_manifest.resolve()
    manifest = _read(manifest_path)
    forbidden = forbidden_inference_paths(manifest)
    if forbidden:
        raise ValueError(
            "case manifest contains oracle-like fields: " + ", ".join(forbidden)
        )
    manifest_cases = {str(row["case_uid"]): row for row in manifest.get("cases") or ()}
    selected = set(str(item) for item in args.only_case)
    result_paths = sorted(partial_root.glob("*/case_result.frozen.json"))
    results = []
    for result_path in result_paths:
        result = _read(result_path)
        case_uid = str(result["case_uid"])
        if selected and case_uid not in selected:
            continue
        if case_uid not in manifest_cases:
            raise ValueError(f"{case_uid}: not present in case manifest")
        if result.get("gold_loaded") is not False:
            raise ValueError(f"{case_uid}: result loaded gold")
        if result.get("human_verdict_loaded") is not False:
            raise ValueError(f"{case_uid}: result loaded a human verdict")
        if result.get("status") != "FROZEN_PENDING_OUTCOME_CRITIC":
            raise ValueError(f"{case_uid}: incomplete result status")
        for request in result.get("critic_requests") or ():
            request_path = Path(str(request["path"])).resolve()
            if sha256_file(request_path) != str(request["sha256"]):
                raise ValueError(f"{case_uid}: critic request hash drift")
        results.append((result_path, result))
    recovered = {str(result["case_uid"]) for _, result in results}
    if selected and recovered != selected:
        raise ValueError(
            f"not all requested cases were complete: requested={sorted(selected)}, "
            f"recovered={sorted(recovered)}"
        )
    if not results:
        raise ValueError("no complete frozen cases found")

    request_rows = [
        request
        for _, result in results
        for request in result.get("critic_requests") or ()
    ]
    protocol = {
        "schema_version": "1.0.0",
        "role": "DEVELOPMENT_SHADOW_NOT_PRODUCTION_COMMIT",
        "recovery_status": "HASH_VALIDATED_COMPLETE_CASES_FROM_INTERRUPTED_BATCH",
        "partial_root": str(partial_root),
        "case_manifest_path": str(manifest_path),
        "case_manifest_sha256": sha256_file(manifest_path),
        "case_count": len(results),
        "request_count": len(request_rows),
        "cases": [
            {
                "case_uid": str(result["case_uid"]),
                "scene_id": str(result["scene_id"]),
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
            }
            for result_path, result in results
        ],
        "critic_requests": request_rows,
        "runtime_human_or_gold_loaded": False,
        "candidate_source": "FINITE_EXECUTOR_CAPABILITIES",
        "candidate_state_deduplication": "ENTITY_ID_INVARIANT_PARTITION_HASH",
        "production_commit_permitted": False,
        "calibration_status": "NOT_YET_FIT",
        "semantic_threshold_count": 0,
        "protocol_uid": "recovered_freeze_protocol_" + _sha256_json(request_rows)[:20],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    _write(output, protocol)
    print(
        json.dumps(
            {
                "status": "PASS",
                "case_count": len(results),
                "request_count": len(request_rows),
                "output": str(output),
                "recovered_case_uids": sorted(recovered),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
