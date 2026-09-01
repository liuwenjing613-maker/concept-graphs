#!/usr/bin/env python3
"""Integrity and blindness checks for Experiment 0 v2 trial packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "private",
    "gt_",
    "routing_label",
    "original_action",
    "decision",
    "target_uid",
    "score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_no}")
                rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_keys(value: Any) -> list[str]:
    keys = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key).lower())
            keys.extend(public_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(public_keys(child))
    return keys


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root = args.packet_root.resolve()
    manifest = read_json(root / "manifest.json")
    worklist_path = root / "worklist.jsonl"
    worklist = read_jsonl(worklist_path)
    errors = []
    action_counts = Counter()
    stratum_counts = Counter()
    case_data: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    if manifest.get("status") != "READY":
        errors.append("manifest status is not READY")
    if manifest.get("worklist_sha256") != sha256_file(worklist_path):
        errors.append("worklist hash mismatch")
    if int(manifest.get("case_count", -1)) != len(worklist):
        errors.append("manifest case_count mismatch")

    for row in worklist:
        case_uid = str(row.get("case_uid") or "")
        case_dir = Path(str(row.get("case_dir") or "")).resolve()
        if root not in case_dir.parents:
            errors.append(f"{case_uid}: case_dir escaped packet root")
            continue
        public_path = case_dir / "case_public.json"
        private_path = case_dir / "case_private.json"
        if not public_path.is_file() or not private_path.is_file():
            errors.append(f"{case_uid}: public/private case missing")
            continue
        public = read_json(public_path)
        private = read_json(private_path)
        case_data[case_uid] = (public, private)
        if public.get("case_uid") != case_uid or private.get("case_uid") != case_uid:
            errors.append(f"{case_uid}: case binding mismatch")
        if private.get("source_public_sha256") != sha256_file(public_path):
            errors.append(f"{case_uid}: public hash binding mismatch")
        if public.get("tminus_snapshot_sha256") != (
            private.get("sampling") or {}
        ).get("tminus_snapshot_sha256", public.get("tminus_snapshot_sha256")):
            errors.append(f"{case_uid}: t^- snapshot mismatch")

        for key in public_keys(public):
            if any(fragment in key for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS):
                errors.append(f"{case_uid}: forbidden public key {key}")
        for name, expected in (public.get("displayed_asset_sha256") or {}).items():
            asset = (case_dir / str(name)).resolve()
            if case_dir not in asset.parents or not asset.is_file():
                errors.append(f"{case_uid}: missing asset {name}")
            elif sha256_file(asset) != expected:
                errors.append(f"{case_uid}: asset hash mismatch {name}")

        action = str(private.get("original_action_type") or "")
        action_counts[action] += 1
        if action == "ATTACH_EXISTING" and not private.get("original_target_code"):
            errors.append(f"{case_uid}: ATTACH missing target code")
        if action == "NEW" and private.get("original_target_code") is not None:
            errors.append(f"{case_uid}: NEW unexpectedly has target code")
        if action not in {"ATTACH_EXISTING", "NEW"}:
            errors.append(f"{case_uid}: invalid original action")

        audit = private.get("private_full_map_gt_audit") or {}
        if not audit.get("all_legal_candidates_displayed"):
            errors.append(f"{case_uid}: legal candidate coverage false")
        candidate_codes = {str(candidate["code"]) for candidate in private["candidates"]}
        if not set(audit.get("legal_candidate_codes") or []).issubset(candidate_codes):
            errors.append(f"{case_uid}: legal candidate code not displayed")
        stratum = str((private.get("sampling") or {}).get("private_auto_routing_label"))
        stratum_counts[stratum] += 1

    repeat_checks = 0
    for row in worklist:
        repeat_of = row.get("repeat_of")
        if not repeat_of:
            continue
        case_uid = str(row["case_uid"])
        if case_uid not in case_data or str(repeat_of) not in case_data:
            errors.append(f"{case_uid}: repeat source missing")
            continue
        public, _ = case_data[case_uid]
        source_public, _ = case_data[str(repeat_of)]
        for key in ("event_uid", "current", "candidates", "displayed_asset_sha256"):
            if public.get(key) != source_public.get(key):
                errors.append(f"{case_uid}: hidden repeat differs in {key}")
        repeat_checks += 1

    report = {
        "schema_version": "experiment0-v2-packet-integrity/1.0",
        "status": "PASS" if not errors else "FAIL",
        "packet_root": str(root),
        "case_count": len(worklist),
        "checked_case_count": len(case_data),
        "repeat_checks": repeat_checks,
        "action_counts": dict(sorted(action_counts.items())),
        "private_stratum_counts": dict(sorted(stratum_counts.items())),
        "error_count": len(errors),
        "errors": errors,
    }
    write_json_atomic(root / "integrity_v2.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
