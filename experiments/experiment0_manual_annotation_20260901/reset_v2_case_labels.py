#!/usr/bin/env python3
"""Back up and unlock selected Experiment 0 v2 cases for re-annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--case-uid", action="append", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def main() -> int:
    args = parse_args()
    packet_root = args.packet_root.resolve()
    targets = sorted(set(args.case_uid))
    worklist = read_jsonl(packet_root / "worklist.jsonl")
    known = {str(row["case_uid"]) for row in worklist}
    unknown = sorted(set(targets) - known)
    if unknown:
        raise ValueError("unknown case_uid: " + ", ".join(unknown))

    labels_root = packet_root / "labels"
    draft_path = labels_root / "blind_drafts.jsonl"
    final_path = labels_root / "event_labels.jsonl"
    drafts = read_jsonl(draft_path)
    finals = read_jsonl(final_path)
    draft_ids = {str(row["case_uid"]) for row in drafts}
    missing_drafts = sorted(set(targets) - draft_ids)
    if missing_drafts:
        raise ValueError("cases are not currently blind-locked: " + ", ".join(missing_drafts))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256("\n".join(targets).encode()).hexdigest()[:8]
    backup_root = labels_root / "reset_backups" / f"{timestamp}_{suffix}"
    backup_root.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for path in (draft_path, final_path):
        if path.exists():
            source_hashes[path.name] = sha256_file(path)
            shutil.copy2(path, backup_root / path.name)

    kept_drafts = [row for row in drafts if str(row["case_uid"]) not in targets]
    kept_finals = [row for row in finals if str(row["case_uid"]) not in targets]
    write_jsonl_atomic(draft_path, kept_drafts)
    write_jsonl_atomic(final_path, kept_finals)

    manifest = {
        "schema_version": "experiment0-v2-label-reset/1.0",
        "created_at_utc": timestamp,
        "packet_root": str(packet_root),
        "case_uids": targets,
        "operation": "UNLOCK_BLIND_AND_FINAL_FOR_REANNOTATION",
        "before": {"blind_drafts": len(drafts), "event_labels": len(finals)},
        "after": {"blind_drafts": len(kept_drafts), "event_labels": len(kept_finals)},
        "removed": {
            "blind_drafts": len(drafts) - len(kept_drafts),
            "event_labels": len(finals) - len(kept_finals),
        },
        "source_sha256": source_hashes,
        "backup_root": str(backup_root),
    }
    (backup_root / "reset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
