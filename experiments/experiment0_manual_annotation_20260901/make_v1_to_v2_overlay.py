#!/usr/bin/env python3
"""Create a non-destructive v1-to-v2 adjudication overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
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


def row_sha256(row: dict[str, Any]) -> str:
    encoded = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def main() -> int:
    args = parse_args()
    labels = read_jsonl(args.v1_labels)
    overlays = []
    for label in labels:
        case_uid = str(label.get("case_uid") or "")
        overlay = {
            "schema_version": "experiment0-v1-to-v2-adjudication-overlay/1.0",
            "v1_case_uid": case_uid,
            "event_uid": label.get("event_uid"),
            "v1_label_row_sha256": row_sha256(label),
            "raw_v1_preserved": True,
            "migration_status": "PRESERVED_NOT_AUTO_MIGRATED",
            "v2_routing_label": None,
            "observation_quality_override": None,
            "adjudication_group": None,
            "reason": (
                "v1 covered ATTACH-only false-attach logic and cannot establish the "
                "five-cell ATTACH/NEW v2 route without renewed review."
            ),
            "invalidated_v1_derived_fields": [
                "private_auto_selection_group",
                "private_auto_is_error",
                "derived.derived_status",
                "derived.derived_action",
                "derived.is_root_false_attach",
            ],
        }
        if case_uid in {"calibration_016", "calibration_024"}:
            overlay.update(
                {
                    "migration_status": "REQUIRES_IDENTITY_ADJUDICATION",
                    "adjudication_group": "calibration_016_024_same_event_disagreement",
                    "reason": (
                        "The hidden repeat disagreed on whether mapper target D is the "
                        "same physical instance; both original answers remain immutable."
                    ),
                }
            )
        elif case_uid == "calibration_020":
            overlay.update(
                {
                    "migration_status": "OBSERVATION_QUALITY_CORRECTED_ONLY",
                    "observation_quality_override": "BACKGROUND_OR_FRAGMENT",
                    "reason": (
                        "Corrected depth/pose-to-ReplicaSSG sidecar and visual audit identify "
                        "the mask as ceiling/background. No v2 routing label is imputed."
                    ),
                }
            )
        elif case_uid in {"calibration_017", "calibration_018", "calibration_019"}:
            overlay.update(
                {
                    "migration_status": "OBSERVATION_QUALITY_CONFIRMED_ONLY",
                    "reason": (
                        "The v1 observation-quality judgement agrees with corrected sidecar "
                        "audit; the routing label still requires v2 review."
                    ),
                }
            )
        overlays.append(overlay)

    overlays.sort(key=lambda row: row["v1_case_uid"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_root / "v1_to_v2_adjudication_overlay.jsonl"
    write_jsonl_atomic(overlay_path, overlays)
    manifest = {
        "schema_version": "experiment0-v1-to-v2-overlay-manifest/1.0",
        "status": "READY",
        "v1_labels": str(args.v1_labels.resolve()),
        "v1_labels_sha256": sha256_file(args.v1_labels),
        "v1_label_count": len(labels),
        "overlay_count": len(overlays),
        "overlay_sha256": sha256_file(overlay_path),
        "policy": (
            "Raw v1 labels are immutable. Overlay corrections never rewrite v1 and are "
            "not mixed with v2 agreement or routing-rate statistics."
        ),
    }
    write_json_atomic(args.output_root / "v1_to_v2_overlay_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
