#!/usr/bin/env python3
"""Build the immutable R2 adjudication overlay after the 15-case review."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from label_logic_v2 import derive_routing_label, validate_blind_label, validate_final_label


DECISIONS: dict[str, dict[str, Any]] = {
    "v2_r2_007": {
        "expected_quality": "BACKGROUND_OR_FRAGMENT",
        "expected_matches": ["B"],
        "blind_overrides": {
            "observation_quality": "BORDERLINE_SINGLE_INSTANCE",
            "physical_instance_note": "局部可见的沙发实例；与候选 B 的位置、几何和历史一致",
        },
        "final_overrides": {
            "notes": "专家裁决：局部可见仍是稳定物理实例，不属于背景碎片",
        },
        "reason_codes": ["PARTIAL_VISIBILITY_NOT_BACKGROUND"],
    },
    "v2_r2_008": {
        "expected_quality": "BACKGROUND_OR_FRAGMENT",
        "expected_matches": ["B"],
        "blind_overrides": {
            "observation_quality": "BORDERLINE_SINGLE_INSTANCE",
            "physical_instance_note": "局部可见的灯实例；与候选 B 的位置和多帧历史一致",
        },
        "final_overrides": {
            "notes": "专家裁决：极小局部观察仍可稳定绑定到灯实例，不属于背景碎片",
        },
        "reason_codes": ["PARTIAL_VISIBILITY_NOT_BACKGROUND"],
    },
    "v2_r2_011": {
        "expected_quality": "CLEAN_SINGLE_INSTANCE",
        "expected_matches": ["A", "D"],
        "blind_overrides": {
            "matching_candidate_codes": ["NONE_SHOWN"],
            "physical_instance_note": "当前局部桌面与展示的同类桌子空间位置不同",
        },
        "final_overrides": {
            "full_map_status": "NO_MATCHING_NODE_EXISTS",
            "outside_matching_node_uids": [],
            "notes": "专家裁决：同类和相似外观不能覆盖 3D 空间不连续证据",
        },
        "reason_codes": ["SAME_CATEGORY_DIFFERENT_INSTANCE", "SPATIAL_DISCONTINUITY"],
    },
    "v2_r2_013": {
        "expected_quality": "CLEAN_SINGLE_INSTANCE",
        "expected_matches": ["B", "C", "D", "E"],
        "blind_overrides": {
            "matching_candidate_codes": ["NONE_SHOWN"],
            "physical_instance_note": "窗户中央的独立百叶面板；相邻面板不是同一物理实例",
        },
        "final_overrides": {
            "full_map_status": "NO_MATCHING_NODE_EXISTS",
            "outside_matching_node_uids": [],
            "notes": "专家裁决：同一窗户系统中的相邻百叶面板仍是不同实例",
        },
        "reason_codes": ["SAME_SYSTEM_DIFFERENT_INSTANCE"],
    },
    "v2_r2_002": {
        "expected_quality": "CLEAN_SINGLE_INSTANCE",
        "expected_matches": ["B", "C", "D", "E"],
        "blind_overrides": {
            "matching_candidate_codes": ["NONE_SHOWN"],
            "physical_instance_note": "窗户中央的独立百叶面板；相邻面板不是同一物理实例",
        },
        "final_overrides": {
            "full_map_status": "NO_MATCHING_NODE_EXISTS",
            "outside_matching_node_uids": [],
            "notes": "专家裁决：v2_r2_013 的隐藏重复，采用同一身份裁决",
        },
        "reason_codes": ["SAME_SYSTEM_DIFFERENT_INSTANCE", "HIDDEN_REPEAT_CONSISTENCY"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def adjudicate_label(
    label: dict[str, Any], decision: dict[str, Any], candidate_codes: set[str]
) -> dict[str, Any]:
    if label["blind"]["observation_quality"] != decision["expected_quality"]:
        raise ValueError(f"{label['case_uid']}: unexpected source quality")
    if sorted(label["blind"]["matching_candidate_codes"]) != sorted(
        decision["expected_matches"]
    ):
        raise ValueError(f"{label['case_uid']}: unexpected source matches")

    blind = copy.deepcopy(label["blind"])
    blind.update(decision["blind_overrides"])
    blind = validate_blind_label(blind, candidate_codes)

    final = copy.deepcopy(label["final"])
    final.update(decision["final_overrides"])
    action = label["reveal"]["original_action_type"]
    final = validate_final_label(final, blind, action)
    derived = derive_routing_label(
        blind,
        final,
        action,
        label["reveal"]["original_target_code"],
    )
    return {"blind": blind, "final": final, "derived": derived}


def main() -> int:
    args = parse_args()
    packet_root = args.packet_root.resolve()
    output_root = args.output_root.resolve()
    labels_path = packet_root / "labels" / "event_labels.jsonl"
    labels = {row["case_uid"]: row for row in read_jsonl(labels_path)}
    if not set(DECISIONS).issubset(labels):
        raise ValueError("R2 source labels do not contain every adjudicated case")

    overlays: list[dict[str, Any]] = []
    for case_uid, decision in DECISIONS.items():
        label = labels[case_uid]
        public = json.loads(
            (packet_root / "cases" / case_uid / "case_public.json").read_text(
                encoding="utf-8"
            )
        )
        candidate_codes = {str(row["code"]) for row in public["candidates"]}
        corrected = adjudicate_label(label, decision, candidate_codes)
        overlays.append(
            {
                "schema_version": "experiment0-v2-r2-adjudication-overlay/1.0",
                "case_uid": case_uid,
                "event_uid": label["event_uid"],
                "source_label_sha256": stable_sha256(label),
                "source_blind": label["blind"],
                "source_final": label["final"],
                "source_derived": label["derived"],
                "reason_codes": decision["reason_codes"],
                "adjudication_basis": "HUMAN_VISUAL_REVIEW_PLUS_FROZEN_TMINUS_AUDIT",
                "adjudicated": corrected,
            }
        )

    overlays.sort(key=lambda row: row["case_uid"])
    overlay_path = output_root / "v2_r2_adjudication_overlay.jsonl"
    atomic_write(
        overlay_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in overlays),
    )

    corrected_labels = copy.deepcopy(labels)
    for row in overlays:
        corrected_labels[row["case_uid"]]["blind"] = row["adjudicated"]["blind"]
        corrected_labels[row["case_uid"]]["final"] = row["adjudicated"]["final"]
        corrected_labels[row["case_uid"]]["derived"] = row["adjudicated"]["derived"]
    route_counts = Counter(
        row["derived"]["routing_label"] for row in corrected_labels.values()
    )
    status_counts = Counter(
        row["derived"]["annotation_status"] for row in corrected_labels.values()
    )
    manifest = {
        "schema_version": "experiment0-v2-r2-adjudication-manifest/1.0",
        "source_packet_root": str(packet_root),
        "source_labels_sha256": file_sha256(labels_path),
        "overlay_count": len(overlays),
        "overlay_case_uids": sorted(DECISIONS),
        "overlay_sha256": file_sha256(overlay_path),
        "raw_labels_preserved": True,
        "corrected_counts": {
            "annotation_status": dict(sorted(status_counts.items())),
            "routing_label": dict(sorted(route_counts.items())),
        },
    }
    atomic_write(
        output_root / "v2_r2_adjudication_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
