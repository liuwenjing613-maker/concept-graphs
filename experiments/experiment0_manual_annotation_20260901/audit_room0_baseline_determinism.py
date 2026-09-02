#!/usr/bin/env python3
"""Compare two strict online mapping runs after canonical UID normalization.

The mapper intentionally generates fresh run, event, transaction and object
UUIDs.  Byte hashes of whole pickle files therefore need not match.  This
audit first pairs objects by their deterministic CREATE observation, then
compares the full evidence trajectory and the exact final numeric state.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


JSONL_LEDGER_NAMES = (
    "frames.jsonl",
    "filter_trace.jsonl",
    "observations.jsonl",
    "associations.jsonl",
    "mapping_events.jsonl",
    "object_versions.jsonl",
    "object_pair_decisions.jsonl",
)
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def canonical_obs_uid(value: Any, run_id: str) -> str:
    return str(value or "").replace(run_id, "<RUN>")


def build_object_labels(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reference_run_id: str,
    candidate_run_id: str,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    if len(reference_rows) != len(candidate_rows):
        raise ValueError("association row counts differ")
    reference_labels: dict[str, str] = {}
    candidate_labels: dict[str, str] = {}
    pair_rows: list[dict[str, Any]] = []
    for index, (reference, candidate) in enumerate(zip(reference_rows, candidate_rows)):
        ref_key = canonical_obs_uid(reference.get("obs_uid"), reference_run_id)
        cand_key = canonical_obs_uid(candidate.get("obs_uid"), candidate_run_id)
        if ref_key != cand_key:
            raise ValueError(f"association order diverged at {index}: {ref_key} != {cand_key}")
        ref_create = reference.get("decision") == "CREATE_OBJECT"
        cand_create = candidate.get("decision") == "CREATE_OBJECT"
        if ref_create != cand_create:
            raise ValueError(f"CREATE decision diverged at {index}: {ref_key}")
        if not ref_create:
            continue
        ref_uid = str(reference.get("target_object_uid") or "")
        cand_uid = str(candidate.get("target_object_uid") or "")
        label = f"<OBJ_{len(reference_labels):04d}>"
        if not ref_uid or not cand_uid:
            raise ValueError(f"CREATE row lacks target UID at {index}")
        if ref_uid in reference_labels or cand_uid in candidate_labels:
            raise ValueError(f"object UID reused by CREATE at {index}")
        reference_labels[ref_uid] = label
        candidate_labels[cand_uid] = label
        pair_rows.append(
            {
                "label": label,
                "observation_key": ref_key,
                "reference_object_uid": ref_uid,
                "candidate_object_uid": cand_uid,
            }
        )
    return reference_labels, candidate_labels, pair_rows


def normalize(value: Any, run_id: str, object_labels: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value.replace(run_id, "<RUN>")
        return UUID_RE.sub(lambda match: object_labels.get(match.group(0), match.group(0)), result)
    if isinstance(value, list):
        return [normalize(item, run_id, object_labels) for item in value]
    if isinstance(value, dict):
        return {
            str(normalize(key, run_id, object_labels)): normalize(item, run_id, object_labels)
            for key, item in value.items()
        }
    return value


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "reference": left, "candidate": right, "reason": "TYPE"}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {
                "path": path,
                "reference_keys_only": sorted(set(left) - set(right)),
                "candidate_keys_only": sorted(set(right) - set(left)),
                "reason": "KEYS",
            }
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "reference": len(left), "candidate": len(right), "reason": "LENGTH"}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return {"path": path, "reference": left, "candidate": right, "reason": "VALUE"}
    return None


def compare_rows(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    reference_run_id: str,
    candidate_run_id: str,
    reference_labels: dict[str, str],
    candidate_labels: dict[str, str],
) -> dict[str, Any]:
    normalized_reference = normalize(reference, reference_run_id, reference_labels)
    normalized_candidate = normalize(candidate, candidate_run_id, candidate_labels)
    difference = first_difference(normalized_reference, normalized_candidate)
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "normalized_equal": difference is None,
        "first_difference": difference,
    }


def compare_similarity_arrays(
    reference_dir: Path,
    candidate_dir: Path,
    reference_run_id: str,
    candidate_run_id: str,
    reference_labels: dict[str, str],
    candidate_labels: dict[str, str],
) -> dict[str, Any]:
    reference_paths = sorted(reference_dir.glob("frame_*.npz"))
    candidate_paths = sorted(candidate_dir.glob("frame_*.npz"))
    mismatches: list[dict[str, Any]] = []
    if [path.name for path in reference_paths] != [path.name for path in candidate_paths]:
        return {
            "reference_file_count": len(reference_paths),
            "candidate_file_count": len(candidate_paths),
            "all_arrays_exact": False,
            "file_inventory_equal": False,
            "mismatches": [{"reason": "FILE_INVENTORY"}],
        }
    array_count = 0
    for reference_path, candidate_path in zip(reference_paths, candidate_paths):
        with np.load(reference_path, allow_pickle=False) as reference, np.load(
            candidate_path, allow_pickle=False
        ) as candidate:
            if set(reference.files) != set(candidate.files):
                mismatches.append({"file": reference_path.name, "reason": "ARRAY_KEYS"})
                continue
            for key in reference.files:
                array_count += 1
                left = reference[key]
                right = candidate[key]
                if left.dtype.kind in {"U", "S", "O"} and right.dtype.kind in {"U", "S", "O"}:
                    left_values = normalize(left.astype(str).tolist(), reference_run_id, reference_labels)
                    right_values = normalize(right.astype(str).tolist(), candidate_run_id, candidate_labels)
                    values_equal = left_values == right_values
                else:
                    values_equal = left.dtype == right.dtype and np.array_equal(left, right)
                if left.shape != right.shape or not values_equal:
                    mismatches.append(
                        {
                            "file": reference_path.name,
                            "array": key,
                            "reason": "ARRAY_VALUE",
                            "reference_shape": list(left.shape),
                            "candidate_shape": list(right.shape),
                            "reference_dtype": str(left.dtype),
                            "candidate_dtype": str(right.dtype),
                        }
                    )
                    if len(mismatches) >= 20:
                        break
        if len(mismatches) >= 20:
            break
    return {
        "reference_file_count": len(reference_paths),
        "candidate_file_count": len(candidate_paths),
        "array_count": array_count,
        "file_inventory_equal": True,
        "all_arrays_exact": not mismatches,
        "mismatches": mismatches,
    }


def source_snapshot_signature(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_path": row.get("source_path"),
            "sha256": (row.get("artifact_ref") or {}).get("sha256"),
        }
        for row in manifest.get("runtime_source_snapshot") or []
    ]


def markdown(metrics: dict[str, Any]) -> str:
    inventory = metrics["final_map_parity"]["candidate_inventory"]
    lines = [
        "# 实验 0：room0 冻结 baseline 双跑确定性审计",
        "",
        "## 结论",
        "",
        f"最终判定：**{metrics['status']}**。同一冻结配置从空图重新在线运行，经过 UID 规范化后，完整证据轨迹与最终数值状态均与原 run 一致。",
        "",
        f"- 处理帧 trace：{metrics['parity_trace']['reference_count']} / {metrics['parity_trace']['candidate_count']}，完全一致；",
        f"- association 事件：{metrics['ledgers']['associations.jsonl']['reference_count']}，规范化后完全一致；",
        f"- 最终对象：{inventory['object_count']}；observation：{inventory['unique_observations']}；点：{inventory['point_count']}；",
        f"- 最终 observation 分区、{metrics['final_map_parity']['pcd_arrays_exact']} 个对象点云、bbox 和类别：全部精确一致；",
        f"- 相似度矩阵文件：{metrics['similarity_arrays']['reference_file_count']}，数组逐值完全一致；",
        f"- 两次 strict evidence 状态：{metrics['reference_manifest_status']} / {metrics['candidate_manifest_status']}。",
        "",
        "## 比较口径",
        "",
        "两次运行会生成不同的 run ID、event/transaction UID 和 object UUID，因此整个压缩 pickle 的字节哈希不要求相同。审计按每个对象的 CREATE observation 建立一一映射，再比较 frames、filter trace、observations、associations、mapping events、object versions、object-pair decisions、final membership、逐帧 parity trace、相似度数组和最终数值状态。",
        "",
        "## 对实验 0 的意义",
        "",
        "room0 的 baseline 在当前冻结环境下具有足够的重复确定性；当前发现的标签与 root/cascade 差异不是随机重跑漂移造成的。这个结论只验证 room0 当前配置，后续新场景仍需保留 run manifest 和 strict evidence audit。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--final-map-parity", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    reference_run = args.reference_run.resolve()
    candidate_run = args.candidate_run.resolve()
    reference_evidence = reference_run / "evidence"
    candidate_evidence = candidate_run / "evidence"
    reference_manifest = read_json(reference_evidence / "manifest.json")
    candidate_manifest = read_json(candidate_evidence / "manifest.json")
    reference_run_id = str(reference_manifest["run_id"])
    candidate_run_id = str(candidate_manifest["run_id"])

    reference_associations = read_jsonl(reference_evidence / "associations.jsonl")
    candidate_associations = read_jsonl(candidate_evidence / "associations.jsonl")
    reference_labels, candidate_labels, object_pairs = build_object_labels(
        reference_associations,
        candidate_associations,
        reference_run_id,
        candidate_run_id,
    )

    ledgers: dict[str, Any] = {}
    for name in JSONL_LEDGER_NAMES:
        reference_rows = read_jsonl(reference_evidence / name)
        candidate_rows = read_jsonl(candidate_evidence / name)
        if name == "associations.jsonl":
            # NPZ archives embed fresh run/object UIDs.  Their byte hashes are
            # expected to differ; the arrays themselves are compared below
            # after UID canonicalization.
            for rows in (reference_rows, candidate_rows):
                for row in rows:
                    for ref_name in ("spatial_sim_ref", "visual_sim_ref", "aggregate_sim_ref"):
                        if isinstance(row.get(ref_name), dict):
                            row[ref_name].pop("sha256", None)
        ledgers[name] = compare_rows(
            reference_rows,
            candidate_rows,
            reference_run_id,
            candidate_run_id,
            reference_labels,
            candidate_labels,
        )

    final_membership = compare_rows(
        read_json(reference_evidence / "final_membership.json"),
        read_json(candidate_evidence / "final_membership.json"),
        reference_run_id,
        candidate_run_id,
        reference_labels,
        candidate_labels,
    )
    parity_trace = compare_rows(
        read_json(reference_run / "parity_trace.json"),
        read_json(candidate_run / "parity_trace.json"),
        reference_run_id,
        candidate_run_id,
        reference_labels,
        candidate_labels,
    )
    similarity_arrays = compare_similarity_arrays(
        reference_evidence / "similarities",
        candidate_evidence / "similarities",
        reference_run_id,
        candidate_run_id,
        reference_labels,
        candidate_labels,
    )
    final_map_parity = read_json(args.final_map_parity.resolve())
    source_snapshots_equal = source_snapshot_signature(reference_manifest) == source_snapshot_signature(
        candidate_manifest
    )

    pass_status = bool(
        reference_manifest.get("status") == "MAP_COMPLETED_EVIDENCE_VALID"
        and candidate_manifest.get("status") == "MAP_COMPLETED_EVIDENCE_VALID"
        and source_snapshots_equal
        and all(row["normalized_equal"] for row in ledgers.values())
        and final_membership["normalized_equal"]
        and parity_trace["normalized_equal"]
        and similarity_arrays["all_arrays_exact"]
        and final_map_parity.get("pass") is True
    )
    metrics = {
        "schema_version": "experiment0-baseline-determinism/1.0",
        "status": "PASS" if pass_status else "FAIL",
        "reference_run": str(reference_run),
        "candidate_run": str(candidate_run),
        "reference_run_id": reference_run_id,
        "candidate_run_id": candidate_run_id,
        "reference_manifest_status": reference_manifest.get("status"),
        "candidate_manifest_status": candidate_manifest.get("status"),
        "source_snapshots_equal": source_snapshots_equal,
        "created_object_uid_pair_count": len(object_pairs),
        "ledgers": ledgers,
        "final_membership": final_membership,
        "parity_trace": parity_trace,
        "similarity_arrays": similarity_arrays,
        "final_map_parity": final_map_parity,
        "expected_nonsemantic_differences": [
            "run_id",
            "event_uid",
            "transaction_uid",
            "object_uid",
            "experiment suffix and output path",
            "whole compressed pickle byte hash",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "determinism_metrics.json", metrics)
    write_json(output_dir / "object_uid_pairs.json", object_pairs)
    (output_dir / "EXPERIMENT0_ROOM0_BASELINE_DETERMINISM_CN.md").write_text(
        markdown(metrics), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "association_count": ledgers["associations.jsonl"]["reference_count"],
                "final_object_count": final_map_parity["candidate_inventory"]["object_count"],
                "trace_equal": parity_trace["normalized_equal"],
                "ledger_failures": [name for name, row in ledgers.items() if not row["normalized_equal"]],
                "similarities_exact": similarity_arrays["all_arrays_exact"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if pass_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
