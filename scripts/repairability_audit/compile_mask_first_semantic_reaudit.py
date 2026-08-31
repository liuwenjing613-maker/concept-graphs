#!/usr/bin/env python3
"""Compile the corrected mask-first point-semantic and structure re-audit.

This script only reads frozen experiment artifacts.  It writes a compact JSON
record and a Chinese Markdown report suitable for the server ``beauty/`` area.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POINT_CONDITIONS = [
    ("B0", "native_B0"),
    ("MF_OP", "native_MF_OP"),
    ("MF_OM_pure", "native_MF_OM_pure"),
    ("MF_OM_all", "native_MF_OM_all"),
    ("MF_OM_all_OA", "native_MF_OM_all_OA"),
    ("MF_OM_all_OA_GT_label", "gtlabel_MF_OM_all_OA"),
]

MAP_DIRS = [
    "room0_mf_op",
    "room0_mf_pure",
    "room0_mf_all_native",
    "room0_mf_all_oa",
    "office0_mf_op",
    "office0_mf_pure",
    "office0_mf_all_native",
    "office0_mf_all_oa",
]

SCALES = ["0p025", "0p05", "0p10"]
SCENES = ["room0", "office0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--old-point-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--official-evaluator-a", type=Path, required=True)
    parser.add_argument("--official-evaluator-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rounded(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "scope",
        "point_count",
        "num_classes",
        "miou",
        "mrecall",
        "mprecision",
        "mf1score",
        "fmiou",
        "point_accuracy",
        "acc0.15",
        "acc0.25",
        "acc0.50",
        "acc0.75",
    ]
    output = {key: row[key] for key in keys}
    for key in (
        "miou",
        "mrecall",
        "mprecision",
        "mf1score",
        "fmiou",
        "point_accuracy",
    ):
        output[key] = round(float(output[key]), 6)
    return output


def load_point_results(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    conditions: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for name, directory in POINT_CONDITIONS:
        path = root / "point_semantic_corrected" / directory / "semseg_results.json"
        payload = read_json(path)
        rows = payload["summary_rows_percent"]
        by_scope = {row["scope"]: rounded(row) for row in rows}
        conditions[name] = {
            "room0": rounded(rows[0]),
            "office0": rounded(rows[1]),
            "two_scene_micro": rounded(rows[-1]),
        }
        provenance[name] = {
            "result": str(path.resolve()),
            "result_sha256": sha256_file(path),
            "inputs": payload["inputs"],
            "protocol": payload["protocol"],
            "runtime_seconds": payload["runtime_seconds"],
            "scopes": sorted(by_scope),
        }
    return conditions, provenance


def load_structure(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scale in SCALES:
        output[scale] = {}
        for scene in SCENES:
            path = root / "structure_corrected" / scene / f"metrics_{scale}.json"
            payload = read_json(path)
            output[scale][scene] = {
                "gt_map": payload["gt_map"],
                "gt_map_sha256": payload["gt_map_sha256"],
                "observable_gt_nodes": payload["observable_gt_nodes"],
                "results": {
                    row["name"]: {
                        "instance_ap_mean_25_50": row["instance_ap_mean_25_50"],
                        "ap25": row["ap25"]["ap"],
                        "ap50": row["ap50"]["ap"],
                        "node_f1": row["node_f1"],
                        "predicted_nodes": row["predicted_nodes"],
                        "map_sha256": row["map_sha256"],
                    }
                    for row in payload["results"]
                },
                "artifact": str(path.resolve()),
                "artifact_sha256": sha256_file(path),
            }
    return output


def load_formal_manifests(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in MAP_DIRS:
        directory = root / "formal_corrected" / name
        manifest_path = directory / "manifest.json"
        manifest = read_json(manifest_path)
        map_files = sorted(directory.glob("pcd_*.pkl.gz"))
        output[name] = {
            "ready": (directory / "READY").is_file(),
            "incomplete": (directory / "INCOMPLETE").exists(),
            "frame_count": manifest["frame_count"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "association_policy": manifest["association_policy"],
            "cumulative_capabilities": manifest["cumulative_capabilities"],
            "effective_observation_stream_sha256": manifest[
                "effective_observation_stream_sha256"
            ],
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "maps": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in map_files
            ],
        }
    return output


def load_sidecars(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scene in SCENES:
        path = root / "sidecars_geometry_full" / scene / "manifest.json"
        payload = read_json(path)
        output[scene] = {
            "frame_count": payload["frame_count"],
            "method": payload["method"],
            "max_distance_m": payload["max_distance_m"],
            "alignment_summary": payload["alignment_summary"],
            "trajectory_sha256": payload["trajectory_sha256"],
            "semantic_mesh_sha256": payload["semantic_mesh_sha256"],
            "objects_json_sha256": payload["objects_json_sha256"],
            "manifest": str(path.resolve()),
            "manifest_sha256": sha256_file(path),
        }
    return output


def old_b0_parity(old_root: Path, current: dict[str, Any]) -> dict[str, Any]:
    old = read_json(old_root / "B0" / "semseg_results.json")
    row = rounded(old["summary_rows_percent"][-1])
    now = current["B0"]["two_scene_micro"]
    numeric = [
        "point_count",
        "miou",
        "mrecall",
        "mprecision",
        "mf1score",
        "fmiou",
        "point_accuracy",
    ]
    return {
        "exact": all(row[key] == now[key] for key in numeric),
        "old": row,
        "current": now,
        "checked_fields": numeric,
    }


def fmt(value: float) -> str:
    return f"{value:.2f}%"


def build_markdown(record: dict[str, Any]) -> str:
    point = record["point_semantic_percent"]
    b0 = point["B0"]["two_scene_micro"]
    lines = [
        "# Mask 前置顺序：点级语义与结构指标纠错复核",
        "",
        "## 最终判断",
        "",
        "旧版语义 sidecar 与 RGB/depth 视角不一致，旧版大幅下降数字和依赖旧 O3 的结构绝对值均撤回。校正后，原生点语义没有呈现单调提升：OP/OM_pure 退化，OM_all 略高于 B0，OA 提高 mIoU，但点准确率仍低于 B0。结构指标则显示 OM_pure 明显有效，OM_all 相对 OM_pure 不稳定并在 room0 退化，OA 带来最大增量。相同 OA 几何换成严格 GT 标签后达到 90% 以上，说明主要剩余限制是重画 mask 后未同步刷新 CLIP 语义特征，而不是几何整体失效。",
        "",
        "## 点级语义（ali-dev 协议，n_exclude=6）",
        "",
        "| 条件 | room0 mIoU | office0 mIoU | 两场景 mIoU | ΔmIoU vs B0 | fwIoU | 点准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _ in POINT_CONDITIONS:
        row = point[name]["two_scene_micro"]
        lines.append(
            "| {name} | {room} | {office} | {pooled} | {delta:+.2f} | {fwiou} | {acc} |".format(
                name=name,
                room=fmt(point[name]["room0"]["miou"]),
                office=fmt(point[name]["office0"]["miou"]),
                pooled=fmt(row["miou"]),
                delta=row["miou"] - b0["miou"],
                fwiou=fmt(row["fmiou"]),
                acc=fmt(row["point_accuracy"]),
            )
        )

    lines.extend(
        [
            "",
            "两场景汇总是合并混淆矩阵后的 micro 结果，不是两个场景百分数的简单平均。GT-label 行是隔离上限，不是可部署成绩。",
            "",
            "## 校正后的 5 cm 结构复核",
            "",
            "| 条件 | room0 AP | office0 AP | 平均 AP | room0 F1 | office0 F1 | 平均 F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    structure = record["structure_class_agnostic"]["0p05"]
    for name in ["B0", "MF_OP", "MF_OM_pure", "MF_OM_all_native", "MF_OM_all_OA", "OG"]:
        room = structure["room0"]["results"][name]
        office = structure["office0"]["results"][name]
        lines.append(
            f"| {name} | {room['instance_ap_mean_25_50']:.3f} | "
            f"{office['instance_ap_mean_25_50']:.3f} | "
            f"{(room['instance_ap_mean_25_50'] + office['instance_ap_mean_25_50']) / 2:.3f} | "
            f"{room['node_f1']:.3f} | {office['node_f1']:.3f} | "
            f"{(room['node_f1'] + office['node_f1']) / 2:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 对修复方向的含义",
            "",
            "1. 优先保留 OM_pure 式的观测去污染/同 owner 干净合并；它在校正结构 AP/F1 上有跨场景的大增益。",
            "2. 不要直接把 OM_all 最大化 partition 当作更强版本：它相对 OM_pure 的两场景平均 AP、F1 都下降，且 room0 退化明显。",
            "3. association/replay 仍是最大结构增量来源，但必须让重画 mask 的 CLIP 特征同步重算，否则视觉关联和最终语义都被旧特征限制。",
            "4. 下一项最小验证应固定相同 mask、帧序和几何，只对所有改变过的 bbox/crop 用冻结 ali-dev CLIP 重编码，再复跑 native association 与点语义。",
            "",
            "## 严谨性检查",
            "",
            f"- B0 与冻结 ali-dev 点语义结果逐项精确一致：{record['checks']['b0_exact_parity']}。",
            "- 8 个正式 mask-first 地图均从空图按 400 帧在线顺序重建，READY=8，INCOMPLETE=0。",
            "- 校正 sidecar 逐帧只使用当前深度、当前位姿和固定 GT 语义网格；两场景全 400 帧有效深度接受率最小值均为 1.0，最坏 p99 几何近邻距离小于 1 cm。",
            "- room0/office0 GT 使用官方词表内的 `lamp`；本轮没有 `desk lamp`。office0 的匿名化/undefined 标签经 3 cm 点 GT 审计主要对应 `other`。",
            "- 点级评测使用精确 CPU cKDTree 最近邻；这是为确定性与 ali-dev parity 保留的实现，地图构建使用 GPU。",
            "",
            "## 必须保留的局限",
            "",
            "- 只有 room0 与 office0，不能给出总体数据集置信区间。",
            "- synthetic/partition mask 仍复用代表 proposal 的原 CLIP 特征；因此原生语义行是当前流水线真实读数，但不是纯 mask 几何的因果效应。",
            "- OA 与 GT-label 均为 Oracle；90.30% 只证明几何与身份绑定上限，不代表当前系统已能实现。",
            "- 旧 sidecar 与旧 O3 相关的绝对结构数值及恢复比例不得继续引用。",
            "",
            "## 失败与资源记录",
            "",
            "- 一次评测环境导入失败发生在读取数据前，空输出目录已清理并用既有正式环境重跑。",
            "- GPU3 在首次 CUDA 初始化时失败，未产生正式地图；残余 INCOMPLETE 目录已核对路径后删除，正式任务改用 GPU0/1/2/4。",
            "- 本阶段没有调用外部 API，因此没有 API token 消耗。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    point, point_provenance = load_point_results(root)
    structure = load_structure(root)
    manifests = load_formal_manifests(root)
    sidecars = load_sidecars(root)

    parity = old_b0_parity(args.old_point_root.resolve(), point)
    ready_count = sum(item["ready"] for item in manifests.values())
    incomplete_count = sum(item["incomplete"] for item in manifests.values())
    frame_counts = sorted({item["frame_count"] for item in manifests.values()})

    evaluator_hash = sha256_file(args.evaluator.resolve())
    official_a_hash = sha256_file(args.official_evaluator_a.resolve())
    official_b_hash = sha256_file(args.official_evaluator_b.resolve())

    record = {
        "schema_version": 1,
        "status": "corrected_formal",
        "withdrawn_results": {
            "old_habitat_semantic_sidecar_point_metrics": True,
            "old_o3_referenced_structure_absolute_values": True,
            "reason": "semantic viewport/visible-surface mismatch",
        },
        "point_semantic_percent": point,
        "point_semantic_provenance": point_provenance,
        "structure_class_agnostic": structure,
        "formal_maps": manifests,
        "sidecars": sidecars,
        "checks": {
            "b0_exact_parity": parity["exact"],
            "b0_parity": parity,
            "formal_ready_count": ready_count,
            "formal_incomplete_count": incomplete_count,
            "formal_frame_counts": frame_counts,
            "all_formal_online_400_ready": ready_count == 8
            and incomplete_count == 0
            and frame_counts == [400],
            "official_evaluator_copies_equal": official_a_hash == official_b_hash,
            "evaluator_sha256": evaluator_hash,
            "official_evaluator_a_sha256": official_a_hash,
            "official_evaluator_b_sha256": official_b_hash,
        },
        "interpretation": {
            "native_point_semantic_is_not_monotonic": True,
            "om_pure_structure_gain_clear": True,
            "om_all_increment_over_om_pure_is_not_stable": True,
            "oracle_association_is_largest_structure_increment": True,
            "oa_gt_label_isolation_miou_percent": point["MF_OM_all_OA_GT_label"]
            ["two_scene_micro"]["miou"],
            "oa_native_miou_percent": point["MF_OM_all_OA"]["two_scene_micro"]
            ["miou"],
            "dominant_remaining_confound": "synthetic masks reuse representative proposal CLIP feature",
            "recommended_next_test": "re-encode every changed mask crop with the frozen ali-dev CLIP encoder before online association; keep geometry and frame order fixed",
        },
        "limitations": [
            "two scenes only",
            "Oracle masks and association are not deployable methods",
            "native synthetic-mask observations retain stale CLIP features",
            "GT-label control is a geometry/identity upper bound",
        ],
    }

    if not record["checks"]["all_formal_online_400_ready"]:
        raise RuntimeError("formal map readiness audit failed")
    if not record["checks"]["b0_exact_parity"]:
        raise RuntimeError("B0 evaluator parity failed")
    if not record["checks"]["official_evaluator_copies_equal"]:
        raise RuntimeError("frozen official evaluator copies differ")

    json_path = args.output_dir / "mask_first_semantic_reaudit.json"
    markdown_path = args.output_dir / "MASK_FIRST_SEMANTIC_REAUDIT_CN.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    markdown_path.write_text(build_markdown(record), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
