#!/usr/bin/env python3
"""Compile the two-scene mask-first cumulative-ladder audit.

This script only reads frozen maps/manifests/evaluations and writes a compact
machine-readable summary plus the human-facing report. It never rebuilds or
modifies a map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


SCENES = ("room0", "office0")
SCALES = {"0p025": 0.025, "0p05": 0.05, "0p10": 0.10}
NEW_STAGES = ("B0", "MF_OP", "MF_OM_pure", "MF_OM_all_native", "MF_OM_all_OA", "OG")
FORWARD_STAGES = ("B0", "FWD_OA", "FWD_OP", "FWD_OM_pure", "MF_OM_all_OA", "OG")
MANIFEST_DIRS = {
    "MF_OP": "mf_op",
    "MF_OM_pure": "mf_pure",
    "MF_OM_all_native": "mf_all_native",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(result: dict) -> dict:
    return {
        "instance_ap_mean_25_50": result["instance_ap_mean_25_50"],
        "ap25": result["ap25"]["ap"],
        "ap50": result["ap50"]["ap"],
        "node_f1": result["node_f1"],
        "semantic_accuracy": result["semantic_accuracy"],
        "semantic_denominator": result["semantic_denominator"],
        "predicted_nodes": result["predicted_nodes"],
        "map_sha256": result["map_sha256"],
    }


def rounded(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    ready_count = 0
    incomplete = []
    for scene in SCENES:
        manifests[scene] = {}
        for stage, suffix in MANIFEST_DIRS.items():
            run_dir = args.root / "formal" / f"{scene}_{suffix}"
            manifest = read_json(run_dir / "manifest.json")
            manifests[scene][stage] = manifest
            ready_count += int((run_dir / "READY").is_file())
            if (run_dir / "INCOMPLETE").exists():
                incomplete.append(str(run_dir))

    evaluations = {}
    raw_evaluations = {}
    for scene in SCENES:
        evaluations[scene] = {}
        raw_evaluations[scene] = {}
        for tag, scale in SCALES.items():
            raw = read_json(args.root / "evaluation" / scene / f"voxel{tag}" / "metrics.json")
            by_name = {entry["name"]: entry for entry in raw["results"]}
            raw_evaluations[scene][tag] = raw
            evaluations[scene][tag] = {
                name: metric_row(by_name[name]) for name in by_name
            }

    fross = {}
    for scene in SCENES:
        raw = read_json(args.root / "fross_correspondence" / f"{scene}_5mm.json")
        fross[scene] = {
            entry["name"]: {
                "f1": entry["f1"],
                "precision": entry["precision"],
                "recall": entry["recall"],
                "mean_selected_purity": entry["mean_selected_purity"],
                "mean_selected_gt_coverage": entry["mean_selected_gt_coverage"],
            }
            for entry in raw["results"]
        }

    five_cm_average = {}
    for stage in evaluations["room0"]["0p05"]:
        rows = [evaluations[scene]["0p05"][stage] for scene in SCENES]
        five_cm_average[stage] = {
            "instance_ap_mean_25_50": mean(row["instance_ap_mean_25_50"] for row in rows),
            "node_f1": mean(row["node_f1"] for row in rows),
        }

    fross_average = {
        stage: {"f1": mean(fross[scene][stage]["f1"] for scene in SCENES)}
        for stage in fross["room0"]
    }

    increments = {}
    for previous, current in zip(NEW_STAGES, NEW_STAGES[1:]):
        increments[f"{previous}->{current}"] = {
            metric: five_cm_average[current][metric] - five_cm_average[previous][metric]
            for metric in ("instance_ap_mean_25_50", "node_f1")
        }
    b0_ap = five_cm_average["B0"]["instance_ap_mean_25_50"]
    og_ap = five_cm_average["OG"]["instance_ap_mean_25_50"]
    gap_recovery = {
        stage: (five_cm_average[stage]["instance_ap_mean_25_50"] - b0_ap) / (og_ap - b0_ap)
        for stage in NEW_STAGES[1:-1]
    }

    runner_hash = sha256(args.runner)
    manifest_hashes = {
        manifests[scene][stage]["input_sha256"]["runner"]
        for scene in SCENES for stage in MANIFEST_DIRS
    }
    op_audits = {
        scene: manifests[scene]["MF_OP"]["provenance_audit"] for scene in SCENES
    }
    parity_sources = {
        "room0": (
            args.root / "smoke" / "room0_mf_all_oa10" / "manifest.json",
            args.root / "parity" / "room0_old_om_all10" / "manifest.json",
        ),
        "office0": (
            args.root / "smoke" / "office0_mf_all_oa10" / "manifest.json",
            args.previous_root / "ladder_smoke" / "office0_om_all10" / "manifest.json",
        ),
    }
    reuse_parity = {}
    for scene, (new_path, old_path) in parity_sources.items():
        new_manifest, old_manifest = read_json(new_path), read_json(old_path)
        fields = ("geometry_sha256", "object_count", "observation_count")
        reuse_parity[scene] = {
            "pass": all(new_manifest[field] == old_manifest[field] for field in fields),
            "new_manifest": str(new_path),
            "old_manifest": str(old_path),
            "geometry_sha256": new_manifest["geometry_sha256"],
            "object_count": new_manifest["object_count"],
            "observation_count": new_manifest["observation_count"],
        }
    label_audit = read_json(args.label_audit)
    quality_gates = {
        "new_online_runs_ready": ready_count,
        "expected_new_online_runs": 6,
        "incomplete_runs": incomplete,
        "single_runner_hash": len(manifest_hashes) == 1 and runner_hash in manifest_hashes,
        "runner_sha256": runner_hash,
        "all_online_from_empty": all(
            manifests[scene][stage]["online_from_empty_map"]
            for scene in SCENES for stage in MANIFEST_DIRS
        ),
        "future_final_lineage_used": any(
            manifests[scene][stage]["future_final_lineage_used_for_mapping"]
            for scene in SCENES for stage in MANIFEST_DIRS
        ),
        "earlier_mask_geometry_duplicated": any(
            manifests[scene][stage]["earlier_mask_geometry_duplicated"]
            for scene in SCENES for stage in MANIFEST_DIRS
        ),
        "op_provenance_missing_total": sum(item["missing"] for item in op_audits.values()),
        "op_provenance_unexpected_total": sum(item["unexpected"] for item in op_audits.values()),
        "dataset_pairing_pass": label_audit["dataset_pairing_pass"],
        "multiscale_outputs": sum(
            (args.root / "evaluation" / scene / f"voxel{tag}" / "metrics.json").is_file()
            for scene in SCENES for tag in SCALES
        ),
        "fross_outputs": sum(
            (args.root / "fross_correspondence" / f"{scene}_5mm.json").is_file()
            for scene in SCENES
        ),
        "reused_oa_two_scene_10_frame_geometry_parity": all(
            item["pass"] for item in reuse_parity.values()
        ),
        "reused_oa_parity": reuse_parity,
    }

    runtime = {}
    for scene in SCENES:
        runtime[scene] = {}
        for stage in MANIFEST_DIRS:
            manifest = manifests[scene][stage]
            runtime[scene][stage] = {
                "frames": manifest["frame_count"],
                "objects": manifest["object_count"],
                "observations": manifest["observation_count"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "peak_rss_mb": manifest["peak_rss_mb"],
            }
        runtime[scene]["MF_OM_all_OA"] = {
            "reused": True,
            "source": str(args.previous_root / "ladder_full" / f"{scene}_om_all"),
            "reason": "same maximal-partition stream plus GT-identity association; exact 10-frame geometry/object/observation parity passed in both scenes",
        }

    summary = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "mask-first cumulative ordering audit",
        "scenes": list(SCENES),
        "frames_per_scene": 400,
        "stage_order": list(NEW_STAGES),
        "stage_definitions": {
            "B0": "frozen baseline",
            "MF_OP": "native processed observations plus owner-identifiable rejected raw observations; native online association",
            "MF_OM_pure": "OP subsumed; one GT-clipped union per owner-identifiable object from all raw proposals; native online association",
            "MF_OM_all_native": "OP and OM_pure subsumed; maximal GT-clipped partition of all overlapping raw proposals plus false-positive suppression; native online association",
            "MF_OM_all_OA": "identical maximal-partition observation stream with GT-identity association; reused from prior formal OM_all after parity audit",
            "OG": "observable GT upper bound",
        },
        "quality_gates": quality_gates,
        "five_cm": {scene: evaluations[scene]["0p05"] for scene in SCENES},
        "five_cm_two_scene_average": five_cm_average,
        "five_cm_cumulative_increments": increments,
        "five_cm_gap_recovery_fraction": gap_recovery,
        "multiscale": evaluations,
        "fross_5mm_0p1m": fross,
        "fross_two_scene_average": fross_average,
        "runtime": runtime,
        "label_ontology_control": {
            "primary_structure_metrics_are_class_agnostic": True,
            "mapping_sha256": label_audit["ontology"]["mapping_sha256"],
            "official_only_is_strict_baseline": True,
            "desk_lamp_present_in_current_two_scene_predictions": False,
            "reviewed_lamp_aliases": label_audit["ontology"]["reviewed_lamp_aliases"],
            "reviewed_aliases_observed": {
                scene: label_audit["label_inventory"][scene]["reviewed_aliases_observed"]
                for scene in SCENES
            },
            "semantic_increment_ontology_robust": False,
        },
        "decision": {
            "op_alone_stable": False,
            "om_pure_vs_b0_stable": True,
            "om_pure_increment_after_op_stable": False,
            "maximal_partition_stable_large_gain": True,
            "association_remains_major_bottleneck_after_mask_repair": True,
            "recommended_order": "partition/decontaminate observations first, then association/replay, then semantic verification",
        },
        "limitations": [
            "Only room0 and office0 were run by explicit user constraint; no population confidence interval is valid.",
            "OM_pure, OM_all, OA, and OG are Oracle capabilities using GT and are not deployable methods.",
            "The large OM_all-to-OA interaction means mask and association effects are not additive independent main effects.",
            "Semantic denominators are small and ontology-sensitive; semantic values are not used to choose the mask/association order.",
            "FROSS-style correspondence is an evaluation-only sensitivity audit, not the official FROSS benchmark pipeline.",
        ],
        "artifacts": {
            "root": str(args.root),
            "previous_formal_root": str(args.previous_root),
            "runner": str(args.runner),
            "label_audit": str(args.label_audit),
        },
    }

    json_path = args.output_dir / "mask_first_order_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    def stage_table(stages):
        lines = ["| stage | room0 AP | office0 AP | avg AP | room0 F1 | office0 F1 | avg F1 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for stage in stages:
            room = evaluations["room0"]["0p05"][stage]
            office = evaluations["office0"]["0p05"][stage]
            avg = five_cm_average[stage]
            lines.append(
                f"| {stage} | {rounded(room['instance_ap_mean_25_50'])} | "
                f"{rounded(office['instance_ap_mean_25_50'])} | "
                f"{rounded(avg['instance_ap_mean_25_50'])} | {rounded(room['node_f1'])} | "
                f"{rounded(office['node_f1'])} | {rounded(avg['node_f1'])} |"
            )
        return "\n".join(lines)

    md = [
        "# Mask-first 累积顺序实验（room0 + office0）",
        "",
        "## 最核心结果",
        "",
        "把 mask partition/去污染放在前面是正确的，但它不能替代关联修复。完整 OM_all 在原生关联下已稳定改善结构；随后加 OA 又产生最大单步提升，说明高质量 mask 之后 association 仍是主要限制。OP 单独恢复被拒绝观测不稳定，不能作为独立主方向。",
        "",
        "## 实验顺序与实现",
        "",
        "所有新建条件均对 400 帧从空图按时间顺序在线构建。阶段按能力累积，但不重复叠加同一几何：",
        "",
        "1. `B0`：冻结基线。",
        "2. `MF_OP`：保留原处理后观测，并恢复可确定 owner 的 rejected raw observation；仍用原生在线关联。",
        "3. `MF_OM_pure`：吸收 OP 能力，将同一 owner 的 raw proposal 先裁剪并合成一个干净观测；仍用原生在线关联。",
        "4. `MF_OM_all_native`：吸收前两步，对所有相交 raw proposal 做最大化 GT partition、去污染和 FP 抑制；仍用原生在线关联。",
        "5. `MF_OM_all_OA`：观测流与上一步完全相同，只把关联换成 GT identity。该定义与此前正式 `OM_all` 相同，smoke parity 通过后直接复用，避免重复全量运行。",
        "6. `OG`：可观测 GT 上限。",
        "",
        "## 5 cm 主评测",
        "",
        "AP 是 AP25/AP50 均值；F1 是 IoU≥0.25 的一对一节点 F1。二者均为类别无关结构指标。",
        "",
        stage_table(NEW_STAGES),
        "",
        "两场景平均的 AP 单步增量：",
        "",
    ]
    for edge, values in increments.items():
        md.append(f"- `{edge}`：{values['instance_ap_mean_25_50']:+.4f}")
    md.extend([
        "",
        f"`MF_OM_all_native` 恢复了 B0→OG 结构 AP 差距的 {gap_recovery['MF_OM_all_native']:.1%}；加 OA 后累计恢复 {gap_recovery['MF_OM_all_OA']:.1%}。因此 mask 修复有明确价值，但仅靠 mask 不够。",
        "",
        "## 与原先 OA-first 顺序的直接对照（5 cm）",
        "",
        stage_table(FORWARD_STAGES),
        "",
        "顺序并非可交换的独立加法。OA-first 的早期提升较小；mask-first 做到 maximal partition 后结构显著改善，但最大的交互增益仍来自在干净观测上做 OA。",
        "",
        "## FROSS-style 敏感性（5 mm 去重、0.1 m 对应）",
        "",
        "| stage | room0 F1 | office0 F1 | avg F1 |",
        "|---|---:|---:|---:|",
    ])
    for stage in NEW_STAGES:
        md.append(
            f"| {stage} | {rounded(fross['room0'][stage]['f1'])} | "
            f"{rounded(fross['office0'][stage]['f1'])} | {rounded(fross_average[stage]['f1'])} |"
        )
    md.extend([
        "",
        "该敏感性下 `MF_OM_all_native → MF_OM_all_OA` 仍在两场景显著增加。反例也被保留：room0 的 `MF_OP` 比 B0 下降，进一步说明 OP 单独不稳定。三档 2.5/5/10 cm 主评测对 `OM_all` 和 `+OA` 的方向一致。",
        "",
        "## 标签与数据集控制",
        "",
        "- Replica/ReplicaSSG 两场景配对审计通过。",
        "- 主结论只使用类别无关 AP/F1，因此 `desk lamp` 与 `lamp` 等别名不会改变 mask/association 顺序判断。",
        "- 严格官方映射是基线；`desk lamp→lamp` 等仅作为预先定义的敏感性规则，不改写 GT。本轮两个场景预测中没有 `desk lamp`，room0 仅实际出现 `ceiling light`，office0 没有灯具扩展别名命中。",
        "- 语义分母小且本体敏感，因此本轮不把 semantic accuracy 用作顺序结论证据。",
        "",
        "## 质量门与资源",
        "",
        f"- 新建正式在线图：{ready_count}/6 READY；INCOMPLETE：{len(incomplete)}。",
        f"- 统一 runner SHA256：`{runner_hash}`；manifest 与文件一致：{quality_gates['single_runner_hash']}。",
        f"- OP 溯源缺失/意外：{quality_gates['op_provenance_missing_total']}/{quality_gates['op_provenance_unexpected_total']}。",
        f"- 复用最终 +OA 的两场景 10 帧几何/对象数/观测数精确 parity：{quality_gates['reused_oa_two_scene_10_frame_geometry_parity']}。",
        "- GPU0/1/2 用于在线构图；GPU3 CUDA 初始化失败后未继续使用，其失败目录已删除。评测为 CPU 几何计算。",
        "",
        "## 最终方向",
        "",
        "推荐顺序冻结为：**observation-level mask partition/去污染 → association/replay → semantic verification**。不要把 OP 当成主修复器，也不要认为先修 mask 就能消除 association 问题。下一步应实现真实的 targeted partition，并用 replay/verify/rollback 检验它能否逼近本 Oracle，而不是继续扩展 final-level merge/split 规则。",
        "",
        "## 局限",
        "",
        "- 仅两个场景，不能外推总体置信区间。",
        "- OM_pure、OM_all、OA、OG 均使用 GT 提供 Oracle 能力，不是可部署方法。",
        "- mask 与 association 存在强交互，不能把各阶段增量解释成互相独立的因果主效应。",
        "- FROSS-style 结果是敏感性审计，不是官方 FROSS benchmark。",
        "",
        "## 产物",
        "",
        f"- 汇总 JSON：`{json_path}`",
        f"- 原始实验：`{args.root}`",
    ])
    md_path = args.output_dir / "MASK_FIRST_ORDER_REPORT.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")

    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "quality_gates": quality_gates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
