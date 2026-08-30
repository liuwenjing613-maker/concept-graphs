#!/usr/bin/env python3
"""Compile the two-scene repairability validation into auditable JSON/Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCENES = ("room0", "office0")
CONDITIONS = (
    "B0",
    "legacy_O1",
    "legacy_O2",
    "OA_online",
    "OP_raw",
    "OM_pure",
    "OM_all",
    "OS_strict",
    "OS_hungarian",
    "OG",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed_results(path: Path) -> dict[str, dict]:
    return {row["name"]: row for row in load(path)["results"]}


def selected_geometry(row: dict) -> dict:
    return {
        "instance_ap_mean_25_50": row["instance_ap_mean_25_50"],
        "node_f1": row["node_f1"],
        "semantic_correct": row["semantic_correct"],
        "semantic_denominator": row["semantic_denominator"],
        "semantic_accuracy": row["semantic_accuracy"],
        "predicted_nodes": row["predicted_nodes"],
    }


def selected_fross(row: dict) -> dict:
    keys = (
        "f1",
        "precision",
        "recall",
        "mean_selected_gt_coverage",
        "mean_selected_purity",
        "fragmented_gt_count",
        "contaminated_prediction_count_by_fross_ratio",
    )
    return {key: row[key] for key in keys}


def selected_proxy(row: dict) -> dict:
    keys = (
        "class_conditioned_f1",
        "unique_class_query_count",
        "unique_class_top1_success_count",
        "unique_class_top1_success_rate",
        "unique_class_top3_success_count",
        "unique_class_top3_success_rate",
        "unique_class_top1_within_1m_count",
        "unique_class_top1_within_1m_rate",
        "semantic_class_count_mae",
        "unique_class_top1_candidate_coverage",
    )
    return {key: row[key] for key in keys}


def gap_fraction(value: float, baseline: float, ceiling: float) -> float | None:
    denominator = ceiling - baseline
    return (value - baseline) / denominator if denominator else None


def compact_replay_case(case_root: Path) -> dict:
    build = load(case_root / "geometry" / "build_manifest.json")
    local = load(case_root / "local" / "local_validation.json")
    global_result = load(case_root / "global" / "parity_result.json")
    return {
        "path": str(case_root),
        "evaluation_role": build["evaluation_role"],
        "obs_uid": build["obs_uid"],
        "build_pass": build["pass"],
        "local_pass": local["pass"],
        "global_parity_pass": global_result["pass"],
        "geometry": {
            "original_points": build["geometry_metrics"]["original_observation_point_count"],
            "restored_points": build["geometry_metrics"]["restored_observation_point_count"],
            "point_support_gain_ratio": build["geometry_metrics"]["point_support_gain_ratio"],
            "raw_mask_area": build["geometry_metrics"]["raw_mask_area"],
            "processed_mask_area": build["geometry_metrics"]["processed_mask_area"],
        },
        "association_effect": local["association_effect"],
        "collateral": local["collateral"],
        "local_checks": local["checks"],
        "global_checks": global_result["checks"],
        "timing_ms": {
            name: (timing or {}).get("suffix_total_wall_ms")
            for name, timing in local["timing"].items()
        },
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--branch-head", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    old_root = args.old_root.resolve()
    source_files: set[Path] = set()

    stage = {}
    for scene in SCENES:
        summary_path = root / "stage_funnel" / scene / "summary.json"
        manifest_path = root / "stage_funnel" / scene / "manifest.json"
        source_files.update((summary_path, manifest_path))
        summary = load(summary_path)
        manifest = load(manifest_path)
        stage[scene] = {
            "thresholds": {
                threshold: summary[threshold]["foreground"]
                for threshold in ("0.3", "0.5", "0.7")
            },
            "gt_alignment_summary": manifest["gt_alignment_summary"],
            "provenance_conservation": manifest["provenance_conservation"],
            "frame_count": manifest["frame_count"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "peak_rss_mb": manifest["peak_rss_mb"],
        }

    ladder = {}
    for scene in SCENES:
        ladder[scene] = {}
        for label, suffix in (
            ("OA_online", "oa"),
            ("OP_raw", "op"),
            ("OM_pure", "om_pure"),
            ("OM_all", "om_all"),
        ):
            path = root / "ladder_full" / f"{scene}_{suffix}" / "manifest.json"
            source_files.add(path)
            value = load(path)
            ladder[scene][label] = {
                key: value[key]
                for key in (
                    "mode",
                    "mode_definition",
                    "frame_count",
                    "observation_count",
                    "object_count",
                    "oracle_object_count",
                    "native_object_count",
                    "online_from_empty_map",
                    "future_final_lineage_used_for_mapping",
                    "oracle_and_unassigned_native_maps_isolated",
                    "elapsed_seconds",
                    "peak_rss_mb",
                    "operation_counts",
                    "assignment_summary",
                )
            }

    geometry = {}
    voxel_names = {"0.025": "voxel0p025", "0.05": "voxel0p05", "0.10": "voxel0p10"}
    for scene in SCENES:
        geometry[scene] = {"scales": {}}
        for scale, directory in voxel_names.items():
            path = root / "evaluation" / scene / directory / "metrics.json"
            source_files.add(path)
            rows = indexed_results(path)
            geometry[scene]["scales"][scale] = {
                condition: selected_geometry(rows[condition]) for condition in CONDITIONS
            }
        primary = geometry[scene]["scales"]["0.05"]
        increments = {}
        for name, before, after in (
            ("association", "B0", "OA_online"),
            ("rejected_raw_observations", "OA_online", "OP_raw"),
            ("owner_identifiable_mask_cleanup", "OP_raw", "OM_pure"),
            ("all_raw_support_partition_cleanup", "OM_pure", "OM_all"),
            ("missing_perception", "OM_all", "OG"),
        ):
            increments[name] = primary[after]["node_f1"] - primary[before]["node_f1"]
        baseline = primary["B0"]["node_f1"]
        ceiling = primary["OG"]["node_f1"]
        geometry[scene]["primary_5cm_increments"] = increments
        geometry[scene]["gap_recovered_5cm"] = {
            condition: gap_fraction(primary[condition]["node_f1"], baseline, ceiling)
            for condition in ("legacy_O1", "legacy_O2", "OA_online", "OP_raw", "OM_pure", "OM_all")
        }

    fross = {}
    for scene in SCENES:
        path = root / "fross_correspondence" / f"{scene}_5mm.json"
        source_files.add(path)
        rows = indexed_results(path)
        fross[scene] = {condition: selected_fross(rows[condition]) for condition in CONDITIONS}
        baseline = fross[scene]["B0"]["f1"]
        ceiling = fross[scene]["OG"]["f1"]
        fross[scene]["gap_recovered"] = {
            condition: gap_fraction(fross[scene][condition]["f1"], baseline, ceiling)
            for condition in ("OA_online", "OP_raw", "OM_pure", "OM_all")
        }

    semantics = {}
    for scene in SCENES:
        path = root / "semantic_only" / scene / "manifest.json"
        source_files.add(path)
        value = load(path)
        semantics[scene] = {
            "strict_match_count": value["strict_match_count"],
            "hungarian_match_count": value["hungarian_match_count"],
            "strict_applied_wrong_label_count": value["variants"]["os_strict"]["applied_wrong_label_count"],
            "weak_hungarian_applied_wrong_label_count": value["variants"]["os_hungarian"]["applied_wrong_label_count"],
            "geometry_and_provenance_unchanged": value["geometry_and_provenance_unchanged"],
        }

    downstream = {}
    for scene in SCENES:
        downstream[scene] = {"scales": {}}
        for scale, filename in voxel_names.items():
            path = root / "downstream_proxy_v2" / scene / f"{filename}.json"
            source_files.add(path)
            rows = indexed_results(path)
            downstream[scene]["scales"][scale] = {
                condition: selected_proxy(rows[condition]) for condition in CONDITIONS
            }

    dataset_label_path = root / "dataset_label_audit" / "audit.json"
    source_files.add(dataset_label_path)
    dataset_label_full = load(dataset_label_path)
    dataset_label_audit = {
        "pass": dataset_label_full["pass"],
        "dataset_pairing_pass": dataset_label_full["dataset_pairing_pass"],
        "dataset_audit": dataset_label_full["dataset_audit"],
        "ontology": dataset_label_full["ontology"],
        "label_inventory": dataset_label_full["label_inventory"],
        "fixed_geometry_match_semantic_sensitivity": dataset_label_full[
            "fixed_geometry_match_semantic_sensitivity"
        ],
        "downstream_proxy_5cm_sensitivity": dataset_label_full[
            "downstream_proxy_5cm_sensitivity"
        ],
        "current_policy_exact_reproduction": dataset_label_full[
            "current_policy_exact_reproduction"
        ],
        "changed_matched_pairs_current_to_reviewed": dataset_label_full[
            "changed_matched_pairs_current_to_reviewed"
        ],
        "mode_conclusions": dataset_label_full["mode_conclusions"],
        "interpretation_rules": dataset_label_full["interpretation_rules"],
    }

    api = {"scenes": {}, "full_total_tokens": 0, "smoke_total_tokens": 0}
    for scene, directory in (("room0", "full_room0"), ("office0", "full_office0")):
        path = root / "gpt_ablation" / directory / "semantic_ablation_summary.json"
        source_files.add(path)
        value = load(path)
        api["scenes"][scene] = {
            "model": value["model"],
            "jobs": value["jobs"],
            "counts": value["counts"],
            "wall_seconds": value["wall_seconds"],
            "inference_failures": value["inference_failures"],
            "errors": value["errors"],
            "branches": value["branches"],
        }
        api["full_total_tokens"] += sum(branch["usage"]["total_tokens"] for branch in value["branches"].values())
    smoke_path = root / "gpt_ablation" / "smoke_room0_1" / "semantic_ablation_summary.json"
    if smoke_path.exists():
        source_files.add(smoke_path)
        smoke = load(smoke_path)
        api["smoke_total_tokens"] = sum(branch["usage"]["total_tokens"] for branch in smoke["branches"].values())
    api["total_tokens_including_smoke"] = api["full_total_tokens"] + api["smoke_total_tokens"]

    prior_path = root / "real_replay_prior_audit" / "prior_geometry_gt_audit.json"
    source_files.add(prior_path)
    prior = load(prior_path)
    mining = {}
    for scene in SCENES:
        path = root / "real_replay_candidate_mining" / f"{scene}.json"
        source_files.add(path)
        value = load(path)
        mining[scene] = {
            "processed_observations_scanned": value["processed_observations_scanned"],
            "unique_frames_scanned": value["unique_frames_scanned"],
            "summary": value["summary"],
            "top_strict_0p5": [
                row
                for row in value["candidates"]
                if row["thresholds"]["0.5"]["restore_candidate"]
            ][:3],
        }
    pilot_root = root / "real_replay_micro_pilot"
    pilot_cases = {
        "room0_strict": pilot_root / "room0_strict_r1_f000088_r0012",
        "office0_relaxed_exploratory": pilot_root
        / "office0_relaxed_r1_f000129_r0010",
    }
    for case_root in pilot_cases.values():
        source_files.update(
            (
                case_root / "geometry" / "build_manifest.json",
                case_root / "local" / "local_validation.json",
                case_root / "global" / "parity_result.json",
            )
        )
    replay = {
        "prior_gt_audit": prior,
        "candidate_mining": mining,
        "micro_pilot": {
            label: compact_replay_case(case_root)
            for label, case_root in pilot_cases.items()
        },
        "partition_capability": {
            "contract_and_pure_execution_available": True,
            "hash_bound_oracle_case_passed": True,
            "preassociation_sparse_replay_integration_available": False,
            "association_stage_behavior": "DEFER",
            "interpretation": "highest-impact capability remains an implementation gap",
        },
    }

    old_manifests = {}
    for scene in SCENES:
        path = old_root / "pilot" / scene / "o1_purity0p5" / "manifest.json"
        source_files.add(path)
        value = load(path)
        old_manifests[scene] = {
            "unmatched_policy": value["unmatched_policy"],
            "purity_threshold": value["purity_threshold"],
            "mixed_predicted_masks_split": value["mixed_predicted_masks_split"],
            "assignment_summary": value["assignment_summary"],
        }

    method_audit = {
        "old_o1_unmatched_policy_uses_frozen_final_b0_lineage": True,
        "old_o1_only_gates_purity_not_recall": True,
        "old_o3_self_evaluation_is_tautological_ceiling": True,
        "old_scalar_rho_publication_ready": False,
        "old_relation_result_used_for_direction": False,
        "room0_b0_frozen_parity_verified": True,
        "office0_b0_independent_frozen_parity_reference_available": False,
        "corrected_ladder_online_from_empty": True,
        "corrected_ladder_future_final_lineage_used": False,
        "corrected_semantic_requires_strict_one_to_one_structure": True,
        "dataset_and_ontology_audit_pass": dataset_label_audit["pass"],
        "semantic_primary_policy": "official-only strict cross-dataset mapping",
        "semantic_comparability_policy": "frozen current aliases reproduced exactly",
        "semantic_expanded_alias_policy": "lamp compounds reported as sensitivity only",
        "old_o1_manifests": old_manifests,
    }

    report = {
        "schema_version": "1.0.0",
        "branch_head_before_validation_commit": args.branch_head,
        "scope": {
            "scenes": list(SCENES),
            "scene_count": 2,
            "reason": "explicit user constraint: each experiment uses two scenes",
            "independent_statistical_unit": "scene",
            "inferential_claim": "none; report paired scene deltas and sign consistency only",
        },
        "method_self_audit": method_audit,
        "stage_funnel": stage,
        "online_oracle_ladder": ladder,
        "geometry_evaluation": geometry,
        "fross_style_sensitivity": fross,
        "strict_semantic_oracle": semantics,
        "downstream_proxy": downstream,
        "dataset_and_label_ontology_audit": dataset_label_audit,
        "gpt_vs_llava_ablation": api,
        "real_replay": replay,
        "decision": {
            "freeze_problem_direction": True,
            "primary_direction": "observation-level mask partition/cleanup using historical provenance, followed by transactional local replay and verify/rollback",
            "secondary_after_structure": "semantic verification/relabeling only after a node passes structural reliability gates",
            "not_supported_as_primary": [
                "final-level false split/false merge repair alone",
                "semantic relabeling alone",
                "blind restoration of all raw geometry",
                "direct LLaVA-to-GPT replacement",
                "relation repair before node stabilization",
            ],
            "method_success_not_yet_claimed": True,
            "blocking_implementation_gap": "PARTITION_OBSERVATION is hash-bound and pure-executable but not integrated into pre-association sparse replay",
        },
        "limitations": [
            "Only two scenes were run by explicit user constraint; no population confidence interval or scene bootstrap is valid.",
            "room0 B0 parity against the frozen reference was verified; office0 was freshly built online from empty but has no independent frozen B0 reference for a parity claim.",
            "OM_pure/OM_all/OG use GT only as Oracle capabilities and are not deployable methods.",
            "The strict semantic cohort is empty in both scenes, so pure semantic benefit is unidentified.",
            "The real micro-pilot has one strict room0 positive and one relaxed office0 exploratory case, not a repair success-rate estimate.",
            "Both real micro-pilot dependency closures cover their whole scene, making outside-closure safety vacuous for both cases.",
            "Clean objects were not run through a reliable automatic diagnosis path; exact no-op executor controls pass, but diagnostic false-mutation rate remains unmeasured.",
            "Relation metrics were excluded because the prior relation comparison mixed models/candidate sets.",
            "FROSS-style correspondence is a sensitivity audit adapted to dense points, not the official FROSS benchmark pipeline.",
            "ReplicaSSG-to-Visual-Genome mapping intentionally collapses some classes (for example sofa to chair); expanded compound aliases are sensitivity results, not annotation truth.",
            "The apparent OM_all-minus-B0 class-F1 delta is not ontology-robust: it is positive under the frozen aliases but zero/negative under official-only mapping, so it is not used as quantitative evidence for semantic benefit.",
            "The frozen office0 config contains a room0 render-camera metadata path, but vis_render=false; RGB/depth/pose selection and depth alignment independently verify office0 data use.",
        ],
        "source_files": {
            str(path): sha256_file(path)
            for path in sorted(source_files, key=lambda item: str(item))
        },
    }

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = output_json.with_suffix(output_json.suffix + ".incomplete")
    tmp_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_json.replace(output_json)

    lines = [
        "# 两场景可修性验证与最终方向",
        "",
        "## 最终判断",
        "",
        "冻结问题方向：**历史 observation 的 mask partition/去污染/重组 + transactional replay + verify/rollback**。",
        "仅修 final false split/false merge/label 不能成为主线；稳定结构后，语义确认是第二阶段。当前只冻结方向，不宣称 partition 方法已经实现成功。",
        "",
        "## 方法自检",
        "",
        "- 旧 O1 对 unmatched observation 使用最终 B0 lineage，存在未来 lineage 依赖；新版 OA/OP/OM 从空图按时间顺序构建，且 manifest 明确 `future_final_lineage_used_for_mapping=false`。",
        "- 旧 O1 只以 purity 判定，未同时要求 visible-instance recall；新版漏斗同时报告 0.3/0.5/0.7 的 purity+recall。",
        "- 旧 O3 对自身评测得到 1.0 是定义上限，不是模型结果；不再压成单一 rho。",
        "- 关系结果因模型/候选集不统一被排除。",
        "- room0 的 B0 冻结 parity 已通过；office0 从空图在线新跑且数据配对通过，但没有独立旧冻结 B0 可作 parity，因此不声称复现了某个旧 office0 数值。",
        "- Replica/ReplicaSSG 数据配对和标签本体单独审计；严格官方映射、原评测别名、灯具复合词扩展三档并列，不按样本事后挑映射。",
        "",
        "## 数据集配对",
        "",
        "| scene | 原始 RGB/depth | 在线帧 | schedule | GT depth 对齐最差 median | 最低 5 cm 内比例 | pass |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for scene in SCENES:
        row = dataset_label_audit["dataset_audit"][scene]
        counts = row["counts"]
        schedule = row["frame_schedule"]
        alignment = row["gt_alignment_summary"]
        lines.append(
            f"| {scene} | {counts['rgb_frames']}/{counts['depth_frames']} | {counts['online_frames']} | {schedule['start']}..{schedule['last']} / {schedule['stride']} | {alignment['max_median_abs_depth_m']:.6f} m | {alignment['min_within_5cm']:.6f} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "两个场景均从各自 Replica 序列读取 2000 帧，按 stride=5 在线处理 400 帧；轨迹哈希、scene alias、GT sidecar 帧号与 B0 schedule 完全一致。office0 配置中的 room0 render-camera 路径只在 `vis_render=true` 时使用，本实验为 false。",
            "",
            "## 5 cm 实例 F1",
            "",
            "| scene | B0 | OA | OP | OM_pure | OM_all | OG |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene in SCENES:
        row = geometry[scene]["scales"]["0.05"]
        lines.append(
            f"| {scene} | {fmt(row['B0']['node_f1'])} | {fmt(row['OA_online']['node_f1'])} | {fmt(row['OP_raw']['node_f1'])} | {fmt(row['OM_pure']['node_f1'])} | {fmt(row['OM_all']['node_f1'])} | {fmt(row['OG']['node_f1'])} |"
        )
    lines.extend(
        [
            "",
            "最大边际在两场景均为 `OM_all - OM_pure`，即利用现有 raw proposal 支撑做理想 partition/cleanup。2.5/5/10 cm 方向一致。",
            "",
            "## 5 mm 去重、0.1 m 点对应敏感性",
            "",
            "| scene | B0 | OA | OP | OM_pure | OM_all | OG |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene in SCENES:
        row = fross[scene]
        lines.append(
            f"| {scene} | {fmt(row['B0']['f1'])} | {fmt(row['OA_online']['f1'])} | {fmt(row['OP_raw']['f1'])} | {fmt(row['OM_pure']['f1'])} | {fmt(row['OM_all']['f1'])} | {fmt(row['OG']['f1'])} |"
        )
    lines.extend(
        [
            "",
            "该敏感性定义下仍是 OM_all 最大；粗 2.5 cm 去重使 OG 自一致性失败，已判为方法失败，不参与结论。",
            "",
            "## 下游代理（5 cm）",
            "",
            "| scene/condition | class F1 | R@1 | R@3 | 1 m | count MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scene in SCENES:
        for condition in ("B0", "OM_all", "OG"):
            row = downstream[scene]["scales"]["0.05"][condition]
            denominator = row["unique_class_query_count"]
            lines.append(
                f"| {scene}/{condition} | {fmt(row['class_conditioned_f1'])} | {row['unique_class_top1_success_count']}/{denominator} | {row['unique_class_top3_success_count']}/{denominator} | {row['unique_class_top1_within_1m_count']}/{denominator} | {fmt(row['semantic_class_count_mae'])} |"
            )
    lines.extend(
        [
            "",
            "OM_all 大幅改善结构，却没有同步改善类别查询，说明语义是结构稳定后的第二瓶颈。",
            "",
            "## 标签本体敏感性（5 cm 下游代理）",
            "",
            "| scene/mode | B0 class F1 | OM_all class F1 | OG class F1 | OM_all-B0 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scene in SCENES:
        modes = dataset_label_audit["downstream_proxy_5cm_sensitivity"][scene]
        for mode in ("official_only", "current_aliases", "reviewed_lamp_aliases"):
            row = modes[mode]
            delta = (
                row["OM_all"]["class_conditioned_f1"]
                - row["B0"]["class_conditioned_f1"]
            )
            lines.append(
                f"| {scene}/{mode} | {fmt(row['B0']['class_conditioned_f1'])} | {fmt(row['OM_all']['class_conditioned_f1'])} | {fmt(row['OG']['class_conditioned_f1'])} | {delta:+.3f} |"
            )
    observed_aliases = {
        scene: dataset_label_audit["label_inventory"][scene][
            "reviewed_aliases_observed"
        ]
        for scene in SCENES
    }
    lines.extend(
        [
            "",
            f"扩展别名在 B0 中的实际命中：`{json.dumps(observed_aliases, ensure_ascii=False, sort_keys=True)}`。`desk lamp` 本轮没有出现；仍显式纳入规则以防后续模型输出。三档结果只用于检验结论是否依赖词表。",
            "官方映射下 OM_all-B0 为 room0 负、office0 零，因此当前类别 F1 的增量不具本体稳健性，不用于定量证明语义收益；三档均显示 OM_all 远低于 OG。",
            "",
            "## 真实 replay",
            "",
        ]
    )
    for label, case in replay["micro_pilot"].items():
        lines.append(
            f"- {label}: build/local/global={case['build_pass']}/{case['local_pass']}/{case['global_parity_pass']}，points {case['geometry']['original_points']}→{case['geometry']['restored_points']} ({fmt(case['geometry']['point_support_gain_ratio'], 1)}×)，association changed={case['association_effect']['changed_recorded_decision']}，closure={case['collateral']['affected_observation_count']} observations。"
        )
    lines.extend(
        [
            "",
            "这证明 geometry overlay/replay/parity 机制可运行，不证明目标对象或 scene 指标已改善。最高收益的 PARTITION_OBSERVATION 仍未接入 pre-association sparse replay。",
            "",
            "## GPT 与 LLaVA",
            "",
            f"正式调用共 {api['full_total_tokens']} tokens，含 smoke 共 {api['total_tokens_including_smoke']} tokens；endpoint 未提供价格，不能换算金额。两场景 GPT vision 与 LLaVA-caption→Terra 的 model-covered accuracy 都为 0，直接替换不受支持。",
            "",
            "## 局限",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- 汇总 JSON：`{output_json}`",
            f"- 原始实验根目录：`{root}`",
            "",
        ]
    )
    output_md = args.output_md.resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = output_md.with_suffix(output_md.suffix + ".incomplete")
    tmp_md.write_text("\n".join(lines), encoding="utf-8")
    tmp_md.replace(output_md)
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
