#!/usr/bin/env python3
"""Audit Experiment 0 labels against the frozen false-attach paper scope.

This is a read-only evaluator.  It separates three questions that must not be
collapsed into one number:

1. Was the human event-level ATTACH/NEW judgement valid?
2. Was the selected node clean at the exact pre-event (t-minus) version?
3. Is the event the first false attachment, rather than a later symptom?

The script deliberately excludes false split, mixed current observations, and
pre-contaminated targets from the paper's strict root-false-attach positives.
It also deep-audits private automatic root candidates so they can be queued for
human review without treating automatic GT labels as formal ground truth.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MAIN_ERROR_ROUTES = {"WRONG_ATTACH_EXISTING", "SHOULD_HAVE_BEEN_NEW"}
FALSE_SPLIT_ROUTE = "WRONG_NEW_FALSE_SPLIT"
STRICT_OBSERVATION_QUALITY = "CLEAN_SINGLE_INSTANCE"
SUFFICIENT_IDENTITY = "SUFFICIENT_FOR_IDENTITY"
EVENT_SEQUENCE_RE = re.compile(r"_e(\d+)$")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def event_sequence_from_uid(value: Any) -> int | None:
    match = EVENT_SEQUENCE_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def grouped(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("experiment0_multidimensional_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_observation_gate(current: dict[str, Any], *, require_human: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if require_human:
        if current.get("human_observation_quality") != STRICT_OBSERVATION_QUALITY:
            reasons.append("HUMAN_QUALITY_NOT_CLEAN")
        if current.get("human_identity_evidence_status") != SUFFICIENT_IDENTITY:
            reasons.append("HUMAN_IDENTITY_EVIDENCE_NOT_SUFFICIENT")
        if current.get("human_identity_routing_eligible") is not True:
            reasons.append("HUMAN_ROUTING_NOT_ELIGIBLE")
    if current.get("gt_assignment_eligible") is not True:
        reasons.append("CORRECTED_GT_NOT_ELIGIBLE")
    purity = current.get("gt_purity")
    if purity is None or float(purity) < 0.90:
        reasons.append("CURRENT_GT_PURITY_LT_0_90")
    if current.get("mask_mixed"):
        reasons.append("CURRENT_MASK_MIXED")
    if current.get("mask_two_foreground"):
        reasons.append("CURRENT_MASK_TWO_FOREGROUND")
    return not reasons, reasons


def target_gate(composition: dict[str, Any] | None) -> dict[str, Any]:
    if not composition:
        return {
            "history_count_ok": False,
            "membership_purity_ok": False,
            "projected_pixel_purity_ok": False,
            "causally_clean": False,
            "reasons": ["EXACT_TMINUS_TARGET_MISSING"],
        }
    member_count = int(composition.get("member_observation_count") or 0)
    dominant = composition.get("dominant_gt_fraction")
    nondominant_pixels = composition.get("projected_gt_non_dominant_pixel_fraction_top2")
    projected_pixel_purity = (
        1.0 - float(nondominant_pixels) if nondominant_pixels is not None else None
    )
    history_ok = member_count >= 2
    membership_ok = dominant is not None and float(dominant) >= 0.90
    pixel_ok = projected_pixel_purity is not None and projected_pixel_purity >= 0.90
    joint_state = composition.get("objective_target_pre_state_joint")
    causally_clean = bool(history_ok and membership_ok and pixel_ok and joint_state == "CLEAN_SINGLE_INSTANCE")
    reasons: list[str] = []
    if not history_ok:
        reasons.append("TARGET_HISTORY_LT_2")
    if not membership_ok:
        reasons.append("TARGET_MEMBERSHIP_PURITY_LT_0_90")
    if not pixel_ok:
        reasons.append("TARGET_PROJECTED_PIXEL_PURITY_LT_0_90")
    if joint_state == "ALREADY_CONTAMINATED":
        reasons.append("TARGET_HAS_PRE_EVENT_CONTAMINATION")
    elif joint_state != "CLEAN_SINGLE_INSTANCE":
        reasons.append("TARGET_CAUSAL_CLEANLINESS_UNCERTAIN")
    return {
        "history_count_ok": history_ok,
        "membership_purity_ok": membership_ok,
        "projected_pixel_purity": projected_pixel_purity,
        "projected_pixel_purity_ok": pixel_ok,
        "causally_clean": causally_clean,
        "joint_state": joint_state,
        "reasons": reasons,
    }


def human_scope_row(row: dict[str, Any]) -> dict[str, Any]:
    current = row.get("current_mask") or {}
    target = row.get("original_target_composition")
    current_ok, current_reasons = current_observation_gate(current, require_human=True)
    target_result = target_gate(target)
    route = str(row.get("routing_label") or "")
    original_action = str(row.get("original_action_type") or "")

    if route == FALSE_SPLIT_ROUTE or original_action != "ATTACH_EXISTING":
        scope_status = "OUT_FALSE_SPLIT_OR_NON_ATTACH"
    elif route not in MAIN_ERROR_ROUTES:
        scope_status = "OUT_NOT_A_FALSE_ATTACH_ERROR"
    elif not current_ok:
        scope_status = "OUT_CURRENT_OBSERVATION_GATE"
    elif not target_result["causally_clean"]:
        scope_status = "OUT_CASCADE_OR_PRECONTAMINATED_TARGET"
    else:
        scope_status = "CORE_STRICT_ROOT_FALSE_ATTACH"

    earliest = row.get("earliest_original_target_contamination") or {}
    first_strict = earliest.get("first_strict_multi_gt") or {}
    first_pixel = earliest.get("first_pixel_mixed_mask") or {}
    first_persistent_pixel = earliest.get("first_persistent_pixel_contamination") or {}
    return {
        "case_uid": row.get("case_uid"),
        "event_uid": row.get("event_uid"),
        "event_sequence": row.get("event_sequence"),
        "event_frame_idx": row.get("event_frame_idx"),
        "sample_kind": row.get("sample_kind"),
        "routing_label": route,
        "original_action_type": original_action,
        "human_causal_role": row.get("causal_role"),
        "human_target_pre_state": row.get("human_target_pre_state"),
        "scope_status": scope_status,
        "current_observation_gate_pass": current_ok,
        "current_observation_gate_reasons": current_reasons,
        "current_gt_id": current.get("gt_id"),
        "current_gt_label": current.get("gt_label"),
        "current_gt_purity": current.get("gt_purity"),
        "current_mask_mixed": current.get("mask_mixed"),
        "current_mask_two_foreground": current.get("mask_two_foreground"),
        "target_gate": target_result,
        "target_member_observation_count": (target or {}).get("member_observation_count"),
        "target_dominant_gt_id": (target or {}).get("dominant_gt_id"),
        "target_membership_dominant_fraction": (target or {}).get("dominant_gt_fraction"),
        "target_pixel_mixed_mask_count": (target or {}).get("pixel_mixed_mask_count"),
        "target_two_foreground_mask_count": (target or {}).get("two_foreground_mask_count"),
        "target_joint_state": (target or {}).get("objective_target_pre_state_joint"),
        "exact_tminus_version_uid": row.get("original_target_exact_tminus_version_uid"),
        "packet_version_uid": row.get("original_target_frozen_packet_version_uid"),
        "packet_exact_version_mismatch": row.get("frozen_vs_exact_version_mismatch"),
        "first_strict_contamination_transition_event_uid": first_strict.get("trigger_event_uid"),
        "first_strict_contamination_association_event_uid": row.get(
            "ledger_earliest_strict_contamination_association_event_uid"
        ),
        "first_strict_contamination_trace_kind": first_strict.get("trace_kind"),
        "first_pixel_mixed_transition_event_uid": first_pixel.get("trigger_event_uid"),
        "first_persistent_pixel_contamination_event_uid": first_persistent_pixel.get("trigger_event_uid"),
    }


def probability_scope_row(row: dict[str, Any]) -> dict[str, Any]:
    current = row.get("current_mask") or {}
    target = row.get("original_target_composition")
    current_ok, current_reasons = current_observation_gate(current, require_human=True)
    target_result = target_gate(target)
    route = str(row.get("routing_label") or "")
    original_action = str(row.get("original_action_type") or "")
    strict_eligible = bool(original_action == "ATTACH_EXISTING" and current_ok and target_result["causally_clean"])
    membership_minimum_eligible = bool(
        original_action == "ATTACH_EXISTING"
        and current_ok
        and target_result["history_count_ok"]
        and target_result["membership_purity_ok"]
        and target_result["projected_pixel_purity_ok"]
    )
    if strict_eligible and route in MAIN_ERROR_ROUTES:
        status = "CORE_ROOT_POSITIVE"
    elif strict_eligible and route == "CORRECT_ATTACH":
        status = "CORE_ELIGIBLE_NEGATIVE"
    elif original_action != "ATTACH_EXISTING":
        status = "OUT_NON_ATTACH"
    elif not current_ok:
        status = "OUT_CURRENT_OBSERVATION_GATE"
    elif not target_result["causally_clean"]:
        status = "OUT_TARGET_NOT_CAUSALLY_CLEAN"
    else:
        status = "OUT_OTHER"
    return {
        "case_uid": row.get("case_uid"),
        "event_uid": row.get("event_uid"),
        "event_sequence": row.get("event_sequence"),
        "routing_label": route,
        "original_action_type": original_action,
        "scope_status": status,
        "strict_eligible": strict_eligible,
        "membership_minimum_eligible": membership_minimum_eligible,
        "current_observation_gate_pass": current_ok,
        "current_observation_gate_reasons": current_reasons,
        "current_gt_purity": current.get("gt_purity"),
        "target_gate": target_result,
        "target_member_observation_count": (target or {}).get("member_observation_count"),
        "target_joint_state": (target or {}).get("objective_target_pre_state_joint"),
    }


def compact_earliest(earliest: dict[str, Any] | None) -> dict[str, Any]:
    earliest = earliest or {}
    result: dict[str, Any] = {}
    for key in (
        "first_pixel_mixed_mask",
        "first_persistent_pixel_contamination",
        "first_strict_multi_gt",
        "first_persistent_multi_gt",
    ):
        item = earliest.get(key) or {}
        examples = (
            item.get("introduced_gt_member_examples")
            or item.get("added_member_examples")
            or item.get("pixel_mixed_mask_examples")
            or []
        )
        result[key] = {
            "trigger_event_uid": item.get("trigger_event_uid"),
            "trigger_event_sequence": item.get("trigger_event_sequence"),
            "operation": item.get("operation"),
            "trace_kind": item.get("trace_kind"),
            "association_event_uids": sorted(
                {str(example.get("association_event_uid")) for example in examples if example.get("association_event_uid")}
            ),
        }
    return result


def load_relevant_observation_geometry(path: Path, obs_uids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            uid = str(row.get("obs_uid") or "")
            if uid not in obs_uids:
                continue
            found[uid] = {
                "n_points": row.get("n_points"),
                "pcd_raw_valid_depth_points": row.get("pcd_raw_valid_depth_points"),
                "pcd_after_dbscan_points": row.get("pcd_after_dbscan_points"),
                "valid_depth_ratio": row.get("valid_depth_ratio"),
                "processed_mask_area": row.get("processed_mask_area"),
                "boundary_touch_ratio": row.get("boundary_touch_ratio"),
            }
            if len(found) == len(obs_uids):
                break
    return found


def auto_candidate_rows(audit: Any, routing_rows: list[dict[str, Any]], human_events: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in routing_rows
        if row.get("private_auto_evaluable") is True
        and row.get("private_auto_episode_role") == "ROOT_CANDIDATE"
        and str(row.get("private_auto_routing_label")) in MAIN_ERROR_ROUTES
        and row.get("private_obs_gt_purity") is not None
        and float(row["private_obs_gt_purity"]) >= 0.90
        and int(row.get("private_target_history_observations") or 0) >= 2
        and row.get("private_target_history_dominant_ratio") is not None
        and float(row["private_target_history_dominant_ratio"]) >= 0.90
    ]
    geometry = load_relevant_observation_geometry(
        audit.paths["observations"], {str(row["obs_uid"]) for row in candidates}
    )
    result: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: int(item.get("event_sequence") or 0)):
        sequence = int(row.get("event_sequence") or event_sequence_from_uid(row.get("event_uid")) or 0)
        target_uid = str(row.get("original_target_object_uid") or "")
        exact_version_uid = str(row.get("target_object_version_before") or "") or None
        target_version = audit.version_at(target_uid, sequence, exact_version_uid)
        composition = audit.version_composition(target_version)
        gate = target_gate(composition)
        gt = audit.gt.get(str(row.get("obs_uid")), {})
        current = {
            "gt_assignment_eligible": gt.get("gt_assignment_eligible"),
            "gt_purity": gt.get("gt_purity"),
            "mask_mixed": gt.get("mask_mixed"),
            "mask_two_foreground": gt.get("mask_two_foreground"),
        }
        current_ok, current_reasons = current_observation_gate(current, require_human=False)
        human = human_events.get(str(row.get("event_uid")))
        if current_ok and gate["causally_clean"]:
            status = "AUTO_CORE_STRICT_CANDIDATE_NEEDS_HUMAN_REVIEW"
        elif not current_ok:
            status = "AUTO_OUT_CURRENT_OBSERVATION_GATE"
        elif (
            gate.get("joint_state") == "UNCERTAIN"
            and gate.get("history_count_ok")
            and gate.get("membership_purity_ok")
            and gate.get("projected_pixel_purity_ok")
        ):
            status = "AUTO_TARGET_CLEANLINESS_UNCERTAIN_NEEDS_HUMAN_REVIEW"
        else:
            status = "AUTO_OUT_PRECONTAMINATED_TARGET"
        result.append(
            {
                "event_uid": row.get("event_uid"),
                "event_sequence": sequence,
                "processed_frame_idx": row.get("processed_frame_idx"),
                "obs_uid": row.get("obs_uid"),
                "auto_routing_label": row.get("private_auto_routing_label"),
                "auto_scope_status": status,
                "current_observation_gate_pass": current_ok,
                "current_observation_gate_reasons": current_reasons,
                "current_gt_id": row.get("private_obs_gt_id"),
                "current_gt_label": row.get("private_obs_gt_label"),
                "current_gt_purity": row.get("private_obs_gt_purity"),
                "target_uid": target_uid,
                "target_exact_tminus_version_uid": (target_version or {}).get("object_version_uid"),
                "target_gt_id_membership_audit": row.get("private_target_gt_id"),
                "target_history_observations_membership_audit": row.get("private_target_history_observations"),
                "target_membership_dominant_ratio_audit": row.get("private_target_history_dominant_ratio"),
                "target_gate": gate,
                "target_member_observation_count_deep": (composition or {}).get("member_observation_count"),
                "target_dominant_gt_id_deep": (composition or {}).get("dominant_gt_id"),
                "target_membership_dominant_fraction_deep": (composition or {}).get("dominant_gt_fraction"),
                "target_projected_non_dominant_pixel_fraction": (composition or {}).get(
                    "projected_gt_non_dominant_pixel_fraction_top2"
                ),
                "target_pixel_mixed_mask_count": (composition or {}).get("pixel_mixed_mask_count"),
                "target_pixel_mixed_mask_examples": (composition or {}).get("pixel_mixed_mask_examples") or [],
                "target_two_foreground_mask_count": (composition or {}).get("two_foreground_mask_count"),
                "target_joint_state": (composition or {}).get("objective_target_pre_state_joint"),
                "earliest_target_contamination": compact_earliest(audit.earliest_contamination(target_version)),
                "observation_geometry": geometry.get(str(row.get("obs_uid"))),
                "association_top1_score": row.get("top1_score"),
                "association_margin": row.get("margin"),
                "already_human_labeled": human is not None,
                "human_case_uid": (human or {}).get("case_uid"),
                "human_routing_label": (human or {}).get("human", {}).get("routing_label"),
                "human_observation_quality": (human or {}).get("human", {}).get("observation_quality"),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_uid",
        "event_uid",
        "event_frame_idx",
        "sample_kind",
        "routing_label",
        "human_causal_role",
        "scope_status",
        "current_gt_id",
        "current_gt_label",
        "current_gt_purity",
        "target_member_observation_count",
        "target_dominant_gt_id",
        "target_membership_dominant_fraction",
        "target_joint_state",
        "packet_exact_version_mismatch",
        "first_strict_contamination_association_event_uid",
        "first_strict_contamination_trace_kind",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_report(metrics: dict[str, Any], human_rows: list[dict[str, Any]], auto_rows: list[dict[str, Any]]) -> str:
    human = metrics["human_error_scope_audit"]
    probability = metrics["probability_sample_scope_audit"]
    automatic = metrics["full_stream_auto_root_candidate_audit"]
    strict_ci = probability["strict_root_rate_wilson_95"]
    ci_text = "不可计算" if strict_ci is None else f"[{strict_ci[0] * 100:.2f}%, {strict_ci[1] * 100:.2f}%]"

    lines = [
        "# 实验 0：room0 主论文范围净化审计",
        "",
        "## 最终判断",
        "",
        "当前标注仍然有效，但它支持的是**事件级身份路由判断**，不能把 14 个错误直接当成主论文的 14 个独立 root false-attach。按最新冻结定义逐层复核后：",
        "",
        f"- 人工错误事件：{human['error_event_count']}；",
        f"- false split / 非 ATTACH 错误：{human['excluded_false_split_or_non_attach_count']}；",
        f"- 当前 observation 通过 strict purity/质量门：{human['current_observation_gate_pass_count']}；",
        f"- ATTACH 错误中，精确 t-minus 目标已预污染：{human['precontaminated_attach_error_count']}；",
        f"- 当前 14 例里可作为主论文严格 root 正例：**{human['strict_core_root_count']}**。",
        "",
        "这不是说人工标注错了。人工标签正确回答了“当前 observation 应该 ATTACH 哪个节点或 NEW”；问题在于后续目标历史审计发现，这些页面上的错误多数已经是更早污染的症状。主论文只允许把第一次从干净状态发生的错误 ATTACH 算作 root。",
        "",
        "## 1. 为什么原来的 14 例不能直接计数",
        "",
        "| case | 路由事实 | 人工因果角色 | 精确 t-minus 目标 | 主论文处置 |",
        "|---|---|---|---|---|",
    ]
    for row in human_rows:
        lines.append(
            "| {case} | {route} | {role} | {target} | {status} |".format(
                case=row.get("case_uid"),
                route=row.get("routing_label"),
                role=row.get("human_causal_role"),
                target=row.get("target_joint_state") or "N/A",
                status=row.get("scope_status"),
            )
        )
    lines += [
        "",
        "最重要的修正是：root/cascade 不能只依据当前截图或目标 top-GT 多数来定，必须读取事件发生前的**精确 object version**，并沿 observation association 与 object-merge parent DAG 查历史。冻结页面版本和精确 t-minus 版本不一致的案例也必须以精确版本为统计依据。",
        "",
        "## 2. 概率样本重新解释",
        "",
        f"150 个概率样本中，原先有 148 个可做一般路由判断，事件级错误为 5/148=3.38%。该数仍可报告为‘clean observation 条件下的路由动作错误率’，但不是主论文 root false-attach 发生率。",
        "",
        f"按主论文 strict 条件过滤后，概率样本中有 {probability['strict_eligible_attach_count']} 个‘当前 observation 纯净 + 精确 t-minus 目标因果干净 + 至少两条目标历史’的 ATTACH 事件；人工确认 root false-attach 为 {probability['strict_core_root_positive_count']}，Wilson 95% 区间为 {ci_text}。",
        "",
        "这个 0 不能解释为问题不存在：这里只是 room0 开发场景，严格分母较小，而且完整流自动审计仍发现待人工复核的候选。它能说明的是：**当前标注尚未给主论文提供已确认的自然 root 正例。**",
        "",
        "## 3. 完整流自动候选的深审计",
        "",
        f"旧的 top-GT 自动逻辑在完整流中给出 {automatic['nominal_auto_root_candidate_count']} 个 main-scope root 候选。加入精确 t-minus 混合历史后，严格干净候选 {automatic['strict_auto_candidate_count']} 个，目标清洁度边界候选 {automatic['uncertain_auto_candidate_count']} 个，明确预污染并排除 {automatic['precontaminated_auto_candidate_count']} 个。自动结果只是选例依据，不能替代人工标签。",
        "",
        "| event | frame | 自动动作真值 | t-minus 状态 | 是否已人工标注 | 处置 |",
        "|---|---:|---|---|---|---|",
    ]
    for row in auto_rows:
        lines.append(
            "| {event} | {frame} | {route} | {state} | {labeled} | {status} |".format(
                event=row.get("event_uid"),
                frame=row.get("processed_frame_idx"),
                route=row.get("auto_routing_label"),
                state=row.get("target_joint_state"),
                labeled="是" if row.get("already_human_labeled") else "否",
                status=row.get("auto_scope_status"),
            )
        )
    lines += [
        "",
        "## 4. 当前能得出的结论",
        "",
        "1. 标注 schema 能稳定表达 ATTACH/NEW 身份事实，标注数据可以继续用于候选覆盖、困难负例和事件级错误分析。",
        "2. 现有 Q2–Q4 mixed-root 回放只能作为旁支/上界分析；mixed mask 不进入主论文 false-attach 正例。",
        "3. 当前不能声称‘方法可行’，也不能进入自动 trigger 或 VLM 主实验；实验 0 的核心自然 root 数仍未建立。",
        "4. 下一步只复核深审计保留下来的自然 root 候选，并在 office0 与未见场景从空图严格在线运行后重复同一协议。",
        "5. baseline 同配置整场重复确定性仍需单独完成；现有局部 B0R parity 不能替代整场双跑。",
        "",
        "## 5. 后续判定顺序",
        "",
        "- 先让人工盲审保留的自动候选，不能直接把自动 GT 当答案；",
        "- 对通过者编译未来独立视角、top-K+NEW 覆盖和 cascade descendants；",
        "- room0/office0 只做开发，不调未见场景阈值；",
        "- 至少 4 个未见场景完成后，再判断自然 root 数、场景覆盖和未来证据是否足以继续；",
        "- Experiment 0 未成立前，不把 mixed-mask detector、VLM 或复杂修复器接回主线。",
        "",
        "## 6. 可复现产物",
        "",
        "- `metrics.json`：所有计数与 Wilson 区间；",
        "- `human_error_scope_rows.jsonl/.csv`：14 例逐例范围裁决；",
        "- `probability_scope_rows.jsonl`：150 个概率样本的严格分母归类；",
        "- `auto_root_candidate_rows.jsonl`：完整流自动 root 候选的精确 t-minus 深审计。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--multidimensional-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(args.multidimensional_module.resolve())
    audit = module.Audit(project_root, output_dir)
    audit.load()

    analysis_root = project_root / "results/experiments/experiment0_manual_annotation_20260901/v2_large_room0_r1/analysis_20260902"
    multidimensional_dir = analysis_root / "multidimensional_analysis"
    if not (multidimensional_dir / "event_multidimensional_table.jsonl").exists():
        multidimensional_dir = (
            project_root / "results/experiments/experiment0_multidimensional_analysis_20260902"
        )
    event_rows = read_jsonl(multidimensional_dir / "event_multidimensional_table.jsonl")
    error_rows = read_jsonl(multidimensional_dir / "error_multidimensional_table.jsonl")
    routing_rows = read_jsonl(audit.paths["routing_audit_records"])
    combined_events = read_jsonl(audit.paths["events"])
    human_by_event = {str(row["event_uid"]): row for row in combined_events}

    human_scope = [human_scope_row(row) for row in error_rows]
    probability_source = [row for row in event_rows if row.get("sample_kind") == "PROBABILITY_SAMPLE"]
    probability_scope = [probability_scope_row(row) for row in probability_source]
    auto_scope = auto_candidate_rows(audit, routing_rows, human_by_event)

    strict_probability = [row for row in probability_scope if row["strict_eligible"]]
    strict_positive = [row for row in strict_probability if row["scope_status"] == "CORE_ROOT_POSITIVE"]
    membership_probability = [row for row in probability_scope if row["membership_minimum_eligible"]]
    membership_positive = [
        row for row in membership_probability if str(row.get("routing_label")) in MAIN_ERROR_ROUTES
    ]
    strict_auto = [
        row for row in auto_scope if row["auto_scope_status"] == "AUTO_CORE_STRICT_CANDIDATE_NEEDS_HUMAN_REVIEW"
    ]
    precontaminated_auto = [
        row for row in auto_scope if row["auto_scope_status"] == "AUTO_OUT_PRECONTAMINATED_TARGET"
    ]
    uncertain_auto = [
        row
        for row in auto_scope
        if row["auto_scope_status"] == "AUTO_TARGET_CLEANLINESS_UNCERTAIN_NEEDS_HUMAN_REVIEW"
    ]
    reviewable_auto = strict_auto + uncertain_auto

    metrics = {
        "schema_version": "experiment0-core-scope-audit/1.0",
        "scope_definition": {
            "analysis_unit": "historical ATTACH_EXISTING decision",
            "included_errors": sorted(MAIN_ERROR_ROUTES),
            "excluded_primary_errors": [FALSE_SPLIT_ROUTE, "MIXED_CURRENT_MASK", "POSE_OR_DYNAMIC", "SEMANTIC_OR_RELATION"],
            "current_observation_gt_purity_min": 0.90,
            "target_membership_purity_min": 0.90,
            "target_projected_pixel_purity_min": 0.90,
            "target_history_observation_min": 2,
            "root_rule": "exact t-minus target is causally clean before the erroneous attachment",
        },
        "human_error_scope_audit": {
            "error_event_count": len(human_scope),
            "scope_status_counts": grouped(human_scope, "scope_status"),
            "excluded_false_split_or_non_attach_count": sum(
                row["scope_status"] == "OUT_FALSE_SPLIT_OR_NON_ATTACH" for row in human_scope
            ),
            "current_observation_gate_pass_count": sum(row["current_observation_gate_pass"] for row in human_scope),
            "attach_error_count": sum(row["original_action_type"] == "ATTACH_EXISTING" for row in human_scope),
            "precontaminated_attach_error_count": sum(
                row["scope_status"] == "OUT_CASCADE_OR_PRECONTAMINATED_TARGET" for row in human_scope
            ),
            "strict_core_root_count": sum(row["scope_status"] == "CORE_STRICT_ROOT_FALSE_ATTACH" for row in human_scope),
            "packet_exact_version_mismatch_count": sum(bool(row["packet_exact_version_mismatch"]) for row in human_scope),
            "human_root_reclassified_out_of_scope_count": sum(
                row.get("human_causal_role") == "ROOT" and row["scope_status"] != "CORE_STRICT_ROOT_FALSE_ATTACH"
                for row in human_scope
            ),
        },
        "probability_sample_scope_audit": {
            "probability_sample_count": len(probability_scope),
            "source_route_counts": grouped(probability_scope, "routing_label"),
            "scope_status_counts": grouped(probability_scope, "scope_status"),
            "strict_eligible_attach_count": len(strict_probability),
            "strict_core_root_positive_count": len(strict_positive),
            "strict_root_rate": safe_ratio(len(strict_positive), len(strict_probability)),
            "strict_root_rate_wilson_95": wilson_interval(len(strict_positive), len(strict_probability)),
            "membership_and_pixel_0_90_minimum_attach_count": len(membership_probability),
            "membership_and_pixel_0_90_minimum_error_count": len(membership_positive),
            "membership_minimum_error_rate": safe_ratio(len(membership_positive), len(membership_probability)),
            "membership_minimum_error_rate_wilson_95": wilson_interval(
                len(membership_positive), len(membership_probability)
            ),
        },
        "full_stream_auto_root_candidate_audit": {
            "nominal_auto_root_candidate_count": len(auto_scope),
            "auto_scope_status_counts": grouped(auto_scope, "auto_scope_status"),
            "strict_auto_candidate_count": len(strict_auto),
            "strict_auto_candidate_human_labeled_count": sum(row["already_human_labeled"] for row in strict_auto),
            "strict_auto_candidate_unlabeled_count": sum(not row["already_human_labeled"] for row in strict_auto),
            "uncertain_auto_candidate_count": len(uncertain_auto),
            "uncertain_auto_candidate_human_labeled_count": sum(row["already_human_labeled"] for row in uncertain_auto),
            "uncertain_auto_candidate_unlabeled_count": sum(not row["already_human_labeled"] for row in uncertain_auto),
            "reviewable_auto_candidate_count": len(reviewable_auto),
            "reviewable_auto_candidate_unlabeled_count": sum(not row["already_human_labeled"] for row in reviewable_auto),
            "precontaminated_auto_candidate_count": len(precontaminated_auto),
        },
        "interpretation": {
            "annotation_valid_for_event_routing": True,
            "annotation_sufficient_for_main_core_root_count": False,
            "method_feasibility_established": False,
            "mixed_root_replays_are_main_scope_evidence": False,
            "next_action": "HUMAN_REVIEW_STRICT_AUTO_CANDIDATES_THEN_REPEAT_ON_NEW_SCENES",
        },
        "runtime_seconds": time.time() - started,
    }

    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "human_error_scope_rows.jsonl", human_scope)
    write_csv(output_dir / "human_error_scope_rows.csv", human_scope)
    write_jsonl(output_dir / "probability_scope_rows.jsonl", probability_scope)
    write_jsonl(output_dir / "auto_root_candidate_rows.jsonl", auto_scope)
    report = render_report(metrics, human_scope, auto_scope)
    (output_dir / "EXPERIMENT0_CORE_SCOPE_AUDIT_CN.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "OK",
                "human_errors": len(human_scope),
                "strict_human_roots": metrics["human_error_scope_audit"]["strict_core_root_count"],
                "strict_probability_denominator": len(strict_probability),
                "strict_probability_roots": len(strict_positive),
                "strict_auto_candidates": len(strict_auto),
                "uncertain_auto_candidates": len(uncertain_auto),
                "output_dir": str(output_dir),
                "runtime_seconds": metrics["runtime_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
