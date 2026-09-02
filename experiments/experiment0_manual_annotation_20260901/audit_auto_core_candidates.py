#!/usr/bin/env python3
"""Deep-audit automatically selected Experiment-0 false-attach candidates.

The private GT routing audit is intentionally only a case selector.  This tool
adds the missing causal check: inspect the exact target object version at t^- and
ask whether the current physical identity had already entered that target.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MAIN_ERROR_ROUTES = {"WRONG_ATTACH_EXISTING", "SHOULD_HAVE_BEEN_NEW"}
EVENT_RE = re.compile(r"_e(\d+)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def event_sequence(value: Any) -> int | None:
    match = EVENT_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def sorted_counts(counter: Counter[int], denominator: int) -> list[dict[str, Any]]:
    return [
        {
            "gt_id": key,
            "count": value,
            "fraction": safe_ratio(value, denominator),
        }
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def current_observation_gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row.get("gt_assignment_eligible") is not True:
        reasons.append("CORRECTED_GT_NOT_ELIGIBLE")
    purity = row.get("gt_purity")
    if purity is None or float(purity) < 0.90:
        reasons.append("CURRENT_GT_PURITY_LT_0_90")
    if row.get("mask_mixed"):
        reasons.append("CURRENT_MASK_MIXED")
    if row.get("mask_two_foreground"):
        reasons.append("CURRENT_MASK_TWO_FOREGROUND")
    return not reasons, reasons


def summarize_target(
    member_uids: list[str],
    gt_by_obs: dict[str, dict[str, Any]],
    association_by_obs: dict[str, dict[str, Any]],
    current_gt_id: int,
) -> dict[str, Any]:
    all_rows = [(uid, gt_by_obs.get(uid)) for uid in member_uids]
    eligible = [
        (uid, row)
        for uid, row in all_rows
        if row and row.get("gt_assignment_eligible") is True and row.get("gt_top_id") is not None
    ]
    gt_counts: Counter[int] = Counter(int(row["gt_top_id"]) for _, row in eligible)
    sorted_ids = sorted(gt_counts, key=lambda item: (-gt_counts[item], item))
    dominant_id = sorted_ids[0] if sorted_ids else None
    second_id = sorted_ids[1] if len(sorted_ids) > 1 else None
    frames_by_gt: dict[int, set[int]] = defaultdict(set)
    projected_pixels: Counter[int] = Counter()
    mixed_rows: list[tuple[str, dict[str, Any]]] = []
    for uid, row in eligible:
        frame = int(row.get("frame_idx") if row.get("frame_idx") is not None else -1)
        top_id = int(row["gt_top_id"])
        frames_by_gt[top_id].add(frame)
        projected_pixels[top_id] += int(row.get("gt_top_pixels") or 0)
        if row.get("gt_second_id") is not None:
            projected_pixels[int(row["gt_second_id"])] += int(row.get("gt_second_pixels") or 0)
        if row.get("mask_mixed") or row.get("mask_two_foreground"):
            mixed_rows.append((uid, row))

    second_count = gt_counts.get(second_id, 0) if second_id is not None else 0
    second_frames = len(frames_by_gt.get(second_id, set())) if second_id is not None else 0
    known_pixels = sum(projected_pixels.values())
    dominant_pixels = max(projected_pixels.values(), default=0)
    projected_non_dominant_fraction = (
        1.0 - float(dominant_pixels) / float(known_pixels) if known_pixels else None
    )
    mixed_frames = {
        int(row.get("frame_idx") if row.get("frame_idx") is not None else -1)
        for _, row in mixed_rows
    }
    persistent_membership = second_count >= 2 and second_frames >= 2
    persistent_pixel = bool(
        len(mixed_rows) >= 2
        and len(mixed_frames) >= 2
        and projected_non_dominant_fraction is not None
        and projected_non_dominant_fraction >= 0.05
    )

    prior_current_examples: list[dict[str, Any]] = []
    for uid, row in eligible:
        top_is_current = int(row["gt_top_id"]) == current_gt_id
        second_is_current = (
            row.get("gt_second_id") is not None
            and int(row["gt_second_id"]) == current_gt_id
            and float(row.get("gt_second_fraction") or 0.0) >= 0.10
        )
        if not (top_is_current or second_is_current):
            continue
        assoc = association_by_obs.get(uid, {})
        prior_current_examples.append(
            {
                "obs_uid": uid,
                "frame_idx": row.get("frame_idx"),
                "association_event_uid": assoc.get("event_uid"),
                "association_event_sequence": assoc.get("event_sequence")
                if assoc.get("event_sequence") is not None
                else event_sequence(assoc.get("event_uid")),
                "gt_top_id": row.get("gt_top_id"),
                "gt_top_label": row.get("gt_top_label"),
                "gt_purity": row.get("gt_purity"),
                "gt_second_id": row.get("gt_second_id"),
                "gt_second_label": row.get("gt_second_label"),
                "gt_second_fraction": row.get("gt_second_fraction"),
                "mask_mixed": bool(row.get("mask_mixed")),
                "mask_two_foreground": bool(row.get("mask_two_foreground")),
                "evidence_kind": "CURRENT_GT_IS_TOP"
                if top_is_current
                else "CURRENT_GT_IS_MATERIAL_SECOND",
            }
        )
    prior_current_examples.sort(
        key=lambda row: (
            int(row.get("association_event_sequence") or 10**18),
            int(row.get("frame_idx") or -1),
            str(row["obs_uid"]),
        )
    )

    coverage = safe_ratio(len(eligible), len(member_uids))
    dominant_fraction = safe_ratio(gt_counts.get(dominant_id, 0), len(eligible)) if dominant_id is not None else None
    projected_pixel_purity = (
        1.0 - projected_non_dominant_fraction
        if projected_non_dominant_fraction is not None
        else None
    )
    if not eligible or (coverage is not None and coverage < 0.80):
        joint_state = "UNCERTAIN"
        joint_reason = "NO_OR_LOW_GT_COVERAGE"
    elif persistent_membership:
        joint_state = "ALREADY_CONTAMINATED"
        joint_reason = "PERSISTENT_MULTI_GT_MEMBERSHIP"
    elif persistent_pixel:
        joint_state = "ALREADY_CONTAMINATED"
        joint_reason = "PERSISTENT_PIXEL_MIXTURE_2OBS_2FRAMES_AND_5PCT"
    elif len(gt_counts) == 1 and not mixed_rows:
        joint_state = "CLEAN_SINGLE_INSTANCE"
        joint_reason = "ONE_MEMBER_GT_AND_NO_PIXEL_MIXTURE"
    else:
        joint_state = "UNCERTAIN"
        joint_reason = "LOW_SUPPORT_MEMBERSHIP_OR_PIXEL_MIXTURE"

    gate_reasons: list[str] = []
    if len(member_uids) < 2:
        gate_reasons.append("TARGET_HISTORY_LT_2")
    if dominant_fraction is None or dominant_fraction < 0.90:
        gate_reasons.append("TARGET_MEMBERSHIP_PURITY_LT_0_90")
    if projected_pixel_purity is None or projected_pixel_purity < 0.90:
        gate_reasons.append("TARGET_PROJECTED_PIXEL_PURITY_LT_0_90")
    if prior_current_examples:
        gate_reasons.append("CURRENT_IDENTITY_ALREADY_PRESENT_BEFORE_EVENT")
    if joint_state == "ALREADY_CONTAMINATED":
        gate_reasons.append("TARGET_HAS_PRE_EVENT_CONTAMINATION")
    elif joint_state != "CLEAN_SINGLE_INSTANCE":
        gate_reasons.append("TARGET_CAUSAL_CLEANLINESS_UNCERTAIN")

    causally_clean = bool(
        len(member_uids) >= 2
        and dominant_fraction is not None
        and dominant_fraction >= 0.90
        and projected_pixel_purity is not None
        and projected_pixel_purity >= 0.90
        and not prior_current_examples
        and joint_state == "CLEAN_SINGLE_INSTANCE"
    )
    return {
        "member_observation_count": len(member_uids),
        "eligible_gt_observation_count": len(eligible),
        "gt_coverage": coverage,
        "gt_id_counts": sorted_counts(gt_counts, len(eligible)),
        "dominant_gt_id": dominant_id,
        "dominant_gt_fraction": dominant_fraction,
        "second_gt_id": second_id,
        "second_gt_count": second_count,
        "second_gt_unique_frame_count": second_frames,
        "projected_gt_pixel_counts_top2": sorted_counts(projected_pixels, known_pixels),
        "projected_pixel_purity": projected_pixel_purity,
        "pixel_mixed_mask_count": len(mixed_rows),
        "pixel_mixed_unique_frame_count": len(mixed_frames),
        "two_foreground_mask_count": sum(bool(row.get("mask_two_foreground")) for _, row in mixed_rows),
        "persistent_multi_gt_2obs_2frames": persistent_membership,
        "persistent_pixel_contamination_2obs_2frames_5pct": persistent_pixel,
        "joint_state": joint_state,
        "joint_state_reason": joint_reason,
        "prior_current_identity_evidence_count": len(prior_current_examples),
        "prior_current_identity_evidence": prior_current_examples[:10],
        "causally_clean": causally_clean,
        "gate_reasons": gate_reasons,
    }


def select_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("private_auto_evaluable") is True
        and row.get("private_auto_episode_role") == "ROOT_CANDIDATE"
        and str(row.get("private_auto_routing_label")) in MAIN_ERROR_ROUTES
        and row.get("private_obs_gt_purity") is not None
        and float(row["private_obs_gt_purity"]) >= 0.90
        and int(row.get("private_target_history_observations") or 0) >= 2
        and row.get("private_target_history_dominant_ratio") is not None
        and float(row["private_target_history_dominant_ratio"]) >= 0.90
    ]


def audit_candidates(
    routing_rows: list[dict[str, Any]],
    versions: dict[str, dict[str, Any]],
    gt_by_obs: dict[str, dict[str, Any]],
    association_by_obs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in sorted(select_candidates(routing_rows), key=lambda item: event_sequence(item.get("event_uid")) or -1):
        obs_uid = str(row["obs_uid"])
        current = gt_by_obs.get(obs_uid, {})
        current_ok, current_reasons = current_observation_gate(current)
        version_uid = str(row.get("target_object_version_before") or "")
        version = versions.get(version_uid)
        if version is None:
            target = None
            status = "AUTO_TARGET_EXACT_TMINUS_VERSION_MISSING"
        else:
            target = summarize_target(
                [str(uid) for uid in version.get("member_observation_uids") or []],
                gt_by_obs,
                association_by_obs,
                int(row["private_obs_gt_id"]),
            )
            if not current_ok:
                status = "AUTO_OUT_CURRENT_OBSERVATION_GATE"
            elif target["causally_clean"]:
                status = "AUTO_CORE_STRICT_CANDIDATE_NEEDS_HUMAN_REVIEW"
            elif target["prior_current_identity_evidence_count"]:
                status = "AUTO_OUT_PRECONTAMINATED_TARGET_CURRENT_IDENTITY_ALREADY_PRESENT"
            elif target["joint_state"] == "UNCERTAIN":
                status = "AUTO_TARGET_CLEANLINESS_UNCERTAIN_NEEDS_HUMAN_REVIEW"
            else:
                status = "AUTO_OUT_PRECONTAMINATED_TARGET"
        results.append(
            {
                "scene": row.get("scene"),
                "event_uid": row.get("event_uid"),
                "event_sequence": row.get("event_sequence")
                if row.get("event_sequence") is not None
                else event_sequence(row.get("event_uid")),
                "processed_frame_idx": row.get("processed_frame_idx"),
                "obs_uid": obs_uid,
                "auto_routing_label": row.get("private_auto_routing_label"),
                "current_gt_id": row.get("private_obs_gt_id"),
                "current_gt_label": row.get("private_obs_gt_label"),
                "current_gt_purity": row.get("private_obs_gt_purity"),
                "current_observation_gate_pass": current_ok,
                "current_observation_gate_reasons": current_reasons,
                "target_object_uid": row.get("original_target_object_uid"),
                "target_exact_tminus_version_uid": version_uid,
                "target": target,
                "auto_scope_status": status,
                "interpretation": "PRIVATE_GT_CASE_SELECTION_ONLY_NEEDS_HUMAN_FOR_FORMAL_LABEL",
            }
        )
    return results


def render_report(scene: str, rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["auto_scope_status"]) for row in rows)
    lines = [
        f"# 实验0：{scene} 自动候选的精确事件前历史审计",
        "",
        "## 结论",
        "",
        f"私有 GT 初筛得到 {len(rows)} 个主范围名义候选。加入精确 `t^-` 目标版本后，严格干净候选 "
        f"{counts.get('AUTO_CORE_STRICT_CANDIDATE_NEEDS_HUMAN_REVIEW', 0)} 个；事件前已含当前身份的级联候选 "
        f"{counts.get('AUTO_OUT_PRECONTAMINATED_TARGET_CURRENT_IDENTITY_ALREADY_PRESENT', 0)} 个；其余边界候选 "
        f"{counts.get('AUTO_TARGET_CLEANLINESS_UNCERTAIN_NEEDS_HUMAN_REVIEW', 0)} 个。",
        "",
        "自动 GT 仅用于挑选值得看的事件，不能替代正式人工标签。这里回答的是候选是否可能是独立 root，"
        "不是评估整个检测/分割系统的所有错误。",
        "",
        "## 候选明细",
        "",
    ]
    for row in rows:
        target = row.get("target") or {}
        identities = ", ".join(
            f"GT{x['gt_id']}={x['count']}"
            for x in target.get("gt_id_counts") or []
        ) or "无可靠 GT"
        lines.extend(
            [
                f"### frame {row.get('processed_frame_idx')} · {row.get('event_uid')}",
                "",
                f"- 当前实例：GT{row.get('current_gt_id')} `{row.get('current_gt_label')}`，纯度 {float(row.get('current_gt_purity') or 0):.3f}；",
                f"- 原目标精确版本：`{row.get('target_exact_tminus_version_uid')}`；",
                f"- 事件前成员：{target.get('member_observation_count')} 条（{identities}）；",
                f"- 事件前当前身份证据：{target.get('prior_current_identity_evidence_count')} 条；混合 mask：{target.get('pixel_mixed_mask_count')} 条；",
                f"- 判定：`{row.get('auto_scope_status')}`；",
                f"- 原因：{', '.join(target.get('gate_reasons') or []) or '无'}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 对下一步的含义",
            "",
            "若严格干净候选为 0，则本场景不能给主论文贡献独立自然 root 正例；它仍可作为混合 mask 导致级联污染的机制证据。"
            "下一步应筛查新的严格在线场景，而不是把这些级联事件重复计作 root。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--observation-gt", type=Path, required=True)
    parser.add_argument("--routing-records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    routing_rows = read_jsonl(args.routing_records)
    gt_by_obs = {str(row["obs_uid"]): row for row in read_jsonl(args.observation_gt)}
    association_rows = read_jsonl(args.evidence_root / "associations.jsonl")
    association_by_obs = {str(row["obs_uid"]): row for row in association_rows}
    candidate_version_uids = {
        str(row.get("target_object_version_before") or "") for row in select_candidates(routing_rows)
    }
    versions = {
        str(row["object_version_uid"]): row
        for row in read_jsonl(args.evidence_root / "object_versions.jsonl")
        if str(row.get("object_version_uid")) in candidate_version_uids
    }
    rows = audit_candidates(routing_rows, versions, gt_by_obs, association_by_obs)
    counts = Counter(str(row["auto_scope_status"]) for row in rows)
    metrics = {
        "schema_version": "experiment0-auto-core-candidate-audit/1.0",
        "status": "READY",
        "scene": args.scene,
        "nominal_main_root_candidate_count": len(rows),
        "scope_status_counts": dict(sorted(counts.items())),
        "strict_clean_candidate_count": counts.get(
            "AUTO_CORE_STRICT_CANDIDATE_NEEDS_HUMAN_REVIEW", 0
        ),
        "precontaminated_current_identity_candidate_count": counts.get(
            "AUTO_OUT_PRECONTAMINATED_TARGET_CURRENT_IDENTITY_ALREADY_PRESENT", 0
        ),
        "uncertain_candidate_count": counts.get(
            "AUTO_TARGET_CLEANLINESS_UNCERTAIN_NEEDS_HUMAN_REVIEW", 0
        ),
        "formal_label_warning": "PRIVATE_GT_SELECTION_ONLY; HUMAN_REVIEW_REQUIRED_FOR_FORMAL_LABELS",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "metrics.json", metrics)
    write_jsonl(args.output_root / "candidate_audit.jsonl", rows)
    (args.output_root / "OFFICE0_CORE_CANDIDATE_AUDIT_CN.md").write_text(
        render_report(args.scene, rows), encoding="utf-8"
    )
    (args.output_root / "READY").write_text("READY\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
