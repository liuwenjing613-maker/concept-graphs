#!/usr/bin/env python3
"""Audit completed v2 labels and compile them into Experiment 0 event records."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from label_logic_v2 import (
    derive_routing_label,
    validate_blind_label,
    validate_final_label,
)


ERROR_LABELS = {
    "WRONG_ATTACH_EXISTING",
    "SHOULD_HAVE_BEEN_NEW",
    "WRONG_NEW_FALSE_SPLIT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def count(rows: Iterable[dict[str, Any]], getter) -> dict[str, int]:
    return dict(sorted(Counter(str(getter(row)) for row in rows).items()))


def label_core(label: dict[str, Any]) -> dict[str, Any]:
    blind = label["blind"]
    final = label["final"]
    derived = label["derived"]
    return {
        "observation_quality": blind["observation_quality"],
        "matching_candidate_codes": sorted(blind["matching_candidate_codes"]),
        "identity_evidence_status": blind["identity_evidence_status"],
        "target_pre_state": final["target_pre_state"],
        "full_map_status": final["full_map_status"],
        "outside_matching_node_uids": sorted(final["outside_matching_node_uids"]),
        "annotation_status": derived["annotation_status"],
        "routing_label": derived["routing_label"],
        "correct_action_type": derived["correct_action_type"],
        "identity_routing_eligible": derived["identity_routing_eligible"],
    }


def queue_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["human"]["annotation_status"] == "COMPLETED"]
    errors = [row for row in completed if row["human"]["is_error"]]
    excluded = [row for row in rows if row["human"]["annotation_status"] == "EXCLUDED"]
    deferred = [row for row in rows if row["human"]["annotation_status"] == "DEFERRED"]
    original_attach = [row for row in completed if row["original_action_type"] == "ATTACH_EXISTING"]
    original_new = [row for row in completed if row["original_action_type"] == "NEW"]
    attach_errors = [row for row in original_attach if row["human"]["is_error"]]
    new_errors = [row for row in original_new if row["human"]["is_error"]]
    return {
        "total": len(rows),
        "completed_route": len(completed),
        "excluded_non_route": len(excluded),
        "deferred_identity": len(deferred),
        "route_coverage": ratio(len(completed), len(rows)),
        "confirmed_errors": len(errors),
        "routing_label_counts": count(completed, lambda row: row["human"]["routing_label"]),
        "conditional_error_rate": ratio(len(errors), len(completed)),
        "conditional_error_wilson95": wilson(len(errors), len(completed)),
        "all_sample_confirmed_error_lower_bound": ratio(len(errors), len(rows)),
        "original_attach": {
            "completed": len(original_attach),
            "errors": len(attach_errors),
            "error_rate": ratio(len(attach_errors), len(original_attach)),
            "wilson95": wilson(len(attach_errors), len(original_attach)),
        },
        "original_new": {
            "completed": len(original_new),
            "errors": len(new_errors),
            "error_rate": ratio(len(new_errors), len(original_new)),
            "wilson95": wilson(len(new_errors), len(original_new)),
        },
        "error_case_uids": [row["case_uid"] for row in errors],
    }


def main() -> int:
    args = parse_args()
    root = args.packet_root.resolve()
    output_root = args.output_root.resolve()
    worklist = read_jsonl(root / "worklist.jsonl")
    private_rows = read_jsonl(root / "private_large_worklist.jsonl")
    drafts = read_jsonl(root / "labels" / "blind_drafts.jsonl")
    labels = read_jsonl(root / "labels" / "event_labels.jsonl")

    structural_errors: list[str] = []
    advisory_flags: list[dict[str, Any]] = []

    def unique_by(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for row in rows:
            uid = str(row.get(key) or "")
            if not uid:
                structural_errors.append(f"{name}: missing {key}")
            elif uid in values:
                structural_errors.append(f"{name}: duplicate {key}={uid}")
            else:
                values[uid] = row
        return values

    work_by_case = unique_by(worklist, "case_uid", "worklist")
    private_by_case = unique_by(private_rows, "case_uid", "private_worklist")
    draft_by_case = unique_by(drafts, "case_uid", "blind_drafts")
    label_by_case = unique_by(labels, "case_uid", "event_labels")
    expected = set(work_by_case)
    for name, mapping in (
        ("private_worklist", private_by_case),
        ("blind_drafts", draft_by_case),
        ("event_labels", label_by_case),
    ):
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        if missing:
            structural_errors.append(f"{name}: missing cases {missing}")
        if extra:
            structural_errors.append(f"{name}: extra cases {extra}")

    compiled: list[dict[str, Any]] = []
    private_cases: dict[str, dict[str, Any]] = {}
    for case_uid in sorted(expected & set(private_by_case) & set(label_by_case)):
        work = work_by_case[case_uid]
        private = private_by_case[case_uid]
        label = label_by_case[case_uid]
        case_dir = Path(str(work["case_dir"]))
        case_private_path = case_dir / "case_private.json"
        if not case_private_path.is_file():
            structural_errors.append(f"{case_uid}: missing case_private.json")
            continue
        case_private = read_json(case_private_path)
        private_cases[case_uid] = case_private
        candidate_codes = {str(row["code"]) for row in case_private["candidates"]}
        try:
            blind = validate_blind_label(label["blind"], candidate_codes)
            final = validate_final_label(
                label["final"], blind, str(case_private["original_action_type"])
            )
            derived = derive_routing_label(
                blind,
                final,
                str(case_private["original_action_type"]),
                case_private.get("original_target_code"),
            )
        except Exception as exc:
            structural_errors.append(f"{case_uid}: label validation failed: {exc}")
            continue
        if blind != label["blind"]:
            structural_errors.append(f"{case_uid}: stored blind label is not normalized")
        if final != label["final"]:
            structural_errors.append(f"{case_uid}: stored final label is not normalized")
        if derived != label["derived"]:
            structural_errors.append(f"{case_uid}: stored derived label differs from recomputation")
        draft = draft_by_case.get(case_uid)
        if draft is not None and draft.get("blind") != label.get("blind"):
            structural_errors.append(f"{case_uid}: locked draft differs from final blind label")
        for field in ("event_uid", "source_frame"):
            if str(work.get(field)) != str(label.get(field)):
                structural_errors.append(f"{case_uid}: label/worklist {field} mismatch")
        if str(case_private.get("event_uid")) != str(label.get("event_uid")):
            structural_errors.append(f"{case_uid}: label/case-private event_uid mismatch")
        if label.get("reveal", {}).get("original_action_type") != case_private.get("original_action_type"):
            structural_errors.append(f"{case_uid}: revealed action mismatch")
        if label.get("reveal", {}).get("original_target_code") != case_private.get("original_target_code"):
            structural_errors.append(f"{case_uid}: revealed target mismatch")
        timeline = label.get("timeline") or {}
        s = timeline.get("s_processed_frame_idx")
        d = timeline.get("d_human_submission_mapper_frame")
        if not isinstance(s, int) or not isinstance(d, int) or s > d:
            structural_errors.append(f"{case_uid}: invalid s<=d timeline")
        if timeline.get("h") is not None or timeline.get("c") is not None:
            structural_errors.append(f"{case_uid}: Experiment 0 must keep h/c null")

        code_to_uid = {
            str(row["code"]): str(row["object_uid"])
            for row in case_private["candidates"]
        }
        shown_codes = list(derived["legal_target_codes_shown"])
        human_target_uids = sorted(
            {code_to_uid[code] for code in shown_codes}
            | set(derived["legal_target_uids_outside"])
        )
        confidence = int(final["confidence"])
        blind_seconds = float(blind.get("blind_review_seconds") or 0)
        final_seconds = float(final.get("final_review_seconds") or 0)
        if confidence <= 2:
            advisory_flags.append({"case_uid": case_uid, "kind": "LOW_CONFIDENCE", "value": confidence})
        if blind_seconds < 3 or final_seconds < 1:
            advisory_flags.append({
                "case_uid": case_uid,
                "kind": "VERY_FAST_REVIEW_HEURISTIC",
                "blind_seconds": blind_seconds,
                "final_seconds": final_seconds,
            })

        compiled.append({
            "schema_version": "experiment0-human-routing-event/1.0",
            "case_uid": case_uid,
            "repeat_of": work.get("repeat_of"),
            "event_uid": label["event_uid"],
            "obs_uid": case_private["obs_uid"],
            "scene": label["scene"],
            "source_frame": label["source_frame"],
            "event_frame_idx": label["event_frame_idx"],
            "sample_kind": work.get("sample_kind"),
            "queue_memberships": private.get("private_queue_memberships") or [],
            "original_action_type": case_private["original_action_type"],
            "original_target_code": case_private.get("original_target_code"),
            "original_target_uid": case_private.get("original_target_uid"),
            "created_object_uid": case_private.get("created_object_uid"),
            "frozen_candidates": [
                {
                    "code": row["code"],
                    "object_uid": row["object_uid"],
                    "object_version_uid": row["object_version_uid"],
                    "aggregate_score": row.get("aggregate_score"),
                    "spatial_score": row.get("spatial_score"),
                    "visual_score": row.get("visual_score"),
                }
                for row in case_private["candidates"]
            ],
            "human": {
                "observation_quality": blind["observation_quality"],
                "matching_candidate_codes": blind["matching_candidate_codes"],
                "identity_evidence_status": blind["identity_evidence_status"],
                "physical_instance_note": blind["physical_instance_note"],
                "target_pre_state": final["target_pre_state"],
                "full_map_status": final["full_map_status"],
                "confidence": confidence,
                "causal_note": final["causal_note"],
                "notes": final["notes"],
                "annotation_status": derived["annotation_status"],
                "routing_label": derived["routing_label"],
                "correct_action_type": derived["correct_action_type"],
                "legal_target_uids": human_target_uids,
                "identity_routing_eligible": derived["identity_routing_eligible"],
                "main_set": derived["main_set"],
                "sensitivity_set": derived["sensitivity_set"],
                "episode_review": derived["episode_review"],
                "is_error": derived["is_error"],
            },
            "private_gt_audit": {
                "auto_evaluable": private.get("private_auto_evaluable"),
                "auto_routing_label": private.get("private_auto_routing_label"),
                "auto_episode_role": private.get("private_auto_episode_role"),
                "causal_group_uid": private.get("private_causal_group_uid"),
                "obs_gt_id": private.get("private_obs_gt_id"),
                "obs_gt_label": private.get("private_obs_gt_label"),
                "obs_gt_purity": private.get("private_obs_gt_purity"),
                "legal_candidate_uids": private.get("private_legal_candidate_uids") or [],
                "all_legal_candidates_displayed": (
                    case_private.get("private_full_map_gt_audit") or {}
                ).get("all_legal_candidates_displayed"),
            },
            "timing": {
                "blind_seconds": blind_seconds,
                "final_seconds": final_seconds,
                "timeline": timeline,
            },
        })

    compiled_by_case = {row["case_uid"]: row for row in compiled}
    base_rows = [row for row in compiled if not row["repeat_of"]]
    repeat_rows = [row for row in compiled if row["repeat_of"]]
    repeat_pairs: list[dict[str, Any]] = []
    for repeat in repeat_rows:
        original_uid = str(repeat["repeat_of"])
        original = compiled_by_case.get(original_uid)
        if original is None:
            structural_errors.append(f"{repeat['case_uid']}: missing repeat source {original_uid}")
            continue
        left = label_core(label_by_case[original_uid])
        right = label_core(label_by_case[repeat["case_uid"]])
        fields = {
            key: left[key] == right[key]
            for key in left
        }
        repeat_pairs.append({
            "original": original_uid,
            "repeat": repeat["case_uid"],
            "event_uid_equal": original["event_uid"] == repeat["event_uid"],
            "quality_agreement": fields["observation_quality"],
            "matching_candidates_agreement": fields["matching_candidate_codes"],
            "identity_evidence_agreement": fields["identity_evidence_status"],
            "routing_label_agreement": fields["routing_label"],
            "eligibility_agreement": fields["identity_routing_eligible"],
            "exact_core_agreement": all(fields.values()),
            "disagreeing_fields": sorted(key for key, value in fields.items() if not value),
            "original_core": left,
            "repeat_core": right,
        })

    def agreement(field: str) -> float | None:
        return ratio(sum(bool(row[field]) for row in repeat_pairs), len(repeat_pairs))

    human_auto_pairs = [
        row for row in base_rows
        if row["human"]["annotation_status"] == "COMPLETED"
        and row["private_gt_audit"]["auto_routing_label"] is not None
    ]
    human_auto_agree = [
        row for row in human_auto_pairs
        if row["human"]["routing_label"] == row["private_gt_audit"]["auto_routing_label"]
    ]
    human_auto_conflicts = [
        {
            "case_uid": row["case_uid"],
            "sample_kind": row["sample_kind"],
            "human": row["human"]["routing_label"],
            "private_auto": row["private_gt_audit"]["auto_routing_label"],
            "confidence": row["human"]["confidence"],
            "human_target_uids": row["human"]["legal_target_uids"],
            "private_gt_target_uids": row["private_gt_audit"]["legal_candidate_uids"],
        }
        for row in human_auto_pairs
        if row["human"]["routing_label"] != row["private_gt_audit"]["auto_routing_label"]
    ]

    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        memberships = row["queue_memberships"] or [row["sample_kind"]]
        for queue in memberships:
            queues[str(queue)].append(row)

    probability = queues.get("PROBABILITY_SAMPLE", [])
    labels_quality_ok = (
        not structural_errors
        and len(labels) == len(worklist)
        and len(repeat_pairs) == 17
        and (agreement("routing_label_agreement") or 0) >= 0.90
        and (agreement("eligibility_agreement") or 0) >= 0.90
    )
    status = "PASS" if labels_quality_ok else "REVIEW_REQUIRED"
    report = {
        "schema_version": "experiment0-v2-large-analysis/1.0",
        "status": status,
        "interpretation": (
            "Human labels are authoritative for the event table. Private projected-GT labels "
            "are audit hints only; disagreements require visual adjudication, not automatic overwrite."
        ),
        "completion": {
            "worklist": len(worklist),
            "private_rows": len(private_rows),
            "blind_drafts": len(drafts),
            "final_labels": len(labels),
            "unique_base_events": len(base_rows),
            "hidden_repeats": len(repeat_rows),
        },
        "structural_validation": {
            "error_count": len(structural_errors),
            "errors": structural_errors,
        },
        "human_label_distributions_base_only": {
            "observation_quality": count(base_rows, lambda row: row["human"]["observation_quality"]),
            "identity_evidence_status": count(base_rows, lambda row: row["human"]["identity_evidence_status"]),
            "annotation_status": count(base_rows, lambda row: row["human"]["annotation_status"]),
            "routing_label": count(base_rows, lambda row: row["human"]["routing_label"]),
            "confidence": count(base_rows, lambda row: row["human"]["confidence"]),
            "target_pre_state": count(base_rows, lambda row: row["human"]["target_pre_state"]),
        },
        "hidden_repeat_consistency": {
            "completed_pairs": len(repeat_pairs),
            "quality_agreement": agreement("quality_agreement"),
            "matching_candidates_agreement": agreement("matching_candidates_agreement"),
            "identity_evidence_agreement": agreement("identity_evidence_agreement"),
            "routing_label_agreement": agreement("routing_label_agreement"),
            "eligibility_agreement": agreement("eligibility_agreement"),
            "exact_core_agreement": agreement("exact_core_agreement"),
            "disagreement_pairs": [row for row in repeat_pairs if not row["exact_core_agreement"]],
            "pairs": repeat_pairs,
        },
        "private_gt_audit_comparison": {
            "comparable_completed": len(human_auto_pairs),
            "exact_route_agreement": ratio(len(human_auto_agree), len(human_auto_pairs)),
            "conflict_count": len(human_auto_conflicts),
            "conflicts": human_auto_conflicts,
            "warning": "Private GT is not a replacement for human identity judgement.",
        },
        "queue_analyses": {
            name: queue_summary(rows) for name, rows in sorted(queues.items())
        },
        "experiment0_room0_probability_result": queue_summary(probability),
        "advisory_flags": {
            "count": len(advisory_flags),
            "items": advisory_flags,
            "warning": "Timing and confidence flags are review aids, not automatic invalidations.",
        },
        "next_step_contract": {
            "use_human_event_table": True,
            "exclude_hidden_repeats_from_event_counts": True,
            "prevalence_source": "PROBABILITY_SAMPLE only",
            "episode_compiler_input": "human_routing_events.jsonl",
            "required_next_outputs": [
                "root-versus-cascade episode assignments",
                "future independent-view availability",
                "top-K-plus-NEW human candidate coverage",
                "B0/B1/B2/B3 replay case list",
            ],
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / "human_routing_events.jsonl", base_rows)
    write_json_atomic(output_root / "annotation_quality_and_experiment0.json", report)

    prob = report["experiment0_room0_probability_result"]
    repeat = report["hidden_repeat_consistency"]
    lines = [
        "# Experiment 0 room0 全量标注质量与初步统计",
        "",
        f"结论：`{status}`。人工标注已编译为 `human_routing_events.jsonl`，隐藏重复不进入事件计数。",
        "",
        "## 完整性",
        "",
        f"- 页面/盲标/最终标签：{len(worklist)}/{len(drafts)}/{len(labels)}",
        f"- 独立事件：{len(base_rows)}；隐藏重复：{len(repeat_rows)}",
        f"- 结构或派生逻辑错误：{len(structural_errors)}",
        "",
        "## 隐藏重复一致性",
        "",
        f"- 路由标签：{repeat['routing_label_agreement']:.3f}",
        f"- 是否可纳入路由：{repeat['eligibility_agreement']:.3f}",
        f"- observation quality：{repeat['quality_agreement']:.3f}",
        f"- 同一实例候选集合：{repeat['matching_candidates_agreement']:.3f}",
        f"- 全核心字段完全一致：{repeat['exact_core_agreement']:.3f}",
        "",
        "## room0 自然概率队列（仅此队列可估计发生率）",
        "",
        f"- 抽样事件：{prob['total']}；可裁决路由：{prob['completed_route']}；排除：{prob['excluded_non_route']}；延后：{prob['deferred_identity']}",
        f"- 已确认路由错误：{prob['confirmed_errors']}",
        f"- 在可裁决事件中的错误率：{prob['conditional_error_rate']}",
        f"- 95% Wilson 区间：{prob['conditional_error_wilson95']}",
        f"- 对全部 150 抽样事件的已确认错误下界：{prob['all_sample_confirmed_error_lower_bound']}",
        f"- 错误类型：{json.dumps(prob['routing_label_counts'], ensure_ascii=False)}",
        "",
        "## 解释与下一步",
        "",
        "- 人工标签是后续事件级实验输入；private GT 只用于冲突复核，不能覆盖人工答案。",
        "- 下一步先编译 root/cascade episode，并统计独立未来视角与候选覆盖；随后才选择 B0/B1/B2/B3 replay 案例。",
        "- room0 是开发场景，不能单独给出跨场景 Go 结论。",
        "",
    ]
    (output_root / "ANNOTATION_QUALITY_AND_EXPERIMENT0_CN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "output_root": str(output_root),
        "completed_labels": len(labels),
        "structural_errors": len(structural_errors),
        "repeat_route_agreement": repeat["routing_label_agreement"],
        "probability_completed": prob["completed_route"],
        "probability_errors": prob["confirmed_errors"],
        "probability_conditional_error_rate": prob["conditional_error_rate"],
        "human_auto_conflicts": len(human_auto_conflicts),
    }, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
