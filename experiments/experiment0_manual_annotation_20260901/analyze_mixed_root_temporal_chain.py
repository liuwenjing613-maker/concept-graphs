#!/usr/bin/env python3
"""Audit the temporal evidence around the GT15/GT19 mixed-instance root.

The corrected instance GT is used only after replay.  This script does not
define an online detector; it establishes whether one-frame quarantine can be
expected to recover naturally and identifies the smallest informative oracle
follow-up experiment.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "mixed-root-temporal-audit/1.0"
DEFAULT_GT_IDS = (15, 19)
DEFAULT_TARGET_ORIGIN = "room0_20260831T111035Z_5c9d86fa_f000123_r0016"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def is_pair_mixed(row: Mapping[str, Any], gt_ids: set[int]) -> bool:
    observed = {
        int(value)
        for value in (row.get("gt_top_id"), row.get("gt_second_id"))
        if value is not None
    }
    return gt_ids.issubset(observed) and bool(
        row.get("mask_mixed") or row.get("mask_two_foreground")
    )


def is_pure_reliable(row: Mapping[str, Any], gt_ids: set[int]) -> bool:
    top_id = row.get("gt_top_id")
    return bool(
        row.get("gt_assignment_eligible")
        and top_id is not None
        and int(top_id) in gt_ids
        and float(row.get("gt_purity") or 0.0) >= 0.8
        and not row.get("mask_mixed")
        and not row.get("mask_two_foreground")
    )


def contiguous_ranges(frames: Iterable[int]) -> list[dict[str, int]]:
    values = sorted(set(int(item) for item in frames))
    if not values:
        return []
    result: list[dict[str, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append(
                {"start_frame": start, "end_frame": previous, "frame_count": previous - start + 1}
            )
            start = value
        previous = value
    result.append(
        {"start_frame": start, "end_frame": previous, "frame_count": previous - start + 1}
    )
    return result


def decision_view(decision: Mapping[str, Any] | None) -> dict[str, Any]:
    if decision is None:
        return {
            "decision_present": False,
            "applied_match": None,
            "applied_target_origin_obs_uid": None,
            "natural_match": None,
            "natural_target_origin_obs_uid": None,
            "best_natural_score": None,
            "best_natural_eligible": None,
        }
    candidates = decision.get("natural_candidates") or []
    best = candidates[0] if candidates else {}
    return {
        "decision_present": True,
        "applied_match": decision.get("applied_match"),
        "applied_target_origin_obs_uid": decision.get(
            "applied_target_origin_obs_uid"
        ),
        "natural_match": decision.get("natural_match"),
        "natural_target_origin_obs_uid": decision.get(
            "natural_target_origin_obs_uid"
        ),
        "best_natural_score": best.get("score"),
        "best_natural_eligible": best.get("eligible"),
    }


def format_ranges(ranges: Iterable[Mapping[str, int]]) -> str:
    values = []
    for row in ranges:
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        values.append(str(start) if start == end else f"{start}–{end}")
    return ", ".join(values) if values else "无"


def markdown(metrics: Mapping[str, Any]) -> str:
    mixed = metrics["mixed_pair_observations"]
    pure = metrics["pure_reliable_observations"]
    routing = metrics["routing"]
    lines = [
        "# Room0 GT15/GT19 混合根因时序审计",
        "",
        "## 结论",
        "",
        "**frame138 不是单帧偶发错误，而是持续出现的分割混合。只隔离 frame138 不可能让实例自然恢复。**",
        "",
        f"- 从 frame{metrics['anchor_frame']} 起，共发现 **{mixed['count']}** 个同时包含 GT15 与 GT19 的混合 observation。",
        f"- 混合最早到最晚：frame{mixed['first_frame']}–frame{mixed['last_frame']}；出现区间：{format_ranges(mixed['contiguous_frame_ranges'])}。",
        f"- 严格纯净证据共有 **{pure['total_count']}** 个：GT15={pure['per_gt']['15']['count']}，GT19={pure['per_gt']['19']['count']}。",
        f"- 首个纯净 GT15 在 frame{pure['per_gt']['15']['first_frame']}；首个纯净 GT19 在 frame{pure['per_gt']['19']['first_frame']}。",
        f"- 这 {pure['total_count']} 个纯净 observation 中，有 **{routing['pure_to_original_target_count']}** 个仍 ATTACH 到同一个旧实体。",
        f"- 最后一个混合 observation 之后的纯净证据：GT15={pure['per_gt']['15']['count_after_last_mixed']}，GT19={pure['per_gt']['19']['count_after_last_mixed']}。",
        "",
        "## 对下一步实验的含义",
        "",
        "1. 单帧删除无效：后续混合 mask 会继续污染旧实体。",
        "2. 只等纯净帧也不够：首个纯净 GT19 仍被原 matcher 高分并入旧实体，没有自行产生 NEW。",
        "3. 最小高信息量上限实验应同时包含两件事：拒收 GT15/GT19 混合 observation；在首个纯净 GT19 处只触发一次 CREATE_INSTANCE，随后让纯净 evidence 按原 matcher 自然回放。",
        "4. 上述 GT 只用于离线选样和结果评测；即便上限实验成功，也不等于在线混合检测已经实现。",
        "",
        "## 评测边界",
        "",
        "- 纯净/可靠定义：gt_assignment_eligible=true、purity≥0.8，且非 MIXED、非 two-foreground。",
        "- 本文件只分析 replay 完成后的校正 GT，不向 replay 注入逐帧身份答案。",
        f"- 回放状态：`{metrics['replay_state_status']}`；decision trace 覆盖 {metrics['decision_trace_count']} 个 observation。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-gt", required=True, type=Path)
    parser.add_argument("--replay-state", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--beauty-summary", type=Path)
    parser.add_argument("--anchor-frame", type=int, default=138)
    parser.add_argument("--target-origin", default=DEFAULT_TARGET_ORIGIN)
    args = parser.parse_args()

    gt_ids = set(DEFAULT_GT_IDS)
    gt_rows = read_jsonl(args.observation_gt.resolve())
    replay_state = read_gzip_json(args.replay_state.resolve())
    decisions = {
        str(row["obs_uid"]): row
        for row in replay_state.get("decision_trace") or []
        if row.get("obs_uid")
    }

    rows_after_anchor = [
        row for row in gt_rows if int(row.get("frame_idx", -1)) >= args.anchor_frame
    ]
    pair_mixed = [row for row in rows_after_anchor if is_pair_mixed(row, gt_ids)]
    pure = [row for row in rows_after_anchor if is_pure_reliable(row, gt_ids)]
    if not pair_mixed:
        raise ValueError("no GT15/GT19 mixed observations found after anchor")

    last_mixed_frame = max(int(row["frame_idx"]) for row in pair_mixed)
    temporal_rows: list[dict[str, Any]] = []
    for row in sorted(pair_mixed + pure, key=lambda item: (int(item["frame_idx"]), str(item["obs_uid"]))):
        category = "PAIR_MIXED" if is_pair_mixed(row, gt_ids) else "PURE_RELIABLE"
        temporal_rows.append(
            {
                "obs_uid": str(row["obs_uid"]),
                "frame_idx": int(row["frame_idx"]),
                "raw_frame": row.get("raw_frame"),
                "category": category,
                "gt_top_id": row.get("gt_top_id"),
                "gt_top_fraction": row.get("gt_purity"),
                "gt_second_id": row.get("gt_second_id"),
                "gt_second_fraction": row.get("gt_second_fraction"),
                "mask_mixed": bool(row.get("mask_mixed")),
                "mask_two_foreground": bool(row.get("mask_two_foreground")),
                **decision_view(decisions.get(str(row["obs_uid"]))),
            }
        )

    mixed_temporal = [row for row in temporal_rows if row["category"] == "PAIR_MIXED"]
    pure_temporal = [row for row in temporal_rows if row["category"] == "PURE_RELIABLE"]
    pure_by_gt: dict[str, Any] = {}
    for gt_id in sorted(gt_ids):
        subset = [row for row in pure_temporal if int(row["gt_top_id"]) == gt_id]
        frames = [int(row["frame_idx"]) for row in subset]
        pure_by_gt[str(gt_id)] = {
            "count": len(subset),
            "frames": frames,
            "first_frame": min(frames) if frames else None,
            "last_frame": max(frames) if frames else None,
            "count_after_last_mixed": sum(frame > last_mixed_frame for frame in frames),
            "to_original_target_count": sum(
                row["applied_target_origin_obs_uid"] == args.target_origin
                for row in subset
            ),
        }

    mixed_target_counts = Counter(
        str(row["applied_target_origin_obs_uid"])
        for row in mixed_temporal
        if row["decision_present"]
    )
    pure_target_counts = Counter(
        str(row["applied_target_origin_obs_uid"])
        for row in pure_temporal
        if row["decision_present"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "analysis_semantics": "POST_REPLAY_CORRECTED_GT_EVALUATION_ONLY",
        "affected_gt_ids": sorted(gt_ids),
        "anchor_frame": args.anchor_frame,
        "target_origin_obs_uid": args.target_origin,
        "replay_state_status": replay_state.get("status"),
        "decision_trace_count": len(decisions),
        "mixed_pair_observations": {
            "count": len(pair_mixed),
            "unique_frame_count": len({int(row["frame_idx"]) for row in pair_mixed}),
            "first_frame": min(int(row["frame_idx"]) for row in pair_mixed),
            "last_frame": last_mixed_frame,
            "contiguous_frame_ranges": contiguous_ranges(
                int(row["frame_idx"]) for row in pair_mixed
            ),
            "with_decision_count": sum(row["decision_present"] for row in mixed_temporal),
            "to_original_target_count": sum(
                row["applied_target_origin_obs_uid"] == args.target_origin
                for row in mixed_temporal
            ),
        },
        "pure_reliable_observations": {
            "total_count": len(pure_temporal),
            "per_gt": pure_by_gt,
        },
        "routing": {
            "mixed_applied_target_origin_counts": dict(mixed_target_counts.most_common()),
            "pure_applied_target_origin_counts": dict(pure_target_counts.most_common()),
            "pure_to_original_target_count": sum(
                row["applied_target_origin_obs_uid"] == args.target_origin
                for row in pure_temporal
            ),
            "all_pure_to_original_target": bool(pure_temporal)
            and all(
                row["applied_target_origin_obs_uid"] == args.target_origin
                for row in pure_temporal
            ),
        },
        "next_experiment": {
            "name": "ORACLE_MIXED_FILTER_PLUS_FIRST_CLEAN_GT19_CREATE",
            "quarantine_observation_count": len(pair_mixed),
            "single_create_trigger_obs_uid": next(
                (
                    str(row["obs_uid"])
                    for row in pure_temporal
                    if int(row["gt_top_id"]) == 19
                ),
                None,
            ),
            "online_claim_allowed": False,
            "reason": (
                "Separate the segmentation-quality ceiling from routing: filter "
                "recurring mixed evidence, create once on the first clean GT19, "
                "then measure natural suffix association."
            ),
        },
        "scientific_conclusion": (
            "RECURRING_MIXED_SEGMENTATION_AND_NO_SPONTANEOUS_IDENTITY_SPLIT"
        ),
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "metrics.json", metrics)
    atomic_jsonl(output_root / "temporal_rows.jsonl", temporal_rows)
    report = markdown(metrics)
    report_path = output_root / "ROOM0_GT15_GT19_TEMPORAL_ROOT_AUDIT_CN.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    if args.beauty_summary:
        beauty_path = args.beauty_summary.resolve()
        beauty_path.parent.mkdir(parents=True, exist_ok=True)
        beauty_path.write_text(report, encoding="utf-8", newline="\n")

    print(
        "[done] mixed={} pure={} pure_to_old={} next_trigger={}".format(
            len(pair_mixed),
            len(pure_temporal),
            metrics["routing"]["pure_to_original_target_count"],
            metrics["next_experiment"]["single_create_trigger_obs_uid"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
