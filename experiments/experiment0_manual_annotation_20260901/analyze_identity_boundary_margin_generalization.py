#!/usr/bin/env python3
"""Cross-case audit of the exploratory identity-boundary score margin rule."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "identity-boundary-margin-generalization/1.0"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an object")
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


def owner_for(state: Mapping[str, Any], obs_uid: str) -> str:
    owners = [
        str(entity_uid)
        for entity_uid, members in (state.get("membership") or {}).items()
        if obs_uid in set(str(item) for item in members or ())
    ]
    if len(owners) != 1:
        raise ValueError(f"{obs_uid} expected one owner, got {owners}")
    return owners[0]


def pure_gt_id(row: Mapping[str, Any] | None) -> int | None:
    if not row:
        return None
    if not row.get("gt_assignment_eligible"):
        return None
    if float(row.get("gt_purity") or 0.0) < 0.8:
        return None
    if row.get("mask_mixed") or row.get("mask_two_foreground"):
        return None
    value = row.get("gt_top_id")
    return int(value) if value is not None else None


def adjudicate_gt(
    *,
    row: Mapping[str, Any] | None,
    target_gt_id: int | None,
    trigger_gt_id: int | None,
) -> str:
    observed = pure_gt_id(row)
    if target_gt_id is None or trigger_gt_id is None or observed is None:
        return "GT_UNAVAILABLE_OR_NOT_RELIABLE"
    if target_gt_id == trigger_gt_id:
        return "GT_ID_COLLISION_OR_UNRESOLVED"
    if observed == trigger_gt_id:
        return "NEW_BRANCH_SUPPORTED"
    if observed == target_gt_id:
        return "OLD_BRANCH_SUPPORTED"
    return "OTHER_GT_INSTANCE"


def case_specs(analysis_root: Path) -> list[dict[str, Any]]:
    mixed_root = analysis_root / "mixed_interval_clean_create"
    mixed_intake = read_json(mixed_root / "intake.json")
    specs = [
        {
            "case_uid": "mixed_interval_gt15_gt19",
            "source": "Q3_ORACLE_FILTER_PLUS_ONE_CLEAN_CREATE",
            "state_path": mixed_root / "Q3_filter_plus_clean_create_state.json.gz",
            "target_probe_obs_uid": mixed_intake["target_origin_obs_uid"],
            "trigger_obs_uid": mixed_intake["clean_create_trigger_obs_uid"],
            "trigger_frame": mixed_intake["clean_create_trigger_frame"],
            "created_entity_uid": mixed_intake["constraint"]["created_entity_uid"],
        }
    ]
    for run_name in (
        "oracle_minimal_replay_0143_no_assoc_boundary",
        "oracle_minimal_replay_v2_r2_013_no_assoc_boundary",
    ):
        run_root = analysis_root / run_name
        compiled = read_json(run_root / "compiled_human_oracle_cases.json")[0]
        case_uid = str(compiled["case_uid"])
        specs.append(
            {
                "case_uid": case_uid,
                "source": run_name,
                "state_path": run_root / case_uid / "branches" / "B3.json.gz",
                "target_probe_obs_uid": compiled["target_origin_obs_uid"],
                "trigger_obs_uid": compiled["anchor_obs_uid"],
                "trigger_frame": compiled["anchor_frame"],
                "created_entity_uid": compiled["constraint"]["created_entity_uid"],
            }
        )
    return specs


def analyze_case(
    spec: Mapping[str, Any],
    observation_gt: Mapping[str, Mapping[str, Any]],
    *,
    margin_delta: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = read_gzip_json(Path(spec["state_path"]))
    target_probe = str(spec["target_probe_obs_uid"])
    trigger_obs = str(spec["trigger_obs_uid"])
    old_entity_uid = owner_for(state, target_probe)
    new_entity_uid = str(spec["created_entity_uid"])
    target_gt_id = pure_gt_id(observation_gt.get(target_probe))
    trigger_gt_id = pure_gt_id(observation_gt.get(trigger_obs))

    rows: list[dict[str, Any]] = []
    for decision in state.get("decision_trace") or ():
        frame_idx = int(decision.get("frame_idx", -1))
        if frame_idx <= int(spec["trigger_frame"]):
            continue
        eligible = {
            str(row["entity_uid"]): row
            for row in decision.get("natural_candidates") or ()
            if row.get("entity_uid") and row.get("eligible")
        }
        if old_entity_uid not in eligible or new_entity_uid not in eligible:
            continue
        old = eligible[old_entity_uid]
        new = eligible[new_entity_uid]
        old_score = float(old["score"])
        new_score = float(new["score"])
        old_preferred = decision.get("applied_match") == old.get("index")
        selected = bool(
            old_preferred and 0.0 < old_score - new_score <= margin_delta
        )
        obs_uid = str(decision["obs_uid"])
        gt_row = observation_gt.get(obs_uid)
        rows.append(
            {
                "case_uid": str(spec["case_uid"]),
                "source": str(spec["source"]),
                "obs_uid": obs_uid,
                "frame_idx": frame_idx,
                "old_entity_uid": old_entity_uid,
                "new_entity_uid": new_entity_uid,
                "old_score": old_score,
                "new_score": new_score,
                "old_minus_new_score": old_score - new_score,
                "old_preferred": old_preferred,
                "selected_at_delta": selected,
                "gt_top_id": gt_row.get("gt_top_id") if gt_row else None,
                "gt_purity": gt_row.get("gt_purity") if gt_row else None,
                "gt_adjudication": adjudicate_gt(
                    row=gt_row,
                    target_gt_id=target_gt_id,
                    trigger_gt_id=trigger_gt_id,
                )
                if selected
                else "NOT_SELECTED",
            }
        )

    selected_rows = [row for row in rows if row["selected_at_delta"]]
    old_margins = sorted(
        float(row["old_minus_new_score"])
        for row in rows
        if row["old_preferred"]
    )
    summary = {
        "case_uid": str(spec["case_uid"]),
        "source": str(spec["source"]),
        "trigger_frame": int(spec["trigger_frame"]),
        "target_probe_obs_uid": target_probe,
        "trigger_obs_uid": trigger_obs,
        "target_gt_id": target_gt_id,
        "trigger_gt_id": trigger_gt_id,
        "gt_identity_pair_adjudicable": bool(
            target_gt_id is not None
            and trigger_gt_id is not None
            and target_gt_id != trigger_gt_id
        ),
        "both_boundary_candidates_eligible_count": len(rows),
        "old_preferred_count": sum(bool(row["old_preferred"]) for row in rows),
        "selected_count": len(selected_rows),
        "selected_obs_uids": [row["obs_uid"] for row in selected_rows],
        "selected_gt_adjudication_counts": counts(
            row["gt_adjudication"] for row in selected_rows
        ),
        "minimum_old_preferred_margin": old_margins[0] if old_margins else None,
    }
    return summary, rows


def counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def markdown(metrics: Mapping[str, Any]) -> str:
    supported_margin = metrics["aggregate"]["minimum_gt_supported_new_margin"]
    unresolved_margin = metrics["aggregate"]["minimum_gt_unresolved_margin"]
    supported_text = "N/A" if supported_margin is None else f"{supported_margin:.6f}"
    unresolved_text = "N/A" if unresolved_margin is None else f"{unresolved_margin:.6f}"
    lines = [
        "# 身份边界低分差规则：跨 CREATE 回放泛化审计",
        "",
        f"- 审计阈值：旧分支领先不超过 {metrics['margin_delta']:.3f}。",
        f"- CREATE 回放：{metrics['case_count']} 个。",
        f"- 两边候选都合格：{metrics['aggregate']['both_eligible_count']} 个；旧边胜出：{metrics['aggregate']['old_preferred_count']} 个。",
        f"- 阈值会触发：{metrics['aggregate']['selected_count']} 个。",
        "",
        "| 案例 | 两边合格 | 旧边胜出 | ≤阈值触发 | GT 判断 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in metrics["cases"]:
        judgment = ", ".join(
            f"{key}={value}"
            for key, value in row["selected_gt_adjudication_counts"].items()
        ) or "无触发"
        lines.append(
            f"| {row['case_uid']} | {row['both_boundary_candidates_eligible_count']} | "
            f"{row['old_preferred_count']} | {row['selected_count']} | {judgment} |"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            f"- 已知支持偏向新身份：{metrics['aggregate']['gt_supported_new_count']} 个。",
            f"- 已知应保留旧身份：{metrics['aggregate']['gt_supported_old_count']} 个。",
            f"- 现有 GT 无法裁决：{metrics['aggregate']['gt_unresolved_count']} 个。",
            f"- 已知正确触发的最小分差：{supported_text}；更小的无法裁决分差：{unresolved_text}。",
            "- 因此任何能覆盖已知正确 frame258 的单一分差阈值，也会先覆盖尚无法裁决的 frame350；必须先补标签或加入第二个可在线判断的条件。",
            "- 目前没有观察到已知反例，但样本只有三个 CREATE 回放，且一个触发无法裁决；0.03 不能冻结为正式阈值。",
            "- 正确下一步是补充可裁决的身份边界正例与负例，在不查看 holdout 结果前冻结规则。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--observation-gt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--beauty-summary", type=Path)
    parser.add_argument("--margin-delta", type=float, default=0.03)
    args = parser.parse_args()

    observation_gt = {
        str(row["obs_uid"]): row
        for row in read_jsonl(args.observation_gt.resolve())
    }
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for spec in case_specs(args.analysis_root.resolve()):
        summary, case_rows = analyze_case(
            spec, observation_gt, margin_delta=float(args.margin_delta)
        )
        summaries.append(summary)
        rows.extend(case_rows)

    selected = [row for row in rows if row["selected_at_delta"]]
    adjudication_counts = counts(row["gt_adjudication"] for row in selected)
    supported_new_margins = [
        float(row["old_minus_new_score"])
        for row in selected
        if row["gt_adjudication"] == "NEW_BRANCH_SUPPORTED"
    ]
    unresolved_margins = [
        float(row["old_minus_new_score"])
        for row in selected
        if row["gt_adjudication"]
        in {
            "GT_ID_COLLISION_OR_UNRESOLVED",
            "GT_UNAVAILABLE_OR_NOT_RELIABLE",
            "OTHER_GT_INSTANCE",
        }
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "analysis_semantics": "POST_REPLAY_CROSS_CASE_AUDIT_NO_REPLAY_MUTATION",
        "margin_delta": float(args.margin_delta),
        "case_count": len(summaries),
        "cases": summaries,
        "aggregate": {
            "both_eligible_count": len(rows),
            "old_preferred_count": sum(bool(row["old_preferred"]) for row in rows),
            "selected_count": len(selected),
            "selected_gt_adjudication_counts": adjudication_counts,
            "gt_supported_new_count": adjudication_counts.get(
                "NEW_BRANCH_SUPPORTED", 0
            ),
            "gt_supported_old_count": adjudication_counts.get(
                "OLD_BRANCH_SUPPORTED", 0
            ),
            "gt_unresolved_count": sum(
                adjudication_counts.get(key, 0)
                for key in (
                    "GT_ID_COLLISION_OR_UNRESOLVED",
                    "GT_UNAVAILABLE_OR_NOT_RELIABLE",
                    "OTHER_GT_INSTANCE",
                )
            ),
            "minimum_gt_supported_new_margin": (
                min(supported_new_margins) if supported_new_margins else None
            ),
            "minimum_gt_unresolved_margin": (
                min(unresolved_margins) if unresolved_margins else None
            ),
            "scalar_threshold_confounded_by_lower_margin_unresolved": bool(
                supported_new_margins
                and unresolved_margins
                and min(unresolved_margins) < min(supported_new_margins)
            ),
        },
        "scientific_conclusion": (
            "PROMISING_BUT_INSUFFICIENT_TO_FREEZE_MARGIN_THRESHOLD"
        ),
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "metrics.json", metrics)
    atomic_jsonl(output_root / "boundary_pair_rows.jsonl", rows)
    report = markdown(metrics)
    (output_root / "IDENTITY_BOUNDARY_MARGIN_GENERALIZATION_CN.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    if args.beauty_summary:
        beauty_path = args.beauty_summary.resolve()
        beauty_path.parent.mkdir(parents=True, exist_ok=True)
        beauty_path.write_text(report, encoding="utf-8", newline="\n")
    print(
        f"[done] cases={len(summaries)} both={len(rows)} "
        f"selected={len(selected)} adjudication={adjudication_counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
