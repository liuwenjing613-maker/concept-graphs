#!/usr/bin/env python3
"""Audit final CREATE_INSTANCE replay partitions with offline observation GT.

Future GT is used only after replay to describe the resulting entities.  This
script never changes constraints, replay decisions, or mapper state.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "oracle-create-partition-audit/1.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def owner_index(membership: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for entity_uid, members in membership.items():
        for obs_uid in members or ():
            owners.setdefault(str(obs_uid), []).append(str(entity_uid))
    return owners


def top_counts(values: Iterable[Any], limit: int = 8) -> list[dict[str, Any]]:
    counts = Counter(str(value) for value in values if value is not None)
    total = sum(counts.values())
    return [
        {"value": value, "count": count, "fraction": count / total if total else None}
        for value, count in counts.most_common(limit)
    ]


def summarize_entity(
    members: Iterable[str],
    observation_gt: Mapping[str, Mapping[str, Any]],
    *,
    expected_gt_id: int | None,
) -> dict[str, Any]:
    member_uids = sorted(set(str(item) for item in members))
    rows = [observation_gt[item] for item in member_uids if item in observation_gt]
    eligible = [row for row in rows if row.get("gt_assignment_eligible")]
    expected_count = (
        sum(int(row.get("gt_top_id")) == int(expected_gt_id) for row in eligible)
        if expected_gt_id is not None
        else None
    )
    frames = [int(row["frame_idx"]) for row in rows if row.get("frame_idx") is not None]
    purities = [float(row["gt_purity"]) for row in eligible if row.get("gt_purity") is not None]
    return {
        "member_observation_count": len(member_uids),
        "observation_gt_row_count": len(rows),
        "eligible_gt_observation_count": len(eligible),
        "missing_gt_observation_count": len(member_uids) - len(rows),
        "frame_range": [min(frames), max(frames)] if frames else None,
        "gt_id_counts": top_counts(row.get("gt_top_id") for row in eligible),
        "gt_distinct_id_count": len(
            {row.get("gt_top_id") for row in eligible if row.get("gt_top_id") is not None}
        ),
        "gt_label_counts": top_counts(row.get("gt_top_label") for row in eligible),
        "detector_class_counts": top_counts(row.get("class_name") for row in rows),
        "mean_mask_gt_purity": mean(purities) if purities else None,
        "expected_gt_id": expected_gt_id,
        "expected_gt_observation_count": expected_count,
        "expected_gt_observation_fraction": (
            expected_count / len(eligible)
            if expected_count is not None and eligible
            else None
        ),
    }


def summarize_expected_gt_presence(
    members: Iterable[str],
    observation_gt: Mapping[str, Mapping[str, Any]],
    expected_gt_id: int | None,
) -> dict[str, Any] | None:
    if expected_gt_id is None:
        return None
    rows = [
        observation_gt[str(item)]
        for item in members
        if str(item) in observation_gt
        and observation_gt[str(item)].get("gt_assignment_eligible")
        and observation_gt[str(item)].get("gt_top_id") is not None
    ]
    matching = [
        row for row in rows if int(row["gt_top_id"]) == int(expected_gt_id)
    ]
    matching.sort(key=lambda row: (int(row.get("frame_idx", -1)), str(row["obs_uid"])))
    return {
        "expected_gt_id": int(expected_gt_id),
        "eligible_observation_count": len(rows),
        "matching_observation_count": len(matching),
        "matching_observation_fraction": len(matching) / len(rows) if rows else None,
        "earliest_matching_frame": (
            int(matching[0]["frame_idx"]) if matching else None
        ),
        "earliest_matching_obs_uid": matching[0]["obs_uid"] if matching else None,
    }


def gt_recovery_metrics(
    *,
    predicted_members: Iterable[str],
    affected_native_members: Iterable[str],
    target_members: Iterable[str],
    observation_gt: Mapping[str, Mapping[str, Any]],
    expected_gt_id: int | None,
) -> dict[str, Any] | None:
    if expected_gt_id is None:
        return None
    predicted = set(str(item) for item in predicted_members)
    affected = set(str(item) for item in affected_native_members)
    target = set(str(item) for item in target_members)

    def eligible_with_gt(items: Iterable[str]) -> set[str]:
        return {
            item
            for item in items
            if item in observation_gt
            and observation_gt[item].get("gt_assignment_eligible")
            and observation_gt[item].get("gt_top_id") is not None
        }

    affected_eligible = eligible_with_gt(affected)
    predicted_eligible = eligible_with_gt(predicted)
    expected = {
        item
        for item in affected_eligible
        if int(observation_gt[item]["gt_top_id"]) == int(expected_gt_id)
    }
    true_positive = expected & predicted
    false_positive = predicted_eligible - expected
    false_negative = expected - predicted
    precision = (
        len(true_positive) / len(predicted_eligible) if predicted_eligible else None
    )
    recall = len(true_positive) / len(expected) if expected else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    residual_in_target = expected & target
    return {
        "scope": "EXPECTED_GT_WITHIN_NATIVE_AFFECTED_PARTITION",
        "expected_gt_id": expected_gt_id,
        "affected_native_observation_count": len(affected),
        "affected_native_eligible_gt_observation_count": len(affected_eligible),
        "affected_native_expected_gt_observation_count": len(expected),
        "predicted_new_eligible_gt_observation_count": len(predicted_eligible),
        "true_positive_observation_count": len(true_positive),
        "false_positive_observation_count": len(false_positive),
        "false_negative_observation_count": len(false_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "residual_expected_gt_in_target_count": len(residual_in_target),
        "residual_expected_gt_in_target_fraction": (
            len(residual_in_target) / len(expected) if expected else None
        ),
    }


def audit_branch(
    state: Mapping[str, Any],
    metrics: Mapping[str, Any],
    compiled: Mapping[str, Any],
    episode: Mapping[str, Any],
    observation_gt: Mapping[str, Mapping[str, Any]],
    affected_native_members: Iterable[str],
) -> dict[str, Any]:
    membership = state.get("membership") or {}
    owners = owner_index(membership)
    evaluation = compiled["label_derived_evaluation"]
    new_probes = list(evaluation["corrected_instance_probe_obs_uids"])
    target_probe = str(evaluation["target_probe_obs_uid"])
    new_probe_owners = {uid: owners.get(str(uid), []) for uid in new_probes}
    new_owner_uids = sorted(
        {entity for values in new_probe_owners.values() for entity in values}
    )
    target_owner_uids = sorted(owners.get(target_probe, []))
    new_owner = new_owner_uids[0] if len(new_owner_uids) == 1 else None
    target_owner = target_owner_uids[0] if len(target_owner_uids) == 1 else None
    offline = episode.get("offline_identity_audit") or {}
    future = episode.get("future_evidence") or {}
    expected_new_gt = future.get("gt_id", offline.get("obs_gt_id"))
    expected_target_gt = offline.get("original_target_gt_id")
    recovery_gt_id = (
        int(expected_new_gt)
        if expected_new_gt is not None
        and expected_target_gt is not None
        and int(expected_new_gt) != int(expected_target_gt)
        else None
    )
    new_members = membership.get(new_owner, ()) if new_owner else ()
    target_members = membership.get(target_owner, ()) if target_owner else ()
    recovery = gt_recovery_metrics(
        predicted_members=new_members,
        affected_native_members=affected_native_members,
        target_members=target_members,
        observation_gt=observation_gt,
        expected_gt_id=recovery_gt_id,
    )
    if recovery is None:
        full_instance_recovery = None
        recovery_conclusion = "GT_INSTANCE_COMPLETENESS_UNDETERMINED"
    else:
        full_instance_recovery = bool(
            metrics["endpoint_correct"]
            and recovery["precision"] == 1.0
            and recovery["recall"] == 1.0
            and metrics["collateral"].get("outside_partition_exact_to_native")
        )
        if full_instance_recovery:
            recovery_conclusion = "FULL_INSTANCE_RECOVERY"
        elif (
            metrics["endpoint_correct"]
            and recovery["true_positive_observation_count"] > 0
            and recovery["precision"] == 1.0
            and recovery["recall"] is not None
            and recovery["recall"] < 1.0
        ):
            recovery_conclusion = "PROBE_PASS_HIGH_PRECISION_PARTIAL_RECALL"
        else:
            recovery_conclusion = "INSTANCE_RECOVERY_FAILED"
    return {
        "endpoint_correct": bool(metrics["endpoint_correct"]),
        "root_action_correct": bool(metrics["root_action"]["correct"]),
        "runtime_invariants_pass": metrics["runtime_invariants"].get("pass"),
        "partition_hash": metrics["partition_hash"],
        "replayed_observation_count": metrics["replayed_observation_count"],
        "runtime_ms": metrics["runtime_ms"],
        "outside_partition_exact_to_native": metrics["collateral"].get(
            "outside_partition_exact_to_native"
        ),
        "changed_outside_observation_count": metrics["collateral"].get(
            "changed_outside_observation_count"
        ),
        "new_probe_owner_uids": new_probe_owners,
        "new_probe_group_has_one_owner": new_owner is not None,
        "new_owner_uid": new_owner,
        "target_owner_uid": target_owner,
        "new_and_target_are_disjoint": bool(
            new_owner and target_owner and new_owner != target_owner
        ),
        "new_owner": summarize_entity(
            new_members,
            observation_gt,
            expected_gt_id=int(expected_new_gt) if expected_new_gt is not None else None,
        )
        if new_owner
        else None,
        "target_owner": summarize_entity(
            target_members,
            observation_gt,
            expected_gt_id=(
                int(expected_target_gt) if expected_target_gt is not None else None
            ),
        )
        if target_owner
        else None,
        "expected_gt_recovery": recovery,
        "expected_gt_recovery_status": (
            "APPLICABLE_DISTINCT_NEW_AND_TARGET_GT"
            if recovery_gt_id is not None
            else "NOT_APPLICABLE_TARGET_GT_UNAVAILABLE_OR_NOT_DISTINCT"
        ),
        "full_instance_recovery": full_instance_recovery,
        "instance_recovery_conclusion": recovery_conclusion,
        "persistent_create_instance_association_veto_count": state.get(
            "persistent_create_instance_association_veto_count"
        ),
        "persistent_create_instance_merge_veto_count": state.get(
            "persistent_create_instance_merge_veto_count"
        ),
    }


def markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# CREATE_INSTANCE 回放分区审计",
        "",
        "离线 GT 只用于回放结束后的结果审计，不参与约束或在线决策。",
        "",
        "| 病例 | 分支 | endpoint | 新实例观测 | 新实例精度 | 整实例召回率 | 残留目标比例 | 范围外精确不变 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        for branch_name, branch in case["branches"].items():
            entity = branch.get("new_owner") or {}
            recovery = branch.get("expected_gt_recovery") or {}
            precision = recovery.get("precision")
            recall = recovery.get("recall")
            residual = recovery.get("residual_expected_gt_in_target_fraction")
            lines.append(
                "| {case} | {branch} | {endpoint} | {members} | {precision} | "
                "{recall} | {residual} | {outside} |".format(
                    case=case["case_uid"],
                    branch=branch_name,
                    endpoint=branch["endpoint_correct"],
                    members=entity.get("member_observation_count"),
                    precision=(f"{precision:.3f}" if precision is not None else "N/A"),
                    recall=(f"{recall:.3f}" if recall is not None else "N/A"),
                    residual=(f"{residual:.3f}" if residual is not None else "N/A"),
                    outside=branch["outside_partition_exact_to_native"],
                )
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 精度与召回率按原始受影响分区中的可靠 observation 计算，不按像素加权。",
            "- endpoint 只检查人工挑选的探针；整实例召回率用于识别“探针成功但大部分同实例观测仍残留”的情况。",
            "- 若原目标 GT 不可用，只能证明人工证据分组被稳定分开，不能声称 GT 实例级分离。",
            "- B2 与 B3 分区一致仍需结合运行时不变量和范围外副作用共同判断。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--observation-gt", type=Path, required=True)
    parser.add_argument("--object-versions", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    gt_rows = read_jsonl(args.observation_gt.resolve())
    observation_gt = {str(row["obs_uid"]): row for row in gt_rows}
    object_versions = {
        str(row["object_version_uid"]): row
        for row in read_jsonl(args.object_versions.resolve())
    }
    episodes = {
        str(row["case_uid"]): row for row in read_jsonl(args.episodes.resolve())
    }
    cases: list[dict[str, Any]] = []
    for run_root in (path.resolve() for path in args.run_root):
        aggregate = json.loads((run_root / "aggregate_metrics.json").read_text())
        compiled_rows = json.loads(
            (run_root / "compiled_human_oracle_cases.json").read_text()
        )
        compiled_by_case = {str(row["case_uid"]): row for row in compiled_rows}
        for case_metrics in aggregate["cases"]:
            case_uid = str(case_metrics["case_uid"])
            compiled = compiled_by_case[case_uid]
            if str(compiled["correct_action_type"]) != "NEW":
                continue
            episode = episodes[case_uid]
            offline = episode.get("offline_identity_audit") or {}
            future = episode.get("future_evidence") or {}
            expected_new_gt = future.get("gt_id", offline.get("obs_gt_id"))
            expected_target_gt = offline.get("original_target_gt_id")
            target_version_uid = str(compiled["target_version_uid"])
            target_version = object_versions[target_version_uid]
            target_version_members = target_version.get("member_observation_uids") or []
            b0r_path = run_root / case_uid / "branches" / "B0R.json.gz"
            with gzip.open(b0r_path, "rt", encoding="utf-8") as handle:
                b0r_state = json.load(handle)
            b0r_membership = b0r_state.get("membership") or {}
            b0r_owners = owner_index(b0r_membership)
            evaluation = compiled["label_derived_evaluation"]
            affected_owner_uids = {
                entity_uid
                for obs_uid in [
                    *evaluation["corrected_instance_probe_obs_uids"],
                    evaluation["target_probe_obs_uid"],
                ]
                for entity_uid in b0r_owners.get(str(obs_uid), ())
            }
            affected_native_members = {
                str(obs_uid)
                for entity_uid in affected_owner_uids
                for obs_uid in b0r_membership.get(entity_uid, ())
            }
            branches: dict[str, Any] = {}
            for branch_name in ("B2", "B3"):
                branch_path = run_root / case_uid / "branches" / f"{branch_name}.json.gz"
                with gzip.open(branch_path, "rt", encoding="utf-8") as handle:
                    state = json.load(handle)
                branches[branch_name] = audit_branch(
                    state,
                    case_metrics["branches"][branch_name],
                    compiled,
                    episode,
                    observation_gt,
                    affected_native_members,
                )
            cases.append(
                {
                    "case_uid": case_uid,
                    "source_run_root": str(run_root),
                    "human_confidence": episode.get("human_confidence"),
                    "human_target_pre_state": episode.get("human_target_pre_state"),
                    "offline_identity_audit": offline,
                    "target_pre_anchor_version_uid": target_version_uid,
                    "target_pre_anchor_version": summarize_entity(
                        target_version_members,
                        observation_gt,
                        expected_gt_id=(
                            int(expected_target_gt)
                            if expected_target_gt is not None
                            else None
                        ),
                    ),
                    "expected_new_gt_already_in_target_pre_anchor": (
                        summarize_expected_gt_presence(
                            target_version_members,
                            observation_gt,
                            int(expected_new_gt) if expected_new_gt is not None else None,
                        )
                    ),
                    "native_affected_owner_uids": sorted(affected_owner_uids),
                    "native_affected_partition": summarize_entity(
                        affected_native_members,
                        observation_gt,
                        expected_gt_id=(
                            int(expected_new_gt)
                            if expected_new_gt is not None
                            else None
                        ),
                    ),
                    "b2_b3_partition_exact": (
                        branches["B2"]["partition_hash"]
                        == branches["B3"]["partition_hash"]
                    ),
                    "branches": branches,
                }
            )

    result = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(cases),
        "observation_gt_use": "POST_REPLAY_EVALUATION_ONLY",
        "cases": cases,
    }
    output_root = args.output_root.resolve()
    atomic_json(output_root / "create_partition_audit.json", result)
    (output_root / "CREATE_PARTITION_AUDIT_CN.md").write_text(
        markdown_report(result), encoding="utf-8", newline="\n"
    )
    print(
        f"[done] cases={len(cases)} output={output_root} ",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
