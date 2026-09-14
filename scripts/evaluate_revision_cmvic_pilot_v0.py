#!/usr/bin/env python3
"""Posthoc exploratory evaluation for the frozen room0 CMVIC pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.auto_constraints import semantic_constraint_fingerprint
from conceptgraph.revision.evidence_split import sha256_file


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object")
            rows.append(value)
    return rows


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _percentile_ranks(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (float(row[field]), row["row_uid"]))
    if len(ordered) == 1:
        return {ordered[0]["row_uid"]: 0.5}
    return {
        row["row_uid"]: index / (len(ordered) - 1) for index, row in enumerate(ordered)
    }


def _risk_coverage(
    rows: Iterable[Mapping[str, Any]], *, method: str, score_field: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row[score_field]), str(row["row_uid"])),
    )
    curve = []
    false_commits = 0
    for index, row in enumerate(ordered, 1):
        false_commits += int(int(row["beneficial_label"]) == 0)
        risk = false_commits / index
        curve.append(
            {
                "method": method,
                "rank": index,
                "coverage": index / len(ordered),
                "risk": risk,
                "case_uid": row["case_uid"],
                "candidate_uid": row["candidate_uid"],
                "score": float(row[score_field]),
                "beneficial_label": int(row["beneficial_label"]),
            }
        )
    label_classes = {int(row["beneficial_label"]) for row in ordered}
    comparable = len(ordered) >= 2 and len(label_classes) == 2
    return (
        {
            "method": method,
            "labeled_candidate_count": len(ordered),
            "positive_count": sum(int(row["beneficial_label"]) for row in ordered),
            "negative_count": sum(1 - int(row["beneficial_label"]) for row in ordered),
            "aurc": (
                sum(item["risk"] for item in curve) / len(curve)
                if curve and comparable
                else None
            ),
            "status": (
                "EXPLORATORY"
                if comparable
                else "ONE_CLASS_ONLY"
                if ordered
                else "INSUFFICIENT_LABELS"
            ),
        },
        curve,
    )


def _critic_preferences(
    *,
    critic_results: Iterable[Path],
    execution_by_case: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], dict[str, Any]],]:
    responses = {}
    for path in critic_results:
        payload = _read(path.resolve())
        for row in payload.get("results") or ():
            if row.get("status") == "PASS":
                responses[str(row["request_uid"])] = row
    values: dict[tuple[str, str], list[float]] = {}
    for case_uid, execution in execution_by_case.items():
        for request_uid, mapping in (
            execution.get("critic_state_mappings") or {}
        ).items():
            response = responses.get(str(request_uid))
            if response is None:
                continue
            preferred = str(response["response"]["critic"]["preferred_state"]).upper()
            label_to_state = {
                str(key).upper(): str(value)
                for key, value in mapping["label_to_state_uid"].items()
            }
            candidate_state = str(mapping["candidate_state_uid"])
            noop_state = str(mapping["noop_state_uid"])
            preferred_state = label_to_state.get(preferred)
            value = (
                1.0
                if preferred_state == candidate_state
                else -1.0
                if preferred_state == noop_state
                else 0.0
            )
            key = (case_uid, str(mapping["candidate_uid"]))
            values.setdefault(key, []).append(value)
    preferences = {}
    audits = {}
    for key, items in values.items():
        stable = len(items) >= 2 and len(set(items)) == 1
        audits[key] = {
            "response_count": len(items),
            "mapped_preferences": list(items),
            "order_consistent": stable,
        }
        if stable:
            preferences[key] = items[0]
    return preferences, audits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="append", required=True, type=Path)
    parser.add_argument("--critic-results", action="append", default=[], type=Path)
    parser.add_argument("--pilot-selection-private", required=True, type=Path)
    parser.add_argument("--human-manifest", required=True, type=Path)
    parser.add_argument(
        "--endpoint-labels",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument("--clean-posthoc-key", required=True, type=Path)
    parser.add_argument("--roundtrip-audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    protocol_paths = [path.resolve() for path in args.freeze_protocol]
    protocols = [_read(path) for path in protocol_paths]
    case_results = {}
    execution_by_case = {}
    protocol_audit = []
    for path, protocol in zip(protocol_paths, protocols):
        if protocol.get("runtime_human_or_gold_loaded") is not False:
            raise ValueError(f"runtime oracle isolation failed: {path}")
        if protocol.get("production_commit_permitted") is not False:
            raise ValueError(f"production commit was not disabled: {path}")
        protocol_audit.append({"path": str(path), "sha256": sha256_file(path)})
        for case_row in protocol.get("cases") or ():
            result_path = Path(str(case_row["result_path"])).resolve()
            result = _read(result_path)
            case_uid = str(result["case_uid"])
            case_results[case_uid] = result
            execution_path = result_path.parent / "execution.private.json"
            if execution_path.is_file():
                execution_by_case[case_uid] = _read(execution_path)

    selection_private_path = args.pilot_selection_private.resolve()
    selection_private = _read(selection_private_path)
    machine_source = {
        str(row["case_uid"]): str(row["source_incident_uid"])
        for row in selection_private.get("eligible_cases") or ()
    }
    clean_key_path = args.clean_posthoc_key.resolve()
    clean_key = _read(clean_key_path)
    clean_source = {
        str(row["blind_case_uid"]): str(row["source_case_uid"])
        for row in clean_key.get("cases") or ()
    }
    human_path = args.human_manifest.resolve()
    human = _read(human_path)
    human_by_incident = {
        str(row["incident_uid"]): row for row in human.get("cases") or ()
    }
    human_by_anchor = {
        str(row["anchor_obs_uid"]): row
        for row in human.get("cases") or ()
        if row.get("anchor_obs_uid")
    }
    beneficial_fingerprints = {
        incident_uid: {
            semantic_constraint_fingerprint(constraint)
            for constraint in row.get("constraints") or ()
            if constraint.get("type") == "CREATE_INSTANCE"
        }
        for incident_uid, row in human_by_incident.items()
    }

    endpoint_label_paths = [path.resolve() for path in args.endpoint_labels]
    endpoint_states: dict[str, set[str]] = {}
    for path in endpoint_label_paths:
        for row in _read_jsonl(path):
            if str(row.get("evidence_sufficient") or "").upper() != "YES":
                continue
            incident_uid = str(row.get("incident_uid") or row.get("case_uid") or "")
            final_state = str(row.get("final_state") or "").upper()
            if not incident_uid or final_state not in {"CORRECT", "WRONG"}:
                continue
            endpoint_states.setdefault(incident_uid, set()).add(final_state)

    def endpoint_consensus(incident_uid: str | None) -> str | None:
        if incident_uid is None:
            return None
        states = endpoint_states.get(incident_uid, set())
        return next(iter(states)) if len(states) == 1 else None

    critic_preferences, critic_audits = _critic_preferences(
        critic_results=args.critic_results,
        execution_by_case=execution_by_case,
    )
    candidate_rows = []
    case_rows = []
    for case_uid, result in sorted(case_results.items()):
        execution = execution_by_case.get(case_uid, {})
        binding_path = execution.get("binding_path")
        anchor_obs_uid = None
        if binding_path and Path(str(binding_path)).is_file():
            anchor_obs_uid = str(_read(Path(str(binding_path)))["obs_uid"])
        source_incident_uid = machine_source.get(case_uid) or clean_source.get(case_uid)
        if source_incident_uid is None and anchor_obs_uid in human_by_anchor:
            source_incident_uid = str(human_by_anchor[anchor_obs_uid]["incident_uid"])
        role = (
            "BLIND_MACHINE_PILOT"
            if case_uid in machine_source
            else "MECHANISM_CLEAN_CONTROL"
            if case_uid in clean_source
            else "MECHANISM_POSITIVE_CONTROL"
        )
        comparisons = {
            str(row["candidate_uid"]): row
            for row in result.get("cmvic_comparisons") or ()
        }
        for score in result.get("primary_candidate_scores") or ():
            candidate_uid = str(score["candidate_uid"])
            comparison = comparisons[candidate_uid]
            assignment_advantage = float(
                score["diagnostics"]["diagnostic_assignment_likelihood"]["advantage"]
            )
            label = None
            label_source = None
            if case_uid in clean_source:
                label = 0
                label_source = "POSTHOC_CLEAN_CONTROL"
            elif (
                source_incident_uid is not None
                and source_incident_uid in human_by_incident
            ):
                label = int(
                    candidate_uid
                    in beneficial_fingerprints.get(source_incident_uid, set())
                )
                label_source = "POSTHOC_HUMAN_CONSTRAINT_FINGERPRINT"
            elif endpoint_consensus(source_incident_uid) == "CORRECT":
                label = 0
                label_source = "POSTHOC_ENDPOINT_LABEL_CONSENSUS_CORRECT"
            row_uid = hashlib.sha256(
                f"{case_uid}:{candidate_uid}".encode("utf-8")
            ).hexdigest()[:20]
            critic_audit = critic_audits.get((case_uid, candidate_uid), {})
            candidate_rows.append(
                {
                    "row_uid": row_uid,
                    "case_uid": case_uid,
                    "candidate_uid": candidate_uid,
                    "role": role,
                    "source_incident_uid": source_incident_uid,
                    "beneficial_label": label,
                    "label_source": label_source,
                    "observable": bool(comparison["observable"]),
                    "projected_difference_pixel_count": int(
                        comparison["candidate"]["projected_difference_pixel_count"]
                    ),
                    "cmvic_advantage": float(score["score_advantage_over_noop"]),
                    "assignment_advantage": assignment_advantage,
                    "vlm_preference": critic_preferences.get((case_uid, candidate_uid)),
                    "vlm_response_count": int(critic_audit.get("response_count") or 0),
                    "vlm_mapped_preferences": list(
                        critic_audit.get("mapped_preferences") or ()
                    ),
                    "vlm_order_consistent": (
                        bool(critic_audit["order_consistent"]) if critic_audit else None
                    ),
                    "runtime_valid": bool(score["valid"]),
                }
            )
        timing = result.get("timing") or {}
        case_rows.append(
            {
                "case_uid": case_uid,
                "role": role,
                "status": result["status"],
                "distinct_candidate_count": int(
                    result.get("distinct_repair_partition_count") or 0
                ),
                "observable_candidate_count": int(
                    result.get("observable_candidate_count") or 0
                ),
                "case_total_wall_ms": float(timing.get("case_total_wall_ms") or 0.0),
                "projection_verifier_wall_ms": float(
                    timing.get("projection_verifier_wall_ms") or 0.0
                ),
                "critic_request_count": len(result.get("critic_requests") or ()),
            }
        )

    labeled_machine = [
        row
        for row in candidate_rows
        if row["role"] == "BLIND_MACHINE_PILOT" and row["beneficial_label"] is not None
    ]
    if labeled_machine:
        cmvic_ranks = _percentile_ranks(labeled_machine, "cmvic_advantage")
        vlm_labeled = [
            row for row in labeled_machine if row["vlm_preference"] is not None
        ]
        vlm_ranks = (
            _percentile_ranks(vlm_labeled, "vlm_preference") if vlm_labeled else {}
        )
        for row in labeled_machine:
            row["cmvic_plus_vlm_rank_score"] = (
                (cmvic_ranks[row["row_uid"]] + vlm_ranks[row["row_uid"]]) / 2.0
                if row["row_uid"] in vlm_ranks
                else None
            )

    method_specs = (
        ("CMVIC", "cmvic_advantage"),
        ("ASSIGNMENT_LIKELIHOOD", "assignment_advantage"),
        ("VLM_CRITIC", "vlm_preference"),
        ("CMVIC_PLUS_VLM_RANK", "cmvic_plus_vlm_rank_score"),
    )
    risk_summaries = []
    curve_rows = []
    for method, field in method_specs:
        available = [row for row in labeled_machine if row.get(field) is not None]
        summary, curve = _risk_coverage(available, method=method, score_field=field)
        risk_summaries.append(summary)
        curve_rows.extend(curve)

    all_distinct = sum(row["distinct_candidate_count"] for row in case_rows)
    all_observable = sum(row["observable_candidate_count"] for row in case_rows)
    machine_cases = [row for row in case_rows if row["role"] == "BLIND_MACHINE_PILOT"]
    machine_distinct = sum(row["distinct_candidate_count"] for row in machine_cases)
    machine_observable = sum(row["observable_candidate_count"] for row in machine_cases)
    case_times = [row["case_total_wall_ms"] for row in case_rows]
    projection_times = [row["projection_verifier_wall_ms"] for row in case_rows]
    positive = [
        row for row in candidate_rows if row["role"] == "MECHANISM_POSITIVE_CONTROL"
    ]
    clean = [row for row in candidate_rows if row["role"] == "MECHANISM_CLEAN_CONTROL"]
    positive_best = max((row["cmvic_advantage"] for row in positive), default=None)
    clean_best = max((row["cmvic_advantage"] for row in clean), default=None)
    converted_assignment_unobservable = sum(
        row["observable"] and abs(row["assignment_advantage"]) <= 1e-12
        for row in candidate_rows
    )
    roundtrip_path = args.roundtrip_audit.resolve()
    roundtrip = _read(roundtrip_path)
    roundtrip_pass = str(roundtrip.get("status") or "").startswith("PASS")

    stop_reasons = []
    if not roundtrip_pass:
        stop_reasons.append("PROJECTION_ROUNDTRIP_NOT_PROVEN")
    if any(
        len(protocol.get("evidence_policy_uids") or ()) > 1 for protocol in protocols
    ):
        stop_reasons.append("MULTIPLE_EVIDENCE_POLICY_UIDS_WITHIN_PROTOCOL")
    fix_reasons = []
    if positive_best is None or positive_best <= 0.0:
        fix_reasons.append("POSITIVE_CONTROL_NOT_FAVORED")
    if clean_best is not None and clean_best > 0.0:
        fix_reasons.append("CLEAN_CONTROL_SEPARATION_FAVORED")
    if not all_observable:
        fix_reasons.append("NO_COUNTERFACTUAL_OBSERVABILITY")
    machine_aurc = {row["method"]: row["aurc"] for row in risk_summaries}
    if (
        machine_aurc.get("CMVIC") is not None
        and machine_aurc.get("ASSIGNMENT_LIKELIHOOD") is not None
        and machine_aurc["CMVIC"] > machine_aurc["ASSIGNMENT_LIKELIHOOD"]
    ):
        fix_reasons.append("CMVIC_MACHINE_AURC_WORSE_THAN_ASSIGNMENT")
    if not labeled_machine:
        fix_reasons.append("MACHINE_HOLDOUT_LABEL_OVERLAP_EMPTY")
    elif len({row["beneficial_label"] for row in labeled_machine}) < 2:
        fix_reasons.append("MACHINE_HOLDOUT_LABELS_ONE_CLASS_ONLY")
    if any(
        row["vlm_response_count"] and row["vlm_order_consistent"] is False
        for row in candidate_rows
    ):
        fix_reasons.append("VLM_ORDER_SWAP_INCONSISTENT")
    decision = "STOP" if stop_reasons else "FIX" if fix_reasons else "GO"

    failure_cases = [
        {
            **row,
            "failure_tags": [
                *(["COUNTERFACTUAL_UNOBSERVABLE"] if not row["observable"] else []),
                *(
                    ["LABELED_BENEFICIAL_NOT_FAVORED"]
                    if row["beneficial_label"] == 1 and row["cmvic_advantage"] <= 0.0
                    else []
                ),
                *(
                    ["LABELED_NONBENEFICIAL_FAVORED"]
                    if row["beneficial_label"] == 0 and row["cmvic_advantage"] > 0.0
                    else []
                ),
            ],
        }
        for row in candidate_rows
        if (
            not row["observable"]
            or (row["beneficial_label"] == 1 and row["cmvic_advantage"] <= 0.0)
            or (row["beneficial_label"] == 0 and row["cmvic_advantage"] > 0.0)
        )
    ]

    results = {
        "schema_version": "1.0.0",
        "evaluation_role": "POSTHOC_EXPLORATORY_CMVIC_PILOT",
        "decision": decision,
        "stop_reasons": stop_reasons,
        "fix_reasons": fix_reasons,
        "headline": {
            "risk_coverage": risk_summaries,
            "counterfactual_observability_coverage": (
                all_observable / all_distinct if all_distinct else None
            ),
            "machine_observability_coverage": (
                machine_observable / machine_distinct if machine_distinct else None
            ),
            "case_total_wall_time_p50_ms": (
                statistics.median(case_times) if case_times else None
            ),
            "projection_verifier_wall_time_p50_ms": (
                statistics.median(projection_times) if projection_times else None
            ),
            "vlm_call_count": sum(row["critic_request_count"] for row in case_rows),
        },
        "mechanism_controls": {
            "positive_control_best_cmvic_advantage": positive_best,
            "clean_control_best_cmvic_advantage": clean_best,
            "assignment_equal_but_projection_observable_count": (
                converted_assignment_unobservable
            ),
            "vlm_order_inconsistent_candidate_count": sum(
                row["vlm_order_consistent"] is False for row in candidate_rows
            ),
        },
        "sample_accounting": {
            "case_count": len(case_rows),
            "candidate_count": len(candidate_rows),
            "machine_case_count": len(machine_cases),
            "machine_labeled_candidate_count": len(labeled_machine),
            "machine_label_overlap_is_sufficient_for_conclusion": (
                len(labeled_machine) >= 4
                and len({row["beneficial_label"] for row in labeled_machine}) == 2
            ),
            "population_inference_forbidden": True,
        },
        "candidate_rows": candidate_rows,
        "case_rows": case_rows,
        "artifact_provenance": {
            "freeze_protocols": protocol_audit,
            "selection_private": {
                "path": str(selection_private_path),
                "sha256": sha256_file(selection_private_path),
            },
            "human_manifest": {
                "path": str(human_path),
                "sha256": sha256_file(human_path),
            },
            "endpoint_labels": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in endpoint_label_paths
            ],
            "clean_posthoc_key": {
                "path": str(clean_key_path),
                "sha256": sha256_file(clean_key_path),
            },
            "roundtrip_audit": {
                "path": str(roundtrip_path),
                "sha256": sha256_file(roundtrip_path),
            },
        },
        "production_commit_count": 0,
        "calibration_ready": False,
        "exploratory_not_significance_claim": True,
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "CMVIC_PILOT_RESULTS.json", results)
    _write(
        output_root / "failure_cases.json",
        {
            "schema_version": "1.0.0",
            "failure_case_count": len(failure_cases),
            "cases": failure_cases,
        },
    )
    curve_path = output_root / "risk_coverage.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "rank",
            "coverage",
            "risk",
            "case_uid",
            "candidate_uid",
            "score",
            "beneficial_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)

    report = [
        "# CMVIC V0 Pilot Report",
        "",
        f"Decision: **{decision}**",
        "",
        "This is an exploratory room0 mechanism pilot, not a significance or population claim.",
        "",
        "## Headline results",
        "",
        f"- Counterfactual observability: {all_observable}/{all_distinct}.",
        f"- Machine observability: {machine_observable}/{machine_distinct}.",
        f"- Positive-control best ΔCMVIC: {positive_best}.",
        f"- Clean-control best ΔCMVIC: {clean_best}.",
        f"- Case wall-time p50: {results['headline']['case_total_wall_time_p50_ms']} ms.",
        f"- Projection wall-time p50: {results['headline']['projection_verifier_wall_time_p50_ms']} ms.",
        f"- Frozen VLM call count: {results['headline']['vlm_call_count']}.",
        "",
        "## Risk–coverage",
        "",
    ]
    for summary in risk_summaries:
        report.append(
            f"- {summary['method']}: AURC={summary['aurc']}, "
            f"n={summary['labeled_candidate_count']} ({summary['status']})."
        )
    report.extend(
        [
            "",
            "## Decision reasons",
            "",
            f"- STOP reasons: {stop_reasons or ['none']}.",
            f"- FIX reasons: {fix_reasons or ['none']}.",
            "",
            "Calibration remains unready and production commits remain zero.",
            "",
        ]
    )
    (output_root / "CMVIC_PILOT_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision,
                "case_count": len(case_rows),
                "candidate_count": len(candidate_rows),
                "machine_labeled_candidate_count": len(labeled_machine),
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
