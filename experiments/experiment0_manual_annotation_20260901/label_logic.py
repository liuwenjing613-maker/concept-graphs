#!/usr/bin/env python3
"""Shared validation and deterministic label derivation for Experiment 0."""

from __future__ import annotations

from typing import Any


OBSERVATION_QUALITIES = {
    "CLEAN_SINGLE_INSTANCE",
    "BORDERLINE_SINGLE_INSTANCE",
    "MIXED_MULTIPLE_INSTANCES",
    "BACKGROUND_OR_FRAGMENT",
    "DUPLICATE_PROPOSAL_SAME_FRAME",
    "DYNAMIC_POSE_DEPTH_ERROR",
    "GRANULARITY_AMBIGUOUS",
    "INSUFFICIENT",
}

TARGET_STATES = {
    "CLEAN_SINGLE_INSTANCE",
    "ALREADY_CONTAMINATED",
    "UNCERTAIN",
}

OUTSIDE_STATUSES = {
    "NOT_NEEDED",
    "MATCH_EXISTS_OUTSIDE",
    "NO_MATCHING_NODE_EXISTS",
    "UNCHECKED",
}

EVIDENCE_STATUSES = {"YES", "PARTIAL", "NO"}

SPECIAL_MATCHES = {"NONE_SHOWN", "UNCERTAIN"}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def validate_blind_label(payload: dict[str, Any], candidate_codes: set[str]) -> dict[str, Any]:
    """Validate the identity judgement made before the mapper choice is revealed."""

    quality = str(payload.get("observation_quality") or "")
    if quality not in OBSERVATION_QUALITIES:
        raise ValueError("请选择当前 observation 的质量类型")

    raw_matches = payload.get("matching_candidate_codes")
    if not isinstance(raw_matches, list) or not raw_matches:
        raise ValueError("请选择同一物理实例候选；看不清时选 UNCERTAIN")
    matches = sorted({str(value) for value in raw_matches})
    invalid = sorted(set(matches) - candidate_codes - SPECIAL_MATCHES)
    if invalid:
        raise ValueError("候选代码不合法：" + ", ".join(invalid))
    special = set(matches) & SPECIAL_MATCHES
    if special and len(matches) != 1:
        raise ValueError("NONE_SHOWN/UNCERTAIN 不能与其他候选同时选择")

    try:
        seconds = round(float(payload.get("blind_review_seconds", 0)), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("盲标用时必须是非负数") from exc
    if seconds < 0:
        raise ValueError("盲标用时必须是非负数")

    return {
        "observation_quality": quality,
        "matching_candidate_codes": matches,
        "physical_instance_note": _clean_text(payload.get("physical_instance_note")),
        "blind_review_seconds": seconds,
    }


def validate_final_label(payload: dict[str, Any], blind: dict[str, Any]) -> dict[str, Any]:
    """Validate the post-reveal adjudication fields."""

    target_state = str(payload.get("target_state") or "")
    if target_state not in TARGET_STATES:
        raise ValueError("请选择系统所选节点在关联前的状态")
    outside = str(payload.get("outside_candidate_status") or "")
    if outside not in OUTSIDE_STATUSES:
        raise ValueError("请选择候选集外是否已有同一实例节点")
    evidence = str(payload.get("evidence_sufficient") or "")
    if evidence not in EVIDENCE_STATUSES:
        raise ValueError("请选择证据是否足够")

    try:
        confidence = int(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度必须是 1–5") from exc
    if confidence < 1 or confidence > 5:
        raise ValueError("置信度必须是 1–5")

    try:
        seconds = round(float(payload.get("final_review_seconds", 0)), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("揭示后复核用时必须是非负数") from exc
    if seconds < 0:
        raise ValueError("揭示后复核用时必须是非负数")

    notes = _clean_text(payload.get("notes"))
    matches = set(blind["matching_candidate_codes"])
    if matches == {"UNCERTAIN"} and outside != "UNCHECKED":
        raise ValueError("盲标为 UNCERTAIN 时，候选集外状态应选 UNCHECKED")
    if matches - SPECIAL_MATCHES and outside != "NOT_NEEDED":
        raise ValueError("已经选出匹配候选时，候选集外状态应选 NOT_NEEDED")
    if matches == {"NONE_SHOWN"} and outside == "NOT_NEEDED":
        raise ValueError("没有展示匹配候选时，必须说明候选集外是否存在正确节点")
    if evidence in {"PARTIAL", "NO"} and not notes:
        raise ValueError("证据为 PARTIAL/NO 时，请在备注中写清缺少什么")

    return {
        "target_state": target_state,
        "outside_candidate_status": outside,
        "evidence_sufficient": evidence,
        "confidence": confidence,
        "final_review_seconds": seconds,
        "notes": notes,
    }


def derive_event_label(
    blind: dict[str, Any],
    final: dict[str, Any],
    selected_target_code: str,
) -> dict[str, Any]:
    """Derive the scientific label; the reviewer never types KEEP/REASSIGN/NEW."""

    quality = blind["observation_quality"]
    matches = set(blind["matching_candidate_codes"])
    target_state = final["target_state"]
    outside = final["outside_candidate_status"]
    evidence = final["evidence_sufficient"]

    if evidence == "NO" or quality == "INSUFFICIENT":
        status = "DEFER_EVIDENCE"
        action = "DEFER"
        eligible_main = False
    elif quality not in {"CLEAN_SINGLE_INSTANCE", "BORDERLINE_SINGLE_INSTANCE"}:
        status = f"EXCLUDE_{quality}"
        action = "EXCLUDE"
        eligible_main = False
    elif evidence == "PARTIAL" or matches == {"UNCERTAIN"} or target_state == "UNCERTAIN":
        status = "DEFER_AMBIGUOUS"
        action = "DEFER"
        eligible_main = False
    elif target_state == "ALREADY_CONTAMINATED":
        status = "CASCADE_OR_PRECONTAMINATED"
        action = "DEFER"
        eligible_main = False
    elif selected_target_code in matches:
        status = "CORRECT_KEEP"
        action = "KEEP"
        eligible_main = True
    elif matches - SPECIAL_MATCHES:
        status = "ROOT_FALSE_ATTACH_REASSIGN"
        action = "REASSIGN"
        eligible_main = True
    elif matches == {"NONE_SHOWN"} and outside == "MATCH_EXISTS_OUTSIDE":
        status = "ROOT_FALSE_ATTACH_REASSIGN_OUTSIDE_TOPK"
        action = "REASSIGN"
        eligible_main = True
    elif matches == {"NONE_SHOWN"} and outside == "NO_MATCHING_NODE_EXISTS":
        status = "ROOT_FALSE_ATTACH_NEW"
        action = "NEW"
        eligible_main = True
    else:
        status = "DEFER_ACTION_TARGET"
        action = "DEFER"
        eligible_main = False

    is_root = status.startswith("ROOT_FALSE_ATTACH_")
    return {
        "derived_status": status,
        "derived_action": action,
        "eligible_main": eligible_main,
        "is_root_false_attach": is_root,
        "main_set": quality == "CLEAN_SINGLE_INSTANCE" and eligible_main,
        "sensitivity_set": quality in {
            "CLEAN_SINGLE_INSTANCE",
            "BORDERLINE_SINGLE_INSTANCE",
        }
        and eligible_main,
    }

