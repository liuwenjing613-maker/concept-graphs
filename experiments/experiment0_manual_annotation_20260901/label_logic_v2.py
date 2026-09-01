#!/usr/bin/env python3
"""Validation and deterministic identity-routing labels for Experiment 0 v2."""

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

ELIGIBLE_QUALITIES = {
    "CLEAN_SINGLE_INSTANCE",
    "BORDERLINE_SINGLE_INSTANCE",
}

IDENTITY_EVIDENCE_STATUSES = {
    "SUFFICIENT_FOR_IDENTITY",
    "PARTIAL",
    "INSUFFICIENT",
}

ORIGINAL_ACTION_TYPES = {"ATTACH_EXISTING", "NEW"}

TARGET_PRE_STATES = {
    "CLEAN_SINGLE_INSTANCE",
    "ALREADY_CONTAMINATED",
    "UNCERTAIN",
    "NOT_APPLICABLE",
}

FULL_MAP_STATUSES = {
    "NOT_NEEDED_MATCH_SHOWN",
    "MATCH_EXISTS_OUTSIDE",
    "NO_MATCHING_NODE_EXISTS",
    "UNCHECKED",
}

SPECIAL_MATCHES = {"NONE_SHOWN", "UNCERTAIN"}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _nonnegative_seconds(value: Any, field_name: str) -> float:
    try:
        seconds = round(float(value or 0), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}必须是非负数") from exc
    if seconds < 0:
        raise ValueError(f"{field_name}必须是非负数")
    return seconds


def validate_blind_label(
    payload: dict[str, Any], candidate_codes: set[str]
) -> dict[str, Any]:
    """Validate the mapper-blind observation and physical-identity judgement."""

    quality = str(payload.get("observation_quality") or "")
    if quality not in OBSERVATION_QUALITIES:
        raise ValueError("请选择当前 observation 的质量类型")

    evidence = str(payload.get("identity_evidence_status") or "")
    if evidence not in IDENTITY_EVIDENCE_STATUSES:
        raise ValueError("请选择身份判断的证据状态")

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
    if matches == ["UNCERTAIN"] and evidence == "SUFFICIENT_FOR_IDENTITY":
        raise ValueError("身份为 UNCERTAIN 时，证据不能标为 SUFFICIENT_FOR_IDENTITY")
    if quality == "INSUFFICIENT" and evidence != "INSUFFICIENT":
        raise ValueError("observation 为 INSUFFICIENT 时，身份证据也必须为 INSUFFICIENT")

    physical_note = _clean_text(payload.get("physical_instance_note"))
    if quality == "GRANULARITY_AMBIGUOUS" and not physical_note:
        raise ValueError(
            "GRANULARITY_AMBIGUOUS 必须说明无法决定的 part-whole 边界；"
            "仅仅只看到实例的一部分不属于粒度歧义"
        )

    return {
        "observation_quality": quality,
        "matching_candidate_codes": matches,
        "identity_evidence_status": evidence,
        "physical_instance_note": physical_note,
        "blind_review_seconds": _nonnegative_seconds(
            payload.get("blind_review_seconds"), "盲标用时"
        ),
    }


def validate_final_label(
    payload: dict[str, Any],
    blind: dict[str, Any],
    original_action_type: str,
) -> dict[str, Any]:
    """Validate post-reveal fields without asking the reviewer for a route label."""

    if original_action_type not in ORIGINAL_ACTION_TYPES:
        raise ValueError("原始路由动作不合法")

    target_pre_state = str(payload.get("target_pre_state") or "")
    if target_pre_state not in TARGET_PRE_STATES:
        raise ValueError("请选择原始目标在事件前的状态")
    if original_action_type == "NEW" and target_pre_state != "NOT_APPLICABLE":
        raise ValueError("原始动作为 NEW 时，目标前状态必须为 NOT_APPLICABLE")
    if original_action_type == "ATTACH_EXISTING" and target_pre_state == "NOT_APPLICABLE":
        raise ValueError("原始动作为 ATTACH 时，必须判断目标前状态")

    full_map_status = str(payload.get("full_map_status") or "")
    if full_map_status not in FULL_MAP_STATUSES:
        raise ValueError("请选择完整 t^- 地图检查结果")

    raw_outside = payload.get("outside_matching_node_uids") or []
    if not isinstance(raw_outside, list):
        raise ValueError("候选外节点 UID 必须是列表")
    outside_uids = sorted({str(value).strip() for value in raw_outside if str(value).strip()})

    matches = set(blind["matching_candidate_codes"])
    if matches == {"UNCERTAIN"} and full_map_status != "UNCHECKED":
        raise ValueError("盲标为 UNCERTAIN 时，完整地图状态必须为 UNCHECKED")
    if matches - SPECIAL_MATCHES and full_map_status != "NOT_NEEDED_MATCH_SHOWN":
        raise ValueError("已经选出展示候选时，完整地图状态应为 NOT_NEEDED_MATCH_SHOWN")
    if matches == {"NONE_SHOWN"} and full_map_status == "NOT_NEEDED_MATCH_SHOWN":
        raise ValueError("未展示匹配节点时，必须给出完整地图检查结果")
    if full_map_status == "MATCH_EXISTS_OUTSIDE" and not outside_uids:
        raise ValueError("MATCH_EXISTS_OUTSIDE 必须填写至少一个事件时节点 UID")
    if full_map_status != "MATCH_EXISTS_OUTSIDE" and outside_uids:
        raise ValueError("只有 MATCH_EXISTS_OUTSIDE 可以填写候选外节点 UID")

    try:
        confidence = int(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度必须是 1–5") from exc
    if not 1 <= confidence <= 5:
        raise ValueError("置信度必须是 1–5")

    notes = _clean_text(payload.get("notes"))
    evidence = blind["identity_evidence_status"]
    if evidence in {"PARTIAL", "INSUFFICIENT"} and not notes:
        raise ValueError("证据为 PARTIAL/INSUFFICIENT 时，请写明缺失证据")

    return {
        "target_pre_state": target_pre_state,
        "full_map_status": full_map_status,
        "outside_matching_node_uids": outside_uids,
        "confidence": confidence,
        "causal_note": _clean_text(payload.get("causal_note")),
        "notes": notes,
        "final_review_seconds": _nonnegative_seconds(
            payload.get("final_review_seconds"), "揭示后复核用时"
        ),
    }


def derive_routing_label(
    blind: dict[str, Any],
    final: dict[str, Any],
    original_action_type: str,
    original_target_code: str | None,
) -> dict[str, Any]:
    """Derive one of the five routing cells while keeping causality separate."""

    if original_action_type not in ORIGINAL_ACTION_TYPES:
        raise ValueError("原始路由动作不合法")
    if original_action_type == "ATTACH_EXISTING" and not original_target_code:
        raise ValueError("ATTACH_EXISTING 缺少原始目标候选代码")
    if original_action_type == "NEW" and original_target_code is not None:
        raise ValueError("NEW 不应绑定事件前目标候选代码")

    quality = blind["observation_quality"]
    evidence = blind["identity_evidence_status"]
    matches = set(blind["matching_candidate_codes"])
    full_map_status = final["full_map_status"]
    shown_targets = sorted(matches - SPECIAL_MATCHES)
    outside_targets = list(final["outside_matching_node_uids"])

    if quality not in ELIGIBLE_QUALITIES:
        annotation_status = "EXCLUDED"
        correct_action = "NOT_APPLICABLE"
        routing_label = f"OUT_OF_SCOPE_{quality}"
    elif (
        evidence != "SUFFICIENT_FOR_IDENTITY"
        or matches == {"UNCERTAIN"}
        or full_map_status == "UNCHECKED"
        or final["target_pre_state"] == "UNCERTAIN"
    ):
        annotation_status = "DEFERRED"
        correct_action = "UNDETERMINED"
        routing_label = "UNDETERMINED"
    else:
        annotation_status = "COMPLETED"
        has_existing_target = bool(shown_targets or outside_targets)
        correct_action = "ATTACH_EXISTING" if has_existing_target else "NEW"
        if original_action_type == "ATTACH_EXISTING":
            if original_target_code in shown_targets:
                routing_label = "CORRECT_ATTACH"
            elif has_existing_target:
                routing_label = "WRONG_ATTACH_EXISTING"
            else:
                routing_label = "SHOULD_HAVE_BEEN_NEW"
        elif has_existing_target:
            routing_label = "WRONG_NEW_FALSE_SPLIT"
        else:
            routing_label = "CORRECT_NEW"

    if annotation_status != "COMPLETED":
        episode_review = "NOT_APPLICABLE"
    elif final["target_pre_state"] == "ALREADY_CONTAMINATED":
        episode_review = "PRECONTAMINATED_REQUIRES_CAUSAL_REVIEW"
    elif routing_label in {
        "WRONG_ATTACH_EXISTING",
        "SHOULD_HAVE_BEEN_NEW",
        "WRONG_NEW_FALSE_SPLIT",
    }:
        episode_review = "ROOT_OR_CASCADE_PENDING_COMPILER"
    else:
        episode_review = "NO_ROUTING_ERROR"

    return {
        "annotation_status": annotation_status,
        "routing_label": routing_label,
        "correct_action_type": correct_action,
        "legal_target_codes_shown": shown_targets,
        "legal_target_uids_outside": outside_targets,
        "identity_routing_eligible": annotation_status == "COMPLETED",
        "main_set": annotation_status == "COMPLETED"
        and quality == "CLEAN_SINGLE_INSTANCE",
        "sensitivity_set": annotation_status == "COMPLETED"
        and quality in ELIGIBLE_QUALITIES,
        "episode_review": episode_review,
        "is_error": routing_label
        in {
            "WRONG_ATTACH_EXISTING",
            "SHOULD_HAVE_BEEN_NEW",
            "WRONG_NEW_FALSE_SPLIT",
        },
    }
