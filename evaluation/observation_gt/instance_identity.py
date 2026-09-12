from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


IDENTITY_RELIABLE = "RELIABLE"
IDENTITY_MIXED = "MIXED_PREDICTION"
IDENTITY_UNKNOWN = "UNKNOWN_PREDICTION"


@dataclass(frozen=True)
class PredictionIdentityConfig:
    min_prediction_identity_purity: float = 0.80

    def validate(self) -> None:
        if not 0.0 <= self.min_prediction_identity_purity <= 1.0:
            raise ValueError("min_prediction_identity_purity must be in [0, 1]")


def infer_prediction_identity(
    object_uid: str,
    member_obs_uids: Iterable[str],
    observation_labels: Mapping[str, Mapping[str, Any]],
    config: PredictionIdentityConfig,
) -> dict[str, Any]:
    config.validate()
    members = tuple(sorted(set(str(item) for item in member_obs_uids)))
    scores: dict[int, float] = defaultdict(float)
    support_counts: dict[int, int] = defaultdict(int)
    matched_point_counts: dict[int, int] = defaultdict(int)
    clean_members: list[str] = []
    missing_members: list[str] = []
    for obs_uid in members:
        label = observation_labels.get(obs_uid)
        if label is None:
            missing_members.append(obs_uid)
            continue
        if label.get("quality_status") != "CLEAN":
            continue
        gt_id = label.get("assigned_gt_instance")
        if gt_id is None:
            continue
        gt_id = int(gt_id)
        weight = float(label.get("top1_purity", 0.0)) * float(
            label.get("gt_support_ratio", 0.0)
        )
        scores[gt_id] += weight
        support_counts[gt_id] += 1
        matched_point_counts[gt_id] += int(label.get("matched_gt_points", 0))
        clean_members.append(obs_uid)
    if not scores:
        return {
            "object_uid": str(object_uid),
            "member_observation_count": len(members),
            "clean_support_count": 0,
            "missing_label_count": len(missing_members),
            "identity_status": IDENTITY_UNKNOWN,
            "gt_instance": None,
            "identity_purity": None,
            "dominant_score": 0.0,
            "total_score": 0.0,
            "dominant_support_count": 0,
            "dominant_matched_gt_points": 0,
            "low_support_identity": True,
            "gt_scores": {},
        }
    ranking = sorted(
        scores,
        key=lambda gt_id: (
            -scores[gt_id],
            -support_counts[gt_id],
            -matched_point_counts[gt_id],
            gt_id,
        ),
    )
    dominant = int(ranking[0])
    total = float(sum(scores.values()))
    purity = float(scores[dominant] / total) if total else 0.0
    status = (
        IDENTITY_RELIABLE
        if purity >= config.min_prediction_identity_purity
        else IDENTITY_MIXED
    )
    return {
        "object_uid": str(object_uid),
        "member_observation_count": len(members),
        "clean_support_count": len(clean_members),
        "missing_label_count": len(missing_members),
        "identity_status": status,
        "gt_instance": dominant,
        "identity_purity": purity,
        "dominant_score": float(scores[dominant]),
        "total_score": total,
        "dominant_support_count": int(support_counts[dominant]),
        "dominant_matched_gt_points": int(matched_point_counts[dominant]),
        "low_support_identity": len(clean_members) == 1,
        "gt_scores": {
            str(gt_id): {
                "score": float(scores[gt_id]),
                "support_count": int(support_counts[gt_id]),
                "matched_gt_points": int(matched_point_counts[gt_id]),
            }
            for gt_id in ranking
        },
    }


def infer_state_identities(
    state: Mapping[str, Iterable[str]],
    observation_labels: Mapping[str, Mapping[str, Any]],
    config: PredictionIdentityConfig,
    object_uids: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected = sorted(set(object_uids if object_uids is not None else state))
    return {
        object_uid: infer_prediction_identity(
            object_uid,
            state.get(object_uid, ()),
            observation_labels,
            config,
        )
        for object_uid in selected
    }


def canonical_predictions(
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[int, str]:
    by_gt: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for identity in identities.values():
        if identity.get("identity_status") != IDENTITY_RELIABLE:
            continue
        gt_id = identity.get("gt_instance")
        if gt_id is not None:
            by_gt[int(gt_id)].append(identity)
    result: dict[int, str] = {}
    for gt_id, candidates in by_gt.items():
        winner = min(
            candidates,
            key=lambda row: (
                -float(row["dominant_score"]),
                -int(row["dominant_support_count"]),
                -int(row["dominant_matched_gt_points"]),
                str(row["object_uid"]),
            ),
        )
        result[gt_id] = str(winner["object_uid"])
    return result


class MappingReplayState:
    """Replay immutable mapping events to recover pre-association membership."""

    def __init__(self) -> None:
        self.members: dict[str, set[str]] = {}

    def apply(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("event_type") or "")
        if event_type == "OBJECT_CREATE":
            object_uid = str(event["object_uid"])
            obs_uid = str(event["obs_uid"])
            if object_uid in self.members:
                raise ValueError(f"OBJECT_CREATE reuses live object {object_uid}")
            self.members[object_uid] = {obs_uid}
        elif event_type == "OBS_ASSOCIATE":
            object_uid = str(event["object_uid"])
            obs_uid = str(event["obs_uid"])
            if object_uid not in self.members:
                raise ValueError(f"OBS_ASSOCIATE targets unknown object {object_uid}")
            self.members[object_uid].add(obs_uid)
        elif event_type == "OBS_DISCARD":
            return
        elif event_type == "OBJECT_DENOISE":
            object_uid = str(event["object_uid"])
            after = event.get("after_summary") or {}
            listed = after.get("member_observation_uids")
            if listed is not None:
                if object_uid not in self.members:
                    raise ValueError(f"OBJECT_DENOISE targets unknown object {object_uid}")
                expected = set(str(item) for item in listed)
                if expected != self.members[object_uid]:
                    raise ValueError(
                        f"OBJECT_DENOISE member drift for {object_uid}: "
                        f"state={len(self.members[object_uid])}, evidence={len(expected)}"
                    )
        elif event_type == "OBJECT_MERGE":
            source_uid = str(event["source_object_uid"])
            target_uid = str(event["target_object_uid"])
            if source_uid not in self.members or target_uid not in self.members:
                raise ValueError(
                    f"OBJECT_MERGE references unknown objects {source_uid}->{target_uid}"
                )
            union = set(self.members[target_uid]) | set(self.members[source_uid])
            documented = event.get("member_union_after")
            if documented is not None and union != set(str(item) for item in documented):
                raise ValueError(f"OBJECT_MERGE member union drift for {source_uid}->{target_uid}")
            self.members[target_uid] = union
            del self.members[source_uid]
        else:
            raise ValueError(f"unsupported mapping event type: {event_type!r}")

    def validate_object_order(self, object_uids: Iterable[str]) -> dict[str, Any]:
        expected = [str(item) for item in object_uids]
        missing = [item for item in expected if item not in self.members]
        extra = [item for item in self.members if item not in set(expected)]
        return {
            "expected_object_count": len(expected),
            "replayed_object_count": len(self.members),
            "missing_from_replay": missing,
            "extra_in_replay": sorted(extra),
            "membership_state_valid": not missing and not extra,
        }
