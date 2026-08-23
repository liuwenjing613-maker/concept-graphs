from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Iterable, Mapping

from .index import ProvenanceIndex


_OBS_SUFFIX = re.compile(r"_f(?P<frame>\d+)_r(?P<raw>\d+)$")


def canonical_obs_key(obs_uid: str) -> str:
    """Return a run-independent observation key for controlled reruns."""
    match = _OBS_SUFFIX.search(str(obs_uid))
    if not match:
        raise ValueError(f"invalid observation uid: {obs_uid}")
    return f"f{int(match.group('frame')):06d}_r{int(match.group('raw')):04d}"


def frame_index(obs_uid: str) -> int:
    match = _OBS_SUFFIX.search(str(obs_uid))
    if not match:
        raise ValueError(f"invalid observation uid: {obs_uid}")
    return int(match.group("frame"))


def stable_entity_uid(members: Iterable[str], *, prefix: str = "derived") -> str:
    canonical = sorted(canonical_obs_key(item) for item in members)
    digest = hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def canonical_membership(
    membership: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    result = {}
    for entity_uid, members in membership.items():
        values = tuple(sorted(dict.fromkeys(str(item) for item in members)))
        if values:
            result[str(entity_uid)] = values
    return dict(sorted(result.items()))


def invert_membership(
    membership: Mapping[str, Iterable[str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for entity_uid, members in membership.items():
        for obs_uid in members:
            if obs_uid in result:
                raise ValueError(f"observation has multiple owners: {obs_uid}")
            result[str(obs_uid)] = str(entity_uid)
    return result


def apply_controlled_membership_corruption(
    clean_membership: Mapping[str, Iterable[str]], case: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    """Apply the named intervention to memberships, never to final geometry."""
    groups = {key: list(values) for key, values in canonical_membership(clean_membership).items()}
    failure_type = str(case["failure_type"]).upper()
    obs_uid = str(case["obs_uid"])
    source = str(case["source_identity_uid"])
    target = case.get("target_identity_uid")

    if source not in groups or obs_uid not in groups[source]:
        raise ValueError("case source observation is not owned by source identity")

    if failure_type == "FALSE_SPLIT":
        groups[source].remove(obs_uid)
        split_uid = str(case.get("corrupted_entity_uid") or stable_entity_uid([obs_uid], prefix="split"))
        groups[split_uid] = [obs_uid]
    elif failure_type == "WRONG_MEMBERSHIP":
        if not target or str(target) not in groups:
            raise ValueError("wrong-membership case requires a valid target identity")
        groups[source].remove(obs_uid)
        groups[str(target)].append(obs_uid)
    elif failure_type == "FALSE_MERGE":
        if not target or str(target) not in groups or str(target) == source:
            raise ValueError("false-merge case requires a distinct target identity")
        groups[str(target)].extend(groups.pop(source))
    else:
        raise ValueError(f"unsupported failure type: {failure_type}")
    return canonical_membership(groups)


class ControlledCaseBuilder:
    """Choose deterministic, real association events from a frozen ledger."""

    def __init__(self, provenance: ProvenanceIndex) -> None:
        self.provenance = provenance
        self.clean_membership = canonical_membership(
            {
                str(row["object_uid"]): row.get("member_observation_uids") or ()
                for row in provenance.final_membership
            }
        )
        self.obs_to_final = invert_membership(self.clean_membership)

    def _identity_for_version(self, version_uid: str | None) -> str | None:
        if not version_uid or version_uid not in self.provenance.object_versions:
            return None
        members = self.provenance.get_member_observations(version_uid)
        votes = Counter(self.obs_to_final[item] for item in members if item in self.obs_to_final)
        return votes.most_common(1)[0][0] if votes else None

    def _candidate_identity(
        self, association: Mapping[str, Any], object_uid: str
    ) -> tuple[str | None, str | None]:
        objects = list(association.get("object_uids_before") or ())
        versions = list(association.get("candidate_object_version_uids") or ())
        version_uid = None
        if object_uid in objects:
            index = objects.index(object_uid)
            if index < len(versions):
                version_uid = str(versions[index])
        identity = self._identity_for_version(version_uid)
        if identity is None and object_uid in self.clean_membership:
            identity = object_uid
        return identity, version_uid

    def _target_options(
        self, association: Mapping[str, Any], *, exclude_identity: str
    ) -> list[dict[str, Any]]:
        result = []
        for rank, candidate in enumerate(association.get("top_candidates") or (), 1):
            object_uid = str(candidate.get("object_uid", ""))
            identity, version_uid = self._candidate_identity(association, object_uid)
            if not identity or identity == exclude_identity:
                continue
            version_members = (
                self.provenance.get_member_observations(version_uid) if version_uid else ()
            )
            result.append(
                {
                    "rank": rank,
                    "object_uid": object_uid,
                    "identity_uid": identity,
                    "version_uid": version_uid,
                    "origin_obs_uid": version_members[0] if version_members else None,
                    "score": float(candidate.get("aggregate_score") or float("-inf")),
                }
            )
        return result

    def _base_case(
        self,
        *,
        failure_type: str,
        association: Mapping[str, Any],
        source_identity: str,
        target: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        obs_uid = str(association["obs_uid"])
        suffix = canonical_obs_key(obs_uid)
        case_uid = f"{failure_type.lower()}_{suffix}"
        target_identity = target.get("identity_uid") if target else None
        clean_version_uid = association.get("target_object_version_before")
        clean_version_members = (
            self.provenance.get_member_observations(str(clean_version_uid))
            if clean_version_uid in self.provenance.object_versions
            else ()
        )
        case = {
            "schema_version": "0.1.0",
            "case_uid": case_uid,
            "failure_type": failure_type,
            "scene_id": str(obs_uid).split("_", 1)[0],
            "anchor_association_event_uid": str(association["event_uid"]),
            "anchor_mapping_event_uid": str(association["mapping_event_uid"]),
            "anchor_event_sequence": int(self.provenance.sequence(dict(association))),
            "frame_idx": frame_index(obs_uid),
            "obs_uid": obs_uid,
            "obs_key": suffix,
            "source_identity_uid": source_identity,
            "target_identity_uid": target_identity,
            "clean_target_object_uid": association.get("target_object_uid"),
            "clean_target_object_version_uid": clean_version_uid,
            "clean_target_origin_obs_uid": (
                clean_version_members[0] if clean_version_members else None
            ),
            "target_object_uid": target.get("object_uid") if target else None,
            "target_object_version_uid": target.get("version_uid") if target else None,
            "target_origin_obs_uid": target.get("origin_obs_uid") if target else None,
            "clean_top1_score": association.get("top1_score"),
            "clean_margin": association.get("margin"),
            "affected_clean_groups": {
                source_identity: list(self.clean_membership[source_identity]),
            },
        }
        if target_identity:
            case["affected_clean_groups"][target_identity] = list(
                self.clean_membership[target_identity]
            )
        return case

    def ranked_candidates(
        self, failure_type: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Return deterministic eligible cases in the same order used by ``select``."""
        failure_type = failure_type.upper()
        ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for association in self.provenance.association_rows:
            obs_uid = str(association["obs_uid"])
            source = self.obs_to_final.get(obs_uid)
            if not source:
                continue
            source_size = len(self.clean_membership[source])
            frame = frame_index(obs_uid)
            if source_size < 6 or not 3 <= frame <= 185:
                continue
            decision = str(association.get("decision", ""))
            top1 = float(association.get("top1_score") or 0.0)
            margin = float(association.get("margin") or 0.0)
            options = self._target_options(association, exclude_identity=source)

            if failure_type == "FALSE_SPLIT" and decision == "MERGE_TO_OBJECT":
                future = sum(
                    self.provenance.sequence(self.provenance.get_association_for_obs(item))
                    > self.provenance.sequence(association)
                    for item in self.clean_membership[source]
                )
                if future < 4:
                    continue
                case = self._base_case(
                    failure_type=failure_type,
                    association=association,
                    source_identity=source,
                    target=None,
                )
                case["corrupted_entity_uid"] = stable_entity_uid([obs_uid], prefix="split")
                case["corruption_plan"] = {
                    "case_uid": case["case_uid"],
                    "frame_idx": frame,
                    "obs_uid": obs_uid,
                    "corruption_type": "FORCE_CREATE",
                    "source_object_uid": association.get("target_object_uid"),
                    "source_origin_obs_uid": case["clean_target_origin_obs_uid"],
                    "seed": 20260822,
                }
                case["oracle_constraint"] = {
                    "type": "SAME_INSTANCE",
                    "entities": [source, case["corrupted_entity_uid"]],
                    "source": "oracle",
                    "evidence_refs": [str(association["event_uid"])],
                }
                ranked.append(((top1, margin, min(source_size, 60), -frame), case))

            elif failure_type == "WRONG_MEMBERSHIP" and decision == "MERGE_TO_OBJECT" and options:
                target = next(
                    (
                        item
                        for item in options
                        if len(self.clean_membership.get(str(item["identity_uid"]), ())) >= 6
                    ),
                    None,
                )
                if not target:
                    continue
                case = self._base_case(
                    failure_type=failure_type,
                    association=association,
                    source_identity=source,
                    target=target,
                )
                case["corruption_plan"] = {
                    "case_uid": case["case_uid"],
                    "frame_idx": frame,
                    "obs_uid": obs_uid,
                    "corruption_type": "FORCE_ASSOCIATE",
                    "source_object_uid": association.get("target_object_uid"),
                    "source_origin_obs_uid": case["clean_target_origin_obs_uid"],
                    "target_object_uid": target["object_uid"],
                    "target_origin_obs_uid": target["origin_obs_uid"],
                    "seed": 20260822,
                }
                case["oracle_constraint"] = {
                    "type": "MOVE_OBSERVATION",
                    "obs_uid": obs_uid,
                    "from": target["identity_uid"],
                    "to": source,
                    "source": "oracle",
                    "evidence_refs": [str(association["event_uid"])],
                }
                plausibility = float(target["score"])
                ranked.append(((plausibility, top1, margin, -frame), case))

            elif failure_type == "FALSE_MERGE" and decision == "CREATE_OBJECT" and options:
                target = next(
                    (
                        item
                        for item in options
                        if len(self.clean_membership.get(str(item["identity_uid"]), ())) >= 6
                    ),
                    None,
                )
                if not target:
                    continue
                case = self._base_case(
                    failure_type=failure_type,
                    association=association,
                    source_identity=source,
                    target=target,
                )
                case["corruption_plan"] = {
                    "case_uid": case["case_uid"],
                    "frame_idx": frame,
                    "obs_uid": obs_uid,
                    "corruption_type": "FORCE_ASSOCIATE",
                    "source_object_uid": None,
                    "target_object_uid": target["object_uid"],
                    "target_origin_obs_uid": target["origin_obs_uid"],
                    "seed": 20260822,
                }
                case["oracle_constraint"] = {
                    "type": "SEPARATE_MEMBER_GROUPS",
                    "groups": {
                        "A": list(self.clean_membership[source]),
                        "B": list(self.clean_membership[str(target["identity_uid"])]),
                    },
                    "source": "oracle",
                    "evidence_refs": [str(association["event_uid"])],
                }
                ranked.append(((float(target["score"]), -abs(20 - source_size), -frame), case))

        if not ranked:
            raise RuntimeError(f"no eligible {failure_type} case in evidence ledger")
        ranked.sort(key=lambda item: (item[0], item[1]["case_uid"]), reverse=True)
        cases = [case for _, case in ranked]
        if limit is not None:
            if limit < 1:
                raise ValueError("candidate limit must be at least one")
            cases = cases[:limit]
        return cases

    def select(self, failure_type: str) -> dict[str, Any]:
        return self.ranked_candidates(failure_type, limit=1)[0]

    def select_smoke_matrix(self) -> list[dict[str, Any]]:
        return [
            self.select("FALSE_SPLIT"),
            self.select("WRONG_MEMBERSHIP"),
            self.select("FALSE_MERGE"),
        ]
