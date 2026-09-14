"""Oracle-free helpers for comparing executed identity-repair outcomes.

The helpers in this module operate on machine provenance and replay states only.
They intentionally canonicalize partitions independently of entity UUIDs so that
two constraints producing the same ownership state are not counted twice.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .auto_constraints import forbidden_inference_paths


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def frame_index(observation: Mapping[str, Any]) -> int:
    """Read a frame index from frozen observation provenance."""

    explicit = observation.get("frame_index")
    if explicit is not None:
        return int(explicit)
    frame_uid = str(observation.get("frame_uid") or "")
    match = re.search(r"_f(\d+)$", frame_uid)
    if match is None:
        raise ValueError(f"cannot derive frame index from {frame_uid!r}")
    return int(match.group(1))


def partition_signature(
    membership: Mapping[str, Iterable[str]],
) -> tuple[tuple[str, ...], ...]:
    """Canonical ownership partition, independent of arbitrary entity IDs."""

    groups = {
        tuple(sorted(set(str(obs_uid) for obs_uid in members or ())))
        for members in membership.values()
    }
    return tuple(sorted(group for group in groups if group))


def partition_hash(membership: Mapping[str, Iterable[str]]) -> str:
    encoded = _canonical_json(partition_signature(membership)).encode("utf-8")
    return "partition_" + hashlib.sha256(encoded).hexdigest()[:20]


def distinct_candidate_partitions(
    noop_partition_hash: str, candidate_partition_hashes: Iterable[str]
) -> tuple[str, ...]:
    """Deduplicate executed candidates and exclude behaviorally identical NO-OP."""

    return tuple(
        sorted(
            set(str(item) for item in candidate_partition_hashes)
            - {str(noop_partition_hash)}
        )
    )


def heldout_assignment_signature(
    state_summary: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Canonical partition of evidence IDs exposed to the outcome critic.

    Geometry and total member counts may differ while every held-out observation
    still has exactly the same owner. In that situation, adding wider versions
    of the same frames cannot reveal which executed partition is preferable.
    """

    groups = {
        tuple(sorted(set(str(item) for item in group.get("evidence_ids") or ())))
        for group in state_summary.get("groups") or ()
    }
    return tuple(sorted(group for group in groups if group))


def heldout_assignments_distinguishable(
    state_summaries: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether anonymous states partition held-out evidence differently."""

    signatures = {
        heldout_assignment_signature(summary) for summary in state_summaries.values()
    }
    return len(signatures) > 1


def evenly_spaced(values: Sequence[str], limit: int) -> tuple[str, ...]:
    """Deterministically retain temporal coverage without score-based picking."""

    unique = tuple(dict.fromkeys(str(value) for value in values))
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(unique) <= limit:
        return unique
    if limit == 1:
        return (unique[len(unique) // 2],)
    indices = [round(index * (len(unique) - 1) / (limit - 1)) for index in range(limit)]
    return tuple(unique[index] for index in indices)


def balanced_partition_sample(
    *,
    values: Sequence[str],
    states: Sequence[Mapping[str, Any]],
    observation_rows: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> tuple[str, ...]:
    """Balance evidence across the common refinement of all replay partitions.

    Each atom contains observations that every candidate state groups in the same
    respective way. Round-robin allocation prevents a long trajectory from
    dominating a shorter competing trajectory, while evenly spaced sampling
    preserves temporal coverage inside every atom.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    unique = tuple(dict.fromkeys(str(value) for value in values))
    owner_signatures = []
    for state in states:
        owner = {}
        for members in (state.get("membership") or {}).values():
            group = tuple(sorted(set(str(item) for item in members or ())))
            group_hash = hashlib.sha256(
                _canonical_json(group).encode("utf-8")
            ).hexdigest()
            for obs_uid in group:
                owner[obs_uid] = group_hash
        owner_signatures.append(owner)

    atoms: dict[tuple[str | None, ...], list[str]] = {}
    for obs_uid in unique:
        signature = tuple(owner.get(obs_uid) for owner in owner_signatures)
        atoms.setdefault(signature, []).append(obs_uid)
    if not atoms:
        return ()

    ordered_atoms = sorted(
        atoms,
        key=lambda signature: (
            frame_index(observation_rows[atoms[signature][0]]),
            signature,
        ),
    )
    allocation = {signature: 0 for signature in ordered_atoms}
    remaining = min(limit, len(unique))
    while remaining:
        progressed = False
        for signature in ordered_atoms:
            if allocation[signature] >= len(atoms[signature]):
                continue
            allocation[signature] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected = []
    for signature in ordered_atoms:
        count = allocation[signature]
        if not count:
            continue
        selected.extend(evenly_spaced(tuple(atoms[signature]), limit=count))
    return tuple(
        sorted(
            selected,
            key=lambda obs_uid: (frame_index(observation_rows[obs_uid]), obs_uid),
        )
    )


def relevant_future_observations(
    *,
    states: Sequence[Mapping[str, Any]],
    root_obs_uids: Iterable[str],
    observation_rows: Mapping[str, Mapping[str, Any]],
    minimum_frame: int,
) -> tuple[str, ...]:
    """Find future observations in any owner touched by an incident root.

    Selection uses the union across all feasible replay outcomes. It therefore
    cannot privilege one candidate merely because that candidate owns more rows.
    """

    roots = {str(item) for item in root_obs_uids}
    relevant: set[str] = set()
    for state in states:
        membership = state.get("membership") or {}
        for members in membership.values():
            group = {str(item) for item in members or ()}
            if group & roots:
                relevant.update(group)
    future = [
        obs_uid
        for obs_uid in relevant
        if obs_uid in observation_rows
        and frame_index(observation_rows[obs_uid]) >= int(minimum_frame)
    ]
    return tuple(
        sorted(
            future,
            key=lambda obs_uid: (frame_index(observation_rows[obs_uid]), obs_uid),
        )
    )


def anonymous_state_summary(
    *,
    state: Mapping[str, Any],
    evidence_id_by_obs: Mapping[str, str],
    observation_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe only the held-out grouping induced by one executed state."""

    object_by_uid = {
        str(row.get("entity_uid")): row for row in state.get("objects") or ()
    }
    groups = []
    for entity_uid, members in (state.get("membership") or {}).items():
        selected = sorted(
            (
                str(obs_uid)
                for obs_uid in members or ()
                if str(obs_uid) in evidence_id_by_obs
            ),
            key=lambda obs_uid: evidence_id_by_obs[obs_uid],
        )
        if not selected:
            continue
        classes = Counter(
            str(observation_rows[obs_uid].get("class_name") or "unknown")
            for obs_uid in selected
        )
        row = object_by_uid.get(str(entity_uid), {})
        groups.append(
            {
                "evidence_ids": [evidence_id_by_obs[obs_uid] for obs_uid in selected],
                "heldout_class_counts": dict(sorted(classes.items())),
                "total_member_observation_count": len(tuple(members or ())),
                "n_points": int(row.get("n_points", 0)),
                "bbox_center": [
                    round(float(value), 4) for value in row.get("bbox_center") or ()
                ],
                "bbox_extent": [
                    round(float(value), 4) for value in row.get("bbox_extent") or ()
                ],
            }
        )
    groups.sort(key=lambda row: tuple(row["evidence_ids"]))
    return {
        "group_count_on_heldout_evidence": len(groups),
        "groups": [
            dict(group_id=f"GROUP_{index:02d}", **row)
            for index, row in enumerate(groups, 1)
        ],
    }


def build_pairwise_critic_prompt(
    *,
    incident_uid: str,
    evidence_rows: Sequence[Mapping[str, Any]],
    state_summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    """Build an action-blind prompt over identical held-out views."""

    payload = {
        "incident_uid": str(incident_uid),
        "heldout_evidence": [
            {
                "evidence_id": str(row["evidence_id"]),
                "frame_index": int(row["frame_index"]),
                "class_name": str(row.get("class_name") or "unknown"),
                "view_type": str(row.get("view_type") or "DETECTION_CROP"),
                **(
                    {
                        "linked_crop_evidence_ids": [
                            str(item)
                            for item in row.get("linked_crop_evidence_ids") or ()
                        ]
                    }
                    if row.get("linked_crop_evidence_ids")
                    else {}
                ),
            }
            for row in evidence_rows
        ],
        "executed_states": {
            str(state_id): dict(summary)
            for state_id, summary in sorted(state_summaries.items())
        },
    }
    forbidden = forbidden_inference_paths(payload)
    if forbidden:
        raise ValueError("oracle-like critic prompt fields: " + ", ".join(forbidden))
    return (
        "Compare the anonymous executed scene-graph states below using only the "
        "attached held-out future views. DETECTION_CROP evidence is assigned to "
        "state groups; WIDE_FRAME_CONTEXT evidence shows the surrounding frame for "
        "its linked crops, while PADDED_LOCAL_CONTEXT enlarges the nearby boundary "
        "around a linked crop. Both context views are supplied only to resolve "
        "physical boundaries or adjacency. "
        "The state labels are randomly assigned and reveal no repair action.\n\n"
        "Use scene-graph object-instance identity: physically connected or assembled "
        "modules of one functional object may remain one instance despite visible "
        "seams; loose accessories and separate freestanding units remain separate. "
        "Repeated installed fixtures at different stable locations are distinct even "
        "when appearance and class match. Never assume connection or separation from "
        "this ontology alone; require the held-out evidence to support it.\n\n"
        "Prefer a state only when repeated appearance, geometry, viewpoint change, and "
        "temporal continuity support its grouping. Same semantic class alone is not "
        "identity evidence. Actively search for counterevidence: one object split across "
        "groups, or distinct objects collapsed into one group. If the held-out views do "
        "not distinguish the states, choose DEFER.\n\n"
        "Return exactly one JSON object with keys preferred_state, confidence, reason, "
        "counterevidence, needed_evidence, and cited_evidence_ids. preferred_state must "
        "be one listed STATE_* value or DEFER. confidence is descriptive only and is not "
        "a commit threshold. Cite at least one listed evidence ID.\n\n"
        "FROZEN HELD-OUT OUTCOMES:\n"
        + json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    )


def signed_pairwise_preference(
    *,
    preferred_partition_hashes: Iterable[str | None],
    candidate_partition_hash: str,
    noop_partition_hash: str,
) -> float:
    """Mean order-swapped critic vote in [-1, 1]; DEFER contributes zero."""

    votes = []
    for preferred in preferred_partition_hashes:
        if preferred is None:
            votes.append(0.0)
        elif str(preferred) == str(candidate_partition_hash):
            votes.append(1.0)
        elif str(preferred) == str(noop_partition_hash):
            votes.append(-1.0)
        else:
            raise ValueError(f"critic selected unknown partition: {preferred}")
    if not votes:
        raise ValueError("at least one critic result is required")
    return sum(votes) / len(votes)
