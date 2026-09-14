#!/usr/bin/env python3
"""Freeze leave-one-view-out CMVIC evidence for identity separation shadows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from conceptgraph.revision.auto_constraints import (
    IncidentBinding,
    enumerate_identity_hypotheses,
    forbidden_inference_paths,
)
from conceptgraph.revision.autonomous_identity import (
    evenly_spaced,
    frame_index,
    partition_hash,
    relevant_future_observations,
)
from conceptgraph.revision.candidate_verifier import CandidateVerifier
from conceptgraph.revision.constraints import ReplayMode, SparseRepairConstraint
from conceptgraph.revision.counterfactual_projection import (
    CAUSAL_GEOMETRY_POLICY,
    CMVIC_STATISTIC_NAME,
    OBSERVED_MASK_POLICY,
    CounterfactualProjectionVerifier,
    ProjectionEvidenceLoader,
    extract_affected_instance_geometries,
    freeze_instance_geometries,
    render_projection_overlay,
)
from conceptgraph.revision.evidence_split import EvidenceSplitManifest, sha256_file
from conceptgraph.revision.snapshot import AnchorStateBuilder
from scripts.freeze_revision_identity_selective_v0 import (
    SceneContext,
    _candidate_seed_version,
    _compact_state,
    _freeze_crops,
    _read,
    _replay,
    _runtime_validity,
    _scoreable_in_every_state,
    _write,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _state_uid(prefix: str, partition_uid: str) -> str:
    digest = hashlib.sha256(str(partition_uid).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _freeze_projected_masks(
    *,
    path: Path,
    projected: Sequence[Any],
) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite projected masks: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(projected, key=lambda item: item.canonical_partition_uid)
    payload: dict[str, Any] = {}
    rows = []
    for index, item in enumerate(ordered):
        key = f"mask_{index:03d}"
        payload[key] = np.asarray(item.mask, dtype=np.uint8)
        rows.append({"array_key": key, **item.audit_dict()})
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "projected_instances": rows,
    }


def _raw_membership_audit(
    *, geometries: Sequence[Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    state_partitions = {
        frozenset(str(item) for item in members)
        for members in (state.get("membership") or {}).values()
        if members
    }
    geometry_partitions = {
        frozenset(item.member_obs_uids) for item in geometries if item.member_obs_uids
    }
    missing = sorted(
        sorted(partition)
        for partition in geometry_partitions
        if partition not in state_partitions
    )
    return {
        "pass": not missing,
        "raw_partition_count": len(geometry_partitions),
        "state_partition_count": len(state_partitions),
        "raw_partitions_missing_from_state": missing,
    }


def _proposal_observations(
    *,
    context: SceneContext,
    binding: IncidentBinding,
    seed_versions: Sequence[str],
    anchor_frame: int,
) -> tuple[str, ...]:
    provenance = context.provenance
    members = []
    for version_uid in seed_versions:
        members.extend(provenance.get_member_observations(version_uid))
    members = sorted(
        {
            obs_uid
            for obs_uid in members
            if obs_uid != binding.obs_uid
            and frame_index(provenance.get_observation(obs_uid)) <= anchor_frame
            and provenance.get_observation(obs_uid).get("crop_ref")
        },
        key=lambda obs_uid: (
            frame_index(provenance.get_observation(obs_uid)),
            obs_uid,
        ),
    )
    return tuple(
        dict.fromkeys((binding.obs_uid,) + evenly_spaced(tuple(members), limit=2))
    )


def _verification_observations(
    *,
    context: SceneContext,
    binding: IncidentBinding,
    aliases: Sequence[str],
    states: Sequence[Mapping[str, Any]],
    anchor_frame: int,
    minimum_frame_gap: int,
    maximum_frames: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    provenance = context.provenance
    roots = [binding.obs_uid]
    roots.extend(binding.aliases[alias].origin_obs_uid for alias in aliases)
    future_pool = relevant_future_observations(
        states=tuple(states),
        root_obs_uids=tuple(roots),
        observation_rows=provenance.observations,
        minimum_frame=anchor_frame + int(minimum_frame_gap),
    )
    scoreable = [
        obs_uid
        for obs_uid in future_pool
        if provenance.get_observation(obs_uid).get("crop_ref")
        and provenance.get_observation(obs_uid).get("processed_mask_ref")
        and _scoreable_in_every_state(obs_uid, states)
    ]
    representative_by_frame: dict[str, str] = {}
    for obs_uid in scoreable:
        frame_uid = str(provenance.get_observation(obs_uid)["frame_uid"])
        representative_by_frame.setdefault(frame_uid, obs_uid)
    ordered_frames = tuple(
        sorted(
            representative_by_frame,
            key=lambda uid: (
                frame_index({"frame_uid": uid}),
                uid,
            ),
        )
    )
    selected_frames = evenly_spaced(ordered_frames, limit=int(maximum_frames))
    selected_obs = tuple(representative_by_frame[uid] for uid in selected_frames)
    return (
        selected_obs,
        tuple(selected_frames),
        len(future_pool),
        len(ordered_frames),
    )


def _projection_prompt(
    *, case_uid: str, image_rows: Sequence[Mapping[str, Any]]
) -> str:
    payload = [
        {
            "evidence_id": str(row["evidence_id"]),
            "frame_index": int(row["frame_index"]),
            "content": str(row["class_name"]),
        }
        for row in image_rows
    ]
    return (
        "Compare two anonymous counterfactual 3D instance partitions using only "
        "the supplied held-out future views. For each frame, the original RGB is "
        "followed by STATE_A and STATE_B overlays. White contours are the frozen "
        "observed 2D instance masks; colored GROUP contours are projected 3D "
        "instance boundaries. Decide which state more consistently explains the "
        "observed masks across views. Actively seek counterevidence: one projected "
        "3D group repeatedly crossing independent observed masks, or multiple "
        "projected groups repeatedly explaining one physical instance. If the "
        "projection is not visible or observed masks conflict, choose DEFER. "
        "Return exactly one JSON object with keys preferred_state, confidence, "
        "reason, counterevidence, needed_evidence, cited_evidence_ids. "
        "preferred_state must be STATE_A, STATE_B, or DEFER. Confidence is "
        "diagnostic only. Cite at least one supplied evidence_id.\n\n"
        + json.dumps(
            {"case_uid": case_uid, "images_in_presented_order": payload},
            indent=2,
            sort_keys=True,
        )
    )


def _request_images(
    *,
    frame_rows: Sequence[Mapping[str, Any]],
    overlay_by_frame_state: Mapping[str, Mapping[str, Mapping[str, Any]]],
    label_to_state: Mapping[str, str],
) -> list[dict[str, Any]]:
    images = []
    for index, frame_row in enumerate(frame_rows, 1):
        frame_uid = str(frame_row["frame_uid"])
        images.append(
            {
                "evidence_id": f"F{index:02d}_RGB",
                "frame_index": int(frame_row["frame_index"]),
                "class_name": "HELDOUT_RGB",
                "sha256": str(frame_row["rgb_sha256"]),
                "path": str(frame_row["rgb_path"]),
            }
        )
        for label in ("STATE_A", "STATE_B"):
            overlay = overlay_by_frame_state[frame_uid][label_to_state[label]]
            images.append(
                {
                    "evidence_id": f"F{index:02d}_{label}",
                    "frame_index": int(frame_row["frame_index"]),
                    "class_name": f"{label}_PROJECTED_BOUNDARIES",
                    "sha256": str(overlay["sha256"]),
                    "path": str(overlay["path"]),
                }
            )
    return images


def _freeze_case(
    *,
    row: Mapping[str, Any],
    context: SceneContext,
    loader: ProjectionEvidenceLoader,
    verifier: CounterfactualProjectionVerifier,
    output_root: Path,
    minimum_frame_gap: int,
    maximum_verification_frames: int,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    case_uid = str(row["case_uid"])
    case_dir = output_root / case_uid
    binding_path = Path(str(row["binding_path"])).resolve()
    if sha256_file(binding_path) != str(row["binding_sha256"]):
        raise ValueError(f"{case_uid}: binding hash drift")
    binding_payload = _read(binding_path)
    forbidden = forbidden_inference_paths(binding_payload)
    if forbidden:
        raise ValueError(f"{case_uid}: oracle-like binding fields: {forbidden}")
    binding = IncidentBinding.from_mapping(binding_payload)
    if binding.case_uid != case_uid:
        raise ValueError(f"{case_uid}: binding case mismatch")

    aliases = tuple(
        sorted(
            alias
            for alias, value in binding.aliases.items()
            if alias.startswith("CANDIDATE_") and value.complete
        )
    )
    if not aliases:
        raise ValueError(f"{case_uid}: no complete identity candidate aliases")
    provenance = context.provenance
    association = provenance.get_event(binding.event_uid)
    seed_versions = tuple(
        _candidate_seed_version(binding=binding, association=association, alias=alias)
        for alias in aliases
    )
    anchor_frame = frame_index(provenance.get_observation(binding.obs_uid))
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=binding.event_uid,
        seed_version_uids=seed_versions,
    )
    snapshot_started = time.perf_counter()
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(anchor_frame)
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        binding.event_uid,
        seed_versions,
        strict=True,
        prefix_state=prefix_state,
        prefix_objects=prefix_objects,
    )
    snapshot_wall_ms = (time.perf_counter() - snapshot_started) * 1000.0

    noop_state, noop_wall_ms = _replay(
        context=context, snapshot=snapshot, closure=closure, constraints=()
    )
    noop_validity = _runtime_validity(
        context=context,
        state=noop_state,
        constraints=(),
        snapshot_valid=bool(snapshot.validation["pass"]),
    )
    noop_partition = partition_hash(noop_state.get("membership") or {})
    replay_rows = []
    for compiled in enumerate_identity_hypotheses(binding, candidate_aliases=aliases):
        if compiled.get("hypothesis_action") != "SEPARATE_MEMBER_GROUPS":
            continue
        primitive = SparseRepairConstraint.from_mapping(
            compiled["candidate_constraint"]
        )
        state, replay_wall_ms = _replay(
            context=context,
            snapshot=snapshot,
            closure=closure,
            constraints=(primitive,),
        )
        validity = _runtime_validity(
            context=context,
            state=state,
            constraints=(primitive,),
            snapshot_valid=bool(snapshot.validation["pass"]),
        )
        replay_rows.append(
            {
                "candidate_uid": str(compiled["constraint_fingerprint"]),
                "target_alias": str(compiled["hypothesis_target_alias"]),
                "constraint": primitive,
                "partition_hash": partition_hash(state.get("membership") or {}),
                "runtime_validity": validity,
                "replay_wall_ms": replay_wall_ms,
                "state": state,
            }
        )

    distinct_by_partition: dict[str, dict[str, Any]] = {}
    for replay_row in replay_rows:
        if replay_row["partition_hash"] == noop_partition:
            continue
        current = distinct_by_partition.get(replay_row["partition_hash"])
        if current is None or (
            not current["runtime_validity"]["valid"]
            and replay_row["runtime_validity"]["valid"]
        ):
            distinct_by_partition[replay_row["partition_hash"]] = replay_row
    if not distinct_by_partition:
        result = {
            "schema_version": "1.0.0",
            "case_uid": case_uid,
            "scene_id": str(row["scene_id"]),
            "anchor_frame": anchor_frame,
            "status": "DEFER_NO_DISTINCT_EXECUTABLE_SEPARATION",
            "observable_candidate_count": 0,
            "critic_requests": [],
            "primary_candidate_scores": [],
            "gold_loaded": False,
            "human_verdict_loaded": False,
            "production_commit_permitted": False,
            "timing": {
                "snapshot_wall_ms": snapshot_wall_ms,
                "noop_replay_wall_ms": noop_wall_ms,
                "candidate_replay_wall_ms": [
                    item["replay_wall_ms"] for item in replay_rows
                ],
                "case_total_wall_ms": (time.perf_counter() - case_started) * 1000.0,
            },
        }
        _write(case_dir / "case_result.frozen.json", result)
        return result

    distinct_rows = [
        distinct_by_partition[key] for key in sorted(distinct_by_partition)
    ]
    endpoint_states = [noop_state] + [item["state"] for item in distinct_rows]
    proposal_obs = _proposal_observations(
        context=context,
        binding=binding,
        seed_versions=seed_versions,
        anchor_frame=anchor_frame,
    )
    proposal_refs, proposal_rows = _freeze_crops(
        provenance=provenance,
        obs_uids=proposal_obs,
        output_dir=case_dir / "proposal_evidence",
        id_prefix="P",
        source_role="PROPOSAL_ONLY_PREANCHOR_CONTEXT",
    )
    (
        verification_obs,
        frame_uids,
        future_count,
        common_frame_count,
    ) = _verification_observations(
        context=context,
        binding=binding,
        aliases=aliases,
        states=endpoint_states,
        anchor_frame=anchor_frame,
        minimum_frame_gap=minimum_frame_gap,
        maximum_frames=maximum_verification_frames,
    )
    if len(frame_uids) < 2:
        raise ValueError(f"{case_uid}: fewer than two common future RGB-D frames")
    verification_refs, verification_rows = _freeze_crops(
        provenance=provenance,
        obs_uids=verification_obs,
        output_dir=case_dir / "verification_evidence",
        id_prefix="V",
        source_role="HELDOUT_FUTURE_OUTCOME_EVIDENCE",
    )
    split = EvidenceSplitManifest.build(
        incident_uid=case_uid,
        anchor_obs_uid=binding.obs_uid,
        anchor_frame=anchor_frame,
        proposal=proposal_refs,
        verification=verification_refs,
        minimum_frame_gap=minimum_frame_gap,
    )
    _write(case_dir / "evidence_split.frozen.json", split.as_dict())

    state_specs = {
        "NOOP": {
            "constraints": (),
            "endpoint_state": noop_state,
            "partition_hash": noop_partition,
            "runtime_validity": noop_validity,
            "candidate_uid": None,
        }
    }
    for index, replay_row in enumerate(distinct_rows, 1):
        uid = _state_uid(f"CANDIDATE_{index:02d}", replay_row["partition_hash"])
        state_specs[uid] = {
            "constraints": (replay_row["constraint"],),
            "endpoint_state": replay_row["state"],
            "partition_hash": replay_row["partition_hash"],
            "runtime_validity": replay_row["runtime_validity"],
            "candidate_uid": replay_row["candidate_uid"],
            "target_alias": replay_row["target_alias"],
        }

    frames = [loader.load_frame(frame_uid) for frame_uid in frame_uids]
    projected_by_frame: dict[str, dict[str, Any]] = {}
    frame_rows = []
    projection_wall_ms = 0.0
    causal_replay_rows = []
    for frame in frames:
        frame_state_projected = {}
        frame_row = frame.audit_dict()
        frame_row.update(
            {
                "rgb_path": str(frame.rgb_path),
                "rgb_sha256": sha256_file(frame.rgb_path),
                "depth_path": str(frame.depth_path),
                "depth_sha256": sha256_file(frame.depth_path),
            }
        )
        frame_rows.append(frame_row)
        for state_uid, spec in state_specs.items():
            causal_started = time.perf_counter()
            (
                causal_state,
                raw_objects,
            ) = context.engine.replay_causal_prefix_from_snapshot_with_objects(
                mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
                snapshot_objects=snapshot.objects,
                snapshot_runtime_ms=snapshot.state["runtime_ms"],
                snapshot_timing=snapshot.state.get("timing"),
                anchor_frame=snapshot.anchor_frame,
                cutoff_frame=frame.frame_index - 1,
                snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
                closure=closure,
                constraints=spec["constraints"],
                current_state=context.native_state,
            )
            causal_wall_ms = (time.perf_counter() - causal_started) * 1000.0
            source_state_hash = str(
                causal_state.get("state_hash")
                or hashlib.sha256(
                    json.dumps(
                        causal_state.get("membership") or {},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            geometries = extract_affected_instance_geometries(
                raw_objects=raw_objects,
                affected_obs_uids=closure.obs_uids,
                source_state_hash=source_state_hash,
            )
            membership_audit = _raw_membership_audit(
                geometries=geometries, state=causal_state
            )
            if not membership_audit["pass"]:
                raise ValueError(
                    f"{case_uid}:{frame.frame_uid}:{state_uid}: "
                    "raw object partitions disagree with causal state"
                )
            geometry_rows = freeze_instance_geometries(
                output_dir=case_dir
                / "projection_private"
                / frame.frame_uid
                / "geometry",
                state_uid=state_uid,
                geometries=geometries,
            )
            projection_started = time.perf_counter()
            projected = verifier.project_state(
                state_uid=state_uid,
                instances=geometries,
                frame=frame,
            )
            projection_ms = (time.perf_counter() - projection_started) * 1000.0
            projection_wall_ms += projection_ms
            projected_artifact = _freeze_projected_masks(
                path=case_dir
                / "projection_private"
                / frame.frame_uid
                / "projected_masks"
                / f"{state_uid}.npz",
                projected=projected,
            )
            frame_state_projected[state_uid] = projected
            causal_replay_rows.append(
                {
                    "frame_uid": frame.frame_uid,
                    "state_uid": state_uid,
                    "causal_cutoff_frame": frame.frame_index - 1,
                    "causal_geometry_policy": CAUSAL_GEOMETRY_POLICY,
                    "state_hash": source_state_hash,
                    "membership_audit": membership_audit,
                    "geometry_artifacts": list(geometry_rows),
                    "projected_mask_artifact": projected_artifact,
                    "causal_replay_wall_ms": causal_wall_ms,
                    "projection_wall_ms": projection_ms,
                }
            )
        projected_by_frame[frame.frame_uid] = frame_state_projected
    _write(case_dir / "frame_evidence.frozen.json", {"frames": frame_rows})

    candidate_scores = []
    comparisons = []
    request_rows = []
    private_mappings = {}
    overlay_by_pair: dict[str, Any] = {}
    candidate_verifier = CandidateVerifier()
    for pair_index, (state_uid, spec) in enumerate(
        ((uid, value) for uid, value in state_specs.items() if uid != "NOOP"), 1
    ):
        comparison = verifier.compare(
            noop_state_uid="NOOP",
            candidate_state_uid=state_uid,
            frames=frames,
            projected_by_frame=projected_by_frame,
        )
        comparisons.append(
            {
                "pair_uid": f"PAIR_{pair_index:02d}",
                "candidate_state_uid": state_uid,
                "candidate_uid": spec["candidate_uid"],
                **comparison.as_dict(),
            }
        )
        score = candidate_verifier.score_identity(
            incident_uid=case_uid,
            candidate_uid=str(spec["candidate_uid"]),
            candidate_state=spec["endpoint_state"],
            noop_state=noop_state,
            split=split,
            runtime_valid=bool(spec["runtime_validity"]["valid"]),
            primary_scorer="CMVIC",
            candidate_cmvic=comparison.candidate,
            noop_cmvic=comparison.noop,
        )
        candidate_scores.append(
            {
                **score.as_dict(),
                "state_uid": state_uid,
                "partition_hash": spec["partition_hash"],
                "target_alias": spec.get("target_alias"),
                "observability_disposition": comparison.disposition,
            }
        )
        if not comparison.observable:
            continue

        pair_uid = f"PAIR_{pair_index:02d}"
        color_seed = f"{case_uid}:{pair_uid}:PHYSICAL_GROUP_COLOR_V0"
        overlay_by_frame_state: dict[str, dict[str, Any]] = {}
        for frame in frames:
            projected_states = projected_by_frame[frame.frame_uid]
            observed_masks, _, _ = verifier.select_common_observed_masks(
                frame=frame,
                projected_by_state={
                    "NOOP": projected_states["NOOP"],
                    state_uid: projected_states[state_uid],
                },
            )
            overlay_by_frame_state[frame.frame_uid] = {}
            for rendered_state_uid in ("NOOP", state_uid):
                overlay_by_frame_state[frame.frame_uid][
                    rendered_state_uid
                ] = render_projection_overlay(
                    frame=frame,
                    projected=projected_states[rendered_state_uid],
                    observed_masks=observed_masks,
                    output_path=case_dir
                    / "critic_overlays"
                    / pair_uid
                    / frame.frame_uid
                    / f"{rendered_state_uid}.png",
                    color_seed=color_seed,
                )
        overlay_by_pair[pair_uid] = overlay_by_frame_state
        for order_index, order in enumerate((("NOOP", state_uid), (state_uid, "NOOP"))):
            label_to_state = {"STATE_A": order[0], "STATE_B": order[1]}
            images = _request_images(
                frame_rows=frame_rows,
                overlay_by_frame_state=overlay_by_frame_state,
                label_to_state=label_to_state,
            )
            prompt = _projection_prompt(case_uid=case_uid, image_rows=images)
            request_uid = f"{case_uid}_{pair_uid}_ORDER_{order_index}"
            request = {
                "schema_version": "2.0.0",
                "request_uid": request_uid,
                "case_uid": case_uid,
                "pair_uid": pair_uid,
                "order_index": order_index,
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "images": images,
                "allowed_state_ids": ["STATE_A", "STATE_B"],
                "allowed_evidence_ids": [str(image["evidence_id"]) for image in images],
                "evidence_split_uid": split.manifest_uid,
                "action_names_hidden_from_critic": True,
                "proposal_evidence_hidden_from_critic": True,
                "state_order_swapped": bool(order_index),
                "projection_evidence_only": True,
            }
            forbidden_request = forbidden_inference_paths(request)
            if forbidden_request:
                raise ValueError(
                    "oracle-like frozen request fields: " + ", ".join(forbidden_request)
                )
            request_path = (
                case_dir / "critic_requests" / f"{request_uid}.json"
            ).resolve()
            _write(request_path, request)
            request_rows.append(
                {
                    "request_uid": request_uid,
                    "case_uid": case_uid,
                    "pair_uid": pair_uid,
                    "order_index": order_index,
                    "path": str(request_path),
                    "sha256": sha256_file(request_path),
                }
            )
            private_mappings[request_uid] = {
                "label_to_state_uid": label_to_state,
                "candidate_state_uid": state_uid,
                "candidate_uid": spec["candidate_uid"],
                "noop_state_uid": "NOOP",
            }

    source_hashes_after = provenance.source_hashes()
    if source_hashes_after != context.source_hashes_before:
        raise ValueError(f"{case_uid}: source evidence mutated during freeze")
    private = {
        "schema_version": "2.0.0",
        "case_uid": case_uid,
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "candidate_aliases": list(aliases),
        "seed_version_uids": list(seed_versions),
        "closure": closure.as_dict(),
        "snapshot_validation": snapshot.validation,
        "noop_partition_hash": noop_partition,
        "noop_runtime_validity": noop_validity,
        "noop_state_audit": _compact_state(noop_state),
        "candidate_replays": [
            {
                "candidate_uid": item["candidate_uid"],
                "target_alias": item["target_alias"],
                "constraint": item["constraint"].as_dict(),
                "partition_hash": item["partition_hash"],
                "runtime_validity": item["runtime_validity"],
                "replay_wall_ms": item["replay_wall_ms"],
                "state_audit": _compact_state(item["state"]),
            }
            for item in replay_rows
        ],
        "causal_replays": causal_replay_rows,
        "critic_state_mappings": private_mappings,
    }
    _write(case_dir / "execution.private.json", private)
    _write(case_dir / "cmvic_scores.frozen.json", {"comparisons": comparisons})
    observable_count = sum(item["observable"] for item in comparisons)
    result = {
        "schema_version": "2.0.0",
        "case_uid": case_uid,
        "scene_id": str(row["scene_id"]),
        "anchor_frame": anchor_frame,
        "observed_current_decision": binding.observed_current_decision,
        "finite_separation_constraint_count": len(replay_rows),
        "distinct_repair_partition_count": len(distinct_rows),
        "observable_candidate_count": observable_count,
        "unobservable_candidate_count": len(comparisons) - observable_count,
        "evidence_split": split.as_dict(),
        "proposal_evidence": proposal_rows,
        "verification_evidence": verification_rows,
        "verification_frame_uids": list(frame_uids),
        "future_pool_count": future_count,
        "common_scoreable_future_frame_count": common_frame_count,
        "primary_candidate_scores": candidate_scores,
        "cmvic_comparisons": comparisons,
        "critic_requests": request_rows,
        "counterfactual_observability_contract": (
            "EXACT_ZERO_VISIBLE_PROJECTED_PARTITION_DIFFERENCE_DEFER_WITHOUT_VLM"
        ),
        "causal_geometry_policy": CAUSAL_GEOMETRY_POLICY,
        "observed_mask_policy": OBSERVED_MASK_POLICY,
        "primary_statistic": CMVIC_STATISTIC_NAME,
        "gold_loaded": False,
        "human_verdict_loaded": False,
        "production_commit_permitted": False,
        "calibration_status": "UNREADY_NEW_EVIDENCE_SCHEMA",
        "semantic_threshold_count": 0,
        "status": (
            "FROZEN_PENDING_OBSERVABLE_CRITIC"
            if request_rows
            else "DEFER_COUNTERFACTUAL_UNOBSERVABLE_NO_VLM"
        ),
        "timing": {
            "context_build_wall_ms_shared": context.build_wall_ms,
            "snapshot_wall_ms": snapshot_wall_ms,
            "noop_replay_wall_ms": noop_wall_ms,
            "candidate_replay_wall_ms": [
                item["replay_wall_ms"] for item in replay_rows
            ],
            "causal_replay_wall_ms": [
                item["causal_replay_wall_ms"] for item in causal_replay_rows
            ],
            "projection_verifier_wall_ms": projection_wall_ms,
            "case_total_wall_ms": (time.perf_counter() - case_started) * 1000.0,
        },
    }
    _write(case_dir / "case_result.frozen.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", action="append", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--minimum-frame-gap", type=int, default=3)
    parser.add_argument("--maximum-verification-frames", type=int, default=4)
    args = parser.parse_args()

    manifests = []
    cases = []
    for manifest_path_value in args.case_manifest:
        manifest_path = manifest_path_value.resolve()
        manifest = _read(manifest_path)
        forbidden = forbidden_inference_paths(manifest)
        if forbidden:
            raise ValueError(
                f"oracle-like case manifest fields in {manifest_path}: {forbidden}"
            )
        manifests.append(
            {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
        )
        cases.extend(dict(row) for row in manifest.get("cases") or ())
    selected = set(str(item) for item in args.only_case)
    if selected:
        cases = [row for row in cases if str(row.get("case_uid")) in selected]
        if {str(row["case_uid"]) for row in cases} != selected:
            raise ValueError("one or more --only-case IDs were not found")
    cases = [row for row in cases if str(row.get("scene_id")) == "room0"]
    if not cases:
        raise ValueError("no room0 cases selected")
    if len({str(row["case_uid"]) for row in cases}) != len(cases):
        raise ValueError("duplicate case UID across manifests")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    room_run = args.room_run.resolve()
    context = SceneContext.build(room_run)
    loader = ProjectionEvidenceLoader(room_run)
    voxel_size = float(loader.config["downsample_voxel_size"])
    verifier = CounterfactualProjectionVerifier(voxel_size=voxel_size)
    run_started = time.perf_counter()
    results = []
    for row in cases:
        results.append(
            _freeze_case(
                row=row,
                context=context,
                loader=loader,
                verifier=verifier,
                output_root=output_root,
                minimum_frame_gap=args.minimum_frame_gap,
                maximum_verification_frames=args.maximum_verification_frames,
            )
        )
        print(
            json.dumps(
                {
                    "case_uid": results[-1]["case_uid"],
                    "status": results[-1]["status"],
                    "observable_candidate_count": results[-1][
                        "observable_candidate_count"
                    ],
                    "request_count": len(results[-1]["critic_requests"]),
                }
            ),
            flush=True,
        )

    request_rows = [
        request for result in results for request in result["critic_requests"]
    ]
    policy_uids = sorted(
        {
            score["evidence_policy_uid"]
            for result in results
            for score in result["primary_candidate_scores"]
        }
    )
    protocol = {
        "schema_version": "2.0.0",
        "role": "CMVIC_DEVELOPMENT_SHADOW_NOT_PRODUCTION_COMMIT",
        "case_manifests": manifests,
        "case_count": len(results),
        "cases": [
            {
                "case_uid": result["case_uid"],
                "scene_id": result["scene_id"],
                "result_path": str(
                    (
                        output_root / result["case_uid"] / "case_result.frozen.json"
                    ).resolve()
                ),
                "status": result["status"],
            }
            for result in results
        ],
        "critic_requests": request_rows,
        "request_count": len(request_rows),
        "runtime_human_or_gold_loaded": False,
        "candidate_source": "FINITE_EXECUTOR_IDENTITY_SEPARATION_CAPABILITY",
        "verification_evidence_policy": (
            "ALL_STATE_UNION_COMMON_SCOREABLE_TEMPORAL_EVEN_SAMPLE"
        ),
        "causal_geometry_policy": CAUSAL_GEOMETRY_POLICY,
        "observed_mask_policy": OBSERVED_MASK_POLICY,
        "primary_statistic": CMVIC_STATISTIC_NAME,
        "evidence_policy_uids": policy_uids,
        "minimum_frame_gap": args.minimum_frame_gap,
        "maximum_verification_frames": args.maximum_verification_frames,
        "critic_position_audit": "TWO_ORDER_SWAPPED_REQUESTS_PER_OBSERVABLE_PAIR",
        "unobservable_vlm_call_count": 0,
        "production_commit_permitted": False,
        "calibration_status": "UNREADY_NEW_EVIDENCE_SCHEMA",
        "semantic_threshold_count": 0,
        "voxel_size": voxel_size,
        "depth_tolerance": verifier.depth_tolerance,
        "total_wall_ms": (time.perf_counter() - run_started) * 1000.0,
    }
    protocol["protocol_uid"] = (
        "cmvic_freeze_protocol_"
        + hashlib.sha256(
            json.dumps(
                protocol,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    _write(output_root / "freeze_protocol.json", protocol)
    print(
        json.dumps(
            {
                "status": "PASS",
                "case_count": len(results),
                "request_count": len(request_rows),
                "evidence_policy_uid_count": len(policy_uids),
                "output_root": str(output_root),
                "total_wall_ms": protocol["total_wall_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
