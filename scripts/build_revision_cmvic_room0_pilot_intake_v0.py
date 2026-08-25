#!/usr/bin/env python3
"""Build an outcome-blind, frame-stratified room0 CMVIC pilot intake."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from conceptgraph.revision.auto_constraints import (
    IncidentBinding,
    enumerate_identity_hypotheses,
    forbidden_inference_paths,
)
from conceptgraph.revision.autonomous_identity import (
    frame_index,
    partition_hash,
    relevant_future_observations,
)
from conceptgraph.revision.constraints import SparseRepairConstraint
from conceptgraph.revision.counterfactual_projection import ProjectionEvidenceLoader
from conceptgraph.revision.evidence_split import sha256_file
from conceptgraph.revision.identity_evidence import IdentityEvidenceBundleBuilder
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.snapshot import AnchorStateBuilder
from scripts.build_revision_identity_machine_intake import (
    _blind_uid,
    _has_machine_future_visibility,
)
from scripts.freeze_revision_identity_selective_v0 import (
    SceneContext,
    _candidate_seed_version,
    _read,
    _replay,
    _runtime_validity,
    _write,
)

QUARTILES = (
    ("Q1_F000_049", 0, 49),
    ("Q2_F050_099", 50, 99),
    ("Q3_F100_149", 100, 149),
    ("Q4_F150_199", 150, 199),
)


def _quartile(anchor_frame: int) -> str:
    for name, start, end in QUARTILES:
        if start <= int(anchor_frame) <= end:
            return name
    raise ValueError(f"anchor frame outside frozen room0 pilot range: {anchor_frame}")


def _rank_key(seed: str, incident_uid: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{incident_uid}".encode("utf-8")).hexdigest()
    return digest, str(incident_uid)


def stratified_round_robin(
    eligible: Sequence[Mapping[str, Any]], *, seed: str, limit: int
) -> list[dict[str, Any]]:
    """Select using only frozen strata and seeded hashes."""

    buckets = {name: [] for name, _, _ in QUARTILES}
    for row in eligible:
        buckets[str(row["anchor_quartile"])].append(dict(row))
    for rows in buckets.values():
        rows.sort(key=lambda row: _rank_key(seed, str(row["source_incident_uid"])))
    selected: list[dict[str, Any]] = []
    cursor = 0
    names = [name for name, _, _ in QUARTILES]
    while len(selected) < int(limit):
        added = False
        for name in names:
            if cursor < len(buckets[name]) and len(selected) < int(limit):
                selected.append(buckets[name][cursor])
                added = True
        if not added:
            break
        cursor += 1
    return selected


def _candidate_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in selection.get("selected") or ()
        if str(row.get("stage")) == "association"
        and not row.get("blocked_checker_ids")
        and row.get("incident_uid")
        and row.get("representative_finding_uid")
    ]


def _evaluate_executable_feasibility(
    *,
    context: SceneContext,
    loader: ProjectionEvidenceLoader,
    binding: IncidentBinding,
    minimum_frame_gap: int,
) -> dict[str, Any]:
    provenance = context.provenance
    association = provenance.get_event(binding.event_uid)
    aliases = tuple(
        sorted(
            alias
            for alias, value in binding.aliases.items()
            if alias.startswith("CANDIDATE_") and value.complete
        )
    )
    if not aliases:
        raise ValueError("no complete candidate context alias")
    seed_versions = tuple(
        _candidate_seed_version(binding=binding, association=association, alias=alias)
        for alias in aliases
    )
    anchor_frame = frame_index(provenance.get_observation(binding.obs_uid))
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=binding.event_uid,
        seed_version_uids=seed_versions,
    )
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(anchor_frame)
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        binding.event_uid,
        seed_versions,
        strict=True,
        prefix_state=prefix_state,
        prefix_objects=prefix_objects,
    )
    noop_state, noop_wall_ms = _replay(
        context=context, snapshot=snapshot, closure=closure, constraints=()
    )
    noop_hash = partition_hash(noop_state.get("membership") or {})
    states = [noop_state]
    executable = []
    for compiled in enumerate_identity_hypotheses(binding, candidate_aliases=aliases):
        if compiled.get("hypothesis_action") != "SEPARATE_MEMBER_GROUPS":
            continue
        constraint = SparseRepairConstraint.from_mapping(
            compiled["candidate_constraint"]
        )
        state, wall_ms = _replay(
            context=context,
            snapshot=snapshot,
            closure=closure,
            constraints=(constraint,),
        )
        validity = _runtime_validity(
            context=context,
            state=state,
            constraints=(constraint,),
            snapshot_valid=bool(snapshot.validation["pass"]),
        )
        candidate_hash = partition_hash(state.get("membership") or {})
        states.append(state)
        executable.append(
            {
                "candidate_uid": str(compiled["constraint_fingerprint"]),
                "target_alias": str(compiled["hypothesis_target_alias"]),
                "partition_hash": candidate_hash,
                "distinct_from_noop": candidate_hash != noop_hash,
                "runtime_valid": bool(validity["valid"]),
                "replay_wall_ms": wall_ms,
            }
        )
    distinct = [
        row for row in executable if row["distinct_from_noop"] and row["runtime_valid"]
    ]
    if not distinct:
        raise ValueError("no runtime-valid distinct CREATE_INSTANCE partition")
    roots = [binding.obs_uid]
    roots.extend(binding.aliases[alias].origin_obs_uid for alias in aliases)
    future = relevant_future_observations(
        states=tuple(states),
        root_obs_uids=tuple(roots),
        observation_rows=provenance.observations,
        minimum_frame=anchor_frame + int(minimum_frame_gap),
    )
    frame_uids = []
    for obs_uid in future:
        observation = provenance.get_observation(obs_uid)
        if not observation.get("processed_mask_ref"):
            continue
        frame_uid = str(observation["frame_uid"])
        if frame_uid in frame_uids:
            continue
        loader.load_frame(frame_uid)
        frame_uids.append(frame_uid)
        if len(frame_uids) >= 2:
            break
    if len(frame_uids) < 2:
        raise ValueError("fewer than two hash-valid projectable future frames")
    return {
        "anchor_frame": anchor_frame,
        "anchor_quartile": _quartile(anchor_frame),
        "native_action": binding.observed_current_decision,
        "complete_candidate_aliases": list(aliases),
        "seed_version_uids": list(seed_versions),
        "noop_partition_hash": noop_hash,
        "distinct_candidate_count": len({row["partition_hash"] for row in distinct}),
        "executable_separation_candidates": executable,
        "future_projectable_frame_count": len(frame_uids),
        "noop_replay_wall_ms": noop_wall_ms,
        "snapshot_validation_pass": bool(snapshot.validation["pass"]),
        "source_immutable": context.source_hashes_before == provenance.source_hashes(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-selection", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--private-audit", required=True, type=Path)
    parser.add_argument("--seed", default="CMVIC_ROOM0_PILOT_V0_20260825")
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--minimum-count", type=int, default=6)
    parser.add_argument("--minimum-frame-gap", type=int, default=3)
    parser.add_argument("--exclude-incident", action="append", default=[])
    args = parser.parse_args()

    if not 1 <= args.minimum_count <= args.target_count:
        raise ValueError("require 1 <= minimum-count <= target-count")
    room_run = args.room_run.resolve()
    selection_path = args.room_selection.resolve()
    selection = _read(selection_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    provenance = ProvenanceIndex(room_run)
    builder = IdentityEvidenceBundleBuilder(provenance)
    context = SceneContext.build(room_run)
    loader = ProjectionEvidenceLoader(room_run)
    eligible = []
    rejected = []
    built_by_uid = {}
    prefiltered = []
    excluded_incidents = {str(item) for item in args.exclude_incident}
    source_candidates = [
        row
        for row in _candidate_rows(selection)
        if str(row["incident_uid"]) not in excluded_incidents
    ]
    for source in source_candidates:
        incident_uid = str(source["incident_uid"])
        finding_uid = str(source["representative_finding_uid"])
        packet = room_run / "audit_validity_gate_endpoint_v2_1" / "cases" / finding_uid
        try:
            case = _read(packet / "case.json")
            review = _read(packet / "review_evidence.json")
            event_uid = str(case["scope"]["event_uid"])
            if str(review.get("case_uid")) != incident_uid:
                raise ValueError("machine review incident mismatch")
            if not _has_machine_future_visibility(
                review,
                association_event_uid=event_uid,
                minimum_frame_gap=args.minimum_frame_gap,
            ):
                raise ValueError("no machine-visible future identity evidence")
            blind_case_uid = _blind_uid("room0", incident_uid)
            built = builder.build(
                case_uid=blind_case_uid,
                association_event_uid=event_uid,
                machine_review=review,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "source_incident_uid": incident_uid,
                    "source_finding_uid": finding_uid,
                    "screening_stage": "STATIC_MACHINE_EVIDENCE_PREFILTER",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        built_by_uid[blind_case_uid] = built
        anchor_frame = frame_index(provenance.get_observation(built.binding.obs_uid))
        prefiltered.append(
            {
                "case_uid": blind_case_uid,
                "scene_id": "room0",
                "source_incident_uid": incident_uid,
                "source_finding_uid": finding_uid,
                "anchor_frame": anchor_frame,
                "anchor_quartile": _quartile(anchor_frame),
                "native_action": built.binding.observed_current_decision,
            }
        )

    deterministic_screen_order = stratified_round_robin(
        prefiltered, seed=str(args.seed), limit=len(prefiltered)
    )
    screened_case_uids = []
    for row in deterministic_screen_order:
        case_uid = str(row["case_uid"])
        screened_case_uids.append(case_uid)
        try:
            feasibility = _evaluate_executable_feasibility(
                context=context,
                loader=loader,
                binding=built_by_uid[case_uid].binding,
                minimum_frame_gap=args.minimum_frame_gap,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "source_incident_uid": str(row["source_incident_uid"]),
                    "source_finding_uid": str(row["source_finding_uid"]),
                    "screening_stage": "EXECUTOR_AND_FUTURE_RGBD_FEASIBILITY",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        eligible.append({**row, **feasibility})
        if len(eligible) >= args.target_count:
            break

    selected = eligible
    if len(selected) < args.minimum_count:
        raise ValueError(
            f"only {len(selected)} eligible cases; minimum is {args.minimum_count}"
        )

    runtime_cases = []
    selected_uids = set()
    for selection_rank, row in enumerate(selected, 1):
        case_uid = str(row["case_uid"])
        selected_uids.add(case_uid)
        built = built_by_uid[case_uid]
        case_dir = output_root / case_uid
        bundle_path = case_dir / "bundle.machine_only.json"
        binding_path = case_dir / "binding.private.json"
        _write(bundle_path, built.inference_bundle)
        _write(binding_path, built.binding.as_dict())
        runtime_cases.append(
            {
                "case_uid": case_uid,
                "scene_id": "room0",
                "anchor_frame": int(row["anchor_frame"]),
                "anchor_quartile": str(row["anchor_quartile"]),
                "native_action": str(row["native_action"]),
                "selection_rank": selection_rank,
                "bundle_path": str(bundle_path.resolve()),
                "bundle_sha256": sha256_file(bundle_path),
                "binding_path": str(binding_path.resolve()),
                "binding_sha256": sha256_file(binding_path),
            }
        )

    selector_path = Path(__file__).resolve()
    runtime_manifest = {
        "schema_version": "1.0.0",
        "role": "CMVIC_ROOM0_BLIND_PILOT_RUNTIME_INPUT",
        "candidate_source": "MACHINE_ASSOCIATION_INCIDENT_WORKLIST_ONLY",
        "selection_policy": "FIXED_SEED_FRAME_QUARTILE_ROUND_ROBIN_AFTER_EXECUTOR_FEASIBILITY",
        "selection_seed": str(args.seed),
        "selector_code_sha256": sha256_file(selector_path),
        "eligible_pool_count": len(eligible),
        "eligible_pool": [
            {
                "case_uid": str(row["case_uid"]),
                "scene_id": "room0",
                "anchor_frame": int(row["anchor_frame"]),
                "anchor_quartile": str(row["anchor_quartile"]),
                "native_action": str(row["native_action"]),
                "selected": str(row["case_uid"]) in selected_uids,
            }
            for row in sorted(eligible, key=lambda item: str(item["case_uid"]))
        ],
        "case_count": len(runtime_cases),
        "cases": runtime_cases,
        "runtime_human_or_gold_loaded": False,
        "review_score_used_for_selection": False,
        "replay_outcome_used_only_as_executable_partition_feasibility": True,
        "static_prefilter_pool_count": len(prefiltered),
        "executor_screened_count": len(screened_case_uids),
        "excluded_prior_control_incidents": sorted(excluded_incidents),
    }
    forbidden = forbidden_inference_paths(runtime_manifest)
    if forbidden:
        raise ValueError("oracle-like runtime fields: " + ", ".join(forbidden))
    runtime_path = args.runtime_manifest.resolve()
    _write(runtime_path, runtime_manifest)
    private_audit = {
        "schema_version": "1.0.0",
        "role": "PRIVATE_CMVIC_PILOT_SELECTION_TRACE_NOT_INFERENCE_INPUT",
        "runtime_manifest_path": str(runtime_path),
        "runtime_manifest_sha256": sha256_file(runtime_path),
        "source_selection_manifest": str(selection_path),
        "source_selection_manifest_sha256": sha256_file(selection_path),
        "human_labels_loaded": False,
        "gold_actions_loaded": False,
        "eligible_cases": eligible,
        "selected_case_uids": [str(row["case_uid"]) for row in selected],
        "deterministic_screen_order": [
            str(row["case_uid"]) for row in deterministic_screen_order
        ],
        "screened_case_uids": screened_case_uids,
        "not_screened_after_target_reached": [
            str(row["case_uid"])
            for row in deterministic_screen_order
            if str(row["case_uid"]) not in set(screened_case_uids)
        ],
        "rejected_cases": rejected,
    }
    _write(args.private_audit.resolve(), private_audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_candidate_count": len(source_candidates),
                "eligible_count": len(eligible),
                "selected_count": len(runtime_cases),
                "quartile_counts": {
                    name: sum(row["anchor_quartile"] == name for row in selected)
                    for name, _, _ in QUARTILES
                },
                "runtime_manifest": str(runtime_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
