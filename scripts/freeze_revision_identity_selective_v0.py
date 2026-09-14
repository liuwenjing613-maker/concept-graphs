#!/usr/bin/env python3
"""Freeze oracle-free identity replay outcomes and held-out critic requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from conceptgraph.revision.auto_constraints import (
    IncidentBinding,
    enumerate_identity_hypotheses,
    forbidden_inference_paths,
)
from conceptgraph.revision.autonomous_identity import (
    anonymous_state_summary,
    balanced_partition_sample,
    build_pairwise_critic_prompt,
    distinct_candidate_partitions,
    evenly_spaced,
    frame_index,
    partition_hash,
    relevant_future_observations,
)
from conceptgraph.revision.candidate_verifier import CandidateVerifier
from conceptgraph.revision.capabilities import enumerate_feasible_actions
from conceptgraph.revision.constraints import ReplayMode, SparseRepairConstraint
from conceptgraph.revision.dependency_graph import TypedDependencyGraph
from conceptgraph.revision.evidence_split import (
    EvidenceReference,
    EvidenceSplitManifest,
    sha256_file,
)
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.replay import CounterfactualReplayEngine
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.snapshot import AnchorStateBuilder, IncrementalPrefixCache
from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine
from conceptgraph.revision.vlm import _load_crop


@dataclass
class SceneContext:
    provenance: ProvenanceIndex
    engine: SparseCounterfactualReplayEngine
    native_state: dict[str, Any]
    dependency_graph: TypedDependencyGraph
    prefix_cache: IncrementalPrefixCache
    source_hashes_before: dict[str, str]
    build_wall_ms: float

    @classmethod
    def build(cls, base_run: Path) -> "SceneContext":
        started = time.perf_counter()
        provenance = ProvenanceIndex(base_run)
        engine = SparseCounterfactualReplayEngine(provenance)
        native = CounterfactualReplayEngine(provenance).clean_state()
        native["branch"] = "NATIVE_FROZEN_ENDPOINT"
        return cls(
            provenance=provenance,
            engine=engine,
            native_state=native,
            dependency_graph=TypedDependencyGraph(provenance),
            prefix_cache=IncrementalPrefixCache(engine),
            source_hashes_before=provenance.source_hashes(),
            build_wall_ms=(time.perf_counter() - started) * 1000.0,
        )


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_seed_version(
    *, binding: IncidentBinding, association: Mapping[str, Any], alias: str
) -> str:
    target = binding.aliases[alias]
    objects = [str(item) for item in association.get("object_uids_before") or ()]
    versions = [
        str(item) for item in association.get("candidate_object_version_uids") or ()
    ]
    matches = [
        versions[index]
        for index, object_uid in enumerate(objects)
        if object_uid == target.entity_uid and index < len(versions)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{binding.case_uid}: cannot bind {alias} to one pre-event version"
        )
    return matches[0]


def _scoreable_in_every_state(
    obs_uid: str, states: Sequence[Mapping[str, Any]]
) -> bool:
    for state in states:
        decisions = {
            str(row.get("obs_uid")): row for row in state.get("decision_trace") or ()
        }
        decision = decisions.get(obs_uid)
        if decision is None:
            return False
        threshold = (decision.get("threshold_semantics") or {}).get("sim_threshold")
        if threshold is None:
            return False
        applied = decision.get("applied_match")
        if applied is not None and not any(
            row.get("score") is not None and int(row.get("index")) == int(applied)
            for row in decision.get("natural_candidates") or ()
        ):
            return False
    return True


def _freeze_crops(
    *,
    provenance: ProvenanceIndex,
    obs_uids: Sequence[str],
    output_dir: Path,
    id_prefix: str,
    source_role: str,
) -> tuple[list[EvidenceReference], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    references = []
    rows = []
    for index, obs_uid in enumerate(obs_uids, 1):
        evidence_id = f"{id_prefix}{index:02d}"
        path = (output_dir / f"{evidence_id}.png").resolve()
        _load_crop(provenance, obs_uid).save(path, format="PNG")
        observation = provenance.get_observation(obs_uid)
        digest = sha256_file(path)
        reference = EvidenceReference.build(
            obs_uid=obs_uid,
            frame_index=frame_index(observation),
            sha256=digest,
            source_role=source_role,
            artifact_path=path,
        )
        references.append(reference)
        rows.append(
            {
                "evidence_id": evidence_id,
                "obs_uid": obs_uid,
                "frame_index": reference.frame_index,
                "class_name": str(observation.get("class_name") or "unknown"),
                "sha256": digest,
                "path": str(path),
                "source_role": source_role,
            }
        )
    return references, rows


def _compact_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_hash": state.get("state_hash"),
        "partition_hash": partition_hash(state.get("membership") or {}),
        "active_object_count": len(state.get("membership") or {}),
        "runtime_ms": state.get("runtime_ms"),
        "timing": state.get("timing") or {},
        "constraint_hit_count": state.get("constraint_hit_count", 0),
        "persistent_create_instance_merge_veto_count": state.get(
            "persistent_create_instance_merge_veto_count", 0
        ),
        "persistent_create_instance_association_veto_count": state.get(
            "persistent_create_instance_association_veto_count", 0
        ),
        "persistent_lineage_redirect_override_count": state.get(
            "persistent_lineage_redirect_override_count", 0
        ),
    }


def _runtime_validity(
    *,
    context: SceneContext,
    state: Mapping[str, Any],
    constraints: Sequence[SparseRepairConstraint],
    snapshot_valid: bool,
) -> dict[str, Any]:
    verification = InvariantVerifier().verify(
        state=state,
        constraints=constraints,
        source_hashes_before=context.source_hashes_before,
        source_hashes_after=context.provenance.source_hashes(),
        known_observation_uids=context.provenance.observations,
    )
    return {
        "valid": bool(verification["pass"] and snapshot_valid),
        "invariants": verification,
        "snapshot_valid": bool(snapshot_valid),
        "source_immutable": (
            context.source_hashes_before == context.provenance.source_hashes()
        ),
    }


def _replay(
    *,
    context: SceneContext,
    snapshot: Any,
    closure: Any,
    constraints: Sequence[SparseRepairConstraint],
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    state = context.engine.replay_local_from_snapshot(
        mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        constraints=list(constraints),
        current_state=context.native_state,
    )
    return state, (time.perf_counter() - started) * 1000.0


def _freeze_case(
    *,
    row: Mapping[str, Any],
    context: SceneContext,
    output_root: Path,
    candidate_alias: str,
    minimum_frame_gap: int,
    maximum_verification_images: int,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    case_uid = str(row["case_uid"])
    case_dir = output_root / case_uid
    binding_path = Path(str(row["binding_path"])).resolve()
    if sha256_file(binding_path) != str(row["binding_sha256"]):
        raise ValueError(f"{case_uid}: binding hash drift")
    binding = IncidentBinding.from_mapping(_read(binding_path))
    if binding.case_uid != case_uid:
        raise ValueError(f"{case_uid}: binding case mismatch")
    if (
        candidate_alias not in binding.aliases
        or not binding.aliases[candidate_alias].complete
    ):
        raise ValueError(f"{case_uid}: incomplete candidate alias {candidate_alias}")

    provenance = context.provenance
    association = provenance.get_event(binding.event_uid)
    if str(association.get("obs_uid")) != binding.obs_uid:
        raise ValueError(f"{case_uid}: event/observation binding mismatch")
    anchor_frame = frame_index(provenance.get_observation(binding.obs_uid))
    seed_version = _candidate_seed_version(
        binding=binding, association=association, alias=candidate_alias
    )
    target = binding.aliases[candidate_alias]

    snapshot_started = time.perf_counter()
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=binding.event_uid,
        seed_version_uids=[seed_version],
    )
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(anchor_frame)
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        binding.event_uid,
        [seed_version],
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

    compiled_rows = enumerate_identity_hypotheses(
        binding, candidate_aliases=[candidate_alias]
    )
    feasible_actions = enumerate_feasible_actions(
        identity_candidate_count=1,
        observed_current_decision=binding.observed_current_decision,
        created_identity_binding_complete=bool(
            binding.created_entity_uid and binding.created_identity_uid
        ),
    )
    replay_rows = []
    states = [noop_state]
    for compiled in compiled_rows:
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
        state_partition_hash = partition_hash(state.get("membership") or {})
        replay_rows.append(
            {
                "candidate_uid": str(compiled["constraint_fingerprint"]),
                "action": str(compiled["hypothesis_action"]),
                "target_alias": str(compiled["hypothesis_target_alias"]),
                "constraint": primitive.as_dict(),
                "partition_hash": state_partition_hash,
                "runtime_validity": validity,
                "replay_wall_ms": replay_wall_ms,
                "state_audit": _compact_state(state),
                "_state": state,
            }
        )
        states.append(state)

    noop_partition_hash = partition_hash(noop_state.get("membership") or {})
    state_by_partition = {noop_partition_hash: noop_state}
    for replay_row in replay_rows:
        state_by_partition.setdefault(
            replay_row["partition_hash"], replay_row["_state"]
        )
    distinct_repair_hashes = list(
        distinct_candidate_partitions(noop_partition_hash, state_by_partition)
    )
    if not distinct_repair_hashes:
        private_replays = [
            {key: value for key, value in row.items() if key != "_state"}
            for row in replay_rows
        ]
        private = {
            "schema_version": "1.0.0",
            "case_uid": case_uid,
            "binding_path": str(binding_path),
            "binding_sha256": sha256_file(binding_path),
            "candidate_alias": candidate_alias,
            "seed_version_uid": seed_version,
            "noop_partition_hash": noop_partition_hash,
            "noop_runtime_validity": noop_validity,
            "noop_state_audit": _compact_state(noop_state),
            "candidate_replays": private_replays,
            "critic_state_mappings": {},
            "precritic_disposition": "NO_DISTINCT_EXECUTABLE_REPAIR",
        }
        _write(case_dir / "execution.private.json", private)
        result = {
            "schema_version": "1.0.0",
            "case_uid": case_uid,
            "scene_id": str(row["scene_id"]),
            "observed_current_decision": binding.observed_current_decision,
            "feasible_actions": list(feasible_actions),
            "finite_constraint_count": len(compiled_rows),
            "unique_executed_partition_count": len(state_by_partition),
            "noop_equivalent_action_count": len(replay_rows),
            "distinct_repair_partition_count": 0,
            "evidence_split": None,
            "proposal_evidence": [],
            "verification_evidence": [],
            "future_pool_count": 0,
            "common_scoreable_future_count": 0,
            "primary_candidate_scores": [],
            "critic_requests": [],
            "timing": {
                "context_build_wall_ms_shared": context.build_wall_ms,
                "snapshot_and_closure_wall_ms": snapshot_wall_ms,
                "noop_replay_wall_ms": noop_wall_ms,
                "candidate_replay_wall_ms": [
                    item["replay_wall_ms"] for item in replay_rows
                ],
                "case_total_wall_ms": (time.perf_counter() - case_started) * 1000.0,
            },
            "gold_loaded": False,
            "human_verdict_loaded": False,
            "semantic_threshold_count": 0,
            "precritic_disposition": "DEFER_NO_DISTINCT_EXECUTABLE_REPAIR",
            "status": "NO_DISTINCT_EXECUTABLE_REPAIR",
        }
        _write(case_dir / "case_result.frozen.json", result)
        return result

    proposal_members = [
        obs_uid
        for obs_uid in provenance.get_member_observations(seed_version)
        if frame_index(provenance.get_observation(obs_uid)) <= anchor_frame
        and obs_uid != binding.obs_uid
    ]
    proposal_members.sort(
        key=lambda obs_uid: (frame_index(provenance.get_observation(obs_uid)), obs_uid)
    )
    proposal_obs = tuple(
        dict.fromkeys(
            (binding.obs_uid,) + evenly_spaced(tuple(proposal_members), limit=2)
        )
    )
    proposal_refs, proposal_rows = _freeze_crops(
        provenance=provenance,
        obs_uids=proposal_obs,
        output_dir=case_dir / "proposal_evidence",
        id_prefix="P",
        source_role="PROPOSAL_ONLY_PREANCHOR_CONTEXT",
    )

    minimum_frame = anchor_frame + minimum_frame_gap
    future_pool = relevant_future_observations(
        states=tuple(state_by_partition.values()),
        root_obs_uids=(binding.obs_uid, str(target.origin_obs_uid)),
        observation_rows=provenance.observations,
        minimum_frame=minimum_frame,
    )
    scoreable_pool = tuple(
        obs_uid
        for obs_uid in future_pool
        if provenance.get_observation(obs_uid).get("crop_ref")
        and _scoreable_in_every_state(obs_uid, tuple(state_by_partition.values()))
    )
    verification_obs = balanced_partition_sample(
        values=scoreable_pool,
        states=tuple(state_by_partition.values()),
        observation_rows=provenance.observations,
        limit=maximum_verification_images,
    )
    if len(verification_obs) < 2:
        raise ValueError(
            f"{case_uid}: fewer than two common scoreable future observations"
        )
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

    verifier = CandidateVerifier()
    score_rows = []
    request_rows = []
    private_mappings = {}
    evidence_id_by_obs = {
        str(item["obs_uid"]): str(item["evidence_id"]) for item in verification_rows
    }
    for repair_index, repair_hash in enumerate(distinct_repair_hashes, 1):
        implementations = [
            item for item in replay_rows if item["partition_hash"] == repair_hash
        ]
        chosen = next(
            (item for item in implementations if item["runtime_validity"]["valid"]),
            implementations[0],
        )
        score = verifier.score_identity(
            incident_uid=case_uid,
            candidate_uid=str(chosen["candidate_uid"]),
            candidate_state=state_by_partition[repair_hash],
            noop_state=noop_state,
            split=split,
            runtime_valid=bool(chosen["runtime_validity"]["valid"]),
        )
        score_rows.append(
            {
                **score.as_dict(),
                "partition_hash": repair_hash,
                "implementation_count": len(implementations),
            }
        )
        pair_uid = f"PAIR_{repair_index:02d}"
        for order_index, order in enumerate(
            (
                (noop_partition_hash, repair_hash),
                (repair_hash, noop_partition_hash),
            )
        ):
            label_to_hash = {"STATE_A": order[0], "STATE_B": order[1]}
            summaries = {
                state_id: anonymous_state_summary(
                    state=state_by_partition[state_hash],
                    evidence_id_by_obs=evidence_id_by_obs,
                    observation_rows=provenance.observations,
                )
                for state_id, state_hash in label_to_hash.items()
            }
            prompt = build_pairwise_critic_prompt(
                incident_uid=case_uid,
                evidence_rows=verification_rows,
                state_summaries=summaries,
            )
            request_uid = f"{case_uid}_{pair_uid}_ORDER_{order_index}"
            request_path = case_dir / "critic_requests" / f"{request_uid}.json"
            request = {
                "schema_version": "1.0.0",
                "request_uid": request_uid,
                "case_uid": case_uid,
                "pair_uid": pair_uid,
                "order_index": order_index,
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "images": verification_rows,
                "allowed_state_ids": ["STATE_A", "STATE_B"],
                "allowed_evidence_ids": [
                    str(item["evidence_id"]) for item in verification_rows
                ],
                "evidence_split_uid": split.manifest_uid,
                "action_names_hidden_from_critic": True,
                "proposal_evidence_hidden_from_critic": True,
                "state_order_swapped": bool(order_index),
            }
            forbidden = forbidden_inference_paths(request)
            if forbidden:
                raise ValueError(
                    "oracle-like frozen request fields: " + ", ".join(forbidden)
                )
            _write(request_path, request)
            private_mappings[request_uid] = {
                "label_to_partition_hash": label_to_hash,
                "candidate_partition_hash": repair_hash,
                "noop_partition_hash": noop_partition_hash,
                "candidate_uid": chosen["candidate_uid"],
            }
            request_rows.append(
                {
                    "request_uid": request_uid,
                    "case_uid": case_uid,
                    "pair_uid": pair_uid,
                    "order_index": order_index,
                    "path": str(request_path.resolve()),
                    "sha256": sha256_file(request_path),
                }
            )

    private_replays = []
    for replay_row in replay_rows:
        private_replays.append(
            {key: value for key, value in replay_row.items() if key != "_state"}
        )
    private = {
        "schema_version": "1.0.0",
        "case_uid": case_uid,
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "candidate_alias": candidate_alias,
        "seed_version_uid": seed_version,
        "noop_partition_hash": noop_partition_hash,
        "noop_runtime_validity": noop_validity,
        "noop_state_audit": _compact_state(noop_state),
        "candidate_replays": private_replays,
        "critic_state_mappings": private_mappings,
    }
    _write(case_dir / "execution.private.json", private)

    result = {
        "schema_version": "1.0.0",
        "case_uid": case_uid,
        "scene_id": str(row["scene_id"]),
        "observed_current_decision": binding.observed_current_decision,
        "feasible_actions": list(feasible_actions),
        "finite_constraint_count": len(compiled_rows),
        "unique_executed_partition_count": len(state_by_partition),
        "noop_equivalent_action_count": sum(
            item["partition_hash"] == noop_partition_hash for item in replay_rows
        ),
        "distinct_repair_partition_count": len(distinct_repair_hashes),
        "evidence_split": split.as_dict(),
        "proposal_evidence": proposal_rows,
        "verification_evidence": verification_rows,
        "future_pool_count": len(future_pool),
        "common_scoreable_future_count": len(scoreable_pool),
        "primary_candidate_scores": score_rows,
        "critic_requests": request_rows,
        "timing": {
            "context_build_wall_ms_shared": context.build_wall_ms,
            "snapshot_and_closure_wall_ms": snapshot_wall_ms,
            "noop_replay_wall_ms": noop_wall_ms,
            "candidate_replay_wall_ms": [
                item["replay_wall_ms"] for item in replay_rows
            ],
            "case_total_wall_ms": (time.perf_counter() - case_started) * 1000.0,
        },
        "gold_loaded": False,
        "human_verdict_loaded": False,
        "semantic_threshold_count": 0,
        "status": "FROZEN_PENDING_OUTCOME_CRITIC",
    }
    _write(case_dir / "case_result.frozen.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--office-run", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--candidate-alias", default="CANDIDATE_1_CONTEXT")
    parser.add_argument("--minimum-frame-gap", type=int, default=3)
    parser.add_argument("--maximum-verification-images", type=int, default=8)
    args = parser.parse_args()

    manifest = _read(args.case_manifest.resolve())
    forbidden = forbidden_inference_paths(manifest)
    if forbidden:
        raise ValueError("oracle-like case manifest fields: " + ", ".join(forbidden))
    cases = list(manifest.get("cases") or ())
    selected = set(str(item) for item in args.only_case)
    if selected:
        cases = [row for row in cases if str(row.get("case_uid")) in selected]
        if len(cases) != len(selected):
            raise ValueError("one or more --only-case IDs were not found")
    if not cases:
        raise ValueError("case manifest selected no cases")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    base_runs = {
        "office0": args.office_run.resolve(),
        "room0": args.room_run.resolve(),
    }
    contexts = {}
    results = []
    run_started = time.perf_counter()
    for row in cases:
        scene_id = str(row["scene_id"])
        if scene_id not in base_runs:
            raise ValueError(f"unknown scene_id: {scene_id}")
        if scene_id not in contexts:
            contexts[scene_id] = SceneContext.build(base_runs[scene_id])
        results.append(
            _freeze_case(
                row=row,
                context=contexts[scene_id],
                output_root=output_root,
                candidate_alias=args.candidate_alias,
                minimum_frame_gap=args.minimum_frame_gap,
                maximum_verification_images=args.maximum_verification_images,
            )
        )

    request_rows = [
        request for result in results for request in result["critic_requests"]
    ]
    protocol = {
        "schema_version": "1.0.0",
        "role": "DEVELOPMENT_SHADOW_NOT_PRODUCTION_COMMIT",
        "case_manifest_path": str(args.case_manifest.resolve()),
        "case_manifest_sha256": sha256_file(args.case_manifest.resolve()),
        "case_count": len(results),
        "request_count": len(request_rows),
        "cases": [
            {
                "case_uid": result["case_uid"],
                "scene_id": result["scene_id"],
                "result_path": str(
                    (
                        output_root / result["case_uid"] / "case_result.frozen.json"
                    ).resolve()
                ),
            }
            for result in results
        ],
        "critic_requests": request_rows,
        "runtime_human_or_gold_loaded": False,
        "candidate_source": "FINITE_EXECUTOR_CAPABILITIES",
        "candidate_state_deduplication": "ENTITY_ID_INVARIANT_PARTITION_HASH",
        "verification_evidence_policy": (
            "ALL_STATE_UNION_THEN_COMMON_REFINEMENT_BALANCE_AND_TEMPORAL_EVEN_SAMPLE"
        ),
        "minimum_frame_gap": args.minimum_frame_gap,
        "maximum_verification_images": args.maximum_verification_images,
        "critic_position_audit": "TWO_ORDER_SWAPPED_REQUESTS_PER_PAIR",
        "production_commit_permitted": False,
        "calibration_status": "NOT_YET_FIT",
        "semantic_threshold_count": 0,
        "total_wall_ms": (time.perf_counter() - run_started) * 1000.0,
        "protocol_uid": "freeze_protocol_" + _sha256_json(request_rows)[:20],
    }
    _write(output_root / "freeze_protocol.json", protocol)
    print(
        json.dumps(
            {
                "status": "PASS",
                "case_count": len(results),
                "request_count": len(request_rows),
                "output_root": str(output_root),
                "total_wall_ms": protocol["total_wall_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
