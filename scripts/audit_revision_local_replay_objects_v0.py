#!/usr/bin/env python3
"""Audit exact state parity between legacy and raw-object sparse replay APIs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from conceptgraph.revision.auto_constraints import IncidentBinding
from conceptgraph.revision.constraints import ReplayMode
from conceptgraph.revision.evidence_split import sha256_file
from conceptgraph.revision.snapshot import AnchorStateBuilder
from scripts.freeze_revision_identity_selective_v0 import (
    SceneContext,
    _candidate_seed_version,
    _read,
    _write,
)


def _object_summary_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "entity_uid": str(row.get("entity_uid") or ""),
                "member_observation_uids": sorted(
                    str(item) for item in row.get("member_observation_uids") or ()
                ),
                "point_count": int(row.get("point_count") or 0),
            }
            for row in state.get("objects") or ()
        ),
        key=lambda row: (
            row["member_observation_uids"],
            row["entity_uid"],
        ),
    )


def _raw_member_partitions(objects: list[Any]) -> list[list[str]]:
    return sorted(
        sorted(str(item) for item in obj.get("obs_uids") or ())
        for obj in objects
        if obj.get("obs_uids")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--candidate-alias", default="CANDIDATE_1_CONTEXT")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    binding_path = args.binding.resolve()
    binding = IncidentBinding.from_mapping(_read(binding_path))
    context = SceneContext.build(args.room_run.resolve())
    provenance = context.provenance
    association = provenance.get_event(binding.event_uid)
    seed_version = _candidate_seed_version(
        binding=binding,
        association=association,
        alias=args.candidate_alias,
    )
    anchor_frame = int(
        provenance.get_observation(binding.obs_uid).get("frame_index")
        or str(provenance.get_observation(binding.obs_uid)["frame_uid"]).rsplit(
            "_f", 1
        )[1]
    )
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=binding.event_uid,
        seed_version_uids=(seed_version,),
    )
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(anchor_frame)
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        binding.event_uid,
        (seed_version,),
        strict=True,
        prefix_state=prefix_state,
        prefix_objects=prefix_objects,
    )
    common = {
        "mode": ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        "snapshot_objects": snapshot.objects,
        "snapshot_runtime_ms": snapshot.state["runtime_ms"],
        "snapshot_timing": snapshot.state.get("timing"),
        "anchor_frame": snapshot.anchor_frame,
        "snapshot_watermark_event_sequence": snapshot.watermark_event_sequence,
        "closure": closure,
        "constraints": (),
        "current_state": context.native_state,
    }
    started = time.perf_counter()
    legacy_state = context.engine.replay_local_from_snapshot(**common)
    legacy_wall_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    raw_state, raw_objects = context.engine.replay_local_from_snapshot_with_objects(
        **common
    )
    raw_wall_ms = (time.perf_counter() - started) * 1000.0

    raw_partitions = _raw_member_partitions(raw_objects)
    state_partitions = sorted(
        sorted(str(item) for item in members)
        for members in (raw_state.get("membership") or {}).values()
        if members
    )
    result = {
        "schema_version": "1.0.0",
        "case_uid": binding.case_uid,
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "candidate_alias": args.candidate_alias,
        "seed_version_uid": seed_version,
        "legacy_state_hash": legacy_state.get("state_hash"),
        "raw_api_state_hash": raw_state.get("state_hash"),
        "state_hash_exact": legacy_state.get("state_hash")
        == raw_state.get("state_hash"),
        "membership_exact": legacy_state.get("membership")
        == raw_state.get("membership"),
        "object_summary_exact": (
            _object_summary_from_state(legacy_state)
            == _object_summary_from_state(raw_state)
        ),
        "raw_object_members_present_in_state": all(
            partition in state_partitions for partition in raw_partitions
        ),
        "raw_object_partition_count": len(raw_partitions),
        "source_hashes_unchanged": (
            context.source_hashes_before == provenance.source_hashes()
        ),
        "snapshot_validation_pass": bool(snapshot.validation["pass"]),
        "legacy_wall_ms": legacy_wall_ms,
        "raw_api_wall_ms": raw_wall_ms,
    }
    result["status"] = (
        "PASS"
        if all(
            result[key]
            for key in (
                "state_hash_exact",
                "membership_exact",
                "object_summary_exact",
                "raw_object_members_present_in_state",
                "source_hashes_unchanged",
                "snapshot_validation_pass",
            )
        )
        else "FAIL"
    )
    _write(args.output.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
