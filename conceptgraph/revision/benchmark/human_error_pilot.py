from __future__ import annotations

import copy
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..constraints import ReplayMode, SparseRepairConstraint
from ..dependency_graph import TypedDependencyGraph
from ..index import ProvenanceIndex
from ..relations import (
    AliDevBaselineRelationBackend,
    load_baseline_frame_records,
    remap_frame_records,
)
from ..replay import CounterfactualReplayEngine
from ..runtime_verify import InvariantVerifier
from ..snapshot import AnchorStateBuilder, IncrementalPrefixCache
from ..sparse_replay import SparseCounterfactualReplayEngine


REPLAYABLE = "REPLAYABLE_ASSOCIATION_CAUSE"
DEFERRED = "DEFER_NON_ASSOCIATION_ROOT"
DESIRED_RELATIONS = {"SAME_OWNER", "DIFFERENT_OWNER"}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("r", encoding="utf-8", newline=None) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), ""):
            digest.update(block.encode("utf-8"))
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("human-error pilot manifest must be a JSON object")
    return value


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expert_queue: Sequence[Mapping[str, Any]] | None = None,
    r1_labels: Sequence[Mapping[str, Any]] | None = None,
    r2_labels: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the frozen six-case cohort before any replay outcome is read."""

    cases = list(manifest.get("cases") or ())
    if len(cases) != 6:
        raise ValueError("pilot must contain exactly six human-confirmed cases")
    case_uids = [str(item.get("case_uid") or "") for item in cases]
    incident_uids = [str(item.get("incident_uid") or "") for item in cases]
    if any(not item for item in case_uids + incident_uids):
        raise ValueError("every pilot case needs case_uid and incident_uid")
    if len(set(case_uids)) != len(case_uids):
        raise ValueError("duplicate case_uid in pilot manifest")
    if len(set(incident_uids)) != len(incident_uids):
        raise ValueError("duplicate incident_uid in pilot manifest")

    error_counts = Counter(str(item.get("endpoint_error_type")) for item in cases)
    if error_counts != Counter({"FALSE_MERGE": 3, "FALSE_SPLIT": 3}):
        raise ValueError("pilot must contain exactly three FALSE_MERGE and three FALSE_SPLIT")
    dispositions = Counter(str(item.get("causal_disposition")) for item in cases)
    if dispositions != Counter({REPLAYABLE: 3, DEFERRED: 3}):
        raise ValueError("pilot must freeze exactly three replayable and three deferred cases")
    scenes = sorted(set(str(item.get("scene_id") or "") for item in cases))
    if any(not item for item in scenes) or len(scenes) > 2:
        raise ValueError("pilot must use one or two named scenes")

    for case in cases:
        human = case.get("human_label") or {}
        if (
            human.get("evidence_sufficient") != "YES"
            or human.get("final_state") != "WRONG"
            or human.get("final_error_type") != case.get("endpoint_error_type")
        ):
            raise ValueError(f"case {case['case_uid']} is not a confirmed endpoint error")
        disposition = str(case["causal_disposition"])
        if disposition == REPLAYABLE:
            if not case.get("anchor_association_event_uid") or not case.get("anchor_obs_uid"):
                raise ValueError(f"replayable case {case['case_uid']} lacks an anchor")
            constraints = list(case.get("constraints") or ())
            if len(constraints) != 1:
                raise ValueError(f"replayable case {case['case_uid']} needs one sparse primitive")
            SparseRepairConstraint.from_mapping(constraints[0])
            desired = (case.get("evaluation") or {}).get("desired_owner_relation")
            if desired not in DESIRED_RELATIONS:
                raise ValueError(f"case {case['case_uid']} has invalid endpoint criterion")
            if len((case.get("evaluation") or {}).get("groups") or ()) < 2:
                raise ValueError(f"case {case['case_uid']} needs at least two evidence groups")
        elif not str(case.get("defer_reason") or "").strip():
            raise ValueError(f"deferred case {case['case_uid']} lacks a reason")

    by_incident = {str(item["incident_uid"]): item for item in cases}
    queue_index = {
        str(item.get("incident_uid")): item for item in (expert_queue or ())
    }
    label_index = {
        str(item.get("incident_uid")): item for item in (r1_labels or ())
    }
    r2_index = {
        str(item.get("incident_uid")): item for item in (r2_labels or ())
    }
    if expert_queue is not None:
        for incident_uid, case in by_incident.items():
            row = queue_index.get(incident_uid)
            if row is None:
                raise ValueError(f"selected incident absent from expert queue: {incident_uid}")
            expected = (
                str(case["scene_id"]),
                str(case["endpoint_error_type"]),
                str(case["representative_finding_uid"]),
            )
            actual = (
                str(row.get("scene_id")),
                str(row.get("endpoint_error_type")),
                str(row.get("representative_finding_uid")),
            )
            if actual != expected:
                raise ValueError(f"expert queue metadata mismatch for {incident_uid}")
    if r1_labels is not None:
        for incident_uid, case in by_incident.items():
            row = label_index.get(incident_uid)
            if row is None:
                raise ValueError(f"selected incident absent from frozen R1 labels: {incident_uid}")
            human = case["human_label"]
            for key in ("evidence_sufficient", "final_state", "final_error_type", "notes"):
                if row.get(key) != human.get(key):
                    raise ValueError(f"frozen R1 label mismatch for {incident_uid}: {key}")
    r2_matches = 0
    if r2_labels is not None:
        for incident_uid, row in r2_index.items():
            case = by_incident.get(incident_uid)
            if case is None:
                continue
            if (
                row.get("evidence_sufficient") != "YES"
                or row.get("final_state") != "WRONG"
                or row.get("final_error_type") != case.get("endpoint_error_type")
            ):
                raise ValueError(f"R2 contradicts selected endpoint label: {incident_uid}")
            r2_matches += 1
    return {
        "pass": True,
        "case_count": len(cases),
        "endpoint_error_type_counts": dict(sorted(error_counts.items())),
        "causal_disposition_counts": dict(sorted(dispositions.items())),
        "scene_ids": scenes,
        "r2_confirmed_selected_case_count": r2_matches,
        "selection_is_purposive": True,
        "population_inference_to_all_confirmed_errors": False,
    }


def verify_frozen_sources(
    manifest: Mapping[str, Any], source_paths: Mapping[str, str | Path]
) -> dict[str, Any]:
    expected = manifest.get("source_artifacts") or {}
    rows = []
    for name, path in sorted(source_paths.items()):
        actual = _sha256_file(path)
        wanted = str((expected.get(name) or {}).get("sha256") or "")
        rows.append(
            {
                "name": name,
                "path": str(Path(path).resolve()),
                "expected_sha256": wanted,
                "actual_sha256": actual,
                "hash_basis": "UTF8_TEXT_WITH_CANONICAL_LF_NEWLINES",
                "pass": bool(wanted) and actual == wanted,
            }
        )
    missing = sorted(set(expected) - set(source_paths))
    if missing or not all(item["pass"] for item in rows):
        raise ValueError(f"frozen human source verification failed; missing={missing}")
    return {"pass": True, "artifacts": rows}


def _owner_index(membership: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for entity_uid, members in membership.items():
        for obs_uid in members or ():
            owners.setdefault(str(obs_uid), []).append(str(entity_uid))
    return owners


def _partition_signature(
    membership: Mapping[str, Iterable[str]], observations: Iterable[str]
) -> tuple[tuple[str, ...], ...]:
    selected = set(str(item) for item in observations)
    groups = {
        tuple(sorted(set(str(item) for item in members or ()) & selected))
        for members in membership.values()
    }
    return tuple(sorted(group for group in groups if group))


def _owner_partition_map(
    membership: Mapping[str, Iterable[str]], observations: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    selected = set(str(item) for item in observations)
    result: dict[str, tuple[str, ...]] = {}
    for members in membership.values():
        group = tuple(sorted(set(str(item) for item in members or ()) & selected))
        for obs_uid in group:
            result[obs_uid] = group
    return result


def evaluate_endpoint_groups(
    membership: Mapping[str, Iterable[str]],
    groups: Mapping[str, Iterable[str]],
    desired_owner_relation: str,
    *,
    probes: Sequence[str] = (),
) -> dict[str, Any]:
    desired = str(desired_owner_relation)
    if desired not in DESIRED_RELATIONS:
        raise ValueError(f"unsupported desired owner relation: {desired}")
    owners = _owner_index(membership)
    normalized = {
        str(name): tuple(sorted(set(str(item) for item in members)))
        for name, members in groups.items()
    }
    group_owners = {
        name: sorted(
            {
                entity_uid
                for obs_uid in members
                for entity_uid in owners.get(obs_uid, ())
            }
        )
        for name, members in normalized.items()
    }
    missing = sorted(
        obs_uid
        for members in normalized.values()
        for obs_uid in members
        if obs_uid not in owners
    )
    duplicated = sorted(
        obs_uid
        for members in normalized.values()
        for obs_uid in members
        if len(owners.get(obs_uid, ())) > 1
    )
    owner_sets = [set(value) for value in group_owners.values()]
    if desired == "SAME_OWNER":
        union = set().union(*owner_sets) if owner_sets else set()
        relation_pass = len(union) == 1
        atomic_groups = all(len(value) == 1 for value in owner_sets)
        disjoint_groups = False
    else:
        atomic_groups = all(len(value) == 1 for value in owner_sets)
        disjoint_groups = all(
            owner_sets[left].isdisjoint(owner_sets[right])
            for left in range(len(owner_sets))
            for right in range(left + 1, len(owner_sets))
        )
        relation_pass = atomic_groups and disjoint_groups
    probe_owners = {
        str(obs_uid): sorted(owners.get(str(obs_uid), ())) for obs_uid in probes
    }
    return {
        "desired_owner_relation": desired,
        "correct": bool(relation_pass and not missing and not duplicated),
        "group_observation_counts": {
            name: len(members) for name, members in normalized.items()
        },
        "group_owner_uids": group_owners,
        "group_owner_counts": {
            name: len(values) for name, values in group_owners.items()
        },
        "all_groups_atomic": atomic_groups,
        "group_owners_pairwise_disjoint": disjoint_groups,
        "missing_evidence_observation_uids": missing,
        "duplicate_evidence_observation_uids": duplicated,
        "probe_owner_uids": probe_owners,
    }


def evaluate_collateral(
    native_membership: Mapping[str, Iterable[str]],
    candidate_membership: Mapping[str, Iterable[str]],
    affected_observations: Iterable[str],
) -> dict[str, Any]:
    native_universe = {
        str(item) for members in native_membership.values() for item in members or ()
    }
    affected = set(str(item) for item in affected_observations)
    outside = native_universe - affected
    native_signature = _partition_signature(native_membership, outside)
    candidate_signature = _partition_signature(candidate_membership, outside)
    native_map = _owner_partition_map(native_membership, outside)
    candidate_map = _owner_partition_map(candidate_membership, outside)
    changed_outside = sorted(
        obs_uid
        for obs_uid in outside
        if native_map.get(obs_uid) != candidate_map.get(obs_uid)
    )
    candidate_owners = _owner_index(candidate_membership)
    missing = sorted(native_universe - set(candidate_owners))
    extra = sorted(set(candidate_owners) - native_universe)
    duplicated = sorted(
        obs_uid for obs_uid, values in candidate_owners.items() if len(values) > 1
    )
    cross_boundary = []
    for entity_uid, members in candidate_membership.items():
        values = set(str(item) for item in members or ())
        if values & affected and values & outside:
            cross_boundary.append(str(entity_uid))
    return {
        "safe": bool(
            native_signature == candidate_signature
            and not missing
            and not extra
            and not duplicated
            and not cross_boundary
        ),
        "affected_observation_count": len(affected),
        "outside_observation_count": len(outside),
        "outside_partition_exact_to_native": native_signature == candidate_signature,
        "changed_outside_observation_count": len(changed_outside),
        "changed_outside_observation_uids": changed_outside,
        "missing_native_observation_uids": missing,
        "extra_observation_uids": extra,
        "duplicate_observation_uids": duplicated,
        "cross_boundary_entity_uids": sorted(cross_boundary),
    }


def _resolve_groups(
    provenance: ProvenanceIndex, specs: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for spec in specs:
        name = str(spec.get("name") or "")
        kind = str(spec.get("source") or "")
        if not name or name in result:
            raise ValueError("evaluation groups need unique non-empty names")
        if kind == "MAPPING_EVENT_FIELD":
            event = provenance.get_event(str(spec["event_uid"]))
            values = event.get(str(spec["field"])) or ()
        elif kind == "FINAL_ENTITY_MEMBERS":
            values = provenance.final_by_object[str(spec["object_uid"])].get(
                "member_observation_uids"
            ) or ()
        else:
            raise ValueError(f"unsupported evaluation group source: {kind}")
        members = tuple(sorted(set(str(item) for item in values)))
        if not members:
            raise ValueError(f"evaluation group {name} is empty")
        result[name] = members
    return result


def _affected_native_observations(
    provenance: ProvenanceIndex, case: Mapping[str, Any]
) -> set[str]:
    result = set()
    for entity_uid in (case.get("evaluation") or {}).get(
        "affected_native_entity_uids"
    ) or ():
        row = provenance.final_by_object.get(str(entity_uid))
        if row is None:
            raise ValueError(f"unknown affected native entity: {entity_uid}")
        result.update(str(item) for item in row.get("member_observation_uids") or ())
    if not result:
        raise ValueError(f"case {case['case_uid']} has no affected native observations")
    return result


def _threshold_trace(association: Mapping[str, Any]) -> dict[str, Any]:
    top1 = association.get("top1_score")
    threshold = association.get("sim_threshold")
    return {
        "comparator": "STRICT_GREATER_THAN",
        "top1_score": float(top1) if top1 is not None else None,
        "sim_threshold": float(threshold) if threshold is not None else None,
        "top1_minus_threshold": (
            float(top1) - float(threshold)
            if top1 is not None and threshold is not None
            else None
        ),
        "recorded_decision": association.get("decision"),
        "equality_semantics": "CREATE_OBJECT",
    }


def _validate_causal_case(
    provenance: ProvenanceIndex, case: Mapping[str, Any]
) -> dict[str, Any]:
    evidence_refs = []
    for obs_uid in case.get("root_observation_uids") or ():
        if str(obs_uid) not in provenance.observations:
            raise ValueError(f"unknown root observation: {obs_uid}")
        evidence_refs.append({"kind": "OBSERVATION", "uid": str(obs_uid)})
    for event_uid in case.get("root_event_uids") or ():
        event = provenance.get_event(str(event_uid))
        evidence_refs.append(
            {
                "kind": "EVENT",
                "uid": str(event_uid),
                "event_type": event.get("event_type", "ASSOCIATION"),
                "event_sequence": provenance.sequence(event),
            }
        )
    result = {
        "pass": True,
        "causal_disposition": case["causal_disposition"],
        "earliest_causal_stage": case.get("earliest_causal_stage"),
        "root_evidence_refs": evidence_refs,
    }
    if case["causal_disposition"] != REPLAYABLE:
        result["defer_reason"] = case["defer_reason"]
        return result

    obs_uid = str(case["anchor_obs_uid"])
    association = provenance.get_association_for_obs(obs_uid)
    if association["event_uid"] != case["anchor_association_event_uid"]:
        raise ValueError(f"anchor association mismatch for {case['case_uid']}")
    if int(str(association["frame_uid"]).rsplit("_f", 1)[-1]) != int(
        case["frame_idx"]
    ):
        raise ValueError(f"anchor frame mismatch for {case['case_uid']}")
    primitive = SparseRepairConstraint.from_mapping(case["constraints"][0])
    if primitive.obs_uid != obs_uid or primitive.applies_at_event_uid != association["event_uid"]:
        raise ValueError(f"constraint scope mismatch for {case['case_uid']}")
    if primitive.constraint_type.value == "CREATE_INSTANCE":
        created = provenance.get_object_version(
            str(association["target_object_version_after"])
        )
        if (
            primitive.created_lineage_uid != created.get("lineage_uid")
            or primitive.created_entity_uid != created.get("object_uid")
        ):
            raise ValueError(f"created identity mismatch for {case['case_uid']}")
    if primitive.constraint_type.value == "ASSIGN_OBSERVATION":
        seeds = [
            provenance.get_object_version(str(uid))
            for uid in case.get("snapshot_seed_version_uids") or ()
        ]
        matching = [
            row
            for row in seeds
            if primitive.target_lineage_uid == row.get("lineage_uid")
            and primitive.target_entity_uid == row.get("object_uid")
            and primitive.target_origin_obs_uid == row.get("origin_observation_uid")
        ]
        if len(matching) != 1:
            raise ValueError(f"assignment target mismatch for {case['case_uid']}")
    result.update(
        {
            "anchor_obs_uid": obs_uid,
            "anchor_association_event_uid": association["event_uid"],
            "anchor_event_sequence": provenance.sequence(association),
            "recorded_anchor_decision": association.get("decision"),
            "threshold_semantics": _threshold_trace(association),
            "constraint": primitive.as_dict(),
        }
    )
    return result


def _affected_geometry(
    state: Mapping[str, Any], observations: Iterable[str]
) -> list[dict[str, Any]]:
    affected = set(str(item) for item in observations)
    rows = []
    for obj in state.get("objects") or ():
        members = set(str(item) for item in obj.get("member_observation_uids") or ())
        if not members & affected:
            continue
        rows.append(
            {
                "entity_uid": str(obj.get("entity_uid")),
                "affected_member_count": len(members & affected),
                "total_member_count": len(members),
                "n_points": int(obj.get("n_points", 0)),
                "bbox_center": obj.get("bbox_center"),
                "bbox_extent": obj.get("bbox_extent"),
                "class_histogram": obj.get("class_histogram") or {},
            }
        )
    return sorted(rows, key=lambda item: item["entity_uid"])


def _relation_signature_outside(
    state: Mapping[str, Any], affected_observations: Iterable[str]
) -> tuple[tuple[Any, ...], ...]:
    affected = set(str(item) for item in affected_observations)
    partitions = {
        str(entity_uid): tuple(sorted(str(item) for item in members or ()))
        for entity_uid, members in (state.get("membership") or {}).items()
    }
    signature = []
    for edge in state.get("edges") or ():
        source = partitions.get(str(edge.get("source_entity_uid")), ())
        target = partitions.get(str(edge.get("target_entity_uid")), ())
        if set(source) & affected or set(target) & affected:
            continue
        signature.append(
            (
                source,
                str(edge.get("relation")),
                target,
                int(edge.get("num_detections", 0)),
            )
        )
    return tuple(sorted(signature))


@dataclass
class HumanSceneContext:
    scene_id: str
    provenance: ProvenanceIndex
    engine: SparseCounterfactualReplayEngine
    native_state: dict[str, Any]
    dependency_graph: TypedDependencyGraph
    prefix_cache: IncrementalPrefixCache
    edge_stream_root: str | Path | None = None
    relation_frame_records: list[dict[str, Any]] | None = None
    relation_input_cache_wall_ms: float = 0.0
    source_hashes_before: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        scene_id: str,
        base_run: str | Path,
        *,
        edge_stream_root: str | Path | None = None,
    ) -> "HumanSceneContext":
        provenance = ProvenanceIndex(base_run)
        engine = SparseCounterfactualReplayEngine(provenance)
        native = CounterfactualReplayEngine(provenance).clean_state()
        native["branch"] = "NATIVE_FROZEN_ENDPOINT"
        return cls(
            scene_id=str(scene_id),
            provenance=provenance,
            engine=engine,
            native_state=native,
            dependency_graph=TypedDependencyGraph(provenance),
            prefix_cache=IncrementalPrefixCache(engine),
            edge_stream_root=edge_stream_root,
            source_hashes_before=provenance.source_hashes(),
        )

    def ensure_relation_input(self) -> list[dict[str, Any]]:
        if self.edge_stream_root is None:
            raise ValueError("relation input was not requested")
        if self.relation_frame_records is None:
            started = time.perf_counter()
            _, records = load_baseline_frame_records(
                self.provenance,
                self.native_state["membership"],
                edge_stream_root=self.edge_stream_root,
            )
            self.relation_frame_records = records
            self.relation_input_cache_wall_ms = (
                time.perf_counter() - started
            ) * 1000.0
        return self.relation_frame_records


def _attach_relations_cached(
    context: HumanSceneContext, states: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    if context.edge_stream_root is None:
        return {"status": "NOT_REQUESTED", "branches": {}}
    records = context.ensure_relation_input()
    branches = {}
    total_started = time.perf_counter()
    for name, state in states.items():
        started = time.perf_counter()
        objects, remapped = remap_frame_records(records, state["membership"])
        backend = AliDevBaselineRelationBackend()
        rebuilt = backend.rebuild(objects=objects, frame_records=remapped)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        rebuilt["runtime_ms"] = runtime_ms
        state["edges"] = rebuilt["output_edges"]
        state.setdefault("timing", {})["relation_rebuild_wall_ms"] = runtime_ms
        branches[name] = rebuilt
    return {
        "status": "PASS"
        if all(item["validation"]["pass"] for item in branches.values())
        else "FAIL",
        "edge_stream_root": str(Path(context.edge_stream_root).resolve()),
        "shared_relation_input_cache_wall_ms": context.relation_input_cache_wall_ms,
        "total_branch_rebuild_wall_ms": (time.perf_counter() - total_started) * 1000.0,
        "branches": branches,
    }


def _mechanism_trace(
    state: Mapping[str, Any], primitive: SparseRepairConstraint
) -> dict[str, Any]:
    decisions = [
        item
        for item in state.get("decision_trace") or ()
        if item.get("obs_uid") == primitive.obs_uid
    ]
    actions = [str((item.get("constraint") or {}).get("action")) for item in decisions]
    merge_veto_count = int(
        state.get("persistent_create_instance_merge_veto_count", 0)
    )
    association_veto_count = int(
        state.get("persistent_create_instance_association_veto_count", 0)
    )
    redirect_count = int(state.get("persistent_lineage_redirect_count", 0))
    redirect_override_count = int(
        state.get("persistent_lineage_redirect_override_count", 0)
    )
    redirect_decisions = [
        {
            "obs_uid": item.get("obs_uid"),
            "natural_match": item.get("natural_match"),
            "historical_default_match": item.get("historical_default_match"),
            "applied_match": item.get("applied_match"),
            "source_lineages": item.get(
                "persistent_lineage_redirect_source_lineages"
            ),
        }
        for item in state.get("decision_trace") or ()
        if (item.get("constraint") or {}).get("reason")
        == "persistent_lineage_redirect"
    ]
    association_vetoes = [
        {
            "obs_uid": item.get("obs_uid"),
            **(item.get("persistent_create_instance_boundary") or {}),
        }
        for item in state.get("decision_trace") or ()
        if (item.get("persistent_create_instance_boundary") or {}).get(
            "overrode_match"
        )
    ]
    if primitive.constraint_type.value == "CREATE_INSTANCE":
        verified = (
            "FORCE_CREATE" in actions
            and merge_veto_count + association_veto_count > 0
        )
    else:
        verified = (
            "FORCE_TARGET" in actions and redirect_override_count > 0
        )
    return {
        "verified": verified,
        "constraint_type": primitive.constraint_type.value,
        "anchor_constraint_actions": actions,
        "persistent_create_instance_merge_veto_count": merge_veto_count,
        "persistent_create_instance_association_veto_count": association_veto_count,
        "persistent_lineage_redirect_count": redirect_count,
        "persistent_lineage_redirect_override_count": redirect_override_count,
        "persistent_lineage_redirects": redirect_decisions,
        "persistent_association_boundary_vetoes": association_vetoes,
        "postprocess_boundary_vetoes": [
            item
            for item in state.get("postprocess_decision_trace") or ()
            if "persistent_create_instance_boundary"
            in (item.get("reject_reasons") or ())
        ],
    }


def _case_timing_summary(states: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, state in states.items():
        timing = state.get("timing") or {}
        snapshot = timing.get("snapshot") or {}
        result[name] = {
            "snapshot_amortized_wall_ms": snapshot.get("snapshot_amortized_wall_ms"),
            "snapshot_cold_upper_bound_wall_ms": snapshot.get(
                "snapshot_cold_upper_bound_wall_ms"
            ),
            "suffix_total_wall_ms": timing.get("suffix_total_wall_ms"),
            "suffix_execute_wall_total_ms": timing.get(
                "suffix_execute_wall_total_ms"
            ),
            "suffix_overlay_wall_total_ms": timing.get("suffix_overlay_wall_total_ms"),
            "suffix_orchestration_wall_ms": timing.get(
                "suffix_orchestration_wall_ms"
            ),
            "suffix_replay_attempt_count": timing.get("suffix_replay_attempt_count"),
            "relation_rebuild_wall_ms": timing.get("relation_rebuild_wall_ms"),
            "invariant_verification_wall_ms": timing.get(
                "invariant_verification_wall_ms"
            ),
            "endpoint_evaluation_wall_ms": timing.get(
                "endpoint_evaluation_wall_ms"
            ),
        }
    return result


def _run_replayable_case(
    context: HumanSceneContext,
    case: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    provenance = context.provenance
    causal = _validate_causal_case(provenance, case)
    primitive = SparseRepairConstraint.from_mapping(case["constraints"][0])
    groups = _resolve_groups(provenance, (case.get("evaluation") or {})["groups"])
    affected = _affected_native_observations(provenance, case)
    seeds = [str(item) for item in case.get("snapshot_seed_version_uids") or ()]
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=str(case["anchor_association_event_uid"]),
        seed_version_uids=seeds,
    )
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(
        int(case["frame_idx"])
    )
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        str(case["anchor_association_event_uid"]),
        seeds,
        strict=True,
        prefix_state=prefix_state,
        prefix_objects=prefix_objects,
    )
    native = copy.deepcopy(context.native_state)
    natural = context.engine.replay_suffix_from_snapshot(
        mode=ReplayMode.NATURAL_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        current_state=native,
    )
    natural["branch"] = "NATURAL_CAUSAL_SUFFIX_REPLAY"
    sparse = context.engine.replay_local_from_snapshot(
        mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        snapshot_objects=snapshot.objects,
        snapshot_runtime_ms=snapshot.state["runtime_ms"],
        snapshot_timing=snapshot.state.get("timing"),
        anchor_frame=snapshot.anchor_frame,
        snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
        closure=closure,
        constraints=[primitive],
        current_state=native,
    )
    sparse["branch"] = "SPARSE_CAUSAL_REPAIR"
    states = {"native": native, "natural": natural, "sparse": sparse}
    relation = _attach_relations_cached(context, states)

    branch_metrics = {}
    verifier = InvariantVerifier()
    probes = [str(item) for item in (case.get("evaluation") or {}).get("probe_obs_uids") or ()]
    for name, state in states.items():
        evaluation_started = time.perf_counter()
        endpoint = evaluate_endpoint_groups(
            state["membership"],
            groups,
            str((case.get("evaluation") or {})["desired_owner_relation"]),
            probes=probes,
        )
        collateral = evaluate_collateral(native["membership"], state["membership"], affected)
        evaluation_ms = (time.perf_counter() - evaluation_started) * 1000.0
        state.setdefault("timing", {})["endpoint_evaluation_wall_ms"] = evaluation_ms
        verify_started = time.perf_counter()
        verification = verifier.verify(
            state=state,
            constraints=[primitive] if name == "sparse" else (),
            source_hashes_before=context.source_hashes_before,
            source_hashes_after=provenance.source_hashes(),
            known_observation_uids=provenance.observations,
        )
        verify_ms = (time.perf_counter() - verify_started) * 1000.0
        verification["runtime_ms"] = verify_ms
        state.setdefault("timing", {})["invariant_verification_wall_ms"] = verify_ms
        branch_metrics[name] = {
            "endpoint": endpoint,
            "collateral": collateral,
            "runtime_invariants": verification,
            "active_object_count": len(state.get("membership") or {}),
            "affected_owner_geometry": _affected_geometry(state, affected),
            "relation": (
                relation.get("branches", {}).get(name)
                if relation.get("status") != "NOT_REQUESTED"
                else {"status": "NOT_REQUESTED"}
            ),
        }

    native_relation_signature = _relation_signature_outside(native, affected)
    relation_outside = {
        name: _relation_signature_outside(state, affected)
        == native_relation_signature
        for name, state in states.items()
    }
    mechanism = _mechanism_trace(sparse, primitive)
    relation_pass = relation.get("status") in {"PASS", "NOT_REQUESTED"}
    source_hashes_unchanged = context.source_hashes_before == provenance.source_hashes()
    contrast_pass = bool(
        not branch_metrics["native"]["endpoint"]["correct"]
        and not branch_metrics["natural"]["endpoint"]["correct"]
        and branch_metrics["sparse"]["endpoint"]["correct"]
        and branch_metrics["sparse"]["collateral"]["safe"]
        and branch_metrics["sparse"]["runtime_invariants"]["pass"]
        and mechanism["verified"]
        and snapshot.validation["pass"]
        and relation_pass
        and relation_outside["sparse"]
        and source_hashes_unchanged
    )
    metrics = {
        "schema_version": "1.0.0",
        "case_uid": case["case_uid"],
        "incident_uid": case["incident_uid"],
        "scene_id": case["scene_id"],
        "endpoint_error_type": case["endpoint_error_type"],
        "causal_disposition": REPLAYABLE,
        "status": "PASS" if contrast_pass else "FAIL",
        "contrast_pass": contrast_pass,
        "contrast_definition": (
            "NATIVE_WRONG_AND_NATURAL_WRONG_AND_SPARSE_CORRECT_WITH_"
            "COLLATERAL_RUNTIME_RELATION_AND_SOURCE_SAFETY"
        ),
        "causal_validation": causal,
        "evaluation_groups": {name: list(values) for name, values in groups.items()},
        "affected_native_observation_count": len(affected),
        "dependency": closure.as_dict(),
        "snapshot_validation": snapshot.validation,
        "mechanism_trace": mechanism,
        "branches": branch_metrics,
        "natural_endpoint_error_reproduced": not branch_metrics["natural"][
            "endpoint"
        ]["correct"],
        "sparse_endpoint_corrected": branch_metrics["sparse"]["endpoint"]["correct"],
        "outside_relation_exact_to_native": relation_outside,
        "relation_rebuild": relation,
        "source_hashes_unchanged": source_hashes_unchanged,
        "timing": _case_timing_summary(states),
        "case_total_wall_ms": (time.perf_counter() - case_started) * 1000.0,
    }
    case_root = output_root / str(case["case_uid"])
    _write_json(case_root / "case_manifest.json", case)
    _write_json(case_root / "causal_validation.json", causal)
    _write_json(case_root / "constraint.json", primitive.as_dict())
    _write_json(case_root / "dependency.json", closure.as_dict())
    _write_json(case_root / "pre_anchor_snapshot.json", snapshot.as_dict())
    _write_json(case_root / "relation_rebuild.json", relation)
    for name, state in states.items():
        _write_json(case_root / "branches" / f"{name}.json", state)
    _write_json(case_root / "metrics.json", metrics)
    return metrics


def _run_deferred_case(
    context: HumanSceneContext,
    case: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    causal = _validate_causal_case(context.provenance, case)
    metrics = {
        "schema_version": "1.0.0",
        "case_uid": case["case_uid"],
        "incident_uid": case["incident_uid"],
        "scene_id": case["scene_id"],
        "endpoint_error_type": case["endpoint_error_type"],
        "causal_disposition": DEFERRED,
        "status": "DEFERRED",
        "replay_attempted": False,
        "repair_claimed": False,
        "causal_validation": causal,
        "methodological_reason": (
            "No executable association target was supported by the frozen evidence; "
            "forcing one would turn the evaluation into an oracle or a category error."
        ),
    }
    case_root = output_root / str(case["case_uid"])
    _write_json(case_root / "case_manifest.json", case)
    _write_json(case_root / "causal_validation.json", causal)
    _write_json(case_root / "metrics.json", metrics)
    return metrics


def _numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    materialized = [float(item) for item in values if item is not None]
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized) if materialized else None,
        "median": statistics.median(materialized) if materialized else None,
        "min": min(materialized) if materialized else None,
        "max": max(materialized) if materialized else None,
    }


def _aggregate_timing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for branch in ("natural", "sparse"):
        branch_rows = [
            (row.get("timing") or {}).get(branch) or {}
            for row in rows
            if row.get("causal_disposition") == REPLAYABLE
        ]
        result[branch] = {
            field: _numeric_summary(item.get(field) for item in branch_rows)
            for field in (
                "snapshot_amortized_wall_ms",
                "snapshot_cold_upper_bound_wall_ms",
                "suffix_total_wall_ms",
                "suffix_execute_wall_total_ms",
                "suffix_overlay_wall_total_ms",
                "suffix_orchestration_wall_ms",
                "relation_rebuild_wall_ms",
                "invariant_verification_wall_ms",
                "endpoint_evaluation_wall_ms",
            )
        }
    return result


def run_human_error_pilot(
    *,
    manifest_path: str | Path,
    scene_base_runs: Mapping[str, str | Path],
    output_root: str | Path,
    source_paths: Mapping[str, str | Path],
    edge_stream_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    manifest = load_manifest(manifest_path)
    source_verification = verify_frozen_sources(manifest, source_paths)
    queue = _read_jsonl(source_paths["expert_queue"])
    r1 = _read_jsonl(source_paths["r1_labels"])
    r2 = _read_jsonl(source_paths["r2_labels"])
    cohort_validation = validate_manifest(
        manifest,
        expert_queue=queue,
        r1_labels=r1,
        r2_labels=r2,
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "frozen_manifest.json", manifest)
    _write_json(output / "source_verification.json", source_verification)
    _write_json(output / "cohort_validation.json", cohort_validation)

    edge_stream_roots = edge_stream_roots or {}
    needed_scenes = sorted(set(str(item["scene_id"]) for item in manifest["cases"]))
    missing_scenes = sorted(set(needed_scenes) - set(scene_base_runs))
    if missing_scenes:
        raise ValueError(f"missing base runs for scenes: {missing_scenes}")
    contexts = {
        scene_id: HumanSceneContext.build(
            scene_id,
            scene_base_runs[scene_id],
            edge_stream_root=edge_stream_roots.get(scene_id),
        )
        for scene_id in needed_scenes
    }

    ordered = sorted(
        manifest["cases"],
        key=lambda item: (
            str(item["scene_id"]),
            0 if item["causal_disposition"] == REPLAYABLE else 1,
            int(item.get("frame_idx", 10**9)),
            str(item["case_uid"]),
        ),
    )
    rows = []
    for case in ordered:
        context = contexts[str(case["scene_id"])]
        if case["causal_disposition"] == REPLAYABLE:
            row = _run_replayable_case(context, case, output)
        else:
            row = _run_deferred_case(context, case, output)
        rows.append(row)

    replayed = [row for row in rows if row["causal_disposition"] == REPLAYABLE]
    deferred = [row for row in rows if row["causal_disposition"] == DEFERRED]
    contrast_passes = sum(bool(row.get("contrast_pass")) for row in replayed)
    sparse_correct = sum(bool(row.get("sparse_endpoint_corrected")) for row in replayed)
    natural_wrong = sum(bool(row.get("natural_endpoint_error_reproduced")) for row in replayed)
    all_sources_unchanged = all(
        context.source_hashes_before == context.provenance.source_hashes()
        for context in contexts.values()
    )
    aggregate = {
        "schema_version": "1.0.0",
        "pilot_uid": manifest.get("pilot_uid"),
        "status": "PASS" if contrast_passes == len(replayed) else "FAIL",
        "pilot_pass": contrast_passes == len(replayed),
        "cohort": cohort_validation,
        "human_confirmed_case_count": len(rows),
        "replayable_case_count": len(replayed),
        "deferred_non_association_root_count": len(deferred),
        "native_wrong_count": sum(
            not row["branches"]["native"]["endpoint"]["correct"] for row in replayed
        ),
        "natural_replay_still_wrong_count": natural_wrong,
        "sparse_causal_repair_correct_count": sparse_correct,
        "strict_contrast_pass_count": contrast_passes,
        "conditional_replayable_repair_rate": (
            contrast_passes / len(replayed) if replayed else None
        ),
        "overall_confirmed_error_repair_yield": contrast_passes / len(rows),
        "deferral_rate": len(deferred) / len(rows),
        "denominator_policy": {
            "conditional_replayable_repair_rate": "strict contrast passes / 3 replayable causes",
            "overall_confirmed_error_repair_yield": "strict contrast passes / all 6 selected confirmed errors",
            "deferred_cases_are_not_counted_as_successes": True,
        },
        "by_endpoint_error_type": {
            error_type: {
                "selected": sum(row["endpoint_error_type"] == error_type for row in rows),
                "replayable": sum(
                    row["endpoint_error_type"] == error_type
                    and row["causal_disposition"] == REPLAYABLE
                    for row in rows
                ),
                "strict_contrast_pass": sum(
                    row["endpoint_error_type"] == error_type
                    and bool(row.get("contrast_pass"))
                    for row in rows
                ),
            }
            for error_type in ("FALSE_MERGE", "FALSE_SPLIT")
        },
        "timing": _aggregate_timing(rows),
        "scene_relation_input_cache_wall_ms": {
            scene_id: context.relation_input_cache_wall_ms
            for scene_id, context in contexts.items()
        },
        "source_hashes_unchanged": all_sources_unchanged,
        "selection_and_claim_limits": {
            "selection": "PURPOSIVE_CAUSAL_PILOT_FROM_40_HUMAN_CONFIRMED_ERRORS",
            "scenes": needed_scenes,
            "generalizes_to_all_40": False,
            "relation_correctness_claimed_without_human_relation_labels": False,
            "geometry_correctness_claimed_without_corrected_geometry_gold": False,
        },
        "cases": [
            {
                "case_uid": row["case_uid"],
                "incident_uid": row["incident_uid"],
                "scene_id": row["scene_id"],
                "endpoint_error_type": row["endpoint_error_type"],
                "causal_disposition": row["causal_disposition"],
                "status": row["status"],
                "contrast_pass": row.get("contrast_pass"),
            }
            for row in rows
        ],
        "run_total_wall_ms": (time.perf_counter() - run_started) * 1000.0,
    }
    _write_json(output / "aggregate_metrics.json", aggregate)
    return aggregate
