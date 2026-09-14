from __future__ import annotations

import json
from collections import deque
from typing import Any, Iterable, Mapping

from .index import LineageIndex, ProvenanceIndex
from .models import DependencyClosure


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if isinstance(item, str)}
    return set()


class CausalTracer:
    """Trace one controlled intervention through the immutable version DAG."""

    def __init__(self, provenance: ProvenanceIndex) -> None:
        self.provenance = provenance
        self.lineage = LineageIndex(provenance)

    def trace(self, case: Mapping[str, Any]) -> dict[str, Any]:
        anchor_uid = str(case["anchor_association_event_uid"])
        anchor = self.provenance.get_event(anchor_uid)
        if anchor.get("obs_uid") != case.get("obs_uid"):
            raise ValueError("case anchor does not reference the injected observation")
        mapping_uid = str(anchor.get("mapping_event_uid"))
        if mapping_uid not in self.provenance.mapping_events:
            raise ValueError("anchor association has no resolvable mapping event")
        mapping_event = self.provenance.get_event(mapping_uid)
        anchor_sequence = self.provenance.sequence(anchor)

        affected_groups = case.get("affected_clean_groups") or {}
        affected_obs = {
            str(item)
            for members in affected_groups.values()
            for item in (members or ())
        }
        affected_obs.add(str(case["obs_uid"]))
        affected_entities = {str(item) for item in affected_groups}
        for key in (
            "source_identity_uid",
            "target_identity_uid",
            "clean_target_object_uid",
            "target_object_uid",
        ):
            if case.get(key):
                affected_entities.add(str(case[key]))

        affected_versions: set[str] = set()
        for version in self.provenance.object_version_rows:
            members = set(version.get("member_observation_uids") or ())
            if members & affected_obs or str(version.get("object_uid")) in affected_entities:
                affected_versions.add(str(version["object_version_uid"]))

        queue = deque(affected_versions)
        while queue:
            version_uid = queue.popleft()
            for child in self.provenance.get_child_versions(version_uid):
                if child not in affected_versions:
                    affected_versions.add(child)
                    queue.append(child)
            row = self.provenance.get_object_version(version_uid)
            affected_entities.add(str(row["object_uid"]))
            affected_obs.update(str(item) for item in row.get("member_observation_uids") or ())

        event_uids: set[str] = {anchor_uid, mapping_uid}
        changed = True
        while changed:
            changed = False
            for event in sorted(self.provenance.events.values(), key=self.provenance.sequence):
                sequence = self.provenance.sequence(event)
                if sequence < anchor_sequence:
                    continue
                event_versions = _strings(event.get("input_object_version_uids")) | _strings(
                    event.get("output_object_version_uids")
                )
                event_obs = _strings(event.get("obs_uid")) | _strings(
                    event.get("source_observation_uids")
                )
                encoded = json.dumps(event, sort_keys=True)
                touches_entity = any(entity in encoded for entity in affected_entities)
                if not (event_versions & affected_versions or event_obs & affected_obs or touches_entity):
                    continue
                before = (len(event_uids), len(affected_versions), len(affected_obs))
                event_uids.add(str(event["event_uid"]))
                affected_versions.update(event_versions)
                affected_obs.update(event_obs)
                for version_uid in event_versions:
                    if version_uid in self.provenance.object_versions:
                        version = self.provenance.get_object_version(version_uid)
                        affected_entities.add(str(version["object_uid"]))
                        affected_obs.update(
                            str(item) for item in version.get("member_observation_uids") or ()
                        )
                changed = changed or before != (
                    len(event_uids),
                    len(affected_versions),
                    len(affected_obs),
                )

        incident_edges = set()
        for entity_uid in affected_entities:
            for event in self.provenance.get_incident_edge_events(entity_uid):
                event_uids.add(str(event["event_uid"]))
                edge_uid = event.get("edge_uid")
                if edge_uid:
                    incident_edges.add(str(edge_uid))

        closure = DependencyClosure.build(
            event_uids=event_uids,
            version_uids=affected_versions,
            entity_uids=affected_entities,
            obs_uids=affected_obs,
            edge_uids=incident_edges,
            start_sequence=anchor_sequence,
            end_sequence=max(
                (self.provenance.sequence(self.provenance.get_event(uid)) for uid in event_uids),
                default=anchor_sequence,
            ),
        )
        trace = [
            {
                "event_uid": anchor_uid,
                "event_type": "ASSOCIATION_DECISION",
                "event_sequence": anchor_sequence,
                "obs_uid": anchor.get("obs_uid"),
                "decision": anchor.get("decision"),
            },
            {
                "event_uid": mapping_uid,
                "event_type": mapping_event.get("event_type"),
                "event_sequence": self.provenance.sequence(mapping_event),
                "output_object_version_uids": mapping_event.get("output_object_version_uids") or [],
            },
        ]
        for version_uid in mapping_event.get("output_object_version_uids") or ():
            if version_uid not in self.provenance.object_versions:
                continue
            trace.append(
                {
                    "object_version_uid": version_uid,
                    "descendant_version_uids": list(self.lineage.resolve_descendants(version_uid)),
                }
            )
        return {
            "case_uid": str(case["case_uid"]),
            "causal_anchor_event_uid": anchor_uid,
            "anchor_frame_uid": anchor.get("frame_uid"),
            "injected_observation_uid": case.get("obs_uid"),
            "injection_recovered": anchor.get("obs_uid") == case.get("obs_uid"),
            "affected_versions": list(closure.version_uids),
            "affected_observations": list(closure.obs_uids),
            "trace": trace,
            "dependency_closure": closure.as_dict(),
        }
