from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .index import ProvenanceIndex
from .models import DependencyClosure


@dataclass(frozen=True, order=True)
class DependencyNode:
    kind: str
    uid: str


class TypedDependencyGraph:
    """Explicit evidence dependency index with no serialized-string matching."""

    def __init__(self, provenance: ProvenanceIndex) -> None:
        self.provenance = provenance
        self.forward: dict[DependencyNode, set[DependencyNode]] = defaultdict(set)
        self.reverse: dict[DependencyNode, set[DependencyNode]] = defaultdict(set)
        self.read_neighbors: dict[DependencyNode, set[DependencyNode]] = defaultdict(set)
        self.node_sequence: dict[DependencyNode, int] = {}
        self.version_entities: dict[str, tuple[str, str]] = {}
        self.versions_by_lineage: dict[str, set[str]] = defaultdict(set)
        self._build()

    @staticmethod
    def node(kind: str, uid: Any) -> DependencyNode:
        return DependencyNode(str(kind).upper(), str(uid))

    def _edge(self, source: DependencyNode, target: DependencyNode) -> None:
        self.forward[source].add(target)
        self.reverse[target].add(source)

    @staticmethod
    def _values(row: Mapping[str, Any], *fields: str) -> tuple[str, ...]:
        result: list[str] = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and value:
                result.append(value)
            elif isinstance(value, (list, tuple, set)):
                result.extend(str(item) for item in value if isinstance(item, str) and item)
        return tuple(result)

    def _build(self) -> None:
        for version in self.provenance.object_version_rows:
            version_uid = str(version["object_version_uid"])
            version_node = self.node("OBJECT_VERSION", version_uid)
            object_uid = str(version["object_uid"])
            lineage_uid = str(version.get("lineage_uid") or object_uid)
            self.version_entities[version_uid] = (object_uid, lineage_uid)
            self.versions_by_lineage[lineage_uid].add(version_uid)
            entity_node = self.node("ENTITY", object_uid)
            lineage_node = self.node("ENTITY_LINEAGE", lineage_uid)
            self._edge(version_node, entity_node)
            trigger_uid = version.get("trigger_event_uid")
            if isinstance(trigger_uid, str) and trigger_uid in self.provenance.events:
                self.node_sequence[version_node] = self.provenance.sequence(
                    self.provenance.get_event(trigger_uid)
                )
            for parent_uid in version.get("parent_version_uids") or ():
                self._edge(
                    self.node("OBJECT_VERSION", parent_uid),
                    version_node,
                )
            for obs_uid in version.get("member_observation_uids") or ():
                self._edge(version_node, self.node("OBS", obs_uid))

        for association in self.provenance.association_rows:
            event_uid = str(association["event_uid"])
            association_node = self.node("ASSOCIATION_EVENT", event_uid)
            sequence = self.provenance.sequence(association)
            self.node_sequence[association_node] = sequence
            obs_node = self.node("OBS", association["obs_uid"])
            self._edge(obs_node, association_node)
            self._edge(association_node, obs_node)
            mapping_uid = association.get("mapping_event_uid")
            if isinstance(mapping_uid, str):
                self._edge(association_node, self.node("MAPPING_EVENT", mapping_uid))
            for version_uid in self._values(association, "target_object_version_before"):
                self._edge(self.node("OBJECT_VERSION", version_uid), association_node)
            for version_uid in self._values(association, "candidate_object_version_uids"):
                self.read_neighbors[association_node].add(
                    self.node("OBJECT_VERSION", version_uid)
                )
            for object_uid in self._values(association, "target_object_uid"):
                self._edge(self.node("ENTITY", object_uid), association_node)
            for object_uid in self._values(association, "object_uids_before"):
                self.read_neighbors[association_node].add(self.node("ENTITY", object_uid))

        for event in self.provenance.mapping_event_rows:
            event_uid = str(event["event_uid"])
            mapping_node = self.node("MAPPING_EVENT", event_uid)
            sequence = self.provenance.sequence(event)
            self.node_sequence[mapping_node] = sequence
            association_uid = event.get("association_event_uid")
            if isinstance(association_uid, str):
                association_node = self.node("ASSOCIATION_EVENT", association_uid)
                self._edge(association_node, mapping_node)
                self._edge(mapping_node, association_node)
            obs_uid = event.get("obs_uid")
            if isinstance(obs_uid, str):
                obs_node = self.node("OBS", obs_uid)
                self._edge(obs_node, mapping_node)
                self._edge(mapping_node, obs_node)
            for version_uid in self._values(event, "input_object_version_uids"):
                self._edge(self.node("OBJECT_VERSION", version_uid), mapping_node)
            for version_uid in self._values(event, "output_object_version_uids"):
                self._edge(mapping_node, self.node("OBJECT_VERSION", version_uid))
            for object_uid in self._values(
                event,
                "object_uid",
                "source_object_uid",
                "target_object_uid",
                "source_entity_uid",
                "target_entity_uid",
            ):
                entity_node = self.node("ENTITY", object_uid)
                self._edge(entity_node, mapping_node)
                self._edge(mapping_node, entity_node)
            if str(event.get("event_type", "")).startswith("EDGE_"):
                edge_uid = event.get("edge_uid") or event_uid
                self._edge(mapping_node, self.node("RELATION_EVENT", edge_uid))

    def forward_closure(
        self,
        *,
        anchor_event_uid: str,
        seed_version_uids: Iterable[str] = (),
        seed_lineage_uids: Iterable[str] = (),
        stop_watermark: int | None = None,
    ) -> DependencyClosure:
        anchor = self.provenance.get_event(str(anchor_event_uid))
        anchor_sequence = self.provenance.sequence(anchor)
        stop = self.provenance.max_sequence if stop_watermark is None else int(stop_watermark)
        anchor_kind = (
            "ASSOCIATION_EVENT"
            if str(anchor_event_uid) in self.provenance.associations
            else "MAPPING_EVENT"
        )
        seeds = {
            self.node(anchor_kind, anchor_event_uid),
            *(self.node("OBJECT_VERSION", uid) for uid in seed_version_uids if uid),
        }
        for lineage_uid in seed_lineage_uids:
            seeds.update(
                self.node("OBJECT_VERSION", uid)
                for uid in self.versions_by_lineage.get(str(lineage_uid), ())
            )
        queue = deque(sorted(seeds))
        visited = set(seeds)
        while queue:
            current = queue.popleft()
            for candidate in sorted(self.forward.get(current, ())):
                sequence = self.node_sequence.get(candidate)
                if sequence is not None and not anchor_sequence <= sequence <= stop:
                    continue
                if candidate not in visited:
                    visited.add(candidate)
                    queue.append(candidate)

        causal_visited = set(visited)
        read_only_nodes = {
            neighbor
            for node in causal_visited
            if node.kind == "ASSOCIATION_EVENT"
            for neighbor in self.read_neighbors.get(node, ())
        }
        visited.update(read_only_nodes)

        event_uids: set[str] = set()
        version_uids: set[str] = set()
        entity_uids: set[str] = set()
        lineage_uids: set[str] = set()
        obs_uids: set[str] = set()
        edge_uids: set[str] = set()
        event_sequences = [anchor_sequence]
        for node in visited:
            if node.kind in {"ASSOCIATION_EVENT", "MAPPING_EVENT"}:
                event_uids.add(node.uid)
                if node in self.node_sequence:
                    event_sequences.append(self.node_sequence[node])
            elif node.kind == "OBJECT_VERSION":
                version_uids.add(node.uid)
            elif node.kind == "ENTITY":
                entity_uids.add(node.uid)
            elif node.kind == "ENTITY_LINEAGE":
                lineage_uids.add(node.uid)
            elif node.kind == "OBS":
                obs_uids.add(node.uid)
            elif node.kind == "RELATION_EVENT":
                edge_uids.add(node.uid)

        for version_uid in version_uids:
            entity = self.version_entities.get(version_uid)
            if entity:
                entity_uids.add(entity[0])
                lineage_uids.add(entity[1])
        closure = DependencyClosure.build(
            event_uids=event_uids,
            version_uids=version_uids,
            entity_uids=entity_uids | lineage_uids,
            obs_uids=obs_uids,
            edge_uids=edge_uids,
            start_sequence=anchor_sequence,
            end_sequence=max(event_sequences),
        )
        return closure
