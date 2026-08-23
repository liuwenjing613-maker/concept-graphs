from __future__ import annotations

import gzip
import json
import pickle
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cases import invert_membership
from .index import ProvenanceIndex


class AliDevBaselineRelationBackend:
    """A semantic-preserving wrapper around ali-dev's unchanged process_edges."""

    def __init__(self) -> None:
        self.invalidated_entities: set[str] = set()
        self.input_relation_types: set[str] = set()
        self.used_process_edges = False

    def invalidate(self, changed_entity_uids: Iterable[str]) -> None:
        self.invalidated_entities.update(str(item) for item in changed_entity_uids)

    @staticmethod
    def _object_id(entity_uid: str) -> uuid.UUID:
        try:
            return uuid.UUID(entity_uid)
        except ValueError:
            return uuid.uuid5(uuid.NAMESPACE_URL, "revision-entity:" + entity_uid)

    def rebuild(
        self,
        *,
        objects: list[dict[str, Any]],
        frame_records: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        from conceptgraph.slam.slam_classes import MapEdgeMapping
        from conceptgraph.slam.utils import process_edges

        map_edges = MapEdgeMapping(objects)
        frames = 0
        input_edges = 0
        for record in sorted(frame_records, key=lambda row: int(row["frame_idx"])):
            frames += 1
            gobs = {
                "detection_class_labels": list(record.get("detection_class_labels") or ()),
                "edges": list(record.get("edges") or ()),
            }
            matches = list(record.get("match_indices") or ())
            input_edges += len(gobs["edges"])
            self.input_relation_types.update(str(edge[1]) for edge in gobs["edges"])
            if gobs["edges"]:
                map_edges = process_edges(
                    matches,
                    gobs,
                    len(objects),
                    objects,
                    map_edges,
                    int(record["frame_idx"]),
                )
                self.used_process_edges = True
            stale = []
            for edge in map_edges.edges_by_index.values():
                if int(record["frame_idx"]) - int(edge.first_detected) > 5 and edge.num_detections < 2:
                    stale.append((edge.obj1_idx, edge.obj2_idx))
            for source, target in stale:
                map_edges.delete_edge(source, target)

        edges = []
        for (source, target), edge in sorted(map_edges.edges_by_index.items()):
            edges.append(
                {
                    "source_entity_uid": str(objects[source]["entity_uid"]),
                    "relation": str(edge.rel_type),
                    "target_entity_uid": str(objects[target]["entity_uid"]),
                    "num_detections": int(edge.num_detections),
                }
            )
        validation = self.validate(objects=objects, edges=edges)
        return {
            "strategy": "GLOBAL_BASELINE_EDGE_REPLAY",
            "backend": "ali-dev process_edges (unchanged)",
            "frames_replayed": frames,
            "input_edge_observations": input_edges,
            "output_edges": edges,
            "input_relation_types": sorted(self.input_relation_types),
            "used_process_edges": self.used_process_edges,
            "informative": input_edges > 0,
            "validation": validation,
        }

    def validate(
        self, *, objects: Iterable[Mapping[str, Any]], edges: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        active = {str(item["entity_uid"]) for item in objects}
        dangling = 0
        self_loops = 0
        novel_types = set()
        malformed = 0
        edge_count = 0
        for edge in edges:
            edge_count += 1
            source = str(edge.get("source_entity_uid", ""))
            target = str(edge.get("target_entity_uid", ""))
            relation = str(edge.get("relation", ""))
            if not source or not target or not relation:
                malformed += 1
            if source not in active or target not in active:
                dangling += 1
            if source == target:
                self_loops += 1
            if relation not in self.input_relation_types:
                novel_types.add(relation)
        return {
            "pass": dangling == 0 and self_loops == 0 and malformed == 0 and not novel_types,
            "edge_count": edge_count,
            "dangling_edge_count": dangling,
            "unexpected_self_loop_count": self_loops,
            "malformed_relation_count": malformed,
            "novel_relation_types": sorted(novel_types),
        }


def load_baseline_frame_records(
    provenance: ProvenanceIndex,
    membership: Mapping[str, Iterable[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct the filtered gobs interface from immutable cached detections."""
    with (provenance.experiment_root / "config_params.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    suffix = str(cfg["detections_exp_suffix"])
    detection_root = provenance.experiment_root.parent / suffix / "detections"
    if not detection_root.is_dir():
        raise FileNotFoundError(f"cached detection directory not found: {detection_root}")

    canonical_membership = {
        str(entity): tuple(str(obs) for obs in members)
        for entity, members in membership.items()
        if tuple(members)
    }
    entity_order = sorted(canonical_membership)
    entity_index = {entity: index for index, entity in enumerate(entity_order)}
    obs_owner = invert_membership(canonical_membership)
    objects = [
        {
            "entity_uid": entity,
            "id": AliDevBaselineRelationBackend._object_id(entity),
            "curr_obj_num": index,
        }
        for index, entity in enumerate(entity_order)
    ]

    obs_by_frame: dict[str, list[dict[str, Any]]] = {}
    for row in provenance.observation_rows:
        obs_uid = str(row["obs_uid"])
        if row.get("status") != "kept" or obs_uid not in obs_owner:
            continue
        obs_by_frame.setdefault(str(row["frame_uid"]), []).append(row)

    frames_path = provenance.evidence_root / "frames.jsonl"
    frames = [json.loads(line) for line in frames_path.open(encoding="utf-8") if line.strip()]
    records = []
    for frame in frames:
        frame_uid = str(frame["frame_uid"])
        observations = sorted(
            obs_by_frame.get(frame_uid, ()),
            key=lambda row: int(row.get("filtered_det_idx", 0)),
        )
        source_frame_id = str(frame.get("source_frame_id", ""))
        directory = detection_root / source_frame_id
        edges_path = directory / "edges.pkl.gz"
        labels_path = directory / "detection_class_labels.pkl.gz"
        if not edges_path.is_file() or not labels_path.is_file():
            continue
        with gzip.open(edges_path, "rb") as handle:
            edges = pickle.load(handle)
        with gzip.open(labels_path, "rb") as handle:
            raw_labels = pickle.load(handle)
        labels = []
        matches = []
        for row in observations:
            raw_index = int(row["raw_det_idx"])
            if raw_index >= len(raw_labels):
                raise ValueError(f"raw detection index out of range in {frame_uid}")
            labels.append(raw_labels[raw_index])
            matches.append(entity_index[obs_owner[str(row["obs_uid"])]])
        records.append(
            {
                "frame_idx": int(frame_uid.rsplit("_f", 1)[-1]),
                "detection_class_labels": labels,
                "edges": edges,
                "match_indices": matches,
            }
        )
    return objects, records
