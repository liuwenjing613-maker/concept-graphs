from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import pickle
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cases import invert_membership
from .index import ProvenanceIndex


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_edge_stream(edge_stream_root: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and strictly validate a frozen frame-level relation stream."""
    root = Path(edge_stream_root)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if manifest_path.name != "manifest.json":
        raise ValueError("edge stream must be a directory or manifest.json")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "PASS":
        raise ValueError(f"edge stream manifest is not PASS: {manifest_path}")
    frames_path = manifest_path.parent / "frames.jsonl"
    if not frames_path.is_file():
        raise FileNotFoundError(f"edge stream frames not found: {frames_path}")
    expected_hash = manifest.get("frames_sha256")
    actual_hash = _sha256_file(frames_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("edge stream frames hash does not match manifest")

    frames: dict[str, dict[str, Any]] = {}
    input_edges = 0
    with frames_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame_id = str(row.get("source_frame_id", ""))
            if not frame_id or frame_id in frames:
                raise ValueError(
                    f"missing or duplicate edge-stream frame at line {line_number}"
                )
            labels = {
                str(item).split(":", 1)[0].strip()
                for item in row.get("input_labels") or ()
            }
            edges = []
            for edge in row.get("edges") or ():
                if not isinstance(edge, (list, tuple)) or len(edge) != 3:
                    raise ValueError(f"malformed edge in {frame_id}: {edge!r}")
                source, relation, target = (str(item) for item in edge)
                if source not in labels or target not in labels:
                    raise ValueError(
                        f"edge endpoint is not bound to frame labels in {frame_id}"
                    )
                if source == target:
                    raise ValueError(f"self-loop relation observation in {frame_id}")
                if not relation:
                    raise ValueError(f"empty relation type in {frame_id}")
                edges.append([source, relation, target])
            row = dict(row)
            row["edges"] = edges
            frames[frame_id] = row
            input_edges += len(edges)
    if int(manifest.get("frame_count", len(frames))) != len(frames):
        raise ValueError("edge stream frame count does not match manifest")
    if int(manifest.get("input_edge_observations", input_edges)) != input_edges:
        raise ValueError("edge stream observation count does not match manifest")
    return manifest, frames


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

        self.input_relation_types.clear()
        self.used_process_edges = False
        map_edges = MapEdgeMapping(objects)
        frames = 0
        input_edges = 0
        nonempty_frames = 0
        for record in sorted(frame_records, key=lambda row: int(row["frame_idx"])):
            frames += 1
            gobs = {
                "detection_class_labels": list(record.get("detection_class_labels") or ()),
                "edges": list(record.get("edges") or ()),
            }
            matches = list(record.get("match_indices") or ())
            input_edges += len(gobs["edges"])
            nonempty_frames += bool(gobs["edges"])
            self.input_relation_types.update(str(edge[1]) for edge in gobs["edges"])
            if gobs["edges"]:
                with contextlib.redirect_stdout(io.StringIO()):
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
                with contextlib.redirect_stdout(io.StringIO()):
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
            "nonempty_input_frames": nonempty_frames,
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


def remap_frame_records(
    frame_records: Iterable[Mapping[str, Any]],
    membership: Mapping[str, Iterable[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remap immutable frame observations to a branch membership without rereading caches."""
    canonical = {}
    for entity, members in membership.items():
        values = tuple(str(obs) for obs in members)
        if values:
            canonical[str(entity)] = values
    entity_order = sorted(canonical)
    entity_index = {entity: index for index, entity in enumerate(entity_order)}
    obs_owner = invert_membership(canonical)
    objects = [
        {
            "entity_uid": entity,
            "id": AliDevBaselineRelationBackend._object_id(entity),
            "curr_obj_num": index,
        }
        for index, entity in enumerate(entity_order)
    ]
    remapped = []
    for source in frame_records:
        record = dict(source)
        observation_uids = [str(item) for item in record.get("observation_uids") or ()]
        if len(observation_uids) != len(record.get("detection_class_labels") or ()):
            raise ValueError("frame observation/label cardinality mismatch")
        missing = [obs_uid for obs_uid in observation_uids if obs_uid not in obs_owner]
        if missing:
            raise ValueError(f"branch membership is missing observations: {missing[:5]}")
        record["match_indices"] = [
            entity_index[obs_owner[obs_uid]] for obs_uid in observation_uids
        ]
        remapped.append(record)
    return objects, remapped


def load_baseline_frame_records(
    provenance: ProvenanceIndex,
    membership: Mapping[str, Iterable[str]],
    *,
    edge_stream_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct the filtered gobs interface from immutable cached detections."""
    with (provenance.experiment_root / "config_params.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    suffix = str(cfg["detections_exp_suffix"])
    detection_root = provenance.experiment_root.parent / suffix / "detections"
    if not detection_root.is_dir():
        raise FileNotFoundError(f"cached detection directory not found: {detection_root}")

    canonical_membership = {}
    for entity, members in membership.items():
        materialized = tuple(str(obs) for obs in members)
        if materialized:
            canonical_membership[str(entity)] = materialized
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
    edge_manifest = None
    edge_frames = None
    if edge_stream_root is not None:
        edge_manifest, edge_frames = load_edge_stream(edge_stream_root)

    records = []
    consumed_edge_frames = set()
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
        observation_uids = []
        for row in observations:
            raw_index = int(row["raw_det_idx"])
            if raw_index >= len(raw_labels):
                raise ValueError(f"raw detection index out of range in {frame_uid}")
            labels.append(raw_labels[raw_index])
            matches.append(entity_index[obs_owner[str(row["obs_uid"])]])
            observation_uids.append(str(row["obs_uid"]))
        if edge_frames is not None:
            if source_frame_id not in edge_frames:
                raise ValueError(f"edge stream is missing source frame {source_frame_id}")
            edge_row = edge_frames[source_frame_id]
            edges = edge_row["edges"]
            consumed_edge_frames.add(source_frame_id)
            raw_label_ids = {str(item).split(" ")[-1] for item in raw_labels}
            for edge in edges:
                if edge[0] not in raw_label_ids or edge[2] not in raw_label_ids:
                    raise ValueError(
                        f"edge stream labels disagree with cached detections in {source_frame_id}"
                    )
        records.append(
            {
                "frame_idx": int(frame_uid.rsplit("_f", 1)[-1]),
                "source_frame_id": source_frame_id,
                "detection_class_labels": labels,
                "edges": edges,
                "match_indices": matches,
                "observation_uids": observation_uids,
                "edge_stream_manifest": (
                    str(Path(edge_stream_root) / "manifest.json")
                    if edge_stream_root is not None and Path(edge_stream_root).is_dir()
                    else str(edge_stream_root)
                    if edge_stream_root is not None
                    else None
                ),
            }
        )
    if edge_frames is not None and consumed_edge_frames != set(edge_frames):
        extra = sorted(set(edge_frames) - consumed_edge_frames)
        raise ValueError(f"edge stream contains unconsumed frames: {extra[:10]}")
    if edge_manifest is not None and not records:
        raise ValueError("edge stream was supplied but no replay records were built")
    return objects, records
