from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import uuid
from pathlib import Path

from conceptgraph.slam.slam_classes import MapEdge, MapEdgeMapping
from conceptgraph.slam.utils import process_edges


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _object_id(entity_uid: str) -> uuid.UUID:
    try:
        return uuid.UUID(entity_uid)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_URL, "revision-entity:" + entity_uid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    objects = [
        {
            "entity_uid": str(row["entity_uid"]),
            "id": _object_id(str(row["entity_uid"])),
            "curr_obj_num": int(row["curr_obj_num"]),
        }
        for row in payload["objects"]
    ]
    records = sorted(payload["records"], key=lambda row: int(row["frame_idx"]))
    map_edges = MapEdgeMapping(objects)
    input_edges = 0
    nonempty_frames = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for record in records:
            edges = list(record.get("edges") or ())
            input_edges += len(edges)
            nonempty_frames += bool(edges)
            if edges:
                map_edges = process_edges(
                    list(record.get("match_indices") or ()),
                    {
                        "detection_class_labels": list(
                            record.get("detection_class_labels") or ()
                        ),
                        "edges": edges,
                    },
                    len(objects),
                    objects,
                    map_edges,
                    int(record["frame_idx"]),
                )
            stale = [
                (edge.obj1_idx, edge.obj2_idx)
                for edge in map_edges.edges_by_index.values()
                if int(record["frame_idx"]) - int(edge.first_detected) > 5
                and edge.num_detections < 2
            ]
            for source, target in stale:
                map_edges.delete_edge(source, target)
    edges = [
        {
            "source_entity_uid": str(objects[source]["entity_uid"]),
            "relation": str(edge.rel_type),
            "target_entity_uid": str(objects[target]["entity_uid"]),
            "num_detections": int(edge.num_detections),
        }
        for (source, target), edge in sorted(map_edges.edges_by_index.items())
    ]
    process_source = inspect.getsource(process_edges)
    mapping_source = inspect.getsource(MapEdge) + inspect.getsource(MapEdgeMapping)
    result = {
        "schema_version": "0.1.0",
        "process_edges_module": inspect.getsourcefile(process_edges),
        "map_edge_module": inspect.getsourcefile(MapEdgeMapping),
        "process_edges_sha256": hashlib.sha256(process_source.encode()).hexdigest(),
        "map_edge_classes_sha256": hashlib.sha256(mapping_source.encode()).hexdigest(),
        "frames_replayed": len(records),
        "nonempty_input_frames": nonempty_frames,
        "input_edge_observations": input_edges,
        "output_edges": edges,
    }
    _write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
