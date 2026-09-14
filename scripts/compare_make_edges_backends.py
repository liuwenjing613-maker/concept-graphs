from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.cases import canonical_membership  # noqa: E402
from conceptgraph.revision.index import ProvenanceIndex  # noqa: E402
from conceptgraph.revision.relations import (  # noqa: E402
    load_baseline_frame_records,
    load_edge_stream,
)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _edge_tuple(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["source_entity_uid"]),
        str(row["relation"]),
        str(row["target_entity_uid"]),
        int(row["num_detections"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one real edge stream through ali-dev and ali-my backends"
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--edge-stream", type=Path, required=True)
    parser.add_argument("--ali-dev-repo", type=Path, required=True)
    parser.add_argument("--ali-my-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    provenance = ProvenanceIndex(args.base_run)
    membership = canonical_membership(
        {
            str(row["object_uid"]): row.get("member_observation_uids") or ()
            for row in provenance.final_membership
        }
    )
    objects, records = load_baseline_frame_records(
        provenance, membership, edge_stream_root=args.edge_stream
    )
    replay_input = args.output_root / "backend_replay_input.json"
    _write_json_atomic(
        replay_input,
        {
            "objects": [
                {
                    "entity_uid": row["entity_uid"],
                    "curr_obj_num": row["curr_obj_num"],
                }
                for row in objects
            ],
            "records": [
                {
                    "frame_idx": row["frame_idx"],
                    "source_frame_id": row["source_frame_id"],
                    "detection_class_labels": row["detection_class_labels"],
                    "edges": row["edges"],
                    "match_indices": row["match_indices"],
                }
                for row in records
            ],
        },
    )
    worker = Path(__file__).with_name("replay_make_edges_backend_worker.py")
    repos = {"ali-dev": args.ali_dev_repo.resolve(), "ali-my": args.ali_my_repo.resolve()}
    results = {}
    for name, repo in repos.items():
        output = args.output_root / f"{name}_backend_replay.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repo)
        subprocess.run(
            [
                str(args.python),
                str(worker),
                "--input",
                str(replay_input),
                "--output",
                str(output),
            ],
            cwd=repo,
            env=environment,
            check=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        result["git_commit"] = _git(repo, "rev-parse", "HEAD")
        result["git_status_porcelain"] = _git(repo, "status", "--porcelain")
        results[name] = result

    dev_edges = {_edge_tuple(row) for row in results["ali-dev"]["output_edges"]}
    my_edges = {_edge_tuple(row) for row in results["ali-my"]["output_edges"]}
    edge_manifest, _ = load_edge_stream(args.edge_stream)
    comparison = {
        "schema_version": "0.1.0",
        "status": "PASS" if dev_edges == my_edges else "FAIL",
        "base_run": str(args.base_run.resolve()),
        "edge_stream": str(args.edge_stream.resolve()),
        "edge_stream_frames_sha256": edge_manifest.get("frames_sha256"),
        "detection_parity": edge_manifest.get("parity"),
        "ali_dev": results["ali-dev"],
        "ali_my": results["ali-my"],
        "process_edges_source_equal": (
            results["ali-dev"]["process_edges_sha256"]
            == results["ali-my"]["process_edges_sha256"]
        ),
        "map_edge_classes_source_equal": (
            results["ali-dev"]["map_edge_classes_sha256"]
            == results["ali-my"]["map_edge_classes_sha256"]
        ),
        "edge_state_equal": dev_edges == my_edges,
        "ali_dev_only_edges": [list(item) for item in sorted(dev_edges - my_edges)],
        "ali_my_only_edges": [list(item) for item in sorted(my_edges - dev_edges)],
        "input_sha256": hashlib.sha256(replay_input.read_bytes()).hexdigest(),
    }
    _write_json_atomic(args.output_root / "backend_parity.json", comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if comparison["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
