import json
from pathlib import Path

from make_v2_large_worklist import main


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_large_worklist_keeps_probability_and_harvest_contract(tmp_path, monkeypatch):
    associations = []
    routes = []
    for index in range(30):
        uid = f"e{index:03d}"
        decision = "CREATE_OBJECT" if index % 5 == 0 else "MERGE_TO_OBJECT"
        associations.append({"event_uid": uid, "decision": decision})
        label = "CORRECT_NEW" if decision == "CREATE_OBJECT" else "CORRECT_ATTACH"
        if index in {10, 11}:
            label = "WRONG_NEW_FALSE_SPLIT" if decision == "CREATE_OBJECT" else "WRONG_ATTACH_EXISTING"
        routes.append(
            {
                "event_uid": uid,
                "decision": decision,
                "private_auto_evaluable": True,
                "private_auto_routing_label": label,
                "processed_frame_idx": index,
                "candidate_count": index % 7,
                "margin": index / 10,
                "private_legal_candidate_uids": [],
            }
        )
    assoc_path = tmp_path / "associations.jsonl"
    route_path = tmp_path / "routes.jsonl"
    output = tmp_path / "out"
    write_jsonl(assoc_path, associations)
    write_jsonl(route_path, routes)
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_v2_large_worklist.py",
            "--associations",
            str(assoc_path),
            "--routing-records",
            str(route_path),
            "--output-root",
            str(output),
            "--probability-count",
            "10",
            "--matched-controls-per-error",
            "1",
            "--hidden-repeat-fraction",
            "0.1",
        ],
    )
    assert main() == 0
    manifest = json.loads((output / "worklist_manifest.json").read_text())
    rows = [json.loads(line) for line in (output / "private_large_worklist.jsonl").read_text().splitlines()]
    assert manifest["probability_sample_count"] == 10
    assert manifest["error_harvest_count"] == 2
    assert manifest["matched_control_count"] == 2
    assert manifest["hidden_repeat_count"] == 2
    assert len(rows) == manifest["total_case_count"]
    assert len({row["case_uid"] for row in rows}) == len(rows)
