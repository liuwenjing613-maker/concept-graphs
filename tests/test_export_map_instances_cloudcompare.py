import gzip
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("instance_id", "<u4"),
        ("is_background", "u1"),
    ]
)


def read_binary_ply(path: Path):
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            assert line
            header_lines.append(line.decode("ascii").rstrip())
            if line == b"end_header\n":
                break
        vertex_line = next(line for line in header_lines if line.startswith("element vertex "))
        count = int(vertex_line.rsplit(" ", 1)[1])
        rows = np.fromfile(handle, dtype=PLY_DTYPE, count=count)
    return header_lines, rows


def test_cli_exports_combined_and_separate_instance_ply(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = run_dir / "pcd_smoke.pkl.gz"
    payload = {
        "objects": [
            {
                "id": "uid-0",
                "class_name": "chair",
                "num_detections": 4,
                "is_background": False,
                "pcd_np": np.array(
                    [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [np.nan, 0, 0]],
                    dtype=np.float64,
                ),
            },
            {
                "id": "uid-1",
                "class_name": "table",
                "num_detections": 2,
                "is_background": True,
                "pcd_np": np.array([[1, 0, 0], [1, 0.01, 0]], dtype=np.float64),
            },
        ],
        "class_names": ["chair", "table"],
    }
    with gzip.open(source, "wb") as handle:
        pickle.dump(payload, handle)

    script = Path(__file__).parents[1] / "scripts" / "export_map_instances_cloudcompare.py"
    result = subprocess.run(
        [sys.executable, str(script), str(run_dir), "--separate"],
        check=True,
        text=True,
        capture_output=True,
    )
    output = run_dir / "cloudcompare_instances"
    header, rows = read_binary_ply(output / "instances_colored.ply")

    assert "Points: 5" in result.stdout
    assert "property uint instance_id" in header
    assert "property uchar is_background" in header
    assert len(rows) == 5
    assert set(rows["instance_id"].tolist()) == {0, 1}
    assert len(set(zip(rows["red"], rows["green"], rows["blue"]))) == 2
    assert rows[rows["instance_id"] == 0]["is_background"].tolist() == [0, 0, 0]
    assert rows[rows["instance_id"] == 1]["is_background"].tolist() == [1, 1]

    metadata = json.loads((output / "instances.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert [item["class_name"] for item in metadata] == ["chair", "table"]
    assert manifest["source_object_count"] == 2
    assert manifest["exported_object_count"] == 2
    assert manifest["total_exported_points"] == 5
    assert manifest["invalid_source_points_removed"] == 1
    assert manifest["unique_instance_colors"] == 2
    assert manifest["combined_ply"]["actual_bytes"] == (output / "instances_colored.ply").stat().st_size
    assert len(list((output / "instances").glob("*.ply"))) == 2


def test_cli_refuses_ambiguous_run_directory(tmp_path):
    for name in ["pcd_one.pkl.gz", "pcd_two.pkl.gz"]:
        with gzip.open(tmp_path / name, "wb") as handle:
            pickle.dump({"objects": []}, handle)
    script = Path(__file__).parents[1] / "scripts" / "export_map_instances_cloudcompare.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "exactly one pcd_*.pkl.gz" in result.stderr
