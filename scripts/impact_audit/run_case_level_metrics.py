#!/usr/bin/env python3
"""Run per-case object/GT audits from a frozen Oracle construction manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path


VOXELS = (0.025, 0.05, 0.10)


def voxel_slug(value: float) -> str:
    return {0.025: "voxel0p025", 0.05: "voxel0p05", 0.10: "voxel0p10"}[value]


def run_one(job: dict, audit_script: Path, output_root: Path) -> dict:
    output_dir = output_root / job["case_id"] / voxel_slug(job["voxel_size"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        return {**job, "status": "reused", "seconds": 0.0, "output_dir": str(output_dir)}

    command = [
        sys.executable,
        str(audit_script),
        "--scene",
        job["scene"],
        "--prediction-map",
        job["prediction_map"],
        "--gt-map",
        job["gt_map"],
        "--voxel-size",
        str(job["voxel_size"]),
        "--output-dir",
        str(output_dir),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    seconds = time.perf_counter() - started
    (output_dir / "run.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "run.stderr.log").write_text(result.stderr, encoding="utf-8")
    status = "completed" if result.returncode == 0 and summary_path.is_file() else "failed"
    return {
        **job,
        "status": status,
        "returncode": result.returncode,
        "seconds": seconds,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-script", type=Path, default=Path(__file__).with_name("object_gt_audit.py"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    construction = json.loads(args.construction_manifest.read_text(encoding="utf-8"))
    jobs = []
    reused_controls = []
    for scene, scene_data in sorted(construction["scenes"].items()):
        for case_id, case_data in sorted(scene_data["case_variants"].items()):
            if case_data.get("no_op", False):
                reused_controls.append({"scene": scene, "case_id": case_id, "family": case_data["family"]})
                continue
            for voxel_size in VOXELS:
                jobs.append(
                    {
                        "scene": scene,
                        "case_id": case_id,
                        "family": case_data["family"],
                        "voxel_size": voxel_size,
                        "prediction_map": case_data["map"],
                        "gt_map": scene_data["o3_source"],
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, job, args.audit_script, args.output_root) for job in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed:02d}/{len(jobs):02d}] {result['status']:9s} "
                f"{result['case_id']} {voxel_slug(result['voxel_size'])} "
                f"{result['seconds']:.2f}s",
                flush=True,
            )

    manifest = {
        "schema_version": "1.0.0",
        "construction_manifest": str(args.construction_manifest),
        "workers": args.workers,
        "wall_seconds": time.perf_counter() - started,
        "job_count": len(jobs),
        "completed": sum(result["status"] in {"completed", "reused"} for result in results),
        "failed": sum(result["status"] == "failed" for result in results),
        "reused_controls": reused_controls,
        "results": sorted(results, key=lambda item: (item["scene"], item["case_id"], item["voxel_size"])),
    }
    manifest_path = args.output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("job_count", "completed", "failed", "wall_seconds")}, indent=2))
    if manifest["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
