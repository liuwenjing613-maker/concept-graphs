from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parse_reuse(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        case_uid, separator, path = value.partition("=")
        if not separator or not case_uid or not path:
            raise ValueError("--reuse-live requires CASE_UID=/absolute/run/path")
        result[case_uid] = str(Path(path).resolve())
    return result


def _live_run_complete(path: Path) -> bool:
    manifest = path / "evidence" / "manifest.json"
    if not manifest.exists():
        return False
    value = _read(manifest)
    return str(value.get("status")) == "MAP_COMPLETED_EVIDENCE_VALID"


def _mapping_command(
    *,
    repo_root: Path,
    selection_root: Path,
    dataset_root: Path,
    scene: str,
    detections_suffix: str,
    case: dict[str, Any],
    exp_suffix: str,
) -> list[str]:
    uid = str(case["case_uid"])
    plan = selection_root / "plans" / f"{uid}.json"
    hydra_dir = selection_root / "hydra" / uid
    latest = selection_root / "latest" / f"{uid}.pkl.gz"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    latest.parent.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable,
        str(repo_root / "conceptgraph" / "slam" / "rerun_realtime_mapping.py"),
        f"dataset_root={dataset_root}",
        f"scene_id={scene}",
        "start=0",
        "end=2000",
        "stride=10",
        f"exp_suffix={exp_suffix}",
        f"detections_exp_suffix={detections_suffix}",
        "force_detection=false",
        "save_detections=false",
        "make_edges=false",
        "use_wandb=false",
        "use_rerun=false",
        "save_rerun=false",
        "vis_render=false",
        "debug_render=false",
        "save_video=false",
        "save_objects_all_frames=false",
        "save_parity_trace=true",
        "save_evidence=true",
        "evidence_mode=best_effort",
        "revision.enabled=true",
        "revision.mode=controlled_validation",
        f"revision.corruption_plan={plan}",
        f"latest_pcd_filepath={latest}",
        f"hydra.run.dir={hydra_dir}",
    ]


def _run_mappings(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, str]:
    selection_root = Path(args.selection_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    reuse = _parse_reuse(args.reuse_live)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    pending = []
    live_runs: dict[str, str] = {}
    for case in cases:
        uid = str(case["case_uid"])
        if uid in reuse:
            path = Path(reuse[uid])
            if not _live_run_complete(path):
                raise RuntimeError(f"reused live run is incomplete: {path}")
            live_runs[uid] = str(path)
            continue
        suffix = f"ali_my_revision_v1_live_{uid}_20260823"
        destination = dataset_root / scene_path(args.scene) / "exps" / suffix
        if _live_run_complete(destination):
            live_runs[uid] = str(destination)
            continue
        pending.append((case, suffix, destination))

    logs = selection_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: list[dict[str, Any]] = []
    failures = []
    cursor = 0
    try:
        while cursor < len(pending) or active:
            while cursor < len(pending) and len(active) < min(args.jobs, len(gpus)):
                case, suffix, destination = pending[cursor]
                gpu = gpus[cursor % len(gpus)]
                cursor += 1
                uid = str(case["case_uid"])
                handle = (logs / f"live_{uid}.log").open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                process = subprocess.Popen(
                    _mapping_command(
                        repo_root=repo_root,
                        selection_root=selection_root,
                        dataset_root=dataset_root,
                        scene=args.scene,
                        detections_suffix=args.detections_suffix,
                        case=case,
                        exp_suffix=suffix,
                    ),
                    cwd=repo_root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active.append(
                    {
                        "process": process,
                        "handle": handle,
                        "case_uid": uid,
                        "gpu": gpu,
                        "destination": destination,
                    }
                )
            time.sleep(1.0)
            remaining = []
            for row in active:
                code = row["process"].poll()
                if code is None:
                    remaining.append(row)
                    continue
                row["handle"].close()
                uid = row["case_uid"]
                if code == 0 and _live_run_complete(row["destination"]):
                    live_runs[uid] = str(row["destination"])
                else:
                    failures.append(
                        {"case_uid": uid, "exit_code": code, "gpu": row["gpu"]}
                    )
                print(
                    json.dumps(
                        {"case_uid": uid, "exit_code": code, "gpu": row["gpu"]}
                    ),
                    flush=True,
                )
            active = remaining
    finally:
        for row in active:
            process = row["process"]
            if process.poll() is None:
                process.terminate()
        for row in active:
            process = row["process"]
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not row["handle"].closed:
                row["handle"].close()
    if failures:
        raise RuntimeError(f"live fidelity mapping failures: {failures}")
    return live_runs


def scene_path(scene: str) -> Path:
    # Kept separate to make accidental writes outside the requested dataset root
    # obvious and testable.
    if not scene or Path(scene).name != scene:
        raise ValueError("scene must be one directory name")
    return Path(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen V1 live-fidelity subset")
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--detections-suffix", default="room0_detections_stride10")
    parser.add_argument("--gpus", default="1,3,5")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--reuse-live", action="append", default=[])
    args = parser.parse_args()
    selection_root = Path(args.selection_root).resolve()
    cases = list(_read(selection_root / "cases.json"))
    live_runs = _run_mappings(args, cases)
    result = {
        "schema_version": "1.0.0",
        "case_count": len(cases),
        "live_runs": live_runs,
        "all_complete": len(live_runs) == len(cases),
    }
    _write(selection_root / "live_orchestration.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
