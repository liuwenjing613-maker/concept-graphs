#!/usr/bin/env python3
"""Launch one fresh online ali-dev mapping run with a blocking gate mode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _bool(value: bool) -> str:
    return str(bool(value)).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("off", "audit", "oracle", "vlm"))
    parser.add_argument("--exp-suffix", required=True)
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--detections-exp-suffix", default="ali_dev_room0_stride5_det_frozen")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--threshold-distance", type=float, default=0.30)
    parser.add_argument("--threshold-scope", choices=("create_only", "both"), default="create_only")
    parser.add_argument("--candidate-iou-threshold", type=float, default=0.85)
    parser.add_argument("--no-candidate-iou-filter", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="high", choices=("none", "low", "medium", "high"))
    parser.add_argument("--base-url", default="https://api.codelink.chat/v1")
    parser.add_argument("--no-api-key-required", action="store_true")
    parser.add_argument("--rerun-connect-addr")
    parser.add_argument("--no-save-pcd", action="store_true")
    parser.add_argument("--no-observation-pcd", action="store_true")
    parser.add_argument("--project-root", default="/home/chenkejun/beauty/conceptgraphs")
    parser.add_argument("--worktree", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default="/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python")
    parser.add_argument(
        "--dataset-root",
        default="/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica",
    )
    parser.add_argument("--dataset-config")
    parser.add_argument("--oracle-gt-path", default="/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/corrected_gt_audit_room0/observation_gt.jsonl")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    worktree = Path(args.worktree).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_config = Path(args.dataset_config).resolve() if args.dataset_config else worktree / "conceptgraph" / "dataset" / "dataconfigs" / "replica" / "replica.yaml"
    exp_root = dataset_root / args.scene / "exps" / args.exp_suffix
    if exp_root.exists():
        raise FileExistsError(f"refusing to reuse any existing map directory: {exp_root}")
    if args.mode == "vlm" and not args.no_api_key_required and not os.environ.get("GATE_API_KEY"):
        raise RuntimeError("GATE_API_KEY must be present only in the process environment for vlm mode")
    if args.mode == "oracle" and not Path(args.oracle_gt_path).is_file():
        raise FileNotFoundError(args.oracle_gt_path)

    use_rerun = bool(args.rerun_connect_addr)
    command = [
        str(Path(args.python).resolve()),
        "conceptgraph/slam/rerun_realtime_mapping.py",
        f"dataset_root={dataset_root}",
        f"dataset_config={dataset_config}",
        f"scene_id={args.scene}",
        f"start={args.start}",
        f"end={args.end}",
        f"stride={args.stride}",
        "image_height=680",
        "image_width=1200",
        "make_edges=false",
        f"use_rerun={_bool(use_rerun)}",
        "save_rerun=false",
        f"rerun_connect_addr={args.rerun_connect_addr or 'null'}",
        "force_detection=false",
        "save_detections=false",
        f"detections_exp_suffix={args.detections_exp_suffix}",
        f"exp_suffix={args.exp_suffix}",
        "save_video=false",
        "save_objects_all_frames=false",
        f"save_pcd={_bool(not args.no_save_pcd)}",
        "save_json=true",
        "periodically_save_pcd=false",
        "save_evidence=true",
        "evidence_mode=strict",
        f"evidence_save_observation_pcd={_bool(not args.no_observation_pcd)}",
        "save_parity_trace=true",
        "device=cuda",
        "revision.enabled=false",
        f"association_gate.mode={args.mode}",
        f"association_gate.margin_threshold={args.margin_threshold}",
        f"association_gate.threshold_distance={args.threshold_distance}",
        f"association_gate.threshold_scope={args.threshold_scope}",
        "association_gate.association_top_k=2",
        "association_gate.create_top_k=3",
        f"association_gate.candidate_iou_filter_enabled={_bool(not args.no_candidate_iou_filter)}",
        f"association_gate.candidate_iou_threshold={args.candidate_iou_threshold}",
        f"association_gate.model={args.model}",
        f"association_gate.reasoning_effort={args.reasoning_effort if args.reasoning_effort != 'none' else 'null'}",
        f"association_gate.base_url={args.base_url}",
        f"association_gate.api_key_required={_bool(not args.no_api_key_required)}",
        f"association_gate.oracle_gt_path={args.oracle_gt_path if args.mode == 'oracle' else 'null'}",
        f"association_gate.max_events={args.max_events}",
        "hydra.run.dir=.",
        "hydra.verbose=false",
        "hydra.job_logging.root.level=INFO",
    ]
    launch_dir = project_root / "results" / "blocking_association_gate_v1" / "launches"
    launch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = launch_dir / f"{args.exp_suffix}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite launch manifest: {manifest_path}")
    manifest = {
        "schema_version": "blocking-association-gate-launch-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": args.mode,
        "fresh_online_map": True,
        "worktree": str(worktree),
        "experiment_root": str(exp_root),
        "cuda_visible_devices": args.gpu,
        "command": command,
        "credentials": "environment only; not recorded",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[launch] mode={args.mode} fresh_output={exp_root}", flush=True)
    print(f"[launch] GPU={args.gpu} frames=[{args.start},{args.end}) stride={args.stride}", flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(worktree) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    completed = subprocess.run(command, cwd=worktree, env=environment, check=False)
    manifest["return_code"] = completed.returncode
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[launch] return_code={completed.returncode} output={exp_root}", flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
