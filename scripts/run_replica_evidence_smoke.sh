#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/home/chenkejun/beauty/conceptgraphs/code/official/ali-my
ROOT=/home/chenkejun/beauty/conceptgraphs
PYTHON=/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python
EXP_SUFFIX=${1:-ali_my_evidence_$(date -u +%Y%m%dT%H%M%SZ)}
SAVE_EVIDENCE=${2:-true}
LOG_PATH=${ROOT}/logs/ali-my/${EXP_SUFFIX}.log

mkdir -p "${ROOT}/logs/ali-my"
cd "${WORKTREE}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH="${WORKTREE}" \
"${PYTHON}" conceptgraph/slam/rerun_realtime_mapping.py \
  dataset_root="${ROOT}/data/Replica" \
  dataset_config="${WORKTREE}/conceptgraph/dataset/dataconfigs/replica/replica.yaml" \
  scene_id=room0 start=0 end=40 stride=20 \
  image_height=680 image_width=1200 \
  make_edges=false use_rerun=false save_rerun=false \
  force_detection=false save_detections=false \
  detections_exp_suffix=smoke_detections_stride20 \
  exp_suffix="${EXP_SUFFIX}" \
  save_video=false save_objects_all_frames=false save_pcd=false save_json=true \
  periodically_save_pcd=false save_evidence="${SAVE_EVIDENCE}" device=cuda \
  2>&1 | tee "${LOG_PATH}"

echo "LOG=${LOG_PATH}"
echo "RUN_ROOT=${ROOT}/data/Replica/room0/exps/${EXP_SUFFIX}"
