#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /root/autodl-tmp/beauty/activate_conceptgraph.sh
export PYTHONPATH="$ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec /root/autodl-tmp/envs/conceptgraph/bin/python "$ROOT/scripts/run_blocking_association_gate.py" \
 --mode vlm --fallback auto --start 0 --end 2000 --stride 5 \
 --exp-suffix "v7_CLIP_no90_new_$(date +%Y%m%d_%H%M%S)_$$" \
 --detections-exp-suffix frozen_clip_frontend_20260912 \
 --project-root /root/autodl-tmp/beauty/conceptgraphs --worktree "$ROOT" \
 --python /root/autodl-tmp/envs/conceptgraph/bin/python-runtime \
 --dataset-root /root/autodl-tmp/data/Replica --output-root /root/autodl-tmp/results/Replica \
 --no-web-link --model Qwen3.6-35B-A3B-FP8 --vlm-urls http://127.0.0.1:18481 --gpu 0 "$@"
