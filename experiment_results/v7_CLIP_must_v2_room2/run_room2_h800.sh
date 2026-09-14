#!/bin/bash
set -euo pipefail
W=/home/chenkejun/beauty/conceptgraphs/code/experiments/v7_CLIP_must_v2
P=/home/chenkejun/beauty/v7_CLIP_must_v2_run_20260914
export PYTHONPATH="$W/.runtime-deps:$W:/home/chenkejun/beauty/conceptgraphs/code/experiments/v7_CLIP/.runtime-deps"
export V7_MODEL_IDENTITY_FILE="$P/h800/model_identity.json"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python
"$PY" -c "import json;from pathlib import Path;p=Path('$P');assert json.loads((p/'input_validation.json').read_text())['status']=='PASS';assert json.loads((p/'h800_smoke_validation.json').read_text())['status']=='PASS'"
"$PY" "$W/scripts/run_blocking_association_gate.py" --mode vlm --fallback auto --scene room2 --start 0 --end 2000 --stride 5 --exp-suffix v7_CLIP_must_v2_room2_H800_20260914 --detections-exp-suffix frozen_clip_frontend_20260912 --worktree "$W" --python "$PY" --dataset-root /home/chenkejun/beauty/conceptgraphs/data/Replica --output-root "$P/Replica" --web-root "$P/web" --web-base-url http://127.0.0.1:18927 --model Qwen3.6-35B-A3B-FP8 --vlm-urls http://127.0.0.1:18481 http://127.0.0.1:18482 --gpu 5
