#!/usr/bin/env bash
set -euo pipefail

# Usage: bash evaluate_instance_pq.sh v4_human2000
# Or change the default EXP below, as in the original evaluate.sh.
EXP="${1:-off2000_10}"
[[ "$EXP" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid experiment name' >&2; exit 2; }
ROOT=/home/chenkejun/beauty/conceptgraphs
PROJECT="$ROOT/code/experiments/ali-dev-blocking-gate-v1-20260903"
TARGET="$ROOT/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/$EXP"
MAP="$TARGET/pcd_${EXP}.pkl.gz"
OUT="$TARGET/instance_pq_fixed_support_$(date +%Y%m%d_%H%M%S_%N)"
test -f "$MAP"
mkdir -- "$OUT"

HF_HUB_OFFLINE=1 /usr/bin/time -v "$ROOT/envs/cg-main/bin/python" \
  "$PROJECT/scripts/eval_replica_instance_pq.py" \
  --project-root "$PROJECT" \
  --replica-semantic-root "$ROOT/data/ReplicaSemanticGT" \
  --replica-ssg-root /data/chenkejun/ReplicaSSG \
  --slam-root "$ROOT/results/main/replica_stride5/Replica" \
  --scene room0 room_0 "$MAP" \
  --output-dir "$OUT" --n-exclude 6 --workers 8 \
  --reference-start 0 --reference-end 2000 --reference-stride 5 \
  --gt-transfer-max-distance-m 0.03 --prediction-max-distance-m 0.05 \
  2>&1 | tee "$OUT/evaluation.log"

printf '\n实例评估结果目录：%s\n' "$OUT"
cat "$OUT/instance_pq.csv"
