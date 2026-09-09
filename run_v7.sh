#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
if [[ "$MODE" != human && "$MODE" != auto ]]; then
  echo "Usage: bash $0 human|auto [--scene room0 --end 2000 --stride 5 --gpu 1 --exp-suffix NAME]"
  echo "Both modes run staged VLM. human: unresolved cases need a choice. auto: discard observation / keep separate."
  echo "Negative decision + max(containment)>90% AND min(containment)>60%: directly approve merge in both modes."
  exit 2
fi
shift
PYTHON="${V7_PYTHON:-/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python}"
export PYTHONPATH="$ROOT/.runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "$ROOT/scripts/run_blocking_association_gate.py" --mode vlm --fallback "$MODE" --exp-suffix "v7_merge_70_${MODE}_$(date +%Y%m%d_%H%M%S)_$$" "$@"
