#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TAG="$(date +%Y%m%d_%H%M%S)_$$"
exec /home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python "$ROOT/scripts/run_v7_1_experiment.py" --exp-suffix "v7_1_auto_${TAG}" --run-dir "/home/chenkejun/beauty/v7_1_20260909/full_${TAG}" "$@"
