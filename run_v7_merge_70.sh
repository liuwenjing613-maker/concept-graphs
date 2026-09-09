#!/usr/bin/env bash
set -euo pipefail
# --dataset-root selects raw inputs; --output-root selects experiment storage.
# Default outputs: b0_dataset/Replica/<scene>/exps/<exp-suffix>.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_v7.sh" "$@"
