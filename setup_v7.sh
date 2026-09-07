#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${V7_PYTHON:-/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python}"
"$PYTHON" -m pip install --upgrade --target "$ROOT/.runtime-deps" --no-deps Pillow==11.0.0 opencv-python-headless==4.12.0.88
PYTHONPATH="$ROOT/.runtime-deps" "$PYTHON" -c 'from PIL import features; assert features.check("raqm"), "RAQM required for validated VLM image rendering"'
