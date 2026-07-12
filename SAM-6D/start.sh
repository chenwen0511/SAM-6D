#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MICROMAMBA_BIN="${MICROMAMBA_BIN:-/home/mui/.local/bin/micromamba}"
SAM6D_ENV_ROOT="${SAM6D_ENV_ROOT:-/home/mui/.micromamba/envs/sam6d}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8004}"

if [ "${CONDA_DEFAULT_ENV:-}" != "sam6d" ]; then
    if [ ! -x "$MICROMAMBA_BIN" ]; then
        echo "Error: micromamba not found: $MICROMAMBA_BIN"
        exit 1
    fi
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mui/.micromamba}"
    eval "$("$MICROMAMBA_BIN" shell hook --shell bash)"
    micromamba activate sam6d
fi

export LD_LIBRARY_PATH="$SAM6D_ENV_ROOT/lib:$SAM6D_ENV_ROOT/lib/python3.9/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export SAM6D_CAD_PATH="${SAM6D_CAD_PATH:-$ROOT_DIR/user_data/models/tray_180mm_centered_mesh_v2.ply}"
export SAM6D_OUTPUT_ROOT="${SAM6D_OUTPUT_ROOT:-$ROOT_DIR/service_outputs}"
export SAM6D_YOLO_WEIGHTS="${SAM6D_YOLO_WEIGHTS:-$ROOT_DIR/user_data/yolo_runs/tray_seg/weights/best.engine}"
export SAM6D_YOLO_IMGSZ="${SAM6D_YOLO_IMGSZ:-640}"
export SAM6D_YOLO_CONF="${SAM6D_YOLO_CONF:-0.25}"
export SAM6D_YOLO_CLASS_ID="${SAM6D_YOLO_CLASS_ID:-0}"

if [ ! -f "$SAM6D_CAD_PATH" ]; then
    echo "Error: SAM6D_CAD_PATH not found: $SAM6D_CAD_PATH"
    exit 1
fi

if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"; then
    echo "Error: port $PORT is already in use"
    echo "Hint: PORT=8005 sh start.sh"
    exit 1
fi

echo "Starting SAM-6D HTTP service on http://$HOST:$PORT"
echo "CAD: $SAM6D_CAD_PATH"
echo "Output: $SAM6D_OUTPUT_ROOT"

exec python3 sam6d_http_service.py --host "$HOST" --port "$PORT" "$@"
