#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MICROMAMBA_BIN="${MICROMAMBA_BIN:-/home/mui/.local/bin/micromamba}"
SAM6D_ENV_ROOT="${SAM6D_ENV_ROOT:-/home/mui/.micromamba/envs/sam6d}"
# conda/micromamba may set HOST to autotools triplet; never use it for uvicorn bind.
BIND_HOST="${SAM6D_HOST:-0.0.0.0}"
PORT="${PORT:-8004}"
HTTP_SERVICE_SCRIPT="${HTTP_SERVICE_SCRIPT:-warmup_http_service.py}"
PID_FILE="${PID_FILE:-$ROOT_DIR/service_outputs/.sam6d_http_${PORT}.pid}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/service_outputs_server.log}"

usage() {
    cat <<EOF
Usage: $(basename "$0") {start|stop|restart|status} [options]

Commands:
  start       Start HTTP service in background (default)
  stop        Stop running service
  restart     Stop then start
  status      Show service status

Options (start / restart):
  -f, --foreground   Run in foreground (logs to terminal, no pid file)
  --port PORT        Listen port (default: 8004, or env PORT)
  --host HOST        Bind address (default: 0.0.0.0, or env SAM6D_HOST)

Environment:
  SAM6D_CAD_PATH, SAM6D_OUTPUT_ROOT, SAM6D_YOLO_* , etc.

Examples:
  $(basename "$0") start
  $(basename "$0") start -f
  PORT=8005 $(basename "$0") restart
  $(basename "$0") stop
EOF
}

setup_env() {
    if [ "${CONDA_DEFAULT_ENV:-}" != "sam6d" ]; then
        if [ ! -x "$MICROMAMBA_BIN" ]; then
            echo "Error: micromamba not found: $MICROMAMBA_BIN"
            exit 1
        fi
        export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mui/.micromamba}"
        eval "$("$MICROMAMBA_BIN" shell hook --shell bash)"
        # conda activate.d scripts reference unset vars (e.g. ADDR2LINE); nounset breaks activate.
        set +u
        micromamba activate sam6d
        set -u
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

    mkdir -p "$(dirname "$PID_FILE")"
    mkdir -p "$SAM6D_OUTPUT_ROOT"
}

pid_is_running() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

read_pid() {
    if [ -f "$PID_FILE" ]; then
        tr -d '[:space:]' < "$PID_FILE"
    fi
}

port_in_use() {
    ss -ltn "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"
}

find_pid_by_port() {
    ss -ltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

is_sam6d_process() {
    local pid="$1"
    local cmd
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmd" == *"$HTTP_SERVICE_SCRIPT"* || "$cmd" == *"sam6d_http_service.py"* ]]
}

do_stop() {
    local pid graceful=1

    pid="$(read_pid || true)"
    if [ -n "${pid:-}" ] && pid_is_running "$pid"; then
        if is_sam6d_process "$pid"; then
            echo "Stopping SAM-6D HTTP service (pid=$pid)..."
            kill -TERM "$pid" 2>/dev/null || true
            for _ in $(seq 1 30); do
                pid_is_running "$pid" || break
                sleep 1
            done
            if pid_is_running "$pid"; then
                echo "Force killing pid=$pid"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        else
            echo "Warning: pid $pid is not $HTTP_SERVICE_SCRIPT, skip kill"
            graceful=0
        fi
    fi

    rm -f "$PID_FILE"

    if port_in_use; then
        pid="$(find_pid_by_port || true)"
        if [ -n "${pid:-}" ] && is_sam6d_process "$pid"; then
            echo "Stopping process on port $PORT (pid=$pid)..."
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            pid_is_running "$pid" && kill -KILL "$pid" 2>/dev/null || true
        elif [ "${graceful:-1}" -eq 1 ]; then
            echo "Warning: port $PORT still in use by another process"
            return 1
        fi
    fi

    echo "Stopped."
}

do_status() {
    local pid

    pid="$(read_pid || true)"
    if [ -n "${pid:-}" ] && pid_is_running "$pid" && is_sam6d_process "$pid"; then
        echo "running (pid=$pid, port=$PORT, host=$BIND_HOST)"
        echo "log: $LOG_FILE"
        return 0
    fi

    if port_in_use; then
        pid="$(find_pid_by_port || true)"
        if [ -n "${pid:-}" ] && is_sam6d_process "$pid"; then
            echo "running (pid=$pid on port $PORT, no pid file)"
            echo "log: $LOG_FILE"
            return 0
        fi
        echo "port $PORT in use, but not by $HTTP_SERVICE_SCRIPT"
        return 1
    fi

    echo "stopped"
    return 1
}

do_start() {
    local foreground=0
    local extra_args=()

    while [ $# -gt 0 ]; do
        case "$1" in
            -f|--foreground)
                foreground=1
                shift
                ;;
            --port)
                PORT="$2"
                PID_FILE="$ROOT_DIR/service_outputs/.sam6d_http_${PORT}.pid"
                shift 2
                ;;
            --host)
                BIND_HOST="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                extra_args+=("$1")
                shift
                ;;
        esac
    done

    setup_env

    if do_status >/dev/null 2>&1; then
        echo "Error: service already running on port $PORT"
        do_status
        exit 1
    fi

    if port_in_use; then
        echo "Error: port $PORT is already in use"
        echo "Hint: PORT=8005 $0 start"
        exit 1
    fi

    echo "Starting SAM-6D warmup HTTP service ($HTTP_SERVICE_SCRIPT) on http://$BIND_HOST:$PORT"
    echo "CAD: $SAM6D_CAD_PATH"
    echo "Output: $SAM6D_OUTPUT_ROOT"

    if [ "$foreground" -eq 1 ]; then
        echo "Log: terminal (foreground)"
        exec python3 "$HTTP_SERVICE_SCRIPT" --host "$BIND_HOST" --port "$PORT" "${extra_args[@]}"
    fi

    nohup python3 "$HTTP_SERVICE_SCRIPT" --host "$BIND_HOST" --port "$PORT" "${extra_args[@]}" \
        >> "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1

    if pid_is_running "$pid"; then
        echo "Started (pid=$pid)"
        echo "Log: $LOG_FILE"
    else
        rm -f "$PID_FILE"
        echo "Error: failed to start, see $LOG_FILE"
        tail -20 "$LOG_FILE" 2>/dev/null || true
        exit 1
    fi
}

CMD="${1:-start}"
shift || true

case "$CMD" in
    start)
        do_start "$@"
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop || true
        sleep 1
        do_start "$@"
        ;;
    status)
        do_status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Error: unknown command '$CMD'"
        usage
        exit 1
        ;;
esac
