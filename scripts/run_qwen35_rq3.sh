#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Reuse the existing PARTNR serving environment; do not install vLLM into ICRL.
SERVE_ENV="${LLM_SERVE_ENV:-$HOME/work/llm-serving/.venv-qwen35}"
MODEL_PATH="${LLM_MODEL_PATH:-/lab/ckpt/jsj412/Qwen3.5-27B}"
MODEL_ID="${ICRL_LLM_MODEL:-qwen35-27b}"
RUN_ROOT="${RQ3_RUN_ROOT:-$REPO/outputs/rq3_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}_$$}"
cd "$REPO"
for arg in "$@"; do
    case "$arg" in --out|--out=*) echo 'Use RQ3_RUN_ROOT to select the output directory.' >&2; exit 2;; esac
done
test -x "$REPO/.venv/bin/python" || { echo 'Run uv sync --locked first.' >&2; exit 1; }
mkdir -p "$RUN_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
SERVER_PID=''
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "${ICRL_LLM_BASE_URL:-}" ]]; then
    test -x "$SERVE_ENV/bin/vllm" || { echo "Missing $SERVE_ENV/bin/vllm; set LLM_SERVE_ENV." >&2; exit 1; }
    test -f "$MODEL_PATH/config.json" || { echo "Missing $MODEL_PATH/config.json; set LLM_MODEL_PATH." >&2; exit 1; }
    PORT="$("$REPO/.venv/bin/python" - <<'PY'
import socket
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    print(s.getsockname()[1])
PY
)"
    export ICRL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
    "$SERVE_ENV/bin/vllm" serve "$MODEL_PATH" \
        --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
        --dtype bfloat16 --tensor-parallel-size 1 --language-model-only \
        --reasoning-parser qwen3 --max-model-len 16384 --max-num-seqs 1 \
        --gpu-memory-utilization 0.75 --enforce-eager \
        > "$RUN_ROOT/vllm.log" 2>&1 &
    SERVER_PID=$!
    echo "Starting model server. Log: $RUN_ROOT/vllm.log"
    READY=0
    for ((i=0; i<180; i++)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -n 80 "$RUN_ROOT/vllm.log"
            exit 1
        fi
        if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
            READY=1
            break
        fi
        sleep 5
    done
    if [[ "$READY" != 1 ]]; then
        echo 'vLLM startup timed out.' >&2
        tail -n 80 "$RUN_ROOT/vllm.log"
        exit 1
    fi
else
    echo 'Using the existing model endpoint; it will not be stopped by this script.'
fi

env JAX_PLATFORMS=cpu "$REPO/.venv/bin/python" -m research.coordination.run \
    --backend llm --base-url "$ICRL_LLM_BASE_URL" --model "$MODEL_ID" \
    "$@" --out "$RUN_ROOT/experiment" 2>&1 | tee "$RUN_ROOT/run.log"
