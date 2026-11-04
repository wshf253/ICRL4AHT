#!/usr/bin/env bash
# Serve Qwen3.8-27B with vLLM inside this job, check that thinking is off, run one
# cooperative-intelligence experiment config, then stop the server.
#
#   bash scripts/run_qwen38_coop.sh research/coop_tom/configs/exp0_smoke.yaml --workers 8
#
# Everything after the config is passed to `python -m research.coop_tom.run`.
# With ICRL_LLM_BASE_URL already set, no server is started and that endpoint is used.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: run_qwen38_coop.sh <config.yaml> [run.py options]}"
shift
# vLLM lives in its own venv (never in the ICRL env). Qwen3.8 may need a newer vLLM than
# the Qwen3.5 serving env; point LLM_SERVE_ENV at one that supports it.
SERVE_ENV="${LLM_SERVE_ENV:-$HOME/work/llm-serving/.venv-qwen35}"
MODEL_PATH="${LLM_MODEL_PATH:-/lab/ckpt/jsj412/Qwen3.8-27B}"
export ICRL_LLM_MODEL="${ICRL_LLM_MODEL:-qwen3.8-27b}"
OUT_ROOT="${COOP_OUT:-$REPO/outputs/coop_tom}"
LOG_DIR="$OUT_ROOT/serve_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}_$$"
# Concurrent episodes (--workers) only help if vLLM batches them.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM="${GPU_MEM:-0.85}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"      # empty = no parser (thinking then appears inline)
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"     # skip the vision tower of multimodal checkpoints
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"                 # 1 = slower but avoids CUDA-graph capture issues

cd "$REPO"
test -x "$REPO/.venv/bin/python" || { echo 'Run uv sync --locked first.' >&2; exit 1; }
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
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
    PORT="$("$REPO/.venv/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
    export ICRL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
    ARGS=(--served-model-name "$ICRL_LLM_MODEL" --host 127.0.0.1 --port "$PORT"
          --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384
          --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization "$GPU_MEM" --enable-prefix-caching)
    [[ -n "$REASONING_PARSER" ]] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
    [[ "$LANGUAGE_MODEL_ONLY" == 1 ]] && ARGS+=(--language-model-only)
    [[ "$ENFORCE_EAGER" == 1 ]] && ARGS+=(--enforce-eager)
    "$SERVE_ENV/bin/vllm" serve "$MODEL_PATH" "${ARGS[@]}" > "$LOG_DIR/vllm.log" 2>&1 &
    SERVER_PID=$!
    echo "Starting vLLM ($MODEL_PATH). Log: $LOG_DIR/vllm.log"
    READY=0
    for ((i=0; i<240; i++)); do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 80 "$LOG_DIR/vllm.log"; exit 1; }
        if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then READY=1; break; fi
        sleep 5
    done
    [[ "$READY" == 1 ]] || { echo 'vLLM startup timed out.' >&2; tail -n 80 "$LOG_DIR/vllm.log"; exit 1; }
else
    echo "Using the existing endpoint $ICRL_LLM_BASE_URL; it will not be stopped by this script."
fi

# Thinking must be off (the experiments assume short JSON replies). Every candidate is
# probed once and reported; the first passing one is used unless COOP_CHAT_TEMPLATE_KWARGS is set.
if ! PICKED="$("$REPO/.venv/bin/python" -m research.coop_tom.check --base-url "$ICRL_LLM_BASE_URL" \
        --model "$ICRL_LLM_MODEL" --pick --report "$LOG_DIR/thinking_check.jsonl")"; then
    cat "$LOG_DIR/thinking_check.jsonl" 2>/dev/null || true
    exit 1
fi
cat "$LOG_DIR/thinking_check.jsonl"
export COOP_CHAT_TEMPLATE_KWARGS="${COOP_CHAT_TEMPLATE_KWARGS:-$PICKED}"
echo "chat_template_kwargs: $COOP_CHAT_TEMPLATE_KWARGS"

"$REPO/.venv/bin/python" -m research.coop_tom.run "$CONFIG" --out "$OUT_ROOT" "$@" 2>&1 | tee "$LOG_DIR/run.log"
