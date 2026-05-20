#!/usr/bin/env bash
# Start a throughput-oriented 1-node 8-GPU SGLang teacher for G2 smoke runs.
#
# Default split is TP=4, DP=2: two teacher replicas, each using four GPUs.
# This is usually a better starting point for online teacher throughput than
# one TP=8 replica when the model fits.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-27b}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

CMD=(
  "${PYTHON_BIN}" -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
)

if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
  CMD+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${EXTRA_SGLANG_ARGS}" ]]; then
  # shellcheck disable=SC2206
  CMD+=(${EXTRA_SGLANG_ARGS})
fi

echo "[teacher] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[teacher] ${SERVED_MODEL_NAME} model=${MODEL_PATH} host=${HOST} port=${PORT} TP=${TP_SIZE} DP=${DP_SIZE}"
printf "[teacher] command:"
printf " %q" "${CMD[@]}"
printf "\n"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" exec "${CMD[@]}"
