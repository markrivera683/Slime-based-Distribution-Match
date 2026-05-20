#!/usr/bin/env bash
# Start the G2 OPD teacher on a dedicated 8-GPU node (two-node main layout).
#
# Listens on 0.0.0.0 so the student node can reach TEACHER_HOST:PORT.
# Default: Qwen3.5-27B, TP=2, DP=4 (four replicas x 2 GPUs on 80GB A100).
#
# Example:
#   MODEL_PATH=/mnt/data/models/Qwen3.5-27B \
#   SERVED_MODEL_NAME=qwen3.5-27b \
#   bash exper_scripts/main_test/start_g2_sglang_teacher_2node.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"

if [[ -f "${SLIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ENV_FILE}"
fi

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-27B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-27b}"
export TP_SIZE="${TP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-4}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---disable-cuda-graph}"

echo "[teacher-2node] bind ${HOST}:${PORT} TP=${TP_SIZE} DP=${DP_SIZE} devices=${CUDA_VISIBLE_DEVICES}"
exec bash "${SLIME_ROOT}/exper_scripts/smoketest/start_g2_sglang_teacher_1node8.sh"
