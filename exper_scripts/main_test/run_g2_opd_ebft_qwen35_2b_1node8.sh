#!/usr/bin/env bash
# Local 1-node 8-GPU OPD run with EBFT-style on-policy credit assignment.
#
# Default layout:
#   teacher Qwen3.5-27B SGLang: GPUs 0,1,2,3
#   student Qwen3.5-2B Slime/Ray: GPUs 4,5,6,7
#
# This is the current OPD mainline:
#   student on-policy rollouts
#   teacher logprobs on those rollouts
#   frozen G1/critic embeddings
#   OPD_CREDIT_ASSIGNMENT=ebft
#   g1_token_advantages -> actor loss
#
# Example:
#   bash exper_scripts/main_test/run_g2_opd_ebft_qwen35_2b_1node8.sh
#
# Reuse an already-running teacher:
#   START_TEACHER=false \
#   TEACHER_API_BASE=http://127.0.0.1:30000 \
#   bash exper_scripts/main_test/run_g2_opd_ebft_qwen35_2b_1node8.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"

if [[ -f "${SLIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ENV_FILE}"
fi

START_TEACHER="${START_TEACHER:-true}"
KEEP_TEACHER_ALIVE="${KEEP_TEACHER_ALIVE:-false}"
PRINT_MODE=false
if [[ "${PRINT_ONLY:-0}" == "1" || "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  PRINT_MODE=true
fi
TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
TEACHER_API_BASE="${TEACHER_API_BASE:-http://${TEACHER_HOST}:${TEACHER_PORT}}"
OPD_TEACHER_RM_URL="${OPD_TEACHER_RM_URL:-${TEACHER_API_BASE%/}/generate}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"

TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STUDENT_CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-2}"
TEACHER_DP_SIZE="${TEACHER_DP_SIZE:-2}"
TEACHER_CONTEXT_LENGTH="${TEACHER_CONTEXT_LENGTH:-2048}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.55}"
TEACHER_EXTRA_SGLANG_ARGS="${TEACHER_EXTRA_SGLANG_ARGS:---disable-cuda-graph --attention-backend triton --sampling-backend pytorch}"
TEACHER_LOG_DIR="${TEACHER_LOG_DIR:-/tmp/slime_opd_1node8}"
TEACHER_LOG="${TEACHER_LOG:-${TEACHER_LOG_DIR}/teacher_${TEACHER_PORT}.log}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-1800}"
TEACHER_POLL_SECONDS="${TEACHER_POLL_SECONDS:-10}"

teacher_pid=""

cleanup_teacher() {
  if [[ -n "${teacher_pid}" && "${KEEP_TEACHER_ALIVE}" != "true" ]]; then
    echo "[1node8-opd-ebft] stopping teacher pid=${teacher_pid}"
    kill "${teacher_pid}" 2>/dev/null || true
    wait "${teacher_pid}" 2>/dev/null || true
  fi
}
trap cleanup_teacher EXIT

wait_for_opd_logprob_endpoint() {
  local url="$1"
  echo "[1node8-opd-ebft] waiting for OPD logprob endpoint: ${url}"
  OPD_TEACHER_RM_URL="${url}" \
  TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}" \
  TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS}" \
  TEACHER_POLL_SECONDS="${TEACHER_POLL_SECONDS}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
import time
import urllib.request

url = os.environ["OPD_TEACHER_RM_URL"]
api_key = os.environ.get("TEACHER_API_KEY", "EMPTY")
deadline = time.time() + float(os.environ.get("TEACHER_WAIT_SECONDS", "1800"))
timeout = float(os.environ.get("TEACHER_POLL_SECONDS", "10"))
payload = {
    "input_ids": [0, 1],
    "sampling_params": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 0,
        "skip_special_tokens": False,
    },
    "return_logprob": True,
    "logprob_start_len": 0,
}
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
last_error = None
while time.time() < deadline:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        token_logprobs = result.get("meta_info", {}).get("input_token_logprobs")
        if isinstance(token_logprobs, list) and len(token_logprobs) >= 2:
            scored = [item[0] for item in token_logprobs[1:] if isinstance(item, (list, tuple)) and item]
            if any(isinstance(value, (int, float)) for value in scored):
                print("[1node8-opd-ebft] OPD logprob endpoint is ready")
                raise SystemExit(0)
        last_error = "missing numeric meta_info.input_token_logprobs"
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    print(f"[1node8-opd-ebft] teacher not ready yet: {last_error}", file=sys.stderr)
    time.sleep(min(10.0, timeout))
print(f"[ERROR] timed out waiting for OPD endpoint {url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

if [[ "${PRINT_MODE}" == "true" ]]; then
  echo "[1node8-opd-ebft] PRINT_ONLY/DRY_RUN_ONLY set; skipping teacher startup and endpoint wait"
elif [[ "${START_TEACHER}" == "true" ]]; then
  mkdir -p "${TEACHER_LOG_DIR}"
  echo "[1node8-opd-ebft] starting teacher on GPUs ${TEACHER_CUDA_VISIBLE_DEVICES}; log=${TEACHER_LOG}"
  HOST="${TEACHER_HOST}" \
  PORT="${TEACHER_PORT}" \
  CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
  TP_SIZE="${TEACHER_TP_SIZE}" \
  DP_SIZE="${TEACHER_DP_SIZE}" \
  CONTEXT_LENGTH="${TEACHER_CONTEXT_LENGTH}" \
  MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC}" \
  SERVED_MODEL_NAME="${TEACHER_MODEL_NAME}" \
  EXTRA_SGLANG_ARGS="${TEACHER_EXTRA_SGLANG_ARGS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SLIME_ROOT}/exper_scripts/smoketest/start_g2_sglang_teacher_1node4.sh" \
    >"${TEACHER_LOG}" 2>&1 &
  teacher_pid="$!"
else
  echo "[1node8-opd-ebft] START_TEACHER=false; using existing teacher at ${OPD_TEACHER_RM_URL}"
fi

if [[ "${PRINT_MODE}" != "true" ]]; then
  wait_for_opd_logprob_endpoint "${OPD_TEACHER_RM_URL}"
fi

export DEPLOY_LAYOUT="${DEPLOY_LAYOUT:-single_node}"
export DEPLOY_ROLE="${DEPLOY_ROLE:-student}"
export CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}"
export TEACHER_API_BASE
export TEACHER_MODEL_NAME
export OPD_TEACHER_RM_URL

export CF_TARGET_MODE="${CF_TARGET_MODE:-opd_onpolicy}"
export OPD_CREDIT_ASSIGNMENT="${OPD_CREDIT_ASSIGNMENT:-ebft}"
export OPD_CF_SCORE_NORMALIZATION="${OPD_CF_SCORE_NORMALIZATION:-mean}"
export OPD_CF_SCORE_TEMPERATURE="${OPD_CF_SCORE_TEMPERATURE:-1.0}"
export OPD_KL_APPLICATION="${OPD_KL_APPLICATION:-auto}"
export G2_OPD_MODE="${G2_OPD_MODE:-opd_ebft_credit_sglang}"
export ENABLE_EFFOPD="${ENABLE_EFFOPD:-false}"
export G1_USE_EBFT_LOSS="${G1_USE_EBFT_LOSS:-false}"

export ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-false}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train.py}"
export COLOCATE="${COLOCATE:-true}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
export CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"
export CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export NUM_EPOCH="${NUM_EPOCH:-1}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-1024}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-${COMPLETION_MAX_LENGTH:-376}}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"

export RUN_NAME="${RUN_NAME:-g2_opd_ebft_qwen35_2b_1node8_$(date +%m%d_%H%M%S)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/slime/outputs}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${RUN_NAME}}"

echo "[1node8-opd-ebft] student GPUs=${CUDA_VISIBLE_DEVICES} teacher=${OPD_TEACHER_RM_URL}"
echo "[1node8-opd-ebft] RUN_NAME=${RUN_NAME} CF_TARGET_MODE=${CF_TARGET_MODE} OPD_CREDIT_ASSIGNMENT=${OPD_CREDIT_ASSIGNMENT}"

bash "${SCRIPT_DIR}/run_g2_opd_qwen35_2b_main.sh" "$@"
