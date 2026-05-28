#!/usr/bin/env bash
# Full standalone pure G1/EBFT-GT Slime/Megatron run for Qwen3.5-2B.
#
# Target is the dataset GT answer. This script intentionally does not enable
# OPD, teacher-completion targets, or G2/cf_l1oo distribution rewards. It
# assembles the Slime CLI, starts Ray, submits the job, and records artifacts
# directly.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# 0. Runtime paths
# ---------------------------------------------------------------------------
# SLIME_ROOT: this repository. MEGATRON_PATH must point to the Megatron-LM tree
# used by Slime. PYTHON_BIN is the environment used for Ray workers.
if [[ -n "${SLIME_ROOT:-}" && "${SLIME_ROOT}" != "${LAUNCHER_SLIME_ROOT}" ]]; then
  echo "[root-guard] ignoring inherited SLIME_ROOT=${SLIME_ROOT}; launcher root is ${LAUNCHER_SLIME_ROOT}" >&2
fi
SLIME_ROOT="${LAUNCHER_SLIME_ROOT}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SLIME_ROOT}/../.." && pwd)}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/slime_runtime/Megatron-LM}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"

# ---------------------------------------------------------------------------
# 1. Model and dataset
# ---------------------------------------------------------------------------
# MODEL_PATH is the HF checkpoint used by SGLang/tokenizer. REF_LOAD is the
# converted Megatron checkpoint for the frozen G1 reference/reward source.
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-2B}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_PATH}}"
REF_LOAD="${REF_LOAD:-/mnt/data/models/Megatron_convert_models/Qwen3.5-2B_torch_dist}"
SLIME_DATA_ROOT="${SLIME_DATA_ROOT:-${PROJECT_ROOT}/data}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-${SLIME_DATA_ROOT}/diff_dataset_prepared}"
SLIME_TRAIN_DATA="${SLIME_TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_slime_qa_100k.jsonl}"

# User-facing length knobs:
# - PROMPT_MAX_LENGTH filters real training prompt+label tokens.
# - COMPLETION_MAX_LENGTH controls fixed-length training completions.
# G1_PROMPT_LENGTH is a padded internal geometry length derived later unless
# explicitly overridden.
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-${G1_MAX_PROMPT_LABEL_LEN:-512}}"
COMPLETION_MAX_LENGTH="${COMPLETION_MAX_LENGTH:-${ROLLOUT_MAX_RESPONSE_LEN:-512}}"

G1_FILTER_TRAIN_DATA="${G1_FILTER_TRAIN_DATA:-true}"
G1_MAX_PROMPT_LABEL_LEN="${G1_MAX_PROMPT_LABEL_LEN:-${PROMPT_MAX_LENGTH}}"
G1_FILTERED_SLIME_TRAIN_DATA="${G1_FILTERED_SLIME_TRAIN_DATA:-${SLIME_TRAIN_DATA%.jsonl}_g1_prompt${G1_MAX_PROMPT_LABEL_LEN}.jsonl}"

# Online eval datasets. ENABLE_SLIME_EVAL=false means these are not passed to
# Slime; this script currently does not run a separate post-eval stage.
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
MBPP_SLIME_EVAL_DATA="${MBPP_SLIME_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa_slime.jsonl}"
HUMANEVAL_SLIME_EVAL_DATA="${HUMANEVAL_SLIME_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa_slime.jsonl}"

# ---------------------------------------------------------------------------
# 2. Parallelism and resource layout
# ---------------------------------------------------------------------------
# Default is async on one 8-GPU node: 4 GPUs for Megatron actor, 4 GPUs for
# SGLang rollout. Set ENABLE_ASYNC_TRAIN=false to use sync colocate mode.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
USER_SET_ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE+x}"
USER_SET_SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH+x}"
USER_SET_SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS+x}"
USER_SET_SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND+x}"
USER_SET_SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND+x}"
USER_SET_SGLANG_DIRECT_WORKER_MODE="${SGLANG_DIRECT_WORKER_MODE+x}"
USER_SET_SLIME_SGLANG_HEALTH_TIMEOUT="${SLIME_SGLANG_HEALTH_TIMEOUT+x}"
USER_SET_SLIME_SGLANG_HEALTH_MAX_WAIT="${SLIME_SGLANG_HEALTH_MAX_WAIT+x}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

# ---------------------------------------------------------------------------
# 3. Training horizon and batch geometry
# ---------------------------------------------------------------------------
# By default this runs one epoch over the rollout dataset. For quick tests,
# override NUM_ROLLOUT; if NUM_ROLLOUT is set, Slime ignores NUM_EPOCH.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
NUM_EPOCH="${NUM_EPOCH:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"

# ---------------------------------------------------------------------------
# 4. SGLang rollout generation
# ---------------------------------------------------------------------------
SGLANG_STABLE_ROLLOUT_MODE="${SGLANG_STABLE_ROLLOUT_MODE:-false}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-1024}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-${COMPLETION_MAX_LENGTH}}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-false}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-}"
SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-}"
SGLANG_ROUTER_DISABLE_CIRCUIT_BREAKER="${SGLANG_ROUTER_DISABLE_CIRCUIT_BREAKER:-true}"
SGLANG_ROUTER_HEALTH_CHECK_ENDPOINT="${SGLANG_ROUTER_HEALTH_CHECK_ENDPOINT:-/health_generate}"
SGLANG_DIRECT_WORKER_MODE="${SGLANG_DIRECT_WORKER_MODE:-false}"
SLIME_SGLANG_HEALTH_TIMEOUT="${SLIME_SGLANG_HEALTH_TIMEOUT:-10}"
SLIME_SGLANG_HEALTH_MAX_WAIT="${SLIME_SGLANG_HEALTH_MAX_WAIT:-1200}"
SLIME_ROUTER_WORKER_WAIT_TIMEOUT="${SLIME_ROUTER_WORKER_WAIT_TIMEOUT:-1200}"
SLIME_ROUTER_WORKER_WAIT_INTERVAL="${SLIME_ROUTER_WORKER_WAIT_INTERVAL:-2}"
SLIME_ROUTER_WORKER_REQUEST_TIMEOUT="${SLIME_ROUTER_WORKER_REQUEST_TIMEOUT:-10}"
SLIME_ROUTER_DISABLE_CIRCUIT_BREAKER="${SLIME_ROUTER_DISABLE_CIRCUIT_BREAKER:-true}"
SLIME_HTTP_REQUEST_TIMEOUT="${SLIME_HTTP_REQUEST_TIMEOUT:-10}"
if [[ "${SGLANG_STABLE_ROLLOUT_MODE}" == "true" || "${SGLANG_STABLE_ROLLOUT_MODE}" == "1" ]]; then
  [[ -z "${USER_SET_SGLANG_DIRECT_WORKER_MODE}" ]] && SGLANG_DIRECT_WORKER_MODE="true"
  [[ -z "${USER_SET_SGLANG_DISABLE_CUDA_GRAPH}" ]] && SGLANG_DISABLE_CUDA_GRAPH="true"
  [[ -z "${USER_SET_SGLANG_MAX_RUNNING_REQUESTS}" ]] && SGLANG_MAX_RUNNING_REQUESTS="16"
  [[ -z "${USER_SET_SGLANG_ATTENTION_BACKEND}" ]] && SGLANG_ATTENTION_BACKEND="triton"
  [[ -z "${USER_SET_SGLANG_SAMPLING_BACKEND}" ]] && SGLANG_SAMPLING_BACKEND="pytorch"
  [[ -z "${USER_SET_SLIME_SGLANG_HEALTH_TIMEOUT}" ]] && SLIME_SGLANG_HEALTH_TIMEOUT="10"
  [[ -z "${USER_SET_SLIME_SGLANG_HEALTH_MAX_WAIT}" ]] && SLIME_SGLANG_HEALTH_MAX_WAIT="1200"
fi

# ---------------------------------------------------------------------------
# 5. Optimizer and PPO-style actor loss knobs
# ---------------------------------------------------------------------------
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
SKIP_INITIAL_UPDATE_WEIGHTS="${SKIP_INITIAL_UPDATE_WEIGHTS:-false}"

# ---------------------------------------------------------------------------
# 6. EBFT reward and loss
# ---------------------------------------------------------------------------
# Phase-1 parity path: frozen Megatron ref produces G1 embeddings/rewards, then
# actor uses EBFT RL+CE loss. KL/entropy parity is intentionally not enabled.
G1_USE_EBFT_LOSS="${G1_USE_EBFT_LOSS:-true}"
G1_APPLY_DENSE_ATTENTION_MASK="${G1_APPLY_DENSE_ATTENTION_MASK:-false}"
G1_QA_MASKING="${G1_QA_MASKING:-false}"
G1_CE_LOSS_COEF="${G1_CE_LOSS_COEF:-0.03}"
G1_PROMPT_LENGTH="${G1_PROMPT_LENGTH:-}"
G1_CONTEXT_LENGTH="${G1_CONTEXT_LENGTH:-8}"
G1_GENERATE_LENGTH="${G1_GENERATE_LENGTH:-8}"
G1_STRIDE="${G1_STRIDE:-8}"
G1_RESPONSE_LENGTH="${G1_RESPONSE_LENGTH:-${COMPLETION_MAX_LENGTH}}"
G1_HIDDEN_STATE_METHOD="${G1_HIDDEN_STATE_METHOD:-last_only}"
G1_EMBEDDING_SOURCE="${G1_EMBEDDING_SOURCE:-megatron_ref}"
G1_REWARD_LOCATION="${G1_REWARD_LOCATION:-trainer}"
G1_REF_FORWARD_MODE="${G1_REF_FORWARD_MODE:-openrlhf_exact}"
# Leave unset for the launcher default. Strict EBFT block-source runs set this
# to strict_block_source through the wrapper script below.
G1_EBFT_LOGPROB_INDEXING="${G1_EBFT_LOGPROB_INDEXING:-}"
USE_WHITENING="${USE_WHITENING:-true}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-1.0}"

# ---------------------------------------------------------------------------
# 7. Megatron execution/performance
# ---------------------------------------------------------------------------
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

# ---------------------------------------------------------------------------
# 8. Online eval
# ---------------------------------------------------------------------------
ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL:-false}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-512}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1536}"

# ---------------------------------------------------------------------------
# 9. Checkpoint/artifacts/Ray
# ---------------------------------------------------------------------------
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SLIME_ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-g1_ebft_gt_qwen35_2b_main_$(date +%m%d_%H%M%S)}"
LOAD_PATH="${LOAD_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/mcore}"
SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/artifacts}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_g1_main}"
DRIVER_LOG="${ARTIFACT_DIR}/ray_job_driver.log"

if [[ -f "${SLIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ENV_FILE}"
fi
if [[ "${SLIME_ROOT:-}" != "${LAUNCHER_SLIME_ROOT}" ]]; then
  echo "[root-guard] ${SLIME_ENV_FILE} set SLIME_ROOT=${SLIME_ROOT:-<unset>}; restoring launcher root ${LAUNCHER_SLIME_ROOT}" >&2
  SLIME_ROOT="${LAUNCHER_SLIME_ROOT}"
fi

if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}:${EXTRA_PYTHONPATH}"
else
  export PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}"
fi
export HF_HOME="${HF_HOME:-${SLIME_ROOT}/caches/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export RAY_TMPDIR
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export SLIME_SGLANG_HEALTH_TIMEOUT
export SLIME_SGLANG_HEALTH_MAX_WAIT
export SLIME_ROUTER_WORKER_WAIT_TIMEOUT
export SLIME_ROUTER_WORKER_WAIT_INTERVAL
export SLIME_ROUTER_WORKER_REQUEST_TIMEOUT
export SLIME_ROUTER_DISABLE_CIRCUIT_BREAKER
export SLIME_HTTP_REQUEST_TIMEOUT

# DLC/DSW images often export HTTP_PROXY/HTTPS_PROXY. The Rust sgl-router
# honors those variables, so its worker health probes to 10.x node IPs must be
# excluded explicitly; otherwise healthy SGLang engines look unavailable and
# router /generate returns 503 no_available_workers.
HOST_IP_FOR_NO_PROXY="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
SLIME_NO_PROXY_EXTRA="127.0.0.1,localhost,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
if [[ -n "${HOST_IP_FOR_NO_PROXY}" ]]; then
  SLIME_NO_PROXY_EXTRA="${SLIME_NO_PROXY_EXTRA},${HOST_IP_FOR_NO_PROXY}"
fi
if [[ -n "${MASTER_ADDR:-}" ]]; then
  SLIME_NO_PROXY_EXTRA="${SLIME_NO_PROXY_EXTRA},${MASTER_ADDR}"
fi
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${NO_PROXY},${SLIME_NO_PROXY_EXTRA}"
else
  export NO_PROXY="${SLIME_NO_PROXY_EXTRA}"
fi
export no_proxy="${NO_PROXY}"

ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-true}"
case "${ENABLE_ASYNC_TRAIN}" in
  true|false) ;;
  *) echo "[ERROR] ENABLE_ASYNC_TRAIN must be true or false, got: ${ENABLE_ASYNC_TRAIN}" >&2; exit 1 ;;
esac

if [[ -z "${TRAIN_ENTRYPOINT+x}" ]]; then
  if [[ "${ENABLE_ASYNC_TRAIN}" == "true" ]]; then
    TRAIN_ENTRYPOINT="train_async.py"
  else
    TRAIN_ENTRYPOINT="train.py"
  fi
fi

if [[ -z "${COLOCATE+x}" ]]; then
  if [[ "${ENABLE_ASYNC_TRAIN}" == "true" ]]; then
    COLOCATE="false"
  else
    COLOCATE="true"
  fi
fi
case "${COLOCATE}" in
  true|false) ;;
  *) echo "[ERROR] COLOCATE must be true or false, got: ${COLOCATE}" >&2; exit 1 ;;
esac
if [[ "${ENABLE_ASYNC_TRAIN}" == "true" && "${COLOCATE}" == "true" ]]; then
  echo "[ERROR] ENABLE_ASYNC_TRAIN=true requires COLOCATE=false because train_async.py does not support colocation" >&2
  exit 1
fi
if [[ "${TRAIN_ENTRYPOINT}" == "train_async.py" && "${COLOCATE}" == "true" ]]; then
  echo "[ERROR] TRAIN_ENTRYPOINT=train_async.py requires COLOCATE=false" >&2
  exit 1
fi

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] required file missing: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "[ERROR] required directory missing: $1" >&2; exit 1; }
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "[ERROR] ${name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

require_bool() {
  local name="$1"
  local value="$2"
  case "${value}" in
    true|false) ;;
    *) echo "[ERROR] ${name} must be true or false, got: ${value}" >&2; exit 1 ;;
  esac
}

require_positive_int "PROMPT_MAX_LENGTH" "${PROMPT_MAX_LENGTH}"
require_positive_int "COMPLETION_MAX_LENGTH" "${COMPLETION_MAX_LENGTH}"
require_positive_int "G1_MAX_PROMPT_LABEL_LEN" "${G1_MAX_PROMPT_LABEL_LEN}"
require_positive_int "ROLLOUT_MAX_CONTEXT_LEN" "${ROLLOUT_MAX_CONTEXT_LEN}"
require_positive_int "ROLLOUT_MAX_RESPONSE_LEN" "${ROLLOUT_MAX_RESPONSE_LEN}"
require_positive_int "G1_CONTEXT_LENGTH" "${G1_CONTEXT_LENGTH}"
require_positive_int "G1_GENERATE_LENGTH" "${G1_GENERATE_LENGTH}"
require_positive_int "G1_STRIDE" "${G1_STRIDE}"
require_positive_int "G1_RESPONSE_LENGTH" "${G1_RESPONSE_LENGTH}"
require_bool "G1_FILTER_TRAIN_DATA" "${G1_FILTER_TRAIN_DATA}"
require_bool "G1_USE_EBFT_LOSS" "${G1_USE_EBFT_LOSS}"
require_bool "G1_QA_MASKING" "${G1_QA_MASKING}"
require_bool "USE_WHITENING" "${USE_WHITENING}"
require_bool "ENABLE_SLIME_EVAL" "${ENABLE_SLIME_EVAL}"
case "${G1_EBFT_LOGPROB_INDEXING}" in
  ""|standard_next_token|strict_block_source) ;;
  *) echo "[ERROR] G1_EBFT_LOGPROB_INDEXING must be empty, standard_next_token, or strict_block_source, got: ${G1_EBFT_LOGPROB_INDEXING}" >&2; exit 1 ;;
esac

if (( G1_MAX_PROMPT_LABEL_LEN != PROMPT_MAX_LENGTH )); then
  echo "[ERROR] G1_MAX_PROMPT_LABEL_LEN=${G1_MAX_PROMPT_LABEL_LEN} must match PROMPT_MAX_LENGTH=${PROMPT_MAX_LENGTH}" >&2
  exit 1
fi
if (( ROLLOUT_MAX_RESPONSE_LEN != COMPLETION_MAX_LENGTH )); then
  echo "[ERROR] ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN} must match COMPLETION_MAX_LENGTH=${COMPLETION_MAX_LENGTH}" >&2
  exit 1
fi
if (( G1_RESPONSE_LENGTH != COMPLETION_MAX_LENGTH )); then
  echo "[ERROR] G1_RESPONSE_LENGTH=${G1_RESPONSE_LENGTH} must match COMPLETION_MAX_LENGTH=${COMPLETION_MAX_LENGTH}" >&2
  exit 1
fi
if (( PROMPT_MAX_LENGTH + COMPLETION_MAX_LENGTH > ROLLOUT_MAX_CONTEXT_LEN )); then
  echo "[ERROR] PROMPT_MAX_LENGTH + COMPLETION_MAX_LENGTH (${PROMPT_MAX_LENGTH}+${COMPLETION_MAX_LENGTH}) must be <= ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN}" >&2
  exit 1
fi
if (( G1_RESPONSE_LENGTH % G1_GENERATE_LENGTH != 0 )); then
  echo "[ERROR] G1_RESPONSE_LENGTH=${G1_RESPONSE_LENGTH} must be divisible by G1_GENERATE_LENGTH=${G1_GENERATE_LENGTH}" >&2
  exit 1
fi
G1_NUM_BLOCKS=$((G1_RESPONSE_LENGTH / G1_GENERATE_LENGTH))
if [[ -z "${G1_PROMPT_LENGTH}" ]]; then
  G1_PROMPT_LENGTH=$(((G1_NUM_BLOCKS - 1) * G1_STRIDE + G1_GENERATE_LENGTH + G1_CONTEXT_LENGTH))
fi
require_positive_int "G1_PROMPT_LENGTH" "${G1_PROMPT_LENGTH}"
EXPECTED_G1_RESPONSE_LENGTH=$((((G1_PROMPT_LENGTH - G1_GENERATE_LENGTH - G1_CONTEXT_LENGTH) / G1_STRIDE + 1) * G1_GENERATE_LENGTH))
if (( G1_PROMPT_LENGTH < PROMPT_MAX_LENGTH )); then
  echo "[ERROR] G1_PROMPT_LENGTH=${G1_PROMPT_LENGTH} must be >= PROMPT_MAX_LENGTH=${PROMPT_MAX_LENGTH}" >&2
  exit 1
fi
if (( G1_PROMPT_LENGTH < G1_GENERATE_LENGTH + G1_CONTEXT_LENGTH )); then
  echo "[ERROR] G1_PROMPT_LENGTH=${G1_PROMPT_LENGTH} is too small for G1_CONTEXT_LENGTH=${G1_CONTEXT_LENGTH} and G1_GENERATE_LENGTH=${G1_GENERATE_LENGTH}" >&2
  exit 1
fi
if (( (G1_PROMPT_LENGTH - G1_GENERATE_LENGTH - G1_CONTEXT_LENGTH) % G1_STRIDE != 0 )); then
  echo "[ERROR] G1_PROMPT_LENGTH=${G1_PROMPT_LENGTH} is not compatible with context=${G1_CONTEXT_LENGTH}, generate=${G1_GENERATE_LENGTH}, stride=${G1_STRIDE}" >&2
  exit 1
fi
if (( EXPECTED_G1_RESPONSE_LENGTH != G1_RESPONSE_LENGTH )); then
  echo "[ERROR] G1 geometry gives response=${EXPECTED_G1_RESPONSE_LENGTH}, expected ${G1_RESPONSE_LENGTH}" >&2
  exit 1
fi

require_file "${SLIME_ROOT}/${TRAIN_ENTRYPOINT}"
require_dir "${MEGATRON_PATH}"
require_dir "${MODEL_PATH}"
require_dir "${REF_LOAD}"
require_file "${RAY_BIN}"
require_file "${SLIME_TRAIN_DATA}"

if [[ "${G1_FILTER_TRAIN_DATA}" == "true" ]]; then
  if [[ -s "${G1_FILTERED_SLIME_TRAIN_DATA}" ]]; then
    echo "[data] using existing filtered train data: ${G1_FILTERED_SLIME_TRAIN_DATA}"
  else
    "${PYTHON_BIN}" "${SLIME_ROOT}/exper_scripts/smoketest/filter_g1_prompt_length.py" \
      --input "${SLIME_TRAIN_DATA}" \
      --output "${G1_FILTERED_SLIME_TRAIN_DATA}" \
      --tokenizer "${HF_CHECKPOINT}" \
      --max-prompt-label-len "${G1_MAX_PROMPT_LABEL_LEN}" \
      --apply-chat-template
  fi
  SLIME_TRAIN_DATA="${G1_FILTERED_SLIME_TRAIN_DATA}"
fi

if [[ "${ENABLE_SLIME_EVAL}" == "true" ]]; then
  require_file "${HUMANEVAL_SLIME_EVAL_DATA}"
  if [[ -s "${MBPP_EVAL_DATA}" && ! -s "${MBPP_SLIME_EVAL_DATA}" ]]; then
    "${PYTHON_BIN}" "${SLIME_ROOT}/scripts/diff_dataset/prepare_slime_jsonl.py" \
      --input "${MBPP_EVAL_DATA}" \
      --output "${MBPP_SLIME_EVAL_DATA}" \
      --input-key question \
      --label-key answer
  fi
  require_file "${MBPP_SLIME_EVAL_DATA}"
fi

NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",") if x.strip()]))
PY
)"
require_positive_int "NUM_GPUS" "${NUM_GPUS}"

if [[ -z "${ACTOR_NUM_GPUS_PER_NODE+x}" ]]; then
  if [[ "${ENABLE_ASYNC_TRAIN}" == "true" ]]; then
    if (( NUM_GPUS >= 8 )); then
      ACTOR_NUM_GPUS_PER_NODE=4
    else
      ACTOR_NUM_GPUS_PER_NODE=$((NUM_GPUS / 2))
      if (( ACTOR_NUM_GPUS_PER_NODE < 1 )); then
        ACTOR_NUM_GPUS_PER_NODE=1
      fi
    fi
  else
    ACTOR_NUM_GPUS_PER_NODE="${NUM_GPUS}"
  fi
fi
require_positive_int "ACTOR_NUM_GPUS_PER_NODE" "${ACTOR_NUM_GPUS_PER_NODE}"

if [[ "${COLOCATE}" == "false" ]]; then
  if [[ -z "${ROLLOUT_NUM_GPUS+x}" ]]; then
    ROLLOUT_NUM_GPUS=$((NUM_GPUS - ACTOR_NUM_GPUS_PER_NODE))
  fi
  require_positive_int "ROLLOUT_NUM_GPUS" "${ROLLOUT_NUM_GPUS}"
  if (( ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS > NUM_GPUS )); then
    echo "[ERROR] ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} + ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be <= NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
  fi
else
  ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${ACTOR_NUM_GPUS_PER_NODE}}"
fi
if [[ "${SGLANG_STABLE_ROLLOUT_MODE}" == "true" || "${SGLANG_STABLE_ROLLOUT_MODE}" == "1" ]]; then
  if [[ "${COLOCATE}" == "false" && -z "${USER_SET_ROLLOUT_NUM_GPUS_PER_ENGINE}" ]]; then
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS}"
  fi
fi
require_positive_int "ROLLOUT_NUM_GPUS_PER_ENGINE" "${ROLLOUT_NUM_GPUS_PER_ENGINE}"

PARALLEL_GROUP_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( PARALLEL_GROUP_SIZE <= 0 )); then
  echo "[ERROR] TP*PP*CP must be positive, got ${PARALLEL_GROUP_SIZE}" >&2
  exit 1
fi
if (( ACTOR_NUM_GPUS_PER_NODE % PARALLEL_GROUP_SIZE != 0 )); then
  echo "[ERROR] ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} must be divisible by TP*PP*CP=${PARALLEL_GROUP_SIZE}" >&2
  exit 1
fi
if [[ "${COLOCATE}" == "false" && $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]]; then
  echo "[ERROR] ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
  exit 1
fi
if [[ "${SGLANG_DIRECT_WORKER_MODE}" == "true" && "${COLOCATE}" == "false" && "${ROLLOUT_NUM_GPUS}" != "${ROLLOUT_NUM_GPUS_PER_ENGINE}" ]]; then
  echo "[ERROR] SGLANG_DIRECT_WORKER_MODE=true requires exactly one rollout worker; set ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS} or SGLANG_STABLE_ROLLOUT_MODE=true." >&2
  exit 1
fi

DP_SIZE=$((ACTOR_NUM_GPUS_PER_NODE / PARALLEL_GROUP_SIZE))
if (( ROLLOUT_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "[ERROR] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}" "${LOAD_PATH}" "${SAVE_PATH}" "${RAY_TMPDIR}"

MODEL_ARGS=(
  --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec
  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --num-attention-heads 8
  --num-query-groups 2
  --kv-channels 256
  --num-layers 24
  --hidden-size 2048
  --ffn-hidden-size 6144
  --use-gated-attention
  --normalization RMSNorm
  --apply-layernorm-1p
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --rotary-percent 0.25
  --swiglu
  --vocab-size 248320
  --rotary-base 10000000
  --attention-output-gate
)

CMD=(
  "${PYTHON_BIN}" "${SLIME_ROOT}/${TRAIN_ENTRYPOINT}"
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MODEL_PATH}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_PATH}"
  --save "${SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
  --prompt-data "${SLIME_TRAIN_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type deepscaler
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
  --advantage-estimator g1
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
  --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity "${RECOMPUTE_GRANULARITY}"
  --recompute-method "${RECOMPUTE_METHOD}"
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --attention-dropout "${ATTENTION_DROPOUT}"
  --hidden-dropout "${HIDDEN_DROPOUT}"
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend "${ATTENTION_BACKEND}"
  --custom-generate-function-path slime.rollout.g1_embedding.generate_fixed_length_for_g1
  --alignment-rew-coef "${ALIGNMENT_REW_COEF}"
  --diversity-rew-coef "${DIVERSITY_REW_COEF}"
  --g1-prompt-length "${G1_PROMPT_LENGTH}"
  --g1-context-length "${G1_CONTEXT_LENGTH}"
  --g1-generate-length "${G1_GENERATE_LENGTH}"
  --g1-stride "${G1_STRIDE}"
  --g1-response-length "${G1_RESPONSE_LENGTH}"
  --g1-hidden-state-method "${G1_HIDDEN_STATE_METHOD}"
  --g1-tokenizer-path "${HF_CHECKPOINT}"
  --g1-embedding-source "${G1_EMBEDDING_SOURCE}"
  --g1-reward-location "${G1_REWARD_LOCATION}"
  --g1-megatron-ref-forward-mode "${G1_REF_FORWARD_MODE}"
  --g1-ce-loss-coef "${G1_CE_LOSS_COEF}"
)
if [[ -n "${G1_EBFT_LOGPROB_INDEXING}" ]]; then
  CMD+=(--g1-ebft-logprob-indexing "${G1_EBFT_LOGPROB_INDEXING}")
fi
if [[ "${USE_WHITENING}" == "true" ]]; then
  CMD+=(--use-whitening)
fi
if [[ "${G1_USE_EBFT_LOSS}" == "true" ]]; then
  CMD+=(--g1-use-ebft-loss)
fi
if [[ "${G1_QA_MASKING}" == "true" ]]; then
  CMD+=(--g1-qa-masking)
fi
if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "true" ]]; then
  CMD+=(--sglang-disable-cuda-graph)
fi
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]]; then
  CMD+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${SGLANG_ATTENTION_BACKEND}" ]]; then
  CMD+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi
if [[ -n "${SGLANG_SAMPLING_BACKEND}" ]]; then
  CMD+=(--sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND}")
fi
if [[ "${SGLANG_ROUTER_DISABLE_CIRCUIT_BREAKER}" == "true" ]]; then
  CMD+=(--router-disable-circuit-breaker)
fi
if [[ -n "${SGLANG_ROUTER_HEALTH_CHECK_ENDPOINT}" ]]; then
  CMD+=(--router-health-check-endpoint "${SGLANG_ROUTER_HEALTH_CHECK_ENDPOINT}")
fi
if [[ "${SGLANG_DIRECT_WORKER_MODE}" == "true" ]]; then
  CMD+=(--sglang-direct-worker-mode)
fi
if [[ "${SKIP_INITIAL_UPDATE_WEIGHTS}" == "true" ]]; then
  CMD+=(--skip-initial-update-weights)
fi
if [[ -n "${NUM_ROLLOUT}" ]]; then
  CMD+=(--num-rollout "${NUM_ROLLOUT}")
else
  CMD+=(--num-epoch "${NUM_EPOCH}")
fi
if [[ "${COLOCATE}" == "true" ]]; then
  CMD+=(--colocate)
else
  CMD+=(--rollout-num-gpus "${ROLLOUT_NUM_GPUS}")
fi
if [[ "${G1_APPLY_DENSE_ATTENTION_MASK}" == "true" ]]; then
  CMD+=(--g1-megatron-ref-apply-dense-attention-mask)
fi
if [[ "${ENABLE_SLIME_EVAL}" == "true" ]]; then
  CMD+=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data mbpp "${MBPP_SLIME_EVAL_DATA}" humaneval "${HUMANEVAL_SLIME_EVAL_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-top-p 1
  )
fi

write_argv_artifact() {
  {
    printf "%q " "${CMD[@]}"
    printf "\n"
  } >"${ARTIFACT_DIR}/argv.sh"
}

write_run_context_artifact() {
  cat >"${ARTIFACT_DIR}/run_context.env" <<EOF
RUN_NAME=${RUN_NAME}
LOAD_PATH=${LOAD_PATH}
SAVE_PATH=${SAVE_PATH}
SLIME_ROOT=${SLIME_ROOT}
LAUNCHER_SLIME_ROOT=${LAUNCHER_SLIME_ROOT}
PROJECT_ROOT=${PROJECT_ROOT}
SLIME_DATA_ROOT=${SLIME_DATA_ROOT}
PREPARED_DATA_DIR=${PREPARED_DATA_DIR}
SLIME_TRAIN_DATA=${SLIME_TRAIN_DATA}
NUM_GPUS=${NUM_GPUS}
DP_SIZE=${DP_SIZE}
NUM_EPOCH=${NUM_EPOCH}
NUM_ROLLOUT=${NUM_ROLLOUT}
PROMPT_MAX_LENGTH=${PROMPT_MAX_LENGTH}
COMPLETION_MAX_LENGTH=${COMPLETION_MAX_LENGTH}
ENABLE_ASYNC_TRAIN=${ENABLE_ASYNC_TRAIN}
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}
COLOCATE=${COLOCATE}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P}
SGLANG_STABLE_ROLLOUT_MODE=${SGLANG_STABLE_ROLLOUT_MODE}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND}
SGLANG_SAMPLING_BACKEND=${SGLANG_SAMPLING_BACKEND}
SGLANG_ROUTER_DISABLE_CIRCUIT_BREAKER=${SGLANG_ROUTER_DISABLE_CIRCUIT_BREAKER}
SGLANG_ROUTER_HEALTH_CHECK_ENDPOINT=${SGLANG_ROUTER_HEALTH_CHECK_ENDPOINT}
SGLANG_DIRECT_WORKER_MODE=${SGLANG_DIRECT_WORKER_MODE}
SLIME_SGLANG_HEALTH_TIMEOUT=${SLIME_SGLANG_HEALTH_TIMEOUT}
SLIME_SGLANG_HEALTH_MAX_WAIT=${SLIME_SGLANG_HEALTH_MAX_WAIT}
SLIME_ROUTER_WORKER_WAIT_TIMEOUT=${SLIME_ROUTER_WORKER_WAIT_TIMEOUT}
SLIME_ROUTER_WORKER_WAIT_INTERVAL=${SLIME_ROUTER_WORKER_WAIT_INTERVAL}
SLIME_ROUTER_WORKER_REQUEST_TIMEOUT=${SLIME_ROUTER_WORKER_REQUEST_TIMEOUT}
SLIME_ROUTER_DISABLE_CIRCUIT_BREAKER=${SLIME_ROUTER_DISABLE_CIRCUIT_BREAKER}
SLIME_HTTP_REQUEST_TIMEOUT=${SLIME_HTTP_REQUEST_TIMEOUT}
NO_PROXY=${NO_PROXY}
no_proxy=${no_proxy}
LR=${LR}
WEIGHT_DECAY=${WEIGHT_DECAY}
ADAM_BETA1=${ADAM_BETA1}
ADAM_BETA2=${ADAM_BETA2}
EPS_CLIP=${EPS_CLIP}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH}
ENTROPY_COEF=${ENTROPY_COEF}
UPDATE_WEIGHTS_INTERVAL=${UPDATE_WEIGHTS_INTERVAL}
SKIP_INITIAL_UPDATE_WEIGHTS=${SKIP_INITIAL_UPDATE_WEIGHTS}
G1_USE_EBFT_LOSS=${G1_USE_EBFT_LOSS}
G1_APPLY_DENSE_ATTENTION_MASK=${G1_APPLY_DENSE_ATTENTION_MASK}
G1_QA_MASKING=${G1_QA_MASKING}
G1_CE_LOSS_COEF=${G1_CE_LOSS_COEF}
G1_MAX_PROMPT_LABEL_LEN=${G1_MAX_PROMPT_LABEL_LEN}
G1_PROMPT_LENGTH=${G1_PROMPT_LENGTH}
G1_NUM_BLOCKS=${G1_NUM_BLOCKS}
G1_CONTEXT_LENGTH=${G1_CONTEXT_LENGTH}
G1_GENERATE_LENGTH=${G1_GENERATE_LENGTH}
G1_STRIDE=${G1_STRIDE}
G1_RESPONSE_LENGTH=${G1_RESPONSE_LENGTH}
G1_HIDDEN_STATE_METHOD=${G1_HIDDEN_STATE_METHOD}
G1_EMBEDDING_SOURCE=${G1_EMBEDDING_SOURCE}
G1_REWARD_LOCATION=${G1_REWARD_LOCATION}
G1_REF_FORWARD_MODE=${G1_REF_FORWARD_MODE}
G1_EBFT_LOGPROB_INDEXING=${G1_EBFT_LOGPROB_INDEXING}
USE_WHITENING=${USE_WHITENING}
ALIGNMENT_REW_COEF=${ALIGNMENT_REW_COEF}
DIVERSITY_REW_COEF=${DIVERSITY_REW_COEF}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT}
HIDDEN_DROPOUT=${HIDDEN_DROPOUT}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS}
ENABLE_SLIME_EVAL=${ENABLE_SLIME_EVAL}
EVAL_INTERVAL=${EVAL_INTERVAL}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT}
EVAL_MAX_PROMPT_LEN=${EVAL_MAX_PROMPT_LEN}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN}
EOF
}

write_argv_artifact || echo "[WARN] failed to write ${ARTIFACT_DIR}/argv.sh; continuing without argv artifact" >&2
if write_run_context_artifact; then
  cp "${ARTIFACT_DIR}/run_context.env" "${ARTIFACT_DIR}/hyperparams.env" || \
    echo "[WARN] failed to copy ${ARTIFACT_DIR}/hyperparams.env; continuing without hyperparams artifact" >&2
else
  echo "[WARN] failed to write ${ARTIFACT_DIR}/run_context.env; continuing without run context artifact" >&2
fi

echo "[main-test] OpenRLHF-G1 aligned Slime run"
echo "[main-test] RUN_NAME=${RUN_NAME}"
echo "[main-test] NUM_EPOCH=${NUM_EPOCH} NUM_ROLLOUT=${NUM_ROLLOUT:-auto} ENABLE_SLIME_EVAL=${ENABLE_SLIME_EVAL} ENABLE_ASYNC_TRAIN=${ENABLE_ASYNC_TRAIN}"
echo "[main-test] WEIGHT_DECAY=${WEIGHT_DECAY} ADAM_BETA2=${ADAM_BETA2} EPS_CLIP_HIGH=${EPS_CLIP_HIGH}"
echo "[preflight] NUM_GPUS=${NUM_GPUS} ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} COLOCATE=${COLOCATE} TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}"
echo "[preflight] SLIME_ROOT=${SLIME_ROOT}"
echo "[preflight] TRAIN_ENTRY=${SLIME_ROOT}/${TRAIN_ENTRYPOINT}"
echo "[preflight] SGLANG_STABLE_ROLLOUT_MODE=${SGLANG_STABLE_ROLLOUT_MODE} ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE} SGLANG_DIRECT_WORKER_MODE=${SGLANG_DIRECT_WORKER_MODE}"
echo "[preflight] TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} ACTOR_DP=${DP_SIZE}"
echo "[preflight] G1_EBFT_LOGPROB_INDEXING=${G1_EBFT_LOGPROB_INDEXING:-launcher-default}"
echo "[submit] command:"
printf "%q " "${CMD[@]}"
echo
echo "[artifact] ${ARTIFACT_DIR}"

if [[ "${PRINT_ONLY:-0}" == "1" || "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

"${RAY_BIN}" stop --force 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
sleep 3

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${RAY_BIN}" start --head \
  --node-ip-address 127.0.0.1 \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir "${RAY_TMPDIR}"

RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<'PY'
import json, os
keys = [
    "PYTHONPATH", "PATH", "VIRTUAL_ENV", "CUDA_HOME", "LD_LIBRARY_PATH",
    "CUDA_DEVICE_MAX_CONNECTIONS", "HF_HOME", "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_XET", "TOKENIZERS_PARALLELISM",
    "RAY_TMPDIR", "PYTHONUNBUFFERED", "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK",
    "SLIME_SGLANG_HEALTH_TIMEOUT", "SLIME_SGLANG_HEALTH_MAX_WAIT",
    "SLIME_ROUTER_WORKER_WAIT_TIMEOUT", "SLIME_ROUTER_WORKER_WAIT_INTERVAL",
    "SLIME_ROUTER_WORKER_REQUEST_TIMEOUT", "SLIME_ROUTER_DISABLE_CIRCUIT_BREAKER",
    "SLIME_HTTP_REQUEST_TIMEOUT", "NO_PROXY", "no_proxy",
]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))
PY
)"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${RAY_BIN}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${CMD[@]}" \
  2>&1 | tee "${DRIVER_LOG}"
SUBMIT_STATUS=${PIPESTATUS[0]}
set -e

printf "%s\n" "${SUBMIT_STATUS}" >"${ARTIFACT_DIR}/ray_job_exit_status.txt"
if [[ -e "${RAY_TMPDIR}/session_latest/logs" ]]; then
  tar -C "${RAY_TMPDIR}/session_latest" -czf "${ARTIFACT_DIR}/ray_session_latest_logs.tgz" logs || true
fi
exit "${SUBMIT_STATUS}"
