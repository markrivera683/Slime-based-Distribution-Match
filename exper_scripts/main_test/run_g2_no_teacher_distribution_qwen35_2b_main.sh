#!/usr/bin/env bash
# Full standalone G2 no-teacher-distribution Slime/Megatron run for Qwen3.5-2B.
#
# Deployment layouts:
#   single_node  - local Ray head with student training only
#   two_node     - 2 Ray nodes, 8 GPU each, no teacher service
#
# Two-node Ray cluster (8+8 GPU) — one orchestrator:
#   bash exper_scripts/main_test/run_g2_no_teacher_distribution_qwen35_2b_2node.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 0. Runtime paths
# ---------------------------------------------------------------------------
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EBFT_DEPS_ROOT="${EBFT_DEPS_ROOT:-${EBFT_ROOT:-${SLIME_ROOT}}}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/slime_runtime/Megatron-LM}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"

# ---------------------------------------------------------------------------
# 1. Model and dataset
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-2B}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_PATH}}"
REF_LOAD="${REF_LOAD:-/mnt/data/models/Megatron_convert_models/Qwen3.5-2B_torch_dist}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
SLIME_TRAIN_DATA="${SLIME_TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_slime_qa_100k.jsonl}"

PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-384}"
COMPLETION_MAX_LENGTH="${COMPLETION_MAX_LENGTH:-376}"
G1_FILTER_TRAIN_DATA="${G1_FILTER_TRAIN_DATA:-true}"
G1_MAX_PROMPT_LABEL_LEN="${G1_MAX_PROMPT_LABEL_LEN:-${PROMPT_MAX_LENGTH}}"
G1_FILTERED_SLIME_TRAIN_DATA="${G1_FILTERED_SLIME_TRAIN_DATA:-${SLIME_TRAIN_DATA%.jsonl}_g1_prompt${G1_MAX_PROMPT_LABEL_LEN}.jsonl}"

MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
MBPP_SLIME_EVAL_DATA="${MBPP_SLIME_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa_slime.jsonl}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
HUMANEVAL_SLIME_EVAL_DATA="${HUMANEVAL_SLIME_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa_slime.jsonl}"

# ---------------------------------------------------------------------------
# 2. Parallelism, deploy layout, and resource layout
# ---------------------------------------------------------------------------
# DEPLOY_LAYOUT:
#   single_node     - default local Ray head
#   two_node        - external 2-node Ray cluster started by the DLC/orchestrator script
DEPLOY_LAYOUT="${DEPLOY_LAYOUT:-single_node}"
DEPLOY_ROLE="${DEPLOY_ROLE:-trainer}"
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-127.0.0.1}"
USE_EXISTING_RAY="${USE_EXISTING_RAY:-false}"

# Default 8-GPU split in sync non-colocate mode: actor 2, critic 2, rollout 4.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

# ---------------------------------------------------------------------------
# 3. Training horizon and batch geometry
# ---------------------------------------------------------------------------
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
NUM_EPOCH="${NUM_EPOCH:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"

# ---------------------------------------------------------------------------
# 4. SGLang rollout generation
# ---------------------------------------------------------------------------
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-1024}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-${COMPLETION_MAX_LENGTH}}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"

# ---------------------------------------------------------------------------
# 5. Optimizer and G2 actor/critic knobs
# ---------------------------------------------------------------------------
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"
ZERO_STAGE="${ZERO_STAGE:-3}"

# ---------------------------------------------------------------------------
# 6. G2 no-teacher distribution matching
# ---------------------------------------------------------------------------
G2_OPD_MODE="${G2_OPD_MODE:-cf_l1oo_no_teacher_distribution}"
CF_TARGET_MODE="${CF_TARGET_MODE:-single}"
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-1}"
CF_TARGET_STD="${CF_TARGET_STD:-0.0}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-${N_SAMPLES_PER_PROMPT:-4}}"

# ---------------------------------------------------------------------------
# 7. G1 embedding/reward path used by G2
# ---------------------------------------------------------------------------
G1_USE_EBFT_LOSS="${G1_USE_EBFT_LOSS:-true}"
G1_APPLY_DENSE_ATTENTION_MASK="${G1_APPLY_DENSE_ATTENTION_MASK:-false}"
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
USE_WHITENING="${USE_WHITENING:-true}"

# ---------------------------------------------------------------------------
# 8. Megatron execution/performance
# ---------------------------------------------------------------------------
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

# ---------------------------------------------------------------------------
# 9. Online eval and optional post-eval
# ---------------------------------------------------------------------------
ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL:-false}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-512}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"

ENABLE_G2_POST_EVAL="${ENABLE_G2_POST_EVAL:-false}"
CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT:-${EBFT_DEPS_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"
CODE_BENCHMARK_PYTHON_BIN="${CODE_BENCHMARK_PYTHON_BIN:-${PYTHON_BIN}}"
CODE_BENCHMARK_BACKEND="${CODE_BENCHMARK_BACKEND:-hf}"
CODE_BENCHMARKS="${CODE_BENCHMARKS:-humaneval,mbpp}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-0}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS:-1024}"
CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.6}"
CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-1.0}"
CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY:-1.0}"
CODE_EVAL_N_SAMPLES="${CODE_EVAL_N_SAMPLES:-16}"
CODE_EVAL_PASSK_LIST="${CODE_EVAL_PASSK_LIST:-1,4,16}"
CODE_EVAL_GREEDY_BATCH_SIZE="${CODE_EVAL_GREEDY_BATCH_SIZE:-16}"
CODE_EVAL_SAMPLE_BATCH_SIZE="${CODE_EVAL_SAMPLE_BATCH_SIZE:-4}"
CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS:-10}"
MBPP_EVAL_CONFIG="${MBPP_EVAL_CONFIG:-}"
MBPP_EVAL_SPLIT="${MBPP_EVAL_SPLIT:-test}"
HUMANEVAL_EVAL_SPLIT="${HUMANEVAL_EVAL_SPLIT:-test}"

# ---------------------------------------------------------------------------
# 10. Checkpoint/artifacts/Ray
# ---------------------------------------------------------------------------
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/slime/outputs}"
RUN_NAME="${RUN_NAME:-g2_no_teacher_distribution_qwen35_2b_main_$(date +%m%d_%H%M%S)}"
LOAD_PATH="${LOAD_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/mcore}"
SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
CRITIC_SAVE_PATH="${CRITIC_SAVE_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/critic_mcore}"
SAVE_HF_PATH_TEMPLATE="${SAVE_HF_PATH_TEMPLATE:-${OUTPUT_ROOT}/${RUN_NAME}/hf/rollout_{rollout_id}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/artifacts}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_g2_no_teacher_distribution_main_${USER:-user}}"
RAY_ADDRESS="http://${RAY_NODE_IP_ADDRESS}:${RAY_DASHBOARD_PORT}"
DRIVER_LOG="${ARTIFACT_DIR}/ray_job_driver.log"
POST_EVAL_LOG="${ARTIFACT_DIR}/post_eval.log"

if [[ -f "${SLIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ENV_FILE}"
fi

export PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}:${EBFT_DEPS_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export RAY_TMPDIR
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"

# two_node student follows G1 main: async Megatron + SGLang rollout by default.
if [[ -z "${ENABLE_ASYNC_TRAIN+x}" ]]; then
  if [[ "${DEPLOY_LAYOUT}" == "two_node" ]]; then
    ENABLE_ASYNC_TRAIN="true"
  else
    ENABLE_ASYNC_TRAIN="false"
  fi
fi
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
    COLOCATE="false"
  fi
fi
# G2 non-async still uses dedicated rollout GPUs (actor+critic+rollout), not colocate.
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
require_positive_int "CF_TEACHER_N_SAMPLES" "${CF_TEACHER_N_SAMPLES}"
require_positive_int "CF_TARGET_NUM_REFS" "${CF_TARGET_NUM_REFS}"
require_bool "G1_FILTER_TRAIN_DATA" "${G1_FILTER_TRAIN_DATA}"
require_bool "G1_USE_EBFT_LOSS" "${G1_USE_EBFT_LOSS}"
require_bool "USE_WHITENING" "${USE_WHITENING}"
require_bool "ENABLE_SLIME_EVAL" "${ENABLE_SLIME_EVAL}"
require_bool "ENABLE_G2_POST_EVAL" "${ENABLE_G2_POST_EVAL}"

case "${DEPLOY_LAYOUT}" in
  single_node|two_node) ;;
  *) echo "[ERROR] DEPLOY_LAYOUT must be single_node or two_node, got: ${DEPLOY_LAYOUT}" >&2; exit 1 ;;
esac
if [[ "${CF_TARGET_MODE}" != "single" ]]; then
  echo "[ERROR] G2 no-teacher distribution requires CF_TARGET_MODE=single." >&2
  exit 1
fi
if [[ "${G1_EMBEDDING_SOURCE}" != "megatron_ref" || "${G1_REWARD_LOCATION}" != "trainer" ]]; then
  echo "[ERROR] G2 cf_l1oo requires G1_EMBEDDING_SOURCE=megatron_ref and G1_REWARD_LOCATION=trainer." >&2
  exit 1
fi
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
if [[ "${ENABLE_G2_POST_EVAL}" == "true" && -z "${POST_EVAL_MODEL_PATH:-}" && -z "${NUM_ROLLOUT}" ]]; then
  echo "[ERROR] ENABLE_G2_POST_EVAL=true requires NUM_ROLLOUT or POST_EVAL_MODEL_PATH so the final HF checkpoint can be located." >&2
  exit 1
fi

require_file "${SLIME_ROOT}/${TRAIN_ENTRYPOINT}"
require_dir "${MEGATRON_PATH}"
require_dir "${MODEL_PATH}"
require_dir "${REF_LOAD}"
require_file "${RAY_BIN}"
require_file "${SLIME_TRAIN_DATA}"

if [[ "${G1_FILTER_TRAIN_DATA}" == "true" ]]; then
  "${PYTHON_BIN}" "${SLIME_ROOT}/exper_scripts/smoketest/filter_g1_prompt_length.py" \
    --input "${SLIME_TRAIN_DATA}" \
    --output "${G1_FILTERED_SLIME_TRAIN_DATA}" \
    --tokenizer "${HF_CHECKPOINT}" \
    --max-prompt-label-len "${G1_MAX_PROMPT_LABEL_LEN}" \
    --apply-chat-template
  SLIME_TRAIN_DATA="${G1_FILTERED_SLIME_TRAIN_DATA}"
fi

if [[ "${ENABLE_SLIME_EVAL}" == "true" ]]; then
  require_file "${HUMANEVAL_SLIME_EVAL_DATA}"
  if [[ -s "${MBPP_EVAL_DATA}" && ! -s "${MBPP_SLIME_EVAL_DATA}" ]]; then
    "${PYTHON_BIN}" "${EBFT_DEPS_ROOT}/scripts/diff_dataset/prepare_slime_jsonl.py" \
      --input "${MBPP_EVAL_DATA}" \
      --output "${MBPP_SLIME_EVAL_DATA}" \
      --input-key question \
      --label-key answer
  fi
  require_file "${MBPP_SLIME_EVAL_DATA}"
fi
if [[ "${ENABLE_G2_POST_EVAL}" == "true" ]]; then
  require_file "${CODE_BENCHMARK_SCRIPT}"
  require_file "${CODE_BENCHMARK_PYTHON_BIN}"
  require_file "${MBPP_EVAL_DATA}"
  require_file "${HUMANEVAL_EVAL_DATA}"
fi

NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",") if x.strip()]))
PY
)"
require_positive_int "NUM_GPUS" "${NUM_GPUS}"
require_positive_int "TENSOR_MODEL_PARALLEL_SIZE" "${TENSOR_MODEL_PARALLEL_SIZE}"
require_positive_int "PIPELINE_MODEL_PARALLEL_SIZE" "${PIPELINE_MODEL_PARALLEL_SIZE}"
require_positive_int "CONTEXT_PARALLEL_SIZE" "${CONTEXT_PARALLEL_SIZE}"
require_positive_int "ROLLOUT_NUM_GPUS_PER_ENGINE" "${ROLLOUT_NUM_GPUS_PER_ENGINE}"

PARALLEL_GROUP_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( PARALLEL_GROUP_SIZE <= 0 )); then
  echo "[ERROR] TP*PP*CP must be positive, got ${PARALLEL_GROUP_SIZE}" >&2
  exit 1
fi

if [[ -z "${CRITIC_NUM_NODES+x}" ]]; then
  CRITIC_NUM_NODES=1
fi
require_positive_int "CRITIC_NUM_NODES" "${CRITIC_NUM_NODES}"

if [[ -z "${CRITIC_NUM_GPUS_PER_NODE+x}" ]]; then
  CRITIC_NUM_GPUS_PER_NODE="${PARALLEL_GROUP_SIZE}"
fi
require_positive_int "CRITIC_NUM_GPUS_PER_NODE" "${CRITIC_NUM_GPUS_PER_NODE}"
if (( CRITIC_NUM_GPUS_PER_NODE % PARALLEL_GROUP_SIZE != 0 )); then
  echo "[ERROR] CRITIC_NUM_GPUS_PER_NODE=${CRITIC_NUM_GPUS_PER_NODE} must be divisible by TP*PP*CP=${PARALLEL_GROUP_SIZE}" >&2
  exit 1
fi
CRITIC_TOTAL_GPUS=$((CRITIC_NUM_NODES * CRITIC_NUM_GPUS_PER_NODE))

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
  elif [[ "${COLOCATE}" == "true" ]]; then
    ACTOR_BUDGET=$((NUM_GPUS - CRITIC_TOTAL_GPUS))
    ACTOR_NUM_GPUS_PER_NODE=$(((ACTOR_BUDGET / PARALLEL_GROUP_SIZE) * PARALLEL_GROUP_SIZE))
  else
    ACTOR_NUM_GPUS_PER_NODE="${PARALLEL_GROUP_SIZE}"
  fi
fi
require_positive_int "ACTOR_NUM_GPUS_PER_NODE" "${ACTOR_NUM_GPUS_PER_NODE}"

if [[ "${COLOCATE}" == "false" ]]; then
  if [[ -z "${ROLLOUT_NUM_GPUS+x}" ]]; then
    ROLLOUT_NUM_GPUS=$((NUM_GPUS - ACTOR_NUM_GPUS_PER_NODE - CRITIC_TOTAL_GPUS))
    if (( ROLLOUT_NUM_GPUS <= 0 )); then
      echo "[ERROR] Default G2 GPU split leaves no rollout GPUs: NUM_GPUS=${NUM_GPUS}, ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}, CRITIC_TOTAL_GPUS=${CRITIC_TOTAL_GPUS}" >&2
      exit 1
    fi
  fi
  require_positive_int "ROLLOUT_NUM_GPUS" "${ROLLOUT_NUM_GPUS}"
  if (( ACTOR_NUM_GPUS_PER_NODE + CRITIC_TOTAL_GPUS + ROLLOUT_NUM_GPUS > NUM_GPUS )); then
    echo "[ERROR] ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} + CRITIC_TOTAL_GPUS=${CRITIC_TOTAL_GPUS} + ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be <= NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
  fi
else
  ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE + CRITIC_TOTAL_GPUS))}"
  if (( ACTOR_NUM_GPUS_PER_NODE + CRITIC_TOTAL_GPUS > NUM_GPUS )); then
    echo "[ERROR] COLOCATE=true requires ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} + CRITIC_TOTAL_GPUS=${CRITIC_TOTAL_GPUS} <= NUM_GPUS=${NUM_GPUS}; reduce actor/critic GPUs or disable colocation." >&2
    exit 1
  fi
fi

if (( ACTOR_NUM_GPUS_PER_NODE % PARALLEL_GROUP_SIZE != 0 )); then
  echo "[ERROR] ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} must be divisible by TP*PP*CP=${PARALLEL_GROUP_SIZE}" >&2
  exit 1
fi
if [[ "${COLOCATE}" == "false" && $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]]; then
  echo "[ERROR] ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
  exit 1
fi

DP_SIZE=$((ACTOR_NUM_GPUS_PER_NODE / PARALLEL_GROUP_SIZE))
if (( ROLLOUT_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "[ERROR] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}" "${LOAD_PATH}" "${SAVE_PATH}" "${CRITIC_SAVE_PATH}" "${RAY_TMPDIR}"

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
  --critic-num-nodes "${CRITIC_NUM_NODES}"
  --critic-num-gpus-per-node "${CRITIC_NUM_GPUS_PER_NODE}"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MODEL_PATH}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_PATH}"
  --save "${SAVE_PATH}"
  --critic-save "${CRITIC_SAVE_PATH}"
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
  --critic-lr "${CRITIC_LR}"
  --critic-lr-head "${CRITIC_LR_HEAD}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
  --advantage-estimator g1
  --distribution-reward-type cf_l1oo
  --cf-target-mode "${CF_TARGET_MODE}"
  --cf-target-num-refs "${CF_TARGET_NUM_REFS}"
  --cf-target-std "${CF_TARGET_STD}"
  --cf-target-seed "${CF_TARGET_SEED}"
  --cf-teacher-lambda "${CF_TEACHER_LAMBDA}"
  --cf-teacher-n-samples "${CF_TEACHER_N_SAMPLES}"
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
  --zero-stage "${ZERO_STAGE}"
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
if [[ -n "${NUM_ROLLOUT}" ]]; then
  require_positive_int "NUM_ROLLOUT" "${NUM_ROLLOUT}"
  CMD+=(--num-rollout "${NUM_ROLLOUT}")
else
  require_positive_int "NUM_EPOCH" "${NUM_EPOCH}"
  CMD+=(--num-epoch "${NUM_EPOCH}")
fi
if [[ "${ENABLE_G2_POST_EVAL}" == "true" ]]; then
  CMD+=(--save-hf "${SAVE_HF_PATH_TEMPLATE}")
fi
if [[ "${USE_WHITENING}" == "true" ]]; then
  CMD+=(--use-whitening)
fi
if [[ "${G1_USE_EBFT_LOSS}" == "true" ]]; then
  CMD+=(--g1-use-ebft-loss)
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

printf "%q " "${CMD[@]}" >"${ARTIFACT_DIR}/argv.sh"
printf "\n" >>"${ARTIFACT_DIR}/argv.sh"
cat >"${ARTIFACT_DIR}/run_context.env" <<EOF
RUN_NAME=${RUN_NAME}
DEPLOY_LAYOUT=${DEPLOY_LAYOUT}
DEPLOY_ROLE=${DEPLOY_ROLE}
RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS}
RAY_ADDRESS=${RAY_ADDRESS}
G2_OPD_MODE=${G2_OPD_MODE}
LOAD_PATH=${LOAD_PATH}
SAVE_PATH=${SAVE_PATH}
CRITIC_SAVE_PATH=${CRITIC_SAVE_PATH}
SAVE_HF_PATH_TEMPLATE=${SAVE_HF_PATH_TEMPLATE}
ARTIFACT_DIR=${ARTIFACT_DIR}
SLIME_ROOT=${SLIME_ROOT}
EBFT_DEPS_ROOT=${EBFT_DEPS_ROOT}
MEGATRON_PATH=${MEGATRON_PATH}
PYTHON_BIN=${PYTHON_BIN}
RAY_BIN=${RAY_BIN}
SLIME_TRAIN_DATA=${SLIME_TRAIN_DATA}
PROMPT_MAX_LENGTH=${PROMPT_MAX_LENGTH}
COMPLETION_MAX_LENGTH=${COMPLETION_MAX_LENGTH}
G1_FILTER_TRAIN_DATA=${G1_FILTER_TRAIN_DATA}
G1_MAX_PROMPT_LABEL_LEN=${G1_MAX_PROMPT_LABEL_LEN}
NUM_GPUS=${NUM_GPUS}
DP_SIZE=${DP_SIZE}
PARALLEL_GROUP_SIZE=${PARALLEL_GROUP_SIZE}
ENABLE_ASYNC_TRAIN=${ENABLE_ASYNC_TRAIN}
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}
COLOCATE=${COLOCATE}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}
CRITIC_NUM_NODES=${CRITIC_NUM_NODES}
CRITIC_NUM_GPUS_PER_NODE=${CRITIC_NUM_GPUS_PER_NODE}
CRITIC_TOTAL_GPUS=${CRITIC_TOTAL_GPUS}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
NUM_EPOCH=${NUM_EPOCH}
NUM_ROLLOUT=${NUM_ROLLOUT}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC}
LR=${LR}
WEIGHT_DECAY=${WEIGHT_DECAY}
ADAM_BETA1=${ADAM_BETA1}
ADAM_BETA2=${ADAM_BETA2}
EPS_CLIP=${EPS_CLIP}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH}
ENTROPY_COEF=${ENTROPY_COEF}
CRITIC_LR=${CRITIC_LR}
CRITIC_LR_HEAD=${CRITIC_LR_HEAD}
ZERO_STAGE=${ZERO_STAGE}
CF_TARGET_MODE=${CF_TARGET_MODE}
CF_TARGET_NUM_REFS=${CF_TARGET_NUM_REFS}
CF_TARGET_STD=${CF_TARGET_STD}
CF_TARGET_SEED=${CF_TARGET_SEED}
CF_TEACHER_LAMBDA=${CF_TEACHER_LAMBDA}
CF_TEACHER_N_SAMPLES=${CF_TEACHER_N_SAMPLES}
G1_USE_EBFT_LOSS=${G1_USE_EBFT_LOSS}
G1_APPLY_DENSE_ATTENTION_MASK=${G1_APPLY_DENSE_ATTENTION_MASK}
G1_CE_LOSS_COEF=${G1_CE_LOSS_COEF}
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
USE_WHITENING=${USE_WHITENING}
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
ENABLE_G2_POST_EVAL=${ENABLE_G2_POST_EVAL}
CODE_BENCHMARK_SCRIPT=${CODE_BENCHMARK_SCRIPT}
CODE_BENCHMARK_PYTHON_BIN=${CODE_BENCHMARK_PYTHON_BIN}
CODE_BENCHMARK_BACKEND=${CODE_BENCHMARK_BACKEND}
CODE_BENCHMARKS=${CODE_BENCHMARKS}
POST_EVAL_MAX_SAMPLES=${POST_EVAL_MAX_SAMPLES}
POST_EVAL_PROMPT_MAX_LEN=${POST_EVAL_PROMPT_MAX_LEN}
CODE_EVAL_MAX_NEW_TOKENS=${CODE_EVAL_MAX_NEW_TOKENS}
CODE_EVAL_TEMPERATURE=${CODE_EVAL_TEMPERATURE}
CODE_EVAL_TOP_P=${CODE_EVAL_TOP_P}
CODE_EVAL_REPETITION_PENALTY=${CODE_EVAL_REPETITION_PENALTY}
CODE_EVAL_N_SAMPLES=${CODE_EVAL_N_SAMPLES}
CODE_EVAL_PASSK_LIST=${CODE_EVAL_PASSK_LIST}
CODE_EVAL_GREEDY_BATCH_SIZE=${CODE_EVAL_GREEDY_BATCH_SIZE}
CODE_EVAL_SAMPLE_BATCH_SIZE=${CODE_EVAL_SAMPLE_BATCH_SIZE}
CODE_EVAL_TIMEOUT_SECONDS=${CODE_EVAL_TIMEOUT_SECONDS}
MBPP_EVAL_CONFIG=${MBPP_EVAL_CONFIG}
MBPP_EVAL_SPLIT=${MBPP_EVAL_SPLIT}
HUMANEVAL_EVAL_SPLIT=${HUMANEVAL_EVAL_SPLIT}
EOF

cp "${ARTIFACT_DIR}/run_context.env" "${ARTIFACT_DIR}/hyperparams.env"

echo "[main-test] G2 no-teacher-distribution Slime/Megatron run"
echo "[main-test] DEPLOY_LAYOUT=${DEPLOY_LAYOUT} DEPLOY_ROLE=${DEPLOY_ROLE} RAY_ADDRESS=${RAY_ADDRESS}"
echo "[main-test] RUN_NAME=${RUN_NAME}"
echo "[main-test] mode=${G2_OPD_MODE} NUM_EPOCH=${NUM_EPOCH} NUM_ROLLOUT=${NUM_ROLLOUT:-auto} ENABLE_SLIME_EVAL=${ENABLE_SLIME_EVAL} ENABLE_G2_POST_EVAL=${ENABLE_G2_POST_EVAL}"
echo "[main-test] cf_l1oo target=${CF_TARGET_MODE} lambda=${CF_TEACHER_LAMBDA} target_refs=${CF_TARGET_NUM_REFS}"
echo "[main-test] ebft_loss=${G1_USE_EBFT_LOSS} ce_loss_coef=${G1_CE_LOSS_COEF}"
echo "[preflight] NUM_GPUS=${NUM_GPUS} ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} CRITIC_NUM_NODES=${CRITIC_NUM_NODES} CRITIC_NUM_GPUS_PER_NODE=${CRITIC_NUM_GPUS_PER_NODE} ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} COLOCATE=${COLOCATE} TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}"
echo "[preflight] TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} ACTOR_DP=${DP_SIZE}"
echo "[submit] command:"
printf "%q " "${CMD[@]}"
echo
echo "[artifact] ${ARTIFACT_DIR}"

if [[ "${PRINT_ONLY:-0}" == "1" || "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${USE_EXISTING_RAY}" == "true" || "${USE_EXISTING_RAY}" == "1" ]]; then
  echo "[ray] using existing Ray cluster at ${RAY_ADDRESS}"
else
  "${RAY_BIN}" stop --force 2>/dev/null || true
  sleep 3

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${RAY_BIN}" start --head \
    --node-ip-address "${RAY_NODE_IP_ADDRESS}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --temp-dir "${RAY_TMPDIR}"
fi

RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<'PY'
import json, os
keys = [
    "PYTHONPATH", "PATH", "VIRTUAL_ENV", "CUDA_HOME", "LD_LIBRARY_PATH",
    "CUDA_DEVICE_MAX_CONNECTIONS", "HF_HOME", "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_XET", "TOKENIZERS_PARALLELISM",
    "RAY_TMPDIR", "PYTHONUNBUFFERED", "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK",
]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))
PY
)"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${RAY_BIN}" job submit \
  --address="${RAY_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${CMD[@]}" \
  2>&1 | tee "${DRIVER_LOG}"
SUBMIT_STATUS=${PIPESTATUS[0]}
set -e

printf "%s\n" "${SUBMIT_STATUS}" >"${ARTIFACT_DIR}/ray_job_exit_status.txt"
if [[ -e "${RAY_TMPDIR}/session_latest/logs" ]]; then
  tar -C "${RAY_TMPDIR}/session_latest" -czf "${ARTIFACT_DIR}/ray_session_latest_logs.tgz" logs || true
fi

EVAL_STATUS=0
if [[ "${ENABLE_G2_POST_EVAL}" == "true" ]]; then
  if (( SUBMIT_STATUS == 0 )); then
    if [[ -z "${POST_EVAL_MODEL_PATH:-}" ]]; then
      FINAL_ROLLOUT_ID=$((NUM_ROLLOUT - 1))
      POST_EVAL_MODEL_PATH="${SAVE_HF_PATH_TEMPLATE//\{rollout_id\}/${FINAL_ROLLOUT_ID}}"
    fi
    POST_EVAL_OUTPUT_DIR="${POST_EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/code_benchmarks}"
    mkdir -p "${POST_EVAL_OUTPUT_DIR}"
    echo "[post-eval] model=${POST_EVAL_MODEL_PATH}"
    echo "[post-eval] output=${POST_EVAL_OUTPUT_DIR}"
    set +e
    "${CODE_BENCHMARK_PYTHON_BIN}" "${CODE_BENCHMARK_SCRIPT}" \
      --model_path "${POST_EVAL_MODEL_PATH}" \
      --output_dir "${POST_EVAL_OUTPUT_DIR}" \
      --benchmarks "${CODE_BENCHMARKS}" \
      --backend "${CODE_BENCHMARK_BACKEND}" \
      --humaneval_dataset "${HUMANEVAL_EVAL_DATA}" \
      --humaneval_split "${HUMANEVAL_EVAL_SPLIT}" \
      --mbpp_dataset "${MBPP_EVAL_DATA}" \
      --mbpp_config "${MBPP_EVAL_CONFIG}" \
      --mbpp_split "${MBPP_EVAL_SPLIT}" \
      --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}" \
      --max_new_tokens "${CODE_EVAL_MAX_NEW_TOKENS}" \
      --sample_temperature "${CODE_EVAL_TEMPERATURE}" \
      --top_p "${CODE_EVAL_TOP_P}" \
      --repetition_penalty "${CODE_EVAL_REPETITION_PENALTY}" \
      --n_samples "${CODE_EVAL_N_SAMPLES}" \
      --passk_list "${CODE_EVAL_PASSK_LIST}" \
      --greedy_batch_size "${CODE_EVAL_GREEDY_BATCH_SIZE}" \
      --sample_batch_size "${CODE_EVAL_SAMPLE_BATCH_SIZE}" \
      --max_samples_per_benchmark "${POST_EVAL_MAX_SAMPLES}" \
      --timeout_seconds "${CODE_EVAL_TIMEOUT_SECONDS}" \
      --skip_missing_toolchains \
      2>&1 | tee "${POST_EVAL_LOG}"
    EVAL_STATUS=${PIPESTATUS[0]}
    set -e
  else
    echo "[post-eval] skipped because training failed with status ${SUBMIT_STATUS}" | tee "${POST_EVAL_LOG}"
  fi
fi
printf "%s\n" "${EVAL_STATUS}" >"${ARTIFACT_DIR}/post_eval_exit_status.txt"

if (( SUBMIT_STATUS != 0 )); then
  exit "${SUBMIT_STATUS}"
fi
exit "${EVAL_STATUS}"
