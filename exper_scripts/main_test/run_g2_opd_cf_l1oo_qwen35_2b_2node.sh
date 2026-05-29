#!/usr/bin/env bash
# Full standalone 2-node OPD-CF-L1OO Slime/Megatron run for Qwen3.5-2B.
#
# Deployment layouts:
#   DEPLOY_ROLE=teacher - start the 8-GPU SGLang teacher scorer on this node
#   DEPLOY_ROLE=student - run the 8-GPU Slime/Megatron student on this node
#
# GPU layout:
#   teacher node 8 GPU: Qwen3.5-27B SGLang scorer, TP=2 DP=4
#   student node 8 GPU: actor=2 GPU, critic/ref=2 GPU, rollout=4 GPU
#   student rollout: 2 SGLang engines x 2 GPU each
#
# Teacher node:
#   DEPLOY_ROLE=teacher bash exper_scripts/main_test/run_g2_opd_cf_l1oo_qwen35_2b_2node.sh
#
# Student node:
#   DEPLOY_ROLE=student TEACHER_HOST=<teacher-ip> \
#   bash exper_scripts/main_test/run_g2_opd_cf_l1oo_qwen35_2b_2node.sh
#
# Mainline semantics:
#   student on-policy rollouts
#   teacher token logprobs on those rollouts
#   cf_l1oo reward / credit assignment over the on-policy group
#   no teacher-completion distribution target
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 0. Runtime paths
# ---------------------------------------------------------------------------
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SLIME_ROOT}/../.." && pwd)}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/slime_runtime/Megatron-LM}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${SLIME_ROOT}/build_conda.sh}"

# DLC/PyTorchJob support. In DLC both pods can run this same script:
# rank0/master starts the teacher, rank1/worker starts the student.
DEPLOY_MODE="${DEPLOY_MODE:-auto}"          # auto|dlc|manual
DLC_AUTO_ENV="${DLC_AUTO_ENV:-true}"
BUILD_SLIME_ENV="${BUILD_SLIME_ENV:-auto}" # auto|true|false
DLC_LOCAL_ROOT="${DLC_LOCAL_ROOT:-/mnt/workspace}"
TEACHER_NODE_RANK="${TEACHER_NODE_RANK:-0}"
STUDENT_NODE_RANK="${STUDENT_NODE_RANK:-1}"
DLC_MODE="false"
DLC_NODE_RANK=""
DLC_MASTER_ADDR=""
DLC_WORLD_SIZE="${WORLD_SIZE:-${PET_WORLD_SIZE:-1}}"

# ---------------------------------------------------------------------------
# 1. Model and dataset
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-2B}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_PATH}}"
REF_LOAD="${REF_LOAD:-/mnt/data/models/Megatron_convert_models/Qwen3.5-2B_torch_dist}"
SLIME_DATA_ROOT="${SLIME_DATA_ROOT:-${PROJECT_ROOT}/data}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-${SLIME_DATA_ROOT}/diff_dataset_prepared}"
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
DEPLOY_LAYOUT="${DEPLOY_LAYOUT:-two_node}"
DEPLOY_ROLE="${DEPLOY_ROLE:-student}"
TEACHER_HOST="${TEACHER_HOST:-}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-127.0.0.1}"
EXPECTED_GPUS_PER_NODE="${EXPECTED_GPUS_PER_NODE:-8}"

if [[ -n "${TEACHER_HOST}" ]]; then
  DEPLOY_LAYOUT="two_node"
  TEACHER_API_BASE="http://${TEACHER_HOST}:${TEACHER_PORT}"
fi

# Student node default uses all 8 local GPUs. Slime/Ray is local to the student
# node; the teacher node is an external SGLang scoring service.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
USER_SET_ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE+x}"
USER_SET_SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH+x}"
USER_SET_SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND+x}"
USER_SET_SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND+x}"
USER_SET_SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS+x}"
USER_SET_SGLANG_DIRECT_WORKER_MODE="${SGLANG_DIRECT_WORKER_MODE+x}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-true}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train_async.py}"
COLOCATE="${COLOCATE:-false}"
# Student 8-GPU split: actor 2 + critic/ref 2 + rollout 4.
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"

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
SGLANG_503_DEBUG_MODE="${SGLANG_503_DEBUG_MODE:-true}"
if [[ "${SGLANG_503_DEBUG_MODE}" == "true" || "${SGLANG_503_DEBUG_MODE}" == "1" ]]; then
  # The DLC failure mode "503 no_available_workers (all circuits open or unhealthy)"
  # is easiest to separate by bypassing the rollout router entirely.
  [[ -z "${USER_SET_ROLLOUT_NUM_GPUS_PER_ENGINE}" ]] && ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS}"
  [[ -z "${USER_SET_SGLANG_DIRECT_WORKER_MODE}" ]] && SGLANG_DIRECT_WORKER_MODE="true"
  [[ -z "${USER_SET_SGLANG_DISABLE_CUDA_GRAPH}" ]] && SGLANG_DISABLE_CUDA_GRAPH="true"
  [[ -z "${USER_SET_SGLANG_ATTENTION_BACKEND}" ]] && SGLANG_ATTENTION_BACKEND="triton"
  [[ -z "${USER_SET_SGLANG_SAMPLING_BACKEND}" ]] && SGLANG_SAMPLING_BACKEND="pytorch"
  [[ -z "${USER_SET_SGLANG_MAX_RUNNING_REQUESTS}" ]] && SGLANG_MAX_RUNNING_REQUESTS="16"
fi
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-1024}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-${COMPLETION_MAX_LENGTH}}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-false}"
SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-true}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-}"
SGLANG_GRAMMAR_BACKEND="${SGLANG_GRAMMAR_BACKEND:-none}"
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
# 6. OPD-CF-L1OO teacher scorer and reward
# ---------------------------------------------------------------------------
G2_OPD_MODE="${G2_OPD_MODE:-opd_cf_l1oo_sglang_2node}"
CF_TARGET_MODE="${CF_TARGET_MODE:-opd_onpolicy}"
OPD_CREDIT_ASSIGNMENT="${OPD_CREDIT_ASSIGNMENT:-cf_l1oo}"
OPD_CF_SCORE_TEMPERATURE="${OPD_CF_SCORE_TEMPERATURE:-1.0}"
OPD_CF_SCORE_NORMALIZATION="${OPD_CF_SCORE_NORMALIZATION:-mean}"
OPD_KL_APPLICATION="${OPD_KL_APPLICATION:-auto}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-4}"
TEACHER_BACKEND="${TEACHER_BACKEND:-remote}"
if [[ -z "${TEACHER_API_BASE:-}" ]]; then
  TEACHER_API_BASE="http://127.0.0.1:${TEACHER_PORT}"
fi
TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
TEACHER_API_STYLE="${TEACHER_API_STYLE:-sglang_generate}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
if [[ "${DEPLOY_LAYOUT}" == "two_node" ]]; then
  TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-240}"
  TEACHER_PREFLIGHT_TIMEOUT="${TEACHER_PREFLIGHT_TIMEOUT:-30}"
else
  TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-120}"
  TEACHER_PREFLIGHT_TIMEOUT="${TEACHER_PREFLIGHT_TIMEOUT:-5}"
fi
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
SKIP_TEACHER_PREFLIGHT="${SKIP_TEACHER_PREFLIGHT:-0}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-3600}"
TEACHER_POLL_SECONDS="${TEACHER_POLL_SECONDS:-10}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-16}"
TEACHER_SGLANG_MULTI_SAMPLE="${TEACHER_SGLANG_MULTI_SAMPLE:-true}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
TEACHER_SYSTEM_PROMPT_TEXT="${TEACHER_SYSTEM_PROMPT_TEXT:-You are a precise assistant. Produce a correct and well-reasoned answer.}"
TEACHER_SYSTEM_PROMPT_ID="${TEACHER_SYSTEM_PROMPT_ID:-g2-opd-main-v1}"
TEACHER_CACHE_ENABLE="${TEACHER_CACHE_ENABLE:-false}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-/mnt/workspace/teacher_cache_shared}"
TEACHER_MAX_RUNNING_REQUESTS="${TEACHER_MAX_RUNNING_REQUESTS:-16}"

OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
OPD_TEACHER_RM_URL="${OPD_TEACHER_RM_URL:-${TEACHER_API_BASE%/}/generate}"
ENABLE_EFFOPD="${ENABLE_EFFOPD:-false}"
EFFOPD_DV_SIZE="${EFFOPD_DV_SIZE:-32}"
EFFOPD_DV_SEED="${EFFOPD_DV_SEED:-42}"
EFFOPD_MAX_K="${EFFOPD_MAX_K:-5}"
EFFOPD_LR_DECAY="${EFFOPD_LR_DECAY:-0.5}"
EFFOPD_VALIDATION_MODE="${EFFOPD_VALIDATION_MODE:-opd_kl_shadow_cf}"
EFFOPD_MAX_TRIGGERS="${EFFOPD_MAX_TRIGGERS:--1}"
EFFOPD_FORCE_WEIGHT_SYNC="${EFFOPD_FORCE_WEIGHT_SYNC:-true}"

# ---------------------------------------------------------------------------
# 7. G1 embedding/reward path used by G2
# ---------------------------------------------------------------------------
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
G1_EBFT_LOGPROB_INDEXING="${G1_EBFT_LOGPROB_INDEXING:-strict_block_source}"
G1_EBFT_ROLLOUT_SAMPLING_MODE="${G1_EBFT_ROLLOUT_SAMPLING_MODE:-block_source}"
G1_EBFT_ROLLOUT_MASK_MODE="${G1_EBFT_ROLLOUT_MASK_MODE:-sparse_ir}"
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
CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT:-${SLIME_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${SLIME_ROOT}/outputs}"
USER_SET_ARTIFACT_DIR="${ARTIFACT_DIR+x}"
default_run_name() {
  local host job_id master_key
  host="$(hostname 2>/dev/null || echo unknown-host)"
  if [[ -n "${PET_JOB_NAME:-}" ]]; then
    echo "g2_opd_cf_l1oo_${PET_JOB_NAME}"
    return 0
  fi
  job_id="$(printf "%s" "${host}" | sed -E 's/-(master|worker)-[0-9]+$//')"
  if [[ -n "${job_id}" && "${job_id}" != "${host}" ]]; then
    echo "g2_opd_cf_l1oo_${job_id}"
    return 0
  fi
  if [[ -n "${MASTER_ADDR:-${PET_MASTER_ADDR:-}}" && "${WORLD_SIZE:-${PET_WORLD_SIZE:-1}}" != "1" ]]; then
    master_key="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
    master_key="$(printf "%s" "${master_key}" | tr -c 'A-Za-z0-9_' '_')"
    echo "g2_opd_cf_l1oo_${master_key}_${WORLD_SIZE:-${PET_WORLD_SIZE:-2}}n"
    return 0
  fi
  echo "g2_opd_cf_l1oo_qwen35_2b_2node_$(date +%m%d_%H%M%S)"
}
RUN_NAME="${RUN_NAME:-$(default_run_name)}"
LOAD_PATH="${LOAD_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/mcore}"
SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
CRITIC_SAVE_PATH="${CRITIC_SAVE_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/critic_mcore}"
SAVE_HF_PATH_TEMPLATE="${SAVE_HF_PATH_TEMPLATE:-${OUTPUT_ROOT}/${RUN_NAME}/hf/rollout_{rollout_id}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/artifacts}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_g2_opd_cf_${PET_NODE_RANK:-${RANK:-student}}}"
DIST_CKPT_STRICTNESS="${DIST_CKPT_STRICTNESS:-log_unexpected}"
RAY_ADDRESS="http://${RAY_NODE_IP_ADDRESS}:${RAY_DASHBOARD_PORT}"
DRIVER_LOG="${ARTIFACT_DIR}/ray_job_driver.log"
POST_EVAL_LOG="${ARTIFACT_DIR}/post_eval.log"

if [[ -f "${SLIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ENV_FILE}"
fi

detect_dlc_mode() {
  DLC_MODE="false"
  DLC_NODE_RANK=""
  DLC_MASTER_ADDR=""
  DLC_WORLD_SIZE="${WORLD_SIZE:-${PET_WORLD_SIZE:-1}}"
  if [[ -n "${PET_NODE_RANK:-}" ]]; then
    DLC_MODE="true"
    DLC_NODE_RANK="${PET_NODE_RANK}"
    DLC_MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
  elif [[ -n "${RANK:-}" && -n "${MASTER_ADDR:-}" && "${DLC_WORLD_SIZE:-1}" -gt 1 ]]; then
    DLC_MODE="true"
    DLC_NODE_RANK="${RANK}"
    DLC_MASTER_ADDR="${MASTER_ADDR}"
  fi
}

resolve_host_ip() {
  local host="$1"
  if [[ "${host}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "${host}"
    return 0
  fi
  getent ahostsv4 "${host}" | awk 'NR==1 {print $1}'
}

get_local_ip() {
  if command -v ip >/dev/null 2>&1; then
    local eth0_ip
    eth0_ip="$(ip -o -4 addr show dev eth0 scope global 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}')"
    if [[ -n "${eth0_ip}" ]]; then
      echo "${eth0_ip}"
      return 0
    fi
  fi
  hostname -I 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i !~ /^127\./ && $i !~ /:/) {print $i; exit}}'
}

setup_no_proxy_env() {
  local local_ip
  local_ip="$(get_local_ip || true)"
  local extra="127.0.0.1,localhost,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

  if [[ -n "${local_ip}" ]]; then
    extra="${extra},${local_ip}"
  fi
  if [[ -n "${RAY_NODE_IP_ADDRESS:-}" ]]; then
    extra="${extra},${RAY_NODE_IP_ADDRESS}"
  fi
  if [[ -n "${TEACHER_HOST:-}" ]]; then
    extra="${extra},${TEACHER_HOST}"
  fi
  if [[ -n "${TEACHER_BIND_HOST:-}" ]]; then
    extra="${extra},${TEACHER_BIND_HOST}"
  fi
  if [[ -n "${DLC_MASTER_ADDR:-}" ]]; then
    extra="${extra},${DLC_MASTER_ADDR}"
  fi
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    extra="${extra},${MASTER_ADDR}"
  fi
  if [[ -n "${PET_MASTER_ADDR:-}" ]]; then
    extra="${extra},${PET_MASTER_ADDR}"
  fi

  if [[ -n "${NO_PROXY:-}" ]]; then
    export NO_PROXY="${NO_PROXY},${extra}"
  else
    export NO_PROXY="${extra}"
  fi
  export no_proxy="${NO_PROXY}"
}

setup_dlc_runtime_env() {
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${DLC_LOCAL_ROOT}/.torch_extensions}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DLC_LOCAL_ROOT}/.triton_cache}"
  export TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-${DLC_LOCAL_ROOT}/teacher_cache_shared}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_g2_opd_cf_${DLC_NODE_RANK:-0}}"
  export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
  [[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
  export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"
  mkdir -p "${DLC_LOCAL_ROOT}/slime_logs" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${RAY_TMPDIR}" "${TEACHER_CACHE_DIR}"
}

setup_dlc_stable_run_name() {
  if [[ -n "${RUN_NAME:-}" ]]; then
    return 0
  fi
  local job_id
  job_id="$(hostname | sed -E 's/^(dlc[a-z0-9]+)-(master|worker)-[0-9]+$/\1/' || true)"
  if [[ -n "${job_id}" && "${job_id}" != "$(hostname)" ]]; then
    RUN_NAME="g2_opd_cf_l1oo_${job_id}"
  fi
}

need_slime_build() {
  [[ "${BUILD_SLIME_ENV}" == "true" || "${BUILD_SLIME_ENV}" == "1" ]] && return 0
  [[ "${BUILD_SLIME_ENV}" == "auto" ]] || return 1
  [[ ! -f "${SLIME_ENV_FILE}" ]] && return 0
  [[ ! -x "${PYTHON_BIN}" ]] && return 0
  return 1
}

maybe_install_cuda129_for_dlc_build() {
  if [[ "${DLC_MODE}" != "true" ]]; then
    return 0
  fi
  if [[ "${INSTALL_CUDA129_FROM_WHEELS_INFRA:-true}" != "true" && "${INSTALL_CUDA129_FROM_WHEELS_INFRA:-true}" != "1" ]]; then
    return 0
  fi
  local installer="${SLIME_ROOT}/scripts/install_cuda129_from_wheels_infra.sh"
  if [[ -x /usr/local/cuda-12.9/bin/nvcc || ! -f "${installer}" ]]; then
    return 0
  fi
  echo "[dlc] installing CUDA 12.9 toolkit from wheels_infra before env build"
  bash "${installer}"
}

maybe_build_slime_env() {
  if [[ "${DLC_MODE}" != "true" || "${DLC_AUTO_ENV}" != "true" ]]; then
    return 0
  fi
  if ! need_slime_build; then
    return 0
  fi
  if [[ ! -f "${BUILD_SCRIPT}" ]]; then
    echo "[ERROR] BUILD_SLIME_ENV requested but BUILD_SCRIPT missing: ${BUILD_SCRIPT}" >&2
    exit 1
  fi
  maybe_install_cuda129_for_dlc_build
  echo "[dlc] rank=${DLC_NODE_RANK} building local Slime env via ${BUILD_SCRIPT}"
  bash "${BUILD_SCRIPT}"
}

source_slime_env() {
  if [[ -f "${SLIME_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${SLIME_ENV_FILE}"
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -d "${VIRTUAL_ENV}/bin" ]]; then
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
  elif [[ -d "$(dirname "${PYTHON_BIN}")" ]]; then
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  fi
  if [[ ! -x "${RAY_BIN}" && -x "$(dirname "${PYTHON_BIN}")/ray" ]]; then
    RAY_BIN="$(dirname "${PYTHON_BIN}")/ray"
  fi
}

wait_opd_logprob_ready() {
  local url="$1"
  echo "[preflight] waiting for OPD logprob endpoint: ${url} (timeout ${TEACHER_WAIT_SECONDS}s)"
  OPD_TEACHER_RM_URL="${url}" \
  TEACHER_API_KEY="${TEACHER_API_KEY}" \
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
deadline = time.time() + float(os.environ.get("TEACHER_WAIT_SECONDS", "3600"))
poll = float(os.environ.get("TEACHER_POLL_SECONDS", "10"))
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
        with urllib.request.urlopen(request, timeout=poll) as response:
            result = json.loads(response.read().decode("utf-8"))
        token_logprobs = result.get("meta_info", {}).get("input_token_logprobs")
        if isinstance(token_logprobs, list) and len(token_logprobs) >= 2:
            scored = [item[0] for item in token_logprobs[1:] if isinstance(item, (list, tuple)) and item]
            if any(isinstance(value, (int, float)) for value in scored):
                print("[preflight] OPD logprob endpoint is ready")
                raise SystemExit(0)
        last_error = "missing numeric meta_info.input_token_logprobs"
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    print(f"[preflight] OPD endpoint not ready: {last_error}", file=sys.stderr, flush=True)
    time.sleep(min(10.0, poll))
print(f"[ERROR] timed out waiting for OPD endpoint {url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

print_shell_command() {
  {
    printf "%q " "$@"
    printf "\n"
  } || echo "[WARN] failed to print shell command" >&2
}

detect_dlc_mode
if [[ "${DEPLOY_MODE}" == "auto" && "${DLC_MODE}" == "true" ]]; then
  DEPLOY_MODE="dlc"
fi
if [[ "${DEPLOY_MODE}" == "dlc" ]]; then
  if [[ "${DLC_MODE}" != "true" || "${DLC_WORLD_SIZE}" != "2" ]]; then
    echo "[ERROR] DEPLOY_MODE=dlc expects WORLD_SIZE=2 with RANK/MASTER_ADDR or PET_* envs; got world=${DLC_WORLD_SIZE}" >&2
    exit 1
  fi
  setup_dlc_stable_run_name
  setup_dlc_runtime_env
  if [[ -z "${USER_SET_ARTIFACT_DIR}" && "${DLC_ARTIFACTS_ON_LOCAL:-true}" == "true" ]]; then
    ARTIFACT_DIR="${DLC_LOCAL_ROOT}/slime_logs/${RUN_NAME}/artifacts"
    DRIVER_LOG="${ARTIFACT_DIR}/ray_job_driver.log"
    POST_EVAL_LOG="${ARTIFACT_DIR}/post_eval.log"
  fi
  maybe_build_slime_env
  source_slime_env
  if [[ "${DLC_NODE_RANK}" == "${TEACHER_NODE_RANK}" ]]; then
    DEPLOY_ROLE="teacher"
    DEPLOY_LAYOUT="two_node"
    TEACHER_BIND_HOST="${TEACHER_BIND_HOST:-0.0.0.0}"
  elif [[ "${DLC_NODE_RANK}" == "${STUDENT_NODE_RANK}" ]]; then
    DEPLOY_ROLE="student"
    DEPLOY_LAYOUT="two_node"
    TEACHER_HOST="$(resolve_host_ip "${DLC_MASTER_ADDR}")"
    if [[ -z "${TEACHER_HOST}" ]]; then
      TEACHER_HOST="${DLC_MASTER_ADDR}"
    fi
    TEACHER_API_BASE="http://${TEACHER_HOST}:${TEACHER_PORT}"
    OPD_TEACHER_RM_URL="${TEACHER_API_BASE%/}/generate"
    RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-$(get_local_ip)}"
  else
    echo "[ERROR] unexpected DLC_NODE_RANK=${DLC_NODE_RANK}; expected teacher=${TEACHER_NODE_RANK} or student=${STUDENT_NODE_RANK}" >&2
    exit 1
  fi
else
  source_slime_env
fi

setup_no_proxy_env

launch_teacher_node() {
  local teacher_model_path="${TEACHER_MODEL_PATH:-/mnt/data/models/Qwen3.5-27B}"
  local teacher_served_model_name="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
  local teacher_bind_host="${TEACHER_BIND_HOST:-0.0.0.0}"
  local teacher_cuda_visible_devices="${TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  local teacher_tp_size="${TEACHER_TP_SIZE:-2}"
  local teacher_dp_size="${TEACHER_DP_SIZE:-4}"
  local teacher_context_length="${TEACHER_CONTEXT_LENGTH:-4096}"
  local teacher_mem_fraction_static="${TEACHER_MEM_FRACTION_STATIC:-0.55}"
  local teacher_max_running_requests="${TEACHER_MAX_RUNNING_REQUESTS:-}"
  local teacher_extra_sglang_args="${TEACHER_EXTRA_SGLANG_ARGS:---disable-cuda-graph}"

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -d "${teacher_model_path}" ]]; then
    echo "[ERROR] TEACHER_MODEL_PATH missing: ${teacher_model_path}" >&2
    exit 1
  fi
  local teacher_num_gpus
  teacher_num_gpus="$(
    TEACHER_CUDA_VISIBLE_DEVICES="${teacher_cuda_visible_devices}" "${PYTHON_BIN}" - <<'PY'
import os
print(len([x for x in os.environ.get("TEACHER_CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",") if x.strip()]))
PY
  )"
  if [[ "${EXPECTED_GPUS_PER_NODE}" != "0" && "${teacher_num_gpus}" != "${EXPECTED_GPUS_PER_NODE}" ]]; then
    echo "[ERROR] teacher node expected ${EXPECTED_GPUS_PER_NODE} visible GPUs, got ${teacher_num_gpus}: ${teacher_cuda_visible_devices}" >&2
    echo "[ERROR] Set EXPECTED_GPUS_PER_NODE=0 to bypass this check for debugging." >&2
    exit 1
  fi

  local cmd=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --model-path "${teacher_model_path}"
    --served-model-name "${teacher_served_model_name}"
    --host "${teacher_bind_host}"
    --port "${TEACHER_PORT}"
    --tp-size "${teacher_tp_size}"
    --dp-size "${teacher_dp_size}"
    --context-length "${teacher_context_length}"
    --mem-fraction-static "${teacher_mem_fraction_static}"
  )
  if [[ -n "${teacher_max_running_requests}" ]]; then
    cmd+=(--max-running-requests "${teacher_max_running_requests}")
  fi
  if [[ -n "${teacher_extra_sglang_args}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${teacher_extra_sglang_args})
  fi

  echo "[2node-opd-cf-l1oo-teacher] CUDA_VISIBLE_DEVICES=${teacher_cuda_visible_devices}"
  echo "[2node-opd-cf-l1oo-teacher] model=${teacher_model_path} served=${teacher_served_model_name}"
  echo "[2node-opd-cf-l1oo-teacher] bind=${teacher_bind_host}:${TEACHER_PORT} TP=${teacher_tp_size} DP=${teacher_dp_size} context=${teacher_context_length} mem=${teacher_mem_fraction_static}"
  printf "[2node-opd-cf-l1oo-teacher] command: " || true
  print_shell_command "${cmd[@]}"

  CUDA_VISIBLE_DEVICES="${teacher_cuda_visible_devices}" exec "${cmd[@]}"
}

if [[ "${DEPLOY_ROLE}" == "teacher" ]]; then
  launch_teacher_node
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
if [[ ! "${SAVE_INTERVAL}" =~ ^(none|off|false)$ ]]; then
  require_positive_int "SAVE_INTERVAL" "${SAVE_INTERVAL}"
fi
require_bool "G1_FILTER_TRAIN_DATA" "${G1_FILTER_TRAIN_DATA}"
require_bool "G1_USE_EBFT_LOSS" "${G1_USE_EBFT_LOSS}"
require_bool "G1_QA_MASKING" "${G1_QA_MASKING}"
require_bool "USE_WHITENING" "${USE_WHITENING}"
require_bool "TEACHER_CACHE_ENABLE" "${TEACHER_CACHE_ENABLE}"
require_bool "TEACHER_SGLANG_MULTI_SAMPLE" "${TEACHER_SGLANG_MULTI_SAMPLE}"
require_bool "ENABLE_SLIME_EVAL" "${ENABLE_SLIME_EVAL}"
require_bool "ENABLE_G2_POST_EVAL" "${ENABLE_G2_POST_EVAL}"
require_bool "SGLANG_DISABLE_CUDA_GRAPH" "${SGLANG_DISABLE_CUDA_GRAPH}"
require_bool "SGLANG_DISABLE_OVERLAP_SCHEDULE" "${SGLANG_DISABLE_OVERLAP_SCHEDULE}"
require_bool "SGLANG_ROUTER_DISABLE_CIRCUIT_BREAKER" "${SGLANG_ROUTER_DISABLE_CIRCUIT_BREAKER}"
require_bool "SGLANG_DIRECT_WORKER_MODE" "${SGLANG_DIRECT_WORKER_MODE}"
case "${G1_EBFT_LOGPROB_INDEXING}" in
  standard_next_token|strict_block_source) ;;
  *) echo "[ERROR] G1_EBFT_LOGPROB_INDEXING must be standard_next_token or strict_block_source, got: ${G1_EBFT_LOGPROB_INDEXING}" >&2; exit 1 ;;
esac
case "${G1_EBFT_ROLLOUT_MASK_MODE}" in
  none|dense4d|sparse_ir) ;;
  *) echo "[ERROR] G1_EBFT_ROLLOUT_MASK_MODE must be none, dense4d, or sparse_ir, got: ${G1_EBFT_ROLLOUT_MASK_MODE}" >&2; exit 1 ;;
esac
case "${G1_EBFT_ROLLOUT_SAMPLING_MODE}" in
  standard|block_source) ;;
  *) echo "[ERROR] G1_EBFT_ROLLOUT_SAMPLING_MODE must be standard or block_source, got: ${G1_EBFT_ROLLOUT_SAMPLING_MODE}" >&2; exit 1 ;;
esac
if [[ "${G1_EBFT_ROLLOUT_SAMPLING_MODE}" == "block_source" ]]; then
  if [[ "${G1_USE_EBFT_LOSS}" != "true" ]]; then
    echo "[ERROR] G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source requires G1_USE_EBFT_LOSS=true" >&2
    exit 1
  fi
  if [[ "${G1_EBFT_LOGPROB_INDEXING}" != "strict_block_source" ]]; then
    echo "[ERROR] G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source requires G1_EBFT_LOGPROB_INDEXING=strict_block_source" >&2
    exit 1
  fi
  if [[ "${G1_EBFT_ROLLOUT_MASK_MODE}" != "dense4d" && "${G1_EBFT_ROLLOUT_MASK_MODE}" != "sparse_ir" ]]; then
    echo "[ERROR] G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source requires G1_EBFT_ROLLOUT_MASK_MODE=dense4d or sparse_ir" >&2
    exit 1
  fi
  if [[ "${G1_EBFT_ROLLOUT_MASK_MODE}" == "dense4d" && "${SGLANG_ATTENTION_BACKEND}" != "torch_native" ]]; then
    echo "[ERROR] G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source with dense4d requires SGLANG_ATTENTION_BACKEND=torch_native" >&2
    exit 1
  fi
  if [[ "${G1_EBFT_ROLLOUT_MASK_MODE}" == "sparse_ir" && "${SGLANG_ATTENTION_BACKEND}" != "triton" ]]; then
    echo "[ERROR] G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source with sparse_ir requires SGLANG_ATTENTION_BACKEND=triton" >&2
    exit 1
  fi
  if [[ "${SGLANG_DISABLE_OVERLAP_SCHEDULE}" != "true" ]]; then
    echo "[ERROR] G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source requires SGLANG_DISABLE_OVERLAP_SCHEDULE=true" >&2
    exit 1
  fi
fi
if [[ "${G1_EBFT_ROLLOUT_MASK_MODE}" != "none" && "${G1_EBFT_ROLLOUT_SAMPLING_MODE}" == "standard" ]]; then
  echo "[ERROR] G1_EBFT_ROLLOUT_MASK_MODE=${G1_EBFT_ROLLOUT_MASK_MODE} is transport only and requires block_source sampling; standard sampling cannot consume EBFT rollout masks" >&2
  exit 1
fi
case "${SGLANG_GRAMMAR_BACKEND}" in
  none|xgrammar|outlines|llguidance) ;;
  *) echo "[ERROR] SGLANG_GRAMMAR_BACKEND must be none, xgrammar, outlines, or llguidance, got: ${SGLANG_GRAMMAR_BACKEND}" >&2; exit 1 ;;
esac

case "${DEPLOY_LAYOUT}" in
  single_node|two_node) ;;
  *) echo "[ERROR] DEPLOY_LAYOUT must be single_node or two_node, got: ${DEPLOY_LAYOUT}" >&2; exit 1 ;;
esac
if [[ "${DEPLOY_LAYOUT}" == "two_node" && -z "${TEACHER_HOST}" ]]; then
  echo "[ERROR] two_node layout requires TEACHER_HOST (teacher node IP/hostname reachable from student)." >&2
  exit 1
fi
if [[ "${TEACHER_BACKEND}" != "remote" ]]; then
  echo "[ERROR] G2+OPD main supports TEACHER_BACKEND=remote only, got: ${TEACHER_BACKEND}" >&2
  exit 1
fi
case "${TEACHER_API_STYLE}" in
  sglang_generate|completions|chat_completions) ;;
  *) echo "[ERROR] TEACHER_API_STYLE must be one of sglang_generate, completions, chat_completions; got: ${TEACHER_API_STYLE}" >&2; exit 1 ;;
esac
if [[ -z "${TEACHER_API_BASE}" || -z "${TEACHER_MODEL_NAME}" || -z "${OPD_TEACHER_RM_URL}" ]]; then
  echo "[ERROR] TEACHER_API_BASE, TEACHER_MODEL_NAME, and OPD_TEACHER_RM_URL must be set." >&2
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
if [[ "${EXPECTED_GPUS_PER_NODE}" != "0" && "${NUM_GPUS}" != "${EXPECTED_GPUS_PER_NODE}" ]]; then
  echo "[ERROR] student node expected ${EXPECTED_GPUS_PER_NODE} visible GPUs, got ${NUM_GPUS}: ${CUDA_VISIBLE_DEVICES}" >&2
  echo "[ERROR] Set EXPECTED_GPUS_PER_NODE=0 to bypass this check for debugging." >&2
  exit 1
fi
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
if [[ "${SGLANG_DIRECT_WORKER_MODE}" == "true" && "${COLOCATE}" == "false" && "${ROLLOUT_NUM_GPUS}" != "${ROLLOUT_NUM_GPUS_PER_ENGINE}" ]]; then
  echo "[ERROR] SGLANG_DIRECT_WORKER_MODE=true requires exactly one rollout worker; set ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS} or SGLANG_503_DEBUG_MODE=true." >&2
  exit 1
fi

DP_SIZE=$((ACTOR_NUM_GPUS_PER_NODE / PARALLEL_GROUP_SIZE))
if (( ROLLOUT_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "[ERROR] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}" "${LOAD_PATH}" "${SAVE_PATH}" "${CRITIC_SAVE_PATH}" "${RAY_TMPDIR}"
if [[ "${TEACHER_CACHE_ENABLE}" == "true" ]]; then
  mkdir -p "${TEACHER_CACHE_DIR}"
fi

write_env_diagnostics() {
  local diag="${ARTIFACT_DIR}/env_diagnostics.txt"
  {
    echo "RUN_NAME=${RUN_NAME}"
    echo "DEPLOY_MODE=${DEPLOY_MODE} DLC_MODE=${DLC_MODE} DLC_NODE_RANK=${DLC_NODE_RANK:-}"
    echo "host=$(hostname 2>/dev/null || true)"
    echo "date=$(date -Is 2>/dev/null || true)"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "RAY_BIN=${RAY_BIN}"
    echo "CUDA_HOME=${CUDA_HOME:-}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    echo "PATH=${PATH}"
    command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc: not found"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi: not found"
    if [[ -x "${PYTHON_BIN}" ]]; then
      "${PYTHON_BIN}" - <<'PY'
import importlib
import sys

print(f"python={sys.executable}")
for name in ["torch", "ray", "sglang", "flashinfer", "sgl_kernel"]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}={version}")
    except Exception as exc:
        print(f"{name}: IMPORT_ERROR {type(exc).__name__}: {exc}")
PY
      timeout 120 "${PYTHON_BIN}" -m pip check || true
    else
      echo "python: not executable"
    fi
  } >"${diag}" 2>&1 || echo "[WARN] failed to write ${diag}; continuing without env diagnostics" >&2
}
write_env_diagnostics

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
  --dist-ckpt-strictness "${DIST_CKPT_STRICTNESS}"
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
  --opd-credit-assignment "${OPD_CREDIT_ASSIGNMENT}"
  --opd-cf-score-temperature "${OPD_CF_SCORE_TEMPERATURE}"
  --opd-cf-score-normalization "${OPD_CF_SCORE_NORMALIZATION}"
  --use-opd
  --opd-type sglang
  --opd-kl-coef "${OPD_KL_COEF}"
  --opd-kl-application "${OPD_KL_APPLICATION}"
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
  --rm-url "${OPD_TEACHER_RM_URL}"
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
CMD+=(
  --g1-ebft-logprob-indexing "${G1_EBFT_LOGPROB_INDEXING}"
  --g1-ebft-rollout-sampling-mode "${G1_EBFT_ROLLOUT_SAMPLING_MODE}"
  --g1-ebft-rollout-mask-mode "${G1_EBFT_ROLLOUT_MASK_MODE}"
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sglang-grammar-backend "${SGLANG_GRAMMAR_BACKEND}"
)
if [[ -n "${NUM_ROLLOUT}" ]]; then
  require_positive_int "NUM_ROLLOUT" "${NUM_ROLLOUT}"
  CMD+=(--num-rollout "${NUM_ROLLOUT}")
else
  require_positive_int "NUM_EPOCH" "${NUM_EPOCH}"
  CMD+=(--num-epoch "${NUM_EPOCH}")
fi
if [[ ! "${SAVE_INTERVAL}" =~ ^(none|off|false)$ ]]; then
  CMD+=(--save "${SAVE_PATH}" --critic-save "${CRITIC_SAVE_PATH}" --save-interval "${SAVE_INTERVAL}")
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
if [[ "${G1_QA_MASKING}" == "true" ]]; then
  CMD+=(--g1-qa-masking)
fi
if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "true" ]]; then
  CMD+=(--sglang-disable-cuda-graph)
fi
if [[ "${SGLANG_DISABLE_OVERLAP_SCHEDULE}" == "true" ]]; then
  CMD+=(--sglang-disable-overlap-schedule)
fi
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]]; then
  CMD+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
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
if [[ "${CF_TARGET_MODE}" == "teacher" ]]; then
  CMD+=(
    --cf-teacher-lambda "${CF_TEACHER_LAMBDA}"
    --cf-teacher-n-samples "${CF_TEACHER_N_SAMPLES}"
    --teacher-backend "${TEACHER_BACKEND}"
    --teacher-api-base "${TEACHER_API_BASE}"
    --teacher-api-key "${TEACHER_API_KEY}"
    --teacher-api-style "${TEACHER_API_STYLE}"
    --teacher-model-name "${TEACHER_MODEL_NAME}"
    --teacher-timeout "${TEACHER_TIMEOUT}"
    --teacher-max-retries "${TEACHER_MAX_RETRIES}"
    --teacher-remote-batch-size "${TEACHER_REMOTE_BATCH_SIZE}"
    --teacher-temperature "${TEACHER_TEMPERATURE}"
    --teacher-top-p "${TEACHER_TOP_P}"
    --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}"
    --teacher-system-prompt-text "${TEACHER_SYSTEM_PROMPT_TEXT}"
    --teacher-system-prompt-id "${TEACHER_SYSTEM_PROMPT_ID}"
  )
  if [[ "${TEACHER_CACHE_ENABLE}" == "true" ]]; then
    CMD+=(--teacher-cache-enable --teacher-cache-dir "${TEACHER_CACHE_DIR}")
  fi
  if [[ "${TEACHER_SGLANG_MULTI_SAMPLE}" == "true" ]]; then
    CMD+=(--teacher-sglang-multi-sample)
  else
    CMD+=(--no-teacher-sglang-multi-sample)
  fi
fi
if [[ "${COLOCATE}" == "true" ]]; then
  CMD+=(--colocate)
else
  CMD+=(--rollout-num-gpus "${ROLLOUT_NUM_GPUS}")
fi
if [[ "${G1_APPLY_DENSE_ATTENTION_MASK}" == "true" ]]; then
  CMD+=(--g1-megatron-ref-apply-dense-attention-mask)
fi
if [[ "${ENABLE_EFFOPD}" == "true" ]]; then
  CMD+=(
    --use-effopd
    --effopd-dv-size "${EFFOPD_DV_SIZE}"
    --effopd-dv-seed "${EFFOPD_DV_SEED}"
    --effopd-max-k "${EFFOPD_MAX_K}"
    --effopd-lr-decay "${EFFOPD_LR_DECAY}"
    --effopd-validation-mode "${EFFOPD_VALIDATION_MODE}"
    --effopd-max-triggers "${EFFOPD_MAX_TRIGGERS}"
  )
  if [[ "${EFFOPD_FORCE_WEIGHT_SYNC}" == "false" ]]; then
    CMD+=(--no-effopd-force-weight-sync)
  fi
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
DEPLOY_LAYOUT=${DEPLOY_LAYOUT}
DEPLOY_ROLE=${DEPLOY_ROLE}
TEACHER_HOST=${TEACHER_HOST}
TEACHER_PORT=${TEACHER_PORT}
RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS}
RAY_ADDRESS=${RAY_ADDRESS}
G2_OPD_MODE=${G2_OPD_MODE}
CF_TARGET_MODE=${CF_TARGET_MODE}
OPD_CREDIT_ASSIGNMENT=${OPD_CREDIT_ASSIGNMENT}
OPD_CF_SCORE_TEMPERATURE=${OPD_CF_SCORE_TEMPERATURE}
OPD_CF_SCORE_NORMALIZATION=${OPD_CF_SCORE_NORMALIZATION}
OPD_KL_APPLICATION=${OPD_KL_APPLICATION}
LOAD_PATH=${LOAD_PATH}
SAVE_PATH=${SAVE_PATH}
CRITIC_SAVE_PATH=${CRITIC_SAVE_PATH}
SAVE_HF_PATH_TEMPLATE=${SAVE_HF_PATH_TEMPLATE}
ARTIFACT_DIR=${ARTIFACT_DIR}
DIST_CKPT_STRICTNESS=${DIST_CKPT_STRICTNESS}
SLIME_ROOT=${SLIME_ROOT}
PROJECT_ROOT=${PROJECT_ROOT}
SLIME_DATA_ROOT=${SLIME_DATA_ROOT}
PREPARED_DATA_DIR=${PREPARED_DATA_DIR}
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
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH}
SGLANG_DISABLE_OVERLAP_SCHEDULE=${SGLANG_DISABLE_OVERLAP_SCHEDULE}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND}
SGLANG_SAMPLING_BACKEND=${SGLANG_SAMPLING_BACKEND}
SGLANG_GRAMMAR_BACKEND=${SGLANG_GRAMMAR_BACKEND}
SGLANG_503_DEBUG_MODE=${SGLANG_503_DEBUG_MODE}
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
CRITIC_LR=${CRITIC_LR}
CRITIC_LR_HEAD=${CRITIC_LR_HEAD}
ZERO_STAGE=${ZERO_STAGE}
CF_TEACHER_LAMBDA=${CF_TEACHER_LAMBDA}
CF_TEACHER_N_SAMPLES=${CF_TEACHER_N_SAMPLES}
TEACHER_BACKEND=${TEACHER_BACKEND}
TEACHER_API_BASE=${TEACHER_API_BASE}
TEACHER_MODEL_NAME=${TEACHER_MODEL_NAME}
TEACHER_API_STYLE=${TEACHER_API_STYLE}
TEACHER_TIMEOUT=${TEACHER_TIMEOUT}
TEACHER_MAX_RETRIES=${TEACHER_MAX_RETRIES}
TEACHER_REMOTE_BATCH_SIZE=${TEACHER_REMOTE_BATCH_SIZE}
TEACHER_SGLANG_MULTI_SAMPLE=${TEACHER_SGLANG_MULTI_SAMPLE}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE}
TEACHER_TOP_P=${TEACHER_TOP_P}
TEACHER_MAX_NEW_TOKENS=${TEACHER_MAX_NEW_TOKENS}
TEACHER_SYSTEM_PROMPT_ID=${TEACHER_SYSTEM_PROMPT_ID}
TEACHER_PREFLIGHT_TIMEOUT=${TEACHER_PREFLIGHT_TIMEOUT}
SKIP_TEACHER_PREFLIGHT=${SKIP_TEACHER_PREFLIGHT}
TEACHER_CACHE_ENABLE=${TEACHER_CACHE_ENABLE}
TEACHER_CACHE_DIR=${TEACHER_CACHE_DIR}
OPD_KL_COEF=${OPD_KL_COEF}
OPD_TEACHER_RM_URL=${OPD_TEACHER_RM_URL}
G1_USE_EBFT_LOSS=${G1_USE_EBFT_LOSS}
G1_APPLY_DENSE_ATTENTION_MASK=${G1_APPLY_DENSE_ATTENTION_MASK}
G1_QA_MASKING=${G1_QA_MASKING}
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
G1_EBFT_LOGPROB_INDEXING=${G1_EBFT_LOGPROB_INDEXING}
G1_EBFT_ROLLOUT_SAMPLING_MODE=${G1_EBFT_ROLLOUT_SAMPLING_MODE}
G1_EBFT_ROLLOUT_MASK_MODE=${G1_EBFT_ROLLOUT_MASK_MODE}
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
}

write_argv_artifact || echo "[WARN] failed to write ${ARTIFACT_DIR}/argv.sh; continuing without argv artifact" >&2
if write_run_context_artifact; then
  cp "${ARTIFACT_DIR}/run_context.env" "${ARTIFACT_DIR}/hyperparams.env" || \
    echo "[WARN] failed to copy ${ARTIFACT_DIR}/hyperparams.env; continuing without hyperparams artifact" >&2
else
  echo "[WARN] failed to write ${ARTIFACT_DIR}/run_context.env; continuing without run context artifact" >&2
fi

echo "[main-test] G2 + OPD Slime/Megatron run"
echo "[main-test] DEPLOY_LAYOUT=${DEPLOY_LAYOUT} DEPLOY_ROLE=${DEPLOY_ROLE} RAY_ADDRESS=${RAY_ADDRESS}"
echo "[main-test] RUN_NAME=${RUN_NAME}"
echo "[main-test] mode=${G2_OPD_MODE} NUM_EPOCH=${NUM_EPOCH} NUM_ROLLOUT=${NUM_ROLLOUT:-auto} ENABLE_SLIME_EVAL=${ENABLE_SLIME_EVAL} ENABLE_G2_POST_EVAL=${ENABLE_G2_POST_EVAL}"
echo "[main-test] teacher=${TEACHER_API_BASE} style=${TEACHER_API_STYLE} model=${TEACHER_MODEL_NAME} opd_rm=${OPD_TEACHER_RM_URL}"
echo "[main-test] cf_target=${CF_TARGET_MODE} opd_credit=${OPD_CREDIT_ASSIGNMENT} opd_cf_norm=${OPD_CF_SCORE_NORMALIZATION} opd_cf_temp=${OPD_CF_SCORE_TEMPERATURE} opd_kl=${OPD_KL_COEF} opd_kl_application=${OPD_KL_APPLICATION}"
echo "[main-test] teacher_completion lambda=${CF_TEACHER_LAMBDA} teacher_samples=${CF_TEACHER_N_SAMPLES} teacher_cache=${TEACHER_CACHE_ENABLE}"
echo "[main-test] ebft_loss=${G1_USE_EBFT_LOSS} ce_loss_coef=${G1_CE_LOSS_COEF}"
echo "[layout] teacher node: 8 GPU scorer, default TEACHER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=2 DP=4"
echo "[layout] student node: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; actor=${ACTOR_NUM_GPUS_PER_NODE}, critic=${CRITIC_NUM_GPUS_PER_NODE}, rollout=${ROLLOUT_NUM_GPUS} (${ROLLOUT_NUM_GPUS_PER_ENGINE} GPU/engine)"
echo "[preflight] NUM_GPUS=${NUM_GPUS} ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} CRITIC_NUM_NODES=${CRITIC_NUM_NODES} CRITIC_NUM_GPUS_PER_NODE=${CRITIC_NUM_GPUS_PER_NODE} ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} COLOCATE=${COLOCATE} TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}"
echo "[preflight] TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} ACTOR_DP=${DP_SIZE}"
echo "[preflight] SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND} SGLANG_DISABLE_OVERLAP_SCHEDULE=${SGLANG_DISABLE_OVERLAP_SCHEDULE} SGLANG_GRAMMAR_BACKEND=${SGLANG_GRAMMAR_BACKEND}"
echo "[preflight] G1_EBFT_LOGPROB_INDEXING=${G1_EBFT_LOGPROB_INDEXING} G1_EBFT_ROLLOUT_SAMPLING_MODE=${G1_EBFT_ROLLOUT_SAMPLING_MODE} G1_EBFT_ROLLOUT_MASK_MODE=${G1_EBFT_ROLLOUT_MASK_MODE}"
echo "[submit] command:"
print_shell_command "${CMD[@]}"
echo "[artifact] ${ARTIFACT_DIR}"

if [[ "${PRINT_ONLY:-0}" == "1" || "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${SKIP_TEACHER_PREFLIGHT}" == "1" ]]; then
  echo "[preflight] skipping teacher API reachability checks because SKIP_TEACHER_PREFLIGHT=1"
else
  if [[ "${CF_TARGET_MODE}" == "teacher" ]]; then
    if [[ "${TEACHER_API_STYLE}" == "sglang_generate" ]]; then
      if [[ "${TEACHER_API_BASE%/}" == */generate ]]; then
        TEACHER_PREFLIGHT_URL="${TEACHER_API_BASE%/}"
      else
        TEACHER_PREFLIGHT_URL="${TEACHER_API_BASE%/}/generate"
      fi
      TEACHER_PREFLIGHT_METHOD="POST_GENERATE"
    else
      TEACHER_PREFLIGHT_URL="${TEACHER_API_BASE%/}/models"
      TEACHER_PREFLIGHT_METHOD="GET_MODELS"
    fi
    echo "[preflight] checking G2 teacher API reachability: ${TEACHER_PREFLIGHT_URL}"
    if ! TEACHER_PREFLIGHT_URL="${TEACHER_PREFLIGHT_URL}" \
         TEACHER_PREFLIGHT_METHOD="${TEACHER_PREFLIGHT_METHOD}" \
         TEACHER_API_KEY="${TEACHER_API_KEY}" \
         TEACHER_PREFLIGHT_TIMEOUT="${TEACHER_PREFLIGHT_TIMEOUT}" \
         "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
import urllib.request

url = os.environ["TEACHER_PREFLIGHT_URL"]
method = os.environ["TEACHER_PREFLIGHT_METHOD"]
timeout = float(os.environ.get("TEACHER_PREFLIGHT_TIMEOUT", "5"))
api_key = os.environ.get("TEACHER_API_KEY", "EMPTY")
if method == "POST_GENERATE":
    payload = {
        "text": "ping",
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 1,
            "skip_special_tokens": True,
        },
        "return_logprob": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
else:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read(1)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
    then
      echo "[ERROR] G2 teacher API preflight failed for ${TEACHER_PREFLIGHT_URL}" >&2
      echo "[ERROR] Start the teacher service first, or set SKIP_TEACHER_PREFLIGHT=1 for special network environments." >&2
      exit 1
    fi
  else
    echo "[preflight] skipping G2 teacher-completion API check for CF_TARGET_MODE=${CF_TARGET_MODE}"
  fi

  if ! wait_opd_logprob_ready "${OPD_TEACHER_RM_URL}"; then
    echo "[ERROR] OPD teacher RM preflight failed for ${OPD_TEACHER_RM_URL}" >&2
    echo "[ERROR] Set OPD_TEACHER_RM_URL to the SGLang /generate endpoint that can return logprobs." >&2
    exit 1
  fi
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[preflight] PREFLIGHT_ONLY=1; exiting before Ray startup"
  exit 0
fi

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
    "SLIME_HTTP_REQUEST_TIMEOUT",
    "NO_PROXY", "no_proxy",
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
