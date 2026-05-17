#!/usr/bin/env bash
# Standalone OpenRLHF-G1-style Slime/Megatron smoke for Qwen3.5-2B.
#
# No bash launcher dependencies: this script assembles the train.py CLI, starts
# Ray, submits the job, and records artifacts itself.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EBFT_ROOT="${EBFT_ROOT:-/mnt/data/ebft-distribution-new/code}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/slime_runtime/Megatron-LM}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"

MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-2B}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_PATH}}"
REF_LOAD="${REF_LOAD:-/mnt/data/models/Megatron_convert_models/Qwen3.5-2B_torch_dist}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
SLIME_TRAIN_DATA="${SLIME_TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_slime_qa_100k.jsonl}"
G1_FILTER_TRAIN_DATA="${G1_FILTER_TRAIN_DATA:-true}"
G1_MAX_PROMPT_LABEL_LEN="${G1_MAX_PROMPT_LABEL_LEN:-384}"
G1_FILTERED_SLIME_TRAIN_DATA="${G1_FILTERED_SLIME_TRAIN_DATA:-${SLIME_TRAIN_DATA%.jsonl}_g1_prompt${G1_MAX_PROMPT_LABEL_LEN}.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
MBPP_SLIME_EVAL_DATA="${MBPP_SLIME_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa_slime.jsonl}"
HUMANEVAL_SLIME_EVAL_DATA="${HUMANEVAL_SLIME_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa_slime.jsonl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
NUM_ROLLOUT="${NUM_ROLLOUT:-25}"

ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-1024}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-376}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"

ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL:-false}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-512}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"

G1_USE_EBFT_LOSS="${G1_USE_EBFT_LOSS:-true}"
G1_APPLY_DENSE_ATTENTION_MASK="${G1_APPLY_DENSE_ATTENTION_MASK:-false}"
G1_CE_LOSS_COEF="${G1_CE_LOSS_COEF:-0.03}"

SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

BALANCE_DATA="${BALANCE_DATA:-false}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-false}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/slime/outputs}"
RUN_NAME="${RUN_NAME:-g1_openrlhf_qwen35_2b_smoke_$(date +%m%d_%H%M%S)}"
LOAD_PATH="${LOAD_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/mcore}"
SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
SMOKE_ARTIFACT_DIR="${SMOKE_ARTIFACT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/smoke_artifacts}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_g1}"

SMOKE_DRIVER_LOG="${SMOKE_ARTIFACT_DIR}/ray_job_driver.log"

if [[ -f "${SLIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ENV_FILE}"
fi

export PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}:${EBFT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export RAY_TMPDIR
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"

ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-false}"
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

require_file "${SLIME_ROOT}/${TRAIN_ENTRYPOINT}"
require_dir "${MEGATRON_PATH}"
require_dir "${MODEL_PATH}"
require_dir "${REF_LOAD}"
require_file "${RAY_BIN}"
require_file "${SLIME_TRAIN_DATA}"

if [[ "${G1_FILTER_TRAIN_DATA}" == "true" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/filter_g1_prompt_length.py" \
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
    "${PYTHON_BIN}" "${EBFT_ROOT}/scripts/diff_dataset/prepare_slime_jsonl.py" \
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
if [[ "${COLOCATE}" == "false" ]]; then
  if (( ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
    echo "[ERROR] ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
    exit 1
  fi
fi
DP_SIZE=$((ACTOR_NUM_GPUS_PER_NODE / PARALLEL_GROUP_SIZE))
if (( ROLLOUT_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "[ERROR] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi

mkdir -p "${SMOKE_ARTIFACT_DIR}" "${LOAD_PATH}" "${SAVE_PATH}" "${RAY_TMPDIR}"

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
  --num-rollout "${NUM_ROLLOUT}"
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
  --entropy-coef 0.0
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --custom-generate-function-path slime.rollout.g1_embedding.generate_fixed_length_for_g1
  --use-whitening
  --alignment-rew-coef 1.0
  --diversity-rew-coef 1.0
  --g1-prompt-length 384
  --g1-context-length 8
  --g1-generate-length 8
  --g1-stride 8
  --g1-response-length 376
  --g1-hidden-state-method last_only
  --g1-tokenizer-path "${HF_CHECKPOINT}"
  --g1-embedding-source megatron_ref
  --g1-reward-location trainer
  --g1-megatron-ref-forward-mode openrlhf_exact
  --g1-use-ebft-loss
  --g1-ce-loss-coef "${G1_CE_LOSS_COEF}"
)
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

printf "%q " "${CMD[@]}" >"${SMOKE_ARTIFACT_DIR}/argv.sh"
printf "\n" >>"${SMOKE_ARTIFACT_DIR}/argv.sh"
cat >"${SMOKE_ARTIFACT_DIR}/run_context.env" <<EOF
RUN_NAME=${RUN_NAME}
LOAD_PATH=${LOAD_PATH}
SAVE_PATH=${SAVE_PATH}
SLIME_TRAIN_DATA=${SLIME_TRAIN_DATA}
NUM_GPUS=${NUM_GPUS}
DP_SIZE=${DP_SIZE}
ENABLE_ASYNC_TRAIN=${ENABLE_ASYNC_TRAIN}
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}
COLOCATE=${COLOCATE}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}
G1_APPLY_DENSE_ATTENTION_MASK=${G1_APPLY_DENSE_ATTENTION_MASK}
EOF

echo "[preflight] NUM_GPUS=${NUM_GPUS} ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} COLOCATE=${COLOCATE} TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}"
echo "[preflight] TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} ACTOR_DP=${DP_SIZE}"
echo "[submit] command:"
printf "%q " "${CMD[@]}"
echo
echo "[artifact] ${SMOKE_ARTIFACT_DIR}"

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
  2>&1 | tee "${SMOKE_DRIVER_LOG}"
SUBMIT_STATUS=${PIPESTATUS[0]}
set -e

printf "%s\n" "${SUBMIT_STATUS}" >"${SMOKE_ARTIFACT_DIR}/ray_job_exit_status.txt"
if [[ -e "${RAY_TMPDIR}/session_latest/logs" ]]; then
  tar -C "${RAY_TMPDIR}/session_latest" -czf "${SMOKE_ARTIFACT_DIR}/ray_session_latest_logs.tgz" logs || true
fi
exit "${SUBMIT_STATUS}"
