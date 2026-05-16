#!/usr/bin/env bash
# Minimal real Ray/Megatron smoke for the trainer-side G1 Megatron/ref embedding path.
#
# This script is owned by the slime G1 migration plan. It calls the EBFT
# diff-dataset launcher only to reuse the existing slime baseline command
# construction and dataset conventions.
set -euo pipefail

EBFT_ROOT="${EBFT_ROOT:-/mnt/data/ebft-distribution-new/code}"
SLIME_ROOT="${SLIME_ROOT:-/mnt/data/distribution-matching-slime/code/slime-0.2.4}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/slime_runtime/Megatron-LM}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"

cd "${EBFT_ROOT}"

MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_PATH}}"
SLIME_TRAIN_DATA="${SLIME_TRAIN_DATA:-/tmp/g1_smoke_short.jsonl}"
REF_LOAD="${REF_LOAD:-/mnt/data/models/Megatron_convert_models/Qwen3.5-4B_torch_dist}"
LOAD_PATH="${LOAD_PATH:-/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_$(date +%m%d_%H%M%S)/mcore}"
SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${SLIME_ROOT}/slime/scripts/models/qwen3.5-4B.sh}"
ALLOW_INFER_MODEL_ARGS="${ALLOW_INFER_MODEL_ARGS:-false}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-376}"
ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL:-false}"
BALANCE_DATA="${BALANCE_DATA:-false}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-false}"
SAVE_INTERVAL="${SAVE_INTERVAL:-999}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# First production smoke uses DP=1 so each trainer rank sees complete
# n_samples_per_prompt groups. DP>1 is supported only with group-aligned splits.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-2}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
RAY_BIN="${RAY_BIN:-/root/venvs/slime/bin/ray}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_g1}"

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

if [[ ! -f "${MODEL_ARGS_SCRIPT}" ]]; then
  echo "[ERROR] MODEL_ARGS_SCRIPT missing: ${MODEL_ARGS_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${REF_LOAD}" || ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
  echo "[ERROR] REF_LOAD must be a Megatron torch-dist checkpoint with latest_checkpointed_iteration.txt: ${REF_LOAD}" >&2
  exit 1
fi
if [[ ! -s "${SLIME_TRAIN_DATA}" ]]; then
  echo "[ERROR] SLIME_TRAIN_DATA missing or empty: ${SLIME_TRAIN_DATA}" >&2
  exit 1
fi
if [[ "${CONTEXT_PARALLEL_SIZE}" != "1" ]]; then
  echo "[ERROR] first G1 Megatron/ref smoke requires CONTEXT_PARALLEL_SIZE=1" >&2
  exit 1
fi
if [[ "${BALANCE_DATA}" != "false" ]]; then
  echo "[ERROR] trainer-side G1 requires BALANCE_DATA=false until group-level balancing is implemented" >&2
  exit 1
fi

NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",") if x.strip()]))
PY
)"
PARALLEL_GROUP_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( PARALLEL_GROUP_SIZE <= 0 || NUM_GPUS % PARALLEL_GROUP_SIZE != 0 )); then
  echo "[ERROR] NUM_GPUS=${NUM_GPUS} must be divisible by TP*PP*CP=${PARALLEL_GROUP_SIZE}" >&2
  exit 1
fi
DP_SIZE=$((NUM_GPUS / PARALLEL_GROUP_SIZE))
if (( ROLLOUT_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "[ERROR] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE} for group-aligned G1 split" >&2
  exit 1
fi

BASE_CMD="$(
  DRY_RUN=true \
  MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT}" \
  ALLOW_INFER_MODEL_ARGS="${ALLOW_INFER_MODEL_ARGS}" \
  MODEL_PATH="${MODEL_PATH}" \
  REF_LOAD="${REF_LOAD}" \
  ADVANTAGE_ESTIMATOR=g1 \
  GROUP_RM=false \
  USE_EBFT_CUSTOM_RM=false \
  CUSTOM_RM_PATH= \
  NUM_ROLLOUT="${NUM_ROLLOUT}" \
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN}" \
  ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL}" \
  BALANCE_DATA="${BALANCE_DATA}" \
  USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE}" \
  TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE}" \
  PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE}" \
  CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  LOAD_PATH="${LOAD_PATH}" \
  SAVE_PATH="${SAVE_PATH}" \
  SLIME_TRAIN_DATA="${SLIME_TRAIN_DATA}" \
  bash scripts/diff_dataset/run_slime_gspo_1node_once.sh | tail -n 1
)"

EXTRA_ARGS=(
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
)
if [[ "${G1_APPLY_DENSE_ATTENTION_MASK:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--g1-megatron-ref-apply-dense-attention-mask)
fi

# Opt-in EBFT policy loss: appends parity-oriented defaults. When CLI flags are
# not registered in slime/utils/arguments.py yet, fail fast (default smoke unchanged).
if [[ "${G1_USE_EBFT_LOSS:-false}" == "true" ]]; then
  _args_py="${SLIME_ROOT}/slime/utils/arguments.py"
  for _needle in "g1-use-ebft-loss" "g1-ce-loss-coef"; do
    if ! grep -qF -- "${_needle}" "${_args_py}"; then
      echo "[ERROR] G1_USE_EBFT_LOSS=true but ${_needle} is not defined in ${_args_py} yet." >&2
      echo "        Drop G1_USE_EBFT_LOSS or add the CLI flags; see refactor_debugging/g1_plan/ebft_loss_validation.md" >&2
      exit 1
    fi
  done
  EXTRA_ARGS+=(--g1-use-ebft-loss --g1-ce-loss-coef 0.03)
fi

eval "set -- ${BASE_CMD}"

if [[ "${G1_USE_EBFT_LOSS:-false}" == "true" ]]; then
  FILTERED_ARGS=()
  while (($#)); do
    case "$1" in
      --use-kl-loss)
        shift
        ;;
      --kl-loss-coef|--kl-loss-type)
        shift
        if (($#)); then
          shift
        fi
        ;;
      --entropy-coef)
        FILTERED_ARGS+=("$1" "0.0")
        shift
        if (($#)); then
          shift
        fi
        ;;
      *)
        FILTERED_ARGS+=("$1")
        shift
        ;;
    esac
  done
  set -- "${FILTERED_ARGS[@]}"
fi

echo "[preflight] NUM_GPUS=${NUM_GPUS} TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} DP=${DP_SIZE}"
echo "[submit] base command:"
printf "%q " "$@" "${EXTRA_ARGS[@]}"
echo

if [[ "${PRINT_ONLY:-0}" == "1" || "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

"${RAY_BIN}" stop --force 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
sleep 3
mkdir -p "${RAY_TMPDIR}"

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
  "RAY_TMPDIR", "PYTHONUNBUFFERED", "G1_RUNTIME_DUMP_PATH",
]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))
PY
)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${RAY_BIN}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "$@" "${EXTRA_ARGS[@]}"
