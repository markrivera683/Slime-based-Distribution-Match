#!/usr/bin/env bash
# Student node for two-node G2 cf_l1oo training without OPD.
#
# Reward / training recipe:
#   --distribution-reward-type cf_l1oo --cf-target-mode teacher
#   --cf-teacher-lambda / --cf-teacher-n-samples
#   remote teacher completions for counterfactual target
#   frozen critic: --critic-lr 0 --critic-lr-head 0
#
# Example:
#   TEACHER_HOST=10.0.0.1 \
#   RAY_NODE_IP_ADDRESS=10.0.0.2 \
#   bash exper_scripts/main_test/run_g2_cf_l1oo_qwen35_2b_main_2node_student.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${TEACHER_HOST:-}" ]]; then
  echo "[ERROR] Set TEACHER_HOST to the teacher node's routable IP or hostname." >&2
  exit 1
fi

export DEPLOY_LAYOUT="${DEPLOY_LAYOUT:-two_node}"
export DEPLOY_ROLE="${DEPLOY_ROLE:-student}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# --- G1-style Slime async: Megatron train + dedicated SGLang rollout GPUs ---
export ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-true}"
export COLOCATE="${COLOCATE:-false}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train_async.py}"

export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
# G2 needs critic GPUs in addition to G1's actor+rollout; keep 4 rollout engines.
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
export CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"
export CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"

# --- Standard G2 cf_l1oo only (no OPD / teacher logprob KL) ---
export G2_OPD_MODE="${G2_OPD_MODE:-cf_l1oo}"
export CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
export CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-4}"
export CRITIC_LR="${CRITIC_LR:-0}"
export CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"
export ZERO_STAGE="${ZERO_STAGE:-3}"

# Megatron ref embeddings feed the G1 advantage path that cf_l1oo sits on.
export G1_EMBEDDING_SOURCE="${G1_EMBEDDING_SOURCE:-megatron_ref}"
export G1_REWARD_LOCATION="${G1_REWARD_LOCATION:-trainer}"
export G1_REF_FORWARD_MODE="${G1_REF_FORWARD_MODE:-openrlhf_exact}"
export USE_WHITENING="${USE_WHITENING:-true}"

# --- SGLang student rollout (2B generation, separate from remote 27B teacher) ---
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"

# --- Remote teacher node: cf_l1oo online completions only ---
export TEACHER_BACKEND="${TEACHER_BACKEND:-remote}"
export TEACHER_PORT="${TEACHER_PORT:-30000}"
export TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
export TEACHER_API_STYLE="${TEACHER_API_STYLE:-sglang_generate}"
export TEACHER_SGLANG_MULTI_SAMPLE="${TEACHER_SGLANG_MULTI_SAMPLE:-true}"
export TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-16}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
export TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
export TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"
export TEACHER_CACHE_ENABLE="${TEACHER_CACHE_ENABLE:-false}"

exec bash "${SCRIPT_DIR}/run_g2_cf_l1oo_qwen35_2b_main.sh" "$@"
