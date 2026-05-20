#!/usr/bin/env bash
# Same-node throughput tuning wrapper:
#   - teacher SGLang server runs on physical GPUs 0,1,2,3
#   - Slime student smoke runs on physical GPUs 4,5,6,7
#
# This wrapper keeps the main G2 smoke recipe intact while setting the GPU
# topology needed for a 4-GPU student. With only 4 student GPUs, actor and
# critic must be colocated with rollout; the non-colocated default needs extra
# rollout GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export COLOCATE="${COLOCATE:-true}"

export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
export CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

export TEACHER_API_BASE="${TEACHER_API_BASE:-http://127.0.0.1:30000}"
export TEACHER_API_STYLE="${TEACHER_API_STYLE:-sglang_generate}"
export TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
export TEACHER_SGLANG_MULTI_SAMPLE="${TEACHER_SGLANG_MULTI_SAMPLE:-true}"
export TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-32}"
export TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-240}"
# Throughput tuning must measure real teacher service latency/QPS, so disable
# cache by default. Override TEACHER_CACHE_ENABLE=true only when testing cache behavior.
export TEACHER_CACHE_ENABLE="${TEACHER_CACHE_ENABLE:-false}"

# Keep throughput tuning runs short by default; override NUM_ROLLOUT for longer
# measurements once a configuration looks healthy.
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"

exec bash "${SCRIPT_DIR}/run_g2_openrlhf_qwen35_2b_smoke.sh"
