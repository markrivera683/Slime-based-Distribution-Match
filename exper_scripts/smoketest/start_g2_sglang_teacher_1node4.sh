#!/usr/bin/env bash
# Start a 4-GPU SGLang teacher on the same node as a 4-GPU Slime student.
#
# Default physical split:
#   teacher: GPUs 0,1,2,3
#   student: GPUs 4,5,6,7
#
# On 80G A100, TP=2, DP=2 is the conservative throughput-tuning start point
# for Qwen3.5-27B. If this leaves enough headroom and your SGLang build supports
# it well, try TP_SIZE=1 DP_SIZE=4 for more replicas.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TP_SIZE="${TP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-2}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"

exec bash "${SCRIPT_DIR}/start_g2_sglang_teacher_1node8.sh"
