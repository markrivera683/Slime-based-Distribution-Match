#!/usr/bin/env bash
# Smoke wrapper for the OPD on-policy EBFT credit-assignment baseline.
#
# This keeps the same OPD teacher-logprob signal as OPD-CF-L1OO but replaces
# CF distribution LOO with pointwise EBFT-style credit plus RLOO shaping.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${SCRIPT_DIR}/opd_smoke_common.sh"

export CF_TARGET_MODE="${CF_TARGET_MODE:-opd_onpolicy}"
export OPD_CREDIT_ASSIGNMENT="${OPD_CREDIT_ASSIGNMENT:-ebft}"
export OPD_CF_SCORE_NORMALIZATION="${OPD_CF_SCORE_NORMALIZATION:-mean}"
export OPD_CF_SCORE_TEMPERATURE="${OPD_CF_SCORE_TEMPERATURE:-1.0}"
export OPD_KL_APPLICATION="${OPD_KL_APPLICATION:-auto}"
export G2_OPD_MODE="${G2_OPD_MODE:-opd_ebft_credit_sglang}"
export ENABLE_EFFOPD="${ENABLE_EFFOPD:-false}"
export G1_USE_EBFT_LOSS="${G1_USE_EBFT_LOSS:-false}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export RUN_NAME="${RUN_NAME:-opd_ebft_credit_smoke_$(date +%m%d_%H%M%S)}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-none}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_opd_${RUN_NAME##*_}}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-4}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.55}"
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"
export SLIME_SGLANG_HEALTH_TIMEOUT="${SLIME_SGLANG_HEALTH_TIMEOUT:-5}"
export SLIME_SGLANG_HEALTH_MAX_WAIT="${SLIME_SGLANG_HEALTH_MAX_WAIT:-180}"

start_mock_opd_rm_if_requested

bash "${SLIME_ROOT}/exper_scripts/main_test/run_g2_opd_qwen35_2b_main.sh" "$@"
