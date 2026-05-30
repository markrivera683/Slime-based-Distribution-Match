#!/usr/bin/env bash
# Mechanism-validation smoke for G2 cf_l1oo + SGLang OPD + EffOPD.
#
# G2 here means cf_l1oo reward/distribution matching. OPD is the SGLang
# teacher-logprob distillation signal. This wrapper intentionally defaults to
# the shadow/proxy EffOPD smoke path (`opd_kl_shadow_cf`) so it validates
# trigger/state/logging without changing the production G2+OPD objective. The
# main launcher defaults remain for real `combined_gate` validation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_EFFOPD="${ENABLE_EFFOPD:-true}"
export EFFOPD_DV_SIZE="${EFFOPD_DV_SIZE:-32}"
export EFFOPD_VALIDATION_MODE="${EFFOPD_VALIDATION_MODE:-opd_kl_shadow_cf}"
export EFFOPD_MAX_TRIGGERS="${EFFOPD_MAX_TRIGGERS:-3}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export RUN_NAME="${RUN_NAME:-g2_opd_effopd_smoke_$(date +%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_g2_opd_qwen35_2b_main.sh" "$@"
