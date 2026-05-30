#!/usr/bin/env bash
# PAI-DLC / PyTorchJob smoke entrypoint for 2-node G2+OPD+EffOPD.
#
# Paste this same command into DLC "启动命令" for both pods:
#   bash /mnt/data/distribution-matching-slime/code/slime-0.2.4/exper_scripts/main_test/run_g2_opd_qwen35_2b_effopd_smoke_2node_dlc.sh
#
# Layout:
#   rank 0 -> SGLang 27B teacher on 8 GPUs
#   rank 1 -> Slime student G2 cf_l1oo + SGLang OPD + EffOPD smoke on 8 GPUs
#
# Defaults are intentionally conservative:
#   - EffOPD is enabled in shadow/proxy smoke mode (`opd_kl_shadow_cf`) by default.
#   - NUM_ROLLOUT=4 triggers steps 1/2/4.
#   - SAVE_INTERVAL=1 persists sidecar state every rollout.
#   - The main launchers keep their real `combined_gate` defaults; this wrapper
#     opts into mechanism-only smoke unless EFFOPD_VALIDATION_MODE is overridden.
#
# To exercise real D_v-gated extrapolation after shadow smoke passes:
#   EFFOPD_VALIDATION_MODE=combined_gate EFFOPD_MAX_TRIGGERS=1 NUM_ROLLOUT=2 bash ...
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export DEPLOY_MODE="${DEPLOY_MODE:-dlc}"
export DLC_AUTO_ENV="${DLC_AUTO_ENV:-true}"
export BUILD_SLIME_ENV="${BUILD_SLIME_ENV:-auto}"

export ENABLE_EFFOPD="${ENABLE_EFFOPD:-true}"
export EFFOPD_VALIDATION_MODE="${EFFOPD_VALIDATION_MODE:-opd_kl_shadow_cf}"
export EFFOPD_DV_SIZE="${EFFOPD_DV_SIZE:-32}"
export EFFOPD_DV_SEED="${EFFOPD_DV_SEED:-42}"
export EFFOPD_MAX_K="${EFFOPD_MAX_K:-5}"
export EFFOPD_LR_DECAY="${EFFOPD_LR_DECAY:-0.5}"
export EFFOPD_MAX_TRIGGERS="${EFFOPD_MAX_TRIGGERS:-3}"
export EFFOPD_FORCE_WEIGHT_SYNC="${EFFOPD_FORCE_WEIGHT_SYNC:-true}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export RUN_NAME="${RUN_NAME:-g2_opd_effopd_dlc_smoke_$(date +%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_g2_opd_qwen35_2b_2node_dlc.sh" "$@"
