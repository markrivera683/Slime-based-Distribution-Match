#!/usr/bin/env bash
# PAI-DLC / PyTorchJob entrypoint for 2-node G2 no-teacher-distribution (8+8 GPU).
#
# Paste into DLC "启动命令" (both pods use the same command):
#   bash /mnt/data/distribution-matching-slime/code/slime-0.2.4/exper_scripts/main_test/run_g2_no_teacher_distribution_qwen35_2b_2node_dlc.sh
#
# Layout (WORLD_SIZE=2):
#   rank 0 (master pod) -> Ray head + Slime job submitter, 8 GPU
#   rank 1 (worker pod) -> Ray worker, 8 GPU
#
# Reward: G2 cf_l1oo with cf-target-mode=single, cf-teacher-lambda=0.
# No teacher service, no teacher cache, no OPD.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export DEPLOY_MODE="${DEPLOY_MODE:-dlc}"
export DLC_AUTO_ENV="${DLC_AUTO_ENV:-true}"
export BUILD_SLIME_ENV="${BUILD_SLIME_ENV:-auto}"
export G2_OPD_MODE="${G2_OPD_MODE:-cf_l1oo_no_teacher_distribution}"
export CF_TARGET_MODE="${CF_TARGET_MODE:-single}"
export CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"

exec bash "${SCRIPT_DIR}/run_g2_no_teacher_distribution_qwen35_2b_2node.sh" "$@"
