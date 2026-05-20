#!/usr/bin/env bash
# PAI-DLC / PyTorchJob entrypoint for 2-node G2 cf_l1oo without OPD (8+8 GPU).
#
# Paste into DLC "启动命令" (both pods use the same command):
#   bash /mnt/data/distribution-matching-slime/code/slime-0.2.4/exper_scripts/main_test/run_g2_cf_l1oo_qwen35_2b_2node_dlc.sh
#
# Layout (WORLD_SIZE=2):
#   rank 0 (master pod) -> SGLang 27B teacher, 8 GPU
#   rank 1 (worker pod) -> Slime student (cf_l1oo only, no OPD), 8 GPU
#
# Do NOT chain build_conda.sh here on every job unless the venv is missing;
# this wrapper builds only on rank 0 when needed and rank 1 waits.
#
# First DLC job on a fresh image (no /usr/local/cuda-12.9):
#   RECREATE_VENV=true BUILD_SLIME_ENV=true bash .../run_g2_cf_l1oo_qwen35_2b_2node_dlc.sh
# Rank 0 auto-installs CUDA 12.9 toolkit from:
#   ${WHEELS_INFRA:-/mnt/data/wheels_infra}/cuda-repo-ubuntu2204-12-9-local_*.deb
# (same OSS path as flash-attn wheels). Set INSTALL_CUDA129_FROM_WHEELS_INFRA=false to skip.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export DEPLOY_MODE="${DEPLOY_MODE:-dlc}"
export DLC_AUTO_ENV="${DLC_AUTO_ENV:-true}"
export BUILD_SLIME_ENV="${BUILD_SLIME_ENV:-auto}"
export G2_OPD_MODE="${G2_OPD_MODE:-cf_l1oo}"

exec bash "${SCRIPT_DIR}/run_g2_cf_l1oo_qwen35_2b_2node.sh" "$@"
