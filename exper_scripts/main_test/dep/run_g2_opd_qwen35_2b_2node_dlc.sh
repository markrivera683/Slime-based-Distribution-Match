#!/usr/bin/env bash
# PAI-DLC / PyTorchJob entrypoint for 2-node G2+OPD (8+8 GPU).
#
# Paste into DLC "启动命令" (both pods use the same command):
#   bash /mnt/data/distribution-matching-slime/code/slime-0.2.4/exper_scripts/main_test/run_g2_opd_qwen35_2b_2node_dlc.sh
#
# Layout (WORLD_SIZE=2):
#   rank 0 (master pod) -> SGLang 27B teacher, 8 GPU
#   rank 1 (worker pod) -> Slime student (cf_l1oo + OPD), 8 GPU
#
# Do NOT chain build_conda.sh here on every job unless the venv is missing.
# /root is pod-local in DLC, so when BUILD_SLIME_ENV is enabled each pod builds
# its own local venv instead of waiting on another pod's /root.
#
# First DLC job on a fresh image (no /usr/local/cuda-12.9):
#   RECREATE_VENV=true BUILD_SLIME_ENV=true bash .../run_g2_opd_qwen35_2b_2node_dlc.sh
# Each pod auto-installs CUDA 12.9 toolkit from:
#   ${WHEELS_INFRA:-/mnt/data/wheels_infra}/cuda-repo-ubuntu2204-12-9-local_*.deb
# (same OSS path as flash-attn wheels). Set INSTALL_CUDA129_FROM_WHEELS_INFRA=false to skip.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export DEPLOY_MODE="${DEPLOY_MODE:-dlc}"
export DLC_AUTO_ENV="${DLC_AUTO_ENV:-true}"
export BUILD_SLIME_ENV="${BUILD_SLIME_ENV:-auto}"

exec bash "${SCRIPT_DIR}/run_g2_opd_qwen35_2b_2node.sh" "$@"
