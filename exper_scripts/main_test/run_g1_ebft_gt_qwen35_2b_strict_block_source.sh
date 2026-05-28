#!/usr/bin/env bash
# Strict block-source variant of run_g1_ebft_gt_qwen35_2b_main.sh.
#
# This keeps the G1 EBFT GT launcher defaults intact and only opts into the
# strict EBFT source-row gather contract via --g1-ebft-logprob-indexing.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export G1_EBFT_LOGPROB_INDEXING="${G1_EBFT_LOGPROB_INDEXING:-strict_block_source}"
export RUN_NAME="${RUN_NAME:-g1_ebft_gt_qwen35_2b_strict_block_source_$(date +%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_g1_ebft_gt_qwen35_2b_main.sh" "$@"
