#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
OPENRLHF_REPO="${OPENRLHF_REPO:-/mnt/data/ebft-distribution-new/code}"
SLIME_ROOT="${SLIME_ROOT:-/mnt/data/distribution-matching-slime/code/slime-0.2.4}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MEGATRON_DUMP="${MEGATRON_DUMP:-artifacts/g1_runtime_megatron_ref_mask_applied_full_group_compact_dump.pt}"
OPENRLHF_DUMP="${OPENRLHF_DUMP:-artifacts/g1_runtime_openrlhf_critic_mask_applied_full_group_compact_dump.pt}"
REPORT_OUT="${REPORT_OUT:-g1_runtime_parity_report_mask_applied_full_group_compact.md}"

export PYTHONPATH="${SLIME_ROOT}:${OPENRLHF_REPO}:${PYTHONPATH:-}"

cd "${BUNDLE_DIR}"

"${PYTHON_BIN}" scripts/dump_openrlhf_g1_from_megatron_dump.py \
  --megatron-dump "${MEGATRON_DUMP}" \
  --out "${OPENRLHF_DUMP}" \
  --model-path "${MODEL_PATH}" \
  --openrlhf-repo "${OPENRLHF_REPO}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

"${PYTHON_BIN}" scripts/compare_g1_runtime_parity.py \
  --megatron-dump "${MEGATRON_DUMP}" \
  --openrlhf-dump "${OPENRLHF_DUMP}" \
  --out "${REPORT_OUT}"
