#!/usr/bin/env bash
# Manual launcher for the OPD CF-L1OO smoke test.
#
# Default mode uses the local mock OPD logprob endpoint, so it validates the
# training plumbing without needing a separate teacher server.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

usage() {
  cat <<'USAGE'
Usage:
  refactor_debugging/OPD/smoke_scripts/manual_opd_cf_l1oo_smoke.sh

Common modes:
  SMOKE_MODE=mock       Use local mock OPD logprob server. Default.
  SMOKE_MODE=real       Use a real OPD_TEACHER_RM_URL /generate endpoint.

Run stages:
  SMOKE_STAGE=full      Run the full smoke. Default.
  SMOKE_STAGE=preflight Only check config and OPD logprob API reachability.
  SMOKE_STAGE=print     Print the train.py command without running it.

Examples:
  SMOKE_STAGE=preflight refactor_debugging/OPD/smoke_scripts/manual_opd_cf_l1oo_smoke.sh

  SMOKE_MODE=real \
  OPD_TEACHER_RM_URL=http://127.0.0.1:30000/generate \
  refactor_debugging/OPD/smoke_scripts/manual_opd_cf_l1oo_smoke.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SMOKE_MODE="${SMOKE_MODE:-mock}"
SMOKE_STAGE="${SMOKE_STAGE:-full}"

case "${SMOKE_MODE}" in
  mock)
    export OPD_SMOKE_MOCK_RM=true
    ;;
  real)
    export OPD_SMOKE_MOCK_RM=false
    if [[ -z "${OPD_TEACHER_RM_URL:-}" ]]; then
      echo "[ERROR] SMOKE_MODE=real requires OPD_TEACHER_RM_URL, e.g. http://127.0.0.1:30000/generate" >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] SMOKE_MODE must be mock or real, got: ${SMOKE_MODE}" >&2
    exit 1
    ;;
esac

case "${SMOKE_STAGE}" in
  full)
    unset PREFLIGHT_ONLY
    unset PRINT_ONLY
    ;;
  preflight)
    export PREFLIGHT_ONLY=1
    unset PRINT_ONLY
    ;;
  print)
    export PRINT_ONLY=1
    unset PREFLIGHT_ONLY
    ;;
  *)
    echo "[ERROR] SMOKE_STAGE must be full, preflight, or print, got: ${SMOKE_STAGE}" >&2
    exit 1
    ;;
esac

export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/slime/outputs}"
export NUM_EPOCH="${NUM_EPOCH:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-none}"

export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-4}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.55}"
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"
export SLIME_SGLANG_HEALTH_TIMEOUT="${SLIME_SGLANG_HEALTH_TIMEOUT:-5}"
export SLIME_SGLANG_HEALTH_MAX_WAIT="${SLIME_SGLANG_HEALTH_MAX_WAIT:-180}"

export RUN_NAME="${RUN_NAME:-opd_cf_l1oo_manual_${SMOKE_MODE}_$(date +%m%d_%H%M%S)}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_opd_${RUN_NAME##*_}}"

echo "[manual-smoke] mode=${SMOKE_MODE} stage=${SMOKE_STAGE} run=${RUN_NAME}"
echo "[manual-smoke] output=${OUTPUT_ROOT}/${RUN_NAME}"
echo "[manual-smoke] ray_tmp=${RAY_TMPDIR}"

cd "${SLIME_ROOT}"
exec "${SCRIPT_DIR}/run_opd_cf_l1oo_smoke.sh" "$@"
