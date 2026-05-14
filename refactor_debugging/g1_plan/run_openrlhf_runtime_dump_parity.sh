#!/usr/bin/env bash

set -euo pipefail

SLIME_ROOT="${SLIME_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
G1_PLAN_DIR="${G1_PLAN_DIR:-${SLIME_ROOT}/refactor_debugging/g1_plan}"
EBFT_REPO="${EBFT_REPO:-/mnt/data/ebft-distribution-new/code}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/workspace}"
STUDENT_VENV="${STUDENT_VENV:-${LOCAL_ROOT}/venvs/.venv}"
OPENRLHF_PYTHON="${OPENRLHF_PYTHON:-${STUDENT_VENV}/bin/python}"
SLIME_PYTHON="${SLIME_PYTHON:-/root/venvs/slime/bin/python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${G1_PLAN_DIR}/artifacts}"
DUMP_PATH="${DUMP_PATH:-${ARTIFACT_DIR}/openrlhf_g1_runtime_fixture.pt}"
RUN_SETUP_ENV="${RUN_SETUP_ENV:-1}"
SKIP_TEACHER="${SKIP_TEACHER:-1}"
GENERATE_LENGTH="${GENERATE_LENGTH:-2}"

mkdir -p "${ARTIFACT_DIR}"

if [[ "${RUN_SETUP_ENV}" == "1" ]]; then
  echo "[1/3] setup EBFT/OpenRLHF env"
  (
    cd "${EBFT_REPO}"
    SKIP_TEACHER="${SKIP_TEACHER}" \
      LOCAL_ROOT="${LOCAL_ROOT}" \
      STUDENT_VENV="${STUDENT_VENV}" \
      bash scripts/setup_env.sh
  )
else
  echo "[1/3] skip setup_env.sh because RUN_SETUP_ENV=${RUN_SETUP_ENV}"
fi

if [[ ! -x "${OPENRLHF_PYTHON}" ]]; then
  echo "[ERROR] OPENRLHF_PYTHON is not executable: ${OPENRLHF_PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${SLIME_PYTHON}" ]]; then
  echo "[ERROR] SLIME_PYTHON is not executable: ${SLIME_PYTHON}" >&2
  exit 1
fi

echo "[2/3] dump OpenRLHF G1 runtime fixture"
PYTHONPATH="${EBFT_REPO}:${PYTHONPATH:-}" \
  "${OPENRLHF_PYTHON}" "${G1_PLAN_DIR}/dump_openrlhf_g1_runtime_fixture.py" \
  --out "${DUMP_PATH}" \
  --generate-length "${GENERATE_LENGTH}"

echo "[3/3] check slime golden parity"
PYTHONPATH="${SLIME_ROOT}:${PYTHONPATH:-}" \
  "${SLIME_PYTHON}" "${G1_PLAN_DIR}/check_slime_g1_dump_parity.py" \
  --dump "${DUMP_PATH}"

echo "[done] OpenRLHF runtime dump parity passed: ${DUMP_PATH}"
