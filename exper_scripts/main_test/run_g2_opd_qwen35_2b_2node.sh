#!/usr/bin/env bash
# Orchestrate G2+OPD teacher (8 GPU) + student (8 GPU) in one entrypoint.
#
# Modes (auto-detected unless DEPLOY_MODE is set):
#   dlc         - 2 DLC/PyTorch pods: rank0=teacher, rank1=student (WORLD_SIZE=2)
#   ssh         - SSH teacher up on TEACHER_NODE, run student on this host
#   local_split - one 8-GPU box: teacher GPUs 0-3, student GPUs 4-7 (debug only)
#
# PAI-DLC 2-pod (recommended startup command for both pods):
#   bash .../exper_scripts/main_test/run_g2_opd_qwen35_2b_2node_dlc.sh
#
# DLC (same command on both pods):
#   bash exper_scripts/main_test/run_g2_opd_qwen35_2b_2node.sh
#
# SSH (run on student node):
#   TEACHER_NODE=teacher-host TEACHER_NODE_IP=10.0.0.1 \
#   bash exper_scripts/main_test/run_g2_opd_qwen35_2b_2node.sh
#
# Local 4+4 debug:
#   DEPLOY_MODE=local_split bash exper_scripts/main_test/run_g2_opd_qwen35_2b_2node.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"

TEACHER_START_SCRIPT="${TEACHER_START_SCRIPT:-${SCRIPT_DIR}/start_g2_sglang_teacher_2node.sh}"
TEACHER_START_SCRIPT_LOCAL="${TEACHER_START_SCRIPT_LOCAL:-${SLIME_ROOT}/exper_scripts/smoketest/start_g2_sglang_teacher_1node4.sh}"
STUDENT_SCRIPT="${STUDENT_SCRIPT:-${SCRIPT_DIR}/run_g2_opd_qwen35_2b_main_2node_student.sh}"

TEACHER_PORT="${TEACHER_PORT:-30000}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-3600}"
TEACHER_POLL_SECONDS="${TEACHER_POLL_SECONDS:-10}"
TEACHER_NODE_RANK="${TEACHER_NODE_RANK:-0}"
STUDENT_NODE_RANK="${STUDENT_NODE_RANK:-1}"

TEACHER_NODE="${TEACHER_NODE:-}"
TEACHER_NODE_IP="${TEACHER_NODE_IP:-}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}"

DEPLOY_MODE="${DEPLOY_MODE:-auto}"
DLC_AUTO_ENV="${DLC_AUTO_ENV:-true}"
BUILD_SLIME_ENV="${BUILD_SLIME_ENV:-false}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${SLIME_ROOT}/build_conda.sh}"
SLIME_ENV_WAIT_SECONDS="${SLIME_ENV_WAIT_SECONDS:-7200}"
DLC_LOCAL_ROOT="${DLC_LOCAL_ROOT:-/mnt/workspace}"
TEACHER_LOG="${TEACHER_LOG:-${DLC_LOCAL_ROOT}/slime_logs/g2_opd_teacher.log}"
TEACHER_PID_FILE="${TEACHER_PID_FILE:-${DLC_LOCAL_ROOT}/slime_logs/g2_opd_teacher.pid}"

setup_dlc_runtime_env() {
  # Mirrors EBFT run_G2_rebase_2node_once.sh: compile caches + Ray on local ext4,
  # HF weights on OSS, teacher cache SQLite off ossfs2.
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${DLC_LOCAL_ROOT}/.torch_extensions}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DLC_LOCAL_ROOT}/.triton_cache}"
  export RAY_TMPDIR="${RAY_TMPDIR:-${DLC_LOCAL_ROOT}/ray_slime_g2_opd}"
  export TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-${DLC_LOCAL_ROOT}/teacher_cache_shared}"
  export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
  [[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
  export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"
  mkdir -p "${DLC_LOCAL_ROOT}/slime_logs" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${RAY_TMPDIR}"
}

setup_dlc_stable_run_name() {
  if [[ -n "${RUN_NAME:-}" ]]; then
    return 0
  fi
  local job_id
  job_id="$(hostname | sed -E 's/^(dlc[a-z0-9]+)-(master|worker)-[0-9]+$/\1/' || true)"
  if [[ -n "${job_id}" && "${job_id}" != "$(hostname)" ]]; then
    export RUN_NAME="g2_opd_${job_id}"
  fi
}

need_slime_build() {
  [[ "${BUILD_SLIME_ENV}" == "true" || "${BUILD_SLIME_ENV}" == "1" ]] && return 0
  [[ "${BUILD_SLIME_ENV}" == "auto" ]] || return 1
  [[ ! -f "${SLIME_ENV_FILE}" ]] && return 0
  [[ ! -x "${PYTHON_BIN}" ]] && return 0
  return 1
}

wait_for_slime_env() {
  local waited=0
  echo "[dlc] rank=${DLC_NODE_RANK:-?} waiting for ${SLIME_ENV_FILE} (timeout ${SLIME_ENV_WAIT_SECONDS}s)"
  while [[ ! -f "${SLIME_ENV_FILE}" || ! -x "${PYTHON_BIN}" ]]; do
    sleep 10
    waited=$((waited + 10))
    if (( waited >= SLIME_ENV_WAIT_SECONDS )); then
      echo "[ERROR] timed out waiting for Slime env at ${SLIME_ENV_FILE}" >&2
      exit 1
    fi
  done
  echo "[dlc] Slime env ready after ${waited}s"
}

maybe_install_cuda129_for_dlc_build() {
  if [[ "${DLC_MODE}" != "true" ]]; then
    return 0
  fi
  if [[ "${INSTALL_CUDA129_FROM_WHEELS_INFRA:-true}" != "true" && "${INSTALL_CUDA129_FROM_WHEELS_INFRA:-true}" != "1" ]]; then
    return 0
  fi
  local installer="${SLIME_ROOT}/scripts/install_cuda129_from_wheels_infra.sh"
  if [[ -x /usr/local/cuda-12.9/bin/nvcc || ! -f "${installer}" ]]; then
    return 0
  fi
  echo "[dlc] installing CUDA 12.9 toolkit from wheels_infra before env build"
  bash "${installer}"
}

maybe_build_slime_env() {
  if [[ "${DLC_MODE}" != "true" || "${DLC_AUTO_ENV}" != "true" ]]; then
    return 0
  fi
  if ! need_slime_build; then
    return 0
  fi
  if [[ ! -f "${BUILD_SCRIPT}" ]]; then
    echo "[ERROR] BUILD_SLIME_ENV requested but BUILD_SCRIPT missing: ${BUILD_SCRIPT}" >&2
    exit 1
  fi
  # /root is pod-local in DLC, so each pod must build its own local venv.
  # Waiting for rank 0's /root/slime_runtime/slime_env.sh would deadlock rank 1.
  maybe_install_cuda129_for_dlc_build
  echo "[dlc] rank=${DLC_NODE_RANK} building local Slime env via ${BUILD_SCRIPT}"
  bash "${BUILD_SCRIPT}"
}

source_slime_env() {
  if [[ -f "${SLIME_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${SLIME_ENV_FILE}"
  fi
  if [[ -d "${VIRTUAL_ENV:-}/bin" ]]; then
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
  elif [[ -d "$(dirname "${PYTHON_BIN}")" ]]; then
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  fi
}

resolve_teacher_host() {
  local host="$1"
  if [[ "${host}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "${host}"
    return 0
  fi
  local ip
  ip="$(resolve_host_ip "${host}" 2>/dev/null || true)"
  if [[ -n "${ip}" ]]; then
    echo "${ip}"
  else
    echo "${host}"
  fi
}

resolve_host_ip() {
  local host="$1"
  if [[ "${host}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "${host}"
    return 0
  fi
  getent ahostsv4 "${host}" | awk 'NR==1 {print $1}'
}

get_local_ip() {
  hostname -I 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i !~ /^127\./ && $i !~ /:/) {print $i; exit}}'
}

detect_dlc_mode() {
  DLC_MODE="false"
  DLC_NODE_RANK=""
  DLC_MASTER_ADDR=""
  DLC_WORLD_SIZE="${WORLD_SIZE:-${PET_WORLD_SIZE:-1}}"
  if [[ -n "${PET_NODE_RANK:-}" ]]; then
    DLC_MODE="true"
    DLC_NODE_RANK="${PET_NODE_RANK}"
    DLC_MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
  elif [[ -n "${RANK:-}" && -n "${MASTER_ADDR:-}" && "${DLC_WORLD_SIZE:-1}" -gt 1 ]]; then
    DLC_MODE="true"
    DLC_NODE_RANK="${RANK}"
    DLC_MASTER_ADDR="${MASTER_ADDR}"
  fi
}

detect_deploy_mode() {
  detect_dlc_mode
  if [[ "${DEPLOY_MODE}" != "auto" ]]; then
    return 0
  fi
  if [[ "${DLC_MODE}" == "true" ]]; then
    DEPLOY_MODE="dlc"
    return 0
  fi
  if [[ "${LOCAL_SPLIT:-false}" == "true" ]]; then
    DEPLOY_MODE="local_split"
    return 0
  fi
  if [[ -n "${TEACHER_NODE}" || -n "${TEACHER_NODE_IP}" ]]; then
    DEPLOY_MODE="ssh"
    return 0
  fi
  echo "[ERROR] Could not detect deploy mode. Set one of:" >&2
  echo "  DEPLOY_MODE=dlc|ssh|local_split" >&2
  echo "  LOCAL_SPLIT=true" >&2
  echo "  TEACHER_NODE / TEACHER_NODE_IP (ssh)" >&2
  echo "  or run inside a 2-pod DLC job (RANK + MASTER_ADDR + WORLD_SIZE=2)" >&2
  exit 1
}

wait_teacher_ready() {
  local host="$1"
  local port="$2"
  local url="http://${host}:${port}/generate"
  echo "[orchestrator] waiting for teacher at ${url} (timeout ${TEACHER_WAIT_SECONDS}s)"
  TEACHER_PREFLIGHT_URL="${url}" \
  TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}" \
  TEACHER_PREFLIGHT_TIMEOUT="${TEACHER_POLL_SECONDS}" \
  TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

url = os.environ["TEACHER_PREFLIGHT_URL"]
timeout = float(os.environ.get("TEACHER_PREFLIGHT_TIMEOUT", "10"))
wait_budget = float(os.environ.get("TEACHER_WAIT_SECONDS", "3600"))
api_key = os.environ.get("TEACHER_API_KEY", "EMPTY")
payload = {
    "text": "ping",
    "sampling_params": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 1,
        "skip_special_tokens": True,
    },
    "return_logprob": False,
}
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
deadline = time.time() + wait_budget
last_err = None
while time.time() < deadline:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(1)
        print(f"[orchestrator] teacher ready: {url}", flush=True)
        raise SystemExit(0)
    except Exception as exc:
        last_err = exc
        time.sleep(timeout)
print(f"[orchestrator] teacher not ready: {last_err}", file=sys.stderr, flush=True)
raise SystemExit(1)
PY
}

cleanup_local_teacher() {
  if [[ -f "${TEACHER_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${TEACHER_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[orchestrator] stopping local teacher pid=${pid}"
      kill "${pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${TEACHER_PID_FILE}"
  fi
}

start_local_teacher_background() {
  local start_script="$1"
  shift
  mkdir -p "$(dirname "${TEACHER_LOG}")" "$(dirname "${TEACHER_PID_FILE}")"
  echo "[orchestrator] starting local teacher: ${start_script}"
  # shellcheck disable=SC2090
  nohup env "$@" bash "${start_script}" >"${TEACHER_LOG}" 2>&1 &
  echo $! >"${TEACHER_PID_FILE}"
  echo "[orchestrator] teacher pid=$(cat "${TEACHER_PID_FILE}") log=${TEACHER_LOG}"
}

start_ssh_teacher_background() {
  local teacher_ip="$1"
  local ssh_target
  if [[ -n "${SSH_USER}" ]]; then
    ssh_target="${SSH_USER}@${teacher_ip}"
  else
    ssh_target="${teacher_ip}"
  fi
  echo "[orchestrator] starting remote teacher via ssh ${ssh_target}"
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${ssh_target}" "bash -lc '
    set -euo pipefail
    if [[ -f \"${SLIME_ENV_FILE}\" ]]; then source \"${SLIME_ENV_FILE}\"; fi
    mkdir -p \"$(dirname \"${TEACHER_LOG}\")\"
    nohup bash \"${TEACHER_START_SCRIPT}\" >\"${TEACHER_LOG}\" 2>&1 &
    echo \$! >\"${TEACHER_PID_FILE}\"
    echo remote_teacher_pid=\$(cat \"${TEACHER_PID_FILE}\")
  '"
}

run_student() {
  local teacher_host="$1"
  export TEACHER_HOST="${teacher_host}"
  export TEACHER_PORT
  export DEPLOY_LAYOUT=two_node
  export DEPLOY_ROLE=student
  export TEACHER_CACHE_ENABLE="${TEACHER_CACHE_ENABLE:-false}"
  exec bash "${STUDENT_SCRIPT}" "$@"
}

launch_dlc_teacher() {
  mkdir -p "$(dirname "${TEACHER_LOG}")"
  # ossfs2 rejects truncating existing log files (EINVAL); pre-delete like EBFT.
  rm -f "${TEACHER_LOG}" 2>/dev/null || true
  echo "[orchestrator] teacher log: ${TEACHER_LOG}"
  export PYTHONUNBUFFERED=1
  set +e
  bash "${TEACHER_START_SCRIPT}" 2>&1 | tee "${TEACHER_LOG}"
  local status=${PIPESTATUS[0]}
  set -e
  exit "${status}"
}

detect_dlc_mode
if [[ "${DLC_MODE}" == "true" && "${DLC_AUTO_ENV}" == "true" ]]; then
  setup_dlc_runtime_env
  setup_dlc_stable_run_name
  maybe_build_slime_env
  source_slime_env
fi

detect_deploy_mode

case "${DEPLOY_MODE}" in
  dlc)
    if [[ "${DLC_MODE}" != "true" ]]; then
      echo "[ERROR] DEPLOY_MODE=dlc but DLC env vars are missing." >&2
      exit 1
    fi
    if [[ "${DLC_WORLD_SIZE}" != "2" ]]; then
      echo "[ERROR] dlc mode expects WORLD_SIZE=2, got ${DLC_WORLD_SIZE}" >&2
      exit 1
    fi
    echo "[orchestrator] DLC mode rank=${DLC_NODE_RANK} master=${DLC_MASTER_ADDR} world=${DLC_WORLD_SIZE} RUN_NAME=${RUN_NAME:-auto}"
    echo "[orchestrator] local ext4: RAY_TMPDIR=${RAY_TMPDIR} TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
    if [[ "${DLC_NODE_RANK}" == "${TEACHER_NODE_RANK}" ]]; then
      echo "[orchestrator] this pod is teacher (rank=${DLC_NODE_RANK})"
      launch_dlc_teacher
    elif [[ "${DLC_NODE_RANK}" == "${STUDENT_NODE_RANK}" ]]; then
      teacher_host="$(resolve_teacher_host "${DLC_MASTER_ADDR}")"
      echo "[orchestrator] this pod is student (rank=${DLC_NODE_RANK}), teacher host=${teacher_host} (master=${DLC_MASTER_ADDR})"
      wait_teacher_ready "${teacher_host}" "${TEACHER_PORT}"
      export RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-$(get_local_ip)}"
      run_student "${teacher_host}" "$@"
    else
      echo "[ERROR] unexpected DLC_NODE_RANK=${DLC_NODE_RANK}; set TEACHER_NODE_RANK/STUDENT_NODE_RANK" >&2
      exit 1
    fi
    ;;
  ssh)
    if [[ -z "${TEACHER_NODE_IP}" ]]; then
      if [[ -z "${TEACHER_NODE}" ]]; then
        echo "[ERROR] ssh mode requires TEACHER_NODE or TEACHER_NODE_IP" >&2
        exit 1
      fi
      TEACHER_NODE_IP="$(resolve_host_ip "${TEACHER_NODE}")"
    fi
    echo "[orchestrator] ssh mode teacher=${TEACHER_NODE_IP} student=$(hostname)"
    start_ssh_teacher_background "${TEACHER_NODE_IP}"
    wait_teacher_ready "${TEACHER_NODE_IP}" "${TEACHER_PORT}"
    export RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-$(get_local_ip)}"
    run_student "${TEACHER_NODE_IP}" "$@"
    ;;
  local_split)
    trap cleanup_local_teacher EXIT INT TERM
    echo "[orchestrator] local_split: teacher GPUs 0-3, student GPUs 4-7"
    start_local_teacher_background "${TEACHER_START_SCRIPT_LOCAL}"
    wait_teacher_ready "127.0.0.1" "${TEACHER_PORT}"
    export CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
    export COLOCATE="${COLOCATE:-true}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
    export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
    export CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"
    export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
    export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
    export TEACHER_HOST="127.0.0.1"
    export DEPLOY_LAYOUT=two_node
    export DEPLOY_ROLE=student
    exec bash "${SCRIPT_DIR}/run_g2_opd_qwen35_2b_main.sh" "$@"
    ;;
  *)
    echo "[ERROR] unknown DEPLOY_MODE=${DEPLOY_MODE}" >&2
    exit 1
    ;;
esac
