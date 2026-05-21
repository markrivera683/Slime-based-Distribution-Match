#!/usr/bin/env bash
# Orchestrate G2 no-teacher-distribution on 2 DLC/PyTorch pods (8+8 GPU).
# rank 0: Ray head + job submitter. rank 1: Ray worker. No teacher service is started.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"

DEPLOY_MODE="${DEPLOY_MODE:-auto}"
DLC_AUTO_ENV="${DLC_AUTO_ENV:-true}"
BUILD_SLIME_ENV="${BUILD_SLIME_ENV:-false}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${SLIME_ROOT}/build_conda.sh}"
SLIME_ENV_WAIT_SECONDS="${SLIME_ENV_WAIT_SECONDS:-7200}"
DLC_LOCAL_ROOT="${DLC_LOCAL_ROOT:-/mnt/workspace}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_WAIT_SECONDS="${RAY_WAIT_SECONDS:-600}"
RAY_HEAD_ADDR_WAIT_SECONDS="${RAY_HEAD_ADDR_WAIT_SECONDS:-600}"
RAY_HEAD_ADDR_DIR="${RAY_HEAD_ADDR_DIR:-/mnt/data/slime/ray_head}"
RAY_HEAD_LOG="${RAY_HEAD_LOG:-${DLC_LOCAL_ROOT}/slime_logs/g2_no_teacher_distribution_ray_head.log}"
RAY_WORKER_LOG="${RAY_WORKER_LOG:-${DLC_LOCAL_ROOT}/slime_logs/g2_no_teacher_distribution_ray_worker.log}"

TEACHER_NODE_RANK="${TEACHER_NODE_RANK:-0}"
STUDENT_NODE_RANK="${STUDENT_NODE_RANK:-1}"

setup_dlc_runtime_env() {
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${DLC_LOCAL_ROOT}/.torch_extensions}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DLC_LOCAL_ROOT}/.triton_cache}"
  export RAY_TMPDIR="${RAY_TMPDIR:-${DLC_LOCAL_ROOT}/ray_slime_g2_no_teacher_distribution}"
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
    export RUN_NAME="g2_no_teacher_distribution_${job_id}"
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

resolve_host_ip() {
  local host="$1"
  if [[ "${host}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "${host}"
    return 0
  fi
  getent ahostsv4 "${host}" | awk 'NR==1 {print $1}'
}

get_local_ip() {
  local preferred_peer="${1:-}"
  local ip_from_hostname
  ip_from_hostname="$(
    hostname -I 2>/dev/null | awk -v peer="${preferred_peer}" '
      function prefix3(ip, parts) {
        split(ip, parts, ".")
        return parts[1] "." parts[2] "." parts[3] "."
      }
      BEGIN {
        peer_prefix = peer ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ ? prefix3(peer) : ""
      }
      {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^127\./ || $i ~ /:/) {
            continue
          }
          if (first == "") {
            first = $i
          }
          if (peer_prefix != "" && index($i, peer_prefix) == 1) {
            print $i
            printed = 1
            exit
          }
        }
      }
      END {
        if (!printed && first != "") {
          print first
        }
      }'
  )"
  if [[ -n "${ip_from_hostname}" && "${preferred_peer}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    local preferred_prefix="${preferred_peer%.*}."
    if [[ "${ip_from_hostname}" == "${preferred_prefix}"* ]]; then
      echo "${ip_from_hostname}"
      return 0
    fi
  fi
  if command -v ip >/dev/null 2>&1; then
    local eth0_ip
    eth0_ip="$(ip -o -4 addr show dev eth0 scope global 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}')"
    if [[ -n "${eth0_ip}" ]]; then
      echo "${eth0_ip}"
      return 0
    fi
  fi
  echo "${ip_from_hostname}"
}

ray_head_addr_file() {
  if [[ -n "${RAY_HEAD_ADDR_FILE:-}" ]]; then
    echo "${RAY_HEAD_ADDR_FILE}"
    return 0
  fi
  local generation="${JOB_FAILOVER_GENERATION:-0}"
  echo "${RAY_HEAD_ADDR_DIR}/${RUN_NAME:-g2_no_teacher_distribution}_${generation}.addr"
}

publish_ray_head_addr() {
  local head_ip="$1"
  local addr_file
  addr_file="$(ray_head_addr_file)"
  mkdir -p "$(dirname "${addr_file}")"
  printf "%s\n" "${head_ip}" >"${addr_file}.tmp"
  mv -f "${addr_file}.tmp" "${addr_file}"
  echo "[orchestrator] published Ray head address ${head_ip} to ${addr_file}"
}

wait_for_published_ray_head_addr() {
  local fallback_ip="$1"
  local addr_file waited head_ip
  addr_file="$(ray_head_addr_file)"
  waited=0
  echo "[orchestrator] waiting for Ray head address file ${addr_file}"
  while [[ ! -s "${addr_file}" ]]; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= RAY_HEAD_ADDR_WAIT_SECONDS )); then
      echo "[WARN] Ray head address file not ready after ${RAY_HEAD_ADDR_WAIT_SECONDS}s; falling back to ${fallback_ip}" >&2
      echo "${fallback_ip}"
      return 0
    fi
  done
  head_ip="$(awk 'NF {print $1; exit}' "${addr_file}")"
  if [[ -z "${head_ip}" ]]; then
    echo "[WARN] Ray head address file ${addr_file} is empty; falling back to ${fallback_ip}" >&2
    echo "${fallback_ip}"
    return 0
  fi
  echo "${head_ip}"
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

wait_ray_head() {
  local address="$1"
  local waited=0
  until "${RAY_BIN}" status --address "${address}" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= RAY_WAIT_SECONDS )); then
      echo "[ERROR] Ray head not ready at ${address} after ${RAY_WAIT_SECONDS}s" >&2
      exit 1
    fi
  done
}

start_ray_head_and_submit() {
  local head_connect_ip="$1"
  local head_bind_ip="$2"
  shift 2
  rm -f "${RAY_HEAD_LOG}" 2>/dev/null || true
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  echo "[orchestrator] starting Ray head bind=${head_bind_ip}:${RAY_PORT} connect=${head_connect_ip}:${RAY_PORT}; log=${RAY_HEAD_LOG}"
  publish_ray_head_addr "${head_bind_ip}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"   "${RAY_BIN}" start --head     --node-ip-address "${head_bind_ip}"     --port "${RAY_PORT}"     --dashboard-host=0.0.0.0     --dashboard-port "${RAY_DASHBOARD_PORT}"     --num-gpus 8     --temp-dir "${RAY_TMPDIR}"     --block >"${RAY_HEAD_LOG}" 2>&1 &
  local ray_pid=$!
  trap 'kill ${ray_pid} 2>/dev/null || true; "${RAY_BIN}" stop --force >/dev/null 2>&1 || true' EXIT INT TERM
  wait_ray_head "${head_bind_ip}:${RAY_PORT}"
  sleep "${WORKER_JOIN_GRACE_SECONDS:-20}"

  export DEPLOY_LAYOUT=two_node
  export DEPLOY_ROLE=trainer
  export USE_EXISTING_RAY=true
  export RAY_NODE_IP_ADDRESS="${head_bind_ip}"
  export RAY_ADDRESS="http://${head_bind_ip}:${RAY_DASHBOARD_PORT}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export G2_OPD_MODE="${G2_OPD_MODE:-cf_l1oo_no_teacher_distribution}"
  export CF_TARGET_MODE="${CF_TARGET_MODE:-single}"
  export CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"
  export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
  export CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-4}"
  export CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
  export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
  exec bash "${SCRIPT_DIR}/run_g2_no_teacher_distribution_qwen35_2b_main.sh" "$@"
}

start_ray_worker_blocking() {
  local head_ip="$1"
  rm -f "${RAY_WORKER_LOG}" 2>/dev/null || true
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  wait_ray_head "${head_ip}:${RAY_PORT}"
  echo "[orchestrator] joining Ray head ${head_ip}:${RAY_PORT}; log=${RAY_WORKER_LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"   "${RAY_BIN}" start --address "${head_ip}:${RAY_PORT}"     --num-gpus 8     --temp-dir "${RAY_TMPDIR}"     --block 2>&1 | tee "${RAY_WORKER_LOG}"
}

detect_dlc_mode
if [[ "${DLC_MODE}" == "true" && "${DLC_AUTO_ENV}" == "true" ]]; then
  setup_dlc_runtime_env
  setup_dlc_stable_run_name
  maybe_build_slime_env
  source_slime_env
fi

if [[ "${DEPLOY_MODE}" == "auto" ]]; then
  if [[ "${DLC_MODE}" == "true" ]]; then
    DEPLOY_MODE="dlc"
  else
    echo "[ERROR] no-teacher 2node launcher currently expects DLC env; set DEPLOY_MODE=dlc." >&2
    exit 1
  fi
fi

if [[ "${DEPLOY_MODE}" != "dlc" || "${DLC_MODE}" != "true" ]]; then
  echo "[ERROR] DEPLOY_MODE=dlc with WORLD_SIZE=2 is required." >&2
  exit 1
fi
if [[ "${DLC_WORLD_SIZE}" != "2" ]]; then
  echo "[ERROR] dlc mode expects WORLD_SIZE=2, got ${DLC_WORLD_SIZE}" >&2
  exit 1
fi

head_connect_ip="$(resolve_host_ip "${DLC_MASTER_ADDR}")"
if [[ -z "${head_connect_ip}" ]]; then
  echo "[ERROR] failed to resolve DLC master address: ${DLC_MASTER_ADDR}" >&2
  exit 1
fi
head_bind_ip="${RAY_HEAD_NODE_IP:-}"
if [[ "${DLC_NODE_RANK}" == "${TEACHER_NODE_RANK}" && -z "${head_bind_ip}" ]]; then
  head_bind_ip="$(get_local_ip "${head_connect_ip}")"
fi
if [[ "${DLC_NODE_RANK}" == "${TEACHER_NODE_RANK}" && -z "${head_bind_ip}" ]]; then
  echo "[ERROR] failed to determine local Ray head bind IP" >&2
  exit 1
fi
echo "[orchestrator] G2 no-teacher distribution DLC rank=${DLC_NODE_RANK} head_connect=${head_connect_ip} head_bind=${head_bind_ip:-n/a} world=${DLC_WORLD_SIZE} RUN_NAME=${RUN_NAME:-auto}"
if [[ "${DLC_NODE_RANK}" == "${TEACHER_NODE_RANK}" ]]; then
  start_ray_head_and_submit "${head_connect_ip}" "${head_bind_ip}" "$@"
elif [[ "${DLC_NODE_RANK}" == "${STUDENT_NODE_RANK}" ]]; then
  head_join_ip="$(wait_for_published_ray_head_addr "${head_connect_ip}")"
  echo "[orchestrator] worker will join Ray head ${head_join_ip}:${RAY_PORT}"
  start_ray_worker_blocking "${head_join_ip}"
else
  echo "[ERROR] unexpected DLC_NODE_RANK=${DLC_NODE_RANK}" >&2
  exit 1
fi
