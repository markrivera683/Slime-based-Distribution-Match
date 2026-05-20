#!/usr/bin/env bash
# Install CUDA 12.9 nvcc toolkit from the local repo deb on OSS (/mnt/data/wheels_infra).
# Used on DLC pods where /usr/local/cuda points at CUDA 13 but torch is cu129.
set -euo pipefail

WHEELS_INFRA="${WHEELS_INFRA:-/mnt/data/wheels_infra}"
if [[ ! -d "${WHEELS_INFRA}" && -d /mnt/data/wheel_infra ]]; then
  WHEELS_INFRA="/mnt/data/wheel_infra"
fi

CUDA129_REPO_DEB="${CUDA129_REPO_DEB:-${WHEELS_INFRA}/cuda-repo-ubuntu2204-12-9-local_12.9.1-575.57.08-1_amd64.deb}"
CUDA129_INSTALL_MARKER="${CUDA129_INSTALL_MARKER:-/mnt/workspace/.slime_cuda129_toolkit_installed}"
CUDA129_SYSTEM_APT_PACKAGES="${CUDA129_SYSTEM_APT_PACKAGES:-libnuma1}"
# Keep this below the full cuda-toolkit for DLC pods. The full toolkit pulls
# GUI/profiler packages such as libegl1/nsight, which can fail on DLC's
# cross-device container filesystem. Library dev headers are still needed by
# source-built CUDA extensions such as transformer-engine/apex.
CUDA129_APT_PACKAGES="${CUDA129_APT_PACKAGES:-cuda-compiler-12-9 cuda-libraries-dev-12-9 cuda-nvtx-12-9}"
INSTALL_CUDA129_FROM_WHEELS_INFRA="${INSTALL_CUDA129_FROM_WHEELS_INFRA:-true}"

nvcc_release_version() {
  local cuda_home="$1"
  if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    return 1
  fi
  "${cuda_home}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1
}

cuda129_ready() {
  local got
  got="$(nvcc_release_version /usr/local/cuda-12.9 || true)"
  [[ "${got}" == "12.9" ]]
}

install_cuda_repo_keyring() {
  local keyring package_name

  package_name="$(dpkg-deb -f "${CUDA129_REPO_DEB}" Package 2>/dev/null || true)"
  if [[ -n "${package_name}" ]]; then
    while IFS= read -r keyring; do
      [[ -f "${keyring}" && "${keyring}" == *-keyring.gpg ]] || continue
      mkdir -p /usr/share/keyrings
      cp -f "${keyring}" /usr/share/keyrings/
      echo "[cuda129] installed local CUDA repo keyring: ${keyring}"
      return 0
    done < <(dpkg -L "${package_name}" 2>/dev/null || true)
  fi

  shopt -s nullglob
  for keyring in /var/cuda-repo-ubuntu2204-12-9-local/*-keyring.gpg /var/cuda-repo-*-local/*-keyring.gpg; do
    mkdir -p /usr/share/keyrings
    cp -f "${keyring}" /usr/share/keyrings/
    echo "[cuda129] installed local CUDA repo keyring: ${keyring}"
    shopt -u nullglob
    return 0
  done
  shopt -u nullglob

  echo "[ERROR] CUDA local repo keyring not found under /var/cuda-repo-*-local after dpkg -i" >&2
  exit 1
}

install_system_apt_packages() {
  local missing=() package

  for package in ${CUDA129_SYSTEM_APT_PACKAGES}; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("${package}")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    return 0
  fi

  echo "[cuda129] installing required system packages: ${missing[*]}"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "${missing[@]}"
}

if [[ "${INSTALL_CUDA129_FROM_WHEELS_INFRA}" != "true" && "${INSTALL_CUDA129_FROM_WHEELS_INFRA}" != "1" ]]; then
  echo "[cuda129] INSTALL_CUDA129_FROM_WHEELS_INFRA=${INSTALL_CUDA129_FROM_WHEELS_INFRA}; skip"
  exit 0
fi

if ! command -v dpkg >/dev/null 2>&1 || ! command -v apt-get >/dev/null 2>&1; then
  echo "[ERROR] dpkg/apt-get required to install CUDA 12.9 from ${CUDA129_REPO_DEB}" >&2
  exit 1
fi

if cuda129_ready; then
  echo "[cuda129] /usr/local/cuda-12.9 already provides nvcc 12.9"
  install_system_apt_packages
  exit 0
fi

if [[ ! -f "${CUDA129_REPO_DEB}" ]]; then
  echo "[ERROR] CUDA 12.9 local repo deb not found: ${CUDA129_REPO_DEB}" >&2
  echo "        Set WHEELS_INFRA or CUDA129_REPO_DEB to the cuda-repo-ubuntu2204-12-9-local *.deb path." >&2
  exit 1
fi

mkdir -p "$(dirname "${CUDA129_INSTALL_MARKER}")" /mnt/workspace 2>/dev/null || true

echo "[cuda129] installing from local repo deb: ${CUDA129_REPO_DEB}"
export DEBIAN_FRONTEND=noninteractive
set +e
dpkg -i "${CUDA129_REPO_DEB}"
dpkg_status=$?
set -e
if (( dpkg_status != 0 )); then
  echo "[cuda129] dpkg -i returned ${dpkg_status}; continuing (package may already be registered)"
fi

install_cuda_repo_keyring
apt-get update -qq
apt-get install -y -qq --no-install-recommends ${CUDA129_APT_PACKAGES}
install_system_apt_packages

if ! cuda129_ready; then
  echo "[ERROR] apt install finished but /usr/local/cuda-12.9/bin/nvcc is not release 12.9" >&2
  exit 1
fi

touch "${CUDA129_INSTALL_MARKER}"
echo "[cuda129] installed ${CUDA129_APT_PACKAGES}; nvcc:"
/usr/local/cuda-12.9/bin/nvcc --version | head -4
