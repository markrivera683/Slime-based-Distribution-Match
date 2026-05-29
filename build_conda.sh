#!/usr/bin/env bash

set -euo pipefail
set -x

# CUDA 12.9 / pip-venv build for environments where the container image cannot be changed.
# This intentionally avoids micromamba/conda and uses the system CUDA toolkit at /usr/local/cuda.
#
# Original conda recipe package groups:
# - CUDA stack: cuda 12.9.1, cuda-nvtx, nccl, cudnn, cuda-python, torch cu129
# - SGLang stack: sglang source checkout + python[all], sgl-router wheel
# - training kernels/libs: flash-attn, transformer_engine, flash-linear-attention, apex,
#   torch_memory_saver, mbridge, Megatron-Bridge, nvidia-modelopt
# - Megatron-LM source checkout + local slime source checkout + slime patches.
#
# cu129 adaptation:
# - Keep the same source commits and patches.
# - Use torch 2.9.1 + cu129 and cuda-python 12.9.
# - Install SGLang with --no-deps and install runtime dependencies manually so
#   dependency resolution cannot silently replace the selected CUDA stack.

export BASE_DIR="${BASE_DIR:-/root/slime_runtime}"
export VENV_DIR="${VENV_DIR:-/root/venvs/slime}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export RECREATE_VENV="${RECREATE_VENV:-false}"
export SGLANG_PROFILE="${SGLANG_PROFILE:-ebft_block_source}"
case "${SGLANG_PROFILE}" in
  ebft_block_source)
    _sglang_profile_repo="https://github.com/markrivera683/sglang_ebft.git"
    _sglang_profile_commit="d3b72348ebaa32903dd9392d4e9deb8e3dd08498"
    ;;
  upstream)
    _sglang_profile_repo="https://github.com/sgl-project/sglang.git"
    _sglang_profile_commit="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
    ;;
  *)
    echo "[ERROR] unknown SGLANG_PROFILE=${SGLANG_PROFILE}; expected ebft_block_source or upstream" >&2
    exit 1
    ;;
esac
export SGLANG_REPO="${SGLANG_REPO:-${_sglang_profile_repo}}"
export SGLANG_COMMIT="${SGLANG_COMMIT:-${_sglang_profile_commit}}"
export MEGATRON_COMMIT="${MEGATRON_COMMIT:-3714d81d418c9f1bca4594fc35f9e8289f652862}"
export MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
export SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.cloud.aliyuncs.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.cloud.aliyuncs.com}"
export PIP_RETRIES="${PIP_RETRIES:-20}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-true}"
export FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.6}"
export WHEELS_INFRA="${WHEELS_INFRA:-/mnt/data/wheels_infra}"
if [[ ! -d "${WHEELS_INFRA}" && -d /mnt/data/wheel_infra ]]; then
  WHEELS_INFRA="/mnt/data/wheel_infra"
fi
export INSTALL_CUDA129_FROM_WHEELS_INFRA="${INSTALL_CUDA129_FROM_WHEELS_INFRA:-true}"
export TORCH_CU129_WHEEL_DIR="${TORCH_CU129_WHEEL_DIR:-${WHEELS_INFRA}}"
_flash_attn_cu129="${WHEELS_INFRA}/flash_attn-2.7.4+cu129torch2.9-cp312-cp312-linux_x86_64.whl"
_flash_attn_legacy="${WHEELS_INFRA}/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl"
if [[ -z "${FLASH_ATTN_WHEEL:-}" ]]; then
  if [[ -f "${_flash_attn_cu129}" ]]; then
    export FLASH_ATTN_WHEEL="${_flash_attn_cu129}"
  else
    export FLASH_ATTN_WHEEL="${_flash_attn_legacy}"
  fi
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-64}"

mkdir -p "${BASE_DIR}" "$(dirname "${VENV_DIR}")" /root/.cargo
touch /root/.cargo/env

if [[ ! -f "${SLIME_DIR}/train.py" ]]; then
  echo "[ERROR] local slime source not found under SLIME_DIR=${SLIME_DIR}" >&2
  exit 1
fi

echo "[sglang] profile=${SGLANG_PROFILE}"
echo "[sglang] repo=${SGLANG_REPO}"
echo "[sglang] commit=${SGLANG_COMMIT}"

if [[ "${RECREATE_VENV}" == "true" && -d "${VENV_DIR}" ]]; then
  echo "[venv] removing existing venv to avoid stale torch/cuda extension ABI mismatches: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

python3.12 -m venv "${VENV_DIR}" --upgrade-deps
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel packaging cmake ninja pybind11

# Match the machine's system CUDA toolkit (nvcc 12.9).
_torch_wheel="${TORCH_CU129_WHEEL_DIR}/torch-2.9.1+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
_torchvision_wheel="${TORCH_CU129_WHEEL_DIR}/torchvision-0.24.1+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
_torchaudio_wheel="${TORCH_CU129_WHEEL_DIR}/torchaudio-2.9.1+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
if [[ -f "${_torch_wheel}" && -f "${_torchvision_wheel}" && -f "${_torchaudio_wheel}" ]]; then
  echo "[torch] installing cu129 wheels from ${TORCH_CU129_WHEEL_DIR}"
  pip install \
    --find-links "${TORCH_CU129_WHEEL_DIR}" \
    --index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --trusted-host mirrors.cloud.aliyuncs.com \
    "${_torch_wheel}" "${_torchvision_wheel}" "${_torchaudio_wheel}"
else
  pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu129
fi
pip install "numpy<2"
pip install cuda-python==12.9.0

assert_torch_cu129() {
  python - <<'PY'
import torch
version = torch.__version__
cuda = torch.version.cuda
print("torch", version)
print("torch.cuda", cuda)
print("cuda_available", torch.cuda.is_available())
if not version.startswith("2.9.1"):
    raise SystemExit(f"[ERROR] expected torch 2.9.1, got {version}")
if cuda != "12.9":
    raise SystemExit(f"[ERROR] expected torch CUDA 12.9, got {cuda}")
PY
}

nvcc_release_version() {
  local cuda_home="$1"
  if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    return 1
  fi
  "${cuda_home}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1
}

apply_cuda_home() {
  export CUDA_HOME="$1"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
}

# DLC images often symlink /usr/local/cuda -> CUDA 13.x while torch cu129 needs nvcc 12.9.
resolve_cuda_home_for_torch() {
  local torch_cuda want got candidate
  torch_cuda="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
  want="${torch_cuda:-12.9}"

  if [[ -n "${CUDA_HOME:-}" ]]; then
    got="$(nvcc_release_version "${CUDA_HOME}" || true)"
    if [[ "${got}" == "${want}" ]]; then
      apply_cuda_home "${CUDA_HOME}"
      echo "[cuda] using CUDA_HOME=${CUDA_HOME} (nvcc ${got}, matches torch ${want})"
      nvcc --version
      return 0
    fi
    echo "[cuda] ignoring CUDA_HOME=${CUDA_HOME} (nvcc ${got:-unknown} != torch ${want})" >&2
  fi

  for candidate in \
    "/usr/local/cuda-${want}" \
    "/usr/local/cuda-12.9" \
    "/usr/local/cuda-12.4" \
    "/usr/local/cuda"; do
    [[ -d "${candidate}" ]] || continue
    got="$(nvcc_release_version "${candidate}" || true)"
    if [[ "${got}" == "${want}" ]]; then
      apply_cuda_home "${candidate}"
      echo "[cuda] resolved CUDA_HOME=${CUDA_HOME} for torch CUDA ${want}"
      nvcc --version
      return 0
    fi
    echo "[cuda] skip ${candidate} (nvcc ${got:-missing}, need ${want})"
  done

  echo "[ERROR] No nvcc ${want} toolkit found for extension builds (apex, flash-attn)." >&2
  echo "        DLC: run ${SLIME_DIR}/scripts/install_cuda129_from_wheels_infra.sh (deb under ${WHEELS_INFRA})" >&2
  echo "        or set CUDA_HOME=/usr/local/cuda-12.9 after installing cuda-compiler-12-9 and cuda-libraries-dev-12-9" >&2
  exit 1
}

maybe_install_cuda129_from_wheels_infra() {
  if [[ "${INSTALL_CUDA129_FROM_WHEELS_INFRA}" != "true" && "${INSTALL_CUDA129_FROM_WHEELS_INFRA}" != "1" ]]; then
    return 0
  fi
  if [[ -x /usr/local/cuda-12.9/bin/nvcc ]]; then
    return 0
  fi
  local installer="${SLIME_DIR}/scripts/install_cuda129_from_wheels_infra.sh"
  if [[ ! -f "${installer}" ]]; then
    echo "[WARN] CUDA 12.9 installer missing: ${installer}" >&2
    return 0
  fi
  bash "${installer}"
}

install_torch_fsdp_shim() {
  # transformer-engine 2.10 imports the newer torch.distributed.fsdp._fully_shard
  # path. If the installed torch does not ship it, provide a narrow compatibility shim.
  python - <<'PY'
from pathlib import Path
import torch

base = Path(torch.__file__).resolve().parent / "distributed" / "fsdp" / "_fully_shard"
if base.exists():
    print(f"[compat] torch FSDP path already exists: {base}")
else:
    base.mkdir(parents=True, exist_ok=True)
    (base / "__init__.py").write_text("from torch.distributed._composable.fully_shard import *\n", encoding="utf-8")
    (base / "_fsdp_common.py").write_text(
        "from torch.distributed._composable.fsdp._fsdp_common import *\n",
        encoding="utf-8",
    )
    print(f"[compat] installed torch FSDP shim: {base}")
PY
}

assert_torch_cu129
maybe_install_cuda129_from_wheels_infra
resolve_cuda_home_for_torch

PY_SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export CUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR:-${PY_SITE_PACKAGES}/nvidia/cudnn/include}"
export CUDNN_LIB_DIR="${CUDNN_LIB_DIR:-${PY_SITE_PACKAGES}/nvidia/cudnn/lib}"
export CPLUS_INCLUDE_PATH="${CUDNN_INCLUDE_DIR}:${CPLUS_INCLUDE_PATH:-}"
export CPATH="${CUDNN_INCLUDE_DIR}:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_LIB_DIR}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${CUDNN_INCLUDE_DIR}/cudnn.h" ]]; then
  echo "[ERROR] cudnn.h not found under ${CUDNN_INCLUDE_DIR}" >&2
  exit 1
fi

clone_or_update() {
  local repo="$1"
  local dest="$2"
  local commit="$3"
  local recursive="${4:-false}"

  if [[ ! -d "${dest}/.git" ]]; then
    if [[ "${recursive}" == "true" ]]; then
      git clone --recursive "${repo}" "${dest}"
    else
      git clone "${repo}" "${dest}"
    fi
  else
    cd "${dest}"
    local current_origin
    current_origin="$(git remote get-url origin 2>/dev/null || true)"
    if [[ "${current_origin}" != "${repo}" ]]; then
      echo "[git] updating origin for ${dest}: ${current_origin:-<none>} -> ${repo}"
      if [[ -n "${current_origin}" ]]; then
        git remote set-url origin "${repo}"
      else
        git remote add origin "${repo}"
      fi
    fi
  fi
  cd "${dest}"
  git fetch --all --tags
  git checkout "${commit}"
  if [[ "${recursive}" == "true" ]]; then
    git submodule update --init --recursive
  fi
}

cd "${BASE_DIR}"
clone_or_update "${SGLANG_REPO}" "${BASE_DIR}/sglang" "${SGLANG_COMMIT}" false

# Do not install "python[all]" dependencies for this pinned SGLang commit.
# Keep torch/cu129 intact by installing runtime dependencies explicitly below.
pip install -e "${BASE_DIR}/sglang/python" --no-deps

# SGLang runtime deps.
# Keep this list explicit so dependency resolution cannot silently replace torch.
pip install \
  IPython \
  aiohttp \
  "apache-tvm-ffi>=0.1.5,<0.2" \
  "anthropic>=0.20.0" \
  blobfile==3.0.0 \
  build \
  compressed-tensors \
  datasets \
  decord2 \
  einops \
  fastapi \
  gguf \
  hf_transfer \
  huggingface_hub \
  interegular \
  "llguidance>=0.7.11,<0.8.0" \
  modelscope \
  msgspec \
  nvidia-ml-py \
  openai-harmony==0.0.4 \
  openai==2.6.1 \
  orjson \
  outlines==0.1.11 \
  packaging \
  partial_json_parser \
  pillow \
  "prometheus-client>=0.20.0" \
  psutil \
  py-spy \
  pybase64 \
  pydantic \
  python-multipart \
  "pyzmq>=25.1.2" \
  sgl-kernel==0.3.21 \
  requests \
  scipy \
  sentencepiece \
  setproctitle \
  soundfile==0.13.1 \
  tiktoken \
  timm==1.0.16 \
  torchao==0.9.0 \
  tqdm \
  transformers==4.57.1 \
  uvicorn \
  uvloop \
  xgrammar==0.1.27 \
  "smg-grpc-proto>=0.3.3" \
  "grpcio>=1.78.0" \
  "grpcio-reflection>=1.78.0" \
  "grpcio-health-checking>=1.78.0"
assert_torch_cu129

if [[ "${INSTALL_FLASHINFER}" == "true" ]]; then
  if [[ -n "${FLASHINFER_VERSION}" ]]; then
    pip install "flashinfer-python==${FLASHINFER_VERSION}" --no-deps
  else
    pip install flashinfer-python
  fi
fi

# Build/runtime deps used by slime's Megatron/SGLang stack.
# flash-attn 2.7.4.post1 is the newest version supported by the pinned Megatron patch.
if [[ -f "${FLASH_ATTN_WHEEL}" ]]; then
  pip install "${FLASH_ATTN_WHEEL}"
else
  MAX_JOBS="${MAX_JOBS}" pip -v install flash-attn==2.7.4.post1 --no-build-isolation
fi
assert_torch_cu129
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
install_torch_fsdp_shim
pip install flash-linear-attention==0.4.1
assert_torch_cu129
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

assert_torch_cu129
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --no-deps
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation --no-deps
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation --no-deps
assert_torch_cu129

# Router wheel from the official slime build recipe.
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall --no-deps
assert_torch_cu129

clone_or_update "${MEGATRON_REPO}" "${BASE_DIR}/Megatron-LM" "${MEGATRON_COMMIT}" true
pip install -e "${BASE_DIR}/Megatron-LM" --no-deps
assert_torch_cu129

pip install \
  accelerate \
  "httpx[http2]" \
  "mcp[cli]" \
  memray \
  numba \
  omegaconf \
  pylatexenc \
  pytest \
  pyyaml \
  qwen_vl_utils \
  "ray[default]" \
  ring_flash_attn \
  tensorboard \
  wandb
assert_torch_cu129
pip install -e "${SLIME_DIR}" --no-deps
assert_torch_cu129

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"
assert_torch_cu129

apply_patch_once() {
  local target_dir="$1"
  local patch_file="$2"
  local marker="$3"

  if [[ ! -f "${patch_file}" ]]; then
    echo "[WARN] patch file not found: ${patch_file}"
    return 0
  fi
  if [[ -f "${marker}" ]]; then
    echo "[patch] already applied: ${patch_file}"
    return 0
  fi
  cd "${target_dir}"
  git apply "${patch_file}"
  touch "${marker}"
}

apply_patch_once "${BASE_DIR}/sglang" "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch" "${BASE_DIR}/sglang/.slime_v059_patch_applied"
apply_patch_once "${BASE_DIR}/Megatron-LM" "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch" "${BASE_DIR}/Megatron-LM/.slime_v059_patch_applied"

cat > "${BASE_DIR}/slime_env.sh" <<EOF
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:\$PATH"
export CUDA_HOME="${CUDA_HOME}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
export PIP_INDEX_URL="${PIP_INDEX_URL}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}"
export PIP_RETRIES="${PIP_RETRIES}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK}"
export SGLANG_PROFILE="${SGLANG_PROFILE}"
export SGLANG_REPO="${SGLANG_REPO}"
export SGLANG_COMMIT="${SGLANG_COMMIT}"
export MEGATRON_PATH="${BASE_DIR}/Megatron-LM"
export SLIME_ROOT="${SLIME_DIR}"
export PYTHONPATH="${BASE_DIR}/Megatron-LM:${SLIME_DIR}:\${PYTHONPATH:-}"
EOF

echo "[done] slime uv environment built."
echo "       source ${BASE_DIR}/slime_env.sh"
