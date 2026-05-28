#!/usr/bin/env bash

start_mock_opd_rm_if_requested() {
  OPD_SMOKE_MOCK_RM="${OPD_SMOKE_MOCK_RM:-true}"
  case "${OPD_SMOKE_MOCK_RM}" in
    true|1|yes) ;;
    false|0|no) return 0 ;;
    *) echo "[ERROR] OPD_SMOKE_MOCK_RM must be true or false, got: ${OPD_SMOKE_MOCK_RM}" >&2; exit 1 ;;
  esac

  PYTHON_BIN="${PYTHON_BIN:-/root/venvs/slime/bin/python}"
  OPD_MOCK_RM_HOST="${OPD_MOCK_RM_HOST:-127.0.0.1}"
  OPD_MOCK_RM_PORT="${OPD_MOCK_RM_PORT:-30123}"
  OPD_MOCK_RM_LOG="${OPD_MOCK_RM_LOG:-/tmp/opd_mock_sglang_logprob_server_${OPD_MOCK_RM_PORT}.log}"
  OPD_MOCK_RM_PID_FILE="${OPD_MOCK_RM_PID_FILE:-/tmp/opd_mock_sglang_logprob_server_${OPD_MOCK_RM_PORT}.pid}"

  export TEACHER_API_BASE="http://${OPD_MOCK_RM_HOST}:${OPD_MOCK_RM_PORT}"
  export OPD_TEACHER_RM_URL="${TEACHER_API_BASE}/generate"
  export TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-mock-sglang-logprob}"

  local mock_server="${SCRIPT_DIR}/mock_sglang_logprob_server.py"
  echo "[smoke] starting mock OPD logprob endpoint: ${OPD_TEACHER_RM_URL}"
  "${PYTHON_BIN}" "${mock_server}" \
    --host "${OPD_MOCK_RM_HOST}" \
    --port "${OPD_MOCK_RM_PORT}" \
    --quiet >"${OPD_MOCK_RM_LOG}" 2>&1 &
  echo $! >"${OPD_MOCK_RM_PID_FILE}"
  OPD_MOCK_RM_PID="$(cat "${OPD_MOCK_RM_PID_FILE}")"
  export OPD_MOCK_RM_PID
  trap cleanup_mock_opd_rm EXIT INT TERM

  wait_for_mock_opd_rm
}

wait_for_mock_opd_rm() {
  local deadline=$((SECONDS + ${OPD_MOCK_RM_WAIT_SECONDS:-15}))
  while (( SECONDS < deadline )); do
    if OPD_TEACHER_RM_URL="${OPD_TEACHER_RM_URL}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import json
import os
import urllib.request

request = urllib.request.Request(
    os.environ["OPD_TEACHER_RM_URL"],
    data=json.dumps(
        {
            "input_ids": [0, 1],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=1) as response:
    payload = json.loads(response.read().decode("utf-8"))
assert payload["meta_info"]["input_token_logprobs"][1][0] is not None
PY
    then
      echo "[smoke] mock OPD logprob endpoint is ready"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] mock OPD logprob endpoint did not become ready; log=${OPD_MOCK_RM_LOG}" >&2
  exit 1
}

cleanup_mock_opd_rm() {
  if [[ -n "${OPD_MOCK_RM_PID:-}" ]] && kill -0 "${OPD_MOCK_RM_PID}" 2>/dev/null; then
    kill "${OPD_MOCK_RM_PID}" 2>/dev/null || true
  fi
}
