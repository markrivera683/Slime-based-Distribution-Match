#!/usr/bin/env python3
"""Minimal SGLang EBFT block_source smoke for dense4d or sparse_ir paths.

This script intentionally does not modify Slime or SGLang source. It uses the
local tiny Llama config/tokenizer with SGLang's dummy weight loader, starts a
short-lived HTTP server, and sends one native /generate request carrying
ebft_position_ids, either ebft_dense_attention_mask or ebft_sparse_ir, and
block_source geometry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.g1_ebft_rollout_mask import build_g1_ebft_rollout_mask_contract


SGLANG_PYTHON = Path("/root/slime_runtime/sglang/python")
DEFAULT_MODEL = Path(
    "/mnt/data/distribution-matching-slime/code/Distributional-Match-Tuning-dev-dlc/local_smoke_llama"
)


def post_json(url: str, payload: dict, timeout: float = 2.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def get_url(url: str, timeout: float = 1.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def require_local_sglang_path() -> None:
    if not (SGLANG_PYTHON / "sglang" / "launch_server.py").exists():
        raise SystemExit(f"BLOCKED: SGLang python path not found: {SGLANG_PYTHON}")


def require_import_deps() -> None:
    missing = [name for name in ("tqdm",) if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "BLOCKED: missing Python dependency/dependencies required before "
            f"SGLang can import: {', '.join(missing)}. "
            "Install the SGLang python project requirements in the active env, "
            "then rerun this script."
        )


def require_tiny_model(model_path: Path) -> None:
    required = ["config.json", "tokenizer.json"]
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        raise SystemExit(
            f"BLOCKED: tiny model directory {model_path} is missing: {', '.join(missing)}"
        )


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path),
        "--tokenizer-path",
        str(args.model_path),
        "--load-format",
        "dummy",
        "--device",
        args.device,
        "--attention-backend",
        args.attention_backend,
        "--dtype",
        args.dtype,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "-1",
        "--max-total-tokens",
        "128",
        "--max-running-requests",
        "1",
        "--disable-overlap-schedule",
    ]
    if args.grammar_backend:
        cmd.extend(["--grammar-backend", args.grammar_backend])
    cmd.extend(args.server_args)
    return cmd


def build_server_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_PYTHON}:{env.get('PYTHONPATH', '')}"
    if args.device == "cpu":
        env.setdefault("SGLANG_USE_CPU_ENGINE", "1")
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def format_server_cmd_for_print(args: argparse.Namespace, env: dict[str, str], cmd: list[str]) -> str:
    env_keys = ["PYTHONPATH"]
    if args.device == "cpu":
        env_keys = ["CUDA_VISIBLE_DEVICES", "SGLANG_USE_CPU_ENGINE", *env_keys]
    env_prefix = " ".join(f"{key}={shlex.quote(env[key])}" for key in env_keys)
    return f"{env_prefix} {' '.join(shlex.quote(part) for part in cmd)}"


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def build_generate_payload(args: argparse.Namespace) -> dict:
    if args.mask_mode == "dense4d" and args.attention_backend != "torch_native":
        raise ValueError("dense4d block_source smoke requires --attention-backend torch_native.")
    if args.mask_mode == "sparse_ir" and args.attention_backend != "triton":
        raise ValueError("sparse_ir block_source smoke requires --attention-backend triton.")

    contract = build_g1_ebft_rollout_mask_contract(
        prompt_length=args.prompt_len,
        context_length=args.context_len,
        generate_length=args.generate_length,
        stride=args.stride,
    )
    if contract.num_blocks != args.num_blocks:
        raise ValueError(
            "Strict EBFT contract num_blocks does not match --num-blocks "
            f"({contract.num_blocks} != {args.num_blocks}); adjust --prompt-len, "
            "--context-len, --generate-length, or --stride."
        )

    input_ids = [1] + list(range(10, 10 + contract.prompt_length - 1))
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": contract.response_length,
            "temperature": 0,
            "ignore_eos": True,
            "stop": [],
            "stop_token_ids": [],
        },
        "stream": False,
        "ebft_rollout_sampling_mode": "block_source",
        "ebft_mask_spec": contract.to_mask_spec(mode=args.mask_mode),
        "ebft_position_ids": contract.position_ids,
    }
    if args.mask_mode == "dense4d":
        payload["ebft_dense_attention_mask"] = contract.to_dense_additive_mask()
    else:
        payload["ebft_sparse_ir"] = contract.to_span_sparse_ir()
    return jsonable(payload)


def assert_block_source_response(body: str, payload: dict) -> None:
    parsed = json.loads(body)
    result = parsed[0] if isinstance(parsed, list) else parsed
    output_ids = result.get("output_ids")
    expected_len = payload["sampling_params"]["max_new_tokens"]
    if not isinstance(output_ids, list) or len(output_ids) != expected_len:
        raise AssertionError(
            f"Expected {expected_len} output_ids, got {output_ids!r}."
        )

    meta_info = result.get("meta_info") or {}
    finish_reason = meta_info.get("finish_reason", result.get("finish_reason"))
    expected_rows = [
        payload["ebft_mask_spec"]["rollout_source_rows"][i : i + payload["ebft_mask_spec"]["num_blocks"]]
        for i in range(
            0,
            payload["ebft_mask_spec"]["response_length"],
            payload["ebft_mask_spec"]["num_blocks"],
        )
    ]
    actual_rows = meta_info.get("ebft_sample_source_rows")
    if actual_rows != expected_rows:
        raise AssertionError(
            "Unexpected ebft_sample_source_rows debug metadata: "
            f"expected {expected_rows!r}, got {actual_rows!r}."
        )
    print(f"output_ids: {output_ids}")
    print(f"length: {len(output_ids)}")
    print(f"ebft_sample_source_rows: {actual_rows}")
    print(f"finish_reason: {finish_reason}")
    print("\nAssertions passed.")


def format_log(lines: list[str], log_lines: int) -> str:
    if log_lines <= 0:
        return "".join(lines)
    return "".join(lines[-log_lines:])


def wait_until_ready(base_url: str, proc: subprocess.Popen, timeout_s: float, log_lines: list[str], max_log_lines: int) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                "BLOCKED: SGLang server exited before becoming ready.\n"
                f"Exit code: {proc.returncode}\n"
                f"Server output:\n{format_log(log_lines, max_log_lines)}"
            )
        try:
            status, _ = get_url(f"{base_url}/health", timeout=1.0)
            if status == 200:
                return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = repr(exc)
        time.sleep(0.5)
    raise SystemExit(
        f"BLOCKED: SGLang server did not become ready: {last_error}\n"
        f"Server output:\n{format_log(log_lines, max_log_lines)}"
    )


def terminate(proc: subprocess.Popen, reader: threading.Thread) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    reader.join(timeout=5)


def start_log_reader(proc: subprocess.Popen, lines: list[str]) -> threading.Thread:
    def read_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    return reader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 selects a free local port")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--attention-backend",
        default="torch_native",
        help="SGLang attention backend to pass through, e.g. torch_native or triton.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["dense4d", "sparse_ir"],
        default="dense4d",
        help="EBFT payload mode. Use sparse_ir with --attention-backend triton.",
    )
    parser.add_argument("--generate-length", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=6)
    parser.add_argument("--context-len", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--log-lines", type=int, default=200, help="server log lines to print; 0 means full log")
    parser.add_argument(
        "--grammar-backend",
        choices=["xgrammar", "outlines", "llguidance", "none"],
        default="none",
        help="SGLang grammar backend; default avoids requiring xgrammar for this unconstrained smoke.",
    )
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra arguments appended to sglang.launch_server after '--'",
    )
    parser.add_argument(
        "--server-log-path",
        type=Path,
        default=Path(__file__).with_suffix(".server.log"),
        help="path where the full captured server log is written",
    )
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    require_local_sglang_path()
    require_tiny_model(args.model_path)
    if args.port == 0:
        args.port = pick_free_port(args.host)

    env = build_server_env(args)
    cmd = build_server_cmd(args)
    payload = build_generate_payload(args)

    print("Server command:")
    print(format_server_cmd_for_print(args, env, cmd))
    print("\n/generate payload:")
    print(json.dumps(payload, indent=2))

    if args.print_only:
        return 0

    require_import_deps()

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    server_log_lines: list[str] = []
    log_reader = start_log_reader(proc, server_log_lines)
    base_url = f"http://{args.host}:{args.port}"
    try:
        wait_until_ready(base_url, proc, args.startup_timeout, server_log_lines, args.log_lines)
        status, body = post_json(f"{base_url}/generate", payload, timeout=30.0)
        print(f"\n/generate status: {status}")
        print(body[:4000])
        assert_block_source_response(body, payload)
        return 0
    finally:
        terminate(proc, log_reader)
        args.server_log_path.write_text("".join(server_log_lines), encoding="utf-8")
        print(f"\nFull server output written to: {args.server_log_path}")
        print("\nServer output:")
        print(format_log(server_log_lines, args.log_lines))


if __name__ == "__main__":
    raise SystemExit(main())
