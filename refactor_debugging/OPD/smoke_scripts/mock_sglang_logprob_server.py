#!/usr/bin/env python3
"""Tiny SGLang-compatible logprob endpoint for OPD smoke tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _token_logprob(token_id: int, position: int) -> float:
    # Deterministic non-uniform scores are enough to exercise OPD credit plumbing.
    return -0.05 * (1 + ((int(token_id) + position) % 7))


class Handler(BaseHTTPRequestHandler):
    server_version = "OPDMockSGLang/0.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._send_json(404, {"error": f"unknown path: {self.path}"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return

        input_ids = request.get("input_ids")
        if input_ids is None:
            text = str(request.get("text", ""))
            input_ids = [ord(ch) % 32000 for ch in text] or [0]
        if not isinstance(input_ids, list) or not input_ids:
            self._send_json(400, {"error": "input_ids must be a non-empty list"})
            return

        token_logprobs = []
        for position, token_id in enumerate(input_ids):
            if position == 0:
                token_logprobs.append([None, int(token_id), None])
            else:
                token_logprobs.append([_token_logprob(int(token_id), position), int(token_id), None])

        self._send_json(
            200,
            {
                "text": "",
                "meta_info": {
                    "input_token_logprobs": token_logprobs,
                    "output_token_logprobs": [],
                    "finish_reason": {"type": "stop"},
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30123)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.quiet = args.quiet
    print(f"[mock-opd-rm] listening on http://{args.host}:{args.port}/generate", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
