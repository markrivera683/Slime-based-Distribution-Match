from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class G2RemoteTeacherConfig:
    api_bases: tuple[str, ...]
    model_name: str
    api_key: str = "EMPTY"
    api_style: str = "completions"
    timeout: int = 120
    max_retries: int = 3
    remote_batch_size: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    system_prompt_text: str = ""
    system_prompt_id: str = ""
    cache_dir: str | None = None
    sglang_multi_sample: bool = True


def g2_remote_teacher_config_from_args(args: Any) -> G2RemoteTeacherConfig:
    api_base = getattr(args, "teacher_api_base", None)
    if not api_base:
        raise ValueError("--teacher-api-base is required to build the G2 remote teacher client")
    api_bases = tuple(item.strip().rstrip("/") for item in str(api_base).split(",") if item.strip())
    if not api_bases:
        raise ValueError("--teacher-api-base did not contain any usable URL")
    model_name = getattr(args, "teacher_model_name", None)
    if not model_name:
        raise ValueError("--teacher-model-name is required to build the G2 remote teacher client")
    return G2RemoteTeacherConfig(
        api_bases=api_bases,
        model_name=str(model_name),
        api_key=str(getattr(args, "teacher_api_key", "EMPTY")),
        api_style=str(getattr(args, "teacher_api_style", "completions")),
        timeout=int(getattr(args, "teacher_timeout", 120)),
        max_retries=int(getattr(args, "teacher_max_retries", 3)),
        remote_batch_size=int(getattr(args, "teacher_remote_batch_size", 8)),
        temperature=float(getattr(args, "teacher_temperature", 0.7)),
        top_p=float(getattr(args, "teacher_top_p", 0.95)),
        max_new_tokens=int(getattr(args, "teacher_max_new_tokens", 512)),
        system_prompt_text=str(getattr(args, "teacher_system_prompt_text", "")),
        system_prompt_id=str(getattr(args, "teacher_system_prompt_id", "")),
        cache_dir=getattr(args, "teacher_cache_dir", None) if getattr(args, "teacher_cache_enable", False) else None,
        sglang_multi_sample=bool(getattr(args, "teacher_sglang_multi_sample", True)),
    )


class G2TeacherCache:
    def __init__(self, cache_dir: str) -> None:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.db_path = path / "teacher_cache.db"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS completions ("
            "cache_key TEXT PRIMARY KEY, completions_json TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def _canonicalize(prompt: str) -> str:
        return "\n".join(line.rstrip() for line in prompt.strip().splitlines())

    @staticmethod
    def _make_key(
        *,
        prompt: str,
        model_name: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        api_style: str,
        system_prompt_id: str,
        system_prompt_text: str,
    ) -> str:
        payload = {
            "prompt": G2TeacherCache._canonicalize(prompt),
            "model_name": model_name,
            "n_samples": int(n_samples),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
            "api_style": api_style,
            "system_prompt_id": system_prompt_id,
            "system_prompt_text": G2TeacherCache._canonicalize(system_prompt_text),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def get(self, key: str) -> list[str] | None:
        row = self._conn.execute("SELECT completions_json FROM completions WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        return list(json.loads(row[0]))

    def put(self, key: str, completions: list[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO completions(cache_key, completions_json, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(completions, ensure_ascii=False), time.time()),
        )
        self._conn.commit()


class G2RemoteTeacherClient:
    """Small remote teacher client for the standard G2 online-teacher foundation.

    This client only returns completion text. Mapping those completions into EBFT
    teacher embeddings is intentionally left to the trainer/rollout integration.
    """

    def __init__(self, config: G2RemoteTeacherConfig) -> None:
        if config.api_style not in ("completions", "chat_completions", "sglang_generate"):
            raise ValueError(f"Unsupported teacher api_style: {config.api_style}")
        self.config = config
        self.cache = G2TeacherCache(config.cache_dir) if config.cache_dir else None
        self._next_base_idx = 0
        self._next_base_lock = Lock()
        self.last_stats: list[dict[str, Any]] = []
        self._last_request_count = 0

    def sample_targets(self, prompts: list[str], n_samples: int) -> list[list[str]]:
        results: list[list[str]] = []
        stats: list[dict[str, Any]] = []
        for prompt in prompts:
            completions, prompt_stats = self._sample_one_with_stats(prompt, n_samples)
            results.append(completions)
            stats.append(prompt_stats)
        self.last_stats = stats
        return results

    def _sample_one(self, prompt: str, n_samples: int) -> list[str]:
        completions, _ = self._sample_one_with_stats(prompt, n_samples)
        return completions

    def _sample_one_with_stats(self, prompt: str, n_samples: int) -> tuple[list[str], dict[str, Any]]:
        start = time.monotonic()
        key = None
        if self.cache is not None:
            key = self.cache._make_key(
                prompt=prompt,
                model_name=self.config.model_name,
                n_samples=n_samples,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                api_style=self.config.api_style,
                system_prompt_id=self.config.system_prompt_id,
                system_prompt_text=self.config.system_prompt_text,
            )
            cached = self.cache.get(key)
            if cached is not None:
                return cached, self._make_stats(
                    cache_hit=True,
                    latency_sec=time.monotonic() - start,
                    num_completions=len(cached),
                    num_requests=0,
                    failed_attempts=0,
                    retries=0,
                )

        completions, request_stats = self._request_with_retries(prompt, n_samples)
        if self.cache is not None and key is not None:
            self.cache.put(key, completions)
        request_stats.update(
            cache_hit=False,
            latency_sec=time.monotonic() - start,
            num_completions=len(completions),
        )
        return completions, self._make_stats(**request_stats)

    def _request_with_retries(self, prompt: str, n_samples: int) -> tuple[list[str], dict[str, Any]]:
        last_error: Exception | None = None
        last_error_detail = ""
        last_url = ""
        num_requests = 0
        failed_attempts = 0
        for attempt in range(1, self.config.max_retries + 1):
            api_base = self._next_api_base()
            last_url = self._request_url(api_base)
            try:
                self._last_request_count = 0
                completions = self._request(api_base, prompt, n_samples)
                num_requests += self._last_request_count or self._expected_request_count(n_samples)
                return completions, {
                    "api_base": api_base,
                    "url": last_url,
                    "num_requests": num_requests,
                    "failed_attempts": failed_attempts,
                    "retries": attempt - 1,
                }
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                failed_attempts += 1
                num_requests += self._last_request_count or self._expected_request_count(n_samples)
                last_error = exc
                last_error_detail = self._format_request_error(exc)
                logger.warning(
                    "G2 remote teacher request failed on attempt %s/%s url=%s: %s",
                    attempt,
                    self.config.max_retries,
                    last_url,
                    last_error_detail,
                )
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(
            f"G2 remote teacher request failed after {self.config.max_retries} retries "
            f"for url={last_url}. Last error: {last_error_detail}"
        ) from last_error

    def _next_api_base(self) -> str:
        with self._next_base_lock:
            idx = self._next_base_idx % len(self.config.api_bases)
            self._next_base_idx += 1
        return self.config.api_bases[idx]

    def _expected_request_count(self, n_samples: int) -> int:
        if self.config.api_style != "sglang_generate":
            return 1
        if self.config.sglang_multi_sample:
            return 1
        return n_samples

    def _request(self, api_base: str, prompt: str, n_samples: int) -> list[str]:
        if self.config.api_style == "sglang_generate":
            return self._request_sglang_generate(api_base, prompt, n_samples)
        return self._request_once(api_base, prompt, n_samples)

    def _request_sglang_generate(self, api_base: str, prompt: str, n_samples: int) -> list[str]:
        if n_samples <= 0:
            return []
        if not self.config.sglang_multi_sample:
            self._last_request_count = n_samples
            return self._request_sglang_generate_concurrent(api_base, prompt, n_samples)
        try:
            self._last_request_count = 1
            return self._request_once(api_base, prompt, n_samples)
        except (urllib.error.HTTPError, ValueError) as exc:
            logger.warning(
                "G2 SGLang multi-sample request failed; falling back to concurrent single-sample requests: %s",
                self._format_request_error(exc) if isinstance(exc, urllib.error.HTTPError) else exc,
            )
            self._last_request_count = 1 + n_samples
            return self._request_sglang_generate_concurrent(api_base, prompt, n_samples)

    def _request_sglang_generate_concurrent(self, api_base: str, prompt: str, n_samples: int) -> list[str]:
        if n_samples <= 0:
            return []
        max_workers = min(n_samples, max(1, int(self.config.remote_batch_size)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._request_once, api_base, prompt, 1) for _ in range(n_samples)]
            completions: list[str] = []
            for future in futures:
                completions.extend(future.result())
        return completions[:n_samples]

    def _request_once(self, api_base: str, prompt: str, n_samples: int) -> list[str]:
        url, payload = self._build_request(api_base, prompt, n_samples)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        try:
            completions = self._parse_response(data)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Failed to parse teacher response for url={url}: {type(exc).__name__}: {exc}") from exc
        if len(completions) < n_samples:
            raise ValueError(
                f"Teacher returned {len(completions)} completions, expected {n_samples} for url={url}"
            )
        return completions[:n_samples]

    def _request_url(self, api_base: str) -> str:
        if self.config.api_style == "sglang_generate":
            return api_base if api_base.endswith("/generate") else f"{api_base}/generate"
        if self.config.api_style == "chat_completions":
            return f"{api_base}/chat/completions"
        return f"{api_base}/completions"

    def _make_stats(self, **overrides: Any) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "cache_hit": False,
            "latency_sec": 0.0,
            "num_completions": 0,
            "api_style": self.config.api_style,
            "api_base": ",".join(self.config.api_bases),
            "url": "",
            "num_requests": 0,
            "failed_attempts": 0,
            "retries": 0,
            "remote_batch_size": int(self.config.remote_batch_size),
            "sglang_multi_sample": bool(self.config.sglang_multi_sample),
        }
        stats.update(overrides)
        stats["latency_sec"] = float(stats["latency_sec"])
        return stats

    @staticmethod
    def _format_request_error(exc: Exception) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            detail = f"HTTPError: status={exc.code} reason={exc.reason}"
            try:
                body = exc.read(4096).decode("utf-8", errors="replace").strip()
            except Exception as body_exc:
                body = f"<failed to read response body: {type(body_exc).__name__}: {body_exc}>"
            if body:
                detail = f"{detail}; response_body={body[:1000]}"
            return detail
        return f"{type(exc).__name__}: {exc}"

    def _build_request(self, api_base: str, prompt: str, n_samples: int) -> tuple[str, dict[str, Any]]:
        if self.config.api_style == "chat_completions":
            messages = []
            if self.config.system_prompt_text:
                messages.append({"role": "system", "content": self.config.system_prompt_text})
            messages.append({"role": "user", "content": prompt})
            return (
                f"{api_base}/chat/completions",
                {
                    "model": self.config.model_name,
                    "messages": messages,
                    "n": int(n_samples),
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_new_tokens,
                },
            )

        full_prompt = f"{self.config.system_prompt_text}\n{prompt}" if self.config.system_prompt_text else prompt
        if self.config.api_style == "sglang_generate":
            sampling_params = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_new_tokens": self.config.max_new_tokens,
                "skip_special_tokens": True,
            }
            if n_samples > 1:
                sampling_params["n"] = int(n_samples)
            return (
                self._request_url(api_base),
                {
                    "text": full_prompt,
                    "sampling_params": sampling_params,
                    "return_logprob": False,
                },
            )
        return (
            f"{api_base}/completions",
            {
                "model": self.config.model_name,
                "prompt": full_prompt,
                "n": int(n_samples),
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_new_tokens,
            },
        )

    def _parse_response(self, data: dict[str, Any]) -> list[str]:
        if self.config.api_style == "sglang_generate":
            if isinstance(data, list):
                completions: list[str] = []
                for item in data:
                    if isinstance(item, dict):
                        completions.extend(self._parse_sglang_generate_dict(item))
                    else:
                        completions.append(str(item))
                return completions
            if not isinstance(data, dict):
                return [str(data)]
            return self._parse_sglang_generate_dict(data)
        choices = data.get("choices", [])
        if self.config.api_style == "chat_completions":
            return [choice.get("message", {}).get("content", "") for choice in choices]
        return [choice.get("text", "") for choice in choices]

    def _parse_sglang_generate_dict(self, data: dict[str, Any]) -> list[str]:
        if "text" in data:
            text = data["text"]
            if isinstance(text, list):
                return [str(item) for item in text]
            return [str(text)]
        if "choices" in data:
            return [str(choice.get("text", "")) for choice in data.get("choices", [])]
        raise KeyError("text")


_CLIENTS: dict[G2RemoteTeacherConfig, G2RemoteTeacherClient] = {}


def _get_client(config: G2RemoteTeacherConfig) -> G2RemoteTeacherClient:
    if config not in _CLIENTS:
        _CLIENTS[config] = G2RemoteTeacherClient(config)
    return _CLIENTS[config]


def uses_standard_g2_remote_teacher(args: Any) -> bool:
    return (
        getattr(args, "distribution_reward_type", "pointwise") == "cf_l1oo"
        and getattr(args, "cf_target_mode", None) == "teacher"
        and getattr(args, "teacher_backend", None) == "remote"
    )


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for message in prompt:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return "\n".join(part for part in parts if part)


def attach_g2_teacher_completions(args: Any, group: list[Sample]) -> list[Sample]:
    """Sample one shared set of remote teacher completions for a prompt group."""
    if not uses_standard_g2_remote_teacher(args):
        return group
    if not group:
        return group

    prompt_text = _prompt_to_text(group[0].prompt)
    mismatched = [
        sample.index if sample.index is not None else idx
        for idx, sample in enumerate(group)
        if _prompt_to_text(sample.prompt) != prompt_text
    ]
    if mismatched:
        raise ValueError(f"G2 teacher prompt group contains mismatched prompts; sample_indices={mismatched}")

    n_teacher = int(getattr(args, "cf_teacher_n_samples", 0))
    if n_teacher <= 0:
        raise ValueError(f"--cf-teacher-n-samples must be positive, got {n_teacher}")

    client = _get_client(g2_remote_teacher_config_from_args(args))
    completions = client.sample_targets([prompt_text], n_teacher)[0]
    if len(completions) != n_teacher:
        raise ValueError(f"G2 remote teacher returned {len(completions)} completions, expected {n_teacher}")

    completion_list = [str(item) for item in completions]
    teacher_stats = client.last_stats[0] if getattr(client, "last_stats", None) else {}
    for sample in group:
        sample.metadata["g2_teacher_completions"] = completion_list
        sample.metadata["g2_teacher_n_samples"] = n_teacher
        sample.metadata["g2_teacher_stats"] = dict(teacher_stats)
    return group
