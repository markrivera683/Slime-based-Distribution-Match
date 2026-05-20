#!/usr/bin/env python3
from __future__ import annotations

import atexit
import argparse
import ast
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MBPP_FEWSHOT_PREFIX = '''"""
Write a function to find the similar elements from the given two tuple lists.
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
"""
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res)

"""
Write a python function to identify non-prime numbers.
assert is_not_prime(2) == False
"""
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result

"""
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
"""
import heapq as hq
def heap_queue_largest(nums,n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
'''


def log(message: str) -> None:
    print(message, flush=True)


def get_model_attention_heads(model_path: str) -> int | None:
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return None

    for candidate in (
        getattr(config, "num_attention_heads", None),
        getattr(getattr(config, "text_config", None), "num_attention_heads", None),
    ):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run code generation benchmarks on a checkpoint.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for benchmark outputs")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="humaneval,mbpp,multipl",
        help="Comma-separated subset of benchmarks to run: humaneval,mbpp,multipl",
    )
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "hf", "vllm"])
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--greedy_temperature", type=float, default=0.0)
    parser.add_argument("--sample_temperature", type=float, default=0.6)
    parser.add_argument("--passk_list", type=str, default="1,4,16")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of sampled completions per prompt")
    parser.add_argument("--greedy_only", action="store_true", default=False, help="Only run greedy decoding and skip sampled pass@k evaluation")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--greedy_batch_size", type=int, default=16)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--max_samples_per_benchmark", type=int, default=0, help="0 means full benchmark")
    parser.add_argument("--timeout_seconds", type=int, default=10)
    parser.add_argument(
        "--detail_preview_chars",
        type=int,
        default=4096,
        help="Max characters to keep for greedy/evaluated-code previews in benchmark_details.jsonl; 0 disables previews",
    )
    parser.add_argument("--skip_missing_toolchains", action="store_true", default=False)

    parser.add_argument("--humaneval_dataset", type=str, default="openai/openai_humaneval")
    parser.add_argument("--humaneval_split", type=str, default="test")
    parser.add_argument("--mbpp_dataset", type=str, default="google-research-datasets/mbpp")
    parser.add_argument("--mbpp_config", type=str, default="sanitized")
    parser.add_argument("--mbpp_split", type=str, default="test")
    parser.add_argument("--multipl_dataset", type=str, default="nuprl/MultiPL-E")
    parser.add_argument("--multipl_configs", type=str, default="humaneval-cpp,humaneval-js,humaneval-ts,humaneval-rs,humaneval-cs,humaneval-go,humaneval-php,humaneval-java")
    parser.add_argument("--multipl_split", type=str, default="test")
    return parser.parse_args()


def load_dataset_split(dataset_path: str, dataset_split: str, config_name: str | None = None):
    ext = os.path.splitext(dataset_path)[-1].lower()

    if os.path.isdir(dataset_path):
        try:
            data = load_from_disk(dataset_path)
        except Exception:
            load_kwargs = {}
            if config_name:
                load_kwargs["name"] = config_name
            data = load_dataset(dataset_path, **load_kwargs)
    elif ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
        file_type = ext.strip(".")
        if file_type == "jsonl":
            file_type = "json"
        data = load_dataset(file_type, data_files=dataset_path)
    else:
        load_kwargs = {}
        if config_name:
            load_kwargs["name"] = config_name
        try:
            return load_dataset(dataset_path, split=dataset_split, **load_kwargs)
        except Exception:
            data = load_dataset(dataset_path, **load_kwargs)

    if isinstance(data, DatasetDict):
        if dataset_split in data:
            return data[dataset_split]
        if "test" in data:
            return data["test"]
        if "validation" in data:
            return data["validation"]
        if "train" in data:
            return data["train"]
        return data[next(iter(data.keys()))]

    return data


def maybe_limit_dataset(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def truncate_at_stop(text: str, stop_tokens: list[str] | tuple[str, ...] | None) -> str:
    if not stop_tokens:
        return text
    min_idx = None
    for stop in stop_tokens:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            min_idx = idx if min_idx is None else min(min_idx, idx)
    return text if min_idx is None else text[:min_idx]


def truncate_preview(text: str | None, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def _sanitize_generated_code(response: str, keep_leading_def: bool = False) -> str:
    if not response:
        return ""
    text = response
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            block_lines = block.splitlines()
            if block_lines and block_lines[0].strip().lower().startswith(("python", "py")):
                block = "\n".join(block_lines[1:])
            text = block
    lines = text.splitlines()
    if keep_leading_def:
        start_idx = 0
        while start_idx < len(lines):
            stripped = lines[start_idx].strip()
            if not stripped or stripped.startswith("#"):
                start_idx += 1
                continue
            if stripped.startswith(("def ", "class ", "import ", "from ", "@")):
                break
            start_idx += 1
        lines = lines[start_idx:]
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped:
            lowered = stripped.lower()
            if stripped.startswith(('"""', "'''")):
                if not keep_leading_def:
                    break
                quote = stripped[:3]
                if stripped.count(quote) >= 2 and stripped != quote:
                    i += 1
                    continue
                i += 1
                while i < len(lines):
                    if quote in lines[i]:
                        i += 1
                        break
                    i += 1
                continue
            if lowered.startswith(("assert ", "print(", "print ", "if __name__")):
                break
            if stripped.startswith(("def ", "class ")) and not line.startswith((" ", "\t")) and not keep_leading_def:
                break
            if lowered.startswith(("# test", "# tests", "#test", "#tests")):
                break
        cleaned_lines.append(line)
        i += 1
    return "\n".join(cleaned_lines).rstrip()


def _build_mbpp_code(
    prompt: str,
    response: str,
    function_name: str,
    helper_code: str,
    function_signature: str | None = None,
    keep_leading_def: bool | None = None,
) -> str:
    prompt_lines = (prompt or "").splitlines()
    prompt_has_target_def = False
    if function_name:
        prompt_has_target_def = any(line.lstrip().startswith(f"def {function_name}(") for line in prompt_lines)
    if not function_name and prompt_lines:
        prompt_has_target_def = any(line.lstrip().startswith(("def ", "class ")) for line in prompt_lines)
    if keep_leading_def is None:
        keep_leading_def = not prompt_has_target_def
    cleaned = _sanitize_generated_code(response or "", keep_leading_def=keep_leading_def)
    if not cleaned.strip():
        cleaned = ""

    lines = cleaned.splitlines() if cleaned else []
    def_indices = [
        idx
        for idx, line in enumerate(lines)
        if line.lstrip().startswith("def ") and not line.startswith((" ", "\t"))
    ]
    code_text = cleaned.rstrip()

    if def_indices:
        start_idx = def_indices[0]
        found_target = False
        if function_name:
            for idx in def_indices:
                line = lines[idx].lstrip()
                if line.startswith(f"def {function_name}("):
                    start_idx = idx
                    found_target = True
                    break
        else:
            found_target = True
        if function_name and not found_target:
            if helper_code:
                helper_code = helper_code.rstrip()
                if helper_code:
                    code_text = helper_code + "\n\n" + code_text
            return code_text.strip()
        code_text = "\n".join(lines[start_idx:]).rstrip()
        if not prompt_has_target_def and found_target:
            full_code = code_text
            if helper_code:
                helper_code = helper_code.rstrip()
                if helper_code:
                    full_code = helper_code + "\n\n" + full_code
            return full_code.strip()

    if not prompt_has_target_def and function_signature:
        signature = function_signature.rstrip()
        if not code_text.strip():
            full_code = signature + "\n    pass"
        else:
            full_code = signature + "\n" + _normalize_python_body(code_text)
    elif not prompt_has_target_def:
        full_code = code_text
    else:
        if code_text.strip():
            full_code = prompt.rstrip() + "\n" + _normalize_python_body(code_text)
        else:
            full_code = prompt.rstrip() + "\n    pass"

    if helper_code:
        helper_code = helper_code.rstrip()
        if helper_code:
            full_code = helper_code + "\n\n" + full_code

    return full_code.strip()


def _normalize_python_body(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "    pass"
    expanded = [line.expandtabs(4) for line in lines]
    indents = [len(line) - len(line.lstrip()) for line in expanded if line.strip()]
    base_indent = min(indents) if indents else 0
    normalized = []
    for line in expanded:
        if not line.strip():
            normalized.append("")
        else:
            stripped = line[base_indent:] if len(line) >= base_indent else line.lstrip()
            normalized.append("    " + stripped.rstrip())
    return "\n".join(normalized)


def _run_python_code_in_subprocess(code: str, unit_tests: list[str], timeout: int = 3) -> tuple[bool, str | None]:
    test_script = textwrap.dedent(
        f"""
        import os
        import signal
        import sys

        _original_exit = sys.exit
        def _safe_exit(code=0):
            raise SystemExit(code)
        sys.exit = _safe_exit

        _original_os_exit = os._exit
        def _safe_os_exit(code=0):
            raise SystemExit(code)
        os._exit = _safe_os_exit

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        namespace = {{
            "__builtins__": __builtins__,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }}

        try:
            exec({code!r}, namespace)
        except SyntaxError:
            print("ERROR:syntax")
            sys.stdout.flush()
            _original_exit(0)
        except SystemExit:
            print("ERROR:syntax")
            sys.stdout.flush()
            _original_exit(0)
        except Exception:
            print("ERROR:runtime")
            sys.stdout.flush()
            _original_exit(0)

        unit_tests = {unit_tests!r}
        try:
            for test in unit_tests:
                stmt = str(test).strip()
                if not stmt:
                    continue
                exec(stmt, namespace)
        except AssertionError:
            print("ERROR:test_failure")
            sys.stdout.flush()
            _original_exit(0)
        except SystemExit:
            print("ERROR:test_failure")
            sys.stdout.flush()
            _original_exit(0)
        except Exception:
            print("ERROR:test_failure")
            sys.stdout.flush()
            _original_exit(0)

        print("SUCCESS")
        sys.stdout.flush()
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            close_fds=True,
            start_new_session=True,
        )
        output = result.stdout.strip()
        if "SUCCESS" in output:
            return True, None
        if "ERROR:syntax" in output:
            return False, "syntax"
        if "ERROR:test_failure" in output:
            return False, "test_failure"
        if "ERROR:runtime" in output:
            return False, "runtime"
        return False, "syntax"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, "syntax"


def run_command(command: list[str], cwd: str, timeout: int) -> tuple[bool, str, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            start_new_session=True,
        )
        ok = result.returncode == 0
        return ok, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as exc:
        return False, "", str(exc)


def estimate_pass_at_k(n: int, c: int, k: int) -> float | None:
    if n < k:
        return None
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(n - c + 1, n + 1):
        product *= 1.0 - k / i
    return 1.0 - product


class TextGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend = args.backend
        self.llm = None
        self.model = None
        self.hf_device = None
        self._closed = False
        atexit.register(self.close)
        log(f"[benchmark] Loading tokenizer from {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models require left padding for batched generation; otherwise
        # right-padded prompts produce broken (often syntactically invalid) outputs.
        self.tokenizer.padding_side = "left"
        self.model_attention_heads = get_model_attention_heads(args.model_path)
        if self.backend == "auto":
            try:
                from vllm import LLM  # noqa: F401
                if self._is_vllm_tp_compatible():
                    self.backend = "vllm"
                else:
                    self.backend = "hf"
            except Exception:
                self.backend = "hf"
        elif self.backend == "vllm":
            self._validate_vllm_tp_compatible()
        log(f"[benchmark] Using generation backend: {self.backend}")
        if self.backend == "vllm":
            self._init_vllm()
        else:
            self._init_hf()

    def _is_vllm_tp_compatible(self) -> bool:
        try:
            self._validate_vllm_tp_compatible()
            return True
        except ValueError as exc:
            log(f"[benchmark] {exc}; falling back to Hugging Face backend")
            return False

    def _validate_vllm_tp_compatible(self):
        if self.args.tp_size <= 1:
            return
        heads = self.model_attention_heads
        if heads is not None and heads % self.args.tp_size != 0:
            raise ValueError(
                f"vLLM tensor_parallel_size={self.args.tp_size} is incompatible with model attention heads={heads}"
            )

    def _init_vllm(self):
        from vllm import LLM

        log("[benchmark] Initializing vLLM engine")
        supplement_dir = Path(__file__).resolve().parents[1] / "supplement"
        if str(supplement_dir) not in sys.path:
            sys.path.insert(0, str(supplement_dir))

        from vllm_generate_progress import (  # noqa: WPS433
            chained_hf_overrides,
            ensure_qwen35_config_registered,
            ensure_qwen35_preprocessor_files,
            fail_fast_on_qwen35_tp_incompatibility,
            gemma4_text_only_hf_overrides,
            prepare_qwen35_text_only_shim_env,
            qwen35_hf_overrides,
        )

        ensure_qwen35_config_registered()
        qwen35_text_only_shim = fail_fast_on_qwen35_tp_incompatibility(
            self.args.model_path,
            max(1, self.args.tp_size),
            enable_text_only_shim=True,
        )
        if qwen35_text_only_shim:
            prepare_qwen35_text_only_shim_env()
            log(
                "[compat] enabling Qwen3.5 text-only shim for benchmark vLLM "
                f"(tp_size={self.args.tp_size})"
            )
        ensure_qwen35_preprocessor_files(self.args.model_path)

        disable_car_env = os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
        disable_custom_all_reduce = disable_car_env not in ("0", "false", "False", "")
        enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") not in ("0", "false", "False", "")

        self.llm = LLM(
            model=self.args.model_path,
            tensor_parallel_size=max(1, self.args.tp_size),
            trust_remote_code=True,
            seed=self.args.seed,
            max_num_seqs=self.args.max_num_seqs,
            enable_prefix_caching=self.args.enable_prefix_caching,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
            hf_overrides=chained_hf_overrides(
                partial(
                    qwen35_hf_overrides,
                    enable_text_only_shim=bool(qwen35_text_only_shim),
                ),
                gemma4_text_only_hf_overrides,
            ),
        )

    def _init_hf(self):
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        log("[benchmark] Initializing Hugging Face model")
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
            try:
                load_kwargs["device_map"] = "auto"
                self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path, **load_kwargs)
            except Exception:
                load_kwargs.pop("device_map", None)
                self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path, **load_kwargs)
                self.model = self.model.to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path, **load_kwargs)
        self.model.eval()
        self.hf_device = next(self.model.parameters()).device
        log(f"[benchmark] HF model ready on device {self.hf_device}")

    def close(self):
        if self._closed:
            return
        self._closed = True

        if self.llm is not None:
            llm_obj = self.llm
            self.llm = None
            try:
                llm_engine = getattr(llm_obj, "llm_engine", None)
                shutdown = getattr(llm_engine, "shutdown", None)
                if callable(shutdown):
                    log("[benchmark] Shutting down vLLM engine")
                    try:
                        shutdown()
                    except TypeError:
                        shutdown(timeout=5)
                    log("[benchmark] vLLM engine shutdown complete")
            except Exception as exc:
                log(f"[benchmark] Warning: failed to cleanly shutdown vLLM engine: {exc}")

        if self.model is not None:
            self.model = None

        self.tokenizer = None
        self.hf_device = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def generate(self, requests: list[dict[str, Any]], temperature: float, n_samples: int, batch_size: int) -> list[list[str]]:
        if self.backend == "vllm":
            return self._generate_vllm(requests, temperature, n_samples)
        return self._generate_hf(requests, temperature, n_samples, batch_size)

    def _truncate_prompt_for_vllm(self, prompt: str) -> str:
        max_len = max(0, int(self.args.prompt_max_len or 0))
        if max_len <= 0:
            return prompt
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= max_len:
            return prompt
        return self.tokenizer.decode(token_ids[:max_len], skip_special_tokens=True)

    def _generate_vllm(self, requests: list[dict[str, Any]], temperature: float, n_samples: int) -> list[list[str]]:
        from vllm import SamplingParams

        grouped: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
        for idx, req in enumerate(requests):
            stop_key = tuple(req.get("stop_tokens") or [])
            grouped.setdefault(stop_key, []).append((idx, req))

        outputs: list[list[str]] = [[] for _ in requests]
        for stop_key, items in grouped.items():
            prompts = [self._truncate_prompt_for_vllm(req["prompt"]) for _, req in items]
            sampling_params = SamplingParams(
                max_tokens=self.args.max_new_tokens,
                top_p=self.args.top_p,
                temperature=temperature,
                repetition_penalty=self.args.repetition_penalty,
                n=n_samples,
                include_stop_str_in_output=True,
                stop=list(stop_key) if stop_key else None,
                # Strip EOS / pad tokens so they don't leak into the executed code as
                # raw "<|endoftext|>"-style strings (causes spurious SyntaxError).
                skip_special_tokens=True,
            )
            generations = self.llm.generate(prompts, sampling_params)
            for (orig_idx, req), generation in zip(items, generations):
                samples = [truncate_at_stop(output.text, req.get("stop_tokens")) for output in generation.outputs]
                outputs[orig_idx] = samples
        return outputs

    def _generate_hf(self, requests: list[dict[str, Any]], temperature: float, n_samples: int, batch_size: int) -> list[list[str]]:
        outputs: list[list[str]] = []
        do_sample = temperature > 0.0
        # Hard guarantee: decoder-only batched generation requires left padding so that the
        # generated continuations align with prompt_len for every sample in the batch.
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            prompts = [item["prompt"] for item in batch]
            tokenized = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.prompt_max_len,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            prompt_len = input_ids.shape[1]
            if torch.cuda.is_available():
                input_ids = input_ids.to(self.hf_device)
                attention_mask = attention_mask.to(self.hf_device)
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.args.max_new_tokens,
                "do_sample": do_sample,
                "top_p": self.args.top_p,
                "repetition_penalty": self.args.repetition_penalty,
                "num_return_sequences": n_samples,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                generate_kwargs["temperature"] = max(temperature, 1e-5)
            with torch.no_grad():
                generated = self.model.generate(**generate_kwargs)
            generated = generated[:, prompt_len:]
            # Strip EOS / pad tokens so they don't leak into the executed code as
            # raw "<|endoftext|>"-style strings (causes spurious SyntaxError).
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for idx, req in enumerate(batch):
                start_idx = idx * n_samples
                end_idx = start_idx + n_samples
                samples = [truncate_at_stop(text, req.get("stop_tokens")) for text in decoded[start_idx:end_idx]]
                outputs.append(samples)
        return outputs


def build_humaneval_code(prompt: str, completion: str) -> str:
    cleaned_response = _sanitize_generated_code(completion or "")
    return prompt.rstrip() + "\n" + cleaned_response


def evaluate_humaneval_completion(prompt: str, completion: str, test_code: str, entry_point: str | None, timeout: int) -> tuple[bool, str | None]:
    full_code = build_humaneval_code(prompt, completion)
    unit_tests = [test_code, f"check({entry_point})"] if entry_point else [test_code]
    return _run_python_code_in_subprocess(full_code, unit_tests, timeout=timeout)


def build_mbpp_evaluated_code(row: dict[str, Any], completion: str) -> str:
    prompt = row["prompt_for_model"]
    function_name = row["function_name"]
    helper_code = row["helper_code"]
    function_signature = row["function_signature"]
    return _build_mbpp_code(prompt, completion, function_name, helper_code, function_signature=function_signature, keep_leading_def=True)


def evaluate_mbpp_completion(row: dict[str, Any], completion: str, timeout: int) -> tuple[bool, str | None]:
    unit_tests = row["unit_tests"]
    code = build_mbpp_evaluated_code(row, completion)
    return _run_python_code_in_subprocess(code, unit_tests, timeout=timeout)


def command_exists(*candidates: str) -> str | None:
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def get_multipl_runtime(language: str) -> tuple[bool, str | None]:
    checks = {
        "cpp": command_exists("g++"),
        "js": command_exists("node"),
        "ts": command_exists("tsc") and command_exists("node"),
        "rs": command_exists("rustc"),
        "cs": (command_exists("mcs") and command_exists("mono")) or (command_exists("csc") and command_exists("mono")),
        "go": command_exists("go"),
        "php": command_exists("php"),
        "java": command_exists("javac") and command_exists("java"),
    }
    tool = checks.get(language)
    return (tool is not None, str(tool) if tool else None)


def evaluate_multipl_completion(language: str, prompt: str, completion: str, tests: str, timeout: int) -> tuple[bool | None, str]:
    runtime_ok, runtime_detail = get_multipl_runtime(language)
    if not runtime_ok:
        return None, f"missing_toolchain:{language}"

    source = f"{prompt}{completion}{tests}"
    with tempfile.TemporaryDirectory(prefix=f"multipl_{language}_") as tmpdir:
        tmp_path = Path(tmpdir)
        try:
            if language == "cpp":
                src = tmp_path / "main.cpp"
                exe = tmp_path / "main"
                src.write_text(source, encoding="utf-8")
                ok, _, err = run_command(["g++", "-std=c++17", "-O2", str(src), "-o", str(exe)], tmpdir, timeout)
                if not ok:
                    return False, f"compile_error:{err[:200]}"
                ok, _, err = run_command([str(exe)], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"

            if language == "js":
                src = tmp_path / "main.js"
                src.write_text(source, encoding="utf-8")
                ok, _, err = run_command(["node", str(src)], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"

            if language == "ts":
                src = tmp_path / "main.ts"
                js_out = tmp_path / "main.js"
                src.write_text(source, encoding="utf-8")
                ok, _, err = run_command(["tsc", str(src), "--target", "es2020", "--module", "commonjs", "--outDir", str(tmp_path)], tmpdir, timeout)
                if not ok:
                    return False, f"compile_error:{err[:200]}"
                ok, _, err = run_command(["node", str(js_out)], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"

            if language == "rs":
                src = tmp_path / "main.rs"
                exe = tmp_path / "main"
                src.write_text(source, encoding="utf-8")
                ok, _, err = run_command(["rustc", str(src), "-O", "-o", str(exe)], tmpdir, timeout)
                if not ok:
                    return False, f"compile_error:{err[:200]}"
                ok, _, err = run_command([str(exe)], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"

            if language == "cs":
                src = tmp_path / "Program.cs"
                exe = tmp_path / "program.exe"
                src.write_text(source, encoding="utf-8")
                compiler = command_exists("mcs", "csc")
                if compiler is None:
                    return None, "missing_toolchain:cs"
                if Path(compiler).name == "mcs":
                    ok, _, err = run_command([compiler, str(src), f"-out:{exe}"], tmpdir, timeout)
                else:
                    ok, _, err = run_command([compiler, f"/out:{exe}", str(src)], tmpdir, timeout)
                if not ok:
                    return False, f"compile_error:{err[:200]}"
                ok, _, err = run_command(["mono", str(exe)], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"

            if language == "go":
                src = tmp_path / "main.go"
                src.write_text(source, encoding="utf-8")
                ok, _, err = run_command(["go", "run", str(src)], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"

            if language == "php":
                src = tmp_path / "main.php"
                src.write_text(source, encoding="utf-8")
                ok, _, err = run_command(["php", str(src)], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"

            if language == "java":
                match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", source)
                class_name = match.group(1) if match else "Main"
                src = tmp_path / f"{class_name}.java"
                src.write_text(source, encoding="utf-8")
                ok, _, err = run_command(["javac", str(src)], tmpdir, timeout)
                if not ok:
                    return False, f"compile_error:{err[:200]}"
                ok, _, err = run_command(["java", "-cp", str(tmp_path), class_name], tmpdir, timeout)
                return ok, "ok" if ok else f"runtime_error:{err[:200]}"
        except Exception as exc:
            return False, f"exception:{exc}"

    return False, "unsupported_language"


# HumanEval needs explicit stop tokens. Without them, models trained on
# markdown-style tutorial code (e.g. the opencode-instruct dataset) emit a
# trailing ``` after a perfectly valid function body, which then gets executed
# verbatim and triggers SyntaxError on every task. We mirror MBPP's stop list
# and add a few HumanEval-specific guards (\nclass and the bare \ndef so the
# model does not start defining unrelated next-tasks below the target body).
HUMANEVAL_STOP_TOKENS = [
    "\nclass",
    "\ndef ",
    "\n\ndef ",
    "\nif __name__",
    "\nprint(",
    "\nassert ",
    '\n"""',
    "\n```",
    "\n#",
]


def build_humaneval_requests(dataset) -> list[dict[str, Any]]:
    requests = []
    for row in dataset:
        prompt = row.get("prompt") or row.get("question") or ""
        requests.append(
            {
                "benchmark": "HumanEval",
                "task_id": row.get("task_id", ""),
                "prompt": prompt,
                "stop_tokens": list(HUMANEVAL_STOP_TOKENS),
                "test_code": row["test"],
                "entry_point": row.get("entry_point"),
            }
        )
    return requests


def build_mbpp_requests(dataset) -> list[dict[str, Any]]:
    requests = []
    for row in dataset:
        prompt = row.get("prompt") or row.get("text") or row.get("question") or ""
        tests = row.get("test_list") or row.get("tests") or []
        test_imports = row.get("test_imports") or []
        test_setup_code = row.get("test_setup_code") or ""
        helper_parts = []
        if isinstance(test_imports, list):
            helper_parts.extend(str(item) for item in test_imports if item)
        elif test_imports:
            helper_parts.append(str(test_imports))
        if test_setup_code:
            helper_parts.append(str(test_setup_code))
        helper_code = "\n".join(part.strip("\n") for part in helper_parts if part).strip()

        function_name = None
        function_signature = None
        code_hint = row.get("code") or row.get("answer") or ""
        for line in str(code_hint).splitlines():
            stripped = line.strip()
            if stripped.startswith("def "):
                function_signature = stripped
                function_name = stripped[4:].split("(", 1)[0].strip()
                break

        prompt_for_model = prompt
        if MBPP_FEWSHOT_PREFIX not in prompt_for_model:
            prompt_for_model = MBPP_FEWSHOT_PREFIX + "\n\n" + prompt_for_model

        requests.append(
            {
                "benchmark": "MBPP",
                "task_id": row.get("task_id", ""),
                "prompt": prompt_for_model,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
                "function_name": function_name,
                "function_signature": function_signature,
                "helper_code": helper_code,
                "unit_tests": tests,
                "prompt_for_model": prompt_for_model,
            }
        )
    return requests


def build_multipl_requests(dataset, config_name: str) -> list[dict[str, Any]]:
    requests = []
    language = config_name.split("-")[-1]
    for idx, row in enumerate(dataset):
        requests.append(
            {
                "benchmark": f"MultiPL-E/{config_name}",
                "task_id": row.get("name", idx),
                "prompt": row["prompt"],
                "stop_tokens": row.get("stop_tokens") or [],
                "tests": row["tests"],
                "language": language,
                "config_name": config_name,
            }
        )
    return requests


def evaluate_request(
    request: dict[str, Any],
    greedy_output: str,
    sampled_outputs: list[str],
    timeout: int,
    passk_list: list[int],
    detail_preview_chars: int,
) -> dict[str, Any]:
    benchmark = request["benchmark"]
    sampled_outputs = sampled_outputs or []
    evaluated_code = ""
    unit_tests_preview: list[str] | str = []
    if benchmark == "HumanEval":
        evaluated_code = build_humaneval_code(request["prompt"], greedy_output)
        unit_tests_preview = [request["test_code"], f"check({request['entry_point']})"] if request["entry_point"] else [request["test_code"]]
        greedy_correct, greedy_error = evaluate_humaneval_completion(
            request["prompt"], greedy_output, request["test_code"], request["entry_point"], timeout
        )
        sample_results = [
            evaluate_humaneval_completion(request["prompt"], sample, request["test_code"], request["entry_point"], timeout)[0]
            for sample in sampled_outputs
        ]
    elif benchmark == "MBPP":
        evaluated_code = build_mbpp_evaluated_code(request, greedy_output)
        unit_tests_preview = request["unit_tests"]
        greedy_correct, greedy_error = evaluate_mbpp_completion(request, greedy_output, timeout)
        sample_results = [evaluate_mbpp_completion(request, sample, timeout)[0] for sample in sampled_outputs]
    else:
        evaluated_code = f"{request['prompt']}{greedy_output}{request['tests']}"
        unit_tests_preview = request["tests"]
        greedy_eval, greedy_error = evaluate_multipl_completion(
            request["language"], request["prompt"], greedy_output, request["tests"], timeout
        )
        if greedy_eval is None:
            return {
                "benchmark": benchmark,
                "task_id": request["task_id"],
                "status": "skipped",
                "reason": greedy_error,
                "greedy_output_preview": truncate_preview(greedy_output, detail_preview_chars),
                "evaluated_code_preview": truncate_preview(evaluated_code, detail_preview_chars),
                "unit_tests_preview": truncate_preview(str(unit_tests_preview), detail_preview_chars),
            }
        greedy_correct = greedy_eval
        sample_results = []
        for sample in sampled_outputs:
            sample_eval, _ = evaluate_multipl_completion(request["language"], request["prompt"], sample, request["tests"], timeout)
            sample_results.append(bool(sample_eval))

    correct_count = sum(bool(x) for x in sample_results)
    record = {
        "benchmark": benchmark,
        "task_id": request["task_id"],
        "status": "ok",
        "greedy_correct": bool(greedy_correct),
        "greedy_error_type": greedy_error,
        "num_samples": len(sample_results),
        "num_correct_samples": correct_count,
    }
    if detail_preview_chars > 0:
        record.update(
            {
                "greedy_output_preview": truncate_preview(greedy_output, detail_preview_chars),
                "evaluated_code_preview": truncate_preview(evaluated_code, detail_preview_chars),
                "unit_tests_preview": truncate_preview(str(unit_tests_preview), detail_preview_chars),
            }
        )
    for k in passk_list:
        value = estimate_pass_at_k(len(sample_results), correct_count, k)
        record[f"pass@{k}"] = value
    return record


def summarize_records(benchmark_name: str, records: list[dict[str, Any]], passk_list: list[int]) -> dict[str, Any]:
    skipped = [r for r in records if r.get("status") == "skipped"]
    completed = [r for r in records if r.get("status") == "ok"]
    summary = {
        "benchmark": benchmark_name,
        "status": "completed" if completed else "skipped",
        "num_tasks": len(records),
        "num_completed": len(completed),
        "num_skipped": len(skipped),
        "skip_reasons": dict(Counter(r.get("reason", "unknown") for r in skipped)),
    }
    if completed:
        summary["greedy_accuracy"] = sum(float(r["greedy_correct"]) for r in completed) / len(completed)
        for k in passk_list:
            values = [r[f"pass@{k}"] for r in completed if r.get(f"pass@{k}") is not None]
            summary[f"pass@{k}"] = sum(values) / len(values) if values else None
    else:
        summary["greedy_accuracy"] = None
        for k in passk_list:
            summary[f"pass@{k}"] = None
    return summary


def run_benchmark_group(
    generator: TextGenerator,
    benchmark_name: str,
    requests: list[dict[str, Any]],
    greedy_temperature: float,
    sample_temperature: float,
    n_samples: int,
    greedy_batch_size: int,
    sample_batch_size: int,
    timeout: int,
    passk_list: list[int],
    detail_preview_chars: int,
    greedy_only: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    log(f"[benchmark] Running {benchmark_name} on {len(requests)} prompts")
    log(f"[benchmark] {benchmark_name}: generating greedy completions")
    greedy_outputs = generator.generate(requests, greedy_temperature, 1, greedy_batch_size)
    if greedy_only:
        sampled_outputs = [[] for _ in requests]
    else:
        log(f"[benchmark] {benchmark_name}: generating sampled completions (n={n_samples})")
        sampled_outputs = generator.generate(requests, sample_temperature, n_samples, sample_batch_size)
    log(f"[benchmark] {benchmark_name}: executing evaluators")

    records = []
    for request, greedy_sample, sampled in zip(requests, greedy_outputs, sampled_outputs):
        greedy_text = greedy_sample[0] if greedy_sample else ""
        record = evaluate_request(request, greedy_text, sampled, timeout, passk_list, detail_preview_chars)
        records.append(record)

    summary = summarize_records(benchmark_name, records, passk_list)
    return summary, records


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    passk_list = [] if args.greedy_only else [int(item.strip()) for item in args.passk_list.split(",") if item.strip()]
    benchmarks_to_run = {item.strip().lower() for item in args.benchmarks.split(",") if item.strip()}
    valid_benchmarks = {"humaneval", "mbpp", "multipl"}
    unknown = benchmarks_to_run - valid_benchmarks
    if unknown:
        raise ValueError(f"Unknown benchmark names: {sorted(unknown)}")
    if not args.greedy_only and not passk_list:
        raise ValueError("passk_list must contain at least one k")
    if passk_list and max(passk_list) > args.n_samples:
        raise ValueError(f"max(passk_list)={max(passk_list)} must be <= n_samples={args.n_samples}")

    generator = None
    try:
        generator = TextGenerator(args)
        summary_rows: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []

        if "humaneval" in benchmarks_to_run:
            log(f"[benchmark] Loading HumanEval dataset: {args.humaneval_dataset} [{args.humaneval_split}]")
            humaneval_dataset = maybe_limit_dataset(
                load_dataset_split(args.humaneval_dataset, args.humaneval_split), args.max_samples_per_benchmark
            )
            humaneval_requests = build_humaneval_requests(humaneval_dataset)
            summary, records = run_benchmark_group(
                generator,
                "HumanEval",
                humaneval_requests,
                args.greedy_temperature,
                args.sample_temperature,
                args.n_samples,
                args.greedy_batch_size,
                args.sample_batch_size,
                args.timeout_seconds,
                passk_list,
                args.detail_preview_chars,
                args.greedy_only,
            )
            summary_rows.append(summary)
            detail_rows.extend(records)

        if "mbpp" in benchmarks_to_run:
            log(f"[benchmark] Loading MBPP dataset: {args.mbpp_dataset} ({args.mbpp_config}) [{args.mbpp_split}]")
            mbpp_dataset = maybe_limit_dataset(
                load_dataset_split(args.mbpp_dataset, args.mbpp_split, config_name=args.mbpp_config),
                args.max_samples_per_benchmark,
            )
            mbpp_requests = build_mbpp_requests(mbpp_dataset)
            summary, records = run_benchmark_group(
                generator,
                "MBPP",
                mbpp_requests,
                args.greedy_temperature,
                args.sample_temperature,
                args.n_samples,
                args.greedy_batch_size,
                args.sample_batch_size,
                args.timeout_seconds,
                passk_list,
                args.detail_preview_chars,
                args.greedy_only,
            )
            summary_rows.append(summary)
            detail_rows.extend(records)

        if "multipl" in benchmarks_to_run:
            multipl_configs = [item.strip() for item in args.multipl_configs.split(",") if item.strip()]
            for config_name in multipl_configs:
                language = config_name.split("-")[-1]
                runtime_ok, runtime_detail = get_multipl_runtime(language)
                benchmark_name = f"MultiPL-E/{config_name}"
                if not runtime_ok:
                    summary_rows.append(
                        {
                            "benchmark": benchmark_name,
                            "status": "skipped",
                            "num_tasks": 0,
                            "num_completed": 0,
                            "num_skipped": 0,
                            "skip_reasons": {f"missing_toolchain:{language}": 1},
                            "greedy_accuracy": None,
                            **{f"pass@{k}": None for k in passk_list},
                        }
                    )
                    log(f"[benchmark] Skipping {benchmark_name}: missing toolchain ({runtime_detail})")
                    if args.skip_missing_toolchains:
                        continue
                    continue

                log(f"[benchmark] Loading MultiPL-E dataset: {args.multipl_dataset} ({config_name}) [{args.multipl_split}]")
                dataset = maybe_limit_dataset(
                    load_dataset_split(args.multipl_dataset, args.multipl_split, config_name=config_name),
                    args.max_samples_per_benchmark,
                )
                requests = build_multipl_requests(dataset, config_name)
                summary, records = run_benchmark_group(
                    generator,
                    benchmark_name,
                    requests,
                    args.greedy_temperature,
                    args.sample_temperature,
                    args.n_samples,
                    args.greedy_batch_size,
                    args.sample_batch_size,
                    args.timeout_seconds,
                    passk_list,
                    args.detail_preview_chars,
                    args.greedy_only,
                )
                summary_rows.append(summary)
                detail_rows.extend(records)

        summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
        details_path = os.path.join(args.output_dir, "benchmark_details.jsonl")
        metadata = {
            "model_path": args.model_path,
            "backend": generator.backend,
            "seed": args.seed,
            "greedy_temperature": args.greedy_temperature,
            "sample_temperature": args.sample_temperature,
            "n_samples": args.n_samples,
            "greedy_only": args.greedy_only,
            "passk_list": passk_list,
            "prompt_max_len": args.prompt_max_len,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_samples_per_benchmark": args.max_samples_per_benchmark,
            "detail_preview_chars": args.detail_preview_chars,
            "summaries": summary_rows,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(details_path, "w", encoding="utf-8") as f:
            for row in detail_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("=" * 72, flush=True)
        print("Benchmark summary", flush=True)
        print("=" * 72, flush=True)
        for summary in summary_rows:
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        log(f"[benchmark] summary written to {summary_path}")
        log(f"[benchmark] details written to {details_path}")
        return 0
    finally:
        if generator is not None:
            generator.close()


if __name__ == "__main__":
    raise SystemExit(main())
