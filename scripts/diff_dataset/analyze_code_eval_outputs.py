#!/usr/bin/env python3
"""Deprecated: analyze code-generation outputs and optionally re-test generated code.

DEPRECATED: use scripts/benchmarks/run_code_generation_benchmarks.py for
pass@k generation, evaluation, and analysis reports.

This is intended for quick post-mortems when pass@1 is unexpectedly low:
it reports output length/format statistics and, for MBPP, delegates code
construction and unit-test execution to scripts/benchmarks/run_code_generation_benchmarks.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>")
LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def first_def_name(code: str) -> str | None:
    for line in str(code or "").splitlines():
        match = re.match(r"\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if match:
            return match.group(1)
    return None


def first_def_signature(code: str) -> tuple[str | None, str | None]:
    for line in str(code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped[4:].split("(", 1)[0].strip(), stripped
    return None, None


def strip_special_tokens(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def fenced_blocks(text: str) -> list[str]:
    if "```" not in text:
        return []
    blocks = []
    parts = text.split("```")
    for block in parts[1::2]:
        lines = block.splitlines()
        if lines and lines[0].strip().lower().startswith(("python", "py")):
            block = "\n".join(lines[1:])
        blocks.append(block)
    return blocks


def choose_code_region(text: str, target_name: str | None) -> str:
    text = strip_special_tokens(text or "")
    blocks = fenced_blocks(text)
    for block in blocks:
        if target_name and re.search(rf"^\s*def\s+{re.escape(target_name)}\s*\(", block, re.MULTILINE):
            return block
    for block in blocks:
        if re.search(r"^\s*def\s+", block, re.MULTILINE):
            return block
    return text


def extract_mbpp_candidate(text: str, target_name: str | None) -> str:
    region = choose_code_region(text, target_name)
    lines = region.splitlines()

    target_idx = None
    if target_name:
        pattern = re.compile(rf"^\s*def\s+{re.escape(target_name)}\s*\(")
        for idx, line in enumerate(lines):
            if pattern.match(line):
                target_idx = idx
                break
    if target_idx is None:
        for idx, line in enumerate(lines):
            if re.match(r"^\s*def\s+", line):
                target_idx = idx
                break
    if target_idx is None:
        return region.strip()

    prefix = []
    for line in lines[:target_idx]:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            prefix.append(line)

    body = []
    for line in lines[target_idx:]:
        stripped = line.strip()
        if body:
            if stripped.startswith(("```", "<|im_end|>", "<|endoftext|>")):
                break
            if re.match(r"^(assert\s|print\s*\(|if __name__)", stripped):
                break
            if re.match(r"^(def|class)\s+", line):
                break
        body.append(line)

    return "\n".join(prefix + body).strip()


def load_benchmark_helpers(repo_root: Path):
    helper_path = repo_root / "scripts" / "benchmarks" / "run_code_generation_benchmarks.py"
    spec = importlib.util.spec_from_file_location("code_bench_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_lengths(values: list[int]) -> dict[str, Any]:
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "mean": round(statistics.mean(values), 2) if values else None,
    }


def truncate_preview(text: str | None, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DEPRECATED: use scripts/benchmarks/run_code_generation_benchmarks.py "
            "for code-generation benchmark analysis."
        )
    )
    parser.add_argument("--benchmark", choices=["mbpp"], default="mbpp")
    parser.add_argument("--source_jsonl", required=True)
    parser.add_argument("--results_jsonl", required=True)
    parser.add_argument("--details_jsonl")
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--examples_jsonl")
    parser.add_argument("--repo_root", default=str(LOCAL_REPO_ROOT))
    parser.add_argument("--timeout_seconds", type=int, default=10)
    parser.add_argument("--preview_chars", type=int, default=1200)
    parser.add_argument("--run_extracted_eval", action="store_true")
    args = parser.parse_args()

    source_rows = load_jsonl(Path(args.source_jsonl))
    result_rows = load_jsonl(Path(args.results_jsonl))
    details_by_idx = {}
    if args.details_jsonl:
        for row in load_jsonl(Path(args.details_jsonl)):
            details_by_idx[int(row.get("source_idx", len(details_by_idx)))] = row

    outputs_by_idx = {}
    for row in result_rows:
        source_idx = int(row.get("source_idx", row.get("idx", len(outputs_by_idx))))
        outputs_by_idx.setdefault(source_idx, row)

    helper = load_benchmark_helpers(Path(args.repo_root)) if args.run_extracted_eval else None
    lengths = []
    extracted_lengths = []
    format_counts = Counter()
    eval_counts = Counter()
    detail_rows = []

    for idx, source in enumerate(source_rows):
        output_row = outputs_by_idx.get(idx, {})
        output = output_row.get("output", "")
        target_name, target_signature = first_def_signature(source.get("answer", ""))
        extracted = extract_mbpp_candidate(output, target_name)
        request = {
            "prompt_for_model": source.get("question", ""),
            "function_name": target_name,
            "function_signature": target_signature,
            "helper_code": "",
            "unit_tests": source.get("tests") or [],
        }

        lengths.append(len(output or ""))
        extracted_lengths.append(len(extracted or ""))
        has_target = bool(target_name and re.search(rf"^\s*def\s+{re.escape(target_name)}\s*\(", output, re.MULTILINE))
        def_count = len(re.findall(r"^\s*def\s+", output or "", re.MULTILINE))

        flags = {
            "has_markdown_fence": "```" in output,
            "has_special_token": any(token in output for token in SPECIAL_TOKENS),
            "has_target_def": has_target,
            "has_any_def": def_count > 0,
            "multiple_defs": def_count > 1,
            "near_1024_char_budget": len(output or "") >= 3500,
        }
        for key, value in flags.items():
            if value:
                format_counts[key] += 1

        extracted_ok = None
        extracted_error = None
        evaluated_code = None
        if helper is not None:
            evaluated_code = helper.build_mbpp_evaluated_code(request, output)
            extracted_ok, extracted_error = helper.evaluate_mbpp_completion(request, output, args.timeout_seconds)
            eval_counts[extracted_error or "ok"] += 1

        detail = {
            "source_idx": idx,
            "task_id": source.get("task_id", idx),
            "target_name": target_name,
            "original_error": details_by_idx.get(idx, {}).get("error_type"),
            "original_correct": details_by_idx.get(idx, {}).get("is_correct"),
            "extracted_correct": extracted_ok,
            "extracted_error": extracted_error,
            "output_chars": len(output or ""),
            "extracted_chars": len(extracted or ""),
            "def_count": def_count,
            **flags,
        }
        if args.preview_chars > 0:
            detail.update(
                {
                    "question_preview": truncate_preview(source.get("question", ""), args.preview_chars),
                    "raw_output_preview": truncate_preview(output, args.preview_chars),
                    "extracted_code_preview": truncate_preview(extracted, args.preview_chars),
                    "evaluated_code_preview": truncate_preview(evaluated_code, args.preview_chars),
                    "tests_preview": truncate_preview("\n".join(source.get("tests") or []), args.preview_chars),
                }
            )
        detail_rows.append(detail)

    original_counts = Counter()
    for row in details_by_idx.values():
        original_counts[row.get("error_type") or "ok"] += 1

    report = {
        "benchmark": "MBPP",
        "total": len(source_rows),
        "result_rows": len(result_rows),
        "original_error_counts": dict(original_counts),
        "format_counts": dict(format_counts),
        "output_chars": summarize_lengths(lengths),
        "extracted_chars": summarize_lengths(extracted_lengths),
        "extracted_eval": {
            "enabled": bool(args.run_extracted_eval),
            "correct": eval_counts.get("ok", 0) if args.run_extracted_eval else None,
            "accuracy_pct": round(eval_counts.get("ok", 0) / len(source_rows) * 100, 2)
            if args.run_extracted_eval and source_rows
            else None,
            "error_counts": dict(eval_counts),
        },
        "inputs": {
            "source_jsonl": args.source_jsonl,
            "results_jsonl": args.results_jsonl,
            "details_jsonl": args.details_jsonl,
        },
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.examples_jsonl:
        write_jsonl(Path(args.examples_jsonl), detail_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
