#!/usr/bin/env python3
"""Extract and validate G1 Megatron/Ray smoke train metrics from logs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any


METRICS = (
    "loss",
    "pg_loss",
    "g1_ebft_ce_loss",
    "ppo_kl",
    "entropy_loss",
    "pg_clipfrac",
)
ZERO_PLACEHOLDERS = ("ppo_kl", "entropy_loss", "pg_clipfrac")
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SCALAR_RE = re.compile(
    rf"(?:^|[\s'\"{{,])(?P<key>(?:train/)?(?:{'|'.join(re.escape(m) for m in METRICS)}))"
    rf"['\"]?\s*[:=]\s*(?P<value>{NUMBER_RE})"
)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _metric_name(key: str) -> str | None:
    key = key.strip("'\"")
    if key in METRICS:
        return key
    if key.startswith("train/") and key.removeprefix("train/") in METRICS:
        return key.removeprefix("train/")
    return None


def _extract_literal_dict(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = ast.literal_eval(line[start : end + 1])
    except (SyntaxError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _record_from_mapping(path: Path, line_no: int, raw_line: str, mapping: dict[str, Any]) -> dict[str, Any] | None:
    metrics: dict[str, float] = {}
    for key, value in mapping.items():
        metric = _metric_name(str(key))
        if metric is None:
            continue
        numeric = _coerce_float(value)
        if numeric is not None:
            metrics[metric] = numeric

    if not metrics:
        return None

    step = mapping.get("train/step", mapping.get("step"))
    return {
        "source": str(path),
        "line": line_no,
        "raw_line": raw_line.rstrip("\n"),
        "step": _coerce_float(step),
        "metrics": metrics,
    }


def extract_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                mapping = _extract_literal_dict(line)
                if mapping is not None:
                    record = _record_from_mapping(path, line_no, line, mapping)
                    if record is not None:
                        records.append(record)
                        continue

                metrics: dict[str, float] = {}
                for match in SCALAR_RE.finditer(line):
                    metric = _metric_name(match.group("key"))
                    if metric is not None:
                        metrics[metric] = float(match.group("value"))
                if metrics:
                    records.append(
                        {
                            "source": str(path),
                            "line": line_no,
                            "raw_line": line.rstrip("\n"),
                            "step": None,
                            "metrics": metrics,
                        }
                    )
    return records


def summarize(
    records: list[dict[str, Any]],
    *,
    ce_coef: float,
    formula_abs_tol: float,
    formula_rel_tol: float,
    zero_tol: float,
) -> dict[str, Any]:
    latest: dict[str, float] = {}
    latest_source: dict[str, dict[str, Any]] = {}
    for record in records:
        for metric, value in record["metrics"].items():
            latest[metric] = value
            latest_source[metric] = {
                "source": record["source"],
                "line": record["line"],
                "step": record["step"],
            }

    missing = [metric for metric in METRICS if metric not in latest]
    checks: dict[str, Any] = {
        "required_metrics_present": not missing,
        "missing_metrics": missing,
    }

    if all(metric in latest for metric in ("loss", "pg_loss", "g1_ebft_ce_loss")):
        expected_loss = latest["pg_loss"] + ce_coef * latest["g1_ebft_ce_loss"]
        abs_error = abs(latest["loss"] - expected_loss)
        rel_base = max(abs(latest["loss"]), abs(expected_loss), 1.0)
        checks["loss_formula"] = {
            "ok": abs_error <= max(formula_abs_tol, formula_rel_tol * rel_base),
            "ce_coef": ce_coef,
            "expected_loss": expected_loss,
            "abs_error": abs_error,
            "abs_tol": formula_abs_tol,
            "rel_tol": formula_rel_tol,
        }

    zero_checks: dict[str, Any] = {}
    for metric in ZERO_PLACEHOLDERS:
        if metric in latest:
            zero_checks[metric] = {
                "ok": abs(latest[metric]) <= zero_tol,
                "value": latest[metric],
                "zero_tol": zero_tol,
            }
    checks["phase1_zero_placeholders"] = zero_checks

    return {
        "records": records,
        "latest": latest,
        "latest_source": latest_source,
        "checks": checks,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    latest = summary["latest"]
    checks = summary["checks"]
    formula = checks.get("loss_formula")
    lines = [
        "# G1 Megatron/Ref Smoke Metric Extraction",
        "",
        "## Latest Metrics",
        "",
    ]
    for metric in METRICS:
        value = latest.get(metric)
        lines.append(f"- `{metric}`: {'MISSING' if value is None else value}")

    lines.extend(["", "## Checks", ""])
    lines.append(f"- required metrics present: `{checks['required_metrics_present']}`")
    if checks["missing_metrics"]:
        lines.append(f"- missing metrics: `{', '.join(checks['missing_metrics'])}`")
    if formula:
        lines.append(
            "- loss formula: "
            f"`loss ~= pg_loss + {formula['ce_coef']} * g1_ebft_ce_loss` -> `{formula['ok']}` "
            f"(expected={formula['expected_loss']}, abs_error={formula['abs_error']})"
        )
    else:
        lines.append("- loss formula: `not evaluated`")

    for metric, check in checks["phase1_zero_placeholders"].items():
        lines.append(f"- Phase 1 `{metric}` zero placeholder: `{check['ok']}` (value={check['value']})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_lines(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["source\tline\tmetric_line"]
    for record in records:
        raw_line = str(record["raw_line"]).replace("\t", "\\t")
        lines.append(f"{record['source']}\t{record['line']}\t{raw_line}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Ray job driver logs or copied Ray log files.")
    parser.add_argument("--ce-coef", type=float, default=0.03)
    parser.add_argument("--formula-abs-tol", type=float, default=1e-6)
    parser.add_argument("--formula-rel-tol", type=float, default=1e-6)
    parser.add_argument("--zero-tol", type=float, default=1e-8)
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-raw-lines", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing_logs = [str(path) for path in args.logs if not path.is_file()]
    if missing_logs:
        raise SystemExit(f"missing log file(s): {', '.join(missing_logs)}")

    summary = summarize(
        extract_records(args.logs),
        ce_coef=args.ce_coef,
        formula_abs_tol=args.formula_abs_tol,
        formula_rel_tol=args.formula_rel_tol,
        zero_tol=args.zero_tol,
    )

    if args.output_json:
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, summary)
    if args.output_raw_lines:
        write_raw_lines(args.output_raw_lines, summary["records"])

    print(json.dumps(summary["latest"], sort_keys=True))

    checks = summary["checks"]
    formula_ok = checks.get("loss_formula", {}).get("ok", False)
    zero_ok = all(check["ok"] for check in checks["phase1_zero_placeholders"].values())
    if args.require_metrics and (not checks["required_metrics_present"] or not formula_ok or not zero_ok):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
