#!/usr/bin/env python
"""Replay a Slime G1 EBFT actor-loss runtime dump against OpenRLHF semantics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slime.utils.g1_ebft_loss import (  # noqa: E402
    G1_EBFT_ACTOR_LOSS_DUMP_FORMAT,
    ebft_mean_rl_ce_over_packed_samples,
)


DEFAULT_OPENRLHF_ROOT = Path("/mnt/data/ebft-distribution-new/code/openrlhf")
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


@dataclass(frozen=True)
class InputSummary:
    name: str
    count: int
    shapes: str
    dtype: str
    sha256: str


@dataclass(frozen=True)
class ComponentDiff:
    name: str
    slime: float
    reference: float
    abs_diff: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class ReplayResult:
    dump_path: Path
    dump_format: str
    source: str
    openrlhf_commit: str
    openrlhf_import_note: str
    qa_masking: bool
    policy_loss_type: str
    ce_coef: float
    atol: float
    rtol: float
    input_summaries: tuple[InputSummary, ...]
    diffs: tuple[ComponentDiff, ...]
    passed: bool


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch.
        return torch.load(path, map_location="cpu")


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _openrlhf_models_dir(openrlhf_root: Path) -> Path | None:
    if (openrlhf_root / "models" / "loss.py").exists():
        return openrlhf_root / "models"
    if (openrlhf_root / "loss.py").exists():
        return openrlhf_root
    return None


def load_openrlhf_ebft_policy_loss(openrlhf_root: Path):
    models_dir = _openrlhf_models_dir(openrlhf_root)
    commit = _git_commit(openrlhf_root)
    if models_dir is None:
        return None, commit, f"OpenRLHF loss.py not found under {openrlhf_root}"

    pkg_name = "_openrlhf_ebft_actor_loss_runtime_ref"
    pkg = sys.modules.get(pkg_name)
    if pkg is None:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    try:
        _load_module(f"{pkg_name}.utils", models_dir / "utils.py")
        loss_module = _load_module(f"{pkg_name}.loss", models_dir / "loss.py")
    except Exception as exc:
        return None, commit, f"OpenRLHF loss.py import unavailable: {exc!r}"

    return loss_module.EBFTPolicyLoss, commit, f"EBFTPolicyLoss from {models_dir / 'loss.py'}"


def _tensor_collection_sha256(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for idx, tensor in enumerate(tensors):
        cpu = tensor.detach().cpu().contiguous()
        h.update(f"{idx}:{tuple(cpu.shape)}:{cpu.dtype}\n".encode())
        buf = io.BytesIO()
        torch.save(cpu, buf)
        h.update(buf.getvalue())
    return h.hexdigest()


def _shape_summary(tensors: list[torch.Tensor]) -> str:
    shapes = [tuple(t.shape) for t in tensors]
    unique = sorted(set(shapes))
    if len(unique) == 1:
        return f"{len(shapes)} x {unique[0]}"
    return ", ".join(str(shape) for shape in shapes)


def _input_summary(name: str, tensors: list[torch.Tensor]) -> InputSummary:
    if not tensors:
        raise ValueError(f"{name} is empty")
    dtypes = sorted({str(t.dtype) for t in tensors})
    return InputSummary(
        name=name,
        count=len(tensors),
        shapes=_shape_summary(tensors),
        dtype=", ".join(dtypes),
        sha256=_tensor_collection_sha256(tensors),
    )


def _as_tensor_list(value: Any, name: str) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError(f"{name} must not be scalar")
        return [row.detach().cpu() for row in value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a tensor list, got {type(value).__name__}")
    if not all(isinstance(t, torch.Tensor) for t in value):
        raise ValueError(f"{name} must contain only tensors")
    return [t.detach().cpu() for t in value]


def _load_dump(path: Path) -> dict[str, Any]:
    dump = _torch_load(path)
    if not isinstance(dump, dict):
        raise ValueError(f"Expected dict dump at {path}, got {type(dump).__name__}")
    if dump.get("format") != G1_EBFT_ACTOR_LOSS_DUMP_FORMAT:
        raise ValueError(
            f"Unsupported dump format {dump.get('format')!r}; expected {G1_EBFT_ACTOR_LOSS_DUMP_FORMAT!r}"
        )
    if not isinstance(dump.get("inputs"), dict) or not isinstance(dump.get("scalars"), dict):
        raise ValueError("Dump must contain dict keys `inputs` and `scalars`")
    return dump


def _can_stack(tensors: list[torch.Tensor]) -> bool:
    return bool(tensors) and len({tuple(t.shape) for t in tensors}) == 1


def _reference_with_fallback(
    *,
    log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    action_mask: list[torch.Tensor],
    qa_mask: list[torch.Tensor],
    qa_masking: bool,
    policy_loss_type: str,
    openrlhf_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, str, str, str]:
    EBFTPolicyLoss, commit, note = load_openrlhf_ebft_policy_loss(openrlhf_root)
    can_use_openrlhf = EBFTPolicyLoss is not None and all(
        _can_stack(tensors) for tensors in (log_probs, advantages, action_mask, qa_mask)
    )

    if can_use_openrlhf:
        mod = EBFTPolicyLoss(policy_loss_type=policy_loss_type)
        rl_ref, ce_ref = mod.forward(
            torch.stack(log_probs, dim=0),
            torch.stack(advantages, dim=0),
            action_mask=torch.stack(action_mask, dim=0).to(dtype=torch.bool),
            qa_masks=torch.stack(qa_mask, dim=0).to(dtype=torch.bool),
            qa_masking=qa_masking,
        )
        return rl_ref.detach().cpu(), ce_ref.detach().cpu(), "OpenRLHF EBFTPolicyLoss", commit, note

    rl_ref, ce_ref = ebft_mean_rl_ce_over_packed_samples(
        per_sample_log_probs_next=log_probs,
        per_sample_adv_next=advantages,
        per_sample_action_mask_next=[t.to(dtype=torch.bool) for t in action_mask],
        per_sample_qa_mask_next=[t.to(dtype=torch.bool) for t in qa_mask],
        qa_masking=qa_masking,
        policy_loss_type=policy_loss_type,
    )
    fallback_reason = note
    if EBFTPolicyLoss is not None:
        fallback_reason = "OpenRLHF available, but dump has non-stackable per-sample shapes; used same-semantics fallback"
    return rl_ref.detach().cpu(), ce_ref.detach().cpu(), "same-semantics fallback", commit, fallback_reason


def _diff(name: str, slime: float, reference: float, *, atol: float, rtol: float) -> ComponentDiff:
    abs_diff = abs(slime - reference)
    tolerance = float(atol + rtol * abs(reference))
    return ComponentDiff(
        name=name,
        slime=slime,
        reference=reference,
        abs_diff=abs_diff,
        tolerance=tolerance,
        passed=abs_diff <= tolerance,
    )


def compare_runtime_dump(
    dump_path: Path | str,
    *,
    openrlhf_root: Path | str = DEFAULT_OPENRLHF_ROOT,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> ReplayResult:
    dump_path = Path(dump_path)
    openrlhf_root = Path(openrlhf_root)
    dump = _load_dump(dump_path)

    inputs = dump["inputs"]
    scalars = dump["scalars"]
    metadata = dump.get("metadata") or {}

    log_probs = _as_tensor_list(inputs.get("log_probs_next"), "log_probs_next")
    advantages = _as_tensor_list(inputs.get("advantages_next"), "advantages_next")
    action_mask = _as_tensor_list(inputs.get("action_mask_next"), "action_mask_next")
    qa_mask = _as_tensor_list(inputs.get("qa_mask_next"), "qa_mask_next")

    if not (len(log_probs) == len(advantages) == len(action_mask) == len(qa_mask)):
        raise ValueError("Input tensor list lengths do not match")

    qa_masking = bool(metadata.get("qa_masking", True))
    policy_loss_type = str(metadata.get("policy_loss_type", "ppo"))
    ce_coef = float(scalars["g1_ce_loss_coef"])

    rl_ref, ce_ref, source, commit, note = _reference_with_fallback(
        log_probs=log_probs,
        advantages=advantages,
        action_mask=action_mask,
        qa_mask=qa_mask,
        qa_masking=qa_masking,
        policy_loss_type=policy_loss_type,
        openrlhf_root=openrlhf_root,
    )
    total_ref = rl_ref + ce_coef * ce_ref

    diffs = (
        _diff("RL", float(scalars["slime_rl_loss"]), float(rl_ref.item()), atol=atol, rtol=rtol),
        _diff("CE", float(scalars["slime_ce_loss"]), float(ce_ref.item()), atol=atol, rtol=rtol),
        _diff("total", float(scalars["slime_total_loss"]), float(total_ref.item()), atol=atol, rtol=rtol),
    )

    return ReplayResult(
        dump_path=dump_path,
        dump_format=str(dump["format"]),
        source=source,
        openrlhf_commit=commit,
        openrlhf_import_note=note,
        qa_masking=qa_masking,
        policy_loss_type=policy_loss_type,
        ce_coef=ce_coef,
        atol=atol,
        rtol=rtol,
        input_summaries=(
            _input_summary("log_probs_next", log_probs),
            _input_summary("advantages_next", advantages),
            _input_summary("action_mask_next", action_mask),
            _input_summary("qa_mask_next", qa_mask),
        ),
        diffs=diffs,
        passed=all(d.passed for d in diffs),
    )


def render_markdown_report(result: ReplayResult) -> str:
    lines = [
        "# G1 EBFT Actor-Loss Runtime Parity",
        "",
        f"- Dump: `{result.dump_path}`",
        f"- Dump format: `{result.dump_format}`",
        f"- Reference source: `{result.source}`",
        f"- OpenRLHF commit: `{result.openrlhf_commit}`",
        f"- OpenRLHF import note: `{result.openrlhf_import_note}`",
        f"- Policy loss type: `{result.policy_loss_type}`",
        f"- QA masking: `{result.qa_masking}`",
        f"- CE coefficient: `{result.ce_coef:.12g}`",
        f"- Tolerance: `atol={result.atol:.3g}`, `rtol={result.rtol:.3g}`",
        f"- Overall: `{'PASS' if result.passed else 'FAIL'}`",
        "",
        "## Input Shape And Hash",
        "",
        "| Field | Count | Shapes | DType | SHA256 |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for summary in result.input_summaries:
        lines.append(
            f"| `{summary.name}` | {summary.count} | `{summary.shapes}` | `{summary.dtype}` | `{summary.sha256}` |"
        )

    lines.extend(
        [
            "",
            "## Loss Diff",
            "",
            "| Component | Slime | Reference | Abs diff | Tolerance | Pass |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for diff in result.diffs:
        lines.append(
            f"| `{diff.name}` | {diff.slime:.12g} | {diff.reference:.12g} | "
            f"{diff.abs_diff:.12g} | {diff.tolerance:.12g} | `{diff.passed}` |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_path", type=Path, help="Path written by G1_EBFT_ACTOR_LOSS_DUMP_PATH")
    parser.add_argument("--output", "-o", type=Path, help="Optional Markdown report output path")
    parser.add_argument("--openrlhf-root", type=Path, default=DEFAULT_OPENRLHF_ROOT)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--fail-on-diff", action="store_true", help="Exit non-zero when any component exceeds tolerance")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = compare_runtime_dump(
        args.dump_path,
        openrlhf_root=args.openrlhf_root,
        atol=args.atol,
        rtol=args.rtol,
    )
    report = render_markdown_report(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 1 if args.fail_on_diff and not result.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
