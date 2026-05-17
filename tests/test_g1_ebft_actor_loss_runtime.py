"""CPU tests for Slime G1 EBFT actor-loss runtime dump replay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from slime.utils.g1_ebft_loss import dump_ebft_actor_loss_runtime, ebft_mean_rl_ce_over_packed_samples

_REPO = Path(__file__).resolve().parents[1]
_COMPARE_PATH = _REPO / "refactor_debugging/g1_plan/compare_g1_ebft_actor_loss_runtime.py"


def _load_compare_module():
    name = "compare_g1_ebft_actor_loss_runtime"
    spec = importlib.util.spec_from_file_location(name, _COMPARE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_small_actor_loss_dump(path: Path, *, total_delta: float = 0.0) -> Path:
    log_probs = [
        torch.tensor([-0.3, -1.2, -0.4, -1.7], dtype=torch.float32),
        torch.tensor([-0.8, -0.6, -1.5, -0.2], dtype=torch.float32),
    ]
    advantages = [
        torch.tensor([0.0, 1.25, -0.75, 0.5], dtype=torch.float32),
        torch.tensor([0.0, -0.5, 1.5, -1.0], dtype=torch.float32),
    ]
    action_mask = [
        torch.tensor([False, True, True, True]),
        torch.tensor([False, False, True, True]),
    ]
    qa_mask = [
        torch.tensor([True, True, False, True]),
        torch.tensor([False, True, True, True]),
    ]
    ce_coef = 0.03
    rl_loss, ce_loss = ebft_mean_rl_ce_over_packed_samples(
        per_sample_log_probs_next=log_probs,
        per_sample_adv_next=advantages,
        per_sample_action_mask_next=action_mask,
        per_sample_qa_mask_next=qa_mask,
        qa_masking=True,
        policy_loss_type="ppo",
    )
    total_loss = rl_loss + ce_coef * ce_loss + torch.tensor(total_delta, dtype=rl_loss.dtype)

    return dump_ebft_actor_loss_runtime(
        path=path,
        per_sample_log_probs_next=log_probs,
        per_sample_adv_next=advantages,
        per_sample_action_mask_next=action_mask,
        per_sample_qa_mask_next=qa_mask,
        qa_masking=True,
        g1_ce_loss_coef=ce_coef,
        slime_rl_loss=rl_loss,
        slime_ce_loss=ce_loss,
        slime_total_loss=total_loss,
        policy_loss_type="ppo",
        metadata={"runtime_kind": "unit_test_fixture"},
    )


def test_actor_loss_runtime_report_replays_small_dump_with_fallback(tmp_path: Path) -> None:
    compare = _load_compare_module()
    dump_path = _write_small_actor_loss_dump(tmp_path / "small_actor_loss_dump.pt")
    report_path = tmp_path / "actor_loss_report.md"

    exit_code = compare.main(
        [
            str(dump_path),
            "--openrlhf-root",
            str(tmp_path / "missing-openrlhf"),
            "--output",
            str(report_path),
            "--fail-on-diff",
        ]
    )

    assert exit_code == 0
    result = compare.compare_runtime_dump(dump_path, openrlhf_root=tmp_path / "missing-openrlhf")
    assert result.passed
    assert result.source == "same-semantics fallback"
    assert all(len(summary.sha256) == 64 for summary in result.input_summaries)

    report = report_path.read_text(encoding="utf-8")
    assert "# G1 EBFT Actor-Loss Runtime Parity" in report
    assert "Input Shape And Hash" in report
    assert "same-semantics fallback" in report
    assert "`PASS`" in report


def test_actor_loss_runtime_diff_logic_fails_on_total_mismatch(tmp_path: Path) -> None:
    compare = _load_compare_module()
    dump_path = _write_small_actor_loss_dump(tmp_path / "bad_total_actor_loss_dump.pt", total_delta=1e-3)

    result = compare.compare_runtime_dump(
        dump_path,
        openrlhf_root=tmp_path / "missing-openrlhf",
        atol=1e-9,
        rtol=0.0,
    )
    diffs = {diff.name: diff for diff in result.diffs}

    assert diffs["RL"].passed
    assert diffs["CE"].passed
    assert not diffs["total"].passed
    assert not result.passed
    assert (
        compare.main(
            [
                str(dump_path),
                "--openrlhf-root",
                str(tmp_path / "missing-openrlhf"),
                "--atol",
                "1e-9",
                "--rtol",
                "0",
                "--fail-on-diff",
            ]
        )
        == 1
    )
