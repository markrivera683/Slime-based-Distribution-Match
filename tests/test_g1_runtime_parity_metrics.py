"""Unit tests for refactor_debugging/g1_plan/compare_g1_runtime_parity.py helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_COMPARE_PATH = _REPO / "refactor_debugging/g1_plan/compare_g1_runtime_parity.py"


def _load_compare_module():
    name = "compare_g1_runtime_parity"
    spec = importlib.util.spec_from_file_location(name, _COMPARE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def c():
    return _load_compare_module()


def test_relative_l2_identity(c):
    t = torch.randn(2, 3, 4)
    assert c._relative_l2(t, t) == pytest.approx(0.0, abs=1e-6)


def test_sample_megatron_hidden_compacts_by_total_lengths(c):
    # THD S=4, two samples of equal real length 2 (stack requires uniform seq in batch).
    h = torch.arange(4 * 1 * 2, dtype=torch.float32).reshape(4, 1, 2)
    dump = {
        "hidden_states_post_sp_gather": h,
        "total_lengths": [2, 2],
    }
    out = c._sample_megatron_hidden(dump)
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0], h[:2, 0])
    assert torch.equal(out[1], h[2:4, 0])


def test_sample_megatron_hidden_raises_on_length_sum_mismatch(c):
    dump = {
        "hidden_states_post_sp_gather": torch.zeros(4, 1, 8),
        "total_lengths": [1, 1],  # sum 2 != 4
    }
    with pytest.raises(ValueError, match="total_lengths"):
        c._sample_megatron_hidden(dump)


def test_early_validate_tokens_count_mismatch(c):
    meg = {"tokens": [torch.tensor([1, 2]), torch.tensor([3, 4])], "g1_qa_masks": [torch.zeros(2), torch.zeros(2)]}
    ohf = {"sequences": torch.zeros(1, 2).long(), "qa_masks": torch.zeros(1, 2).long()}
    with pytest.raises(ValueError, match="tokens sample count mismatch"):
        c._early_validate_tokens_and_masks(meg, ohf)


def test_check_embedding_family_pass(c):
    a = torch.randn(4, 5, 8)
    b = a.clone()
    cos = c._cosine_stats(a, b)
    close = c._close_stats(a, b)
    rl = c._relative_l2(a, b)
    thr = c.EmbeddingThresholds()
    ok, fails = c._check_embedding_family("t", cos, close, rl, thr)
    assert ok and fails == []


def test_check_embedding_family_cosine_fail(c):
    a = torch.ones(2, 3, 4)
    b = torch.zeros(2, 3, 4)
    cos = c._cosine_stats(a, b)  # min cosine 0
    close = c._close_stats(a, b)
    rl = c._relative_l2(a, b)
    thr = c.EmbeddingThresholds(cosine_min=0.998)
    ok, fails = c._check_embedding_family("t", cos, close, rl, thr)
    assert not ok
    assert any("cosine_min" in f for f in fails)


def test_cli_mismatch_exit_code(tmp_path):
    """Shape mismatch should fail before metrics and exit non-zero with strict exit."""
    d1 = tmp_path / "m.pt"
    d2 = tmp_path / "o.pt"
    torch.save(
        {
            "tokens": [torch.tensor([1, 2])],
            "g1_qa_masks": [torch.zeros(2, dtype=torch.long)],
            "hidden_states_post_sp_gather": torch.zeros(2, 1, 4),
            "total_lengths": [2],
            "g1_gen_embedding": [torch.zeros(1, 4)],
            "g1_gt_embedding": [torch.zeros(1, 4)],
        },
        d1,
    )
    torch.save(
        {
            "sequences": torch.tensor([[1, 2]]).long(),
            "qa_masks": torch.zeros(1, 2, dtype=torch.long),
            # OpenRLHF [B,S,...] with S=3 vs Megatron compact S=2 -> shape mismatch before metrics.
            "hidden_states": torch.zeros(1, 3, 1, 4),
            "g1_gen_embedding": [torch.zeros(1, 4)],
            "g1_gt_embedding": [torch.zeros(1, 4)],
        },
        d2,
    )
    out = tmp_path / "r.md"
    proc = subprocess.run(
        [sys.executable, str(_COMPARE_PATH), "--megatron-dump", str(d1), "--openrlhf-dump", str(d2), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    joined = (proc.stderr + proc.stdout).lower()
    assert "shape mismatch" in joined or "hidden states" in joined
