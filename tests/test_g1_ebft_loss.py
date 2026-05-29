"""CPU tests for OpenRLHF-aligned EBFT policy loss primitives."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from slime.utils.g1_ebft_loss import (
    ebft_build_next_token_action_qa_advantages,
    ebft_compute_rl_ce_scalars_after_masks,
    ebft_mean_rl_ce_over_packed_samples,
    openrlhf_masked_mean,
)


OPENRLHF_EBFT_LOSS_ATOL = 1e-7
OPENRLHF_EBFT_LOSS_RTOL = 1e-6
OPENRLHF_EBFT_FIXTURE_SHAPE = (2, 6)
OPENRLHF_EBFT_KL_COEF = 0.25


def _load_openrlhf_ebft_policy_loss_from_checkout():
    """Load only OpenRLHF loss.py so optional package deps do not hide loss parity."""

    rlhf_models = Path("/mnt/data/ebft-distribution-new/code/openrlhf/models")
    if not rlhf_models.exists():
        return None, "OpenRLHF reference checkout not mounted at /mnt/data/ebft-distribution-new/code/openrlhf"

    pkg_name = "_openrlhf_ebft_ref"
    pkg = sys.modules.get(pkg_name)
    if pkg is None:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(rlhf_models)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    def load_module(module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {module_name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    try:
        load_module(f"{pkg_name}.utils", rlhf_models / "utils.py")
        loss_module = load_module(f"{pkg_name}.loss", rlhf_models / "loss.py")
    except Exception as exc:  # pragma: no cover - depends on optional local checkout
        return None, f"OpenRLHF loss.py import unavailable: {exc!r}"

    return loss_module.EBFTPolicyLoss, f"source-file import from {rlhf_models / 'loss.py'}"


def _openrlhf_ebft_actor_loss_fixture() -> dict[str, torch.Tensor | dict[bool, tuple[torch.Tensor, torch.Tensor]]]:
    log_probs = torch.tensor(
        [
            [-0.2, -1.3, -0.7, -0.4, -2.0, -0.1],
            [-1.1, -0.5, -0.9, -1.7, -0.3, -2.2],
        ],
        dtype=torch.float32,
    )
    advantages = torch.tensor(
        [
            [0.0, 0.0, 1.5, -0.25, 0.75, -1.0],
            [0.0, 0.2, -0.5, 1.25, -1.5, 0.5],
        ],
        dtype=torch.float32,
    )
    # False = prompt/CE token, True = generated/RL token in OpenRLHF's next-token layout.
    action_mask = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, False, False, True, True, True],
        ],
        dtype=torch.bool,
    )
    # QA mask is already shifted like OpenRLHF ebft_actor.py passes qa_masks[:, 1:] to EBFTPolicyLoss.
    qa_masks = torch.tensor(
        [
            [True, False, True, True, False, True],
            [False, True, True, True, False, True],
        ],
        dtype=torch.bool,
    )
    golden = {
        False: (torch.tensor(-1.0 / 6.0, dtype=torch.float32), torch.tensor(19.0 / 24.0, dtype=torch.float32)),
        True: (torch.tensor(-23.0 / 48.0, dtype=torch.float32), torch.tensor(9.0 / 20.0, dtype=torch.float32)),
    }
    return {
        "log_probs": log_probs,
        "advantages": advantages,
        "action_mask": action_mask,
        "qa_masks": qa_masks,
        "golden": golden,
    }


def _openrlhf_ebft_kl_fixture() -> dict[str, torch.Tensor]:
    action_log_probs = torch.tensor(
        [
            [-0.2, -12.0, -0.8, 0.0],
            [2.0, -0.5, -11.5, -0.1],
        ],
        dtype=torch.float32,
    )
    base_action_log_probs = torch.tensor(
        [
            [-0.7, 3.0, -2.8, 0.5],
            [1.0, 0.0, 1.0, -20.5],
        ],
        dtype=torch.float32,
    )
    # True = generated token. Counts differ by row to lock in OpenRLHF's global reduction.
    action_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, False, True, False],
        ],
        dtype=torch.bool,
    )
    return {
        "action_log_probs": action_log_probs,
        "base_action_log_probs": base_action_log_probs,
        "action_mask": action_mask,
    }


def _manual_openrlhf_compute_approx_kl(
    action_log_probs: torch.Tensor,
    base_action_log_probs: torch.Tensor,
    *,
    kl_estimator: str,
) -> torch.Tensor:
    log_ratio = action_log_probs.float() - base_action_log_probs.float()
    if kl_estimator == "k1":
        kl = log_ratio
    elif kl_estimator == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_estimator == "k3":
        kl = (-log_ratio).exp() - 1 + log_ratio
    else:
        raise ValueError(f"Unknown OpenRLHF KL estimator: {kl_estimator}")
    return kl.clamp(min=-10, max=10)


def _manual_openrlhf_ebft_kl_scalar(
    *,
    action_log_probs: torch.Tensor,
    base_action_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    kl_coef: float,
    kl_estimator: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kl_coef == 0:
        kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
    else:
        kl = _manual_openrlhf_compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            kl_estimator=kl_estimator,
        )
    kl_scalar = openrlhf_masked_mean(kl, action_mask.to(kl.dtype))
    return kl_scalar, kl_scalar * kl_coef


def _manual_openrlhf_ebft_fixture_loss(
    *,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    qa_masks: torch.Tensor,
    qa_masking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    qa_eff = qa_masks if qa_masking else torch.ones_like(qa_masks, dtype=torch.bool)
    rl_mask = action_mask & qa_eff
    ce_mask = (~action_mask) & qa_eff
    ratio = (log_probs - log_probs.detach()).exp()
    rl_loss = openrlhf_masked_mean(-ratio * advantages, rl_mask.to(log_probs.dtype), dim=-1).mean()
    ce_loss = openrlhf_masked_mean(-log_probs, ce_mask.to(log_probs.dtype), dim=-1).mean()
    return rl_loss, ce_loss


def _slime_packed_fixture_loss(
    *,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    qa_masks: torch.Tensor,
    qa_masking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ebft_mean_rl_ce_over_packed_samples(
        per_sample_log_probs_next=list(log_probs.unbind(0)),
        per_sample_adv_next=list(advantages.unbind(0)),
        per_sample_action_mask_next=list(action_mask.unbind(0)),
        per_sample_qa_mask_next=list(qa_masks.unbind(0)),
        qa_masking=qa_masking,
        policy_loss_type="ppo",
    )


def test_openrlhf_masked_mean_zero_row_falls_back_to_unmasked_mean() -> None:
    t = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    m = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    row = openrlhf_masked_mean(t, m, dim=-1)
    assert torch.isclose(row[0], torch.tensor((1.0 + 3.0) / 2))
    assert torch.isclose(row[1], torch.tensor((10.0 + 20.0 + 30.0) / 3))


def _hand_ebft_ppo_rl_ce(
    log_probs_next: torch.Tensor,
    advantages_next: torch.Tensor,
    action_mask_next: torch.Tensor,
    qa_mask_next: torch.Tensor,
    *,
    qa_masking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from slime.utils.g1_ebft_loss import openrlhf_masked_mean as mm

    lp = log_probs_next.unsqueeze(0)
    adv = advantages_next.unsqueeze(0)
    am = action_mask_next.unsqueeze(0)
    qa = qa_mask_next.unsqueeze(0)

    qa_eff = qa if qa_masking else torch.ones_like(qa, dtype=torch.bool)

    rl_mask_bool = am.to(dtype=torch.bool) & qa_eff
    ce_mask_bool = (~am.to(dtype=torch.bool)) & qa_eff

    log_ratio = lp - lp.detach()
    ratio = log_ratio.exp()
    surr = -ratio * adv

    rl = mm(surr, rl_mask_bool.to(lp.dtype), dim=-1).mean()
    ce = mm(-lp, ce_mask_bool.to(lp.dtype), dim=-1).mean()
    return rl, ce


def test_parity_ratio_one_no_clip_prompt_ce_response_rl() -> None:
    torch.manual_seed(0)

    lp = torch.randn(10, dtype=torch.float32)

    qa_raw = torch.ones(11, dtype=torch.long)
    qa_raw[:3] = 0
    pl, rl = 4, 7
    L = pl + rl
    assert L == qa_raw.numel()
    seq = torch.zeros(L, dtype=torch.long)
    resp_adv = torch.randn(rl, dtype=torch.float32)

    _, qa_next, adv_next = ebft_build_next_token_action_qa_advantages(
        full_sequence_1d=seq,
        response_advantages_1d=resp_adv,
        qa_mask_full_1d=qa_raw,
        prompt_length=pl,
        response_length=rl,
    )
    lm1 = L - 1
    prompt_len = pl
    am = torch.zeros(lm1, dtype=torch.bool)
    am[prompt_len - 1 : prompt_len - 1 + rl] = True

    for qa_masking in (False, True):
        rl_s, ce_s = ebft_compute_rl_ce_scalars_after_masks(
            log_probs_next=lp.unsqueeze(0),
            advantages_next=adv_next.unsqueeze(0),
            action_mask_next=am.unsqueeze(0),
            qa_mask_next=qa_next.unsqueeze(0),
            qa_masking=qa_masking,
            policy_loss_type="ppo",
        )

        rl_h, ce_h = _hand_ebft_ppo_rl_ce(lp, adv_next, am, qa_next, qa_masking=qa_masking)

        torch.testing.assert_close(rl_s, rl_h)
        torch.testing.assert_close(ce_s, ce_h)


def test_mean_over_packed_samples_avg_rows() -> None:
    qa_masking = True
    lp_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    lp_b = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)

    adv_a = torch.tensor([9.0, 1.0, 0.0], dtype=torch.float32)
    adv_b = torch.tensor([-1.0, 2.0, 7.0], dtype=torch.float32)

    action_a = torch.tensor([False, True, False], dtype=torch.bool)
    action_b = torch.tensor([True, False, True], dtype=torch.bool)

    qa_a = torch.tensor([True, True, False], dtype=torch.bool)
    qa_b = torch.ones(3, dtype=torch.bool)

    r_packed, c_packed = ebft_mean_rl_ce_over_packed_samples(
        per_sample_log_probs_next=[lp_a, lp_b],
        per_sample_adv_next=[adv_a, adv_b],
        per_sample_action_mask_next=[action_a, action_b],
        per_sample_qa_mask_next=[qa_a, qa_b],
        qa_masking=qa_masking,
        policy_loss_type="ppo",
    )

    rows_rl = []
    rows_ce = []
    for tensors in [(lp_a, adv_a, action_a, qa_a), (lp_b, adv_b, action_b, qa_b)]:
        lp_i, adv_i, am_i, qa_i = tensors
        r_i, c_i = ebft_compute_rl_ce_scalars_after_masks(
            log_probs_next=lp_i.unsqueeze(0),
            advantages_next=adv_i.unsqueeze(0),
            action_mask_next=am_i.unsqueeze(0),
            qa_mask_next=qa_i.unsqueeze(0),
            qa_masking=qa_masking,
            policy_loss_type="ppo",
        )
        rows_rl.append(r_i)
        rows_ce.append(c_i)

    torch.testing.assert_close(r_packed, (rows_rl[0] + rows_rl[1]) / 2)
    torch.testing.assert_close(c_packed, (rows_ce[0] + rows_ce[1]) / 2)


@pytest.mark.parametrize(
    ("qa_masking", "expected_rl_counts", "expected_ce_counts"),
    [
        (False, [4, 3], [2, 3]),
        (True, [3, 2], [1, 2]),
    ],
)
def test_slime_packed_samples_match_openrlhf_ebft_golden_fixture(
    qa_masking: bool,
    expected_rl_counts: list[int],
    expected_ce_counts: list[int],
) -> None:
    """Deterministic fallback for OpenRLHF EBFT actor-loss scalar parity."""

    fixture = _openrlhf_ebft_actor_loss_fixture()
    log_probs = fixture["log_probs"]
    advantages = fixture["advantages"]
    action_mask = fixture["action_mask"]
    qa_masks = fixture["qa_masks"]
    golden = fixture["golden"]
    assert isinstance(log_probs, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert isinstance(action_mask, torch.Tensor)
    assert isinstance(qa_masks, torch.Tensor)
    assert isinstance(golden, dict)

    assert tuple(log_probs.shape) == OPENRLHF_EBFT_FIXTURE_SHAPE
    assert tuple(advantages.shape) == OPENRLHF_EBFT_FIXTURE_SHAPE
    assert tuple(action_mask.shape) == OPENRLHF_EBFT_FIXTURE_SHAPE
    assert tuple(qa_masks.shape) == OPENRLHF_EBFT_FIXTURE_SHAPE

    qa_eff = qa_masks if qa_masking else torch.ones_like(qa_masks, dtype=torch.bool)
    assert (action_mask & qa_eff).sum(dim=-1).tolist() == expected_rl_counts
    assert ((~action_mask) & qa_eff).sum(dim=-1).tolist() == expected_ce_counts

    rl_slime, ce_slime = _slime_packed_fixture_loss(
        log_probs=log_probs,
        advantages=advantages,
        action_mask=action_mask,
        qa_masks=qa_masks,
        qa_masking=qa_masking,
    )
    rl_manual, ce_manual = _manual_openrlhf_ebft_fixture_loss(
        log_probs=log_probs,
        advantages=advantages,
        action_mask=action_mask,
        qa_masks=qa_masks,
        qa_masking=qa_masking,
    )
    rl_golden, ce_golden = golden[qa_masking]

    torch.testing.assert_close(rl_slime, rl_manual, atol=OPENRLHF_EBFT_LOSS_ATOL, rtol=OPENRLHF_EBFT_LOSS_RTOL)
    torch.testing.assert_close(ce_slime, ce_manual, atol=OPENRLHF_EBFT_LOSS_ATOL, rtol=OPENRLHF_EBFT_LOSS_RTOL)
    torch.testing.assert_close(rl_slime, rl_golden, atol=OPENRLHF_EBFT_LOSS_ATOL, rtol=OPENRLHF_EBFT_LOSS_RTOL)
    torch.testing.assert_close(ce_slime, ce_golden, atol=OPENRLHF_EBFT_LOSS_ATOL, rtol=OPENRLHF_EBFT_LOSS_RTOL)


@pytest.mark.parametrize("kl_estimator", ["k1", "k2", "k3"])
def test_openrlhf_ebft_kl_scalar_fixture_matches_mask_reduction_and_coef(kl_estimator: str) -> None:
    """Document OpenRLHF EBFT KL semantics without enabling Slime's training KL path."""

    fixture = _openrlhf_ebft_kl_fixture()
    action_log_probs = fixture["action_log_probs"]
    base_action_log_probs = fixture["base_action_log_probs"]
    action_mask = fixture["action_mask"]

    kl_tokens = _manual_openrlhf_compute_approx_kl(
        action_log_probs,
        base_action_log_probs,
        kl_estimator=kl_estimator,
    )
    expected_scalar = (kl_tokens * action_mask.to(kl_tokens.dtype)).sum() / action_mask.sum()
    per_row_scalar = openrlhf_masked_mean(kl_tokens, action_mask.to(kl_tokens.dtype), dim=-1).mean()

    kl_scalar, kl_contribution = _manual_openrlhf_ebft_kl_scalar(
        action_log_probs=action_log_probs,
        base_action_log_probs=base_action_log_probs,
        action_mask=action_mask,
        kl_coef=OPENRLHF_EBFT_KL_COEF,
        kl_estimator=kl_estimator,
    )

    torch.testing.assert_close(kl_scalar, expected_scalar, atol=OPENRLHF_EBFT_LOSS_ATOL, rtol=OPENRLHF_EBFT_LOSS_RTOL)
    torch.testing.assert_close(
        kl_contribution,
        expected_scalar * OPENRLHF_EBFT_KL_COEF,
        atol=OPENRLHF_EBFT_LOSS_ATOL,
        rtol=OPENRLHF_EBFT_LOSS_RTOL,
    )
    assert not torch.isclose(kl_scalar, per_row_scalar)


def test_openrlhf_ebft_kl_zero_coef_short_circuits_base_log_probs() -> None:
    """OpenRLHF skips approximate KL when init_kl_coef/kl_coef is zero."""

    fixture = _openrlhf_ebft_kl_fixture()
    action_log_probs = fixture["action_log_probs"]
    action_mask = fixture["action_mask"]
    nan_base_action_log_probs = torch.full_like(action_log_probs, float("nan"))

    kl_scalar, kl_contribution = _manual_openrlhf_ebft_kl_scalar(
        action_log_probs=action_log_probs,
        base_action_log_probs=nan_base_action_log_probs,
        action_mask=action_mask,
        kl_coef=0.0,
        kl_estimator="k3",
    )

    torch.testing.assert_close(kl_scalar, torch.tensor(0.0))
    torch.testing.assert_close(kl_contribution, torch.tensor(0.0))


def test_thd_full_sequence_logprob_gather_returns_1d_samples(monkeypatch) -> None:
    megatron_loss = pytest.importorskip(
        "slime.backends.megatron_utils.loss",
        reason="Megatron dependency is not available in this environment.",
    )
    monkeypatch.setattr(megatron_loss.mpu, "get_tensor_model_parallel_group", lambda: None)

    def fake_calc(logits_chunk, tokens_targets, tp_group, with_entropy=False, chunk_size=None):
        del logits_chunk, tp_group, with_entropy, chunk_size
        return torch.arange(tokens_targets.numel(), dtype=torch.float32).view(-1, 1), None

    monkeypatch.setattr(megatron_loss, "calculate_log_probs_and_entropy", fake_calc)

    args = type("Args", (), {"qkv_format": "thd", "rollout_temperature": 1.0, "log_probs_chunk_size": 0})()
    outs = megatron_loss.thd_packed_full_sequence_next_log_probs_from_logits(
        torch.zeros(1, 6, 4, dtype=torch.float32),
        args=args,
        unconcat_tokens=[torch.tensor([1, 2, 3, 4])],
        total_lengths=[4],
    )

    assert outs[0].shape == (3,)


def test_thd_pair_axis_logprob_gather_uses_strict_source_rows(monkeypatch) -> None:
    megatron_loss = pytest.importorskip(
        "slime.backends.megatron_utils.loss",
        reason="Megatron dependency is not available in this environment.",
    )
    monkeypatch.setattr(megatron_loss.mpu, "get_tensor_model_parallel_group", lambda: None)

    def fake_calc(logits_chunk, tokens_targets, tp_group, with_entropy=False, chunk_size=None):
        del tp_group, with_entropy, chunk_size
        return logits_chunk.gather(dim=-1, index=tokens_targets.reshape(-1, 1)).reshape(-1, 1), None

    monkeypatch.setattr(megatron_loss, "calculate_log_probs_and_entropy", fake_calc)

    vocab_size = 32
    total_len = 10
    logits = torch.zeros(1, total_len, vocab_size, dtype=torch.float32)
    for row in range(total_len):
        for token in range(vocab_size):
            logits[0, row, token] = row * 1000 + token
    tokens = torch.tensor([0, 1, 2, 3, 4, 5, 10, 20, 11, 21], dtype=torch.long)
    args = type("Args", (), {"qkv_format": "thd", "rollout_temperature": 1.0, "log_probs_chunk_size": 0})()

    strict_out = megatron_loss.thd_packed_pair_axis_log_probs_from_logits(
        logits,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[total_len],
        per_sample_source_rows=[torch.tensor([0, 1, 2, 3, 4, 1, 3, 6, 7])],
        per_sample_target_positions=[torch.arange(1, total_len)],
    )[0]
    standard_out = megatron_loss.thd_packed_pair_axis_log_probs_from_logits(
        logits,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[total_len],
        per_sample_source_rows=[torch.arange(total_len - 1)],
        per_sample_target_positions=[torch.arange(1, total_len)],
    )[0]

    torch.testing.assert_close(strict_out, torch.tensor([1, 1002, 2003, 3004, 4005, 1010, 3020, 6011, 7021.0]))
    torch.testing.assert_close(standard_out, torch.tensor([1, 1002, 2003, 3004, 4005, 5010, 6020, 7011, 8021.0]))
    assert not torch.equal(strict_out[-4:], standard_out[-4:])


def test_g1_ebft_policy_loss_dispatches_logprob_indexing(monkeypatch) -> None:
    megatron_loss = pytest.importorskip(
        "slime.backends.megatron_utils.loss",
        reason="Megatron dependency is not available in this environment.",
    )
    monkeypatch.setattr(megatron_loss.mpu, "get_context_parallel_world_size", lambda: 1)

    called = []

    def fake_standard(*args, **kwargs):
        del args, kwargs
        called.append("standard")
        return [torch.zeros(3, dtype=torch.float32)]

    def fake_strict(*args, **kwargs):
        del args, kwargs
        called.append("strict")
        return [torch.ones(3, dtype=torch.float32)]

    def fake_mean(*, per_sample_log_probs_next, **kwargs):
        return per_sample_log_probs_next[0].sum(), per_sample_log_probs_next[0].sum() + 1

    monkeypatch.setattr(megatron_loss, "thd_packed_full_sequence_next_log_probs_from_logits", fake_standard)
    monkeypatch.setattr(megatron_loss, "thd_packed_pair_axis_log_probs_from_logits", fake_strict)
    monkeypatch.setattr(megatron_loss, "ebft_mean_rl_ce_over_packed_samples", fake_mean)

    base_args = {
        "qkv_format": "thd",
        "allgather_cp": False,
        "advantage_estimator": "g1",
        "loss_type": "policy_loss",
        "use_opsm": False,
        "entropy_coef": 0.0,
        "g1_qa_masking": True,
        "g1_ce_loss_coef": 0.03,
        "g1_prompt_length": 2,
        "g1_context_length": 1,
        "g1_generate_length": 1,
        "g1_stride": 1,
        "g1_response_length": 1,
    }
    batch = {
        "unconcat_tokens": [torch.tensor([5, 6, 7])],
        "total_lengths": [3],
        "response_lengths": [1],
        "ebft_action_mask_next": [torch.tensor([False, True, True])],
        "ebft_qa_mask_next": [torch.tensor([True, True, True])],
        "ebft_advantages_next": [torch.zeros(3)],
        "ebft_logprob_source_rows": [torch.tensor([0, 0, 1])],
        "ebft_logprob_target_positions": [torch.tensor([1, 2, 2])],
    }
    logits = torch.zeros(1, 3, 8, dtype=torch.float32)

    args_default = type("Args", (), base_args)()
    loss_default, metrics_default = megatron_loss.policy_loss_function_g1_ebft(
        args_default,
        batch,
        logits,
        lambda x: x,
    )
    assert called[-1] == "standard"
    torch.testing.assert_close(loss_default, torch.tensor(0.03))
    torch.testing.assert_close(metrics_default["pg_loss"], torch.tensor(0.0))

    args_standard = type("Args", (), {**base_args, "g1_ebft_logprob_indexing": "standard_next_token"})()
    loss_standard, metrics_standard = megatron_loss.policy_loss_function_g1_ebft(
        args_standard,
        batch,
        logits,
        lambda x: x,
    )
    assert called[-1] == "standard"
    torch.testing.assert_close(loss_standard, torch.tensor(0.03))
    torch.testing.assert_close(metrics_standard["pg_loss"], torch.tensor(0.0))

    args_strict = type("Args", (), {**base_args, "g1_ebft_logprob_indexing": "strict_block_source"})()
    loss_strict, metrics_strict = megatron_loss.policy_loss_function_g1_ebft(
        args_strict,
        batch,
        logits,
        lambda x: x,
    )
    assert called[-1] == "strict"
    torch.testing.assert_close(loss_strict, torch.tensor(3.12))
    torch.testing.assert_close(metrics_strict["pg_loss"], torch.tensor(3.0))


def test_g1_ebft_policy_loss_strict_metadata_drives_action_gradients(monkeypatch) -> None:
    megatron_loss = pytest.importorskip(
        "slime.backends.megatron_utils.loss",
        reason="Megatron dependency is not available in this environment.",
    )
    monkeypatch.setattr(megatron_loss.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(megatron_loss.mpu, "get_tensor_model_parallel_group", lambda: None)

    def fake_calc(logits_chunk, tokens_targets, tp_group, with_entropy=False, chunk_size=None):
        del tp_group, with_entropy, chunk_size
        return logits_chunk.gather(dim=-1, index=tokens_targets.reshape(-1, 1)).reshape(-1, 1), None

    monkeypatch.setattr(megatron_loss, "calculate_log_probs_and_entropy", fake_calc)

    args = type(
        "Args",
        (),
        {
            "qkv_format": "thd",
            "allgather_cp": False,
            "advantage_estimator": "g1",
            "loss_type": "policy_loss",
            "use_opsm": False,
            "entropy_coef": 0.0,
            "g1_qa_masking": True,
            "g1_ce_loss_coef": 0.0,
            "g1_prompt_length": 6,
            "g1_context_length": 2,
            "g1_generate_length": 2,
            "g1_stride": 2,
            "g1_response_length": 4,
            "g1_ebft_logprob_indexing": "strict_block_source",
            "rollout_temperature": 1.0,
            "log_probs_chunk_size": 0,
        },
    )()
    tokens = torch.tensor([0, 1, 2, 3, 4, 5, 10, 20, 11, 21], dtype=torch.long)
    logits = torch.zeros(1, 10, 32, dtype=torch.float32, requires_grad=True)
    batch = {
        "unconcat_tokens": [tokens],
        "total_lengths": [10],
        "response_lengths": [4],
        "ebft_action_mask_next": [torch.tensor([False, False, False, False, False, True, True, True, True])],
        "ebft_qa_mask_next": [torch.tensor([False, False, False, False, False, True, True, True, True])],
        "ebft_advantages_next": [torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])],
        "ebft_logprob_source_rows": [torch.tensor([0, 1, 2, 3, 4, 1, 3, 6, 7])],
        "ebft_logprob_target_positions": [torch.arange(1, 10)],
    }

    loss, metrics = megatron_loss.policy_loss_function_g1_ebft(args, batch, logits, lambda x: x)
    loss.backward()

    torch.testing.assert_close(metrics["pg_loss"], torch.tensor(-2.5))
    expected_grad = torch.zeros_like(logits)
    expected_grad[0, 1, 10] = -0.25
    expected_grad[0, 3, 20] = -0.5
    expected_grad[0, 6, 11] = -0.75
    expected_grad[0, 7, 21] = -1.0
    torch.testing.assert_close(logits.grad, expected_grad)
    torch.testing.assert_close(loss.detach(), torch.tensor(-2.5))


def test_optional_openrlhf_ebft_policy_loss_matches_slime_packed_golden_fixture() -> None:
    """When OpenRLHF is available, compare its actor loss to Slime on the exact same fixture."""

    EBFTPolicyLoss, import_note = _load_openrlhf_ebft_policy_loss_from_checkout()
    if EBFTPolicyLoss is None:
        pytest.skip(import_note)

    fixture = _openrlhf_ebft_actor_loss_fixture()
    log_probs = fixture["log_probs"]
    advantages = fixture["advantages"]
    action_mask = fixture["action_mask"]
    qa_masks = fixture["qa_masks"]
    golden = fixture["golden"]
    assert isinstance(log_probs, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert isinstance(action_mask, torch.Tensor)
    assert isinstance(qa_masks, torch.Tensor)
    assert isinstance(golden, dict)

    mod = EBFTPolicyLoss(policy_loss_type="ppo")

    for qa_masking in (False, True):
        rl_openrlhf, ce_openrlhf = mod.forward(
            log_probs,
            advantages,
            action_mask=action_mask,
            qa_masks=qa_masks,
            qa_masking=qa_masking,
        )
        rl_slime, ce_slime = _slime_packed_fixture_loss(
            log_probs=log_probs,
            advantages=advantages,
            action_mask=action_mask,
            qa_masks=qa_masks,
            qa_masking=qa_masking,
        )
        rl_golden, ce_golden = golden[qa_masking]

        torch.testing.assert_close(
            rl_slime,
            rl_openrlhf,
            atol=OPENRLHF_EBFT_LOSS_ATOL,
            rtol=OPENRLHF_EBFT_LOSS_RTOL,
            msg=f"RL mismatch for qa_masking={qa_masking}; OpenRLHF import={import_note}",
        )
        torch.testing.assert_close(
            ce_slime,
            ce_openrlhf,
            atol=OPENRLHF_EBFT_LOSS_ATOL,
            rtol=OPENRLHF_EBFT_LOSS_RTOL,
            msg=f"CE mismatch for qa_masking={qa_masking}; OpenRLHF import={import_note}",
        )
        torch.testing.assert_close(rl_openrlhf, rl_golden, atol=OPENRLHF_EBFT_LOSS_ATOL, rtol=OPENRLHF_EBFT_LOSS_RTOL)
        torch.testing.assert_close(ce_openrlhf, ce_golden, atol=OPENRLHF_EBFT_LOSS_ATOL, rtol=OPENRLHF_EBFT_LOSS_RTOL)
