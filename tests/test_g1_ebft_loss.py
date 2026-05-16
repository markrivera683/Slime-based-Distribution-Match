"""CPU tests for OpenRLHF-aligned EBFT policy loss primitives."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from slime.backends.megatron_utils import loss as megatron_loss
from slime.utils.g1_ebft_loss import (
    ebft_build_next_token_action_qa_advantages,
    ebft_compute_rl_ce_scalars_after_masks,
    ebft_mean_rl_ce_over_packed_samples,
    openrlhf_masked_mean,
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


def test_thd_full_sequence_logprob_gather_returns_1d_samples(monkeypatch) -> None:
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


def test_optional_openrlhf_ebf_policy_loss_import_parity_if_repo_present() -> None:
    """When the OpenRLHF checkout is mounted, slime EBFT must match ``EBFTPolicyLoss``."""

    rlhf_models = Path("/mnt/data/ebft-distribution-new/code/openrlhf/models")
    if not rlhf_models.exists():
        pytest.skip("OpenRLHF reference checkout not mounted at expected path.")

    try:
        repo_root = str(rlhf_models.parent.parent.resolve())
        if repo_root not in __import__("sys").path:
            __import__("sys").path.insert(0, repo_root)
        from openrlhf.models.loss import EBFTPolicyLoss  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency chain
        pytest.skip(f"OpenRLHF reference import unavailable ({exc}); install deps or skip in CI.")

    torch.manual_seed(3)
    B, S = 2, 9
    log_probs = torch.randn(B, S, dtype=torch.float32, requires_grad=True)
    advantages = torch.randn(B, S, dtype=torch.float32)
    action_mask = torch.zeros(B, S, dtype=torch.bool)
    action_mask[:, 5:] = True
    qa_masks = torch.ones(B, S, dtype=torch.bool)
    qa_masks[0, 2:4] = False

    mod = EBFTPolicyLoss(policy_loss_type="ppo")

    for qa_masking in (False, True):
        rl_o, ce_o = mod.forward(
            log_probs,
            advantages,
            action_mask=action_mask,
            qa_masks=qa_masks,
            qa_masking=qa_masking,
        )

        qa_next = qa_masks if qa_masking else torch.ones_like(qa_masks)

        rl_s, ce_s = ebft_compute_rl_ce_scalars_after_masks(
            log_probs_next=log_probs,
            advantages_next=advantages,
            action_mask_next=action_mask.to(dtype=torch.bool),
            qa_mask_next=qa_next.to(dtype=torch.bool),
            qa_masking=qa_masking,
            policy_loss_type="ppo",
        )

        torch.testing.assert_close(rl_s, rl_o)
        torch.testing.assert_close(ce_s, ce_o)
