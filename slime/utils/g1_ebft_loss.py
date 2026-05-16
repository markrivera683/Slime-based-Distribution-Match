"""OpenRLHF ``EBFTPolicyLoss`` parity helpers (torch-only / CPU-testable).

The Megatron trainer imports this module alongside ``attach_ebft_g1_next_token_contract_to_batch``
in ``slime/utils/g1_ebft_data_contract.py``.
"""

from __future__ import annotations

import torch


def openrlhf_masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Match OpenRLHF ``masked_mean`` (fallback to unmasked mean when mask sums to zero)."""

    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)

    mask = mask.to(dtype=tensor.dtype)

    if dim is None:
        denom = mask.sum()
        if denom == 0:
            return tensor.mean()
        return (tensor * mask).sum() / denom

    masked_sum = (tensor * mask).sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)
    mean_all = tensor.mean(dim=dim, keepdim=keepdim)

    safe_div = masked_sum / denom.clamp(min=1)

    return torch.where(denom == 0, mean_all, safe_div)


def ebft_build_next_token_action_qa_advantages(
    *,
    full_sequence_1d: torch.Tensor,
    response_advantages_1d: torch.Tensor,
    qa_mask_full_1d: torch.Tensor | None,
    prompt_length: int | None = None,
    response_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``[L-1]`` next-token ``action_mask``, ``qa_mask_next`` (``qa[:,1:]``), and advantages.

    When ``qa_mask_full_1d`` is ``None``, ``qa_mask_next`` is all-ones (OpenRLHF layout before
    loss; loss may still apply ``qa_masking=False`` → ones inside ``EBFTPolicyLoss``).
    """
    seq = full_sequence_1d.reshape(-1).to(device=response_advantages_1d.device)
    resp_adv = response_advantages_1d.reshape(-1)
    device = seq.device
    dtype = resp_adv.dtype if resp_adv.is_floating_point else torch.float32
    resp_adv = resp_adv.to(device=device, dtype=dtype)

    L = int(seq.numel())
    R = int(resp_adv.numel())
    lm1 = L - 1

    if prompt_length is not None and response_length is not None:
        if int(prompt_length) + int(response_length) != L:
            raise ValueError(
                f"full sequence length {L} != g1_prompt_length + g1_response_length "
                f"({int(prompt_length)} + {int(response_length)})"
            )
        if R != int(response_length):
            raise ValueError(
                f"response_advantages length {R} != g1_response_length ({int(response_length)})"
            )
        prompt_len = int(prompt_length)
    else:
        prompt_len = L - R
        if prompt_len < 1:
            raise ValueError(f"invalid prompt_length {prompt_len} for L={L}, R={R}")
        if R != L - prompt_len:
            raise ValueError(
                f"response_advantages length {R} incompatible with inferred prompt_length={prompt_len} (L={L})"
            )

    if qa_mask_full_1d is None:
        qa_next = torch.ones(lm1, dtype=torch.bool, device=device)
    else:
        qa_full = qa_mask_full_1d.to(device=device).reshape(-1)
        if int(qa_full.numel()) != L:
            raise ValueError(f"g1_qa_mask length {qa_full.numel()} != full sequence length {L}")
        q_slice = qa_full[1:]
        qa_next = q_slice.to(torch.bool) if q_slice.dtype == torch.bool else q_slice.ne(0)

    action = torch.zeros(lm1, dtype=torch.bool, device=device)
    adv_vec = torch.zeros(lm1, dtype=dtype, device=device)

    start = prompt_len - 1
    end = start + R
    if end > lm1:
        raise ValueError(f"Invalid EBFT slice [{start}, {end}) for length L-1={lm1}")

    action[start:end] = True
    adv_vec[start:end] = resp_adv

    return action, qa_next, adv_vec


def build_ebft_g1_next_token_tensors(
    *,
    g1_full_sequence: torch.Tensor,
    g1_qa_mask: torch.Tensor | None,
    response_advantages: torch.Tensor,
    g1_prompt_length: int,
    g1_response_length: int,
    qa_masking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed OpenRLHF G1 geometry helper (matches ``prompt_length + response_length == L``)."""

    qa_in = g1_qa_mask if qa_masking else None
    return ebft_build_next_token_action_qa_advantages(
        full_sequence_1d=g1_full_sequence,
        response_advantages_1d=response_advantages,
        qa_mask_full_1d=qa_in,
        prompt_length=int(g1_prompt_length),
        response_length=int(g1_response_length),
    )


def ebft_compute_rl_ce_scalars_after_masks(
    *,
    log_probs_next: torch.Tensor,
    advantages_next: torch.Tensor,
    action_mask_next: torch.Tensor,
    qa_mask_next: torch.Tensor,
    qa_masking: bool,
    policy_loss_type: str = "ppo",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scalar ``(rl_loss, ce_loss)`` grouped like ``EBFTPolicyLoss.forward``."""

    if tuple(log_probs_next.shape) != tuple(action_mask_next.shape):
        raise ValueError(
            f"log_probs_next shape {tuple(log_probs_next.shape)} != action_mask_next shape {tuple(action_mask_next.shape)}"
        )

    qa_eff = qa_mask_next if qa_masking else torch.ones_like(qa_mask_next, dtype=torch.bool)

    rl_mask_bool = action_mask_next.to(dtype=torch.bool) & qa_eff
    ce_mask_bool = (~action_mask_next.to(dtype=torch.bool)) & qa_eff

    log_ratio = log_probs_next - log_probs_next.detach()
    if policy_loss_type == "ppo":
        ratio = log_ratio.exp()
    elif policy_loss_type == "gspo":
        denom = rl_mask_bool.to(log_ratio.dtype).sum(dim=-1).clamp(min=1)
        seq_log_ratio = (log_ratio * rl_mask_bool.to(log_ratio.dtype)).sum(dim=-1) / denom
        ratio = seq_log_ratio.exp().unsqueeze(-1) * rl_mask_bool.to(log_ratio.dtype)
    else:
        raise ValueError(f"Unsupported EBFT policy_loss_type: {policy_loss_type!r}")

    surr_loss = -ratio * advantages_next

    rl_mask = rl_mask_bool.to(dtype=log_probs_next.dtype)
    ce_mask = ce_mask_bool.to(dtype=log_probs_next.dtype)

    ce_loss_tensor = -log_probs_next

    rl_loss_row = openrlhf_masked_mean(surr_loss, rl_mask, dim=-1)
    ce_loss_row = openrlhf_masked_mean(ce_loss_tensor, ce_mask, dim=-1)

    return rl_loss_row.mean(), ce_loss_row.mean()


def ebft_mean_rl_ce_over_packed_samples(
    *,
    per_sample_log_probs_next: list[torch.Tensor],
    per_sample_adv_next: list[torch.Tensor],
    per_sample_action_mask_next: list[torch.Tensor],
    per_sample_qa_mask_next: list[torch.Tensor],
    qa_masking: bool,
    policy_loss_type: str = "ppo",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Megatron-packed micro-batch: one OpenRLHF row per concatenated sequence."""

    n = len(per_sample_log_probs_next)
    if not (
        n == len(per_sample_adv_next) == len(per_sample_action_mask_next) == len(per_sample_qa_mask_next)
    ):
        raise ValueError("Mismatched per-sample EBFT tensor list lengths.")

    rl_scalars = []
    ce_scalars = []
    for lp, adv, am, qa in zip(
        per_sample_log_probs_next,
        per_sample_adv_next,
        per_sample_action_mask_next,
        per_sample_qa_mask_next,
        strict=True,
    ):
        if lp.shape != adv.shape or lp.shape != am.shape or lp.shape != qa.shape:
            raise ValueError(
                f"Per-sample shape mismatch lp={tuple(lp.shape)} adv={tuple(adv.shape)} "
                f"action={tuple(am.shape)} qa={tuple(qa.shape)}"
            )
        r_i, c_i = ebft_compute_rl_ce_scalars_after_masks(
            log_probs_next=lp.unsqueeze(0),
            advantages_next=adv.unsqueeze(0),
            action_mask_next=am.unsqueeze(0).to(dtype=torch.bool),
            qa_mask_next=qa.unsqueeze(0).to(dtype=torch.bool),
            qa_masking=qa_masking,
            policy_loss_type=policy_loss_type,
        )
        rl_scalars.append(r_i)
        ce_scalars.append(c_i)

    return torch.stack(rl_scalars).mean(), torch.stack(ce_scalars).mean()
