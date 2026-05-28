"""OpenRLHF ``EBFTPolicyLoss`` parity helpers (torch-only / CPU-testable).

The Megatron trainer imports this module alongside ``attach_ebft_g1_next_token_contract_to_batch``
in ``slime/utils/g1_ebft_data_contract.py``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from slime.utils.g1_core import get_num_strided_blocks


G1_EBFT_ACTOR_LOSS_DUMP_ENV = "G1_EBFT_ACTOR_LOSS_DUMP_PATH"
G1_EBFT_ACTOR_LOSS_DUMP_FORMAT = "slime_g1_ebft_actor_loss_runtime_v1"
G1_EBFT_LOGPROB_INDEXING_STANDARD = "standard_next_token"
G1_EBFT_LOGPROB_INDEXING_STRICT_BLOCK = "strict_block_source"
G1_EBFT_LOGPROB_INDEXING_CHOICES = (
    G1_EBFT_LOGPROB_INDEXING_STANDARD,
    G1_EBFT_LOGPROB_INDEXING_STRICT_BLOCK,
)


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


def build_strict_block_source_target_pairs(
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return OpenRLHF EBFT source rows and target positions for generated tokens.

    OpenRLHF block prediction uses boundary logits for the first generated step:
    ``context_length + block_idx * stride - 1``. Later steps use the previous
    generated token of the same block. Returned tensors are sample-local
    positions in the full ``prompt + response`` sequence.
    """

    prompt_length = int(prompt_length)
    context_length = int(context_length)
    generate_length = int(generate_length)
    stride = int(stride)
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1 for strict EBFT block source rows, got {context_length}")

    num_blocks = get_num_strided_blocks(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    source_rows: list[int] = []
    target_positions: list[int] = []
    for step_idx in range(generate_length):
        for block_idx in range(num_blocks):
            response_idx = step_idx * num_blocks + block_idx
            if step_idx == 0:
                source = context_length + block_idx * stride - 1
            else:
                source = prompt_length + (step_idx - 1) * num_blocks + block_idx
            source_rows.append(source)
            target_positions.append(prompt_length + response_idx)

    return (
        torch.tensor(source_rows, dtype=torch.long, device=device),
        torch.tensor(target_positions, dtype=torch.long, device=device),
    )


def build_ebft_g1_logprob_pair_axis(
    *,
    prompt_length: int,
    response_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
    indexing: str = G1_EBFT_LOGPROB_INDEXING_STANDARD,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pair-axis ``(source_logit_row, target_token_pos, action_mask)`` tensors.

    ``standard_next_token`` is the existing Slime behavior: row ``i`` predicts
    token ``i + 1``. ``strict_block_source`` matches OpenRLHF EBFT's processed
    logits: prompt CE pairs remain standard, while generated/action pairs gather
    logits from the block-prediction source rows.
    """

    prompt_length = int(prompt_length)
    response_length = int(response_length)
    context_length = int(context_length)
    generate_length = int(generate_length)
    stride = int(stride)
    if indexing not in G1_EBFT_LOGPROB_INDEXING_CHOICES:
        raise ValueError(f"Unsupported G1 EBFT logprob indexing: {indexing!r}")
    if prompt_length < 1:
        raise ValueError(f"prompt_length must be >= 1, got {prompt_length}")
    if response_length < 0:
        raise ValueError(f"response_length must be >= 0, got {response_length}")

    total_length = prompt_length + response_length
    target_positions = torch.arange(1, total_length, dtype=torch.long, device=device)
    source_rows = torch.arange(0, total_length - 1, dtype=torch.long, device=device)
    action_mask = torch.zeros(total_length - 1, dtype=torch.bool, device=device)
    if response_length:
        action_start = prompt_length - 1
        action_end = action_start + response_length
        action_mask[action_start:action_end] = True

    if indexing == G1_EBFT_LOGPROB_INDEXING_STANDARD or response_length == 0:
        return source_rows, target_positions, action_mask

    num_blocks = get_num_strided_blocks(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    expected_response_length = int(generate_length) * int(num_blocks)
    if response_length != expected_response_length:
        raise ValueError(
            "strict_block_source requires response_length == generate_length * num_blocks "
            f"({response_length} != {generate_length} * {num_blocks})"
        )

    action_sources, action_targets = build_strict_block_source_target_pairs(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
        device=device,
    )
    source_rows[action_start:action_end] = action_sources
    target_positions[action_start:action_end] = action_targets
    return source_rows, target_positions, action_mask


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


def _detach_cpu_tensor_list(values: list[torch.Tensor], *, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
    outs = []
    for tensor in values:
        out = tensor.detach().cpu()
        if dtype is not None:
            out = out.to(dtype=dtype)
        outs.append(out.contiguous())
    return outs


def dump_ebft_actor_loss_runtime(
    *,
    path: str | os.PathLike[str],
    per_sample_log_probs_next: list[torch.Tensor],
    per_sample_adv_next: list[torch.Tensor],
    per_sample_action_mask_next: list[torch.Tensor],
    per_sample_qa_mask_next: list[torch.Tensor],
    qa_masking: bool,
    g1_ce_loss_coef: float,
    slime_rl_loss: torch.Tensor,
    slime_ce_loss: torch.Tensor,
    slime_total_loss: torch.Tensor,
    policy_loss_type: str = "ppo",
    metadata: dict | None = None,
) -> Path:
    """Write a CPU runtime dump for one Slime EBFT actor-loss microbatch."""

    dump_path = Path(path).expanduser()
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": G1_EBFT_ACTOR_LOSS_DUMP_FORMAT,
        "metadata": {
            "qa_masking": bool(qa_masking),
            "policy_loss_type": policy_loss_type,
            **(metadata or {}),
        },
        "inputs": {
            "log_probs_next": _detach_cpu_tensor_list(per_sample_log_probs_next),
            "advantages_next": _detach_cpu_tensor_list(per_sample_adv_next),
            "action_mask_next": _detach_cpu_tensor_list(per_sample_action_mask_next, dtype=torch.bool),
            "qa_mask_next": _detach_cpu_tensor_list(per_sample_qa_mask_next, dtype=torch.bool),
        },
        "scalars": {
            "g1_ce_loss_coef": float(g1_ce_loss_coef),
            "slime_rl_loss": float(slime_rl_loss.detach().float().cpu().item()),
            "slime_ce_loss": float(slime_ce_loss.detach().float().cpu().item()),
            "slime_total_loss": float(slime_total_loss.detach().float().cpu().item()),
        },
    }

    with tempfile.NamedTemporaryFile(prefix=f".{dump_path.name}.", suffix=".tmp", dir=dump_path.parent, delete=False) as f:
        tmp_path = Path(f.name)

    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, dump_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return dump_path


def maybe_dump_ebft_actor_loss_runtime(
    *,
    per_sample_log_probs_next: list[torch.Tensor],
    per_sample_adv_next: list[torch.Tensor],
    per_sample_action_mask_next: list[torch.Tensor],
    per_sample_qa_mask_next: list[torch.Tensor],
    qa_masking: bool,
    g1_ce_loss_coef: float,
    slime_rl_loss: torch.Tensor,
    slime_ce_loss: torch.Tensor,
    slime_total_loss: torch.Tensor,
    policy_loss_type: str = "ppo",
    metadata: dict | None = None,
) -> Path | None:
    """Opt-in runtime dump controlled by ``G1_EBFT_ACTOR_LOSS_DUMP_PATH``."""

    dump_path = os.environ.get(G1_EBFT_ACTOR_LOSS_DUMP_ENV)
    if not dump_path:
        return None

    return dump_ebft_actor_loss_runtime(
        path=dump_path,
        per_sample_log_probs_next=per_sample_log_probs_next,
        per_sample_adv_next=per_sample_adv_next,
        per_sample_action_mask_next=per_sample_action_mask_next,
        per_sample_qa_mask_next=per_sample_qa_mask_next,
        qa_masking=qa_masking,
        g1_ce_loss_coef=g1_ce_loss_coef,
        slime_rl_loss=slime_rl_loss,
        slime_ce_loss=slime_ce_loss,
        slime_total_loss=slime_total_loss,
        policy_loss_type=policy_loss_type,
        metadata=metadata,
    )
