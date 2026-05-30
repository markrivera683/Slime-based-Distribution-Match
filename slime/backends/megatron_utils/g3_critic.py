from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from slime.utils.g1_core import expand_block_rewards_to_token_advantages, whiten_embeddings_batched
from slime.utils.g2_core import compute_opd_cf_l1oo_rewards
from slime.utils.g3_ema import G3EMAAdapterController, G3EMAFeatureStepResult, is_g3_opd_fused_mode


@dataclass
class G3CriticClosureResult:
    """Outputs produced by the critic-owned G3 OPD/EMA closure."""

    feature_step: G3EMAFeatureStepResult
    token_advantages: list[torch.Tensor]
    scalar_rewards: list[float]
    block_rewards: torch.Tensor
    teacher_scores: torch.Tensor
    normalized_base_embedding: torch.Tensor
    reward_embedding: torch.Tensor


def _teacher_log_probs_to_g3_group_scores(
    *,
    teacher_log_probs: list[torch.Tensor] | None,
    num_groups: int,
    n_samples_per_prompt: int,
    device: torch.device,
    normalization: str,
) -> torch.Tensor:
    if teacher_log_probs is None:
        raise ValueError("G3 critic closure requires teacher_log_probs.")
    expected = int(num_groups) * int(n_samples_per_prompt)
    if len(teacher_log_probs) != expected:
        raise ValueError(
            "G3 critic closure teacher_log_probs must align with rollout samples; "
            f"got {len(teacher_log_probs)} for {expected} samples."
        )

    scores = []
    for sample_idx, log_probs in enumerate(teacher_log_probs):
        tensor = log_probs.detach().float().reshape(-1).to(device=device)
        if tensor.numel() == 0:
            raise ValueError(f"G3 critic closure teacher_log_probs[{sample_idx}] is empty.")
        if normalization == "mean":
            score = tensor.mean()
        elif normalization == "sum":
            score = tensor.sum()
        else:
            raise ValueError(
                "Unsupported --opd-cf-score-normalization "
                f"{normalization!r}; expected 'mean' or 'sum'."
            )
        scores.append(score)

    return torch.stack(scores, dim=0).view(1, int(num_groups), int(n_samples_per_prompt))


def _stack_g3_base_embeddings(
    args: Namespace,
    gen_embeddings: list[torch.Tensor],
    response_lengths: list[int],
) -> torch.Tensor:
    if not gen_embeddings:
        raise ValueError("G3 critic closure requires non-empty gen_embeddings.")
    if len(gen_embeddings) != len(response_lengths):
        raise ValueError(
            "G3 critic closure requires one response_length per embedding; "
            f"got {len(gen_embeddings)} embeddings and {len(response_lengths)} response lengths."
        )

    n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", len(gen_embeddings)))
    if n_samples_per_prompt <= 1:
        raise ValueError("G3 critic closure requires n_samples_per_prompt > 1.")
    if len(gen_embeddings) % n_samples_per_prompt != 0:
        raise ValueError(
            f"G3 sample count {len(gen_embeddings)} is not divisible by n_samples_per_prompt={n_samples_per_prompt}."
        )

    gen = torch.stack([tensor.float() for tensor in gen_embeddings], dim=0)
    if gen.ndim != 3:
        raise ValueError(
            "G3 critic closure expects each gen embedding to have shape [num_blocks, hidden_dim], "
            f"got stacked shape {tuple(gen.shape)}."
        )

    num_samples, num_blocks, _ = gen.shape
    expected_num_blocks = getattr(args, "g1_num_blocks", None)
    if expected_num_blocks is not None and num_blocks != int(expected_num_blocks):
        raise ValueError(f"G3 num_blocks {num_blocks} != expected {int(expected_num_blocks)}.")

    expected_response_length = getattr(args, "g1_response_length", None)
    for idx, response_length in enumerate(response_lengths):
        if expected_response_length is not None and int(response_length) != int(expected_response_length):
            raise ValueError(
                f"G3 response_length {response_length} != expected {int(expected_response_length)} at sample {idx}."
            )
        if int(response_length) % num_blocks != 0:
            raise ValueError(
                f"G3 response_length {response_length} is not divisible by num_blocks={num_blocks} at sample {idx}."
            )

    num_groups = num_samples // n_samples_per_prompt
    gen = gen.view(num_groups, n_samples_per_prompt, num_blocks, gen.shape[-1]).unsqueeze(0)
    gen = F.normalize(gen, p=2, dim=-1)
    if bool(getattr(args, "use_whitening", False)):
        gen, _ = whiten_embeddings_batched(
            gen,
            gen,
            whiten_tol=float(getattr(args, "whiten_tol", 1e-5)),
        )
    return gen.contiguous()


def run_g3_opd_critic_closure_from_embeddings(
    args: Namespace,
    controller: G3EMAAdapterController,
    gen_embeddings: list[torch.Tensor],
    response_lengths: list[int],
    teacher_log_probs: list[torch.Tensor] | None,
) -> G3CriticClosureResult:
    """Run the local critic-owned G3 closure from detached critic embeddings.

    This helper intentionally starts at the Megatron-hidden-to-G1-embedding
    boundary so it can be tested on CPU. The critic supplies detached base
    features; the helper normalizes/whitens them, trains only the feature
    adapter against detached EMA OPD-CF targets, updates EMA, then computes
    sync-ready no-grad token advantages and scalar rewards.
    """
    if not is_g3_opd_fused_mode(args):
        raise ValueError("G3 critic closure requires --g3-enable with OPD cf_l1oo on-policy mode.")

    base_embedding = _stack_g3_base_embeddings(args, gen_embeddings, response_lengths)
    _, num_groups, n_samples_per_prompt, num_blocks, _ = base_embedding.shape
    teacher_scores = _teacher_log_probs_to_g3_group_scores(
        teacher_log_probs=teacher_log_probs,
        num_groups=num_groups,
        n_samples_per_prompt=n_samples_per_prompt,
        device=base_embedding.device,
        normalization=str(getattr(args, "opd_cf_score_normalization", "mean")),
    )

    feature_step = controller.train_feature_step(
        base_embedding,
        teacher_scores,
        cf_num_freqs=int(getattr(args, "cf_num_freqs", 128)),
        cf_sigma=float(getattr(args, "cf_sigma", 1.0)),
        cf_seed=int(getattr(args, "cf_seed", 43)),
        cf_alpha=float(getattr(args, "cf_alpha", 0.5)),
        cf_beta=float(getattr(args, "cf_beta", 0.5)),
        score_temperature=float(getattr(args, "opd_cf_score_temperature", 1.0)),
    )

    with torch.no_grad():
        controller.live_adapter.eval()
        reward_embedding = controller.live_adapter(base_embedding.detach())
        block_rewards = compute_opd_cf_l1oo_rewards(
            reward_embedding,
            teacher_scores,
            cf_num_freqs=int(getattr(args, "cf_num_freqs", 128)),
            cf_sigma=float(getattr(args, "cf_sigma", 1.0)),
            cf_seed=int(getattr(args, "cf_seed", 43)),
            cf_alpha=float(getattr(args, "cf_alpha", 0.5)),
            cf_beta=float(getattr(args, "cf_beta", 0.5)),
            cf_reward_scale=float(getattr(args, "cf_reward_scale", 1.0)),
            score_temperature=float(getattr(args, "opd_cf_score_temperature", 1.0)),
        )

    flat_rewards = block_rewards.squeeze(0).reshape(len(gen_embeddings), num_blocks)
    token_advantages: list[torch.Tensor] = []
    scalar_rewards: list[float] = []
    for idx, response_length in enumerate(response_lengths):
        generate_length = int(response_length) // num_blocks
        token_advantages.append(
            expand_block_rewards_to_token_advantages(
                flat_rewards[idx],
                generate_length=generate_length,
                response_length=int(response_length),
            ).detach()
        )
        scalar_rewards.append(float(flat_rewards[idx].mean().detach().cpu().item()))

    return G3CriticClosureResult(
        feature_step=feature_step,
        token_advantages=token_advantages,
        scalar_rewards=scalar_rewards,
        block_rewards=flat_rewards.detach(),
        teacher_scores=teacher_scores.detach(),
        normalized_base_embedding=base_embedding.detach(),
        reward_embedding=reward_embedding.detach(),
    )
