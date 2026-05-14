from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch

from slime.utils.g1_core import (
    compute_pointwise_rewards,
    compute_rloo_shaped_rewards,
    expand_block_rewards_to_token_advantages,
)
from slime.utils.types import Sample


def _metadata_tensor(sample: Sample, key: str) -> torch.Tensor:
    if key not in sample.metadata:
        raise KeyError(f"Sample metadata is missing required G1 key: {key}")
    return torch.as_tensor(sample.metadata[key], dtype=torch.float32)


def _get_arg(args: Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def compute_group_g1_rewards(args: Namespace, samples: list[Sample]) -> list[float]:
    """Compute OpenRLHF G1 pointwise rewards for one prompt group.

    This reward path expects upstream rollout/fixture code to attach per-block
    critic embeddings in sample.metadata:
      - g1_gen_embedding: [num_blocks, hidden_dim]
      - g1_gt_embedding: [num_blocks, hidden_dim]
    """
    if not samples:
        return []

    n_samples_per_prompt = int(_get_arg(args, "n_samples_per_prompt", len(samples)))
    if len(samples) != n_samples_per_prompt:
        raise ValueError(f"G1 group size {len(samples)} != n_samples_per_prompt {n_samples_per_prompt}")

    gen_embedding = torch.stack([_metadata_tensor(sample, "g1_gen_embedding") for sample in samples], dim=0)
    gt_embedding = torch.stack([_metadata_tensor(sample, "g1_gt_embedding") for sample in samples], dim=0)
    if gen_embedding.ndim != 3:
        raise ValueError(f"Expected per-sample G1 embeddings shaped [num_blocks, hidden_dim], got {gen_embedding.shape}")
    if gen_embedding.shape != gt_embedding.shape:
        raise ValueError(f"G1 gen/gt embedding shapes must match, got {gen_embedding.shape} vs {gt_embedding.shape}")

    expected_num_blocks = _get_arg(args, "g1_num_blocks", None)
    if expected_num_blocks is not None and gen_embedding.shape[1] != int(expected_num_blocks):
        raise ValueError(f"G1 num_blocks {gen_embedding.shape[1]} != expected {int(expected_num_blocks)}")

    gen_embedding = gen_embedding.unsqueeze(0).unsqueeze(0)
    gt_embedding = gt_embedding.unsqueeze(0).unsqueeze(0)

    alignment_rew_coef = float(_get_arg(args, "alignment_rew_coef", 1.0))
    diversity_rew_coef = float(_get_arg(args, "diversity_rew_coef", 1.0))
    rewards, gt_rewards, diversity_rewards = compute_pointwise_rewards(
        gen_embedding,
        gt_embedding,
        alignment_rew_coef=alignment_rew_coef,
        diversity_rew_coef=diversity_rew_coef,
        use_whitening=bool(_get_arg(args, "use_whitening", True)),
    )
    shaped_rewards, baseline = compute_rloo_shaped_rewards(
        rewards,
        diversity_rewards,
        gt_rewards,
        n_samples_per_prompt=n_samples_per_prompt,
        alignment_rew_coef=alignment_rew_coef,
        diversity_rew_coef=diversity_rew_coef,
    )

    rewards = rewards.squeeze(0)
    gt_rewards = gt_rewards.squeeze(0)
    diversity_rewards = diversity_rewards.squeeze(0)
    shaped_rewards = shaped_rewards.squeeze(0)
    baseline = baseline.squeeze(0)

    scalar_rewards: list[float] = []
    for idx, sample in enumerate(samples):
        block_shaped_rewards = shaped_rewards[idx]
        num_blocks = int(block_shaped_rewards.numel())
        expected_response_length = _get_arg(args, "g1_response_length", None)
        if expected_response_length is not None and sample.response_length != int(expected_response_length):
            raise ValueError(
                f"G1 response_length {sample.response_length} != expected {int(expected_response_length)} at sample {idx}"
            )
        if sample.response_length % num_blocks != 0:
            raise ValueError(
                f"response_length {sample.response_length} is not divisible by G1 num_blocks {num_blocks}"
            )
        generate_length = sample.response_length // num_blocks
        token_advantages = expand_block_rewards_to_token_advantages(
            block_shaped_rewards,
            generate_length=generate_length,
            response_length=sample.response_length,
        )

        sample.metadata["g1_rewards"] = rewards[idx].detach().cpu().tolist()
        sample.metadata["g1_gt_rewards"] = gt_rewards[idx].detach().cpu().tolist()
        sample.metadata["g1_diversity_rewards"] = diversity_rewards[idx].detach().cpu().tolist()
        sample.metadata["g1_rloo_baseline"] = baseline[idx].detach().cpu().tolist()
        sample.metadata["g1_token_advantages"] = token_advantages.detach().cpu().tolist()

        scalar_reward = float(rewards[idx].mean().detach().cpu().item())
        scalar_rewards.append(scalar_reward)

    return scalar_rewards


async def custom_rm(args: Namespace, samples: list[Sample], **kwargs) -> list[float]:
    return compute_group_g1_rewards(args, samples)
