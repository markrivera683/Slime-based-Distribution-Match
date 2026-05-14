#!/usr/bin/env python
"""Dump an OpenRLHF G1 pointwise runtime fixture.

This intentionally runs under the EBFT/OpenRLHF student environment and imports
OpenRLHF's reward helpers directly. It does not start a full training job; the
goal is to capture the exact intermediate tensors for a deterministic embedding
fixture so slime can use them as golden data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openrlhf.utils.embedding_utils import (
    get_alignment_rewards,
    get_diversity_rewards,
    whiten_embeddings_batched,
)


def fixed_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Return [micro, group, n_samples, num_blocks, hidden] tensors."""
    gen_embedding = torch.tensor(
        [
            [
                [1.0, 0.2, -0.1, 0.4, 0.7],
                [0.3, 1.1, 0.5, -0.2, 0.0],
                [0.0, 0.4, 1.2, 0.3, -0.5],
            ],
            [
                [0.6, -0.3, 0.8, 0.1, 1.0],
                [1.2, 0.1, -0.4, 0.6, 0.2],
                [0.5, 1.0, 0.2, -0.7, 0.3],
            ],
            [
                [-0.2, 0.9, 0.3, 1.1, 0.4],
                [0.4, -0.5, 1.3, 0.2, 0.8],
                [1.1, 0.2, 0.0, 0.5, -0.3],
            ],
            [
                [0.7, 0.4, 0.6, -0.8, 0.1],
                [-0.1, 0.8, 0.2, 1.0, 0.5],
                [0.2, -0.6, 0.9, 0.4, 1.2],
            ],
        ],
        dtype=torch.float32,
    )
    gt_embedding = torch.tensor(
        [
            [
                [0.9, 0.1, 0.0, 0.5, 0.6],
                [0.2, 1.0, 0.3, -0.1, 0.4],
                [0.1, 0.5, 1.1, 0.2, -0.4],
            ],
            [
                [0.4, -0.2, 0.9, 0.3, 0.8],
                [1.0, 0.2, -0.2, 0.5, 0.1],
                [0.6, 0.8, 0.4, -0.6, 0.2],
            ],
            [
                [-0.1, 0.7, 0.5, 0.9, 0.3],
                [0.5, -0.3, 1.1, 0.4, 0.6],
                [1.0, 0.3, 0.1, 0.6, -0.2],
            ],
            [
                [0.8, 0.5, 0.4, -0.7, 0.2],
                [0.0, 0.9, 0.1, 0.8, 0.7],
                [0.3, -0.4, 1.0, 0.5, 1.1],
            ],
        ],
        dtype=torch.float32,
    )
    return gen_embedding.unsqueeze(0).unsqueeze(0), gt_embedding.unsqueeze(0).unsqueeze(0)


def compute_rloo_baseline(
    diversity_rewards: torch.Tensor,
    gt_rewards: torch.Tensor,
    *,
    n_samples_per_prompt: int,
    alignment_rew_coef: float,
    diversity_rew_coef: float,
) -> torch.Tensor:
    original_shape = diversity_rewards.shape
    diversity_rewards = diversity_rewards.reshape(
        diversity_rewards.shape[0],
        -1,
        n_samples_per_prompt,
        diversity_rewards.shape[-1],
    )
    gt_rewards = gt_rewards.reshape(gt_rewards.shape[0], -1, n_samples_per_prompt, gt_rewards.shape[-1])

    gt_baseline = (gt_rewards.sum(2, keepdim=True) - gt_rewards) / float(n_samples_per_prompt - 1)
    if float(diversity_rew_coef) != 0.0 and n_samples_per_prompt > 2:
        diversity_baseline = (
            diversity_rewards.sum(2, keepdim=True) - 2.0 * diversity_rewards
        ) / float(n_samples_per_prompt - 2)
    else:
        diversity_baseline = torch.zeros_like(diversity_rewards)

    baseline = float(alignment_rew_coef) * gt_baseline - float(diversity_rew_coef) * diversity_baseline
    return baseline.reshape(original_shape)


def compute_dump(*, use_whitening: bool, generate_length: int) -> dict[str, torch.Tensor | int | bool | float]:
    n_samples_per_prompt = 4
    alignment_rew_coef = 1.0
    diversity_rew_coef = 1.0
    gen_embedding, gt_embedding = fixed_embeddings()

    whitened_gen_embedding = gen_embedding
    whitened_gt_embedding = gt_embedding
    if use_whitening:
        whitened_gen_embedding, whitened_gt_embedding = whiten_embeddings_batched(
            gen_embedding,
            gt_embedding,
            whiten_tol=1e-5,
            normalize=False,
        )

    gt_rewards = get_alignment_rewards(whitened_gen_embedding, whitened_gt_embedding)
    diversity_rewards = get_diversity_rewards(whitened_gen_embedding, per_token=False)

    gt_rewards = gt_rewards.reshape(gt_rewards.shape[0], -1, gt_rewards.shape[-1]) * 2
    diversity_rewards = diversity_rewards.reshape(diversity_rewards.shape[0], -1, diversity_rewards.shape[-1]) * 2
    rewards = alignment_rew_coef * gt_rewards - diversity_rew_coef * diversity_rewards

    baseline = compute_rloo_baseline(
        diversity_rewards,
        gt_rewards,
        n_samples_per_prompt=n_samples_per_prompt,
        alignment_rew_coef=alignment_rew_coef,
        diversity_rew_coef=diversity_rew_coef,
    )
    shaped_rewards = rewards - baseline
    expanded_rewards = torch.stack([sample_rewards.repeat(generate_length) for sample_rewards in shaped_rewards[0]])

    return {
        "gen_embedding": gen_embedding,
        "gt_embedding": gt_embedding,
        "whitened_gen_embedding": whitened_gen_embedding,
        "whitened_gt_embedding": whitened_gt_embedding,
        "gt_rewards": gt_rewards,
        "diversity_rewards": diversity_rewards,
        "rewards": rewards,
        "baseline": baseline,
        "shaped_rewards": shaped_rewards,
        "expanded_rewards": expanded_rewards,
        "use_whitening": use_whitening,
        "generate_length": generate_length,
        "n_samples_per_prompt": n_samples_per_prompt,
        "alignment_rew_coef": alignment_rew_coef,
        "diversity_rew_coef": diversity_rew_coef,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output .pt path")
    parser.add_argument("--generate-length", type=int, default=2)
    parser.add_argument("--no-whitening", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump = compute_dump(use_whitening=not args.no_whitening, generate_length=args.generate_length)
    torch.save(dump, output_path)

    print(f"[openrlhf-dump] wrote {output_path}")
    for key in [
        "gen_embedding",
        "whitened_gen_embedding",
        "gt_rewards",
        "diversity_rewards",
        "rewards",
        "baseline",
        "shaped_rewards",
        "expanded_rewards",
    ]:
        value = dump[key]
        if isinstance(value, torch.Tensor):
            print(f"[openrlhf-dump] {key}: shape={tuple(value.shape)} mean={value.float().mean().item():.8f}")


if __name__ == "__main__":
    main()
