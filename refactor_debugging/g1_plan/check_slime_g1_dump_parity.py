#!/usr/bin/env python
"""Check slime G1 math against an OpenRLHF runtime dump fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from slime.utils.g1_core import (
    compute_pointwise_rewards,
    compute_rloo_shaped_rewards,
    expand_block_rewards_to_token_advantages,
    whiten_embeddings_batched,
)


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        max_abs = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_abs={max_abs}\n{exc}") from exc
    print(f"[slime-parity] {name}: ok shape={tuple(actual.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="OpenRLHF dump .pt path")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    dump_path = Path(args.dump)
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)

    gen_embedding = dump["gen_embedding"]
    gt_embedding = dump["gt_embedding"]
    use_whitening = bool(dump["use_whitening"])
    generate_length = int(dump["generate_length"])
    n_samples_per_prompt = int(dump["n_samples_per_prompt"])
    alignment_rew_coef = float(dump["alignment_rew_coef"])
    diversity_rew_coef = float(dump["diversity_rew_coef"])

    if use_whitening:
        whitened_gen, whitened_gt = whiten_embeddings_batched(gen_embedding, gt_embedding, normalize=False)
        assert_close(
            "whitened_gen_embedding",
            whitened_gen,
            dump["whitened_gen_embedding"],
            rtol=args.rtol,
            atol=args.atol,
        )
        assert_close(
            "whitened_gt_embedding",
            whitened_gt,
            dump["whitened_gt_embedding"],
            rtol=args.rtol,
            atol=args.atol,
        )

    rewards, gt_rewards, diversity_rewards = compute_pointwise_rewards(
        gen_embedding,
        gt_embedding,
        alignment_rew_coef=alignment_rew_coef,
        diversity_rew_coef=diversity_rew_coef,
        use_whitening=use_whitening,
    )
    assert_close("gt_rewards", gt_rewards, dump["gt_rewards"], rtol=args.rtol, atol=args.atol)
    assert_close("diversity_rewards", diversity_rewards, dump["diversity_rewards"], rtol=args.rtol, atol=args.atol)
    assert_close("rewards", rewards, dump["rewards"], rtol=args.rtol, atol=args.atol)

    shaped_rewards, baseline = compute_rloo_shaped_rewards(
        rewards,
        diversity_rewards,
        gt_rewards,
        n_samples_per_prompt=n_samples_per_prompt,
        alignment_rew_coef=alignment_rew_coef,
        diversity_rew_coef=diversity_rew_coef,
    )
    assert_close("baseline", baseline, dump["baseline"], rtol=args.rtol, atol=args.atol)
    assert_close("shaped_rewards", shaped_rewards, dump["shaped_rewards"], rtol=args.rtol, atol=args.atol)

    expanded_rewards = torch.stack(
        [
            expand_block_rewards_to_token_advantages(
                sample_rewards,
                generate_length=generate_length,
                response_length=sample_rewards.numel() * generate_length,
            )
            for sample_rewards in shaped_rewards.squeeze(0)
        ]
    )
    assert_close("expanded_rewards", expanded_rewards, dump["expanded_rewards"], rtol=args.rtol, atol=args.atol)
    print(f"[slime-parity] PASS dump={dump_path}")


if __name__ == "__main__":
    main()
