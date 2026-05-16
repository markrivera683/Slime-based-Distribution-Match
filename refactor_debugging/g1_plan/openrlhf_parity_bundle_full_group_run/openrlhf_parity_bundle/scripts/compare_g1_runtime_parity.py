#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def _flatten_embeddings(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([t.float() for t in values], dim=0)


def _sample_megatron_hidden(megatron_dump: dict) -> torch.Tensor:
    hidden = megatron_dump["hidden_states_post_sp_gather"].float()
    if hidden.ndim != 3 or hidden.shape[1] != 1:
        raise ValueError(f"Expected Megatron THD hidden [S, 1, H], got {hidden.shape}")
    total_lengths = [int(x) for x in megatron_dump["total_lengths"]]
    chunks = []
    cursor = 0
    for length in total_lengths:
        chunks.append(hidden[cursor : cursor + length, 0])
        cursor += length
    return torch.stack(chunks, dim=0)


def _cosine_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    cos = F.cosine_similarity(a.reshape(-1, a.shape[-1]).float(), b.reshape(-1, b.shape[-1]).float(), dim=-1)
    return {
        "mean": float(cos.mean().item()),
        "median": float(cos.median().item()),
        "min": float(cos.min().item()),
        "max": float(cos.max().item()),
    }


def _close_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _optional_tensor_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool | None:
    if left is None or right is None:
        return None
    return bool(torch.equal(left, right))


def _optional_mask_semantics_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool | None:
    if left is None or right is None:
        return None
    return bool(torch.equal(left == 0, right == 0))


def _require_same_count(name: str, left: list, right: list) -> None:
    if len(left) != len(right):
        raise ValueError(f"{name} sample count mismatch: Megatron={len(left)} OpenRLHF={len(right)}")


def _mask_summary(mask: torch.Tensor | None) -> dict[str, object] | None:
    if mask is None:
        return None
    allowed = mask == 0
    finite = torch.isfinite(mask)
    return {
        "shape": tuple(mask.shape),
        "allowed_count": int(allowed.sum().item()),
        "blocked_count": int((~allowed).sum().item()),
        "finite_count": int(finite.sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dump", required=True)
    parser.add_argument("--openrlhf-dump", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    megatron = torch.load(args.megatron_dump, map_location="cpu", weights_only=False)
    openrlhf = torch.load(args.openrlhf_dump, map_location="cpu", weights_only=False)

    megatron_tokens = torch.stack([t.long() for t in megatron["tokens"]], dim=0)
    openrlhf_tokens = openrlhf["sequences"].long()
    token_match = bool(torch.equal(megatron_tokens, openrlhf_tokens))
    qa_match = bool(torch.equal(torch.stack([t.long() for t in megatron["g1_qa_masks"]], dim=0), openrlhf["qa_masks"].long()))
    position_match = _optional_tensor_equal(megatron.get("g1_position_ids"), openrlhf.get("position_ids"))
    mask_match = _optional_mask_semantics_equal(megatron.get("g1_attention_mask"), openrlhf.get("attention_mask"))
    megatron_mask_summary = _mask_summary(megatron.get("g1_attention_mask"))
    openrlhf_mask_summary = _mask_summary(openrlhf.get("attention_mask"))

    megatron_hidden = _sample_megatron_hidden(megatron)
    openrlhf_hidden = openrlhf["hidden_states"].squeeze(-2).float()
    hidden_cos = _cosine_stats(megatron_hidden, openrlhf_hidden)

    megatron_gen = _flatten_embeddings(megatron["g1_gen_embedding"])
    megatron_gt = _flatten_embeddings(megatron["g1_gt_embedding"])
    openrlhf_gen = _flatten_embeddings(openrlhf["g1_gen_embedding"])
    openrlhf_gt = _flatten_embeddings(openrlhf["g1_gt_embedding"])
    _require_same_count("gen embeddings", megatron["g1_gen_embedding"], openrlhf["g1_gen_embedding"])
    _require_same_count("gt embeddings", megatron["g1_gt_embedding"], openrlhf["g1_gt_embedding"])
    gen_cos = _cosine_stats(megatron_gen, openrlhf_gen)
    gt_cos = _cosine_stats(megatron_gt, openrlhf_gt)
    reward_stats = None
    if "scalar_rewards" in megatron or "scalar_rewards" in openrlhf:
        if "scalar_rewards" not in megatron or "scalar_rewards" not in openrlhf:
            raise ValueError("scalar_rewards present in only one dump")
        _require_same_count("scalar_rewards", megatron["scalar_rewards"], openrlhf["scalar_rewards"])
        reward_stats = _close_stats(torch.tensor(megatron["scalar_rewards"]), torch.tensor(openrlhf["scalar_rewards"]))
    advantage_cos = None
    if "g1_token_advantages" in megatron or "g1_token_advantages" in openrlhf:
        if "g1_token_advantages" not in megatron or "g1_token_advantages" not in openrlhf:
            raise ValueError("g1_token_advantages present in only one dump")
        _require_same_count(
            "g1_token_advantages",
            megatron["g1_token_advantages"],
            openrlhf["g1_token_advantages"],
        )
        advantage_cos = _cosine_stats(
            torch.stack([t.float() for t in megatron["g1_token_advantages"]], dim=0).unsqueeze(-1),
            torch.stack([t.float() for t in openrlhf["g1_token_advantages"]], dim=0).unsqueeze(-1),
        )

    report = f"""# G1 Runtime Parity Report

## Inputs

- Megatron dump: `{args.megatron_dump}`
- OpenRLHF dump: `{args.openrlhf_dump}`

## Contract Checks

- Token IDs match: `{token_match}`
- QA masks match: `{qa_match}`
- Position ids match: `{position_match}`
- Attention mask tensors match: `{mask_match}`
- Megatron ref forward mode: `{megatron.get("g1_megatron_ref_forward_mode", "standard")}`
- Megatron attention mask status: `{megatron.get("g1_attention_mask_status", "not_dumped")}`
- Megatron attention mask applied: `{megatron.get("g1_attention_mask_applied", False)}`
- Megatron hidden shape: `{tuple(megatron_hidden.shape)}`
- OpenRLHF hidden shape: `{tuple(openrlhf_hidden.shape)}`
- Megatron gen embedding shape: `{tuple(megatron_gen.shape)}`
- OpenRLHF gen embedding shape: `{tuple(openrlhf_gen.shape)}`

## Mask Summary

- Megatron mask summary: `{megatron_mask_summary}`
- OpenRLHF mask summary: `{openrlhf_mask_summary}`

## Cosine Similarity

### Full Hidden States

```text
mean   = {hidden_cos['mean']:.8f}
median = {hidden_cos['median']:.8f}
min    = {hidden_cos['min']:.8f}
max    = {hidden_cos['max']:.8f}
```

### Gen Block Embeddings

```text
mean   = {gen_cos['mean']:.8f}
median = {gen_cos['median']:.8f}
min    = {gen_cos['min']:.8f}
max    = {gen_cos['max']:.8f}
```

### GT Block Embeddings

```text
mean   = {gt_cos['mean']:.8f}
median = {gt_cos['median']:.8f}
min    = {gt_cos['min']:.8f}
max    = {gt_cos['max']:.8f}
```

### Rewards

```text
{f"max_abs  = {reward_stats['max_abs']:.8f}" if reward_stats is not None else "not available"}
{f"mean_abs = {reward_stats['mean_abs']:.8f}" if reward_stats is not None else "not available"}
```

### Token Advantages

```text
{f"mean   = {advantage_cos['mean']:.8f}" if advantage_cos is not None else "not available"}
{f"median = {advantage_cos['median']:.8f}" if advantage_cos is not None else "not available"}
{f"min    = {advantage_cos['min']:.8f}" if advantage_cos is not None else "not available"}
{f"max    = {advantage_cos['max']:.8f}" if advantage_cos is not None else "not available"}
```

## Interpretation

This report compares the current Slime Megatron/ref fast path against OpenRLHF Critic on identical token IDs and QA masks.

Tight tensor equality is expected only after both staged contracts pass: OpenRLHF position ids must drive Megatron RoPE, and EBFT dense strided mask semantics must reach the Megatron attention backend. If the report says the Megatron dense mask was built but not applied, remaining generated-block gaps should be treated as attention-mask work, not full exactness.
"""
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[g1-runtime-parity] wrote {output_path}")
    print(report)


if __name__ == "__main__":
    main()
