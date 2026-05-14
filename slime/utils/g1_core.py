from __future__ import annotations

import torch
import torch.nn.functional as F


def get_num_strided_blocks(prompt_length: int, context_length: int, generate_length: int, stride: int) -> int:
    """Return the OpenRLHF G1 strided-block count."""
    remainder = prompt_length - generate_length - context_length
    if remainder < 0 or remainder % stride != 0:
        raise ValueError(
            "Invalid G1 strided-block geometry: "
            f"prompt_length={prompt_length}, context_length={context_length}, "
            f"generate_length={generate_length}, stride={stride}"
        )
    return remainder // stride + 1


def prepare_strided_token_blocks(
    prompts: list[torch.Tensor],
    full_sequences: list[torch.Tensor],
    *,
    prompt_length: int,
    stride: int,
    num_blocks: int,
    n_samples_per_prompt: int,
    context_length: int,
    generate_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match OpenRLHF's token block layout before embedding construction."""
    prompts_tensor = torch.stack(prompts)
    full_sequences_tensor = torch.stack(full_sequences)

    prompts_tensor = prompts_tensor.reshape(
        prompts_tensor.shape[0],
        prompts_tensor.shape[1] // n_samples_per_prompt,
        n_samples_per_prompt,
        prompts_tensor.shape[2],
    )
    full_sequences_tensor = full_sequences_tensor.reshape(
        full_sequences_tensor.shape[0],
        full_sequences_tensor.shape[1] // n_samples_per_prompt,
        n_samples_per_prompt,
        full_sequences_tensor.shape[2],
    )

    gt_blocks = prompts_tensor[:, :, :, context_length:].unfold(3, generate_length, stride)
    gen_blocks = full_sequences_tensor[:, :, :, prompt_length:]
    gen_blocks = gen_blocks.reshape(
        gen_blocks.shape[0],
        gen_blocks.shape[1],
        gen_blocks.shape[2],
        generate_length,
        num_blocks,
    )
    gen_blocks = gen_blocks.transpose(-1, -2)
    return gen_blocks, gt_blocks


def whiten_embeddings_batched(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    *,
    whiten_tol: float = 1e-5,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenRLHF-compatible whitening across the sample axis at dim=2."""
    if gen_embedding.shape != gt_embedding.shape:
        raise ValueError(f"gen_embedding and gt_embedding must share shape, got {gen_embedding.shape} vs {gt_embedding.shape}")
    if gen_embedding.ndim not in (5, 7):
        raise ValueError(f"Expected 5D or 7D embeddings with sample axis at dim=2, got {gen_embedding.shape}")

    ndim = gen_embedding.ndim
    perm = [0, 1] + list(range(3, ndim - 1)) + [2, ndim - 1]
    inv_perm = [0] * ndim
    for idx, value in enumerate(perm):
        inv_perm[value] = idx

    gen_perm = gen_embedding.permute(*perm).contiguous()
    gt_perm = gt_embedding.permute(*perm).contiguous()

    *batch_dims, n_samples, embed_dim = gen_perm.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= int(dim)

    gen_flat = gen_perm.reshape(batch_size, n_samples, embed_dim).float()
    gt_flat = gt_perm.reshape(batch_size, n_samples, embed_dim).float()

    try:
        u, singular_values, _ = torch.linalg.svd(gen_flat, full_matrices=False)
    except torch._C._LinAlgError:
        noise_scale = 1e-6 * gen_flat.abs().mean()
        gen_noisy = gen_flat + noise_scale * torch.randn_like(gen_flat)
        try:
            u, singular_values, _ = torch.linalg.svd(gen_noisy, full_matrices=False)
        except torch._C._LinAlgError:
            if normalize:
                return F.normalize(gen_embedding, p=2, dim=-1), F.normalize(gt_embedding, p=2, dim=-1)
            return gen_embedding, gt_embedding

    max_singular_value = singular_values.max(dim=-1, keepdim=True).values
    inv_s = torch.where(
        singular_values > whiten_tol * max_singular_value,
        1.0 / (singular_values + 1e-12),
        torch.zeros_like(singular_values),
    )
    whitening = (u * inv_s.unsqueeze(-2)) @ u.transpose(-1, -2)

    gen_whitened = whitening @ gen_flat
    gt_whitened = whitening @ gt_flat

    gen_whitened = gen_whitened.to(dtype=gen_embedding.dtype).reshape(*batch_dims, n_samples, embed_dim)
    gt_whitened = gt_whitened.to(dtype=gt_embedding.dtype).reshape(*batch_dims, n_samples, embed_dim)

    gen_whitened = gen_whitened.permute(*inv_perm).contiguous()
    gt_whitened = gt_whitened.permute(*inv_perm).contiguous()

    if normalize:
        gen_whitened = F.normalize(gen_whitened, p=2, dim=-1)
        gt_whitened = F.normalize(gt_whitened, p=2, dim=-1)

    return gen_whitened, gt_whitened


def get_alignment_rewards(gen_embedding: torch.Tensor, gt_embedding: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(gen_embedding, gt_embedding, dim=-1)


@torch.no_grad()
def get_diversity_rewards(gen_embedding: torch.Tensor, *, per_token: bool = False) -> torch.Tensor:
    if gen_embedding.shape[2] <= 1:
        if per_token:
            return torch.zeros(
                gen_embedding.shape[0],
                gen_embedding.shape[1],
                gen_embedding.shape[2],
                gen_embedding.shape[3],
                gen_embedding.shape[-2],
                device=gen_embedding.device,
                dtype=gen_embedding.dtype,
            )
        return torch.zeros(
            gen_embedding.shape[0],
            gen_embedding.shape[1],
            gen_embedding.shape[2],
            gen_embedding.shape[3],
            device=gen_embedding.device,
            dtype=gen_embedding.dtype,
        )

    if per_token:
        reorg = gen_embedding.permute(0, 1, 3, 2, 4, 5)
        n_samples = gen_embedding.shape[2]
        lhs = reorg.unsqueeze(3).repeat(1, 1, 1, n_samples, 1, 1, 1)
        rhs = reorg.unsqueeze(4).repeat(1, 1, 1, 1, n_samples, 1, 1)
        full_sims = torch.sum(lhs * rhs, dim=-1)
        diagonal = torch.eye(full_sims.shape[-2], device=full_sims.device, dtype=torch.bool)
        sims = full_sims.masked_fill(diagonal.view(1, 1, 1, full_sims.shape[-2], full_sims.shape[-2], 1), 0.0)
        diversity_rewards = sims.sum(dim=-2) / (full_sims.shape[-2] - 1)
        return diversity_rewards.permute(0, 1, 3, 2, 4)

    reorg = gen_embedding.permute(0, 1, 3, 2, 4)
    n_samples = gen_embedding.shape[2]
    lhs = reorg.unsqueeze(3).repeat(1, 1, 1, n_samples, 1, 1)
    rhs = reorg.unsqueeze(4).repeat(1, 1, 1, 1, n_samples, 1)
    full_sims = torch.sum(lhs * rhs, dim=-1)
    diagonal = torch.eye(full_sims.shape[-1], device=full_sims.device, dtype=torch.bool)
    sims = full_sims.masked_fill(diagonal.view(1, 1, 1, full_sims.shape[-1], full_sims.shape[-1]), 0.0)
    diversity_rewards = sims.sum(dim=-1) / (full_sims.shape[-1] - 1)
    return diversity_rewards.permute(0, 1, 3, 2)


def compute_pointwise_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    *,
    alignment_rew_coef: float = 1.0,
    diversity_rew_coef: float = 1.0,
    use_whitening: bool = True,
    whiten_tol: float = 1e-5,
    per_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if use_whitening:
        gen_embedding, gt_embedding = whiten_embeddings_batched(
            gen_embedding,
            gt_embedding,
            whiten_tol=whiten_tol,
            normalize=False,
        )

    alignment = get_alignment_rewards(gen_embedding, gt_embedding)
    diversity = get_diversity_rewards(gen_embedding, per_token=per_token)

    if per_token:
        alignment = alignment.reshape(alignment.shape[0], -1, alignment.shape[-2], alignment.shape[-1])
        diversity = diversity.reshape(diversity.shape[0], -1, diversity.shape[-2], diversity.shape[-1])
    else:
        alignment = alignment.reshape(alignment.shape[0], -1, alignment.shape[-1])
        diversity = diversity.reshape(diversity.shape[0], -1, diversity.shape[-1])

    alignment = alignment * 2
    diversity = diversity * 2
    rewards = float(alignment_rew_coef) * alignment - float(diversity_rew_coef) * diversity
    return rewards, alignment, diversity


@torch.no_grad()
def compute_rloo_baseline(
    diversity_rewards: torch.Tensor,
    gt_rewards: torch.Tensor,
    *,
    n_samples_per_prompt: int,
    alignment_rew_coef: float = 1.0,
    diversity_rew_coef: float = 1.0,
) -> torch.Tensor:
    original_shape = diversity_rewards.shape

    if diversity_rewards.ndim == 3:
        diversity_rewards = diversity_rewards.reshape(
            diversity_rewards.shape[0],
            -1,
            n_samples_per_prompt,
            diversity_rewards.shape[-1],
        )
        gt_rewards = gt_rewards.reshape(gt_rewards.shape[0], -1, n_samples_per_prompt, gt_rewards.shape[-1])
    else:
        diversity_rewards = diversity_rewards.reshape(
            diversity_rewards.shape[0],
            -1,
            n_samples_per_prompt,
            diversity_rewards.shape[-2],
            diversity_rewards.shape[-1],
        )
        gt_rewards = gt_rewards.reshape(
            gt_rewards.shape[0],
            -1,
            n_samples_per_prompt,
            gt_rewards.shape[-2],
            gt_rewards.shape[-1],
        )

    if n_samples_per_prompt <= 1:
        return torch.zeros_like(diversity_rewards).reshape(original_shape)

    denom_loo = float(n_samples_per_prompt - 1)
    gt_baseline = (gt_rewards.sum(2, keepdim=True) - gt_rewards) / denom_loo

    if float(diversity_rew_coef) != 0.0 and n_samples_per_prompt > 2:
        denom_div = float(n_samples_per_prompt - 2)
        diversity_baseline = diversity_rewards.sum(2, keepdim=True) / denom_div - (
            2.0 * diversity_rewards
        ) / denom_div
    else:
        diversity_baseline = torch.zeros_like(diversity_rewards)

    baseline = float(alignment_rew_coef) * gt_baseline - float(diversity_rew_coef) * diversity_baseline
    return baseline.reshape(original_shape)


def compute_rloo_shaped_rewards(
    rewards: torch.Tensor,
    diversity_rewards: torch.Tensor,
    gt_rewards: torch.Tensor,
    *,
    n_samples_per_prompt: int,
    alignment_rew_coef: float = 1.0,
    diversity_rew_coef: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = compute_rloo_baseline(
        diversity_rewards,
        gt_rewards,
        n_samples_per_prompt=n_samples_per_prompt,
        alignment_rew_coef=alignment_rew_coef,
        diversity_rew_coef=diversity_rew_coef,
    )
    return rewards - baseline, baseline


def expand_block_rewards_to_token_advantages(
    shaped_rewards: torch.Tensor,
    *,
    generate_length: int,
    response_length: int | None = None,
) -> torch.Tensor:
    if shaped_rewards.ndim == 1:
        expanded = shaped_rewards.repeat(generate_length)
    elif shaped_rewards.ndim == 2:
        expanded = shaped_rewards.repeat(1, generate_length)
    elif shaped_rewards.ndim == 3:
        expanded = shaped_rewards.reshape(shaped_rewards.shape[0], -1)
    else:
        raise ValueError(f"Unsupported shaped reward rank: {shaped_rewards.ndim}")

    if response_length is not None and expanded.shape[-1] != response_length:
        raise ValueError(f"Expanded G1 advantages length {expanded.shape[-1]} != response_length {response_length}")
    return expanded
