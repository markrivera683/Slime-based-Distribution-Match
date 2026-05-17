from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from slime.rollout.rm_hub.g1_core import compute_group_g1_rewards
from slime.rollout.g1_embedding import (
    G1EmbeddingConfig,
    assert_g1_critic_checkpoint_is_transformers_hf,
    build_g1_full_sequence_inputs,
    build_g1_prompt_inputs,
    hidden_states_to_g1_embeddings,
)
from slime.ray import rollout as rollout_module
from slime.ray.rollout import RolloutManager
from slime.backends.megatron_utils import data as data_module
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.g1_fast import (
    build_megatron_rotary_pos_emb_from_position_ids,
    build_openrlhf_g1_attention_mask_and_position_ids,
    compute_g1_token_advantages_from_embeddings,
    megatron_hidden_to_g1_embeddings,
    openrlhf_dense_mask_thd_attention,
    pack_openrlhf_g1_attention_mask,
)
from slime.utils.g1_core import (
    compute_pointwise_rewards,
    compute_rloo_baseline,
    compute_rloo_shaped_rewards,
    expand_block_rewards_to_token_advantages,
    get_num_strided_blocks,
    prepare_strided_token_blocks,
    whiten_embeddings_batched,
)
from slime.utils.types import Sample


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        table = {"a": 1, "b": 2, "c": 3, "d": 4, "x": 5, "y": 6}
        return [table[ch] for ch in str(text)]


def _fixed_openrlhf_block_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic block-level G1 fixture shaped [micro, group, N, blocks, hidden]."""
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


def _openrlhf_reference_whiten(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    *,
    whiten_tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    u, singular_values, _ = torch.linalg.svd(gen_flat, full_matrices=False)
    max_singular_value = singular_values.max(dim=-1, keepdim=True).values
    inv_s = torch.where(
        singular_values > whiten_tol * max_singular_value,
        1.0 / (singular_values + 1e-12),
        torch.zeros_like(singular_values),
    )
    whitening = (u * inv_s.unsqueeze(-2)) @ u.transpose(-1, -2)

    gen_whitened = (whitening @ gen_flat).to(dtype=gen_embedding.dtype)
    gt_whitened = (whitening @ gt_flat).to(dtype=gt_embedding.dtype)
    gen_whitened = gen_whitened.reshape(*batch_dims, n_samples, embed_dim).permute(*inv_perm).contiguous()
    gt_whitened = gt_whitened.reshape(*batch_dims, n_samples, embed_dim).permute(*inv_perm).contiguous()
    return gen_whitened, gt_whitened


def _openrlhf_reference_diversity(gen_embedding: torch.Tensor) -> torch.Tensor:
    micro_batches, groups, n_samples, num_blocks, _ = gen_embedding.shape
    diversity = torch.empty(
        micro_batches,
        groups,
        n_samples,
        num_blocks,
        dtype=gen_embedding.dtype,
        device=gen_embedding.device,
    )
    for micro_idx in range(micro_batches):
        for group_idx in range(groups):
            for sample_idx in range(n_samples):
                for block_idx in range(num_blocks):
                    other_sims = []
                    for other_idx in range(n_samples):
                        if other_idx == sample_idx:
                            continue
                        other_sims.append(
                            torch.sum(
                                gen_embedding[micro_idx, group_idx, sample_idx, block_idx]
                                * gen_embedding[micro_idx, group_idx, other_idx, block_idx]
                            )
                        )
                    diversity[micro_idx, group_idx, sample_idx, block_idx] = torch.stack(other_sims).mean()
    return diversity


def _openrlhf_reference_pointwise_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    *,
    alignment_rew_coef: float = 1.0,
    diversity_rew_coef: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alignment = F.cosine_similarity(gen_embedding, gt_embedding, dim=-1)
    diversity = _openrlhf_reference_diversity(gen_embedding)

    alignment = alignment.reshape(alignment.shape[0], -1, alignment.shape[-1]) * 2
    diversity = diversity.reshape(diversity.shape[0], -1, diversity.shape[-1]) * 2
    rewards = float(alignment_rew_coef) * alignment - float(diversity_rew_coef) * diversity
    return rewards, alignment, diversity


def _openrlhf_reference_rloo_baseline(
    diversity_rewards: torch.Tensor,
    gt_rewards: torch.Tensor,
    *,
    n_samples_per_prompt: int,
    alignment_rew_coef: float = 1.0,
    diversity_rew_coef: float = 1.0,
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


def _assert_fixed_embedding_parity(*, use_whitening: bool) -> None:
    gen_embedding, gt_embedding = _fixed_openrlhf_block_embeddings()
    reference_gen = gen_embedding
    reference_gt = gt_embedding

    if use_whitening:
        reference_gen, reference_gt = _openrlhf_reference_whiten(reference_gen, reference_gt)
        actual_gen, actual_gt = whiten_embeddings_batched(gen_embedding, gt_embedding, normalize=False)
        torch.testing.assert_close(actual_gen, reference_gen, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_gt, reference_gt, rtol=1e-5, atol=1e-5)

    expected_rewards, expected_gt_rewards, expected_diversity_rewards = _openrlhf_reference_pointwise_rewards(
        reference_gen,
        reference_gt,
    )
    rewards, gt_rewards, diversity_rewards = compute_pointwise_rewards(
        gen_embedding,
        gt_embedding,
        use_whitening=use_whitening,
    )
    torch.testing.assert_close(rewards, expected_rewards, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gt_rewards, expected_gt_rewards, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(diversity_rewards, expected_diversity_rewards, rtol=1e-5, atol=1e-5)

    expected_baseline = _openrlhf_reference_rloo_baseline(
        expected_diversity_rewards,
        expected_gt_rewards,
        n_samples_per_prompt=4,
    )
    shaped_rewards, baseline = compute_rloo_shaped_rewards(
        rewards,
        diversity_rewards,
        gt_rewards,
        n_samples_per_prompt=4,
    )
    torch.testing.assert_close(baseline, expected_baseline, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(shaped_rewards, expected_rewards - expected_baseline, rtol=1e-5, atol=1e-5)

    generate_length = 2
    token_advantages = torch.stack(
        [
            expand_block_rewards_to_token_advantages(
                sample_rewards,
                generate_length=generate_length,
                response_length=sample_rewards.numel() * generate_length,
            )
            for sample_rewards in shaped_rewards.squeeze(0)
        ]
    )
    expected_token_advantages = torch.stack(
        [sample_rewards.repeat(generate_length) for sample_rewards in (expected_rewards - expected_baseline).squeeze(0)]
    )
    torch.testing.assert_close(token_advantages, expected_token_advantages, rtol=1e-5, atol=1e-5)


def _assert_megatron_g1_rollout_contract(train_data: dict, samples: list[Sample], args: Namespace) -> None:
    expected_sequence_len = int(args.g1_prompt_length) + int(args.g1_response_length)
    assert train_data["response_lengths"] == [int(args.g1_response_length)] * len(
        samples
    ), "rollout fixed response length contract changed before trainer-side G1"

    assert len(train_data["g1_full_sequences"]) == len(samples), "rollout omitted g1_full_sequences for a sample"
    assert len(train_data["g1_qa_masks"]) == len(samples), "rollout omitted g1_qa_masks for a sample"
    for idx, (sample, full_sequence, qa_mask) in enumerate(
        zip(samples, train_data["g1_full_sequences"], train_data["g1_qa_masks"], strict=True)
    ):
        assert len(full_sequence) == expected_sequence_len, (
            f"rollout g1_full_sequences length mismatch at sample {idx}: "
            f"{len(full_sequence)} != {expected_sequence_len}"
        )
        assert len(qa_mask) == expected_sequence_len, (
            f"rollout g1_qa_masks length mismatch at sample {idx}: {len(qa_mask)} != {expected_sequence_len}"
        )
        assert full_sequence[-int(args.g1_response_length) :] == sample.tokens[-int(args.g1_response_length) :], (
            f"rollout response token tail mismatch at sample {idx}"
        )
        assert qa_mask[-int(args.g1_response_length) :] == [1] * int(args.g1_response_length), (
            f"rollout g1_qa_masks response tail is not all action tokens at sample {idx}"
        )


def _assert_dp_split_keeps_prompt_groups_together(refs: list, *, n_samples_per_prompt: int) -> None:
    group_to_rank = {}
    for rank, ref in enumerate(refs):
        sample_indices = ref.inner["sample_indices"]
        assert len(sample_indices) % n_samples_per_prompt == 0, (
            f"DP split produced a partial G1 prompt group on rank {rank}: {sample_indices}"
        )
        for offset in range(0, len(sample_indices), n_samples_per_prompt):
            chunk = sample_indices[offset : offset + n_samples_per_prompt]
            group_ids = {idx // n_samples_per_prompt for idx in chunk}
            assert len(group_ids) == 1, f"DP split broke a prompt group on rank {rank}: {chunk}"
            group_id = group_ids.pop()
            assert group_id not in group_to_rank, (
                f"DP split sent prompt group {group_id} to both rank {group_to_rank[group_id]} and rank {rank}"
            )
            group_to_rank[group_id] = rank


def test_get_num_strided_blocks_matches_diff_dataset_g1_geometry():
    assert get_num_strided_blocks(prompt_length=384, context_length=8, generate_length=8, stride=8) == 47


def test_prepare_strided_token_blocks_matches_openrlhf_layout():
    prompts = [torch.arange(0, 16).reshape(2, 8)]
    full_sequences = [torch.arange(0, 32).reshape(2, 16)]

    gen_blocks, gt_blocks = prepare_strided_token_blocks(
        prompts,
        full_sequences,
        prompt_length=8,
        stride=2,
        num_blocks=2,
        n_samples_per_prompt=2,
        context_length=2,
        generate_length=4,
    )

    assert gen_blocks.shape == (1, 1, 2, 2, 4)
    assert gt_blocks.shape == (1, 1, 2, 2, 4)
    torch.testing.assert_close(gen_blocks[0, 0, 0], torch.tensor([[8, 10, 12, 14], [9, 11, 13, 15]]))
    torch.testing.assert_close(gt_blocks[0, 0, 0], torch.tensor([[2, 3, 4, 5], [4, 5, 6, 7]]))


def test_g1_prompt_pack_contract_marks_answer_and_padding():
    config = G1EmbeddingConfig(prompt_length=6, context_length=2, generate_length=2, stride=2, response_length=4)
    sample = Sample(prompt="ab", label="cd")

    prompt_ids, doc_ids, qa_mask = build_g1_prompt_inputs(
        tokenizer=_FakeTokenizer(),
        sample=sample,
        config=config,
    )

    torch.testing.assert_close(prompt_ids, torch.tensor([1, 2, 3, 4, 0, 0]))
    torch.testing.assert_close(doc_ids, torch.tensor([0, 0, 0, 0, -1, -1]))
    torch.testing.assert_close(qa_mask, torch.tensor([0, 0, 1, 1, 0, 0]))


def test_g1_full_sequence_contract_requires_fixed_response_length():
    config = G1EmbeddingConfig(prompt_length=6, context_length=2, generate_length=2, stride=2, response_length=4)
    sample = Sample(prompt="ab", label="cd", tokens=[1, 2, 5, 6, 5, 6], response_length=4)

    full_sequence, doc_ids, qa_mask = build_g1_full_sequence_inputs(
        tokenizer=_FakeTokenizer(),
        sample=sample,
        config=config,
    )

    assert full_sequence.numel() == config.prompt_length + config.response_length
    assert qa_mask.numel() == config.prompt_length + config.response_length
    torch.testing.assert_close(full_sequence, torch.tensor([1, 2, 3, 4, 0, 0, 5, 6, 5, 6]))
    torch.testing.assert_close(full_sequence[-config.response_length :], torch.tensor(sample.tokens[-config.response_length :]))
    torch.testing.assert_close(doc_ids, torch.tensor([0, 0, 0, 0, -1, -1]))
    torch.testing.assert_close(qa_mask, torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]))

    sample.response_length = 3
    with pytest.raises(ValueError, match="response_length=4"):
        build_g1_full_sequence_inputs(tokenizer=_FakeTokenizer(), sample=sample, config=config)


def test_hidden_states_to_g1_embeddings_matches_openrlhf_block_order():
    config = G1EmbeddingConfig(prompt_length=8, context_length=2, generate_length=2, stride=2, response_length=6)
    hidden_states = torch.arange(14, dtype=torch.float32).reshape(1, 14, 1, 1)
    qa_masks = torch.ones(1, 14, dtype=torch.long)

    gen_embedding, gt_embedding = hidden_states_to_g1_embeddings(hidden_states, qa_masks, config)

    torch.testing.assert_close(gt_embedding, torch.tensor([[[3.0], [5.0], [7.0]]]))
    torch.testing.assert_close(gen_embedding, torch.tensor([[[11.0], [12.0], [13.0]]]))


def test_megatron_hidden_to_g1_embeddings_matches_shared_block_helper(monkeypatch):
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    args = Namespace(
        qkv_format="thd",
        g1_prompt_length=8,
        g1_context_length=2,
        g1_generate_length=2,
        g1_stride=2,
        g1_response_length=6,
        g1_hidden_state_method="last_only",
        g1_qa_masking=False,
        g1_document_masking=False,
        n_samples_per_prompt=1,
    )
    hidden_seq = torch.arange(28, dtype=torch.float32).reshape(14, 2)
    hidden_states = hidden_seq.unsqueeze(1)
    qa_mask = torch.ones(14, dtype=torch.long)

    gen_embedding, gt_embedding = megatron_hidden_to_g1_embeddings(
        hidden_states,
        args=args,
        total_lengths=[14],
        g1_qa_masks=[qa_mask],
    )
    expected_gen, expected_gt = hidden_states_to_g1_embeddings(
        F.normalize(hidden_seq, p=2, dim=-1).reshape(1, 14, 1, 2),
        qa_mask.reshape(1, 14),
        G1EmbeddingConfig(prompt_length=8, context_length=2, generate_length=2, stride=2, response_length=6),
    )

    torch.testing.assert_close(gen_embedding[0], expected_gen[0])
    torch.testing.assert_close(gt_embedding[0], expected_gt[0])


def test_rloo_baseline_matches_openrlhf_formula_for_pointwise_rewards():
    gt_rewards = torch.tensor([[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]]])
    diversity_rewards = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

    baseline = compute_rloo_baseline(
        diversity_rewards,
        gt_rewards,
        n_samples_per_prompt=4,
        alignment_rew_coef=1.0,
        diversity_rew_coef=1.0,
    )

    reshaped_gt = gt_rewards.reshape(1, 1, 4, 2)
    reshaped_div = diversity_rewards.reshape(1, 1, 4, 2)
    expected_gt = (reshaped_gt.sum(2, keepdim=True) - reshaped_gt) / 3.0
    expected_div = reshaped_div.sum(2, keepdim=True) / 2.0 - 2.0 * reshaped_div / 2.0
    expected = (expected_gt - expected_div).reshape_as(gt_rewards)

    torch.testing.assert_close(baseline, expected)


def test_expand_block_rewards_keeps_openrlhf_generation_step_order():
    block_rewards = torch.tensor([1.0, 2.0, 3.0])

    advantages = expand_block_rewards_to_token_advantages(
        block_rewards,
        generate_length=2,
        response_length=6,
    )

    torch.testing.assert_close(advantages, torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))


def test_fixed_embedding_fixture_matches_openrlhf_pointwise_reference_without_whitening():
    _assert_fixed_embedding_parity(use_whitening=False)


def test_fixed_embedding_fixture_matches_openrlhf_pointwise_reference_with_whitening():
    _assert_fixed_embedding_parity(use_whitening=True)


def test_group_g1_reward_matches_fixed_embedding_reference_metadata():
    gen_embedding, gt_embedding = _fixed_openrlhf_block_embeddings()
    reference_gen, reference_gt = _openrlhf_reference_whiten(gen_embedding, gt_embedding)
    expected_rewards, expected_gt_rewards, expected_diversity_rewards = _openrlhf_reference_pointwise_rewards(
        reference_gen,
        reference_gt,
    )
    expected_baseline = _openrlhf_reference_rloo_baseline(
        expected_diversity_rewards,
        expected_gt_rewards,
        n_samples_per_prompt=4,
    )
    expected_shaped_rewards = expected_rewards - expected_baseline

    generate_length = 2
    args = Namespace(
        n_samples_per_prompt=4,
        alignment_rew_coef=1.0,
        diversity_rew_coef=1.0,
        use_whitening=True,
    )
    samples = [
        Sample(
            response_length=expected_rewards.shape[-1] * generate_length,
            metadata={
                "g1_gen_embedding": gen_embedding[0, 0, idx].tolist(),
                "g1_gt_embedding": gt_embedding[0, 0, idx].tolist(),
            },
        )
        for idx in range(4)
    ]

    scalar_rewards = compute_group_g1_rewards(args, samples)

    for idx, sample in enumerate(samples):
        torch.testing.assert_close(torch.tensor(sample.metadata["g1_rewards"]), expected_rewards[0, idx])
        torch.testing.assert_close(torch.tensor(sample.metadata["g1_gt_rewards"]), expected_gt_rewards[0, idx])
        torch.testing.assert_close(
            torch.tensor(sample.metadata["g1_diversity_rewards"]),
            expected_diversity_rewards[0, idx],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(torch.tensor(sample.metadata["g1_rloo_baseline"]), expected_baseline[0, idx])
        torch.testing.assert_close(
            torch.tensor(sample.metadata["g1_token_advantages"]),
            expected_shaped_rewards[0, idx].repeat(generate_length),
            rtol=1e-5,
            atol=1e-5,
        )
        assert scalar_rewards[idx] == pytest.approx(float(expected_rewards[0, idx].mean().item()))


def test_trainer_side_g1_advantages_match_group_rm_math():
    gen_embedding, gt_embedding = _fixed_openrlhf_block_embeddings()
    args = Namespace(
        n_samples_per_prompt=4,
        alignment_rew_coef=1.0,
        diversity_rew_coef=1.0,
        use_whitening=True,
        g1_response_length=6,
    )
    token_advantages, scalar_rewards = compute_g1_token_advantages_from_embeddings(
        args,
        [gen_embedding[0, 0, idx] for idx in range(4)],
        [gt_embedding[0, 0, idx] for idx in range(4)],
        [6, 6, 6, 6],
    )
    samples = [
        Sample(
            response_length=6,
            metadata={
                "g1_gen_embedding": gen_embedding[0, 0, idx].tolist(),
                "g1_gt_embedding": gt_embedding[0, 0, idx].tolist(),
            },
        )
        for idx in range(4)
    ]
    expected_scalar_rewards = compute_group_g1_rewards(args, samples)

    assert scalar_rewards == pytest.approx(expected_scalar_rewards)
    for idx, sample in enumerate(samples):
        torch.testing.assert_close(token_advantages[idx], torch.tensor(sample.metadata["g1_token_advantages"]))


def test_group_g1_reward_populates_token_advantages_from_metadata_embeddings():
    args = Namespace(
        n_samples_per_prompt=4,
        alignment_rew_coef=1.0,
        diversity_rew_coef=1.0,
        use_whitening=False,
    )
    samples = []
    for idx in range(4):
        sample = Sample(
            response_length=4,
            metadata={
                "g1_gen_embedding": [[1.0 + idx, 0.0], [0.0, 1.0 + idx]],
                "g1_gt_embedding": [[1.0, 0.0], [0.0, 1.0]],
            },
        )
        samples.append(sample)

    rewards = compute_group_g1_rewards(args, samples)

    assert len(rewards) == 4
    for sample in samples:
        assert "g1_rewards" in sample.metadata
        assert "g1_gt_rewards" in sample.metadata
        assert "g1_diversity_rewards" in sample.metadata
        assert "g1_rloo_baseline" in sample.metadata
        assert len(sample.metadata["g1_token_advantages"]) == sample.response_length


def test_default_train_data_conversion_preserves_g1_token_advantages():
    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = Namespace(
        reward_key=None,
        advantage_estimator="g1",
        rewards_normalization=False,
    )
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None

    samples = [
        Sample(
            index=0,
            tokens=[1, 2, 3],
            response_length=2,
            reward=0.5,
            metadata={"g1_token_advantages": [0.1, 0.2]},
            status=Sample.Status.COMPLETED,
        ),
        Sample(
            index=1,
            tokens=[1, 4, 5],
            response_length=2,
            reward=-0.5,
            metadata={"g1_token_advantages": [-0.1, -0.2]},
            status=Sample.Status.COMPLETED,
        ),
    ]

    train_data = rollout_manager_class._convert_samples_to_train_data(manager, samples)

    assert train_data["g1_token_advantages"] == [[0.1, 0.2], [-0.1, -0.2]]
    assert train_data["loss_masks"] == [[1, 1], [1, 1]]


def test_train_data_conversion_prepares_megatron_g1_sequences_without_rollout_advantages(monkeypatch):
    from slime.utils import processing_utils

    monkeypatch.setattr(processing_utils, "load_tokenizer", lambda *args, **kwargs: _FakeTokenizer())
    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = Namespace(
        reward_key=None,
        advantage_estimator="g1",
        rewards_normalization=False,
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        g1_tokenizer_path="unused",
        hf_checkpoint="unused",
        g1_prompt_length=6,
        g1_context_length=2,
        g1_generate_length=2,
        g1_stride=2,
        g1_response_length=4,
        n_samples_per_prompt=2,
        g1_openrlhf_repo="/unused",
        g1_hidden_state_method="last_only",
        g1_embedding_device="cuda",
        g1_embedding_dtype="bfloat16",
        g1_qa_masking=False,
        g1_document_masking=False,
    )
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    samples = [
        Sample(index=0, prompt="ab", label="cd", tokens=[1, 2, 5, 6, 5, 6], response_length=4, reward=0.0),
        Sample(index=1, prompt="ab", label="cd", tokens=[1, 2, 6, 5, 6, 5], response_length=4, reward=0.0),
    ]

    train_data = rollout_manager_class._convert_samples_to_train_data(manager, samples)

    assert "g1_token_advantages" not in train_data
    assert train_data["g1_full_sequences"][0] == [1, 2, 3, 4, 0, 0, 5, 6, 5, 6]
    assert train_data["g1_qa_masks"][0] == [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    _assert_megatron_g1_rollout_contract(train_data, samples, manager.args)


def test_g1_advantage_estimator_consumes_precomputed_token_advantages(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    args = Namespace(
        use_rollout_logprobs=False,
        kl_coef=0.0,
        advantage_estimator="g1",
        use_opd=False,
        normalize_advantages=False,
    )
    rollout_data = {
        "log_probs": [torch.zeros(3), torch.zeros(3)],
        "rewards": [0.0, 0.0],
        "g1_token_advantages": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1.0, -2.0, -3.0])],
        "response_lengths": [3, 3],
        "total_lengths": [5, 5],
        "loss_masks": [torch.ones(3), torch.ones(3)],
    }

    loss_module.compute_advantages_and_returns(args, rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(rollout_data["advantages"][1], torch.tensor([-1.0, -2.0, -3.0]))
    torch.testing.assert_close(rollout_data["returns"][0], rollout_data["advantages"][0])


def test_g1_advantage_estimator_rejects_response_level_shape_mismatch(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)

    rollout_data = {
        "log_probs": [torch.zeros(4)],
        "rewards": [0.0],
        "g1_token_advantages": [torch.tensor([1.0, 2.0, 3.0])],
        "response_lengths": [4],
        "total_lengths": [6],
        "loss_masks": [torch.ones(4)],
    }

    with pytest.raises(ValueError, match="G1 token advantage length 3 != response_length 4 at sample 0"):
        loss_module.compute_advantages_and_returns(
            Namespace(
                use_rollout_logprobs=False,
                kl_coef=0.0,
                advantage_estimator="g1",
                use_opd=False,
                normalize_advantages=False,
            ),
            rollout_data,
        )


def test_group_g1_reward_rejects_response_lengths_not_aligned_to_blocks():
    args = Namespace(
        n_samples_per_prompt=2,
        alignment_rew_coef=1.0,
        diversity_rew_coef=1.0,
        use_whitening=False,
    )
    samples = [
        Sample(
            response_length=3,
            metadata={
                "g1_gen_embedding": [[1.0, 0.0], [0.0, 1.0]],
                "g1_gt_embedding": [[1.0, 0.0], [0.0, 1.0]],
            },
        ),
        Sample(
            response_length=3,
            metadata={
                "g1_gen_embedding": [[0.0, 1.0], [1.0, 0.0]],
                "g1_gt_embedding": [[1.0, 0.0], [0.0, 1.0]],
            },
        ),
    ]

    with pytest.raises(ValueError, match="not divisible"):
        compute_group_g1_rewards(args, samples)


def test_group_g1_reward_rejects_configured_response_length_mismatch():
    args = Namespace(
        n_samples_per_prompt=2,
        alignment_rew_coef=1.0,
        diversity_rew_coef=1.0,
        use_whitening=False,
        g1_response_length=4,
    )
    samples = [
        Sample(
            response_length=2,
            metadata={
                "g1_gen_embedding": [[1.0, 0.0], [0.0, 1.0]],
                "g1_gt_embedding": [[1.0, 0.0], [0.0, 1.0]],
            },
        ),
        Sample(
            response_length=2,
            metadata={
                "g1_gen_embedding": [[0.0, 1.0], [1.0, 0.0]],
                "g1_gt_embedding": [[1.0, 0.0], [0.0, 1.0]],
            },
        ),
    ]

    with pytest.raises(ValueError, match="response_length"):
        compute_group_g1_rewards(args, samples)


def test_g1_train_data_conversion_requires_all_token_advantages():
    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = Namespace(
        reward_key=None,
        advantage_estimator="g1",
        rewards_normalization=False,
    )
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    samples = [
        Sample(
            index=0,
            tokens=[1, 2, 3],
            response_length=2,
            reward=0.0,
            metadata={"g1_token_advantages": [0.1, 0.2]},
            status=Sample.Status.COMPLETED,
        ),
        Sample(
            index=1,
            tokens=[1, 4, 5],
            response_length=2,
            reward=0.0,
            metadata={},
            status=Sample.Status.COMPLETED,
        ),
    ]

    with pytest.raises(ValueError, match="every sample"):
        rollout_manager_class._convert_samples_to_train_data(manager, samples)


def test_split_train_data_by_dp_preserves_g1_token_advantages(monkeypatch):
    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = Namespace(balance_data=False)
    monkeypatch.setattr(rollout_module.ray, "put", lambda value: value)

    data = {
        "tokens": [[1, 2, 3], [1, 4, 5], [1, 6, 7], [1, 8, 9]],
        "response_lengths": [2, 2, 2, 2],
        "rewards": [0.0, 0.1, 0.2, 0.3],
        "truncated": [False, False, False, False],
        "loss_masks": [[1, 1], [1, 1], [1, 1], [1, 1]],
        "sample_indices": [0, 1, 2, 3],
        "g1_token_advantages": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
        "g1_full_sequences": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "g1_qa_masks": [[1, 1], [1, 1], [1, 1], [1, 1]],
    }

    refs = rollout_manager_class._split_train_data_by_dp(manager, data, dp_size=2)

    assert refs[0].inner["g1_token_advantages"] == [[0.1, 0.2], [0.5, 0.6]]
    assert refs[1].inner["g1_token_advantages"] == [[0.3, 0.4], [0.7, 0.8]]
    assert refs[0].inner["g1_full_sequences"] == [[1, 2], [5, 6]]
    assert refs[1].inner["g1_qa_masks"] == [[1, 1], [1, 1]]


def test_split_train_data_by_dp_keeps_megatron_g1_prompt_groups_together(monkeypatch):
    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = Namespace(
        advantage_estimator="g1",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        n_samples_per_prompt=4,
        balance_data=False,
    )
    monkeypatch.setattr(rollout_module.ray, "put", lambda value: value)

    data = {
        "tokens": [[idx] for idx in range(16)],
        "response_lengths": [2] * 16,
        "rewards": [0.0] * 16,
        "truncated": [False] * 16,
        "loss_masks": [[1, 1] for _ in range(16)],
        "sample_indices": list(range(16)),
        "g1_full_sequences": [[idx, idx] for idx in range(16)],
        "g1_qa_masks": [[1, 1] for _ in range(16)],
    }

    refs = rollout_manager_class._split_train_data_by_dp(manager, data, dp_size=2)

    assert refs[0].inner["sample_indices"] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert refs[1].inner["sample_indices"] == [4, 5, 6, 7, 12, 13, 14, 15]
    _assert_dp_split_keeps_prompt_groups_together(refs, n_samples_per_prompt=4)


def test_trainer_side_g1_keeps_sequences_when_ebft_enabled():
    actor_source = (Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/actor.py").read_text(
        encoding="utf-8"
    )
    cleanup_guard = 'if not bool(getattr(self.args, "g1_use_ebft_loss", False)):'
    guard_idx = actor_source.index(cleanup_guard)
    cleanup_block = actor_source[guard_idx : guard_idx + 260]

    assert 'rollout_data.pop("g1_full_sequences", None)' in cleanup_block
    assert 'rollout_data.pop("g1_qa_masks", None)' in cleanup_block


def test_split_train_data_by_dp_rejects_megatron_g1_balance_data(monkeypatch):
    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = Namespace(
        advantage_estimator="g1",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        n_samples_per_prompt=4,
        balance_data=True,
    )
    monkeypatch.setattr(rollout_module.ray, "put", lambda value: value)
    data = {
        "tokens": [[idx] for idx in range(8)],
        "response_lengths": [2] * 8,
        "rewards": [0.0] * 8,
        "truncated": [False] * 8,
        "loss_masks": [[1, 1] for _ in range(8)],
        "g1_full_sequences": [[idx, idx] for idx in range(8)],
        "g1_qa_masks": [[1, 1] for _ in range(8)],
    }

    with pytest.raises(ValueError, match="group-aligned DP split"):
        rollout_manager_class._split_train_data_by_dp(manager, data, dp_size=2)


def test_split_train_data_by_dp_rejects_megatron_g1_uneven_group_count(monkeypatch):
    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = Namespace(
        advantage_estimator="g1",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        n_samples_per_prompt=4,
        balance_data=False,
    )
    monkeypatch.setattr(rollout_module.ray, "put", lambda value: value)
    data = {
        "tokens": [[idx] for idx in range(12)],
        "response_lengths": [2] * 12,
        "rewards": [0.0] * 12,
        "truncated": [False] * 12,
        "loss_masks": [[1, 1] for _ in range(12)],
        "g1_full_sequences": [[idx, idx] for idx in range(12)],
        "g1_qa_masks": [[1, 1] for _ in range(12)],
    }

    with pytest.raises(ValueError, match="must be divisible by dp_size"):
        rollout_manager_class._split_train_data_by_dp(manager, data, dp_size=2)


def test_g1_megatron_ref_trainer_rejects_dynamic_batch_size():
    """Trainer-side Megatron G1 reshapes embeddings by contiguous prompt groups."""
    from slime.utils.arguments import assert_g1_megatron_ref_trainer_stable_microbatch_order

    incompatible = Namespace(
        advantage_estimator="g1",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        use_dynamic_batch_size=True,
    )
    with pytest.raises(ValueError, match="use_dynamic_batch_size"):
        assert_g1_megatron_ref_trainer_stable_microbatch_order(incompatible)

    fixed = Namespace(
        advantage_estimator="g1",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        use_dynamic_batch_size=False,
    )
    assert_g1_megatron_ref_trainer_stable_microbatch_order(fixed)


def test_log_rollout_data_skips_g1_integer_payloads(monkeypatch):
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module, "gather_log_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(data_module, "log_multi_turn_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_module, "log_passrate", lambda *args, **kwargs: None)

    rollout_data = {
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [4],
        "rewards": [1.0],
        "g1_full_sequences": [torch.tensor([1, 2, 3, 4], dtype=torch.long)],
        "g1_qa_masks": [torch.tensor([0, 1, 1, 1], dtype=torch.long)],
    }

    data_module.log_rollout_data(
        0,
        Namespace(
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
            ci_test=False,
        ),
        rollout_data,
    )


def test_append_g1_runtime_dump_accumulates_dump_sample_count(tmp_path):
    from slime.backends.megatron_utils.loss import _append_g1_runtime_dump

    meta = {"data_parallel_rank": 0, "tensor_model_parallel_rank": 0}

    def _micro_payload(tokens_len: int) -> dict:
        t = torch.arange(tokens_len, dtype=torch.long)
        masks = torch.ones(tokens_len, dtype=torch.long)
        embed = torch.zeros((1, 4, 8), dtype=torch.float32)
        return {
            "total_lengths": [int(tokens_len)],
            "tokens": [t.clone()],
            "g1_qa_masks": [masks],
            "g1_gen_embedding": [embed.clone()],
            "g1_gt_embedding": [embed.clone()],
            "hidden_states_post_sp_gather": torch.zeros(int(tokens_len), 8),
            "g1_dump_writer_metadata": meta,
        }

    path = tmp_path / "dump.pt"
    _append_g1_runtime_dump(path, _micro_payload(6))
    first = torch.load(path, map_location="cpu", weights_only=False)
    assert first["dump_sample_count_total"] == 1
    assert len(first["total_lengths"]) == 1
    assert first["g1_dump_writer_metadata"]["data_parallel_rank"] == 0

    _append_g1_runtime_dump(path, _micro_payload(10))
    merged = torch.load(path, map_location="cpu", weights_only=False)
    assert merged["dump_sample_count_total"] == 2
    assert merged["total_lengths"] == [6, 10]
    assert merged["hidden_states_post_sp_gather"].shape[0] == 16


def test_megatron_ref_smoke_script_uses_trainer_side_g1_path():
    script = "/mnt/data/distribution-matching-slime/code/slime-0.2.4/refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh"
    with open(script, encoding="utf-8") as f:
        content = f.read()

    assert "slime.rollout.g1_embedding.generate_fixed_length_for_g1" in content
    assert "--g1-embedding-source megatron_ref" in content
    assert "--g1-reward-location trainer" in content
    assert "--g1-critic-model-path" not in content
    assert "GROUP_RM=false" in content
    assert "PRINT_ONLY" in content
    assert "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} must be divisible by DP_SIZE" in content
    assert "CONTEXT_PARALLEL_SIZE=1" in content


def test_openrlhf_g1_position_ids_match_expected_strided_layout():
    args = Namespace(
        g1_prompt_length=6,
        g1_context_length=2,
        g1_generate_length=2,
        g1_stride=2,
        g1_response_length=4,
        n_samples_per_prompt=2,
        g1_hidden_state_method="last_only",
        g1_qa_masking=False,
        g1_document_masking=False,
    )

    attention_mask, position_ids = build_openrlhf_g1_attention_mask_and_position_ids(
        args,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2, 3, 4, 5, 2, 4, 3, 5]]))
    assert tuple(attention_mask.shape) == (1, 1, 10, 10)
    assert attention_mask[0, 0, 6, 0].item() == 0.0
    assert attention_mask[0, 0, 6, 1].item() == 0.0
    assert attention_mask[0, 0, 6, 2].item() < -1e20
    assert attention_mask[0, 0, 8, 6].item() == 0.0
    assert attention_mask[0, 0, 8, 7].item() < -1e20


def test_custom_megatron_rope_uses_non_monotonic_openrlhf_positions():
    class _Rotary:
        rotary_interleaved = False
        seq_len_interpolation_factor = None

        def __init__(self):
            self.inv_freq = torch.tensor([1.0, 0.5])

    position_ids = torch.tensor([[0, 1, 2, 2, 4, 3]])
    custom = build_megatron_rotary_pos_emb_from_position_ids(_Rotary(), position_ids).squeeze(1).squeeze(1)
    default = build_megatron_rotary_pos_emb_from_position_ids(
        _Rotary(),
        torch.arange(position_ids.numel()).view_as(position_ids),
    ).squeeze(1).squeeze(1)

    torch.testing.assert_close(custom[2], custom[3])
    assert not torch.equal(custom, default)


def test_pack_openrlhf_g1_attention_mask_preserves_sample_blocks():
    min_value = torch.finfo(torch.float32).min
    mask = torch.full((2, 1, 3, 3), min_value)
    mask[0, 0, 0, 0] = 0.0
    mask[0, 0, 1, :2] = 0.0
    mask[1, 0, 0, 0] = 0.0
    mask[1, 0, 1, :2] = 0.0
    mask[1, 0, 2, :3] = 0.0

    packed = pack_openrlhf_g1_attention_mask(mask, total_lengths=[2, 3])

    assert tuple(packed.shape) == (1, 1, 5, 5)
    torch.testing.assert_close(packed[:, :, :2, :2], mask[:1, :, :2, :2])
    torch.testing.assert_close(packed[:, :, 2:5, 2:5], mask[1:2, :, :3, :3])
    assert packed[0, 0, 1, 2].item() == min_value
    assert packed[0, 0, 4, 1].item() == min_value


def test_openrlhf_dense_mask_thd_attention_applies_arbitrary_mask():
    query = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    key = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    value = torch.tensor([[[10.0, 0.0]], [[20.0, 0.0]]])
    min_value = torch.finfo(torch.float32).min
    mask = torch.tensor([[[[0.0, min_value], [min_value, 0.0]]]])

    output = openrlhf_dense_mask_thd_attention(query, key, value, mask, softmax_scale=1.0)

    torch.testing.assert_close(output[:, 0], torch.tensor([[10.0, 0.0], [20.0, 0.0]]))


def test_assert_g1_critic_checkpoint_rejects_sglang_auto_map(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"auto_map": {"AutoConfig": "sglang.srt.configs.qwen3_5.Qwen3_5Config"}, "model_type": "qwen3_5"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SGLang-patched"):
        assert_g1_critic_checkpoint_is_transformers_hf(str(tmp_path))


def test_assert_g1_critic_checkpoint_rejects_transformers_unknown_model_type(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SGLang config class|AutoConfig/AutoModelForCausalLM"):
        assert_g1_critic_checkpoint_is_transformers_hf(str(tmp_path))


def test_assert_g1_critic_checkpoint_accepts_hf_native_config(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}',
        encoding="utf-8",
    )
    assert_g1_critic_checkpoint_is_transformers_hf(str(tmp_path))


def test_assert_g1_critic_checkpoint_missing_config_json(tmp_path):
    with pytest.raises(ValueError, match="config.json"):
        assert_g1_critic_checkpoint_is_transformers_hf(str(tmp_path))
