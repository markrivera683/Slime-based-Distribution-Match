from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from slime.rollout.rm_hub.g1_core import compute_group_g1_rewards
from slime.rollout.g1_embedding import (
    G1EmbeddingConfig,
    build_g1_full_sequence_inputs,
    build_g1_prompt_inputs,
    hidden_states_to_g1_embeddings,
)
from slime.ray import rollout as rollout_module
from slime.ray.rollout import RolloutManager
from slime.backends.megatron_utils import loss as loss_module
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

    torch.testing.assert_close(full_sequence, torch.tensor([1, 2, 3, 4, 0, 0, 5, 6, 5, 6]))
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
    }

    refs = rollout_manager_class._split_train_data_by_dp(manager, data, dp_size=2)

    assert refs[0].inner["g1_token_advantages"] == [[0.1, 0.2], [0.5, 0.6]]
    assert refs[1].inner["g1_token_advantages"] == [[0.3, 0.4], [0.7, 0.8]]
