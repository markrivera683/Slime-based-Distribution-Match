"""CPU-only smoke coverage for the strict G1 EBFT training batch path."""

from argparse import Namespace

import pytest
import torch

from slime.backends.megatron_utils import data as data_module
from slime.ray import rollout as rollout_module
from slime.ray.rollout import RolloutManager
from slime.utils.g1_ebft_data_contract import attach_ebft_g1_next_token_contract_to_batch
from slime.utils.types import Sample


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        table = {"a": 1, "b": 2, "c": 3, "d": 4, "x": 5, "y": 6}
        return [table[ch] for ch in str(text)]


def _strict_smoke_args() -> Namespace:
    return Namespace(
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
        n_samples_per_prompt=1,
        g1_openrlhf_repo="/unused",
        g1_hidden_state_method="last_only",
        g1_embedding_device="cuda",
        g1_embedding_dtype="bfloat16",
        g1_qa_masking=True,
        g1_document_masking=False,
        g1_use_ebft_loss=True,
        g1_ebft_logprob_indexing="strict_block_source",
        qkv_format="thd",
        allgather_cp=False,
        loss_type="policy_loss",
        use_opsm=False,
        entropy_coef=0.0,
        g1_ce_loss_coef=0.0,
        rollout_temperature=1.0,
        log_probs_chunk_size=0,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        use_opd=False,
        normalize_advantages=False,
    )


def _tensorize_train_data(train_data: dict) -> None:
    for key in ("tokens", "loss_masks", "g1_full_sequences", "g1_qa_masks", "g1_token_advantages"):
        train_data[key] = [torch.as_tensor(value) for value in train_data[key]]


def test_strict_ebft_train_data_iterator_contract_and_loss_use_block_source_rows(monkeypatch) -> None:
    megatron_loss = pytest.importorskip(
        "slime.backends.megatron_utils.loss",
        reason="Megatron dependency is not available in this environment.",
    )
    from slime.utils import processing_utils

    monkeypatch.setattr(processing_utils, "load_tokenizer", lambda *args, **kwargs: _TinyTokenizer())
    monkeypatch.setattr(rollout_module.ray, "put", lambda value: value)
    monkeypatch.setattr(megatron_loss.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(megatron_loss.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(megatron_loss.mpu, "get_tensor_model_parallel_group", lambda: None)

    def fake_calc(logits_chunk, tokens_targets, tp_group, with_entropy=False, chunk_size=None):
        del tp_group, with_entropy, chunk_size
        return logits_chunk.gather(dim=-1, index=tokens_targets.reshape(-1, 1)).reshape(-1, 1), None

    monkeypatch.setattr(megatron_loss, "calculate_log_probs_and_entropy", fake_calc)

    rollout_manager_class = getattr(getattr(RolloutManager, "__ray_metadata__", None), "modified_class", RolloutManager)
    manager = object.__new__(rollout_manager_class)
    manager.args = _strict_smoke_args()
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None

    samples = [
        Sample(
            index=0,
            prompt="ab",
            label="cd",
            tokens=[1, 2, 5, 6, 5, 6],
            response_length=4,
            reward=0.0,
            status=Sample.Status.COMPLETED,
        )
    ]

    train_data = rollout_manager_class._convert_samples_to_train_data(manager, samples)
    assert train_data["tokens"][0] == [1, 2, 5, 6, 5, 6]
    assert train_data["g1_full_sequences"][0] == [1, 2, 3, 4, 0, 0, 5, 6, 5, 6]
    assert train_data["ebft_logprob_source_rows"][0] == [0, 1, 2, 3, 4, 1, 3, 6, 7]

    train_data["total_lengths"] = [len(tokens) for tokens in train_data["tokens"]]
    train_data["log_probs"] = [torch.zeros(4)]
    train_data["g1_token_advantages"] = [torch.tensor([1.0, 2.0, 3.0, 4.0])]
    _tensorize_train_data(train_data)
    megatron_loss.compute_advantages_and_returns(manager.args, train_data)

    iterator = data_module.DataIterator(train_data, micro_batch_size=1)
    batch = iterator.get_next(
        [
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "advantages",
            "g1_full_sequences",
            "g1_qa_masks",
            "ebft_logprob_source_rows",
            "ebft_logprob_target_positions",
            "ebft_logprob_indexing",
        ]
    )
    assert batch["total_lengths"] == [6]

    data_module.prepare_g1_ebft_tokens_for_batch(batch, manager.args)
    assert batch["tokens"][0].tolist() == train_data["g1_full_sequences"][0].tolist()
    assert batch["total_lengths"] == [10]

    attach_ebft_g1_next_token_contract_to_batch(batch, manager.args)
    batch["unconcat_tokens"] = batch["tokens"]
    torch.testing.assert_close(batch["ebft_logprob_source_rows"][0], torch.tensor([0, 1, 2, 3, 4, 1, 3, 6, 7]))
    torch.testing.assert_close(batch["ebft_logprob_target_positions"][0], torch.arange(1, 10))

    logits = torch.zeros(1, 10, 16, dtype=torch.float32, requires_grad=True)
    loss, metrics = megatron_loss.policy_loss_function_g1_ebft(manager.args, batch, logits, lambda x: x)
    loss.backward()

    torch.testing.assert_close(metrics["pg_loss"], torch.tensor(-2.5))
    expected_grad = torch.zeros_like(logits)
    expected_grad[0, 1, 5] = -0.25
    expected_grad[0, 3, 6] = -0.5
    expected_grad[0, 6, 5] = -0.75
    expected_grad[0, 7, 6] = -1.0
    torch.testing.assert_close(logits.grad, expected_grad)
    assert logits.grad[0, 5, 5].item() == 0.0
    assert logits.grad[0, 6, 6].item() == 0.0
    assert logits.grad[0, 7, 5].item() == 0.0
    assert logits.grad[0, 8, 6].item() == 0.0
