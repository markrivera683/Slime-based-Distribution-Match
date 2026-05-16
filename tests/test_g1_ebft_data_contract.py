"""CPU tests for G1 EBFT next-token data contract (masks + advantages layout)."""

from argparse import Namespace

import pytest
import torch

from slime.utils.g1_ebft_data_contract import (
    attach_ebft_g1_next_token_contract_to_batch,
    build_ebft_g1_next_token_tensors,
)


def test_build_ebft_g1_next_token_geometry_384_376():
    pl, rl = 384, 376
    L = pl + rl
    seq = torch.arange(L, dtype=torch.long)
    qa = torch.ones(L, dtype=torch.long)
    resp_adv = torch.linspace(0.1, 0.5, rl)

    action, qa_next, adv_next = build_ebft_g1_next_token_tensors(
        g1_full_sequence=seq,
        g1_qa_mask=qa,
        response_advantages=resp_adv,
        g1_prompt_length=pl,
        g1_response_length=rl,
    )

    assert action.shape == (L - 1,)
    assert qa_next.shape == (L - 1,)
    assert adv_next.shape == (L - 1,)
    assert qa_next.dtype == torch.bool
    assert qa_next.all()

    assert action.sum().item() == rl
    assert int(action[: pl - 1].sum()) == 0
    assert action[pl - 1 : pl - 1 + rl].all()

    assert adv_next[: pl - 1].abs().max().item() == 0.0
    torch.testing.assert_close(adv_next[pl - 1 : pl - 1 + rl], resp_adv)
    tail = adv_next[pl - 1 + rl :]
    assert tail.numel() == 0 or tail.abs().max().item() == 0.0


def test_qa_mask_shift_matches_openrlhf_slice():
    pl, rl = 384, 376
    L = pl + rl
    qa = torch.zeros(L, dtype=torch.long)
    qa[:200] = 1
    qa[pl:] = 1
    seq = torch.zeros(L, dtype=torch.long)
    resp_adv = torch.zeros(rl)

    _, qa_next, _ = build_ebft_g1_next_token_tensors(
        g1_full_sequence=seq,
        g1_qa_mask=qa,
        response_advantages=resp_adv,
        g1_prompt_length=pl,
        g1_response_length=rl,
        qa_masking=True,
    )
    expected = qa[1:].ne(0)
    assert torch.equal(qa_next, expected)


def test_qa_masking_off_matches_openrlhf_ones():
    pl, rl = 384, 376
    L = pl + rl
    seq = torch.zeros(L, dtype=torch.long)
    qa = torch.zeros(L, dtype=torch.long)
    resp_adv = torch.zeros(rl)
    _, qa_next, _ = build_ebft_g1_next_token_tensors(
        g1_full_sequence=seq,
        g1_qa_mask=qa,
        response_advantages=resp_adv,
        g1_prompt_length=pl,
        g1_response_length=rl,
        qa_masking=False,
    )
    assert qa_next.shape == (L - 1,)
    assert qa_next.all()


def test_maybe_attach_populates_lists():
    ns = Namespace(g1_use_ebft_loss=True, g1_prompt_length=384, g1_response_length=376)
    pl, rl = 384, 376
    L = pl + rl
    batch = {
        "advantages": [torch.ones(rl), torch.ones(rl) * 2],
        "g1_full_sequences": [torch.zeros(L, dtype=torch.long), torch.zeros(L, dtype=torch.long)],
        "g1_qa_masks": [torch.ones(L, dtype=torch.long), torch.ones(L, dtype=torch.long)],
    }
    attach_ebft_g1_next_token_contract_to_batch(batch, ns)
    assert "ebft_action_mask_next" in batch
    assert len(batch["ebft_action_mask_next"]) == 2
    assert batch["ebft_action_mask_next"][0].shape == (L - 1,)
    assert batch["ebft_seq_len_m1"] == [L - 1, L - 1]
    torch.testing.assert_close(batch["ebft_advantages_next"][1][pl - 1 : pl - 1 + rl], batch["advantages"][1])


def test_build_rejects_length_mismatch():
    with pytest.raises(ValueError, match="full sequence length"):
        build_ebft_g1_next_token_tensors(
            g1_full_sequence=torch.zeros(100),
            g1_qa_mask=torch.zeros(100),
            response_advantages=torch.zeros(376),
            g1_prompt_length=384,
            g1_response_length=376,
        )


def test_maybe_attach_no_op_without_flag():
    batch = {
        "advantages": [torch.ones(376)],
        "g1_full_sequences": [torch.zeros(760)],
        "g1_qa_masks": [torch.ones(760)],
    }
    attach_ebft_g1_next_token_contract_to_batch(batch, Namespace(g1_use_ebft_loss=False))
    assert "ebft_action_mask_next" not in batch


def test_attach_no_op_without_advantages_even_when_ebft_enabled():
    ns = Namespace(
        g1_use_ebft_loss=True,
        g1_prompt_length=384,
        g1_response_length=376,
        g1_qa_masking=False,
    )
    batch = {"g1_full_sequences": [torch.zeros(760, dtype=torch.long)]}
    attach_ebft_g1_next_token_contract_to_batch(batch, ns)
    assert "ebft_action_mask_next" not in batch


def test_attach_requires_g1_full_sequences_when_ebft_enabled():
    ns = Namespace(
        g1_use_ebft_loss=True,
        g1_prompt_length=384,
        g1_response_length=376,
        g1_qa_masking=False,
    )
    batch = {"advantages": [torch.ones(376)]}
    with pytest.raises(ValueError, match="g1_full_sequences"):
        attach_ebft_g1_next_token_contract_to_batch(batch, ns)


def test_attach_qa_masking_requires_g1_qa_masks():
    pl, rl = 384, 376
    L = pl + rl
    ns = Namespace(
        g1_use_ebft_loss=True,
        g1_prompt_length=pl,
        g1_response_length=rl,
        g1_qa_masking=True,
    )
    batch = {
        "advantages": [torch.ones(rl)],
        "g1_full_sequences": [torch.zeros(L, dtype=torch.long)],
    }
    with pytest.raises(ValueError, match="g1_qa_masks"):
        attach_ebft_g1_next_token_contract_to_batch(batch, ns)


def test_maybe_attach_without_qa_masks_when_masking_off():
    ns = Namespace(
        g1_use_ebft_loss=True,
        g1_prompt_length=384,
        g1_response_length=376,
        g1_qa_masking=False,
    )
    pl, rl = 384, 376
    L = pl + rl
    batch = {
        "advantages": [torch.ones(rl)],
        "g1_full_sequences": [torch.zeros(L, dtype=torch.long)],
        "total_lengths": [L],
        "response_lengths": [rl],
    }
    attach_ebft_g1_next_token_contract_to_batch(batch, ns)
    assert "ebft_action_mask_next" in batch
    assert len(batch["ebft_action_mask_next"]) == 1
