"""CPU-only strict EBFT rollout mask contract tests."""

import pytest
import torch

from slime.utils.g1_ebft_rollout_mask import GENERATED_SOURCE_KIND
from slime.utils.g1_ebft_rollout_mask import PROMPT_SOURCE_KIND
from slime.utils.g1_ebft_rollout_mask import build_g1_ebft_rollout_mask_contract
from slime.utils.g1_ebft_rollout_mask import ebft_span_sparse_ir_to_dense_allowed_matrix


def test_tiny_rollout_mask_contract_layout_and_metadata() -> None:
    contract = build_g1_ebft_rollout_mask_contract(
        prompt_length=6,
        context_length=2,
        generate_length=2,
        stride=2,
        qa_values=[0, 0, 1, 0, 0, 1],
        doc_ids=[0, 0, 7, 8, 7, 8],
    )

    assert contract.num_blocks == 2
    assert contract.response_length == 4
    assert contract.full_sequence_length == 10
    assert contract.position_ids == (0, 1, 2, 3, 4, 5, 2, 4, 3, 5)
    assert contract.response_positions == (2, 4, 3, 5)
    assert contract.rollout_anchor_positions == (2, 4, 3, 5)
    assert contract.target_positions == (6, 7, 8, 9)

    assert contract.rollout_source_rows == (1, 3, 6, 7)
    assert contract.rollout_source_rows_by_step() == ((1, 3), (6, 7))
    assert contract.rollout_source_kinds == (
        PROMPT_SOURCE_KIND,
        PROMPT_SOURCE_KIND,
        GENERATED_SOURCE_KIND,
        GENERATED_SOURCE_KIND,
    )

    assert contract.logprob_source_rows == (1, 3, 6, 7)
    assert contract.logprob_target_positions == contract.target_positions
    assert contract.response_qa_values == (1, 0, 0, 1)
    assert contract.response_doc_ids == (7, 7, 8, 8)


def test_dense_allowed_matrix_and_sparse_edges_are_equivalent() -> None:
    contract = build_g1_ebft_rollout_mask_contract(
        prompt_length=6,
        context_length=2,
        generate_length=2,
        stride=2,
    )

    dense = contract.to_dense_allowed_matrix()
    sparse_dense = torch.zeros_like(dense)
    for query_position, key_position in contract.to_sparse_allowed_edges():
        sparse_dense[0, 0, query_position, key_position] = True

    torch.testing.assert_close(dense, sparse_dense)
    span_dense = ebft_span_sparse_ir_to_dense_allowed_matrix(contract.to_sparse_ir())
    torch.testing.assert_close(dense, span_dense)

    assert dense[0, 0, 6, 0]
    assert dense[0, 0, 6, 1]
    assert not dense[0, 0, 6, 2]
    assert dense[0, 0, 6, 6]
    assert dense[0, 0, 8, 6]
    assert not dense[0, 0, 8, 7]

    additive = contract.to_dense_additive_mask(dtype=torch.float32)
    assert additive[0, 0, 6, 0].item() == 0.0
    assert additive[0, 0, 8, 6].item() == 0.0
    assert additive[0, 0, 6, 2].item() < -1e20


def test_document_masking_resets_position_ids_and_filters_edges() -> None:
    contract = build_g1_ebft_rollout_mask_contract(
        prompt_length=6,
        context_length=2,
        generate_length=2,
        stride=2,
        doc_ids=[0, 0, 0, 0, 1, 1],
        document_masking=True,
    )

    assert contract.position_ids == (0, 1, 2, 3, 0, 1, 2, 0, 3, 1)
    assert contract.response_doc_ids == (0, 1, 0, 1)

    dense = contract.to_dense_allowed_matrix()
    assert dense[0, 0, 6, 0]
    assert dense[0, 0, 6, 1]
    assert not dense[0, 0, 7, 0]
    assert dense[0, 0, 7, 7]


def test_rollout_mask_contract_rejects_invalid_geometry_and_metadata() -> None:
    with pytest.raises(ValueError, match="stride must be >= 1"):
        build_g1_ebft_rollout_mask_contract(
            prompt_length=6,
            context_length=2,
            generate_length=2,
            stride=0,
        )

    with pytest.raises(ValueError, match="Invalid G1 strided-block geometry"):
        build_g1_ebft_rollout_mask_contract(
            prompt_length=5,
            context_length=2,
            generate_length=2,
            stride=2,
        )

    with pytest.raises(ValueError, match="doc_ids length"):
        build_g1_ebft_rollout_mask_contract(
            prompt_length=6,
            context_length=2,
            generate_length=2,
            stride=2,
            doc_ids=[0, 1],
        )

    with pytest.raises(ValueError, match="document_masking requires doc_ids"):
        build_g1_ebft_rollout_mask_contract(
            prompt_length=6,
            context_length=2,
            generate_length=2,
            stride=2,
            document_masking=True,
        )
