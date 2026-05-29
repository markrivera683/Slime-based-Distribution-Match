"""CPU-only EBFT span sparse IR contract tests."""

from copy import deepcopy
import runpy
from pathlib import Path

import pytest
import torch

from slime.utils.g1_ebft_rollout_mask import build_g1_ebft_rollout_mask_contract
from slime.utils.g1_ebft_rollout_mask import ebft_span_sparse_ir_to_dense_allowed_matrix
from slime.utils.g1_ebft_rollout_mask import validate_ebft_span_sparse_ir


CONTRACT_VALIDATOR = (
    Path(__file__).resolve().parents[1]
    / "refactor_debugging"
    / "blockwise"
    / "validate_sglang_sparse_ir_contract.py"
)


def _tiny_sparse_ir():
    contract = build_g1_ebft_rollout_mask_contract(
        prompt_length=6,
        context_length=2,
        generate_length=2,
        stride=2,
    )
    return contract, contract.to_span_sparse_ir()


def test_span_sparse_ir_to_dense_matches_allowed_edge_contract() -> None:
    contract, sparse_ir = _tiny_sparse_ir()

    dense_from_spans = ebft_span_sparse_ir_to_dense_allowed_matrix(sparse_ir)

    torch.testing.assert_close(dense_from_spans, contract.to_dense_allowed_matrix())
    assert dense_from_spans[0, 0, 6, 0]
    assert dense_from_spans[0, 0, 6, 1]
    assert not dense_from_spans[0, 0, 6, 2]
    assert dense_from_spans[0, 0, 8, 6]
    assert not dense_from_spans[0, 0, 8, 7]


def test_span_sparse_ir_has_canonical_shape_bounds_and_order() -> None:
    _contract, sparse_ir = _tiny_sparse_ir()

    assert sparse_ir["version"] == 1
    assert sparse_ir["layout"] == "ebft_block_strided_v1"
    assert sparse_ir["seq_len"] == 10
    assert sparse_ir["query_len"] == 10
    assert sparse_ir["prefix_len"] == 6
    assert len(sparse_ir["q_indptr"]) == sparse_ir["query_len"] + 1
    assert sparse_ir["q_indptr"] == (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    for query_position in range(sparse_ir["query_len"]):
        prev_end = -1
        begin = sparse_ir["q_indptr"][query_position]
        end = sparse_ir["q_indptr"][query_position + 1]
        for span_idx in range(begin, end):
            span_start = sparse_ir["span_starts"][span_idx]
            span_end = sparse_ir["span_ends"][span_idx]
            assert 0 <= span_start < span_end <= sparse_ir["seq_len"]
            assert span_start > prev_end
            prev_end = span_end

    assert sparse_ir["span_starts"][6:8] == (0, 6)
    assert sparse_ir["span_ends"][6:8] == (2, 7)
    assert sparse_ir["span_starts"][10:13] == (0, 6, 8)
    assert sparse_ir["span_ends"][10:13] == (2, 7, 9)


def test_sglang_accepts_slime_sparse_ir_layout_contract() -> None:
    namespace = runpy.run_path(str(CONTRACT_VALIDATOR))

    namespace["validate_contract"]()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "version"),
        ("layout", "allowed_edges", "layout"),
        ("seq_len", 0, "seq_len"),
        ("query_len", 11, "query_len"),
        ("prefix_len", 11, "prefix_len"),
    ],
)
def test_span_sparse_ir_rejects_invalid_header_fields(field: str, value: object, message: str) -> None:
    _contract, sparse_ir = _tiny_sparse_ir()
    invalid_ir = deepcopy(sparse_ir)
    invalid_ir[field] = value

    with pytest.raises(ValueError, match=message):
        validate_ebft_span_sparse_ir(invalid_ir)


def test_span_sparse_ir_rejects_invalid_index_and_span_fields() -> None:
    _contract, sparse_ir = _tiny_sparse_ir()

    invalid_ir = deepcopy(sparse_ir)
    invalid_ir["q_indptr"] = invalid_ir["q_indptr"][:-1]
    with pytest.raises(ValueError, match="q_indptr length"):
        validate_ebft_span_sparse_ir(invalid_ir)

    invalid_ir = deepcopy(sparse_ir)
    invalid_ir["q_indptr"] = list(invalid_ir["q_indptr"])
    invalid_ir["q_indptr"][-1] += 1
    with pytest.raises(ValueError, match="number of spans"):
        validate_ebft_span_sparse_ir(invalid_ir)

    invalid_ir = deepcopy(sparse_ir)
    invalid_ir["span_ends"] = list(invalid_ir["span_ends"])
    invalid_ir["span_ends"][0] = sparse_ir["seq_len"] + 1
    with pytest.raises(ValueError, match="span bounds"):
        validate_ebft_span_sparse_ir(invalid_ir)

    invalid_ir = deepcopy(sparse_ir)
    invalid_ir["span_starts"] = list(invalid_ir["span_starts"])
    invalid_ir["span_starts"][7] = 1
    with pytest.raises(ValueError, match="sorted"):
        validate_ebft_span_sparse_ir(invalid_ir)
