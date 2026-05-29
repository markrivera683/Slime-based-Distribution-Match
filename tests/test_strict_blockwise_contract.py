"""CPU-only strict EBFT block prediction layout contract tests."""

from argparse import Namespace

import torch

from slime.utils.g1_core import get_num_strided_blocks
from slime.utils.g1_ebft_loss import build_ebft_g1_next_token_tensors
from slime.utils.g1_ebft_loss import build_ebft_g1_logprob_pair_axis


def _tiny_args() -> Namespace:
    return Namespace(
        g1_prompt_length=6,
        g1_context_length=2,
        g1_generate_length=2,
        g1_stride=2,
        g1_response_length=4,
        n_samples_per_prompt=1,
        g1_hidden_state_method="last_only",
        g1_qa_masking=True,
        g1_document_masking=False,
    )


def _flatten_block_major_response(block_major: torch.Tensor) -> torch.Tensor:
    return block_major.t().contiguous().reshape(-1)


def _prompt_side_strided_suffix(
    prompt_values: torch.Tensor,
    *,
    context_length: int,
    generate_length: int,
    stride: int,
) -> torch.Tensor:
    return prompt_values[context_length:].unfold(0, generate_length, stride).t().contiguous().reshape(-1)


def _strict_position_ids(
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
) -> torch.Tensor:
    num_blocks = get_num_strided_blocks(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    prompt_positions = torch.arange(prompt_length)
    block_starting_positions = torch.arange(num_blocks) * stride + context_length
    generated_positions = torch.stack(
        [block_starting_positions + step_idx for step_idx in range(generate_length)],
        dim=0,
    ).reshape(-1)
    return torch.cat([prompt_positions, generated_positions])


def _strict_block_logit_positions(
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
) -> list[torch.Tensor]:
    num_blocks = get_num_strided_blocks(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    steps = []
    for step_idx in range(generate_length):
        if step_idx == 0:
            positions = torch.tensor([context_length + block_idx * stride - 1 for block_idx in range(num_blocks)])
        else:
            start = prompt_length + (step_idx - 1) * num_blocks
            positions = torch.arange(start, start + num_blocks)
        steps.append(positions)
    return steps


def _strict_source_target_map(
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
) -> list[dict[str, int]]:
    num_blocks = get_num_strided_blocks(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    rows = []
    for step_idx in range(generate_length):
        for block_idx in range(num_blocks):
            response_idx = step_idx * num_blocks + block_idx
            if step_idx == 0:
                source_logit_row = context_length + block_idx * stride - 1
            else:
                source_logit_row = prompt_length + (step_idx - 1) * num_blocks + block_idx
            rows.append(
                {
                    "source_logit_row": source_logit_row,
                    "target_pos": prompt_length + response_idx,
                    "response_idx": response_idx,
                    "step": step_idx,
                    "block": block_idx,
                }
            )
    return rows


def _strict_next_token_pair_axis(
    *,
    prompt_length: int,
    response_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
) -> list[dict[str, int | bool]]:
    prompt_pairs = [
        {
            "source_logit_row": target_pos - 1,
            "target_pos": target_pos,
            "is_action": False,
        }
        for target_pos in range(1, prompt_length)
    ]
    action_pairs = [
        {
            "source_logit_row": row["source_logit_row"],
            "target_pos": row["target_pos"],
            "is_action": True,
        }
        for row in _strict_source_target_map(
            prompt_length=prompt_length,
            context_length=context_length,
            generate_length=generate_length,
            stride=stride,
        )
    ]
    assert len(action_pairs) == response_length
    return prompt_pairs + action_pairs


def _fake_score_table(*, max_source_row: int, target_tokens: list[int]) -> list[dict[int, int]]:
    return [{token: source_row * 1000 + token for token in target_tokens} for source_row in range(max_source_row + 1)]


def _gather_fake_scores(
    score_table: list[dict[int, int]],
    source_rows: list[int],
    target_tokens: list[int],
) -> list[int]:
    return [score_table[source_row][target_token] for source_row, target_token in zip(source_rows, target_tokens)]


def test_tiny_strict_blockwise_layout_is_cpu_only_contract():
    args = _tiny_args()
    prompt_length = args.g1_prompt_length
    context_length = args.g1_context_length
    generate_length = args.g1_generate_length
    stride = args.g1_stride
    response_length = args.g1_response_length
    num_blocks = get_num_strided_blocks(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )

    assert num_blocks == 2
    assert response_length == 4

    block_major_response = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
    time_major_response = _flatten_block_major_response(block_major_response)
    torch.testing.assert_close(time_major_response, torch.tensor([10, 20, 11, 21]))

    position_ids = _strict_position_ids(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    torch.testing.assert_close(position_ids, torch.tensor([0, 1, 2, 3, 4, 5, 2, 4, 3, 5]))

    step0, step1 = _strict_block_logit_positions(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    torch.testing.assert_close(step0, torch.tensor([1, 3]))
    torch.testing.assert_close(step1, torch.tensor([6, 7]))

    strict_source_target_map = _strict_source_target_map(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    assert strict_source_target_map == [
        {"source_logit_row": 1, "target_pos": 6, "response_idx": 0, "step": 0, "block": 0},
        {"source_logit_row": 3, "target_pos": 7, "response_idx": 1, "step": 0, "block": 1},
        {"source_logit_row": 6, "target_pos": 8, "response_idx": 2, "step": 1, "block": 0},
        {"source_logit_row": 7, "target_pos": 9, "response_idx": 3, "step": 1, "block": 1},
    ]
    strict_source_rows = [row["source_logit_row"] for row in strict_source_target_map]
    strict_target_positions = [row["target_pos"] for row in strict_source_target_map]
    assert strict_source_rows == [1, 3, 6, 7]
    assert strict_target_positions == [6, 7, 8, 9]

    prompt_target_positions = _prompt_side_strided_suffix(
        torch.arange(prompt_length),
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    torch.testing.assert_close(prompt_target_positions, torch.tensor([2, 4, 3, 5]))

    prompt_qa = torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.long)
    prompt_doc = torch.tensor([0, 0, 7, 8, 7, 8], dtype=torch.long)
    generated_qa_suffix = _prompt_side_strided_suffix(
        prompt_qa,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    generated_doc_suffix = _prompt_side_strided_suffix(
        prompt_doc,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    torch.testing.assert_close(generated_qa_suffix, prompt_qa[prompt_target_positions])
    torch.testing.assert_close(generated_doc_suffix, prompt_doc[prompt_target_positions])
    torch.testing.assert_close(generated_qa_suffix, torch.tensor([1, 0, 0, 1]))
    torch.testing.assert_close(generated_doc_suffix, torch.tensor([7, 7, 8, 8]))

    full_sequence = torch.cat([torch.arange(prompt_length), time_major_response])
    full_action_mask = torch.cat(
        [
            torch.zeros(prompt_length, dtype=torch.bool),
            torch.ones(response_length, dtype=torch.bool),
        ]
    )
    action_mask_next, qa_mask_next, advantages_next = build_ebft_g1_next_token_tensors(
        g1_full_sequence=full_sequence,
        g1_qa_mask=torch.ones(prompt_length + response_length, dtype=torch.long),
        response_advantages=torch.arange(response_length, dtype=torch.float32),
        g1_prompt_length=prompt_length,
        g1_response_length=response_length,
        qa_masking=True,
    )

    torch.testing.assert_close(action_mask_next, full_action_mask[1:])
    standard_action_rows = action_mask_next.nonzero(as_tuple=False).flatten().tolist()
    assert standard_action_rows == [5, 6, 7, 8]
    assert standard_action_rows != strict_source_rows
    assert action_mask_next.sum().item() == response_length
    assert not action_mask_next[: prompt_length - 1].any()
    assert action_mask_next[prompt_length - 1 : prompt_length - 1 + response_length].all()
    assert qa_mask_next.all()
    torch.testing.assert_close(
        advantages_next[prompt_length - 1 : prompt_length - 1 + response_length],
        torch.arange(response_length, dtype=torch.float32),
    )


def test_tiny_strict_pair_axis_and_fake_logits_gather_are_not_standard_shift():
    args = _tiny_args()
    prompt_length = args.g1_prompt_length
    context_length = args.g1_context_length
    generate_length = args.g1_generate_length
    stride = args.g1_stride
    response_length = args.g1_response_length

    block_major_response = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
    time_major_response = _flatten_block_major_response(block_major_response)
    full_sequence = torch.cat([torch.arange(prompt_length), time_major_response])

    strict_pair_axis = _strict_next_token_pair_axis(
        prompt_length=prompt_length,
        response_length=response_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    target_positions = [row["target_pos"] for row in strict_pair_axis]
    strict_source_rows = [row["source_logit_row"] for row in strict_pair_axis]
    action_pairs = [row for row in strict_pair_axis if row["is_action"]]

    assert target_positions == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert strict_source_rows == [0, 1, 2, 3, 4, 1, 3, 6, 7]
    assert [row["target_pos"] for row in action_pairs] == [6, 7, 8, 9]
    assert [row["source_logit_row"] for row in action_pairs] == [1, 3, 6, 7]

    standard_raw_next_token_source_rows = list(range(prompt_length + response_length - 1))
    assert standard_raw_next_token_source_rows == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert standard_raw_next_token_source_rows != strict_source_rows
    assert standard_raw_next_token_source_rows[: prompt_length - 1] == strict_source_rows[: prompt_length - 1]

    action_target_tokens = [full_sequence[target_pos].item() for target_pos in [6, 7, 8, 9]]
    action_standard_source_rows = [standard_raw_next_token_source_rows[target_pos - 1] for target_pos in [6, 7, 8, 9]]
    action_strict_source_rows = [row["source_logit_row"] for row in action_pairs]
    score_table = _fake_score_table(
        max_source_row=max(standard_raw_next_token_source_rows),
        target_tokens=action_target_tokens,
    )

    strict_scores = _gather_fake_scores(score_table, action_strict_source_rows, action_target_tokens)
    standard_scores = _gather_fake_scores(score_table, action_standard_source_rows, action_target_tokens)

    assert action_standard_source_rows == [5, 6, 7, 8]
    assert action_strict_source_rows == [1, 3, 6, 7]
    assert action_target_tokens == [10, 20, 11, 21]
    assert strict_scores == [1010, 3020, 6011, 7021]
    assert standard_scores == [5010, 6020, 7011, 8021]
    assert strict_scores != standard_scores


def test_core_pair_axis_helper_matches_openrlhf_strict_source_rows():
    args = _tiny_args()
    source_rows, target_positions, action_mask = build_ebft_g1_logprob_pair_axis(
        prompt_length=args.g1_prompt_length,
        response_length=args.g1_response_length,
        context_length=args.g1_context_length,
        generate_length=args.g1_generate_length,
        stride=args.g1_stride,
        indexing="strict_block_source",
    )

    torch.testing.assert_close(target_positions, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    torch.testing.assert_close(source_rows, torch.tensor([0, 1, 2, 3, 4, 1, 3, 6, 7]))
    torch.testing.assert_close(action_mask, torch.tensor([False, False, False, False, False, True, True, True, True]))
