#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BlockwiseConfig:
    prompt_length: int
    context_length: int
    generate_length: int
    stride: int

    @property
    def num_blocks(self) -> int:
        remainder = self.prompt_length - self.generate_length - self.context_length
        if remainder < 0 or remainder % self.stride != 0:
            raise ValueError(
                "Invalid block geometry: "
                f"prompt_length={self.prompt_length}, context_length={self.context_length}, "
                f"generate_length={self.generate_length}, stride={self.stride}"
            )
        return remainder // self.stride + 1

    @property
    def response_length(self) -> int:
        return self.generate_length * self.num_blocks

    @property
    def full_sequence_length(self) -> int:
        return self.prompt_length + self.response_length


def parse_int_list(raw: str | None, *, expected: int, name: str, default: list[int]) -> list[int]:
    if raw is None:
        values = list(default)
    else:
        values = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} integers, got {len(values)}: {values}")
    return values


def strided_blocks(values: list[int], config: BlockwiseConfig) -> list[list[int]]:
    blocks: list[list[int]] = []
    for block_idx in range(config.num_blocks):
        start = config.context_length + block_idx * config.stride
        end = start + config.generate_length
        block = values[start:end]
        if len(block) != config.generate_length:
            raise ValueError(f"Block {block_idx} has length {len(block)}, expected {config.generate_length}")
        blocks.append(block)
    return blocks


def time_major_flat(blocks: list[list[int]], config: BlockwiseConfig) -> list[int]:
    return [blocks[block_idx][step] for step in range(config.generate_length) for block_idx in range(config.num_blocks)]


def build_position_ids(config: BlockwiseConfig) -> list[int]:
    prompt_positions = list(range(config.prompt_length))
    response_positions = [
        config.context_length + block_idx * config.stride + step
        for step in range(config.generate_length)
        for block_idx in range(config.num_blocks)
    ]
    return prompt_positions + response_positions


def build_response_layout(
    *,
    config: BlockwiseConfig,
    prompt_tokens: list[int],
    prompt_qa: list[int],
    prompt_doc: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(config.generate_length):
        for block_idx in range(config.num_blocks):
            response_index = step * config.num_blocks + block_idx
            target_sequence_pos = config.prompt_length + response_index
            prompt_anchor_pos = config.context_length + block_idx * config.stride + step
            if step == 0:
                source_logit_pos = prompt_anchor_pos - 1
                source_kind = "prompt_anchor"
            else:
                source_logit_pos = config.prompt_length + (step - 1) * config.num_blocks + block_idx
                source_kind = "previous_generated_token"
            rows.append(
                {
                    "step": step,
                    "block": block_idx,
                    "response_index": response_index,
                    "target_sequence_pos": target_sequence_pos,
                    "prompt_anchor_pos": prompt_anchor_pos,
                    "source_logit_pos": source_logit_pos,
                    "source_kind": source_kind,
                    "target_token": prompt_tokens[prompt_anchor_pos],
                    "target_qa": prompt_qa[prompt_anchor_pos],
                    "target_doc": prompt_doc[prompt_anchor_pos],
                }
            )
    return rows


def build_strict_source_target_map(layout: list[dict[str, Any]]) -> list[dict[str, int]]:
    return [
        {
            "source_logit_row": int(row["source_logit_pos"]),
            "target_pos": int(row["target_sequence_pos"]),
            "response_idx": int(row["response_index"]),
            "step": int(row["step"]),
            "block": int(row["block"]),
        }
        for row in layout
    ]


def build_pair_axis(
    *,
    config: BlockwiseConfig,
    full_sequence: list[int],
    layout: list[dict[str, Any]],
) -> dict[str, Any]:
    strict_source_by_target = {
        int(row["target_sequence_pos"]): int(row["source_logit_pos"])
        for row in layout
    }
    source_kind_by_target = {
        int(row["target_sequence_pos"]): str(row["source_kind"])
        for row in layout
    }
    target_positions = list(range(1, config.full_sequence_length))
    standard_source_rows = [target_pos - 1 for target_pos in target_positions]
    strict_source_rows = [
        strict_source_by_target.get(target_pos, target_pos - 1)
        for target_pos in target_positions
    ]
    target_tokens = [full_sequence[target_pos] for target_pos in target_positions]
    generated_target_positions = list(range(config.prompt_length, config.full_sequence_length))

    pairs = []
    for target_pos, target_token, standard_source, strict_source in zip(
        target_positions,
        target_tokens,
        standard_source_rows,
        strict_source_rows,
        strict=True,
    ):
        pairs.append(
            {
                "target_pos": target_pos,
                "target_token": target_token,
                "strict_source_row": strict_source,
                "standard_source_row": standard_source,
                "source_kind": source_kind_by_target.get(target_pos, "standard_prompt_shift"),
                "is_generated_target": target_pos >= config.prompt_length,
            }
        )

    return {
        "pairs": pairs,
        "pair_target_positions": target_positions,
        "target_tokens": target_tokens,
        "strict_pair_source_rows": strict_source_rows,
        "standard_pair_source_rows": standard_source_rows,
        "generated_pair_target_positions": generated_target_positions,
        "strict_action_source_rows": [
            strict_source_by_target[target_pos]
            for target_pos in generated_target_positions
        ],
        "standard_generated_source_rows": [
            target_pos - 1
            for target_pos in generated_target_positions
        ],
    }


def build_fake_gather(pair_axis: dict[str, Any]) -> dict[str, Any]:
    source_rows = sorted(
        set(pair_axis["strict_pair_source_rows"])
        | set(pair_axis["standard_pair_source_rows"])
    )
    target_tokens = sorted(set(pair_axis["target_tokens"]))
    logits = {
        source_row: {
            target_token: source_row * 1000 + target_token
            for target_token in target_tokens
        }
        for source_row in source_rows
    }

    def gather(source_rows_to_gather: list[int]) -> list[int]:
        return [
            logits[source_row][target_token]
            for source_row, target_token in zip(source_rows_to_gather, pair_axis["target_tokens"], strict=True)
        ]

    generated_count = len(pair_axis["generated_pair_target_positions"])
    strict_pair_scores = gather(pair_axis["strict_pair_source_rows"])
    standard_pair_scores = gather(pair_axis["standard_pair_source_rows"])

    return {
        "score_formula": "score = source_row * 1000 + target_token",
        "logits": logits,
        "target_tokens": pair_axis["target_tokens"],
        "strict_gathered_target_logprobs": strict_pair_scores,
        "standard_gathered_target_logprobs": standard_pair_scores,
        "strict_generated_target_logprobs": strict_pair_scores[-generated_count:],
        "standard_generated_target_logprobs": standard_pair_scores[-generated_count:],
    }


def build_standard_next_token_shift(
    *,
    config: BlockwiseConfig,
    full_qa: list[int],
    response_advantages: list[int],
) -> dict[str, Any]:
    length_minus_one = config.full_sequence_length - 1
    action_next = [False] * length_minus_one
    advantages_next = [0] * length_minus_one
    start = config.prompt_length - 1
    end = start + config.response_length
    for row, advantage in zip(range(start, end), response_advantages, strict=True):
        action_next[row] = True
        advantages_next[row] = advantage
    return {
        "logit_rows": list(range(length_minus_one)),
        "target_positions": list(range(1, config.full_sequence_length)),
        "qa_next": [bool(v) for v in full_qa[1:]],
        "action_next": action_next,
        "advantages_next": advantages_next,
    }


def build_strict_block_shift(
    *,
    config: BlockwiseConfig,
    layout: list[dict[str, Any]],
    response_advantages: list[int],
) -> dict[str, Any]:
    action_by_logit_row = [False] * (config.full_sequence_length - 1)
    qa_by_logit_row = [False] * (config.full_sequence_length - 1)
    doc_by_logit_row = [None] * (config.full_sequence_length - 1)
    target_by_logit_row = [None] * (config.full_sequence_length - 1)
    advantages_by_logit_row = [0] * (config.full_sequence_length - 1)

    for row, advantage in zip(layout, response_advantages, strict=True):
        source = int(row["source_logit_pos"])
        action_by_logit_row[source] = True
        qa_by_logit_row[source] = bool(row["target_qa"])
        doc_by_logit_row[source] = int(row["target_doc"])
        target_by_logit_row[source] = int(row["target_sequence_pos"])
        advantages_by_logit_row[source] = advantage

    return {
        "logit_rows": list(range(config.full_sequence_length - 1)),
        "action_by_logit_row": action_by_logit_row,
        "qa_target_by_logit_row": qa_by_logit_row,
        "doc_target_by_logit_row": doc_by_logit_row,
        "target_position_by_logit_row": target_by_logit_row,
        "advantages_by_logit_row": advantages_by_logit_row,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = BlockwiseConfig(
        prompt_length=args.prompt_length,
        context_length=args.context_length,
        generate_length=args.generate_length,
        stride=args.stride,
    )

    default_tokens = list(range(100, 100 + config.prompt_length))
    default_qa = [0] * config.prompt_length
    for idx in range(config.context_length, config.prompt_length):
        default_qa[idx] = 1 if idx % 3 != 1 else 0
    default_doc = [0 if idx < config.context_length + config.stride else 1 for idx in range(config.prompt_length)]

    prompt_tokens = parse_int_list(args.prompt_tokens, expected=config.prompt_length, name="prompt_tokens", default=default_tokens)
    prompt_qa = parse_int_list(args.prompt_qa, expected=config.prompt_length, name="prompt_qa", default=default_qa)
    prompt_doc = parse_int_list(args.prompt_doc, expected=config.prompt_length, name="prompt_doc", default=default_doc)

    token_blocks = strided_blocks(prompt_tokens, config)
    qa_blocks = strided_blocks(prompt_qa, config)
    doc_blocks = strided_blocks(prompt_doc, config)

    generated_tokens = time_major_flat(token_blocks, config)
    generated_qa = time_major_flat(qa_blocks, config)
    generated_doc = time_major_flat(doc_blocks, config)
    response_advantages = list(range(1, config.response_length + 1))
    full_sequence = prompt_tokens + generated_tokens
    full_qa = prompt_qa + generated_qa
    full_doc = prompt_doc + generated_doc
    position_ids = build_position_ids(config)
    layout = build_response_layout(
        config=config,
        prompt_tokens=prompt_tokens,
        prompt_qa=prompt_qa,
        prompt_doc=prompt_doc,
    )
    strict_source_target_map = build_strict_source_target_map(layout)
    standard_next_token_shift = build_standard_next_token_shift(
        config=config,
        full_qa=full_qa,
        response_advantages=response_advantages,
    )
    strict_block_shift = build_strict_block_shift(
        config=config,
        layout=layout,
        response_advantages=response_advantages,
    )
    pair_axis = build_pair_axis(
        config=config,
        full_sequence=full_sequence,
        layout=layout,
    )
    fake_gather = build_fake_gather(pair_axis)
    standard_action_rows = [
        row_idx for row_idx, is_action in enumerate(standard_next_token_shift["action_next"]) if is_action
    ]
    strict_action_source_rows = [row["source_logit_row"] for row in strict_source_target_map]
    strict_action_target_positions = [row["target_pos"] for row in strict_source_target_map]

    expected_tiny_position_ids = [0, 1, 2, 3, 4, 5, 2, 4, 3, 5]
    expected_tiny_strict_source_rows = [1, 3, 6, 7]
    expected_tiny_strict_target_positions = [6, 7, 8, 9]
    expected_tiny_standard_action_rows = [5, 6, 7, 8]
    expected_tiny_pair_target_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_tiny_strict_pair_source_rows = [0, 1, 2, 3, 4, 1, 3, 6, 7]
    is_tiny_fixture = (
        config.prompt_length == 6
        and config.context_length == 2
        and config.generate_length == 2
        and config.stride == 2
    )

    checks = {
        "num_blocks_formula": config.num_blocks
        == (config.prompt_length - config.generate_length - config.context_length) // config.stride + 1,
        "response_length_is_generate_times_blocks": config.response_length == config.generate_length * config.num_blocks,
        "response_layout_is_time_major": generated_tokens == [
            prompt_tokens[config.context_length + block_idx * config.stride + step]
            for step in range(config.generate_length)
            for block_idx in range(config.num_blocks)
        ],
        "generated_qa_matches_strided_unfold": generated_qa == time_major_flat(qa_blocks, config),
        "generated_doc_matches_strided_unfold": generated_doc == time_major_flat(doc_blocks, config),
        "generated_qa_not_all_one": any(value != 1 for value in generated_qa),
        "generated_doc_not_all_one": any(value != 1 for value in generated_doc),
        "strict_step0_uses_prompt_anchor": all(
            row["source_kind"] == "prompt_anchor" and row["source_logit_pos"] == row["prompt_anchor_pos"] - 1
            for row in layout
            if row["step"] == 0
        ),
        "strict_later_steps_use_previous_generated_token": all(
            row["source_kind"] == "previous_generated_token"
            and row["source_logit_pos"] == config.prompt_length + (row["step"] - 1) * config.num_blocks + row["block"]
            for row in layout
            if row["step"] > 0
        ),
        "tiny_fixture_position_ids": (not is_tiny_fixture) or position_ids == expected_tiny_position_ids,
        "tiny_fixture_strict_source_rows": (not is_tiny_fixture)
        or strict_action_source_rows == expected_tiny_strict_source_rows,
        "tiny_fixture_strict_target_positions": (not is_tiny_fixture)
        or strict_action_target_positions == expected_tiny_strict_target_positions,
        "tiny_fixture_standard_action_rows": (not is_tiny_fixture)
        or standard_action_rows == expected_tiny_standard_action_rows,
        "tiny_fixture_standard_rows_differ_from_strict_source_rows": (not is_tiny_fixture)
        or standard_action_rows != strict_action_source_rows,
        "pair_axis_targets_match_standard_shift": pair_axis["pair_target_positions"]
        == standard_next_token_shift["target_positions"],
        "pair_axis_target_tokens_match_full_sequence_shift": pair_axis["target_tokens"] == full_sequence[1:],
        "pair_axis_standard_rows_match_standard_shift": pair_axis["standard_pair_source_rows"]
        == standard_next_token_shift["logit_rows"],
        "pair_axis_generated_rows_match_action_views": pair_axis["strict_action_source_rows"]
        == strict_action_source_rows
        and pair_axis["standard_generated_source_rows"] == standard_action_rows,
        "fake_gather_pair_lengths_match": len(fake_gather["strict_gathered_target_logprobs"])
        == len(pair_axis["pair_target_positions"])
        and len(fake_gather["standard_gathered_target_logprobs"]) == len(pair_axis["pair_target_positions"]),
        "fake_gather_generated_strict_differs_from_standard": fake_gather["strict_generated_target_logprobs"]
        != fake_gather["standard_generated_target_logprobs"],
        "tiny_fixture_pair_target_positions": (not is_tiny_fixture)
        or pair_axis["pair_target_positions"] == expected_tiny_pair_target_positions,
        "tiny_fixture_strict_pair_source_rows": (not is_tiny_fixture)
        or pair_axis["strict_pair_source_rows"] == expected_tiny_strict_pair_source_rows,
        "standard_qa_next_is_full_qa_shifted_left": standard_next_token_shift["qa_next"] == [bool(v) for v in full_qa[1:]],
    }

    return {
        "config": asdict(config) | {
            "num_blocks": config.num_blocks,
            "response_length": config.response_length,
            "full_sequence_length": config.full_sequence_length,
        },
        "prompt": {
            "tokens": prompt_tokens,
            "qa": prompt_qa,
            "doc": prompt_doc,
        },
        "strided_unfold_block_major": {
            "tokens": token_blocks,
            "qa": qa_blocks,
            "doc": doc_blocks,
        },
        "time_major_response": {
            "tokens": generated_tokens,
            "qa": generated_qa,
            "doc": generated_doc,
        },
        "full_sequence": {
            "tokens": full_sequence,
            "qa": full_qa,
            "doc": full_doc,
            "position_ids": position_ids,
        },
        "strict_block_prediction_layout": layout,
        "strict_source_target_map": strict_source_target_map,
        "pair_axis": pair_axis,
        "fake_gather": fake_gather,
        "standard_contiguous_next_token_shift": standard_next_token_shift | {
            "action_rows": standard_action_rows,
        },
        "strict_block_prediction_shift": strict_block_shift | {
            "source_rows": strict_action_source_rows,
            "target_positions": strict_action_target_positions,
        },
        "checks": checks,
    }


def format_bool(value: bool) -> str:
    return "OK" if value else "FAIL"


def print_report(report: dict[str, Any]) -> None:
    config = report["config"]
    print("Blockwise strict EBFT preflight")
    print(
        "geometry: "
        f"prompt_length={config['prompt_length']} context_length={config['context_length']} "
        f"generate_length={config['generate_length']} stride={config['stride']} "
        f"num_blocks={config['num_blocks']} response_length={config['response_length']}"
    )
    print()

    print("Prompt")
    print(f"  tokens: {report['prompt']['tokens']}")
    print(f"  qa:     {report['prompt']['qa']}")
    print(f"  doc:    {report['prompt']['doc']}")
    print()

    print("Time-major response from strided unfold")
    print(f"  tokens: {report['time_major_response']['tokens']}")
    print(f"  qa:     {report['time_major_response']['qa']}")
    print(f"  doc:    {report['time_major_response']['doc']}")
    print()

    print("Full sequence")
    print(f"  tokens:       {report['full_sequence']['tokens']}")
    print(f"  qa:           {report['full_sequence']['qa']}")
    print(f"  doc:          {report['full_sequence']['doc']}")
    print(f"  position_ids: {report['full_sequence']['position_ids']}")
    print()

    print("Strict block prediction layout")
    print("  step block resp_idx target_pos prompt_anchor source_logit source_kind              token qa doc")
    for row in report["strict_block_prediction_layout"]:
        print(
            "  "
            f"{row['step']:>4} {row['block']:>5} {row['response_index']:>8} "
            f"{row['target_sequence_pos']:>10} {row['prompt_anchor_pos']:>13} "
            f"{row['source_logit_pos']:>12} {row['source_kind']:<24} "
            f"{row['target_token']:>5} {row['target_qa']:>2} {row['target_doc']:>3}"
        )
    print()

    print("Strict source-target map")
    print("  source_logit_row target_pos response_idx step block")
    for row in report["strict_source_target_map"]:
        print(
            "  "
            f"{row['source_logit_row']:>16} {row['target_pos']:>10} {row['response_idx']:>12} "
            f"{row['step']:>4} {row['block']:>5}"
        )
    print()

    pair_axis = report["pair_axis"]
    print("Pair-axis contract")
    print(f"  pair_target_positions:         {pair_axis['pair_target_positions']}")
    print(f"  strict_pair_source_rows:       {pair_axis['strict_pair_source_rows']}")
    print(f"  standard_pair_source_rows:     {pair_axis['standard_pair_source_rows']}")
    print(f"  strict action source rows:     {pair_axis['strict_action_source_rows']}")
    print(f"  standard generated source rows:{pair_axis['standard_generated_source_rows']}")
    print("  target_pos target_token strict_source standard_source source_kind")
    for row in pair_axis["pairs"]:
        print(
            "  "
            f"{row['target_pos']:>10} {row['target_token']:>12} {row['strict_source_row']:>13} "
            f"{row['standard_source_row']:>15} {row['source_kind']}"
        )
    print()

    fake_gather = report["fake_gather"]
    print("Fake logits gather demo")
    print(f"  {fake_gather['score_formula']}")
    print(f"  target_tokens:                    {fake_gather['target_tokens']}")
    print(f"  strict gathered target logprobs:  {fake_gather['strict_gathered_target_logprobs']}")
    print(f"  standard gathered target scores:  {fake_gather['standard_gathered_target_logprobs']}")
    print(f"  strict generated target logprobs: {fake_gather['strict_generated_target_logprobs']}")
    print(f"  standard generated target scores: {fake_gather['standard_generated_target_logprobs']}")
    print()

    standard = report["standard_contiguous_next_token_shift"]
    strict = report["strict_block_prediction_shift"]
    print("Next-token shift views")
    print(f"  standard action row indexes:        {standard['action_rows']}")
    print(f"  strict source logit row indexes:    {strict['source_rows']}")
    print(f"  strict target position indexes:     {strict['target_positions']}")
    print(f"  standard qa_next=full_qa[1:]:        {standard['qa_next']}")
    print(f"  standard action rows:               {standard['action_next']}")
    print(f"  strict action by source logit row:  {strict['action_by_logit_row']}")
    print(f"  strict target by source logit row:  {strict['target_position_by_logit_row']}")
    print(f"  strict qa by source logit row:      {strict['qa_target_by_logit_row']}")
    print()

    print("Checks")
    for name, ok in report["checks"].items():
        print(f"  {format_bool(bool(ok)):>4}  {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a CPU-only strict EBFT block-pred layout fixture.",
    )
    parser.add_argument("--prompt-length", type=int, default=6)
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--generate-length", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--prompt-tokens", default=None, help="Comma-separated prompt token ids.")
    parser.add_argument("--prompt-qa", default=None, help="Comma-separated prompt QA mask values.")
    parser.add_argument("--prompt-doc", default=None, help="Comma-separated prompt doc ids.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)

    return 0 if all(bool(value) for value in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
