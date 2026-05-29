#!/usr/bin/env python3
"""Validate the Slime/SGLang EBFT sparse IR layout contract without imports."""

from __future__ import annotations

import ast
from pathlib import Path


SLIME_MASK_PATH = Path(
    "/mnt/data/distribution-matching-slime/code/slime_tmp/slime/utils/g1_ebft_rollout_mask.py"
)
SGLANG_FORWARD_BATCH_PATH = Path(
    "/root/slime_runtime/sglang/python/sglang/srt/model_executor/forward_batch_info.py"
)
SLIME_LAYOUT_CONSTANT = "EBFT_SPAN_SPARSE_IR_LAYOUT"
SGLANG_ACCEPTED_LAYOUTS_CONSTANT = "EBFT_SPARSE_IR_ACCEPTED_LAYOUTS"


def _read_assignment(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    raise AssertionError(f"{path} does not define {name}")


def _literal_eval_with_names(node: ast.AST, names: dict[str, object]) -> object:
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_eval_with_names(element, names) for element in node.elts)
    return ast.literal_eval(node)


def load_slime_layout() -> str:
    value = ast.literal_eval(_read_assignment(SLIME_MASK_PATH, SLIME_LAYOUT_CONSTANT))
    if not isinstance(value, str):
        raise AssertionError(
            f"{SLIME_LAYOUT_CONSTANT} must be a string, got {type(value).__name__}"
        )
    return value


def load_sglang_accepted_layouts(slime_layout: str) -> tuple[str, ...]:
    value = _literal_eval_with_names(
        _read_assignment(SGLANG_FORWARD_BATCH_PATH, SGLANG_ACCEPTED_LAYOUTS_CONSTANT),
        {"EBFT_SPARSE_IR_CANONICAL_LAYOUT": slime_layout},
    )
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise AssertionError(
            f"{SGLANG_ACCEPTED_LAYOUTS_CONSTANT} must be a tuple[str, ...], got {value!r}"
        )
    return value


def validate_contract() -> None:
    slime_layout = load_slime_layout()
    accepted_layouts = load_sglang_accepted_layouts(slime_layout)
    if slime_layout not in accepted_layouts:
        raise AssertionError(
            f"Slime emits layout {slime_layout!r}, but SGLang accepts only {accepted_layouts!r}"
        )
    for alias in ("q_spans", "query_spans", "span"):
        if alias not in accepted_layouts:
            raise AssertionError(
                f"SGLang sparse IR compatibility alias {alias!r} is missing"
            )


def main() -> int:
    validate_contract()
    print("OK: Slime EBFT sparse IR layout is accepted by SGLang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
