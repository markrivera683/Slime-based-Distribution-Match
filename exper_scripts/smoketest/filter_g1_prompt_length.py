#!/usr/bin/env python3
"""Filter Slime jsonl rows to the strict OpenRLHF G1 prompt+label budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _render_prompt(tokenizer, prompt: Any, *, apply_chat_template: bool) -> str:
    if not apply_chat_template:
        if isinstance(prompt, str):
            return prompt
        parts = []
        for message in prompt:
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return "\n".join(part for part in parts if part)

    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--max-prompt-label-len", type=int, default=384)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"[g1-filter] reuse existing {args.output}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    max_seen = 0
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rendered_prompt = _render_prompt(
                tokenizer,
                row.get(args.prompt_key),
                apply_chat_template=args.apply_chat_template,
            )
            label = row.get(args.label_key)
            if label is None:
                raise ValueError(f"row {line_no} is missing label key {args.label_key!r}")

            prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
            label_ids = tokenizer.encode(str(label), add_special_tokens=False)
            total_len = len(prompt_ids) + len(label_ids)
            max_seen = max(max_seen, total_len)
            if total_len <= args.max_prompt_label_len:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

    print(
        "[g1-filter] "
        f"input={args.input} output={args.output} kept={kept} dropped={dropped} "
        f"max_seen_prompt_label_len={max_seen} limit={args.max_prompt_label_len}"
    )
    if kept == 0:
        raise ValueError("G1 prompt-length filter kept zero rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
