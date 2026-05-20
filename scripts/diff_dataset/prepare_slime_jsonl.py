#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def convert_row(row: dict[str, Any], idx: int, input_key: str, label_key: str, system_prompt: str) -> dict[str, Any] | None:
    question = str(row.get(input_key) or row.get("question") or row.get("input") or "").strip()
    label = str(row.get(label_key) or row.get("answer") or row.get("output") or "").strip()
    if not question or not label:
        return None
    if system_prompt:
        question = f"{system_prompt.strip()}\n\n{question}"
    metadata = {k: v for k, v in row.items() if k not in {input_key, label_key, "question", "answer", "input", "output"}}
    metadata.setdefault("source_idx", row.get("source_idx", idx))
    return {"prompt": [{"role": "user", "content": question, "step_loss_mask": 1}], "label": label, "metadata": metadata}


def validate(path: Path, max_errors: int = 20) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    for line_no, row in enumerate(read_jsonl(path), 1):
        count += 1
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            errors.append(f"line {line_no}: prompt must be a non-empty list")
        elif not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            errors.append(f"line {line_no}: each prompt message needs role/content")
        if not str(row.get("label") or "").strip():
            errors.append(f"line {line_no}: empty label")
        if len(errors) >= max_errors:
            break
    return count, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert diff-dataset JSONL to slime prompt/label JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="question")
    parser.add_argument("--label-key", default="answer")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if args.validate_only:
        count, errors = validate(source)
        print(f"[validate] {source}: rows={count}, errors={len(errors)}")
        for err in errors:
            print(f"[ERROR] {err}")
        if errors:
            raise SystemExit(1)
        return

    def rows() -> Iterable[dict[str, Any]]:
        emitted = 0
        for idx, row in enumerate(read_jsonl(source)):
            converted = convert_row(row, idx, args.input_key, args.label_key, args.system_prompt)
            if converted is None:
                continue
            yield converted
            emitted += 1
            if args.max_samples > 0 and emitted >= args.max_samples:
                break

    count = write_jsonl(output, rows())
    valid_count, errors = validate(output)
    print(f"[convert] {source} -> {output}: wrote={count}, validated={valid_count}, errors={len(errors)}")
    for err in errors:
        print(f"[ERROR] {err}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
