#!/usr/bin/env python3
"""
Post-eval analysis: reads batch_inference output (eval_results.jsonl) and the
original eval dataset, joins on prompt text, runs math_verify accuracy check,
and writes a detailed JSON report + prints a summary.

Usage:
  python scripts/analyze_eval_results.py \
      --eval_results /root/outputs/run_xxx/eval_results.jsonl \
      --eval_dataset /mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl \
      --input_key question --label_key answer \
      [--report_path /root/outputs/run_xxx/eval_analysis.json]
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from math_verify import parse, verify
    from openrlhf.utils.math_verifier import get_llm_answer, verify_llm_answer
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


# ---------------------------------------------------------------------------
# Plan-B fallback answer extractor.
#
# Background: the ebft-distill model frequently writes a plain-English answer
# at the end of its reasoning ("Therefore, the answer is 5.") but does NOT
# wrap it in `\boxed{...}`. `get_llm_answer` only looks for boxed/parseable
# top-level math, so these answers get bucketed into `reasoning_incomplete`
# and obscure the real signal -- the model produced an answer, it's just
# missing the SFT-era structural marker.
#
# This fallback runs ONLY when the primary extractor fails. It scans the
# trailing region of the output for a small set of high-precision patterns
# and returns the first parseable candidate. False positives become
# `wrong_answer` (still informative) instead of false-negative
# `reasoning_incomplete`.
# ---------------------------------------------------------------------------

# Patterns are (regex, group_idx_for_answer). Each is anchored toward end-of-
# output via the surrounding driver code (we only feed in the trailing tail).
# Order matters: more specific patterns first.
_FALLBACK_PATTERNS = [
    # "Final Answer: X" / "Final Answer is X" / "The final answer is X"
    re.compile(
        r"(?:final\s+answer|the\s+final\s+answer)\s*(?:is|[:=])\s*([^.\n<]+?)"
        r"(?:\.|<eos>|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    # "The answer is X" / "Answer: X" / "Answer = X"
    re.compile(
        r"(?:the\s+answer|answer)\s*(?:is|[:=])\s*([^.\n<]+?)"
        r"(?:\.|<eos>|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Therefore/Thus/Hence, ... is/equals/= X" -- last one only.
    # Use lazy match so the first "is/=" doesn't swallow earlier filler.
    re.compile(
        r"(?:therefore|thus|hence|so)[,\s]+[^.\n]*?"
        r"(?:\bis\b|\bequals\b|=)\s*([^.\n<]+?)"
        r"(?:\.|<eos>|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Trailing "= X." or "= X<eos>" on the very last line — common in
    # algebraic derivations that just stop after the punchline.
    re.compile(
        r"=\s*([^=\n<]+?)\s*(?:\.\s*<eos>|<eos>|\.\s*\Z|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
]


def _safe_parse(s):
    """Wrap math_verify.parse in a timeout-tolerant try/except.

    parse() has been observed to hang/timeout on pathological strings.
    The driver (`run_with_timeout` upstream) handles process-level timeouts;
    here we just absorb exceptions so a single bad candidate doesn't kill
    the whole eval.
    """
    if not HAS_MATH_VERIFY:
        return None
    try:
        return parse(s)
    except Exception:
        return None


def extract_answer_with_fallback(text, max_tail_chars=2000):
    """Try fallback strategies after \\boxed-style extraction has failed.

    Returns (parsed_expr, pattern_name) or (None, None).
    """
    if not text or not HAS_MATH_VERIFY:
        return None, None
    tail = text[-max_tail_chars:]
    for pat in _FALLBACK_PATTERNS:
        # Use the LAST match (rightmost in the tail) since answers come at
        # the end. finditer + take last is cheap on a 2000-char string.
        last = None
        for m in pat.finditer(tail):
            last = m
        if last is None:
            continue
        candidate = last.group(1).strip().strip(".,;:$ \t")
        if not candidate or len(candidate) > 200:
            continue
        # Try parsing as raw, then as $...$ math, then as \\boxed{...}
        for wrapped in (candidate, f"${candidate}$", f"\\boxed{{{candidate}}}"):
            parsed = _safe_parse(wrapped)
            if parsed:
                return parsed, pat.pattern[:60] + "..."
    return None, None


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_dataset(path, input_key, label_key):
    """Load gold labels from the eval dataset file or HF-format folder.

    Returns a tuple ``(text_lookup, gold_by_idx)``:

    - ``text_lookup``: ``{prompt_text[:500].strip(): gold}`` — legacy match
      path; fails when the prediction's ``input`` field has been wrapped with
      a prompt template (e.g. ``"Problem: {q}\\n\\nSolution: "``) because the
      prefix shifts the first 500 chars away from the bare question.
    - ``gold_by_idx``: ``[gold_for_row_0, gold_for_row_1, ...]`` — robust path
      keyed by ``source_idx`` (which vllm_generate_progress.py always writes
      and which corresponds to the row position in the eval dataset for stage1
      and to the original test_qa.jsonl position for stage2 retry subsets).

    The caller should try ``gold_by_idx`` first and fall back to ``text_lookup``
    so older eval_results without ``source_idx`` still work.
    """
    if path.endswith(".jsonl") or path.endswith(".json"):
        rows = load_jsonl(path)
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset(path, split="test")
            rows = [dict(r) for r in ds]
        except Exception:
            rows = load_jsonl(os.path.join(path, "test.jsonl"))

    text_lookup = {}
    gold_by_idx = []
    for r in rows:
        gold = r.get(label_key, "")
        gold_by_idx.append(gold)
        prompt_text = r.get(input_key, "")
        if prompt_text:
            key = prompt_text.strip()[:500]
            text_lookup[key] = gold
    return text_lookup, gold_by_idx


# Common EOS-style markers we treat as "the model immediately gave up". This
# list is intentionally cross-tokenizer (Gemma / Llama / Qwen / GPT-NeoX) since
# `analyze_eval_results.py` is consumed by all of G1/G2/G3/baseline/Teacher.
# A pure-EOS output is a strong "early-quit" signal worth surfacing on its own;
# previously these got bucketed into `too_short`, hiding ~16% of stage1 outputs
# behind a generic label.
_PURE_EOS_MARKERS = (
    "<eos>",
    "<|endoftext|>",
    "</s>",
    "<|im_end|>",
    "<|end|>",
    "<|end_of_text|>",
)


def _is_pure_eos(model_output: str) -> bool:
    """True iff the entire output is whitespace + at most one EOS-like marker."""
    s = model_output.strip()
    if not s:
        return False  # this is `empty_output`, classified separately
    # Allow leading/trailing whitespace and a single EOS marker, nothing else.
    for m in _PURE_EOS_MARKERS:
        if s == m:
            return True
    return False


def classify_output(model_output, gold_answer):
    """Classify a single (model_output, gold_answer) pair."""
    if not HAS_MATH_VERIFY:
        return None, "no_math_verify", "math_verify not installed"

    if not gold_answer or not gold_answer.strip():
        return None, "missing_gold", "Gold answer is empty"

    gold_boxed = parse(f"\\boxed{{{gold_answer}}}")
    if not gold_boxed:
        gold_boxed = parse(gold_answer)
    if not gold_boxed:
        return None, "unparseable_gold", f"Cannot parse gold: {gold_answer[:80]}"

    if not model_output or not model_output.strip():
        return False, "empty_output", "Model produced no output"

    # Promote pure-EOS to its own category before the generic "too_short" path.
    # This lets us see how often the model is *immediately giving up* (a known
    # failure mode with packed-stream distill training) vs. just generating a
    # short non-answer.
    if _is_pure_eos(model_output):
        return False, "pure_eos", "Model emitted only an EOS marker (immediate quit)"

    # Plan-B: try BOTH the primary (math_verify-style) extractor and the
    # plain-English fallback. The primary extractor is permissive enough that
    # it almost always returns *something* (often the rightmost LaTeX-ish
    # token in the text), so if we only ran fallback when primary returned
    # None we'd never get any rescues. Instead we try primary first; if its
    # candidate verifies, great; if not, we *also* try fallback. The fallback
    # operates on the trailing region only, where natural-language answer
    # statements tend to live.
    pred_primary, resp_type = get_llm_answer(model_output)
    pred_fallback, fallback_pat = extract_answer_with_fallback(model_output)

    raw_gold = parse(gold_answer)

    def _verify(pred):
        if not pred:
            return False
        try:
            if verify(pred, gold_boxed):
                return "boxed"
        except Exception:
            pass
        if raw_gold:
            try:
                if verify(pred, raw_gold):
                    return "raw"
            except Exception:
                pass
        return False

    # Resolve correctness in priority order: primary-vs-boxed > primary-vs-raw
    # > fallback-vs-boxed > fallback-vs-raw. The fallback path needs explicit
    # tagging in the category so we can audit/contrast strict-vs-fallback acc.
    primary_match = _verify(pred_primary)
    if primary_match == "boxed":
        return True, "correct", ""
    if primary_match == "raw":
        return True, "correct_raw_match", "Matches raw (non-boxed) gold"

    fallback_match = _verify(pred_fallback)
    if fallback_match == "boxed":
        return True, "correct_fallback", \
            f"Plain-text answer matched gold (via {fallback_pat})"
    if fallback_match == "raw":
        return True, "correct_raw_match_fallback", \
            f"Plain-text answer matched raw gold (via {fallback_pat})"

    # Neither extractor produced a verifying answer. Classify the failure.
    if pred_primary is None and pred_fallback is None:
        if len(model_output.strip()) < 30:
            return False, "too_short", "Output too short, no parseable answer"
        return False, "no_answer_extracted", "Cannot extract answer from model output"

    out_lower = model_output.lower()
    has_steps = bool(re.search(
        r'step\s*\d|first|then|therefore|thus|hence|so\s+we|let\s+', out_lower))
    has_eq = bool(re.search(r'[=<>]', model_output))
    has_boxed = "\\boxed" in model_output

    # Both wrong: prefer the more informative tag.
    # \boxed{} present + wrong -> classic wrong_answer (model committed via marker)
    # fallback found a plain-text answer (typically clearer than primary's
    #   middle-of-text math expr) -> wrong_answer_fallback (committed in prose)
    # Otherwise we fall through to the heuristic reasoning/calculation buckets.
    if has_boxed:
        return False, "wrong_answer", \
            f"Has \\boxed but answer is wrong (resp_type={resp_type})"
    if pred_fallback is not None:
        return False, "wrong_answer_fallback", \
            f"Plain-text answer extracted but wrong (via {fallback_pat})"
    if has_steps and has_eq:
        return False, "reasoning_incomplete", "Has reasoning steps but wrong/no final answer"
    if has_eq and not has_steps:
        return False, "calculation_error", "Has equations but answer is wrong"

    return False, "no_reasoning", "No recognisable reasoning towards the answer"


def _quantile(sorted_values, p):
    """Nearest-rank quantile on a presorted list. Returns 0 for empty input."""
    if not sorted_values:
        return 0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    idx = int(len(sorted_values) * p / 100)
    if idx >= len(sorted_values):
        idx = len(sorted_values) - 1
    return sorted_values[idx]


def _try_load_tokenizer(tokenizer_path):
    """Best-effort load of an HF tokenizer. Returns (tokenizer, error_str_or_None)."""
    if not tokenizer_path:
        return None, "no --tokenizer_path provided"
    if not os.path.exists(tokenizer_path):
        return None, f"path does not exist: {tokenizer_path}"
    try:
        from transformers import AutoTokenizer  # local import: optional dep
    except Exception as exc:  # pragma: no cover - env-dependent
        return None, f"transformers not importable: {exc}"
    try:
        tk = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        return None, f"AutoTokenizer.from_pretrained failed: {exc}"
    return tk, None


def main():
    parser = argparse.ArgumentParser(description="Analyze eval_results.jsonl")
    parser.add_argument("--eval_results", type=str, required=True,
                        help="Path to eval_results.jsonl from batch_inference")
    parser.add_argument("--eval_dataset", type=str, required=True,
                        help="Path to eval dataset (jsonl/json/HF folder)")
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--label_key", type=str, default="answer")
    parser.add_argument("--report_path", type=str, default=None,
                        help="Output JSON report path (default: eval_analysis.json next to eval_results)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help=("Optional HF tokenizer path (usually MODEL_PATH). "
                              "If provided, output lengths are also reported in tokens "
                              "and the fraction hitting --max_new_tokens is computed."))
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help=("Generation cap that produced these results "
                              "(e.g. vLLM SamplingParams.max_tokens). Used only "
                              "to compute the 'hit cap' fraction; requires --tokenizer_path."))
    parser.add_argument("--hit_cap_tolerance", type=int, default=4,
                        help=("Treat tokens >= max_new_tokens - tolerance as 'hit cap'. "
                              "Tolerance covers small discrepancies between vLLM's "
                              "internal detokenize/retokenize roundtrip and ours."))
    args = parser.parse_args()

    if not os.path.isfile(args.eval_results):
        print(f"[ERROR] eval_results not found: {args.eval_results}")
        sys.exit(1)

    if args.report_path is None:
        base_dir = os.path.dirname(args.eval_results)
        args.report_path = os.path.join(base_dir, "eval_analysis.json")

    print("=" * 70)
    print("  Post-Eval Analysis")
    print("=" * 70)

    print(f"\n[1] Loading eval results: {args.eval_results}")
    results = load_jsonl(args.eval_results)
    print(f"    Loaded {len(results)} predictions")

    print(f"\n[2] Loading gold labels: {args.eval_dataset}")
    text_lookup, gold_by_idx = load_eval_dataset(
        args.eval_dataset, args.input_key, args.label_key
    )
    print(f"    Loaded {len(gold_by_idx)} gold entries "
          f"(text-keys: {len(text_lookup)})")

    print(f"\n[3] Matching predictions to gold answers ...")
    matched = 0
    unmatched = 0
    matched_via_idx = 0
    matched_via_text = 0
    records = []

    for i, r in enumerate(results):
        prompt = r.get("input", "")
        model_output = r.get("output", "")
        # Robust matching: prefer ``source_idx`` (always written by
        # vllm_generate_progress.py and stable under prompt-template wrapping)
        # and only fall back to text-based lookup when source_idx is absent or
        # out of range. This fixes the previous failure mode where adding a
        # prompt template like "Problem: {q}\n\nSolution: " caused the prompt
        # text in eval_results to no longer match the bare question text in
        # the gold dataset, sending 100% of rows into the `unmatched` bucket.
        gold = None
        match_via = None
        sidx_raw = r.get("source_idx", None)
        try:
            sidx = int(sidx_raw) if sidx_raw is not None else None
        except (TypeError, ValueError):
            sidx = None
        if sidx is not None and 0 <= sidx < len(gold_by_idx):
            gold = gold_by_idx[sidx]
            match_via = "source_idx"
            matched_via_idx += 1
        else:
            key = prompt.strip()[:500]
            gold = text_lookup.get(key, None)
            if gold is not None:
                match_via = "text"
                matched_via_text += 1

        if gold is None:
            unmatched += 1
            records.append({
                "idx": i,
                "source_idx": r.get("source_idx", i),
                "attempt_idx": r.get("attempt_idx", 0),
                "prompt": prompt[:200],
                "model_output": model_output[:500],
                "gold_answer": None,
                "is_correct": None,
                "category": "unmatched",
                "detail": "Could not find gold answer for this prompt",
            })
            continue

        matched += 1
        correct, category, detail = classify_output(model_output, gold)
        records.append({
            "idx": i,
            "source_idx": r.get("source_idx", i),
            "attempt_idx": r.get("attempt_idx", 0),
            "prompt": prompt[:200],
            "model_output": model_output[:500],
            "gold_answer": gold,
            "is_correct": correct,
            "category": category,
            "detail": detail,
            "match_via": match_via,
        })

    print(f"    Matched: {matched}, Unmatched: {unmatched}  "
          f"(via source_idx: {matched_via_idx}, via text: {matched_via_text})")

    evaluated = [r for r in records if r["is_correct"] is not None]
    n_correct = sum(1 for r in evaluated if r["is_correct"])
    n_evaluated = len(evaluated)

    cats = Counter(r["category"] for r in records)

    char_lengths = []
    for r in results:
        out = r.get("output", "")
        char_lengths.append(len(out))
    n_outputs = max(1, len(char_lengths))
    avg_chars = sum(char_lengths) / n_outputs
    empty_outputs = sum(1 for l in char_lengths if l < 5)
    char_lengths_sorted = sorted(char_lengths)

    # Optional token-length stats. Off by default; on when --tokenizer_path is
    # provided. We compute these here so the printed SUMMARY block can carry
    # both units side-by-side, and --max_new_tokens (if given) gives us an
    # "actually hit the cap" fraction (more useful than chars-vs-cap).
    tokenizer, tokenizer_err = _try_load_tokenizer(args.tokenizer_path)
    token_lengths = []
    avg_tokens = None
    token_quantiles = None
    hit_cap_count = None
    hit_cap_pct = None
    cap_used = None
    if tokenizer is not None:
        try:
            for r in results:
                out = r.get("output", "")
                if not out:
                    token_lengths.append(0)
                    continue
                # add_special_tokens=False so we measure the model's *generated*
                # token count, not BOS/EOS tokens injected by the tokenizer.
                token_lengths.append(len(tokenizer.encode(out, add_special_tokens=False)))
            avg_tokens = sum(token_lengths) / n_outputs
            tl_sorted = sorted(token_lengths)
            token_quantiles = {
                "p50": _quantile(tl_sorted, 50),
                "p90": _quantile(tl_sorted, 90),
                "p99": _quantile(tl_sorted, 99),
                "max": tl_sorted[-1] if tl_sorted else 0,
            }
            if args.max_new_tokens is not None and args.max_new_tokens > 0:
                cap_used = int(args.max_new_tokens)
                threshold = max(0, cap_used - max(0, args.hit_cap_tolerance))
                hit_cap_count = sum(1 for tl in token_lengths if tl >= threshold)
                hit_cap_pct = hit_cap_count / n_outputs * 100.0
        except Exception as exc:
            # Don't fail analysis just because tokenization blew up.
            print(f"  [WARN] token-length stats failed: {exc}")
            token_lengths = []
            avg_tokens = None
            token_quantiles = None
            hit_cap_count = None
            hit_cap_pct = None
            cap_used = None
    else:
        if args.tokenizer_path or args.max_new_tokens is not None:
            print(f"  [INFO] token-length stats disabled: {tokenizer_err}")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total predictions:      {len(results)}")
    print(f"  Matched to gold:        {matched}")
    print(f"  Evaluated (parseable):  {n_evaluated}")
    if n_evaluated > 0:
        acc = n_correct / n_evaluated * 100
        print(f"  Correct:                {n_correct}/{n_evaluated} ({acc:.1f}%)")
    else:
        acc = 0.0
        print(f"  Correct:                N/A (no evaluable samples)")
    if avg_tokens is not None:
        print(f"  Avg output length:      {avg_chars:.0f} chars  /  {avg_tokens:.0f} tokens")
        q = token_quantiles
        print(f"  Token quantiles:        p50={q['p50']}  p90={q['p90']}  "
              f"p99={q['p99']}  max={q['max']}")
        if cap_used is not None:
            print(f"  Hit max_new_tokens:     {hit_cap_count}/{n_outputs} "
                  f"({hit_cap_pct:.1f}%)   "
                  f"[cap={cap_used}, tol={args.hit_cap_tolerance}]")
    else:
        print(f"  Avg output length:      {avg_chars:.0f} chars   "
              f"(token stats unavailable; pass --tokenizer_path to enable)")
    print(f"  Empty/very short (<5 chars): {empty_outputs}")

    # Plan-A diagnostics: surface the "model immediately gave up" rate. We
    # already classified outputs above; just slice the records by category to
    # avoid re-tokenizing or re-string-matching anything.
    pure_eos_count = sum(1 for r in records if r.get("category") == "pure_eos")
    pure_eos_pct = pure_eos_count / len(records) * 100.0 if records else 0.0
    print(f"  Pure-EOS (immediate quit): {pure_eos_count}/{len(records)} "
          f"({pure_eos_pct:.1f}%)   "
          f"[matches one of {{{', '.join(_PURE_EOS_MARKERS)}}}]")

    print(f"\n  Category breakdown:")
    for cat, cnt in cats.most_common():
        pct = cnt / len(records) * 100
        print(f"    {cat:30s}  {cnt:4d}  ({pct:5.1f}%)")

    if any(r["category"] == "too_short" or r["category"] == "empty_output"
           for r in records):
        short_count = sum(1 for r in records
                         if r["category"] in ("too_short", "empty_output"))
        print(f"\n  [WARNING] {short_count} samples had too-short/empty output.")
        print(f"            Consider increasing max_new_tokens.")

    print(f"\n  Sample outputs per category:")
    shown = set()
    for cat_name in ["correct", "correct_fallback",
                     "correct_raw_match", "correct_raw_match_fallback",
                     "wrong_answer", "wrong_answer_fallback",
                     "reasoning_incomplete", "calculation_error",
                     "no_answer_extracted", "pure_eos", "too_short",
                     "empty_output", "no_reasoning", "unparseable_gold",
                     "unmatched"]:
        for r in records:
            if r["category"] == cat_name and cat_name not in shown:
                shown.add(cat_name)
                print(f"\n    [{cat_name}]")
                print(f"      Q: {r['prompt'][:120]}")
                print(f"      Gold: {str(r['gold_answer'])[:80]}")
                print(f"      Model: {r['model_output'][:200]}")
                break

    summary = {
        "total_predictions": len(results),
        "matched": matched,
        "matched_via_source_idx": matched_via_idx,
        "matched_via_text": matched_via_text,
        "unmatched": unmatched,
        "evaluated": n_evaluated,
        "correct": n_correct,
        "accuracy_pct": round(acc, 2),
        "avg_output_length_chars": round(avg_chars, 1),
        "char_length_quantiles": {
            "p50": _quantile(char_lengths_sorted, 50),
            "p90": _quantile(char_lengths_sorted, 90),
            "p99": _quantile(char_lengths_sorted, 99),
            "max": char_lengths_sorted[-1] if char_lengths_sorted else 0,
        },
        "empty_or_very_short": empty_outputs,
        "categories": dict(cats),
        "math_verify_available": HAS_MATH_VERIFY,
    }
    if avg_tokens is not None:
        summary["avg_output_length_tokens"] = round(avg_tokens, 1)
        summary["token_length_quantiles"] = token_quantiles
    else:
        summary["avg_output_length_tokens"] = None
        summary["token_length_quantiles"] = None
    if cap_used is not None:
        summary["max_new_tokens"] = cap_used
        summary["hit_max_new_tokens_count"] = hit_cap_count
        summary["hit_max_new_tokens_pct"] = round(hit_cap_pct, 2)
        summary["hit_max_new_tokens_tolerance"] = args.hit_cap_tolerance
    else:
        summary["max_new_tokens"] = None
        summary["hit_max_new_tokens_count"] = None
        summary["hit_max_new_tokens_pct"] = None
    # Plan-A diagnostics surfaced as top-level summary fields too, so that
    # downstream merge / dashboards don't have to dig into `categories`.
    summary["pure_eos_count"] = pure_eos_count
    summary["pure_eos_pct"] = round(pure_eos_pct, 2)
    summary["pure_eos_markers"] = list(_PURE_EOS_MARKERS)

    report = {
        "summary": summary,
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved to: {args.report_path}")
    print("=" * 70)

    return 0 if n_evaluated == 0 or acc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
