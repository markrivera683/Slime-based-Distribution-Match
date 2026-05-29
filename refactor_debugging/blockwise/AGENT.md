# Blockwise EBFT Agent Notes

Date: 2026-05-28

Scope: `refactor_debugging/blockwise/` is a CPU-only preflight workspace for
strict EBFT block prediction. It is used to reason about layout, source/target
logit indexing, QA/doc metadata, and next-token masks before touching GPU
rollout or Megatron training code.

## Current Status

- Added `inspect_blockwise_contract.py`.
  - Runs without GPU or model weights.
  - Builds a tiny fixture with `prompt_length=6`, `context_length=2`,
    `generate_length=2`, `stride=2`.
  - Prints time-major response layout, full sequence, QA/doc suffixes,
    position ids, strict source-target rows, standard next-token rows, and a
    fake-logits gather demo.
  - Supports `--json` for machine-readable inspection.

- Added `README.md`.
  - Documents the purpose of the blockwise preflight tools.
  - Explains the difference between standard next-token shift and strict
    OpenRLHF EBFT source-row gather.

- Added CPU-only tests in the repo test area.
  - `tests/test_strict_blockwise_contract.py` checks the tiny blockwise
    contract and fake-gather behavior.
  - `pytest` was not available in the environment during this pass, so tests
    were verified by `py_compile` and manual function calls.

- Important correction found during source inspection:
  - OpenRLHF strict block prediction uses first-step source rows
    `context_length + block_idx * stride - 1`.
  - For the tiny fixture, strict generated/action source rows are
    `[1, 3, 6, 7]`.
  - The standard generated next-token source rows are `[5, 6, 7, 8]`.
  - Therefore strict EBFT cannot be implemented as raw standard shift over the
    generated region.

## Current Implementation Branch

Active implementation work was moved to a copied repo:

`/mnt/data/distribution-matching-slime/code/slime_tmp`

Branch:

`dev/blk`

That branch contains the first core-code pass:

- `--g1-ebft-logprob-indexing {standard_next_token, strict_block_source}`
- strict pair-axis helpers in `slime/utils/g1_ebft_loss.py`
- pair-axis fields attached in `slime/utils/g1_ebft_data_contract.py`
- pair-axis logprob gather in `slime/backends/megatron_utils/loss.py`
- default behavior remains `standard_next_token`

## Key Contract

For a full sequence of length `L = prompt_length + response_length`, the EBFT
loss still consumes a pair axis of length `L - 1`.

Standard indexing:

```text
source_rows   = [0, 1, 2, ..., L-2]
target_tokens = [1, 2, 3, ..., L-1]
```

Strict block-source indexing:

```text
prompt/CE pairs: standard source row -> target token
generated/action pairs: OpenRLHF block-pred source row -> generated target
```

Tiny fixture:

```text
target_positions       = [1,2,3,4,5,6,7,8,9]
strict_source_rows     = [0,1,2,3,4,1,3,6,7]
standard_source_rows   = [0,1,2,3,4,5,6,7,8]
strict action sources  = [1,3,6,7]
standard action rows   = [5,6,7,8]
```

## Plan

1. Keep this directory CPU-only.
   - Do not import Megatron, SGLang, Ray, or CUDA-only paths here.
   - Keep scripts runnable from repo root with plain `python`.

2. Use the preflight contract as the source of truth for future strict work.
   - Any strict implementation must reproduce the source-target pair axis.
   - Any change to off-by-one behavior should be checked against OpenRLHF
     `prepare_logprobs()` and `generate_strided_blocks()`.

3. In `slime_tmp/dev/blk`, finish CPU tests for the new core helpers.
   - Pair-axis helper tests.
   - Data-contract attach tests.
   - Fake-logits gather tests.
   - Dispatch tests for `standard_next_token` vs `strict_block_source`.

4. After CPU tests pass, move to GPU smoke only when cards are available.
   - First target: THD, CP=1, TP=1, tiny microbatch.
   - Check finite loss/grad and runtime dumps.
   - Do not claim full strict EBFT parity until actor forward also uses strict
     attention mask and position ids.

5. Later strict parity work.
   - Add strict rollout actor or model-forward backend.
   - Share one dense-mask/position-id builder across rollout, actor logprob,
     ref/teacher logprob, and embedding paths.
   - Compare against OpenRLHF on tiny deterministic fixtures.

## Validation Commands

From repo root:

```bash
python refactor_debugging/blockwise/inspect_blockwise_contract.py
python refactor_debugging/blockwise/inspect_blockwise_contract.py --json
python -m py_compile refactor_debugging/blockwise/inspect_blockwise_contract.py
```

When `pytest` is available:

```bash
pytest tests/test_strict_blockwise_contract.py
```
