# Strict EBFT Blockwise Implementation Notes

This directory contains CPU-only debug material for the strict G1 EBFT
block-prediction contract. It is meant as a practical handoff note for future
agents/users before they involve SGLang, Ray, Megatron, or GPUs.

## Current Status

- CPU preflight: covered by `inspect_blockwise_contract.py`.
- CPU unit/smoke coverage: strict pair-axis helpers, argument validation, data
  contract wiring, loss gather behavior, and a tiny rollout-to-loss smoke are
  represented in tests.
- GPU/Ray parity: not established yet. The strict path is wired for the trainer
  batch/loss flow, but it still needs an end-to-end distributed run against the
  OpenRLHF/EBFT reference behavior.

## What `strict_block_source` Means

`--g1-ebft-logprob-indexing` selects how EBFT log-probs are gathered for
`--g1-use-ebft-loss`.

- `standard_next_token` is the default. It keeps the existing Slime convention:
  logit row `i` predicts token position `i + 1`.
- `strict_block_source` keeps prompt CE pairs on the normal next-token axis, but
  gathers generated/action log-probs from the EBFT block-prediction source row
  for each generated target token.

For a block target at `(step, block)`:

- `step == 0`: the source logit row is the prompt anchor row immediately before
  the generated target token for that block.
- `step > 0`: the source logit row is the previous generated token in the same
  block.

That source-row to target-token mapping is the key strict contract. Duplicate
source rows are expected in small fixtures because one prompt row may be used as
both a normal prompt CE source and a generated/action source.

## Why Standard SGLang/`generate` Is Not Enough

SGLang generation returns the normal autoregressive response stream. That is
useful for sampling, but strict EBFT block prediction trains on a different
layout:

- the response region is time-major: all blocks for step 0, then all blocks for
  step 1, and so on;
- generated tokens, QA masks, and document metadata are produced from the same
  strided prompt unfold, not from a synthetic all-one response mask;
- the generated/action loss does not always use the contiguous previous token as
  the logit source.

In other words, a standard `generate` response can tell us which sampled tokens
exist, but it cannot by itself tell the trainer which packed logit row should be
used for every strict EBFT generated target. The strict path therefore carries
explicit pair-axis metadata through rollout and batch construction.

## Block Geometry

The strict layout assumes:

```text
num_blocks = (prompt_length - generate_length - context_length) // stride + 1
response_length = generate_length * num_blocks
```

The default tiny fixture is:

```text
prompt_length=6
context_length=2
generate_length=2
stride=2
num_blocks=2
response_length=4
```

Its full position ids are:

```text
[0, 1, 2, 3, 4, 5, 2, 4, 3, 5]
```

The prompt-side positions are `0..5`. The generated region is time-major:
`step0/block0 -> 2`, `step0/block1 -> 4`, `step1/block0 -> 3`, and
`step1/block1 -> 5`.

For the same fixture, the complete pair axis is:

```text
pair_target_positions:   [1, 2, 3, 4, 5, 6, 7, 8, 9]
strict_pair_source_rows: [0, 1, 2, 3, 4, 1, 3, 6, 7]
```

The generated/action part of that strict source-target map is:

```text
source rows:      [1, 3, 6, 7]
target positions: [6, 7, 8, 9]
```

The standard contiguous next-token shift would mark generated action rows
`[5, 6, 7, 8]`. That is the core mismatch this implementation fixes.

## Rollout Metadata

When all of these are true:

- `--g1-use-ebft-loss`
- `--g1-ebft-logprob-indexing strict_block_source`
- rollout has built `g1_full_sequences`

the rollout conversion attaches:

- `ebft_logprob_source_rows`: one list per sample, length `full_sequence_len - 1`;
- `ebft_logprob_target_positions`: one list per sample, also length
  `full_sequence_len - 1`;
- `ebft_logprob_indexing`: the string `strict_block_source`.

These fields describe the log-prob pair axis used by the trainer. Prompt CE
pairs are standard next-token pairs; generated/action pairs use the strict block
source rows.

The batch/data contract then validates this metadata, converts it to tensors,
and also populates:

- `ebft_action_mask_next`
- `ebft_qa_mask_next`
- `ebft_advantages_next`
- `ebft_seq_len_m1`

If rollout pair metadata is absent, the trainer can rebuild the pair axis from
the strict G1 geometry. If rollout metadata is present, the trainer checks that
it matches the configured indexing and action-mask shape.

## Batch And Loss Flow

The strict trainer flow is:

1. Rollout builds `g1_full_sequences` and `g1_qa_masks` from the strided EBFT
   prompt layout.
2. Rollout attaches `ebft_logprob_*` pair-axis metadata when strict indexing is
   enabled.
3. `prepare_g1_ebft_tokens_for_batch` replaces `batch["tokens"]` with
   `batch["g1_full_sequences"]` for strict EBFT actor-loss batches, and resets
   `total_lengths`/`response_lengths` to the full strict sequence geometry.
4. `attach_ebft_g1_next_token_contract_to_batch` builds next-token action, QA,
   and advantage tensors and carries the strict log-prob pair axis into the
   Megatron batch.
5. `policy_loss_function_g1_ebft` gathers log-probs:
   - `standard_next_token`: dense full-sequence `row i -> token i + 1`;
   - `strict_block_source`: explicit `source_row -> target_position` pairs.
6. EBFT RL and prompt CE scalars are computed over the packed per-sample lists.

This path currently requires `qkv_format=thd`, context parallel world size 1,
`advantage_estimator=g1`, `loss_type=policy_loss`, no KL/OPSM term, and
`entropy_coef=0`.

## CLI Flags

Use the strict path with:

```bash
--g1-use-ebft-loss \
--g1-ebft-logprob-indexing strict_block_source
```

The flag default is:

```bash
--g1-ebft-logprob-indexing standard_next_token
```

Argument validation rejects `strict_block_source` unless the G1 strided block
geometry is valid and:

```text
g1_response_length == g1_generate_length * num_blocks
```

## Launching A Strict G1 EBFT GT Run

The production launcher default remains unchanged:

```bash
exper_scripts/main_test/run_g1_ebft_gt_qwen35_2b_main.sh
```

To opt into strict EBFT block-source indexing, use the wrapper:

```bash
exper_scripts/main_test/run_g1_ebft_gt_qwen35_2b_strict_block_source.sh
```

The wrapper sets:

```bash
G1_EBFT_LOGPROB_INDEXING=strict_block_source
```

and then delegates to the same G1 EBFT GT launcher. All existing environment
overrides still work, for example:

```bash
NUM_ROLLOUT=1 \
SGLANG_STABLE_ROLLOUT_MODE=true \
exper_scripts/main_test/run_g1_ebft_gt_qwen35_2b_strict_block_source.sh
```

For non-GPU command inspection only, use:

```bash
PRINT_ONLY=1 \
G1_FILTER_TRAIN_DATA=false \
exper_scripts/main_test/run_g1_ebft_gt_qwen35_2b_strict_block_source.sh
```

The run artifacts include `G1_EBFT_LOGPROB_INDEXING` in
`run_context.env`/`hyperparams.env`, and the printed command should include:

```bash
--g1-ebft-logprob-indexing strict_block_source
```

The launcher pins `SLIME_ROOT` to the checkout containing the script. This is
intentional: shared runtime env files such as `/root/slime_runtime/slime_env.sh`
may export an older `SLIME_ROOT`, but they should not redirect
`train_async.py`/`train.py` away from the active repository. `PRINT_ONLY=1`
prints explicit `SLIME_ROOT` and `TRAIN_ENTRY` preflight lines before the full
submitted command so the final train target is visible without scanning the
long argv.

## CPU Smoke Status

The CPU smoke currently verifies the tiny end-to-end strict data path without
running SGLang/Ray actors or CUDA kernels:

- samples are converted into strict `g1_full_sequences`;
- strict pair-axis metadata is attached as `[0, 1, 2, 3, 4, 1, 3, 6, 7]` for
  the tiny fixture;
- the batch preparation swaps model input to `g1_full_sequences`;
- the EBFT data contract materializes pair-axis tensors;
- the loss gathers gradients from strict source rows, not standard generated
  rows.

If Megatron is unavailable in the local environment, the strict smoke test is
expected to skip rather than fail.

## Validation Commands

From the repository root:

```bash
python refactor_debugging/blockwise/inspect_blockwise_contract.py
```

Machine-readable preflight:

```bash
python refactor_debugging/blockwise/inspect_blockwise_contract.py --json
```

Focused CPU tests:

```bash
pytest tests/test_g1_ebft_launcher_contract.py
pytest tests/test_strict_blockwise_contract.py
pytest tests/test_g1_ebft_arguments.py
pytest tests/test_g1_ebft_data_contract.py
pytest tests/test_g1_ebft_loss.py
pytest tests/test_g1_ebft_strict_smoke.py
```

Combined strict EBFT CPU pass:

```bash
pytest \
  tests/test_g1_ebft_launcher_contract.py \
  tests/test_strict_blockwise_contract.py \
  tests/test_g1_ebft_arguments.py \
  tests/test_g1_ebft_data_contract.py \
  tests/test_g1_ebft_loss.py \
  tests/test_g1_ebft_strict_smoke.py
```

## Debug Preflight Script

`inspect_blockwise_contract.py` uses only the Python standard library. It does
not import torch and does not allocate CUDA tensors.

The script prints the strict block-prediction layout and a fake-logits gather
demo. The fake table is readable as `logits[source_row][target_token]`, with
scores defined by:

```text
score = source_row * 1000 + target_token
```

For the default fixture, strict generated target gathers read rows
`[1, 3, 6, 7]`, while standard generated target gathers read rows
`[5, 6, 7, 8]`, producing visibly different scores.

The defaults include non-all-one QA/doc metadata so generated metadata flow is
easy to inspect. You can override the tiny prompt:

```bash
python refactor_debugging/blockwise/inspect_blockwise_contract.py \
  --prompt-length 6 \
  --context-length 2 \
  --generate-length 2 \
  --stride 2 \
  --prompt-tokens 10,11,12,13,14,15 \
  --prompt-qa 0,0,1,0,0,1 \
  --prompt-doc 0,0,0,0,1,1
```

The command exits nonzero if any built-in contract check fails.

## Current Limitations

- No GPU/Ray parity run has been completed for this strict path.
- No confirmed parity report against OpenRLHF EBFT runtime tensors/logprobs is
  checked in yet.
- Context parallelism is intentionally out of scope for the strict EBFT actor
  loss path.
- KL/OPSM, entropy loss, and non-`policy_loss` modes are intentionally rejected
  while the strict contract is being stabilized.
- Standard SGLang response logprobs should not be treated as strict EBFT
  generated/action logprobs unless they have been remapped to this explicit pair
  axis.

## Next Steps

1. Run the focused CPU tests above in a clean environment with Megatron
   importable and record the result.
2. Add a small GPU single-rank/Ray validation that dumps `ebft_logprob_*`,
   packed token lengths, and gathered log-probs.
3. Compare strict Slime gathered log-probs and masks against the EBFT/OpenRLHF
   reference path for the same tiny fixture.
4. Only after parity is established, consider broadening support for context
   parallelism, KL/OPSM, entropy, or rollout-provided strict logprobs.
