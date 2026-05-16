# Step 4: Minimal Slime G1 Smoke Configuration

## Goal

Run the smallest slime G1 smoke after Step 3 embedding contract tests pass. The active smoke now targets the Megatron/fast path:

- SGLang rollout produces fixed 376-token responses.
- `slime.rollout.g1_embedding.generate_fixed_length_for_g1` does not attach rollout-side embeddings.
- `RolloutManager` prepares `g1_full_sequences` / `g1_qa_masks`.
- `MegatronTrainRayActor` switches to the frozen `ref` snapshot, captures hidden states, and writes `g1_token_advantages`.
- `advantage_estimator=g1` consumes precomputed token advantages.
- Loss is finite for one to a few train steps.

This smoke does not yet prove OpenRLHF loss exactness, because slime still uses PPO-style policy loss until Step 5.

**Scope:** the old `generate_with_g1_embeddings` + OpenRLHF `Critic` path is now a correctness stash only, because Slime/SGLang and OpenRLHF need incompatible Transformers versions for Qwen3.5. The smoke should use trainer-side Megatron `ref` embeddings via `--g1-embedding-source megatron_ref --g1-reward-location trainer`.

## Smoke Status

PASSED on 2026-05-16 with the maintained script:

```bash
refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

The passing run used `TP=4`, `PP=2`, `DP=1`, `CP=1`, wrote one checkpoint to `/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_0516_001913/mcore`, and reached a finite actor train step with `g1_megatron_embeddings_time ~= 11.3s`.

A second group-aligned DP smoke also passed with `TP=2`, `PP=1`, `DP=4`, `CP=1`, `ROLLOUT_BATCH_SIZE=4`, and output `/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_0516_013755/mcore`.

Use `PRINT_ONLY=1` or `DRY_RUN_ONLY=1` with the script to print the composed training command without starting Ray. The script preflights `REF_LOAD`, `SLIME_TRAIN_DATA`, `CONTEXT_PARALLEL_SIZE=1`, `BALANCE_DATA=false`, and `ROLLOUT_BATCH_SIZE % DP_SIZE == 0`.

## Required Custom Paths

```text
--custom-generate-function-path slime.rollout.g1_embedding.generate_fixed_length_for_g1
```

## Required G1 Arguments

```text
--advantage-estimator g1
--use-whitening
--alignment-rew-coef 1.0
--diversity-rew-coef 1.0
--rollout-max-response-len 376
--n-samples-per-prompt 4
--g1-embedding-source megatron_ref
--g1-reward-location trainer
```

Do not pass `--normalize-advantages` for the first G1 smoke. It is false by default and should remain false to preserve the precomputed G1 advantage distribution.

## Required Embedding Arguments

The active Megatron smoke does **not** pass `--g1-critic-model-path`; it uses the frozen Megatron `ref` snapshot. The old HF/OpenRLHF critic path still requires a compatible HF checkpoint, but it is not part of this smoke.

```text
--g1-prompt-length 384
--g1-context-length 8
--g1-generate-length 8
--g1-stride 8
--g1-response-length 376
--g1-hidden-state-method last_only
--g1-tokenizer-path <same tokenizer path as policy, usually --hf-checkpoint>
--g1-embedding-source megatron_ref
--g1-reward-location trainer
```

First version keeps these flags off:

```text
# do not pass:
--g1-qa-masking
--g1-document-masking
```

This matches diff-dataset G1 defaults where `qa_masking=False` and `document_masking=False`.

## Minimal Smoke Template

Add the following arguments to the existing verified slime baseline launch command:

```bash
G1_ARGS=(
  --advantage-estimator g1
  --custom-generate-function-path slime.rollout.g1_embedding.generate_fixed_length_for_g1
  --use-whitening
  --alignment-rew-coef 1.0
  --diversity-rew-coef 1.0
  --rollout-max-response-len 376
  --n-samples-per-prompt 4
  --g1-prompt-length 384
  --g1-context-length 8
  --g1-generate-length 8
  --g1-stride 8
  --g1-response-length 376
  --g1-hidden-state-method last_only
  --g1-tokenizer-path "${HF_CHECKPOINT}"
  --g1-embedding-source megatron_ref
  --g1-reward-location trainer
)
```

The smoke should run with:

- very small `--rollout-batch-size`
- `--num-steps-per-rollout 1` or equivalent one-step setup
- no async
- no OPD
- no dynamic filtering
- no `--balance-data` for trainer-side G1 unless group-level balancing is implemented
- for DP>1, use group-aligned DP split and require `rollout_batch_size % dp_size == 0`
- first smoke can use `TP=4`, `PP=2`, `DP=1`, `CP=1`
- DP smoke can use `TP=2`, `PP=1`, `DP=4`, `CP=1`, `ROLLOUT_BATCH_SIZE=4`, `GLOBAL_BATCH_SIZE=16`

## Expected Runtime Fields

After train-data conversion, every local training sample must have:

```text
rollout_data["g1_full_sequences"]  # [760]
rollout_data["g1_qa_masks"]        # [760]
```

Before `compute_advantages_and_returns`, trainer-side Megatron G1 must add:

```text
rollout_data["g1_token_advantages"] # one [376] tensor per sample
```

## Pass Criteria

The smoke passes only if:

1. Rollout does not truncate or stop early.
2. Every response has length 376.
3. Every group has exactly 4 samples in rollout order.
4. Trainer-side Megatron hidden capture produces `[47, hidden_dim]` gen/gt embeddings internally.
5. `g1_token_advantages` length equals 376.
6. `RolloutManager._convert_samples_to_train_data` includes `g1_full_sequences` / `g1_qa_masks`.
7. `MegatronTrainRayActor` writes CUDA float32 `g1_token_advantages`.
8. `compute_advantages_and_returns(..., advantage_estimator="g1")` runs.
9. Actor loss is finite for at least one train step.

## Expected Failures

Fail loudly and fix before scaling if:

- prompt + label does not fit in `g1_prompt_length`
- response length is not exactly 376
- `--ref-load` is missing or no `ref` snapshot is available
- context parallel size is greater than 1 (first fast path fails loudly)
- `--normalize-advantages` is enabled
- `--group-rm` / rollout-side custom RM is accidentally enabled for this trainer-side path
- `balance_data=true` is enabled for trainer-side G1
- `rollout_batch_size` / prompt group count cannot be evenly split across DP ranks

## After Smoke

If the smoke passes, the next decision is whether to:

1. keep using `ref` as the frozen embedding source for short experiments,
2. add an independent Megatron embedding/critic role with separate checkpoint loading, or
3. add EBFTPolicyLoss + CE to move closer to full OpenRLHF G1 exactness.

Current branch decision: keep the PPO-style loss for now and do not call the path full exact until `g1_runtime_parity_plan.md` and the Step 5 loss decision are resolved.
