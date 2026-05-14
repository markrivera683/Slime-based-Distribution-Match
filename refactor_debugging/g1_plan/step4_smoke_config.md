# Step 4: Minimal Slime G1 Smoke Configuration

## Goal

Run the smallest slime G1 smoke after Step 3 embedding contract tests pass. This smoke verifies:

- SGLang rollout produces fixed 376-token responses.
- `slime.rollout.g1_embedding.generate_with_g1_embeddings` writes `g1_gen_embedding` / `g1_gt_embedding`.
- `slime.rollout.rm_hub.g1_core.custom_rm` writes `g1_token_advantages`.
- `advantage_estimator=g1` consumes precomputed token advantages.
- Loss is finite for one to a few train steps.

This smoke does not yet prove OpenRLHF loss exactness, because slime still uses PPO-style policy loss until Step 5.

## Required Custom Paths

```text
--custom-generate-function-path slime.rollout.g1_embedding.generate_with_g1_embeddings
--custom-rm-path slime.rollout.rm_hub.g1_core.custom_rm
```

## Required G1 Arguments

```text
--advantage-estimator g1
--group-rm
--use-whitening
--alignment-rew-coef 1.0
--diversity-rew-coef 1.0
--rollout-max-response-len 376
--n-samples-per-prompt 4
```

Do not pass `--normalize-advantages` for the first G1 smoke. It is false by default and should remain false to preserve the precomputed G1 advantage distribution.

## Required Embedding Arguments

```text
--g1-prompt-length 384
--g1-context-length 8
--g1-generate-length 8
--g1-stride 8
--g1-response-length 376
--g1-hidden-state-method last_only
--g1-openrlhf-repo /mnt/data/ebft-distribution-new/code
--g1-critic-model-path <same HF path used by OpenRLHF G1 critic_pretrain>
--g1-tokenizer-path <same tokenizer path as policy, usually --hf-checkpoint>
--g1-embedding-device cuda
--g1-embedding-dtype bfloat16
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
  --group-rm
  --custom-generate-function-path slime.rollout.g1_embedding.generate_with_g1_embeddings
  --custom-rm-path slime.rollout.rm_hub.g1_core.custom_rm
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
  --g1-openrlhf-repo /mnt/data/ebft-distribution-new/code
  --g1-critic-model-path "${G1_CRITIC_MODEL_PATH}"
  --g1-tokenizer-path "${HF_CHECKPOINT}"
  --g1-embedding-device cuda
  --g1-embedding-dtype bfloat16
)
```

The smoke should run with:

- very small `--rollout-batch-size`
- `--num-steps-per-rollout 1` or equivalent one-step setup
- no async
- no OPD
- no dynamic filtering
- no `--balance-data` until the DP contract has been validated

## Expected Runtime Fields

After custom generate, every sample must have:

```text
metadata["g1_gen_embedding"]       # [47, hidden_dim]
metadata["g1_gt_embedding"]        # [47, hidden_dim]
```

After group RM, every sample must also have:

```text
metadata["g1_rewards"]             # [47]
metadata["g1_gt_rewards"]          # [47]
metadata["g1_diversity_rewards"]   # [47]
metadata["g1_rloo_baseline"]       # [47]
metadata["g1_token_advantages"]    # [376]
```

## Pass Criteria

The smoke passes only if:

1. Rollout does not truncate or stop early.
2. Every response has length 376.
3. Every group has exactly 4 samples.
4. Every sample has `[47, hidden_dim]` gen/gt embeddings.
5. `g1_token_advantages` length equals 376.
6. `RolloutManager._convert_samples_to_train_data` includes `g1_token_advantages`.
7. `MegatronTrainRayActor._get_rollout_data` moves them to CUDA float32 tensors.
8. `compute_advantages_and_returns(..., advantage_estimator="g1")` runs.
9. Actor loss is finite for at least one train step.

## Expected Failures

Fail loudly and fix before scaling if:

- prompt + label does not fit in `g1_prompt_length`
- response length is not exactly 376
- `--g1-critic-model-path` is missing
- OpenRLHF import path is unavailable
- `--normalize-advantages` is enabled
- group RM is disabled

## After Smoke

If the smoke passes, the next decision is whether to:

1. keep this as a slow correctness path and implement Megatron/fast embedding producer, or
2. first add EBFTPolicyLoss + CE to move closer to full OpenRLHF G1 exactness.
