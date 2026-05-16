# G1 Megatron/Ref Smoke Follow-up Artifact

## Summary

本轮把已通过的 `/tmp/run_g1_smoke_megatron_ref.sh` 从临时脚本提升为 slime workspace 内的维护脚本，并补上 trainer-side G1 在 `DP>1` 时最关键的 correctness 约束：**DP split 必须按 prompt group 对齐**，不能按单条 sample 拆散 `n_samples_per_prompt` 组。

当前 active path 仍是：

```text
SGLang rollout
  -> generate_fixed_length_for_g1
  -> RolloutManager prepares g1_full_sequences / g1_qa_masks
  -> MegatronTrainRayActor switches to ref
  -> forward_only captures decoder hidden
  -> trainer-side G1 embeddings / pointwise reward / RLOO
  -> g1_token_advantages
  -> advantage_estimator=g1
  -> actor_train
```

## Implemented

### 1. Maintained Smoke Script

新增：

- `refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh`

该脚本保留在 slime 的 G1 plan 下，因为它验证的是 slime 的 Megatron/ref G1 fast path；EBFT repo 只作为 diff-dataset runner 和数据约定来源。

脚本默认使用：

```text
MODEL_PATH=/mnt/data/models/Qwen3.5-4B
REF_LOAD=/mnt/data/models/Megatron_convert_models/Qwen3.5-4B_torch_dist
SLIME_TRAIN_DATA=/tmp/g1_smoke_short.jsonl
TENSOR_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
RAY_TMPDIR=/tmp/ray_g1
```

关键 CLI：

```text
--custom-generate-function-path slime.rollout.g1_embedding.generate_fixed_length_for_g1
--use-whitening
--g1-tokenizer-path "${HF_CHECKPOINT}"
--g1-embedding-source megatron_ref
--g1-reward-location trainer
```

它明确不使用：

```text
--g1-critic-model-path
--custom-rm-path
--group-rm
```

### 2. Group-aligned DP Split

更新：

- `slime/ray/rollout.py`

当满足：

```text
advantage_estimator == "g1"
g1_embedding_source == "megatron_ref"
g1_reward_location == "trainer"
```

时，`_split_train_data_by_dp` 不再按 sample round-robin 分配，而是按 prompt group 分配。

例如 `n_samples_per_prompt=4`、4 个 prompt group、`dp_size=2`：

```text
global order:
  [p0_s0 p0_s1 p0_s2 p0_s3]
  [p1_s0 p1_s1 p1_s2 p1_s3]
  [p2_s0 p2_s1 p2_s2 p2_s3]
  [p3_s0 p3_s1 p3_s2 p3_s3]

rank0:
  [p0_s0 p0_s1 p0_s2 p0_s3]
  [p2_s0 p2_s1 p2_s2 p2_s3]

rank1:
  [p1_s0 p1_s1 p1_s2 p1_s3]
  [p3_s0 p3_s1 p3_s2 p3_s3]
```

This preserves the local assumption in `g1_fast.py` that embeddings are laid out as:

```text
[group0 x K], [group1 x K], ...
```

Failure conditions now include:

- `balance_data=true` for this path.
- sample count not divisible by `n_samples_per_prompt`.
- prompt group count not divisible by `dp_size`.

### 3. Tests

更新：

- `tests/test_g1_core.py`

新增 coverage：

- group-aligned DP partition keeps full prompt groups together.
- trainer-side G1 rejects `balance_data=true`.
- trainer-side G1 rejects uneven prompt group count across DP ranks.
- maintained smoke script uses `generate_fixed_length_for_g1`, `megatron_ref`, `trainer`, and does not use OpenRLHF Critic flags.

## Smoke Result Being Preserved

The successful smoke run was:

```text
Job: raysubmit_cghTC7hgqPu6u6kK
Output: /mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_0516_001913/mcore
Parallelism: TP=4, PP=2, DP=1, CP=1
```

Key logs:

```text
Rollout generation: 4/4
rollout/response_len/mean: 376.0
Timer g1_megatron_embeddings end: 12.7s
Timer actor_train end: 86.5s
train/loss: 5.587935447692871e-09
Job succeeded
exit_code: 0
```

The group-aligned DP smoke also passed:

```text
Job: raysubmit_ATzSrkrBUMpXAnnh
Output: /mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_0516_013755/mcore
Parallelism: TP=2, PP=1, DP=4, CP=1
Rollout generation: 16/16
rollout/response_len/mean: 376.0
Timer g1_megatron_embeddings end: 1.3s
Timer actor_train end: 28.7s
train/loss: 3.4924596548080444e-09
Job succeeded
exit_code: 0
```

## Debug Findings Preserved

1. **Ray tmp path must be short.**
   `RAY_TMPDIR=/tmp/ray_g1` avoids AF_UNIX socket path length failures.

2. **Megatron model must be recursively unwrapped for hooks.**
   Runtime object shape is `DDP -> Float16Module -> GPTModel`; the hidden hook must attach to the inner `GPTModel.decoder`.

3. **Sequence-parallel hidden must be gathered.**
   Without `gather_from_sequence_parallel_region`, the captured hidden can be only a shard (e.g. 384 tokens), not the full G1 sequence.

4. **Trainer-side G1 intermediates must not enter logging.**
   `g1_full_sequences` and `g1_qa_masks` are Long tensors and must be removed after `g1_token_advantages` are computed.

5. **DP must preserve prompt groups.**
   RLOO is group-relative. Sample-level DP split can silently mix prompts; group-aligned split is required for correctness.

6. **TP=8 is invalid for Qwen3.5-4B.**
   `num_query_groups=4`, so `tensor_model_parallel_size` must divide 4. The passing smoke used `TP=4, PP=2` to keep `DP=1`.

## What This Still Does Not Prove

- OpenRLHF Critic hidden and Megatron/ref hidden runtime numerical parity.
- EBFTPolicyLoss + CE exact actor loss parity.
- Arbitrary DP with `balance_data=true`.
- CP>1.
- Variable-length responses / early-stop samples.

## Next Recommended Work

1. Use `run_g1_megatron_ref_smoke.sh` as the canonical smoke entrypoint.
2. If scaling beyond the tested `DP=4`, keep `balance_data=false` and set `rollout_batch_size % dp_size == 0`.
3. Add runtime parity dumps comparing OpenRLHF hidden/groom with Megatron/ref hidden/groom.
4. Keep EBFTPolicyLoss deferred for the current branch; revisit only after runtime parity and loss diagnostics justify an exact-loss branch.

See also:

- `refactor_debugging/g1_plan/g1_runtime_parity_plan.md`
- `refactor_debugging/g1_plan/g1_runtime_parity_report.md`
- `refactor_debugging/g1_plan/step5_loss_decision.md`

Runtime parity result summary:

```text
Token IDs match: True
QA masks match: True
Full hidden cosine mean: 0.9591
Gen block embedding cosine mean: 0.9057
GT block embedding cosine mean: 0.9990
```

The generated-block gap is expected until Megatron/ref reproduces OpenRLHF's EBFT dense strided attention mask and position-id semantics.
