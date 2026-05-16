# G1 Exact Implementation Artifact

## Summary

本轮把 G1 迁移从“reward/RLOO/token advantage 数学已对齐”推进到了“慢 HF/OpenRLHF embedding path 可以接入 slime rollout 小闭环”，随后又切换到 **Megatron/fast embedding path** 的第一版 trainer-side 实现。

**阶段定性（replan L29–31）：** 当前 `slime.rollout.g1_embedding` 等实现是 **temporary / debug training path**——**不修改 Megatron 内部**，在 rollout 之后、group RM 之前写入 embedding metadata；用于 strict 闭环与 parity。**后续计划仍是以 Megatron（或 fast path）替换该 stash**，本阶段不宣称已是最终训练架构。

更新：OpenRLHF in-process `Critic` 因 Qwen3.5 `transformers` / SGLang 环境冲突暂停。当前活动实现改为 `--g1-embedding-source megatron_ref --g1-reward-location trainer`：rollout 准备固定长度 G1 序列，Megatron actor 在训练侧切到 frozen `ref` snapshot，捕获 hidden，直接写 `g1_token_advantages`。

Smoke 更新：Megatron/ref trainer-side smoke 已在真实 Ray/Megatron 上通过，维护脚本为 `refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh`。通过配置包括 `TP=4, PP=2, DP=1, CP=1` 以及 group-aligned `TP=2, PP=1, DP=4, CP=1`，并成功完成 `g1_megatron_embeddings`、`actor_train` 和 checkpoint save。

没有编辑 `/root/.cursor/plans/g1-exact-replan_7cd78e3e.plan.md`。所有变更都落在 workspace 内。

## What Was Implemented

### 1. Updated G1 Planning Docs

更新了以下文档，使状态和当前代码一致：

- `refactor_debugging/g1_plan/g1_replan.md`
- `refactor_debugging/g1_plan/step0_g1_scope.md`
- `refactor_debugging/g1_plan/g1_flow_prompt_to_update.md`

关键更新：

- baseline 已由用户确认通过，不再阻塞 G1。
- Step 2 固定 embedding / helper-level parity 已完成。
- 下一步已从 Step 2 改为 Step 3：rollout embedding source。
- 明确 `run_openrlhf_runtime_dump_parity.sh` 只是 helper-level / fixed embedding parity，不是完整 `prompt -> rollout -> critic hidden -> groom` runtime parity。
- 明确当前 slime PPO-style loss 只能作为小闭环验证，不能宣称 full OpenRLHF G1 exact。

### 2. Added Step 3 Embedding Contract

新增：

- `refactor_debugging/g1_plan/step3_embedding_contract.md`

定义了第一版慢 HF/OpenRLHF embedding path 的契约：

- `prompt_length=384`
- `context_length=8`
- `generate_length=8`
- `stride=8`
- `num_blocks=47`
- `response_length=376`
- `n_samples_per_prompt=4`
- 每条 sample metadata:
  - `g1_gen_embedding`: `[47, hidden_dim]`
  - `g1_gt_embedding`: `[47, hidden_dim]`

第一版明确拒绝：

- truncation
- stop early
- variable-length response
- `response_length != 376`

### 3. Added Slow HF/OpenRLHF Embedding Producer

新增：

- `slime/rollout/g1_embedding.py`

主要内容：

- `G1EmbeddingConfig`
- `g1_embedding_config_from_args`
- `build_g1_prompt_inputs`
- `build_g1_full_sequence_inputs`
- `hidden_states_to_g1_embeddings`
- `SlowOpenRLHFG1EmbeddingProducer`
- `attach_g1_embeddings`
- `generate_with_g1_embeddings`

作用：

```text
custom_generate wrapper
  -> 调默认 SGLang generate
  -> 检查 response_length == 376
  -> 构造 OpenRLHF-style full_sequence / doc_ids / qa_masks
  -> 调慢 HF/OpenRLHF Critic
  -> hidden -> block embedding
  -> 写入 Sample.metadata["g1_gen_embedding"]
  -> 写入 Sample.metadata["g1_gt_embedding"]
```

历史接入方式（slow/correctness stash，不是当前 active smoke）：

```text
--custom-generate-function-path slime.rollout.g1_embedding.generate_with_g1_embeddings
--group-rm
--custom-rm-path slime.rollout.rm_hub.g1_core.custom_rm
```

当前 active smoke 使用 `generate_fixed_length_for_g1`，并在 trainer side 通过 `--g1-embedding-source megatron_ref --g1-reward-location trainer` 生成 `g1_token_advantages`。

### 4. Added G1 Embedding CLI Arguments

更新：

- `slime/utils/arguments.py`

新增参数：

```text
--g1-prompt-length
--g1-context-length
--g1-generate-length
--g1-stride
--g1-response-length
--g1-critic-model-path
--g1-tokenizer-path
--g1-openrlhf-repo
--g1-hidden-state-method
--g1-embedding-device
--g1-embedding-dtype
--g1-qa-masking
--g1-document-masking
```

其中 `--g1-critic-model-path` 必须是 **Transformers `AutoModelForCausalLM` 能加载的 HF 原生目录**（`config.json` 不得通过 `auto_map` 指向 `sglang.*`）。若 `--hf-checkpoint` / `MODEL_PATH` 已被 SGLang 改写，应单独准备一份 Hub 原始快照或 HF-only 副本给 critic。

这些参数让 G1 几何和模型路径通过命令行配置，而不是写死在代码里。

### 5. Tightened G1 RM Validation

更新：

- `slime/rollout/rm_hub/g1_core.py`

新增校验：

- gen / gt embedding shape 必须一致。
- 如果传入 `args.g1_num_blocks`，embedding block 数必须匹配。
- 如果传入 `args.g1_response_length`，sample response length 必须匹配。

这避免 `[47, D]` contract 被静默破坏。

### 6. Tightened Rollout To Train Data Contract

更新：

- `slime/ray/rollout.py`

之前逻辑只检查 `samples[0]` 是否有 `g1_token_advantages`。现在：

- 当 `advantage_estimator == "g1"` 时，每条 sample 都必须有 `metadata["g1_token_advantages"]`。
- 任意 sample 缺字段都会明确报错。

这修掉了一个容易隐藏的 G1 数据丢失 bug。

更新：trainer-side Megatron/ref G1 现在使用 group-aligned DP split。对于 `advantage_estimator=g1` + `g1_embedding_source=megatron_ref` + `g1_reward_location=trainer`，DP split 按 prompt group 分配，要求 `balance_data=false` 且 prompt group 数可被 DP 整除，避免把 `n_samples_per_prompt` 组拆散后错误计算 RLOO。该路径已通过 `DP=4` 真实 smoke。

### 7. Added Step 4 Smoke Configuration

新增：

- `refactor_debugging/g1_plan/step4_smoke_config.md`

里面写了最小 slime G1 smoke run 所需参数：

```text
--advantage-estimator g1
--custom-generate-function-path slime.rollout.g1_embedding.generate_fixed_length_for_g1
--use-whitening
--alignment-rew-coef 1.0
--diversity-rew-coef 1.0
--rollout-max-response-len 376
--n-samples-per-prompt 4
--g1-embedding-source megatron_ref
--g1-reward-location trainer
```

并说明第一版 trainer-side path 不要启用：

- async
- OPD
- dynamic filtering
- normalize advantages
- balance data

### 8. Added Step 5 Loss Decision Document

新增：

- `refactor_debugging/g1_plan/step5_loss_decision.md`

明确当前情况：

- 当前 slime 仍使用 PPO-style loss。
- OpenRLHF G1 使用 `EBFTPolicyLoss + CE`。
- 小闭环通过前不要改 loss。
- 若后续继续追求 full exact，需要补：
  - ratio 固定为 1
  - 无 PPO clip
  - prompt / QA 区域 CE
  - `ce_loss_coef=0.03`
  - OpenRLHF 风格 mask 与聚合

### 9. Expanded Tests

更新：

- `tests/test_g1_core.py`

测试数从 11 增加到 28。

新增覆盖：

- prompt + label packing contract
- answer mask / padding doc id contract
- fixed response length contract
- hidden states -> G1 block embedding order
- configured response length mismatch rejection
- G1 train_data 要求所有 sample 都有 `g1_token_advantages`
- DP split 保留 `g1_token_advantages`
- Megatron hidden -> G1 embeddings 与共享 block/groom helper 一致
- trainer-side G1 embedding/reward/RLOO 直接生成 token advantages
- `megatron_ref + trainer` 模式下 train_data 准备 `g1_full_sequences` / `g1_qa_masks`
- trainer-side G1 DP split 按 prompt group 对齐
- trainer-side G1 拒绝 `balance_data=true` 和无法按 DP 均分的 group 数
- smoke 脚本不使用 OpenRLHF Critic / `--g1-critic-model-path`

## Validation

已运行：

```bash
/root/venvs/slime/bin/python -m py_compile \
  slime/rollout/g1_embedding.py \
  tests/test_g1_core.py \
  refactor_debugging/g1_plan/check_slime_g1_dump_parity.py \
  refactor_debugging/g1_plan/dump_openrlhf_g1_runtime_fixture.py
```

已运行：

```bash
bash -n refactor_debugging/g1_plan/run_openrlhf_runtime_dump_parity.sh
```

已运行：

```bash
/root/venvs/slime/bin/python -m pytest tests/test_g1_core.py -q
```

结果：

```text
28 passed, 5 warnings
```

Warnings 来自 Megatron / Transformer Engine 的 deprecation warning，与本轮 G1 逻辑无关。

## Current G1 Flow After This Change

```text
SGLang rollout
  -> custom_generate: generate_fixed_length_for_g1
  -> fixed response length check: 376
  -> RolloutManager train_data adds g1_full_sequences / g1_qa_masks
  -> actor switches to ref snapshot
  -> Megatron forward_only captures final hidden states
  -> trainer-side hidden -> block embeddings -> reward/RLOO
  -> rollout_data["g1_token_advantages"]
  -> loss.py advantage_estimator == "g1"
  -> PPO-style training smoke
```

## What This Does Not Yet Prove

This does not yet prove full OpenRLHF G1 exactness.

Still missing:

- full `prompt -> Megatron ref hidden -> groom` runtime golden parity
- independent Megatron embedding/critic role
- EBFTPolicyLoss + CE exact loss branch
- variable-length / truncation support
- arbitrary DP / `balance_data=true` support beyond group-aligned split

## Known Risks

### `build_conda.sh` CUDNN Version

Current diff shows:

```text
nvidia-cudnn-cu12==9.16.0.29
```

This is not part of the G1 logic. It should be confirmed separately.

这个不是问题，就用这个版本

### First Version Fixed Length

The slow embedding path intentionally rejects responses whose length is not 376. This is correct for the first exact-path smoke, but must be revisited later.

### Prompt Packing Approximation

The slow slime path reconstructs OpenRLHF-style prompt tokens from `prompt + label`. It does not yet fully reproduce OpenRLHF multi-document packing when several QA pairs are packed into a single 384-token chunk.

### Loss Is Not Exact Yet

Until Step 5 is implemented, use names like:

```text
slime_g1_reward_smoke
slime_g1_advantage
```

Do not call this full `slime_g1_exact` in experimental conclusions.

## Next Action

Use the maintained Megatron/fast smoke described in:

- `refactor_debugging/g1_plan/step4_smoke_config.md`
- `refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh`

The key command additions are:

```text
--advantage-estimator g1
--custom-generate-function-path slime.rollout.g1_embedding.generate_fixed_length_for_g1
--use-whitening
--rollout-max-response-len 376
--g1-tokenizer-path <tokenizer path>
--g1-embedding-source megatron_ref
--g1-reward-location trainer
```

