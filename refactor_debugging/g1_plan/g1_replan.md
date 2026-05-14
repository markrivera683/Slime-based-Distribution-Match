# Slime G1 Replan

## Goal

在当前 slime 项目中复刻 OpenRLHF G1 核心算法，同时保持迁移顺序清晰：

1. 先跑通 slime baseline。
2. 再验证 G1 数学等价。
3. 再补 rollout embedding 链路。
4. 再决定是否补 OpenRLHF 风格 actor loss。
5. 最后再做 async / OPD 等加速实验。

当前阶段不要直接追求加速。没有稳定 baseline 和 G1 小闭环之前，不引入 async、OPD、dynamic batch 等变量。

## Current State

当前分支已经完成了 G1 的中段管线草稿，并已通过固定 embedding / OpenRLHF helper 级 parity：

- `slime/utils/g1_core.py`：实现 strided block 几何、whitening、alignment/diversity reward、RLOO baseline、block reward 到 token advantage 展开。
- `slime/rollout/rm_hub/g1_core.py`：实现 group custom RM，从 `Sample.metadata` 读取 `g1_gen_embedding` / `g1_gt_embedding`，写回 `g1_token_advantages`。
- `slime/ray/rollout.py`：把 `g1_token_advantages` 透传到 train batch，并支持 DP split。
- `slime/backends/megatron_utils/actor.py`：把 `g1_token_advantages` 转成 GPU float tensor。
- `slime/backends/megatron_utils/loss.py`：新增 `advantage_estimator == "g1"`，直接消费预计算 token advantages。
- `slime/utils/arguments.py`：新增 `--advantage-estimator g1`、`--alignment-rew-coef`、`--diversity-rew-coef`、`--use-whitening`。
- `tests/test_g1_core.py`：覆盖核心公式、RM metadata 写入、train_data 透传、loss 消费预计算 advantages；当前目标测试已通过。
- `refactor_debugging/g1_plan/run_openrlhf_runtime_dump_parity.sh`：用 OpenRLHF helper 生成固定 embedding golden，再用 slime 校验 reward/RLOO/token advantage parity。

关键缺口：

- 还没有实现从真实 rollout / critic hidden states 生成 `g1_gen_embedding` 和 `g1_gt_embedding`。
- 当前 slime 训练侧仍使用 PPO-style policy loss，没有完全复刻 OpenRLHF 的 `EBFTPolicyLoss`。
- 当前 v3 plan 的 Stage 2 写的是“接受整段近似”，如果目标是复刻 G1 核心算法，需要先做 exact parity，再决定是否保留近似版本作为加速实验。
- OpenRLHF G1 默认启用 whitening，slime 现在需要显式传 `--use-whitening`。

## Step 0: Freeze Scope And Baseline Rules

目标：明确当前迁移只做 G1，不把 G2/G3、CF teacher、OPD、async 一起混进来。

执行顺序：

1. 固定目标算法为 OpenRLHF G1：
  - `distribution_reward_type=pointwise`
  - `cf_target_mode=single`
  - no teacher
  - `advantage_estimator=rloo`
  - `use_whitening=true`
  - frozen critic embedding source
2. 明确当前 slime 目标分两层：
  - 第一层：G1 reward / RLOO / token advantage 数学等价。
  - 第二层：是否复刻 OpenRLHF actor loss。
3. 明确暂停项：
  - 暂不做 async。
  - 暂不做 OPD。
  - 暂不做 G2/G3。
  - 暂不做 CF reward。
  - 暂不做性能 sweep。

产出：

- 一份确认后的 G1 scope。
- 一组必须固定的实验参数。

通过标准：

- 后续每个实现任务都能归入“G1 parity”或“slime baseline”之一。
- 不再把加速和算法复刻混在同一个调试循环里。

## Step 1: Slime Baseline First

目标：先证明 slime 原生训练闭环能跑通，不引入 G1 custom reward。

状态：用户已确认 baseline 跑过且无问题；后续 G1 不再被 Step 1 阻塞。

执行顺序：

1. 复用当前数据准备链路。
2. 启动同步 slime baseline：
  - 使用 `train.py`。
  - 不使用 async。
  - 不使用 OPD。
  - 不使用 G1 custom RM。
  - 不使用 `--advantage-estimator g1`。
3. 小步数跑通：
  - 目标不是追求最终指标。
  - 目标是训练、保存、转换、评测链路不崩。
4. 做 checkpoint round-trip：
  - slime checkpoint 转回 HF。
  - 跑 HumanEval pass@1 sanity。

产出：

- slime baseline 启动命令。
- 一次可复跑的小步数日志。
- checkpoint 转换结果。
- HumanEval pass@1 sanity 结果。

通过标准：

- baseline 能稳定复跑。
- checkpoint 可以转回 HF。
- pass@1 不明显崩坏。

阻塞规则：

- Step 1 不通过，不进入 G1 端到端实现。

## Step 2: G1 Math Parity

目标：不依赖真实 rollout，先验证 slime 的 G1 数学与 OpenRLHF G1 对齐。

状态：已完成固定 embedding parity 与 OpenRLHF helper-level dump parity；尚未覆盖完整 `prompt -> rollout -> critic hidden -> groom` runtime parity。

执行顺序：

1. 从 OpenRLHF diff-dataset G1 中固定要对齐的数学：
  - strided block 数量。
  - generated block / ground-truth block 排布。
  - whitening。
  - alignment reward。
  - diversity reward。
  - `alignment_rew_coef * alignment - diversity_rew_coef * diversity`。
  - RLOO baseline。
  - block reward 展开到 token advantage。
2. 构造最小 fixture：
  - 固定 `n_samples_per_prompt=4`。
  - 固定 `num_blocks`。
  - 固定 hidden dim。
  - 使用写死的 `g1_gen_embedding` / `g1_gt_embedding`，不跑真实模型、不跑 rollout、不从 critic 动态取 hidden states。
  - fixture 必须可重复、可追溯，所有输入 tensor 形状和数值都在测试中显式声明。
3. 在 slime 中验证：
  - `compute_pointwise_rewards` 数值。
  - `compute_rloo_baseline` 数值。
  - `compute_rloo_shaped_rewards` 数值。
  - `expand_block_rewards_to_token_advantages` 顺序。
  - `compute_group_g1_rewards` 写入 metadata 的字段。
4. 覆盖两条路径：
  - `use_whitening=False`：先排除 whitening 干扰，验证 pointwise reward / RLOO / token advantage 基本公式。
  - `use_whitening=True`：对齐 diff-dataset G1 主路径，whitening 使用 `normalize=False`，样本轴固定在 dim=2。
5. 明确默认参数：
  - 对齐 OpenRLHF G1 时必须显式启用 `--use-whitening`。
  - `alignment_rew_coef=1.0`。
  - `diversity_rew_coef=1.0`。
  - `ce_loss_coef=0.03` 仅属于 actor loss parity，不参与 reward fixture 数学。

产出：

- 固定 embedding parity fixture。
- 对齐 OpenRLHF diff-dataset G1 公式的单元测试。
- 明确的 shape、sample 顺序、block 顺序、token 展开顺序文档。

通过标准：

- 对同一组固定 embedding，slime 与 OpenRLHF reference 的以下输出必须逐项 `assert_close`：
  - whitened `gen_embedding` / `gt_embedding`。
  - alignment rewards。
  - diversity rewards。
  - combined pointwise rewards。
  - RLOO baseline。
  - shaped rewards。
  - token advantages。
- non-whitening 与 whitening 两条路径都必须通过。
- token advantage 展开顺序必须按 OpenRLHF 源码行为验证：block-level `shaped_reward.repeat(1, generate_max_len)`，即 `[block0, block1, ..., blockK, block0, block1, ..., blockK]` 这种按 generation step 重复整段 block reward 的顺序。

阻塞规则：

- Step 2 不通过，不补真实 rollout embedding。

## Step 3: Add Rollout Embedding Source

目标：补齐当前最大缺口，让 slime rollout 能生成 `g1_gen_embedding` 和 `g1_gt_embedding`。

执行顺序：

1. 先用慢 HF/OpenRLHF embedding path 接入 slime 小闭环：
  - 复刻 OpenRLHF G1 的 frozen critic hidden states。
  - 第一版 `hidden_state_method=last_only`。
  - 固定 `embed_method=last_token`。
  - 第一版 `qa_masking=False`、`document_masking=False`。
  - 第一版 `feature_map_type=identity`。
2. 确定 token 几何：
  - `prompt_length`
  - `context_length`
  - `generate_length`
  - `stride`
  - `num_blocks = (prompt_length - generate_length - context_length) / stride + 1`
  - 第一版固定 `prompt_length=384`、`context_length=8`、`generate_length=8`、`stride=8`、`response_length=376`
3. 在 rollout 或 custom generate 路径中构造 block：
  - generated blocks。
  - ground-truth blocks。
  - 每个 sample 对应 `[num_blocks, hidden_dim]` 的 embedding。
4. 写入 `Sample.metadata`：
  - `g1_gen_embedding`
  - `g1_gt_embedding`
5. 让现有 `slime.rollout.rm_hub.g1_core.custom_rm` 消费这些字段。
6. 增加契约校验：
  - group size 必须等于 `n_samples_per_prompt`。
  - 每个 sample 的 `num_blocks` 必须一致。
  - `response_length % num_blocks == 0`。
  - 缺 metadata 时直接报错，不静默 fallback。

产出：

- embedding 生成设计说明。
- metadata contract。
- 最小 rollout fixture 或 mock integration test。

通过标准：

- rollout 结束后，每条样本都带有合法的 `g1_gen_embedding` / `g1_gt_embedding`。
- group RM 能成功写出 `g1_token_advantages`。
- 完整 runtime parity 仍需单独用 critic/groom dump 验证；helper-level fixed embedding parity 不代表这一步已完成。

## Step 4: Connect G1 End To End

目标：跑通 slime 的 G1 reward/advantage 训练小闭环。

执行顺序：

1. 启动参数最小集合：
  - `--advantage-estimator g1`
  - `--group-rm`
  - `--custom-rm-path slime.rollout.rm_hub.g1_core.custom_rm`
  - `--use-whitening`
  - `--alignment-rew-coef 1.0`
  - `--diversity-rew-coef 1.0`
2. 禁用额外变量：
  - 不启用 async。
  - 不启用 OPD。
  - 不启用 dynamic filtering。
  - 不启用 dynamic batch sweep。
3. 跑极小步数训练。
4. 检查关键字段：
  - `g1_rewards`
  - `g1_gt_rewards`
  - `g1_diversity_rewards`
  - `g1_rloo_baseline`
  - `g1_token_advantages`
5. 检查训练侧：
  - `g1_token_advantages` 长度等于 `response_length`。
  - DP split 后字段不丢。
  - GPU tensor dtype 为 float32。
  - loss finite。

产出：

- 一次 G1 小闭环运行日志。
- reward / advantage debug dump。
- 训练 loss sanity。

通过标准：

- 训练一步或少量步数不报错。
- G1 token advantage 被训练侧消费。
- loss 没有 NaN / Inf。

## Step 5: Decide Actor Loss Parity

目标：明确 slime 是否需要复刻 OpenRLHF `EBFTPolicyLoss`。

背景：

- OpenRLHF G1 的 actor loss 是 REINFORCE-like RL loss + CE loss。
- 当前 slime 是 PPO-style policy loss，G1 只替换了 advantage。

执行顺序：

1. 先保留当前 slime PPO-style loss 跑一个最小 G1 实验，作为 embedding/advantage 小闭环验证。
2. 对比 OpenRLHF G1 的同口径小实验：
  - reward 分布。
  - advantage 分布。
  - loss 曲线。
  - pass@1 / pass@16 sanity。
3. 如果 slime PPO-style loss 方向一致：
  - 将当前路线命名为 `slime_g1_reward` 或 `slime_g1_equiv_reward`。
  - 暂不改 loss。
4. 如果差异明显：
  - 新增 G1/EBFT policy loss 分支。
  - 支持 RL loss on generation tokens。
  - 支持 CE loss on prompt / QA mask 区域。
  - 增加 `ce_loss_coef`。

产出：

- 是否复刻 actor loss 的决策记录。
- 如果需要，输出 loss 接入设计。

通过标准：

- 能解释当前 slime G1 与 OpenRLHF G1 的差异来自 reward、embedding、还是 loss。

## Step 6: Strengthen Tests

目标：把 G1 迁移中的关键契约固定住，避免后续加速时破坏算法语义。

执行顺序：

1. 保留现有 `tests/test_g1_core.py`。
2. 补充 OpenRLHF fixture parity test。
3. 补充 group RM contract test：
  - 缺 embedding 报错。
  - group size 不等于 `n_samples_per_prompt` 报错。
  - `response_length` 与 `num_blocks` 不整除报错。
4. 补充 train data contract test：
  - `_convert_samples_to_train_data` 不丢 `g1_token_advantages`。
  - `_split_train_data_by_dp` 不丢 `g1_token_advantages`。
5. 补充 CP 场景 test：
  - `slice_log_prob_with_cp` 后 advantage 与 local response tokens 对齐。

产出：

- 一组可在本地快速运行的 G1 tests。

通过标准：

- 未来改 rollout、DP、CP、loss 时，G1 contract 能被测试及时捕获。

## Step 7: Promote To Real Experiment

目标：从小闭环进入同口径 G1 实验。

执行顺序：

1. 固定 baseline 与 G1 的 token budget。
2. 固定评测：
  - HumanEval pass@1 / pass@16。
  - MBPP pass@1 / pass@16。
3. 固定记录口径：
  - rollout time。
  - reward / embedding time。
  - actor train time。
  - save / eval time。
  - total wall-clock。
4. 跑 `slime_baseline`。
5. 跑 `slime_g1`。
6. 与 OpenRLHF G1 做同口径对比。

产出：

- baseline vs slime G1 对比表。
- 指标和 wall-clock 曲线。

通过标准：

- slime G1 至少方向上不弱于 slime baseline。
- 能说明速度收益或开销瓶颈在哪里。

## Step 8: Only Then Start Acceleration

目标：在 G1 正确性明确后，再做 slime 加速能力验证。

执行顺序：

1. 以 Step 7 的 slime G1 作为固定基线。
2. 单因子打开 async。
3. 单因子调整 `update_weights_interval`。
4. 单因子打开 dynamic batch。
5. 单因子打开 balance-data。
6. 单因子调 SGLang concurrency / memory 参数。
7. 每个因素只在收益明确时保留。

产出：

- speed sweep 表。
- 最终推荐启动脚本。

通过标准：

- 每个保留项对 wall-clock 有明确收益。
- 加速后 pass@ 指标没有明显退化。

## Step 9: OPD And Future Work

目标：G1 + acceleration 稳定后，再考虑 OPD 和后续创新。

执行顺序：

1. 先用最便宜的 SGLang teacher 形态验证 OPD。
2. 只对比：
  - slime G1 accelerated
  - slime G1 accelerated + OPD
3. 如果 OPD 有收益，再考虑 Megatron teacher。
4. G2/G3 是否迁移，等 G1 报告完成后再决策。

产出：

- OPD pilot 结果。
- 是否迁移 G2/G3 的决策建议。

通过标准：

- OPD 对 pass@ 或收敛速度有可观测收益。
- 没有收益时不继续扩大复杂度。

## Suggested Agent Split

后续实际写代码时建议并行拆 agent：

- `openrlhf-g1-parity`：提取 OpenRLHF exact G1 数学、shape、fixture。
- `slime-rollout-embedding`：设计并实现 slime embedding 生成和 metadata 写入。
- `slime-loss-parity`：判断并实现可选 EBFTPolicyLoss。
- `test-contracts`：补 parity、RM、DP、CP 测试。
- `runner-baseline`：跑 Stage 1 baseline 与 checkpoint round-trip。
- `runner-g1`：跑 G1 小闭环和同口径实验。

## Immediate Next Step

下一步执行 Step 3：Add Rollout Embedding Source。

不要先做 async、OPD 或 EBFT loss。先用慢 HF/OpenRLHF embedding path 把 `g1_gen_embedding` / `g1_gt_embedding` 写入 slime `Sample.metadata`，跑通 G1 小闭环，再决定 Megatron fast path 与 EBFT loss。