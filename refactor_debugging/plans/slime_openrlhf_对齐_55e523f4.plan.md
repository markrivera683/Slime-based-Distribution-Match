---
name: slime openrlhf 对齐
overview: 已完成对当前 slime 框架、G1/G2/G2+OPD 扩展，以及旧 OpenRLHF EBFT/G2/G3 实现的只读梳理。后续任何修改会先按这份对齐图定位入口、数据契约和 reward/loss 路径。
todos:
  - id: index-critical-files
    content: 维护当前 slime 与旧 OpenRLHF 的关键文件索引，后续问题先从索引定位。
    status: pending
  - id: trace-workflows
    content: 按 G1、G2 no-teacher、G2 teacher、G2+OPD、旧 G3 五条 workflow 分别追踪入口脚本到 reward/loss。
    status: pending
  - id: verify-contracts
    content: 遇到实现改动时优先验证 sample metadata、train_data、embedding shape、prompt group 顺序和 teacher 样本契约。
    status: pending
  - id: compare-openrlhf
    content: 涉及算法语义时以旧 OpenRLHF 的 embedding_utils 与 ebft_experience_maker 为事实源做对照。
    status: pending
isProject: false
---

# Slime 与 OpenRLHF 对齐阅读计划

## 已确认的主线

- 当前 slime 主框架是 `[train.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/train.py)` / `[train_async.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/train_async.py)` 驱动 Ray 编排，rollout 走 `[slime/rollout/sglang_rollout.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/rollout/sglang_rollout.py)`，训练侧走 Megatron `[slime/backends/megatron_utils/actor.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/backends/megatron_utils/actor.py)`。
- G1/EBFT 在 slime 中的核心是固定长度 rollout + trainer-side Megatron ref embedding + `[slime/utils/g1_core.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/utils/g1_core.py)` + `[slime/utils/g1_ebft_loss.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/utils/g1_ebft_loss.py)`。
- G2 no-teacher 的核心是 `distribution_reward_type=cf_l1oo` + `cf_target_mode=single`，不拉 remote teacher，不允许 OPD，reward 在 actor ref 路径算。
- G2 cf_l1oo teacher / G2+OPD 的核心是 `cf_target_mode=teacher`，rollout 后用 `[slime/rollout/g2_teacher.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/rollout/g2_teacher.py)` 拉 teacher completions，critic 侧算 teacher embedding 和 `g1_token_advantages`。
- OPD 是正交叠加：`[slime/rollout/on_policy_distillation.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/rollout/on_policy_distillation.py)` 从 SGLang teacher `/generate` 拿 token logprob，`[slime/backends/megatron_utils/loss.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/backends/megatron_utils/loss.py)` 用 reverse KL 改 advantages。

## 与旧 OpenRLHF 的对应关系

- 旧入口是 `[openrlhf/cli/train_ebft_ray.py](/mnt/data/ebft-distribution-new/code/openrlhf/cli/train_ebft_ray.py)`，主循环在 `[openrlhf/trainer/ebft_trainer.py](/mnt/data/ebft-distribution-new/code/openrlhf/trainer/ebft_trainer.py)`。
- 旧 reward/experience 核心在 `[openrlhf/trainer/ppo_utils/ebft_experience_maker.py](/mnt/data/ebft-distribution-new/code/openrlhf/trainer/ppo_utils/ebft_experience_maker.py)`，其中 `make_experience` 负责 critic hidden、embedding、teacher embedding、`cf_l1oo` reward 和 advantage 前置数据。
- 旧数学核心在 `[openrlhf/utils/embedding_utils.py](/mnt/data/ebft-distribution-new/code/openrlhf/utils/embedding_utils.py)`：`whiten_embeddings_batched`、`_build_cf_target_embedding`、`get_cf_l1oo_rewards` 是 slime `[slime/utils/g2_core.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/utils/g2_core.py)` 的对齐来源。
- 旧 G3 额外包含 EMA、feature adapter、critic classifier/direct discrepancy，主要在 `[openrlhf/trainer/ray/ebft_critic.py](/mnt/data/ebft-distribution-new/code/openrlhf/trainer/ray/ebft_critic.py)` 和 `[openrlhf/models/critic.py](/mnt/data/ebft-distribution-new/code/openrlhf/models/critic.py)`；当前 slime 尚未完整实现 G3/vicinal。

## 后续处理原则

- 改任何 G1/G2 逻辑前，先检查 `[slime/utils/arguments.py](/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/utils/arguments.py)` 的参数契约，避免把 no-teacher、teacher、OPD、G3 语义混在一起。
- 查 G2 数值问题时，优先按 `sample metadata -> train_data -> teacher/student embedding -> cf_l1oo rewards -> token advantages -> EBFT loss` 这条链路排查。
- 查脚本/部署问题时，区分两种 2-node：G2 teacher/OPD 是 teacher pod + student pod；G2 no-teacher 是 16 GPU Ray 集群，无 teacher 服务。
- 如果后续要继续迁移 OpenRLHF G3，重点补齐 slime 的 `vicinal target`、feature adapter、EMA critic、classifier/direct discrepancy，而不是复用现有 OPD 路径。

