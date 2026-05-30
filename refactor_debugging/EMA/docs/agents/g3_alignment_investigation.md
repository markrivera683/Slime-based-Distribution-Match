# G3 Alignment Investigation 调研记录

> 状态：规划/设计调研文档。本文档不是实现结果。

## 调研结论

source EBFT 的 G3 路线与当前用户选择不同。source EBFT G3 使用 teacher completions target，并在 critic 侧有 adapter+EMA 训练逻辑，同时 reward 类型仍围绕 `cf_l1oo`。

当前 repo 的目标不是复刻 teacher-completion target G3，而是在已有 G1/G2/OPD-CF-L1OO 基础上实现 OPD-fused G3。

## Source EBFT G3 事实

参考方向来自 source repo：

- `/mnt/data/ebft-distribution-new/code/scripts/diff_dataset/`
- `/mnt/data/ebft-distribution-new/code/openrlhf/trainer/ppo_utils/ebft_experience_maker.py`

已知事实：

- source EBFT G3 使用 teacher completions target。
- source EBFT G3 使用 adapter+EMA critic。
- source EBFT G3 使用 `cf_l1oo`。
- source G3 script 中有 `CF_TEACHER_LAMBDA=0.6`。
- source G3 script 中有 `CF_TEACHER_N_SAMPLES=4`。
- source G3 adapter rank 为 64。
- source G3 adapter dropout 为 0。
- source G3 使用 `EMA_BETA=0.99`。
- source G3 使用 `CRITIC_LR=0`。
- source G3 使用 `CRITIC_LR_HEAD=5e-5`。

这些事实用于理解 lineage，但不代表当前 v1 要实现 teacher-completion target。

## 当前 Repo 事实

当前 repo 已具备：

- G1/G2 基础。
- OPD-CF-L1OO 基础。
- `cf_l1oo` reward 类型。
- `cf_target_mode=opd_onpolicy` 路径。
- `--use-opd` 相关路径。

当前 repo 尚缺：

- true learnable feature adapter。
- EMA adapter / EMA feature geometry。
- critic-side OPD-onpolicy G3 feature training closure。

另一个重要事实：当前 G2 validation 会冻结 critic/head。G3 v1 需要允许 adapter optimizer，但仍不训练 backbone 或 critic value head，因此 validation/optimizer 逻辑必须小心区分 adapter params 和 critic/head params。

## 对当前计划的影响

OPD-fused G3 的 v1 应采用：

- `--distribution-reward-type cf_l1oo`
- `--cf-target-mode opd_onpolicy`
- `--use-opd`
- critic/Megatron-side G3 reward/feature computation
- adapter + EMA geometry

并明确排除：

- source EBFT teacher-completion target。
- teacher embeddings for `opd_onpolicy`。
- backbone unfreeze。
- critic value head training。
