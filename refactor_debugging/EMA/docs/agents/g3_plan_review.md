# G3 Plan Review 评审记录

> 状态：规划/设计评审文档。本文档记录计划修正，不表示代码已实现。

## P0：adapter 没有训练信号

原计划的最大风险是把 adapter 放在 reward embedding path，但没有建立 critic-side differentiable training closure。

问题原因：

- reward/advantage 在训练 actor 前会变成 detached/scalar。
- actor 消费的是 synchronized rewards/token advantages。
- scalar reward path 无法把梯度传回 adapter。

最终修正：

- adapter 训练必须发生在 critic/Megatron 侧。
- critic 侧构造 feature loss。
- live adapter embeddings 参与 feature loss。
- EMA embeddings detached 后作为 reward target geometry。
- actor 侧只消费同步后的 rewards/token advantages。

## P0：当前 `opd_onpolicy` reward 不是 critic-side

另一个核心问题是当前 `opd_onpolicy` reward/feature computation 不应继续作为 actor-side 或 detached reward path 的附属逻辑。

最终修正：

- G3 implementation 应移动/拥有 OPD-onpolicy G3 reward/feature computation 到 critic/Megatron 侧。
- critic-side computation 是同步 rewards/token advantages 的 source。

## P1：validation/optimizer 冲突

当前 G2 validation 会冻结 critic/head。G3 v1 又需要训练 adapter，因此容易出现两类冲突：

- validation 误以为任何 critic-side optimizer 都违反冻结规则。
- optimizer 参数集合误包含 backbone 或 critic value head。

最终修正：

- G3 branch 允许 adapter optimizer。
- adapter optimizer 只能包含 adapter/G3 trainable params。
- backbone 不训练。
- critic value head 不训练。
- v1 不加入 `feature_adapter_unfreeze_layers`。

## P1：EMA update point 不清楚

EMA 更新点必须明确，否则可能出现未 step 就更新、失败 step 后更新、或与梯度图耦合的问题。

最终修正：

- successful optimizer step 之后更新 EMA。
- EMA 参数不参与梯度。
- EMA target embeddings 必须 detached。
- checkpoint 保存 live adapter、adapter optimizer、EMA adapter。
- checkpoint 缺失 EMA 时从 live adapter 初始化。

## P1：adapter insertion point 不清楚

adapter 插入位置如果太晚，例如 reward scalar 后，就无法保持 feature geometry 训练意义。

最终修正：

- hidden-state L2 normalize 后进入 adapter。
- adapter 位于 G1/G2 block pooling/construction 之前。
- adapter 结构为 `LayerNorm -> Linear down -> GELU -> Dropout -> Linear up -> residual`。
- up projection zero-init，确保初始 identity parity。

## 最终 v1 范围

保留：

- OPD-fused G3。
- `cf_l1oo + cf_target_mode=opd_onpolicy + use_opd`。
- learnable residual feature adapter。
- EMA feature geometry。
- critic-side closure。

排除：

- source EBFT teacher-completion target G3。
- teacher embeddings for `opd_onpolicy`。
- `feature_adapter_unfreeze_layers`。
- backbone training。
- critic value head training。
