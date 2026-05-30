# G3 OPD-Fused EMA Alignment 设计说明

> 状态：规划/设计文档。本文档不是已实现代码说明。

## 决策

用户选择的 G3 路线是 **OPD-fused G3**，不是 source EBFT 中的 teacher-completion target G3。当前实现方向应复用 repo 已有的 G1/G2/OPD-CF-L1OO 基础，在 `cf_l1oo + cf_target_mode=opd_onpolicy + use_opd` 上增加 learnable residual feature adapter 与 EMA feature geometry。

必须保持的启动约束：

- `--distribution-reward-type cf_l1oo`
- `--cf-target-mode opd_onpolicy`
- `--use-opd`

## 关键修正：adapter 必须有训练闭环

P0 级别的问题是：如果只是把 adapter 放进 reward embedding path，adapter 不会得到有效训练信号。原因是 reward/advantage 在进入 actor 更新前会变成 detached/scalar，actor 侧消费的是同步后的 rewards/token advantages，而不是可反传的 feature graph。

因此，G3 v1 的实现边界应调整为：

- OPD-onpolicy G3 reward/feature computation 由 critic/Megatron 侧拥有。
- critic 侧构造可微 feature loss，用于训练 G3 adapter。
- actor 侧只消费同步后的 rewards/token advantages。
- G3 分支允许 adapter optimizer，但不训练 backbone，也不训练 critic value head。

## Feature Adapter 设计

adapter 输入是 hidden features，先做 L2 normalize，再进入轻量 residual adapter。推荐结构：

1. hidden-state L2 normalize。
2. `LayerNorm`。
3. `Linear` down projection，rank 由 `--feature-adapter-rank` 控制。
4. `GELU`。
5. `Dropout`，概率由 `--feature-adapter-dropout` 控制。
6. `Linear` up projection。
7. residual add。
8. 输出再进入 G1/G2 的 block pooling/construction。

up projection 应 zero-init，让 adapter 初始接近 identity/residual parity，降低启用初期对现有 reward geometry 的扰动。

插入点：adapter 应位于 G1/G2 block pooling/construction 之前，而不是 reward scalar 之后。

## EMA Feature Geometry

EMA 维护 adapter/G3 trainable params 的 shadow copy：

```text
ema = beta * ema + (1 - beta) * live
```

计划规则：

- EMA 在 successful optimizer step 之后更新。
- EMA embeddings detached 后用于 reward target geometry。
- checkpoint 应包含 live adapter、adapter optimizer、EMA adapter。
- 如果 checkpoint 缺失 EMA adapter，应从当前 live adapter 初始化 EMA，而不是直接失败。

## CLI 计划

v1 使用：

- `--g3-enable`
- `--feature-adapter-enable`
- `--feature-adapter-rank`
- `--feature-adapter-dropout`
- `--enable-ema`
- `--ema-beta`
- `--g3-adapter-lr`
- `--g3-feature-loss-coef`

v1 不包含 `feature_adapter_unfreeze_layers`。当前目标是只训练 adapter/G3 trainable params，不扩大到 backbone unfreeze。

## 验证边界

开启 G3 OPD-fused v1 时，应验证：

- 必须是 `--distribution-reward-type cf_l1oo`。
- 必须是 `--cf-target-mode opd_onpolicy`。
- 必须启用 `--use-opd`。
- `opd_onpolicy` 路径不需要 teacher embeddings。
- G3 branch 允许 adapter optimizer。
- 不训练 backbone。
- 不训练 critic value head。

## Launcher 方向

后续实现应更新当前 OPD-CF 1-node/2-node launcher，让 print-only 或 dry-run 输出能明确展示 G3/adapter/EMA 相关 flag。deprecated dep script 是 legacy only，不应成为新路径的主要入口。
