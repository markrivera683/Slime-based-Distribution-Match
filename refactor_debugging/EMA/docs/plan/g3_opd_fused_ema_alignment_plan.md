# G3 OPD-Fused EMA Alignment 实施计划

> 状态：规划/设计文档。本文档描述未来实现步骤，不表示代码已完成。

## 目标

在当前 repo 已有 G1/G2/OPD-CF-L1OO 基础上，实现 OPD-fused G3：

- 基础组合：`cf_l1oo + cf_target_mode=opd_onpolicy + use_opd`
- 新增 learnable residual feature adapter
- 新增 EMA feature geometry
- 将 G3 reward/feature computation 放到 critic/Megatron 侧拥有
- actor 只消费同步后的 rewards/token advantages

## 非目标

- 不实现 source EBFT teacher-completion target G3。
- v1 不加入 `feature_adapter_unfreeze_layers`。
- v1 不训练 backbone。
- v1 不训练 critic value head。
- deprecated dep script 不作为主要 launcher 更新对象。

## 阶段 1：参数与校验

新增/接入 CLI：

- `--g3-enable`
- `--feature-adapter-enable`
- `--feature-adapter-rank`
- `--feature-adapter-dropout`
- `--enable-ema`
- `--ema-beta`
- `--g3-adapter-lr`
- `--g3-feature-loss-coef`

校验规则：

- `--g3-enable` 要求 `--distribution-reward-type cf_l1oo`。
- `--g3-enable` 要求 `--cf-target-mode opd_onpolicy`。
- `--g3-enable` 要求 `--use-opd`。
- G3 adapter optimizer 不应与当前 G2 “freeze critic/head” 校验冲突。
- 明确报错：G3 OPD-fused 不使用 teacher embeddings。

## 阶段 2：adapter 模块

实现 residual feature adapter：

```text
L2-normalized hidden features
  -> LayerNorm
  -> Linear down
  -> GELU
  -> Dropout
  -> Linear up, zero-init
  -> residual
```

要求：

- adapter 位于 hidden-state normalize 之后。
- adapter 位于 G1/G2 block pooling/construction 之前。
- 初始 zero-init up projection 后应接近 identity parity。
- 参数集合只包含 adapter/G3 trainable params。

## 阶段 3：critic-side G3 closure

这是 v1 的关键实现点。

计划把 OPD-onpolicy G3 reward/feature computation 移到 critic/Megatron 侧，形成可微闭环：

- critic 侧拿 live adapter embeddings。
- EMA adapter 生成 detached target embeddings。
- 构造 feature loss。
- `--g3-feature-loss-coef` 控制 loss 权重。
- adapter optimizer 用 `--g3-adapter-lr`。
- optimizer step 成功后更新 EMA。
- actor 侧只读取同步后的 rewards/token advantages。

注意：不能只在 reward embedding path 放 adapter，否则 reward/advantage detached 后 adapter 没有梯度。

## 阶段 4：EMA 与 checkpoint

EMA 规则：

```text
ema = beta * ema + (1 - beta) * live
```

实现要求：

- EMA params 不参与梯度。
- successful optimizer step 之后更新 EMA。
- EMA embeddings 用于 reward target geometry 时必须 detached。
- checkpoint 保存 live adapter、adapter optimizer、EMA adapter。
- 从旧 checkpoint 恢复时，如果缺失 EMA adapter，则用 live adapter 初始化 EMA。

## 阶段 5：同步与 launcher

同步边界：

- critic/Megatron 侧是 G3 reward/feature computation source。
- actor 侧消费 synchronized rewards/token advantages。
- 不让 actor 侧 adapter reward path 成为训练信号来源。

Launcher：

- 后续更新当前 OPD-CF 1-node launcher。
- 后续更新当前 OPD-CF 2-node launcher。
- print-only/dry-run 输出应包含 G3/adapter/EMA flags。
- deprecated dep script 仅保留 legacy 说明。

## 测试计划

建议新增或更新以下测试：

- parser/validation：G3 要求 `cf_l1oo + opd_onpolicy + use_opd`。
- adapter identity parity：zero-init up projection 时输出接近原 feature。
- gradient only on adapter：backbone 和 critic value head 无梯度/无 optimizer step。
- feature loss differentiability：feature loss 能反传到 adapter。
- EMA equation：验证 `ema = beta * ema + (1 - beta) * live`。
- checkpoint missing EMA init：旧 checkpoint 缺失 EMA 时从 live 初始化。
- no teacher embeddings for opd_onpolicy：OPD-fused G3 不请求 teacher embeddings。
- critic-side sync source：reward/token advantage 来源是 critic-side computation。
- launcher print-only contracts：1-node/2-node OPD-CF launcher 输出包含预期 flags。

## 完成标准

- G3 OPD-fused 只在明确 flag 组合下启用。
- adapter 有 critic-side differentiable loss。
- actor 不承担 adapter 训练闭环。
- EMA 更新点和 checkpoint 行为可测试。
- launcher 文档和 print-only contract 与 v1 CLI 一致。
