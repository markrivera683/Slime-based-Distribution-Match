# EMA / G3 OPD-Fused 规划文档

> 状态：规划/设计文档。这里描述的是后续实现方向，不表示代码已经完成。

本目录记录 `refactor_debugging/EMA` 下关于 G3、OPD-CF-L1OO、feature adapter、EMA feature geometry 的调研和实施计划。当前用户选择的是 **OPD-fused G3** 路线，而不是 source EBFT 中基于 teacher-completion target 的 G3 路线。

## 文档索引

- [g3_opd_fused_ema_alignment.md](g3_opd_fused_ema_alignment.md)：总体设计对齐说明。
- [plan/g3_opd_fused_ema_alignment_plan.md](plan/g3_opd_fused_ema_alignment_plan.md)：分阶段实施计划和测试计划。
- [agents/g3_alignment_investigation.md](agents/g3_alignment_investigation.md)：调研事实记录。
- [agents/g3_plan_review.md](agents/g3_plan_review.md)：计划评审和关键修正。

## 最终方向

当前 repo 已经有 G1/G2/OPD-CF-L1OO 基础。G3 v1 计划在以下组合上叠加：

- `--distribution-reward-type cf_l1oo`
- `--cf-target-mode opd_onpolicy`
- `--use-opd`
- learnable residual feature adapter
- EMA feature geometry

核心修正是：adapter 必须有 critic-side training closure。仅把 adapter 放进 reward embedding path 不会产生有效梯度，因为 rewards/advantages 会变成 detached/scalar，actor 侧消费的只是同步后的 reward/token advantage。

因此，G3 的 OPD-onpolicy reward/feature computation 应由 critic/Megatron 侧移动/拥有；actor 只消费已经同步好的 rewards/token advantages。

## v1 CLI 范围

计划新增/使用这些 flag：

- `--g3-enable`
- `--feature-adapter-enable`
- `--feature-adapter-rank`
- `--feature-adapter-dropout`
- `--enable-ema`
- `--ema-beta`
- `--g3-adapter-lr`
- `--g3-feature-loss-coef`

v1 不加入 `feature_adapter_unfreeze_layers`，避免与“不训练 backbone 或 critic value head”的边界冲突。

## 启动脚本说明

后续实现应更新当前 OPD-CF 1-node/2-node launcher。deprecated dep script 只作为 legacy 参考，不作为主要实现入口。
