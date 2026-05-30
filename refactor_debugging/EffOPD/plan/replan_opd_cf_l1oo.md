# EffOPD 接入 OPD-CF-L1OO 重规划

## 背景 / 当前发现

EffOPD 目前已经具备 controller / hook 骨架，相关逻辑集中在
`slime/backends/megatron_utils/effopd/`，并已在 Megatron actor 路径中预留接入点。
后续工作不应从零重写 EffOPD，而应把现有骨架连接到 OPD-CF-L1OO 的主路径：
`cf_target_mode=opd_onpolicy`。

已知阻塞点：当前 EffOPD 的 validation / argument 约束看起来仍要求
`cf_target_mode=teacher`，而主 OPD-CF-L1OO launcher / docs 使用
`cf_target_mode=opd_onpolicy`。后续实现需要放宽或重新核验这组约束，避免 EffOPD 被参数校验挡在
OPD-CF-L1OO 正式路径之外。

相关路径：

- `slime/utils/arguments.py`
- `slime/backends/megatron_utils/effopd/`
- `slime/backends/megatron_utils/actor.py`
- `train_async.py`
- `tests/test_effopd_core.py`
- `tests/test_g1_ebft_arguments.py`
- `tests/test_g2_opd_blockwise_launcher_contract.py`

## 目标语义

1. EffOPD 是 OPD-CF-L1OO 的 checkpoint-level acceleration：它只加速检查点级别的参数推进，
   不替代 OPD loss，不引入新的可训练参数。
2. `D_v` gate 使用组合评分：以 CF-L1OO reward 作为主信号，减去 reverse-KL proxy；
   默认不采用 shadow-first 策略。
3. 被接受的 extrapolation 只改变模型参数；optimizer / scheduler 状态保持不变。
   下一次 delta 必须基于实际被接受后的权重计算。
4. candidate extrapolation 从 base checkpoint 出发，使用 `2^k` scales；候选之间不能链式叠加。

## 后续实现计划

1. 梳理参数校验
   - 检查 `slime/utils/arguments.py` 中 EffOPD 与 `cf_target_mode` 的约束。
   - 将 EffOPD 允许范围扩展到 `cf_target_mode=opd_onpolicy`，或确认已有等价路径并补足测试。
   - 保持 teacher 模式兼容，不改变非 EffOPD 训练默认行为。

2. 连接 actor / async 训练路径
   - 在 `slime/backends/megatron_utils/actor.py` 中确认 EffOPD hook 的调用点位于 OPD-CF-L1OO
     完成一次可接受 checkpoint / step 边界之后。
   - 在 `train_async.py` 路径中核验 EffOPD controller 的生命周期，确保 on-policy rollout 与
     参数接受逻辑不会互相越界。

3. 明确 candidate 生成与接受逻辑
   - 每轮以 base checkpoint 和最新 accepted checkpoint 计算 delta。
   - 从 base checkpoint 生成 `scale=2^k` 的候选参数。
   - gate 接受后只写入模型参数，不重置 optimizer / scheduler。
   - 记录 accepted scale、组合评分、CF-L1OO reward、reverse-KL proxy，方便排查训练流。

4. 文档与 launcher 对齐
   - 更新 EffOPD 文档，明确它是 OPD-CF-L1OO 的加速器而不是新 loss。
   - 对齐主 OPD-CF-L1OO launcher / docs 中的 `cf_target_mode=opd_onpolicy`。

## 验证计划

建议优先跑窄测试，再做 launcher dry-run：

```bash
pytest tests/test_effopd_core.py
pytest tests/test_g1_ebft_arguments.py
pytest tests/test_g2_opd_blockwise_launcher_contract.py
```

建议补充或更新的测试点：

- EffOPD + `cf_target_mode=opd_onpolicy` 参数组合通过校验。
- EffOPD + teacher 模式仍保持原有兼容性。
- `D_v` gate 默认使用 “CF-L1OO reward - reverse-KL proxy” 组合评分。
- candidate scales 从同一个 base checkpoint 生成，不链式生成。
- accepted extrapolation 只改模型参数，不改 optimizer / scheduler state。

建议 dry-run：

```bash
bash examples/.../opd_cf_l1oo_*.sh --dry-run
bash exper_scripts/.../opd_cf_l1oo_*.sh --dry-run
```

实际脚本名以后续实现 agent 在仓库中确认的主 OPD-CF-L1OO launcher 为准；dry-run 重点检查
`--cf-target-mode opd_onpolicy` 与 EffOPD 开关能同时存在，并且没有被 argument validation 拒绝。

## 风险 / 假设

- 假设现有 EffOPD controller / hook 骨架已经覆盖 checkpoint 选择、候选评估和接受入口；后续主要是接线、
  语义校准和测试补齐。
- `D_v` gate 的 reverse-KL proxy 需要与现有 OPD-CF-L1OO 指标口径一致，否则会出现 gate 分数和训练日志解释不一致。
- optimizer / scheduler state 保持不变可能与某些权重同步或 ZeRO / Megatron 状态缓存路径冲突，需要在 actor
  与 async 训练路径分别验证。
- 若当前 EffOPD validation 事实上依赖 teacher-target 特征或 teacher embedding，改成 `opd_onpolicy` 时需要明确替代来源，
  不能用 shadow-first 作为默认退路。
