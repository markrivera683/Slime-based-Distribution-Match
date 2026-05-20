# Slime 对齐 OpenRLHF EBFT G1 进度核查

## 结论

G1 对齐**不是完全完成**。更准确的判断是：Slime 的 G1 主链路 / Phase-1 parity path 已基本接入，但仍需实测和若干语义差异处理。

当前 Slime 侧已经具备固定长度 rollout、Megatron ref 侧 G1 embedding、pointwise reward、RLOO shaped rewards、whitening、token advantage 展开，以及 `--g1-use-ebft-loss` 对 OpenRLHF `EBFTPolicyLoss` 风格 actor loss 的接入路径。这些属于“代码能力已存在”。

但这不等同于“实验或数值 parity 已验证”。Slime 与 EBFT/OpenRLHF 在数据字段、chat template、KL/entropy、reward 主路径、eval/post-eval、mask/position 诊断路径默认行为，以及若干训练超参上仍存在差异或待确认项。因此文档中应将“已实现能力”和“已验证等价”严格区分。

## EBFT/OpenRLHF 中的 G1/G2/G3

### G1

EBFT/OpenRLHF 的 G1 以 `scripts/diff_dataset/run_G1_rebase.sh` 及其调用链为事实源。它是单脚本单机训练，使用 Qwen3.5-2B student，无在线 teacher。

G1 的关键配置包括：`distribution_reward_type=pointwise`、`CF_TARGET_MODE=single`、`CF_TEACHER_LAMBDA=0`，训练入口为 `train_ebft_ray`。默认关闭训练中 eval，训练后会运行 code benchmark baseline shell。数据由 `prepare_code_datasets.py` 生成 JSONL 和 `manifest.env`，关键字段为 `input_key=question`、`label_key=answer`、`output_key=answer`，并依赖 `PROMPT`、`CONTEXT`、`GENERATE`、`STRIDE` 这组关键几何参数。

### G2

EBFT/OpenRLHF 的标准 G2 以 `scripts/diff_dataset/run_G2_rebase_2node_once.sh` 及其 `JOB_SCRIPT` 调用链为事实源。它在 G1 基础上引入 `cf_l1oo` 和在线 vLLM teacher，critic head 默认冻结，即 `CRITIC_LR_HEAD=0`。

标准 G2 的主结构包含 Ray job、teacher API base/cache/prefetch、`zero_stage=3`，默认 `eval_steps=-1`，但仍会传入 eval CLI 参数，训练后使用 `run_code_generation_benchmarks.py`。需要注意，`no_teacher_distribution`、`vicinal`、`distillPure` 等是变体，不应与标准 G2 混淆。

标准 G2 不包含 feature adapter、EMA、classifier/direct discrepancy、diversity/alignment，也不启用 `ce_loss_coef`。

### G3

EBFT/OpenRLHF 的 G3 以 `scripts/diff_dataset/run_G3_rebase_2node_once.sh` 及其调用链为事实源。它保持 G2 的 `cf_l1oo` 和远端 teacher 主结构，同时增加了 EMA、feature adapter、可训练 critic head、critic classifier/direct discrepancy、diversity/alignment，以及 `ce_loss_coef`。

因此，G3 不是简单的脚本参数变体，而是在 G2 主链路上叠加更多 critic、特征和对齐相关机制。

## Slime G1 对齐状态对照

| 类别 | Slime 当前状态 | 证据文件 |
| --- | --- | --- |
| 已对齐 | 已有 G1 主脚本和 smoketest 脚本，自称 Phase-1 parity path，可作为 G1 对齐入口。 | `exper_scripts/main_test/run_g1_openrlhf_qwen35_2b_main.sh`、`exper_scripts/smoketest/run_g1_openrlhf_qwen35_2b_smoke.sh`、`exper_scripts/smoketest/qwen3.5-2B.sh` |
| 已对齐 | 支持固定长度 rollout 和 G1 embedding 相关路径。 | `slime/rollout/g1_embedding.py`、`slime/utils/g1_core.py` |
| 已对齐 | Megatron ref 侧存在 G1 embedding 接入。 | `slime/backends/megatron_utils/actor.py` |
| 已对齐 | 支持 pointwise reward、RLOO shaped rewards、whitening 和 token advantage 展开。 | `slime/utils/g1_core.py`、`slime/backends/megatron_utils/loss.py` |
| 已对齐 | `--g1-use-ebft-loss` 接入 OpenRLHF `EBFTPolicyLoss` 风格 actor loss。 | `slime/utils/arguments.py`、`slime/backends/megatron_utils/loss.py` |
| 已对齐 | 存在 `openrlhf_exact` mask/position 诊断路径。 | `slime/backends/megatron_utils/actor.py`、`slime/backends/megatron_utils/g1_fast.py` |
| 已对齐 | 支持 prompt/label 长度过滤。 | `filter_g1_prompt_length.py` |
| 未对齐/存疑 | KL/entropy 显式不做 parity，且 G1 EBFT loss 与 `use_kl_loss` 存在冲突关系。 | `exper_scripts/main_test/run_g1_openrlhf_qwen35_2b_main.sh`、`slime/backends/megatron_utils/loss.py` |
| 未对齐/存疑 | EBFT 使用 `no_chat_template` 和 `question`/`answer` 字段；Slime 使用 apply-chat-template 和 `prompt`/`label` 字段。需要确认数据语义是否真正等价。 | `exper_scripts/main_test/run_g1_openrlhf_qwen35_2b_main.sh`、`exper_scripts/smoketest/run_g1_openrlhf_qwen35_2b_smoke.sh` |
| 未对齐/存疑 | Slime 的 `rm-type=deepscaler` 不是 EBFT embedding reward 主路径，只能视为统一 RM 钩子的背景能力，不能作为 G1 reward parity 的充分证据。 | `exper_scripts/main_test/run_g1_openrlhf_qwen35_2b_main.sh`、`slime/utils/arguments.py` |
| 未对齐/存疑 | Slime 主脚本默认不运行独立 post-eval；EBFT G1 训练后会运行 code benchmark baseline shell。 | `exper_scripts/main_test/run_g1_openrlhf_qwen35_2b_main.sh`、`exper_scripts/smoketest/run_g1_openrlhf_qwen35_2b_smoke.sh` |
| 未对齐/存疑 | `openrlhf_exact` 默认存在，但 dense attention mask 默认不应用，数值 parity 仍需实验验证。 | `slime/backends/megatron_utils/actor.py`、`slime/backends/megatron_utils/g1_fast.py` |
| 未对齐/存疑 | main/smoke 脚本的部分超参与 EBFT 默认值不同，需要逐项确认哪些是有意差异，哪些会影响 parity。 | `exper_scripts/main_test/run_g1_openrlhf_qwen35_2b_main.sh`、`exper_scripts/smoketest/run_g1_openrlhf_qwen35_2b_smoke.sh` |

## 需要避免的误判

Slime 已有 G1 相关代码能力，并不意味着已经完成 OpenRLHF EBFT G1 的实验复现或数值 parity。

特别是以下点不能被扩大解释：

- `--g1-use-ebft-loss` 表示 actor loss 路径接入了 EBFTPolicyLoss 风格逻辑，但不自动证明整体训练行为等价。
- `openrlhf_exact` 诊断路径存在，不等于默认运行路径已经具备完全一致的 attention mask、position id 和 logprob 数值。
- `rm-type=deepscaler` 是 Slime 的统一 reward model 钩子背景，不是 EBFT G1 embedding reward 主路径的直接对应物。
- Phase-1 parity path 更适合理解为“主链路对齐入口”，不是“已完成 parity 验收”。

## 小结：下一步对齐 run_G2

Slime 当前没有等价的 `run_G2` 脚本，也没有完整的 `cf_l1oo + online teacher` CF 目标训练分支。`actor.py` 中已有的 teacher/log_probs 相关逻辑更接近 OPD/distillation，不能直接等价为 EBFT 标准 G2。

建议后续以 EBFT `scripts/diff_dataset/run_G2_rebase_2node_once.sh` 的 `JOB_SCRIPT` 为事实源，拆出 Slime G2 对齐任务：

- 明确标准 G2 的最小闭环：`cf_l1oo`、在线 vLLM teacher、teacher API base/cache/prefetch、Ray job、`zero_stage=3`、critic head 冻结。
- 将标准 G2 与 `no_teacher_distribution`、`vicinal`、`distillPure` 等变体分开建模，避免把变体能力误认为标准 G2。
- 盘点 Slime 现有 teacher/log_probs、OPD/distillation 逻辑与 EBFT G2 teacher CF 目标的差异，决定是复用、重构还是新增分支。
- 明确 eval 参数传递和 post-eval 行为：EBFT G2 默认 `eval_steps=-1` 但仍传 eval CLI，训练后跑 `run_code_generation_benchmarks.py`。
- 在实现前先形成参数对照表和最小 smoke 脚本，再做数值 parity 检查。
- G3 对齐应在 G2 主链路稳定后再处理，因为 G3 额外引入 EMA、feature adapter、可训 critic head、discrepancy、diversity/alignment 和 `ce_loss_coef`。
