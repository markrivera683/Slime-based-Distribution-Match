# DMFT / Distribution-Matching Finetuning 项目总览

## 0. 这个文档是给谁看的

本文档面向后续接手项目的 coding agent、research agent、实验 agent、写作 agent。目标不是宣传方法，而是让 agent 在不读完整历史对话、不打开 Notion/飞书、不重新推断上下文的情况下，迅速理解这个项目在做什么、为什么做、方法结构是什么、当前实验在验证什么、下一步应该补什么。

项目当前名称可以暂称为 **DMFT: Distribution-Matching Finetuning**。它是在 EBFT（Embedding-Based Feature Training）思想上的扩展：保留“在 feature space 中定义 reward”的核心，但把 reward 从单个 sample 对单个 ground-truth feature 的点对点匹配，升级为一组 samples 与目标答案分布之间的分布匹配。

---

## 1. 项目背景：为什么要做 DMFT

### 1.1 原始 EBFT 的基本设定

原始 EBFT 的核心想法是：不要只用 binary correctness 或 token-level cross entropy 来训练模型，而是在某个固定 critic network 的 feature space 中衡量生成答案与 ground truth 的相似度。

记：

- prompt / context 为 `c`
- ground truth answer 为 `y_gt`
- student model 生成的一个回答为 `ŷ_j`
- critic network 为 `φ_c(·)`
- `z_j = φ_c(ŷ_j)` 表示第 j 个生成结果的 feature
- `z_gt = φ_c(y_gt)` 表示 ground truth 的 feature

原始 EBFT 大致等价于把目标答案压缩成一个 Dirac delta 单点分布：

```math
ν_c = δ(φ_c(y_gt))
```

然后对每个生成结果单独打分。一个 sample 离 `φ_c(y_gt)` 越近，reward 越高。

这种设计的优点是：

1. 比 binary correctness 更平滑。
2. 比 token-level imitation 更关注语义或功能层面的相似度。
3. 可以在 critic feature space 里定义连续 reward。

但它的问题也很明显：它仍然是 **pointwise matching**。

### 1.2 单点奖励的问题

在数学推理、代码生成、开放式 reasoning 等任务里，同一个 prompt 往往有多个正确答案或多个正确推理路径。

数学场景：

- 同一道题可以用代数解。
- 也可以用几何解。
- 也可以用构造法或归纳法。
- 它们 token-level 完全不同，但都是正确 reasoning path。

代码场景：

- 同一个需求可以用不同数据结构实现。
- 可以递归，也可以迭代。
- 可以用标准库，也可以手写逻辑。
- 可以有不同 coding style，但功能等价。

如果 reward 只让学生靠近一个 `y_gt`，模型实际上会被诱导去模仿某一个固定答案，而不是学习“正确答案族”的分布结构。

本质问题：

> 单点 reward 把原本多模态的正确输出空间压缩成了一个 Dirac 点。

DMFT 的出发点就是：

> 对于一个 prompt，模型的多个采样结果本身形成一个经验分布。训练目标不应该只问“这个 sample 像不像 ground truth”，而应该问“这一组 samples 的整体分布像不像正确答案分布”。

---

## 2. 项目的核心主张

DMFT 的核心主张有三层。

第一层：reward 粒度升级。

从：

```text
单个生成结果 vs 单个 ground truth feature
```

变成：

```text
一组生成结果形成的 empirical distribution vs target answer distribution
```

第二层：target 表达升级。

从：

```text
single GT feature
```

变成：

```text
single / vicinal / teacher-generated target distribution
```

第三层：feature geometry 升级。

从：

```text
固定 critic feature space
```

变成：

```text
critic backbone frozen，但通过 lightweight adapter + EMA 让 feature geometry 逐步适配 distribution matching
```

因此，项目从 G1 到 G3 的演化是：

```text
G1: pointwise reward + single GT + frozen feature
G2: distribution reward + richer target distribution + frozen feature
G3: distribution reward + teacher target + adaptive feature geometry
```

---

## 3. 方法版本总览

| 方法                 | Reward                       | Target                         | Teacher | Feature Space | 核心问题                                          |
| ------------------ | ---------------------------- | ------------------------------ | ------- | ------------- | --------------------------------------------- |
| G1 / EBFT baseline | pointwise feature reward     | single GT                      | 否       | frozen critic | 原始 EBFT 是否有效                                  |
| G2-single          | CF + LOO distribution reward | single GT                      | 否       | frozen critic | distribution reward 本身是否有收益                   |
| G2-vicinal         | CF + LOO distribution reward | vicinal GT distribution        | 否       | frozen critic | 把单点 target 平滑成局部分布是否有收益                       |
| G2-teacher         | CF + LOO distribution reward | teacher empirical distribution | 是       | frozen critic | 用真实 teacher 多样输出作为 target 是否有收益               |
| G3                 | CF + LOO distribution reward | teacher empirical distribution | 是       | adapter + EMA | feature geometry 是否需要适配 distribution matching |

---

## 4. G1：原始 EBFT baseline

G1 可以理解为原始 EBFT 或 pointwise EBFT baseline。

它做的事情是：

1. 对 prompt `c` 生成一个或多个 student outputs。
2. 用固定 critic `φ_c` 提取每个 output 的 feature。
3. 用同一个 critic 提取 ground truth 的 feature。
4. 每个 sample 单独与 GT feature 比较。
5. 把 feature-level similarity / distance 转成 reward。
6. 用该 reward 更新 student policy。

G1 的研究意义：

- 它是整个方法的起点。
- 它检验“feature-level reward”相比传统 SFT / binary reward 是否有意义。
- 它也是 G2/G3 的必要对照组。

G1 的核心限制：

- 只看单个 sample。
- 只对齐单个 GT。
- 不显式建模同一 prompt 下输出的多样性。
- 对 coding/reasoning 的多解性表达不足。

---

## 5. G2：Distribution Reward

G2 是项目的核心创新层。它让 reward 从 pointwise 变成 distribution-level。

### 5.1 Student empirical distribution

对于同一个 prompt `c`，从当前 student policy 中采样 `N` 个输出：

```math
\hat{y}_1, \hat{y}_2, ..., \hat{y}_N
```

经过 critic feature extractor：

```math
z_j = φ_c(\hat{y}_j)
```

这组 features 构成 student 当前在 prompt `c` 下的经验分布：

```math
μ_{θ,c} = \frac{1}{N}\sum_{j=1}^{N}δ(z_j)
```

直觉：

> `μ_{θ,c}` 是当前模型对这个 prompt 的“输出行为指纹”。

如果模型真的学会了这个任务，它在同一个 prompt 下采样出来的一组答案，不一定 token-level 一样，但应该在语义、功能、推理结构上落在正确答案区域内。

### 5.2 Target distribution

目标不再只是 `δ(φ_c(y_gt))`，而是某种目标答案分布 `ν_c`。

G2 支持至少三类 target construction：

1. `single`
2. `vicinal`
3. `teacher`

这三类 target 见第 7 节。

### 5.3 Distribution discrepancy

G2 的核心问题是：如何度量 student distribution `μ_{θ,c}` 与 target distribution `ν_c` 的距离。

当前方案使用 **Characteristic Function Discrepancy**，简称 CFD 或 CF discrepancy。

理由：

- characteristic function 理论上可以刻画完整分布。
- 它在频率域比较分布，而不只是比较均值。
- 对 shape、spread、phase、multimodal structure 更敏感。
- 可以通过 random frequency Monte Carlo approximation 高效实现。

---

## 6. Characteristic Function Discrepancy，CFD

### 6.1 特征函数定义

对于 feature distribution `μ`，其 characteristic function 为：

```math
ψ_μ(t) = E_{z \sim μ}[e^{i t^T z}]
```

因为实际分布是 empirical distribution，所以可以用 samples 求平均。

对固定 random frequencies：

```math
t_l \sim \mathcal{N}(0, σ^{-2}I), \quad l=1,...,L
```

当前默认：

```text
L = 128
σ = 1.0
```

对每个 frequency `t_l`，计算 student empirical characteristic function：

```math
ψ_μ(t_l)=\frac{1}{N}\sum_j e^{i t_l^T z_j}
```

实现时拆成实部和虚部：

```math
Re ψ_μ(t_l)=\frac{1}{N}\sum_j cos(t_l^T z_j)
```

```math
Im ψ_μ(t_l)=\frac{1}{N}\sum_j sin(t_l^T z_j)
```

同理对 target distribution `ν` 计算 `ψ_ν(t_l)`。

### 6.2 固定 random frequencies

频率 `t_l` 是预采样并固定的，不是训练参数。

这样做的原因：

1. 避免 frequency 本身漂移导致 reward 非平稳。
2. 保证 reward function 对 student features 可微。
3. 实现上类似 random Fourier features，计算稳定。
4. 方便不同 step 之间保持一致的 distribution metric。

### 6.3 Amplitude + phase discrepancy

当前 CFD 使用 amplitude 和 phase 的组合。

简化写法：

```math
D_{CF}(μ,ν)=\frac{1}{L}\sum_l \sqrt{α(|ψ_ν(t_l)|-|ψ_μ(t_l)|)^2 + βp_l}
```

其中 phase loss：

```math
p_l = 2(|ψ_ν(t_l)||ψ_μ(t_l)| - R_μR_ν - I_μI_ν)
```

`R_μ, I_μ` 分别是 student characteristic function 在该 frequency 的实部和虚部，`R_ν, I_ν` 是 target 的实部和虚部。

直觉：

- amplitude 比较分布在频率域上的整体强度或 spread。
- phase 比较分布的位置和形状方向。
- 只比较 amplitude 可能把不同位置或不同形状的分布误判为相近。
- phase 项让 metric 对 distribution geometry 更敏感。

### 6.4 Agent 需要注意的实现点

实现 CFD 时，agent 应检查：

1. feature tensor shape 是否正确。
2. student samples 与 target samples 是否按 prompt 对齐。
3. frequencies 是否只初始化一次。
4. frequencies 是否放在正确 device 和 dtype 上。
5. `cos/sin` 的 projection 是否在 feature dimension 上正确 inner product。
6. amplitude/phase 的数值稳定项是否存在，例如 `sqrt(x + eps)`。
7. mixed precision 下是否有 overflow / underflow。
8. group 内 sample 数 `N` 是否大于 1，否则 LOO 不成立。

---

## 7. Leave-One-Out attribution：把 group reward 拆成 sample reward

### 7.1 为什么需要 LOO

CFD 给的是 group-level discrepancy：

```math
D_{CF}(X,Y)
```

其中 `X` 是 student sample set，`Y` 是 target sample set。

但 policy gradient / RL-style finetuning 通常需要每个 sample 的 reward：

```math
r_j
```

所以必须把 set-level reward 分解到 sample-level。

### 7.2 LOO reward 定义

DMFT 使用 leave-one-out marginal contribution：

```math
r_j = D_{CF}(X \setminus j, Y) - D_{CF}(X,Y)
```

解释：

- 如果去掉第 j 个 sample 后，distribution discrepancy 变大，说明这个 sample 对匹配 target distribution 有正贡献。
- 这种 sample 应该获得正 reward。
- 如果去掉它以后 discrepancy 变小，说明它拖累了 distribution matching，应获得低 reward 或负 reward。

### 7.3 与 Shapley value 的关系

LOO 可以理解为 Shapley value 的简化近似。

完整 Shapley value 需要枚举所有 coalition，计算量不可接受。LOO 只看“从完整集合中移除当前元素”的边际贡献，成本低，解释性强。

### 7.4 向量化实现

朴素实现：

对每个 sample j 重新计算一次 `D_CF(X \ j, Y)`，复杂且慢。

优化实现：

先计算所有 sample 在所有 frequencies 上的：

```text
cos(t_l^T z_j)
sin(t_l^T z_j)
```

得到 full sum / mean：

```math
mean = sum / N
```

对每个 j，LOO mean 可以直接算：

```math
mean_{-j} = (sum - value_j)/(N-1)
```

这样不需要循环重算所有投影。

### 7.5 复杂度

主要复杂度：

```math
O(B · G · K · (N+M) · L · d)
```

符号解释：

| 符号 | 含义                                  |
| -- | ----------------------------------- |
| B  | micro batch size                    |
| G  | group 数                             |
| K  | block 数                             |
| N  | student samples per prompt，通常 4 或 8 |
| M  | target samples 数                    |
| L  | CF frequencies，默认 128               |
| d  | critic feature dimension            |

当前判断：

- N 通常较小，LOO 本身不是最大瓶颈。
- 主要开销在 critic feature forward 和 frequency projection。
- 如果要加速，优先考虑 critic feature caching、teacher target precompute、projection kernel 优化、异步 rollout/training pipeline。

---

## 8. Target distribution construction

G2/G3 的 target distribution 是关键，因为如果 target 仍然是 single GT point，那么 distribution matching 的潜力无法充分释放。

当前有三类 target mode。

### 8.1 single target

定义：

```math
ν_c = δ(φ_c(y_gt))
```

即只使用 ground truth feature。

作用：

- 作为最保守的对照组。
- 检验“即便 target 是单点，distribution reward 是否也比 pointwise reward 更好”。
- 帮助区分 reward-side upgrade 和 target-side upgrade。

注意：

如果 G2-single 有提升，说明 CF + LOO 本身有价值。如果 G2-single 没有提升而 G2-teacher 有提升，则主要收益可能来自 teacher target。

### 8.2 vicinal target

定义：

在 `φ_c(y_gt)` 周围加 Gaussian noise，构造局部 target cloud。

直觉：

- 将单点 GT 平滑成局部邻域。
- 低成本模拟“答案附近的可接受区域”。
- 不需要 teacher generation。

优点：

- 成本低。
- 可控。
- 适合做 target-side smoothing baseline。

缺点：

- 噪声不一定对应真实可行答案。
- 只是在 feature space 中做局部扰动，不能真正表达多种 reasoning path。
- 可能只提供 regularization，而不是语义多样性。

### 8.3 teacher target

定义：

使用 frozen teacher model 对同一个 prompt 采样 `M` 个 completions：

```math
y_i^T \sim q_T(·|c)
```

用 critic 提取 feature，构造 teacher empirical distribution。

混合 GT 和 teacher 的一般形式：

```math
ν_c = (1-λ)δ(φ_c(y_gt)) + λ \frac{1}{M}\sum_{i=1}^{M}δ(φ_c(y_i^T))
```

当：

```text
λ = 1
```

退化为 pure teacher target：

```math
ν_c = \frac{1}{M}\sum_{i=1}^{M}δ(φ_c(y_i^T))
```

当前配置中提到：

```text
cf_teacher_lambda = 1
```

也就是 pure teacher target。

### 8.4 teacher target 的直觉

好的 teacher 对同一道题可能生成多种正确或近似正确的解法。这些解法在 feature space 中形成一个经验分布。这个分布比单个 GT 更接近“正确答案族”的真实结构。

DMFT 不是要求 student 抄 teacher 的某一个答案，而是让 student 的输出分布靠近 teacher 的答案分布。

这和普通 distillation 的区别是：

- 普通 SFT / offline distillation：更像 sample-level imitation。
- DMFT teacher target：更像 feature-distribution-level distillation。

### 8.5 当前模型配置

根据已有报告，当前主配置为：

| 角色      | 模型          | 状态                        |
| ------- | ----------- | ------------------------- |
| Student | Qwen3.5-4B  | 可训练                       |
| Teacher | Qwen3.5-27B | Frozen，只提供 target support |
| 参数量比    | 约 6.75x     | teacher 大于 student        |

之前还提到过 Qwen0.8B 相关实验，teacher-student scale gap 约 33.75x。这个结果用于说明即使 teacher-student gap 很大，teacher target 仍可能有效。

Agent 注意：模型名称、版本号和实际 checkpoint 路径必须以当前代码/实验配置为准。本文档只记录已有报告中的上下文。

---

## 9. G3：Adaptive Feature Geometry

G2 G假设 critic feature space 是固定且合理的。但这个假设不一定成立。

如果 critic embedding 本身不能很好区分：

- 正确 vs 错误答案
- 不同 reasoning mode
- 功能等价代码
- 表面相似但语义错误答案

那么再好的 distribution metric 也只是在一个不理想的坐标系中做匹配。

G3 的目标是：

> 不大规模 finetune critic backbone，而是用轻量 adapter 让 feature geometry 逐步适配 distribution matching。

### 9.1 为什么不能直接解冻 critic

完全解冻 critic backbone 的风险：

1. reward non-stationary 太强。
2. student 和 critic 可能互相追逐，导致训练不稳定。
3. critic feature geometry 可能 collapse。
4. 计算和显存开销增加。
5. 难以判断收益来自 student 变强还是 critic 漂移。

所以 G3 采用更保守的局部适配方式。

### 9.2 Residual Bottleneck Adapter

Adapter 架构：

```math
x → LayerNorm → Linear(d,r) → GELU → Dropout → Linear(r,d) → residual add
```

当前报告中的配置：

| 参数              | 值                    |
| --------------- | -------------------- |
| Adapter type    | residual\_bottleneck |
| Bottleneck rank | 64                   |
| Dropout         | 0.0                  |
| Unfreeze layers | 0                    |
| Trainable       | adapter + small head |
| Backbone        | frozen               |

Adapter 不是插入 transformer 每一层，而是作用在 critic 提取出的 hidden feature stream 上。

流程大致是：

```text
text → frozen critic backbone → hidden features → adapter → feature/reward computation
```

### 9.3 Zero-init up projection

关键工程细节：adapter 的 up projection zero-init。

这意味着训练开始时 adapter 近似恒等映射，不会立即破坏原始 critic feature geometry。

因此：

- G3 early stage 接近 G2。
- 随着训练推进，adapter 慢慢调整 feature geometry。
- 这种设计有利于稳定训练。

### 9.4 EMA target geometry

G3 使用 EMA 来稳定 target-side feature geometry。

更新形式：

```math
θ_{EMA} ← β θ_{EMA} + (1-β) θ_{online}
```

当前报告中：

```text
β = 0.99
```

作用：

- online adapter 可以学习。
- target geometry 以更慢速度变化。
- 避免 student 和 target representation 同时剧烈漂移。
- 类似 target network / momentum encoder 的稳定机制。

---

## 10. 训练数据与实验上下文

根据已有对话上下文，当前训练数据集提到的是：

```text
Open-Code-Instruct 100k
```

任务类型主要是 coding instruction / code generation。

Agent 注意：以上是历史上下文，不是最终实验结论。任何写入论文或报告的数字必须重新从日志、表格或 evaluation result 中读取。

---

##

## 13. 不用做的，你只负责明白这个项目的前身是什么就好，不用做所谓的“需要可视化的内容”（如下）

这个项目特别适合做可视化，因为它的核心故事是“从点到分布”。

### 13.1 Pipeline 图

建议画一张主图，包含：

1. 输入 prompt。
2. Student 采样 N 个 outputs。
3. Teacher / GT 构造 target samples。
4. Critic 提取 features。
5. Student features 形成 empirical distribution。
6. Target features 形成 target distribution。
7. CFD 计算 distribution discrepancy。
8. LOO attribution 生成 per-sample reward。
9. Policy optimization 更新 student。
10. G3 中 adapter + EMA 的位置。

图的中心 message：

```text
Reward is assigned to each sample by its marginal contribution to matching the target feature distribution.
```

### 13.2 Feature distribution visualization

建议用 PCA / UMAP / t-SNE 展示：

- GT feature point。
- Teacher target cloud。
- Student samples before training。
- Student samples after G1。
- Student samples after G2。
- Student samples after G3。

想证明的现象：

- G1 可能向单点收缩。
- G2/G3 更好覆盖 target support。
- G3 的 feature geometry 更分离 correct/incorrect regions。
-

## 14. 当前工程与加速方向这个是着重要考虑的地方！

已有对话中提到后续要加入“加速的创新点”，并考虑使用 `slime` 或类似框架做 async。以及OPD

当前可能瓶颈：

1. Student rollout：每个 prompt 要采样 N 个 outputs。
2. Teacher target generation：如果在线生成，成本很高。
3. Critic feature extraction：student + target 都要过 critic。
4. Frequency projection：`(N+M) × L × d` 级别 projection。
5. LOO reward attribution：虽然已向量化，但仍有额外 group-level 操作。

建议工程策略：

| 优化                           | 说明                                   |
| ---------------------------- | ------------------------------------ |
| teacher target offline cache | teacher completions 和 features 尽量预计算 |
| critic feature cache         | 对固定 GT / teacher target features 缓存  |
| async rollout                | rollout 与 training 解耦，提高 GPU 利用率     |
| vectorized LOO               | 避免 per-sample loop                   |
| fused projection             | 优化 `z @ t` 的频率投影                     |
| mixed precision guard        | `cos/sin/sqrt` 部分注意数值稳定              |
| block-level batching         | 按 prompt group 合并计算                  |
| target refresh schedule      | teacher target 不必每 step 重新采样         |

Agent 如果接手工程，应优先检查：

1. teacher samples 是否已经缓存。
2. target features 是否重复计算。
3. critic 是否和 student 共用 GPU 导致资源争用。
4. rollout 是否阻塞 training。
5. dataloader 是否正确按 prompt group 组织 N 个 samples。

---

## 18. Agent 接手时的最低上下文

如果一个新 agent 只能读 20 行，它至少要知道：

1. 项目叫 DMFT，目标是改进 EBFT。
2. 原始 EBFT 是 pointwise feature reward：每个 sample 对齐 single GT feature。
3. DMFT 的核心是 distribution matching：同一 prompt 下 student 的 N 个 samples 形成 empirical feature distribution。
4. Target distribution 可以是 single GT、vicinal GT cloud、teacher completions。
5. Distribution discrepancy 当前用 Characteristic Function Discrepancy。
6. Group-level discrepancy 通过 Leave-One-Out 转成 per-sample reward。
7. G2 是 frozen critic 上的 distribution reward。
8. G3 加 residual bottleneck adapter 和 EMA target geometry，让 feature space 适配。
9. 当前主场景是 coding，数据集提到 Open-Code-Instruct 100k。
10. 当前 student/teacher 提到 Qwen3.5-4B / Qwen3.5-27B。
11. 评测重点不能只看 greedy，要看 pass\@1 和 pass\@16/pass\@k。
12. 20\. 当前最重要的结论

DMFT 目前不是一个“已经完全验证完”的项目，而是一个 research direction 已经成型、early signal 看起来不错、但仍需要系统 baseline 和消融来确认贡献归因的项目。

它最强的研究叙事是：

> 传统 feature-based finetuning 把正确答案空间压成一个点；DMFT 把同一 prompt 下的 student outputs 看成一个分布，并让这个分布去匹配由 GT、vicinal samples 或 teacher completions 构成的目标答案分布。通过 characteristic-function discrepancy 衡量分布距离，再用 leave-one-out attribution 把 group-level reward 分配给单个 sample。进一步地，G3 用 adapter + EMA 让 critic feature geometry 稳定地适配这个分布匹配目标。
>
>
