下面按“目标函数 → 参数动态机制 → EffOPD 加速算法 → 理论解释 → 实现细节”的顺序详细讲。核心结论先放在前面：这篇论文的 OPD 加速不是简单把 learning rate 调大，而是利用 OPD 训练早期已经形成的稳定参数方向，对当前更新方向做受控线性外推。论文把这个方法叫 EffOPD。它依赖两个观察：一是 OPD 早期已经知道该更新哪些模块；二是 OPD 早期已经锁定接近最终解的低秩更新方向。

一、OPD 本身的训练目标

先定义学生模型为 (\pi_\theta)，教师模型为 (\pi^*)。输入 prompt 为 (x)，学生自己生成轨迹：

[
y=(y_1,\dots,y_T)\sim \pi_\theta(\cdot|x)
]

注意“on-policy”的关键就在这里：训练样本不是 teacher 生成的，也不是离线固定数据，而是当前 student 自己采样出来的。然后 teacher 在 student 自己走出来的 token 序列上给监督信号。

OPD 的目标是最小化 student 与 teacher 的 reverse KL：

[
J_{\mathrm{OPD}}(\theta)
========================

\min_\theta
\mathbb{E}*{x\sim D,; y\sim \pi*\theta(\cdot|x)}
\left[
D_{\mathrm{KL}}\left(
\pi_\theta(y|x);|;\pi^*(y|x)
\right)
\right]
]

这里的直觉是：student 自己生成当前分布下会遇到的 response，然后 teacher 告诉 student：在这些位置上，你的 token 分布应该往 teacher 靠近。

更细地展开到 token-level，论文给出的 OPD 梯度形式是：

[
\nabla_\theta J_{\mathrm{OPD}}(\theta)
======================================

\mathbb{E}*{x\sim D,; y\sim \pi*\theta(\cdot|x)}
\left[
\sum_{t=1}^{T}
\sum_{t'=t}^{T}
\left(
\log \pi_\theta(y_{t'}|x,y_{<t'})
---------------------------------

\log \pi^*(y_{t'}|x,y_{<t'})
\right)
\nabla_\theta \log \pi_\theta(y_t|x,y_{<t})
\right]
]

这个公式比普通 token-level KD 更复杂，因为它考虑了 on-policy trajectory 中前面动作对后面 token 分布的影响。直观地说，第 (t) 个 token 的 policy gradient 会影响后续 (t'\ge t) 的分布差异。

实践里常用近似，把这种长程 credit assignment 简化为 immediate token-level reverse KL，也就是论文中的近似式：

[
\nabla_\theta J_{\mathrm{OPD}}(\theta)
\approx
\mathbb{E}*{x\sim D,; y\sim \pi*\theta(\cdot|x)}
\left[
\sum_{t=1}^{T}
\left(
\log \pi_\theta(y_t|x,y_{<t})
-----------------------------

\log \pi^*(y_t|x,y_{<t})
\right)
\nabla_\theta \log \pi_\theta(y_t|x,y_{<t})
\right]
]

这一项非常重要。它说明 OPD 的每个 token 都有一个 dense supervision signal：

[
\log \pi_\theta(y_t|c_t)-\log \pi^*(y_t|c_t)
]

其中：

[
c_t=(x,y_{<t})
]

如果 student 给某个 token 的 log-prob 比 teacher 偏高或偏低，梯度都会直接调整。这和 RLVR 很不一样。RLVR 通常只有最终答案对错的 sparse reward，而 OPD 在每个 token 上都能得到 teacher distribution 的细粒度信号。论文进一步强调，dense supervision 只是表层解释，更深层原因是 OPD 的参数更新方向更稳定、更低秩、更少冗余。

二、RL 与 OPD 的参数更新差异

论文统一用 base model 参数 (W_{\mathrm{Base}}) 作为起点，定义训练后的参数更新：

[
\Delta W_{\mathrm{RL/OPD}}
==========================

W_{\mathrm{RL/OPD}}-W_{\mathrm{Base}}
]

然后比较两个问题：相同更新 norm 下，谁带来更多性能提升；达到相同性能时，谁需要更小的参数移动。

他们做了一个 scaling analysis：固定最终更新方向，只改变更新幅度：

[
W(\alpha)
=========

W_{\mathrm{Base}}+\alpha \Delta W_{\mathrm{RL/OPD}},
\quad
\alpha\in[0,1]
]

也就是说，方向不变，只缩放大小。如果 OPD 在相同 (|\alpha\Delta W|) 下性能更高，就说明 OPD 的更新方向里 task-relevant signal 更密集；如果 RL 更新 norm 很大但性能不对应增长，就说明 RL 有更多 weakly task-correlated components。论文图 2 的结果正是这个结论：OPD 用更小的参数更新达到相近甚至更好的推理性能。

这个分析为加速提供了第一层依据：OPD 的训练路径更“直”。如果路径很直，后面就可以沿当前方向外推。

三、机制 1：Functional Redundancy Avoidance

论文的第一个核心机制叫 Functional Redundancy Avoidance，即“功能冗余避免”。

Transformer 的不同模块对 reasoning 的贡献不是均匀的。论文把参数更新拆成 embedding、MLP、attention，并进一步看不同层的位置。结论是：embedding 更新对 reasoning gain 基本没用；中间层 MLP 更关键；bottom/top layers 的边际收益较低。论文图 3 的 sliding-window intervention 显示，中间层干预带来的性能增益更高，而底层和顶层较低。

形式上，设 Transformer 有 (L) 层，每层有 Attention 和 MLP。对于某个中心层 (l)，论文定义滑动窗口：

[
\mathcal{W}_l
=============

\left{
i\in\mathbb{Z}
\mid
\max(1,l-8)\le i\le \min(L,l+8)
\right}
]

窗口宽度最多 17 层。然后只把某个窗口中的 OPD 或 RL 更新注入 base model，其他部分保持 base 不变，观察性能变化。

MLP intervention 可以写成：

[
W^{(\mathrm{interv})}_{\mathrm{MLP},l}
======================================

\begin{cases}
W^{(i,\mathrm{MLP})}*{\mathrm{Base}}
+
\Delta W^{(i,\mathrm{MLP})}*{\mathrm{RL/OPD}},
&
i\in \mathcal{W}*l
\
W^{(i,\mathrm{MLP})}*{\mathrm{Base}},
&
i\notin \mathcal{W}_l
\end{cases}
]

同时 Attention 保持 base：

[
W^{(i,\mathrm{Attn})}=W^{(i,\mathrm{Attn})}_{\mathrm{Base}}
]

Attention intervention 类似：

[
W^{(\mathrm{interv})}_{\mathrm{Attn},l}
=======================================

\begin{cases}
W^{(i,\mathrm{Attn})}*{\mathrm{Base}}
+
\Delta W^{(i,\mathrm{Attn})}*{\mathrm{RL/OPD}},
&
i\in \mathcal{W}*l
\
W^{(i,\mathrm{Attn})}*{\mathrm{Base}},
&
i\notin \mathcal{W}_l
\end{cases}
]

同时 MLP 保持 base。

这个实验的关键发现不是“OPD 和 RL 用了完全不同的模块”，而是：OPD 和 RL 的敏感区域大致相似，都依赖模型已有的 reasoning-critical structures；区别在于 RL 会在低敏感区域积累较大 norm，而 OPD 会抑制这些低效区域，把更新集中到更有用的中间层模块。论文把这称为 OPD 在 module-allocation level 的 foresight。

用一句更工程化的话说：OPD 的更新预算利用率更高；RL 花了更多参数位移去改一些“不怎么涨分”的地方。

四、机制 2：Early Low-Rank Lock-in

第二个核心机制叫 Early Low-Rank Lock-in，即“早期低秩锁定”。

对任意一个参数矩阵更新：

[
\Delta W\in \mathbb{R}^{m\times n}
]

做 SVD：

[
\Delta W
========

U\Sigma V^\top
]

其中：

[
\Sigma=\mathrm{diag}(\sigma_1,\sigma_2,\dots,\sigma_r)
]

并且：

[
\sigma_1\ge \sigma_2\ge \dots \ge \sigma_r>0
]

论文用几个指标衡量更新是否集中在少数方向上。

第一个是 spectral norm：

[
|\Delta W|_2=\sigma_1
]

它表示最大奇异方向上的更新强度。

第二个是 spectral-to-Frobenius ratio：

[
\rho
====

\frac{\sigma_1}
{|\Delta W|_F}
==============

\frac{\sigma_1}
{\sqrt{\sum_{j=1}^{r}\sigma_j^2}}
]

如果 (\rho) 越大，说明最大主方向占比越高，更新越集中。

第三个是 effective rank：

[
\mathrm{rank}_{\mathrm{eff}}
============================

\exp
\left(
-\sum_{i=1}^{r}
\bar{\sigma}_i\log \bar{\sigma}_i
\right)
]

其中：

[
\bar{\sigma}_i
==============

\frac{\sigma_i}{\sum_{j=1}^{r}\sigma_j}
]

effective rank 越低，说明奇异值分布越集中，更新越接近低秩结构。

第四个是 Top-1% Subspace Norm Ratio。令：

[
k=\lceil r/100\rceil
]

取前 (k) 个奇异方向构造低秩近似：

[
\Delta W_k
==========

U_{:,1:k}\Sigma_{1:k,1:k}V_{:,1:k}^{\top}
]

然后定义：

[
R_{\mathrm{Top\text{-}1%}}
==========================

# \frac{|\Delta W_k|_F}{|\Delta W|_F}

\sqrt{
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{j=1}^{r}\sigma_j^2}
}
]

如果这个值接近 1，说明前 1% 的奇异方向已经吸收了大部分更新能量。论文表 1 显示，OPD 在多个模型尺度上都有更低 effective rank、更高 Top-1% 能量占比。比如 8B 模型上，OPD 的 Top-1% subspace norm ratio 是 94.7%，RL 是 88.5%；OPD 的 effective rank 是 2341，RL 是 2754。

这个结果说明：OPD 的更新不是均匀散在高维参数空间里，而是集中在少数 dominant subspaces 上。

五、主子空间和尾部子空间的功能差异

论文进一步把更新频谱分为 principal subspace 和 tail subspace。

Top-(k)% principal update 定义为：

[
\Delta W_{\mathrm{Top}\text{-}k%}
=================================

U_{:,1:k}\Sigma_{1:k,1:k}V_{:,1:k}^{\top}
]

Bottom-(k)% tail update 可以定义为：

[
\Delta W_{\mathrm{Bottom}\text{-}k%}
====================================

U_{:,r-k+1:r}\Sigma_{r-k+1:r,r-k+1:r}V_{:,r-k+1:r}^{\top}
]

对 Top-(k)%，论文会把 OPD 和 RL 的 Frobenius norm 调到一致，再比较性能。这是在控制更新大小后，只比较方向质量。结果是，前 10% rank 就能恢复超过 95% 的 full-model reasoning performance，而且 OPD 的 Top-(k)% 方向质量持续优于 RL。

对 Bottom-(k)%，论文不做 norm equalization，而是看训练原始状态下 tail subspace 的实际贡献。结果是 tail directions 的边际收益很低；RL 的 tail norm 明显更大，但性能提升有限。论文给出的现象是，RL 的 bottom-50% tail subspace norm 大约是 OPD 的 1.6 到 2.5 倍，但带来的 gain 很小。

这点非常关键。它解释了为什么 OPD 能被外推：有效性能主要沿少数主方向增长；如果这些主方向早期已经稳定，那么后续训练多数是在增加这些方向上的幅度，而不是找新方向。

六、早期 checkpoint 的 norm scaling 证据

论文还有一个很直接的实验：拿早期 OPD checkpoint，比如训练进度 10% 的模型，保留每个模块的方向不变，只把每个模块的 Frobenius norm 放大到最终 checkpoint 的水平。

设第 (t) 个 checkpoint 的某个模块更新为：

[
\Delta W_{t}^{(m)}
==================

W_t^{(m)}-W_{\mathrm{Base}}^{(m)}
]

最终 checkpoint 为 (T)：

[
\Delta W_T^{(m)}
================

W_T^{(m)}-W_{\mathrm{Base}}^{(m)}
]

module-wise norm scaling 可以写成：

[
\widetilde{\Delta W}_{t}^{(m)}
==============================

\Delta W_t^{(m)}
\cdot
\frac{
|\Delta W_T^{(m)}|_F
}{
|\Delta W_t^{(m)}|_F
}
]

然后组合出：

[
\widetilde{W}_t
===============

W_{\mathrm{Base}}
+
{\widetilde{\Delta W}_{t}^{(m)}}_m
]

这相当于问：如果早期方向不变，只补足 magnitude，模型能恢复多少最终性能？

论文结果是：10% training progress 的 OPD checkpoint 经过这种 scaling 后，可以恢复约 80% 的最终 reasoning performance，同时 KL divergence 也下降，说明输出分布更接近 teacher。

附录里还给了一个带 (\beta) 的全局 scaling 形式：

[
\Delta W_{\mathrm{scaled}}
==========================

\Delta W_{\mathrm{early}}
+
\Delta W_{\mathrm{early}}
\cdot
\frac{
\beta
\left(
|\Delta W_{\mathrm{final}}|_F
-----------------------------

|\Delta W_{\mathrm{early}}|*F
\right)
}{
|\Delta W*{\mathrm{early}}|_F
}
]

可以化简成：

[
\Delta W_{\mathrm{scaled}}
==========================

\left[
1+
\beta
\left(
\frac{|\Delta W_{\mathrm{final}}|*F}
{|\Delta W*{\mathrm{early}}|*F}
-1
\right)
\right]
\Delta W*{\mathrm{early}}
]

当 (\beta=0) 时，不额外放大：

[
\Delta W_{\mathrm{scaled}}=\Delta W_{\mathrm{early}}
]

当 (\beta=1) 时，scaled update 的 norm 匹配 final update 的 norm。论文发现，(\beta) 从 0 增大时性能上升，大约 (\beta\approx 0.8) 后开始平台化，超过约 1.2 后性能下降。这说明“方向早期是对的”，但不能无限放大；过度放大会把噪声或非任务相关成分也放大。

七、EffOPD 的核心算法

现在进入加速算法本体。

令 (W_t) 表示经过第 (t) 次 OPD update 后的模型参数。EffOPD 不在每一步都外推，而是在指数间隔 checkpoint 上触发：

[
t=2^n
]

其中 (n=0,1,2,\dots)。也就是第 1、2、4、8、16……步触发 extrapolation search。

为什么用指数间隔？因为早期方向变化快，需要更频繁检查；后期方向已经更稳定，可以减少检查频率，降低 validation overhead。同时，两个指数 checkpoint 之间的 displacement 足以代表一段局部训练方向。

在第一个 checkpoint，即 (t=1)，用初始参数到 (W_1) 的位移作为方向：

[
\Delta_0=W_1-W_0
]

对于后续 checkpoint，论文定义局部方向为当前指数 checkpoint 与上一个指数 checkpoint 的参数差：

[
\Delta_n
========

## W_{2^n}

W_{2^{n-1}}
]

这个 (\Delta_n) 是 EffOPD 的核心方向估计。它不是单步梯度，而是一段 OPD 训练累计出来的 parameter displacement。论文认为，由于 OPD 方向早期稳定，所以这个 displacement 可以近似后续优化方向。

然后从当前参数 (W_{2^n}) 出发，沿 (\Delta_n) 生成 5 个候选：

[
W_{\mathrm{fn},k}
=================

W_{2^n}
+
2^k\Delta_n,
\quad
k=1,2,\dots,5
]

注意这里是 (2^k)，不是 (k)。所以候选依次是：

[
W_{\mathrm{fn},1}=W_{2^n}+2\Delta_n
]

[
W_{\mathrm{fn},2}=W_{2^n}+4\Delta_n
]

[
W_{\mathrm{fn},3}=W_{2^n}+8\Delta_n
]

[
W_{\mathrm{fn},4}=W_{2^n}+16\Delta_n
]

[
W_{\mathrm{fn},5}=W_{2^n}+32\Delta_n
]

这相当于沿当前 OPD 已经确认的方向大胆前进，但不是直接无条件接受。

论文随机采样 50 个训练样本构造 lightweight validation set：

[
D_v\subset D,\quad |D_v|=50
]

定义验证函数：

[
V_{D_v}(W)
]

这个验证函数可以理解为在小验证集上的任务分数、reward、accuracy 或 distillation validation objective。论文没有把它写死为某一种指标，而是抽象为 (V_{D_v}(\cdot))。

初始化：

[
W_{\mathrm{acc}}=W_{2^n}
]

[
v_{\mathrm{acc}}=V_{D_v}(W_{2^n})
]

然后按 (k=1,2,\dots,5) 顺序评估候选。如果候选不差于当前接受模型：

[
V_{D_v}(W_{\mathrm{fn},k})
\ge
v_{\mathrm{acc}}
]

则接受：

[
W_{\mathrm{acc}}
\leftarrow
W_{\mathrm{fn},k}
]

[
v_{\mathrm{acc}}
\leftarrow
V_{D_v}(W_{\mathrm{fn},k})
]

如果某个候选第一次失败，即：

[
V_{D_v}(W_{\mathrm{fn},k})
<
v_{\mathrm{acc}}
]

则立即停止搜索。最终参数设为：

[
W_{2^n}^{\mathrm{EffOPD}}
=========================

W_{\mathrm{acc}}
]

如果 (k=1) 的候选都失败，那么：

[
W_{\mathrm{acc}}=W_{2^n}
]

此时 EffOPD 退化为 vanilla OPD，不做外推。这个设计保证了保守性。

八、EffOPD 的伪代码

可以写成下面这样：

```text
Input:
  initial student parameters W0
  teacher model π*
  training data D
  validation sample size m = 50
  max extrapolation candidates K = 5

Train OPD normally.

For each OPD step t:
    perform one normal OPD update:
        W_t ← OPD_Update(W_{t-1}, π*, batch)

    if t is power of 2:
        n ← log2(t)

        if t == 1:
            Δ_n ← W_1 - W_0
        else:
            Δ_n ← W_{2^n} - W_{2^{n-1}}

        sample lightweight validation set D_v from D, |D_v| = 50

        W_acc ← W_{2^n}
        v_acc ← V_{D_v}(W_acc)

        for k = 1 to 5:
            W_candidate ← W_{2^n} + 2^k Δ_n
            v_candidate ← V_{D_v}(W_candidate)

            if v_candidate >= v_acc:
                W_acc ← W_candidate
                v_acc ← v_candidate
            else:
                break

        W_{2^n} ← W_acc
        continue OPD training from W_acc
```

工程上要注意，候选 (W_{\mathrm{fn},k}) 都是从原始 (W_{2^n}) 加不同倍数的 (\Delta_n)，不是从上一个 accepted candidate 继续递推加。也就是说，候选是同一条射线上的多个离散点。

九、EffOPD 和调大学习率的区别

表面看，EffOPD 像是“走大步”，但它和简单增大学习率不是一回事。

增大学习率会改变每个 mini-batch 梯度更新：

[
W_{t+1}=W_t-\eta \nabla L_t
]

如果 (\eta) 过大，梯度噪声、batch variance、teacher-student mismatch 都会被放大，容易振荡。

EffOPD 的形式更像 checkpoint-level extrapolation：

[
W_{\mathrm{new}}
================

W_t+\gamma(W_t-W_{\mathrm{prev}})
]

其中：

[
\gamma=2^k
]

它利用的是一段训练轨迹的累计方向，而不是单个 noisy gradient。并且每次外推都经过小验证集筛选。论文图 7 的 ablation 也显示，learning rate 变大会带来震荡，而 EffOPD 通过 validation feedback 过滤过激 extrapolation，稳定性更好。

十、为什么 EffOPD 能加速

假设在某段训练区间内，OPD 的参数轨迹可以近似写成：

[
W_{t+s}
\approx
W_t + a_s d
]

其中 (d) 是已经锁定的主方向，(a_s) 是随训练增长的幅度。vanilla OPD 每一步只是小幅增加 (a_s)。EffOPD 用：

[
\Delta_n=W_{2^n}-W_{2^{n-1}}
]

估计这个方向，然后直接尝试：

[
W_{2^n}+2^k\Delta_n
]

这相当于提前完成未来若干 OPD step 才会积累到的 magnitude。如果方向真的稳定，那么这个外推点会落在接近后续训练轨迹的位置上，验证集分数不会下降，于是被接受。若方向不稳定或 magnitude 过冲，验证集会拒绝。

因此，EffOPD 的加速本质是：把“后续主要只是沿同一方向增大 norm”的训练过程压缩成少数几次参数外推。

这和论文提出的 Early Low-Rank Lock-in 正好对应：OPD 早期已经锁定 dominant subspace，后续训练主要是在该 subspace 内增长幅度，而不是大规模调整方向。

十一、局部二次模型下的理论解释

论文附录给了一个局部几何解释。设 token context 为：

[
c=(x,y_{<t})
]

student logits 为：

[
z_\theta(c)\in \mathbb{R}^{V}
]

teacher logits 为：

[
z^*(c)\in \mathbb{R}^{V}
]

base model 参数为 (\theta_0)，参数位移为：

[
\Delta \theta=\theta-\theta_0
]

在 (\theta_0) 附近一阶展开：

[
z_\theta(c)
===========

z_{\theta_0}(c)
+
J_c\Delta\theta
+
O(|\Delta\theta|^2)
]

其中：

[
J_c
===

\left.
\frac{\partial z_\theta(c)}{\partial \theta}
\right|_{\theta=\theta_0}
]

忽略高阶项：

[
z_\theta(c)
\approx
z_0(c)+J_c\Delta\theta
]

定义 teacher-base logit residual：

[
r_c
===

z^*(c)-z_0(c)
]

那么 student-teacher logit discrepancy 是：

[
z_\theta(c)-z^*(c)
\approx
J_c\Delta\theta-r_c
]

当 student 和 teacher 分布接近时，KL 可以用 logit 空间的二阶形式近似。定义 base distribution：

[
p_0(c)=\mathrm{softmax}(z_0(c))
]

Fisher matrix：

[
F_c
===

\mathrm{Diag}(p_0(c))-p_0(c)p_0(c)^\top
]

则单个 context 上的 KL 近似为：

[
D_{\mathrm{KL}}(p_\theta|p^*)
\approx
\frac{1}{2}
(J_c\Delta\theta-r_c)^\top
F_c
(J_c\Delta\theta-r_c)
]

对 on-policy contexts 取期望：

[
L_{\mathrm{OPD}}(\Delta\theta)
\approx
\frac{1}{2}
\mathbb{E}_c
\left[
(J_c\Delta\theta-r_c)^\top
F_c
(J_c\Delta\theta-r_c)
\right]
]

展开：

[
L_{\mathrm{OPD}}(\Delta\theta)
\approx
\frac{1}{2}
\Delta\theta^\top
\mathbb{E}_c[J_c^\top F_cJ_c]
\Delta\theta
------------

\Delta\theta^\top
\mathbb{E}_c[J_c^\top F_cr_c]
+
\mathrm{const}
]

定义：

[
A=\mathbb{E}_c[J_c^\top F_cJ_c]
]

[
b=\mathbb{E}_c[J_c^\top F_cr_c]
]

于是 OPD 的局部目标变成：

[
L_{\mathrm{OPD}}(\Delta\theta)
==============================

## \frac{1}{2}\Delta\theta^\top A\Delta\theta

b^\top \Delta\theta
+
\mathrm{const}
]

梯度为：

[
g(\Delta\theta)
===============

# \nabla_{\Delta\theta}L_{\mathrm{OPD}}

A\Delta\theta-b
]

梯度下降为：

[
\Delta\theta_{s+1}
==================

\Delta\theta_s-\eta(A\Delta\theta_s-b)
]

即：

[
\Delta\theta_{s+1}
==================

(I-\eta A)\Delta\theta_s+\eta b
]

从 base model 开始：

[
\Delta\theta_0=0
]

递推得到：

[
\Delta\theta_s
==============

\eta
\sum_{j=0}^{s-1}
(I-\eta A)^j b
]

如果步长满足收敛条件，可以写成闭式：

[
\Delta\theta_s
==============

\left[
I-(I-\eta A)^s
\right]
A^{-1}b
]

如果对 (A) 做特征分解：

[
A=U\Lambda U^\top
]

[
\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)
]

并把 (b) 投影到特征向量上：

[
b=U\beta
]

其中：

[
\beta_i=\langle b,u_i\rangle
]

则：

[
\Delta\theta_s
==============

\sum_{i:\lambda_i>0}
\frac{
1-(1-\eta\lambda_i)^s
}{
\lambda_i
}
\beta_i u_i
]

这个公式非常关键。它告诉我们，每个方向 (u_i) 的更新幅度由三件事决定：

[
\beta_i
]

即 teacher residual 在该方向上的驱动力；

[
\lambda_i
]

即该方向的局部曲率或敏感度；

[
1-(1-\eta\lambda_i)^s
]

即有限步训练中的增长因子。

如果 (b) 主要集中在前 (k) 个方向上，也就是：

[
|P_{U_k^\perp}b|
\le
\epsilon |b|,
\quad
\epsilon\ll 1
]

其中：

[
U_k=\mathrm{span}{u_1,\dots,u_k}
]

那么整个更新轨迹就会早期被限制在这个低维子空间里。如果再有谱间隙：

[
\lambda_k\gg \lambda_{k+1}
]

前 (k) 个方向会更早被激活和饱和。这就是 Early Low-Rank Lock-in 的理论来源。

十二、模块级冗余抑制的理论解释

把参数拆成 (M) 个模块：

[
\Delta\theta
============

(\Delta\theta_1,\Delta\theta_2,\dots,\Delta\theta_M)
]

Jacobian 也拆成：

[
J_c=
[J_{c,1},J_{c,2},\dots,J_{c,M}]
]

那么每个模块的 driving term 是：

[
b_m
===

\mathbb{E}*c[J*{c,m}^\top F_cr_c]
]

而 (A) 的 block 是：

[
A_{mn}
======

\mathbb{E}*c[J*{c,m}^\top F_cJ_{c,n}]
]

局部最优满足：

[
\sum_{n=1}^{M}
A_{mn}\Delta\theta_n^*
======================

b_m
]

如果模块间耦合不强，即 (A_{mn}) 在 (m\ne n) 时较小，那么近似有：

[
\Delta\theta_m^*
\approx
A_{mm}^{-1}b_m
]

于是，如果某个模块和 teacher residual 的耦合很弱：

[
b_m\approx 0
]

那么：

[
\Delta\theta_m^*\approx 0
]

这就是 Functional Redundancy Avoidance 的理论形式：如果某个模块对匹配 teacher 行为分布贡献小，OPD 会自然给它很小的更新。比如 embedding 或底层/顶层模块，可能 (b_m) 较小，所以更新被抑制。

十三、为什么 RL 不容易这样外推

RL 的 policy gradient 大致是：

[
g_{\mathrm{RL}}
===============

\mathbb{E}*{x,y\sim \pi*\theta}
\left[
\sum_{t=1}^{T}
A_t
\nabla_\theta
\log \pi_\theta(y_t|c_t)
\right]
]

其中 (A_t) 是 advantage。把 log-prob 梯度写成 logits Jacobian 形式：

[
\nabla_\theta \log \pi_\theta(y_t|c_t)
======================================

J_{c_t}^\top
(e_{y_t}-p_\theta(\cdot|c_t))
]

所以：

[
g_{\mathrm{RL}}
===============

\mathbb{E}
\left[
\sum_{t=1}^{T}
A_t
J_{c_t}^\top
(e_{y_t}-p_\theta(\cdot|c_t))
\right]
]

这个信号有几个问题：第一，(A_t) 在 RLVR 中经常来自最终对错，稀疏且高方差；第二，(e_{y_t}-p_\theta) 是随机采样 token 造成的 full-support noise；第三，credit assignment 更长，早期方向不稳定。因此 RL 的参数路径更容易分散、摇摆、在 tail directions 积累 norm。

OPD 的驱动力则是：

[
b=\mathbb{E}_c[J_c^\top F_cr_c]
]

其中 (r_c=z^*(c)-z_0(c)) 是 teacher-base residual。这个 residual 往往集中在关键 reasoning token、关键格式 token、答案 token 或中间推理结构上。因此它经过 (F_c) 和 (J_c^\top) 投影回参数空间后，也更可能集中在少数有效方向上。这个差异解释了为什么 OPD 的方向更适合外推，而 RL 不适合直接用同样的 extrapolation。

十四、EffOPD 的实际收益

论文在数学推理和代码任务上测试了 EffOPD，包括 Eurus-RL-Code、DeepMath-103K，以及 Codeforces、Taco、AIME24、AIME25、AIME26、MINERVA、GPQA 等评测。实验模型规模覆盖 1.5B、4B、14B、32B。结果是，EffOPD 通常在约 10 个训练 step 内开始收敛，而 vanilla OPD 往往需要 30 到 40 step，平均约 3× 加速，同时最终性能基本保持。

它相比 AlphaOPD、ExOPD 的核心差异是：不是固定 extrapolation，而是通过 (D_v) 自适应选择可接受的 extrapolation magnitude。固定外推容易过冲；EffOPD 用小验证集判断“当前方向还能不能继续走”。

十五、实现时最关键的工程点

第一，必须保存上一个指数 checkpoint：

[
W_{2^{n-1}}
]

否则无法计算：

[
\Delta_n=W_{2^n}-W_{2^{n-1}}
]

第二，候选参数不一定要真的保存 5 份完整模型。大模型上可以采用 in-place delta apply：

[
W \leftarrow W_{2^n}+2^k\Delta_n
]

评估完后 revert 回 (W_{2^n})，或者继续替换到下一个 candidate。FSDP/ZeRO 场景下要按 shard 操作，避免额外完整参数副本。

第三，验证集 (D_v) 虽然很小，但要和训练任务同分布。论文使用 50 个训练样本作为 lightweight validation set。它的作用不是精确估计最终 benchmark，而是检测“沿这个方向继续走是否还有效”。论文 ablation 认为 (D_v) 难度不是核心因素，关键是能提供方向有效性的反馈。

第四，接受条件最好不要过于激进。论文写的是：

[
V_{D_v}(W_{\mathrm{fn},k})\ge v_{\mathrm{acc}}
]

工程上如果验证噪声较大，可以改成带 margin 的版本：

[
V_{D_v}(W_{\mathrm{fn},k})
\ge
v_{\mathrm{acc}}-\epsilon
]

或者更保守：

[
V_{D_v}(W_{\mathrm{fn},k})
\ge
v_{\mathrm{acc}}+\epsilon
]

具体取决于验证指标的方差。

第五，外推后继续正常 OPD，而不是停止训练。EffOPD 是插入在 OPD 训练中的 checkpoint-level acceleration module。它不替代 OPD loss，也不引入新 trainable parameters。

十六、用一句话总结算法本质

vanilla OPD 的参数轨迹大致是：

[
W_0
\rightarrow
W_1
\rightarrow
W_2
\rightarrow
W_4
\rightarrow
W_8
\rightarrow
\dots
]

EffOPD 在 (W_{2^n}) 处估计方向：

[
\Delta_n=W_{2^n}-W_{2^{n-1}}
]

然后尝试跳到：

[
W_{2^n}+2\Delta_n,;
W_{2^n}+4\Delta_n,;
W_{2^n}+8\Delta_n,;
W_{2^n}+16\Delta_n,;
W_{2^n}+32\Delta_n
]

用小验证集选择最远但不掉分的位置。它之所以有效，是因为 OPD 早期已经完成了“方向选择”，后续主要是“幅度增长”。EffOPD 直接补偿这个幅度增长，从而减少训练步数。




这份研究报告深入探讨了同策略蒸馏（On-Policy Distillation, OPD）高效性的底层参数动力学机制，并基于此提出了核心加速算法 **EffOPD**。

以下是关于 OPD 加速涉及的核心算法、数学公式及其理论支撑的详尽解析：

### 一、 核心加速算法：EffOPD (Efficient OPD)

EffOPD 的核心思想是**定向外推 (Directional Extrapolation)**。研究发现，OPD 在训练极早期（前 10%-30%）就能锁定最终的优化方向。因此，EffOPD 并不等待模型通过成千上万步微小的梯度更新到达终点，而是尝试沿着已确定的方向跨出一大步。

#### 1. 触发机制与方向估计
算法在训练步数呈**指数级间隔**（即 $t = 2^n$ 步，如第 1, 2, 4, 8... 步）时触发外推搜索。
*   **局部更新方向公式 (1)**：
    $$\Delta_n = W_{2^n} - W_{2^{n-1}}$$
    这里 $\Delta_n$ 捕获了两个指数检查点之间的累计参数位移。由于 OPD 的更新方向具有极高的稳定性，这个 $\Delta_n$ 被用作后续更新方向的局部近似。

#### 2. 候选参数生成
算法沿着估计的方向 $\Delta_n$ 生成 5 个具有不同跨度的候选方案：
*   **线性外推公式 (2)**：
    $$\tilde{W}_{n,k} = W_{2^n} + 2^k \Delta_n, \quad k \in \{1, 2, 3, 4, 5\}$$
    其中 $2^k$ 是外推缩放系数（即步长为 2, 4, 8, 16, 32 倍位移），旨在通过大步跳跃直接逼近最终模型。

#### 3. 自适应验证与接受逻辑
为了防止过度外推导致模型崩溃，EffOPD 引入了**轻量级验证集 $D_v$**（仅随机采样 50 个样本，计算开销远低于常规训练步）。
*   **接受准则 (3)**：依次评估候选者 $\tilde{W}_{n,k}$。如果验证得分 $V_{D_v}$ 提升，则接受该跨越并更新参数：
    $$W_{acc} \leftarrow \tilde{W}_{n,k}, \quad v_{acc} \leftarrow V_{D_v}(\tilde{W}_{n,k})$$。
*   **搜索终止**：一旦某个候选者无法提升性能，搜索立即停止，算法将当前最优的 $W_{acc}$ 作为结果，并在后续步数中退回常规 OPD 训练。这种机制确保了加速过程的安全性。

---

### 二、 算法加速的底层特性：OPD 的“远见 (Foresight)”

EffOPD 之所以有效，是因为 OPD 具备两大核心特性，作者将其统称为“远见机制”。

#### 1. 模块化层面的“功能冗余规避” (Property 1)
OPD 能够早期识别出对推理任务贡献不大的区域，并抑制其更新。
*   **发现**：OPD 的参数更新非常精简，它会将更新集中在对推理至关重要的中间层 MLP 模块上，而抑制 Embedding 层或模型底层/顶层的更新。
*   **验证方法**：作者使用了**滑动窗口干预 (Sliding-window Intervention)** 框架（公式 10-12）来定位这些更新的功能贡献。通过将特定层的参数替换回基座模型并观察性能下降，证明了 OPD 的更新确实高度集中且无冗余。

#### 2. 几何层面的“早期低秩锁定” (Property 2)
OPD 的参数更新在空间上表现出极强的结构性约束。
*   **谱集中性**：通过对更新矩阵进行奇异值分解 (SVD) $\Delta W = U \Sigma V^\top$（公式 13），作者发现 OPD 的 Top-1% 子空间捕获了超过 90% 的更新能量。
*   **早期对齐**：OPD 更新的主成分子空间在训练早期就与最终训练完成的模型高度对齐（余弦相似度极高）。
*   **量化公式 (19)**：通过将早期检查点的更新向量规模放大（Rescaling）到最终水平：
    $$\Delta W_{scaled} = \Delta W_{early} + \Delta W_{early} \times \beta \cdot \frac{\|\Delta W_{final}\|_F - \|\Delta W_{early}\|_F}{\|\Delta W_{early}\|_F}$$。
    实验发现，仅凭 10% 的训练进度，通过适当放大幅值（$\beta \approx 1.0$），就能恢复最终模型约 80% 的性能。这证明了 OPD 早期已经找准了方向，只是幅值还不够。

---

### 三、 数学理论视角：OPD 动力学的局部线性化

在附录 F.5 中，作者通过对学生模型在基座模型附近进行泰勒展开，提供了理论上的解释。

#### 1. 局部目标函数
OPD 的损失函数可以近似为一个凸二次函数：
$$L_{OPD}(\Delta \theta) \approx \frac{1}{2} \Delta \theta^\top A \Delta \theta - b^\top \Delta \theta + \text{const}$$
*   **$A = \mathbb{E}_c[J_c^\top F_c J_c]$** 是曲率矩阵（Gram 矩阵），决定了参数更新的路径几何。
*   **$b = \mathbb{E}_c[J_c^\top F_c r_c]$** 是驱动项，其中 $r_c$ 是教师与基座模型之间的 Logit 残差。

#### 2. 更新路径的闭式解
在该近似下，梯度下降产生的参数更新路径 $\Delta \theta_s$（第 $s$ 步）具有解析形式：
$$\Delta \theta_s = \sum_{i:\lambda_i>0} \frac{1 - (1 - \eta \lambda_i)^s}{\lambda_i} \beta_i u_i$$
*   其中 $\lambda_i, u_i$ 是矩阵 $A$ 的特征对，$\beta_i$ 是驱动信号 $b$ 在特征方向上的投影。
*   **核心结论**：由于残差信号 $r_c$ 通常只在少数关键 Token（如推理步骤、最终答案）上非常显著，因此驱动项 $b$ 是天生**低秩**的。这意味着更新路径被限制在一个由早期确定的低维子空间内，这从理论上解释了为什么我们可以通过 EffOPD 进行线性外推加速。

### 总结
EffOPD 算法通过**公式 (1)** 计算局部位移确定“登山路径”，通过**公式 (2)** 进行跨越式的“大步跳跃”，并通过**自适应验证机制**确保不偏离路径。这种方法在保持性能的同时，将训练速度平均提升了 **3 倍**。