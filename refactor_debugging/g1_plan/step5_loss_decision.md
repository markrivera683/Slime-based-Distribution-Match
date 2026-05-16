# Step 5: EBFT Loss Decision

## Status

Step 4 Megatron/ref trainer-side G1 smoke has passed. Current slime G1 path is still not full OpenRLHF loss exactness because the actor loss remains PPO-style.

Decision for the current branch: **do not implement EBFTPolicyLoss yet**. Keep the passed path named as a G1 reward/advantage smoke until runtime parity and loss diagnostics justify a separate exact-loss branch.

**Loss parity spec:** Exact OpenRLHF `EBFTPolicyLoss` semantics, mask layout, temperature, KL, and aggregation are written in [`ebft_loss_parity_contract.md`](ebft_loss_parity_contract.md). Use that file as the single contract when implementation is ungated.

## Strict gate before implementation (parent plan)

The **strict gate comes first**: do not land Megatron EBFT actor loss code until Phase 1 parity/diagnostics close, including:

- **Dump safety:** no shared-path DP dump races; rank-scoped or rank-0-only writes; no unsafe unlink/load/append/save; writer metadata; consistent counts across dumped tensors.
- **Ordering:** grouped `n_samples_per_prompt` ordering is provably preserved before RLOO, or dynamic batching is disabled / order restored for trainer-side Megatron G1.
- **Pipeline parallel:** G1 keys and loss inputs are defined for `PP>1` (last-stage-only vs all stages documented and enforced); loss must not rely on keys missing on non-last stages.
- **Parity metrics:** compact full-group reports with **max_abs / mean_abs / relative L2** and explicit thresholds — not cosine-only evidence — for hidden states, gen/GT embeddings, rewards, and token advantages.

Only after that gate: implement the gated EBFT loss mode per [`ebft_loss_parity_contract.md`](ebft_loss_parity_contract.md) (ratio=1, no clip, full-sequence log probs, masks, CE coef 0.03, KL and temperature rules, no advantage normalization unless proven equivalent).

## Current Slime Behavior

The current slime implementation does:

```text
g1_token_advantages
  -> compute_advantages_and_returns(..., advantage_estimator="g1")
  -> policy_loss_function
  -> PPO-style ratio / clipping path
```

This is enough to validate:

- fixed-length rollout is produced
- Megatron `ref` hidden capture produces G1 embeddings on the trainer side
- trainer-side G1 computes token advantages
- train data carries those advantages
- Megatron training can consume them

It is not enough to claim OpenRLHF G1 exact training dynamics.

## OpenRLHF G1 Behavior

OpenRLHF diff-dataset G1 uses `EBFTPolicyLoss`:

```text
log_ratio = log_probs - detach(log_probs)
ratio = exp(log_ratio) = 1
rl_loss = masked_mean(-ratio * advantages, action_mask & qa_masks)
ce_loss = masked_mean(-log_probs, ~action_mask & qa_masks)
total = rl_ctl * rl_loss + ce_ctl * ce_loss + kl_ctl * kl_loss
```

For diff-dataset G1:

- `ce_loss_coef = 0.03`
- `qa_masking = False` by default
- `init_kl_coef = 0.0`
- reward/advantage uses RLOO and pointwise embedding rewards

## Exactness Gap

| Area | Current slime | OpenRLHF G1 |
| --- | --- | --- |
| policy gradient | PPO ratio + clip | ratio is effectively 1, no PPO clip |
| prompt CE | absent in G1 path | CE on `~action_mask & qa_masks` |
| masks | response `loss_masks` | `action_mask & qa_masks` and prompt CE mask |
| aggregation | slime per-token/per-sample reducer | `masked_mean(..., dim=-1).mean()` |
| advantage normalization | possible if `--normalize-advantages` is passed | not part of EBFTPolicyLoss itself |

## Decision Gate

Do not implement EBFT loss until the **strict gate** above passes and the post-smoke diagnostics justify the extra prompt-logprob/mask work. The current Megatron exact-parity branch adds a gated `--g1-megatron-ref-forward-mode openrlhf_exact` diagnostic mode for OpenRLHF position/RoPE parity and dense EBFT mask parity. Dense mask application is proven through the slow ref-only torch THD fallback, not through the standard Transformer Engine `thd` fast path.

The passed smokes have proven:

1. Megatron/ref trainer-side embeddings are produced
2. `g1_token_advantages` length is 376
3. one training step is finite
4. reward / advantage distributions are logged
5. mask-applied embedding runtime parity is close to OpenRLHF on the one-sample reference:

```text
Full hidden cosine mean ~= 0.99935502
Gen block embedding cosine mean ~= 0.99980551
GT block embedding cosine mean ~= 0.99889201
```
6. full-group reward and advantage parity is available and aligned closely enough to justify moving the next effort to loss exactness:

```text
Full-group reward max_abs ~= 0.00221942
Full-group reward mean_abs ~= 0.00108439
Full-group token advantage sample_cos_mean ~= 0.99966836
Full-group token advantage sample_cos_min ~= 0.99948490
Full-group token advantage max_abs ~= 0.02138489
Full-group token advantage mean_abs ~= 0.00396629
```

One reporting caveat remains: the first full-group hidden-state report included THD padding in the Megatron hidden dump, which polluted the full-hidden cosine while leaving block embeddings, rewards, and advantages valid. The compact full-group Megatron dump removes that padding and should be used for the final full-hidden report refresh.

7. gated EBFT RL+CE loss smoke now reaches a successful Ray job with `G1_USE_EBFT_LOSS=true` and `G1_APPLY_DENSE_ATTENTION_MASK=true`:

```text
Job 'raysubmit_sBwPYa6Q2PycQDyD' succeeded
```

This validates the first-phase EBFT integration path. KL/entropy parity remains out of scope for this phase.

Next compare in a separate loss-focused plan:

- slime PPO-style + G1 advantage
- OpenRLHF G1 reference
- optional slime EBFT-style branch

The embedding/reward/advantage gate is now effectively open for a separate EBFT loss parity plan. Do not mix that implementation into this diagnostic parity branch without a dedicated loss plan and tests.

Useful diagnostics from the passed smoke include `train/pg_clipfrac`, `train/train_rollout_logprob_abs_diff`, `train/loss`, `train/pg_loss`, and `rollout/g1_token_advantages`.

The passed smoke logged:

```text
train/loss ~= 5.59e-09
train/pg_loss ~= 5.59e-09
train/pg_clipfrac = 0.0
train/train_rollout_logprob_abs_diff ~= 0.1463
```

These values are sufficient to show the current PPO-style path is numerically finite, but not sufficient to claim OpenRLHF actor-loss exactness.

## If Exact Loss Is Required

Add a separate loss mode rather than overloading PPO silently. Full semantic and shape requirements are in [`ebft_loss_parity_contract.md`](ebft_loss_parity_contract.md).

Required design (summary):

- a flag such as `--g1-use-ebft-loss` or a dedicated `loss_type`
- ratio fixed to 1 for RL term (`log_probs - detach(log_probs)`); no PPO clipping
- no TIS / OPSM in this branch
- prompt CE term with coefficient `--g1-ce-loss-coef`, default `0.03`
- QA mask support, including `qa_masks[:, 1:]` alignment
- aggregation compatible with `masked_mean(..., dim=-1).mean()`; match OpenRLHF KL reduction and approximate-KL clamp if KL is enabled
- continuation-only temperature scaling if reference `temperature != 1`
- disable Slime advantage normalization for strict parity unless proven equivalent

Required train data additions:

- enough prompt/full-sequence mask information to compute CE
- `qa_masks` aligned to logits positions if `qa_masking` is enabled
- clear distinction between response-only G1 advantages and prompt CE tokens

## Naming Rule

Until EBFT loss is implemented and validated:

- call the path `slime_g1_reward_smoke` or `slime_g1_advantage`
- do not call experiment results `full_g1_exact`

After EBFT loss parity is implemented and validated:

- `slime_g1_exact` can refer to embedding + reward/RLOO + advantage + EBFT loss parity

## Minimum Tests For Loss Exactness

Before enabling exact loss in real experiments:

1. unit test ratio=1 behavior against OpenRLHF `EBFTPolicyLoss`
2. unit test CE mask on prompt positions
3. unit test response RL mask
4. unit test aggregation on equal-length and variable-length masks
5. smoke test one Megatron train step with finite loss
