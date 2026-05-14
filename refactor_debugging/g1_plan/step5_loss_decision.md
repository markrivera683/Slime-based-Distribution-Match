# Step 5: EBFT Loss Decision

## Status

Decision is deferred until the Step 4 G1 smoke passes. Current slime G1 path is not full OpenRLHF loss exactness.

## Current Slime Behavior

The current slime implementation does:

```text
g1_token_advantages
  -> compute_advantages_and_returns(..., advantage_estimator="g1")
  -> policy_loss_function
  -> PPO-style ratio / clipping path
```

This is enough to validate:

- embedding metadata is produced
- group RM computes G1 token advantages
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

Do not implement EBFT loss before Step 4 smoke passes. The smoke must first prove:

1. slow HF/OpenRLHF embeddings are produced
2. `g1_token_advantages` length is 376
3. one training step is finite
4. reward / advantage distributions are logged

Only then compare:

- slime PPO-style + G1 advantage
- OpenRLHF G1 reference
- optional slime EBFT-style branch

## If Exact Loss Is Required

Add a separate loss mode rather than overloading PPO silently.

Required design:

- a flag such as `--g1-use-ebft-loss` or a dedicated `loss_type`
- ratio fixed to 1 for RL term
- no PPO clipping
- no TIS / OPSM in this branch
- prompt CE term with coefficient `--g1-ce-loss-coef`, default `0.03`
- QA mask support, including the OpenRLHF shift semantics
- aggregation compatible with `masked_mean(..., dim=-1).mean()`

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
