# OpenRLHF KL parity contract for G1 EBFT

This is the Phase 3 contract for OpenRLHF-style KL parity in the gated G1 EBFT loss work. It documents the target semantics and CPU fixture coverage only. It does **not** authorize enabling `--g1-use-ebft-loss` together with Slime training KL (`--use-kl-loss`), and it does not change the trainer path.

## Scope

In scope:

- OpenRLHF EBFT actor-side KL scalar semantics.
- `base_action_log_probs` provenance and tensor alignment.
- `compute_approx_kl` estimator behavior and clamp.
- `kl_coef == 0` short-circuit behavior.
- Mask/reduction/temperature semantics needed for parity.
- Why Slime's current PPO/GRPO KL path is not a drop-in replacement.

Out of scope:

- Wiring KL into Slime's gated EBFT training path.
- Relaxing the existing `--g1-use-ebft-loss` and `--use-kl-loss` incompatibility.
- Entropy parity.

## Authoritative OpenRLHF references

- `openrlhf/trainer/ray/ebft_actor.py`
  - Extracts `base_action_log_probs = experience.base_action_log_probs`.
  - Recomputes current `action_log_probs` with the EBFT actor.
  - If `args.use_kl_loss` and `args.init_kl_coef > 0`, computes `compute_approx_kl(action_log_probs, base_action_log_probs, kl_estimator=args.kl_estimator)`.
  - If `args.init_kl_coef <= 0`, sets `kl = torch.zeros_like(action_log_probs, ...)`.
  - Reduces with `masked_mean(kl, experience.action_mask)`.
  - Adds `kl_loss * kl_ctl` to the actor loss.
- `openrlhf/trainer/ppo_utils/ebft_experience_maker.py`
  - Populates `base_action_log_probs` by calling the initial/reference actor group `forward_strided_blocks(...)` over the same full sequences, action masks, prompt length, generation step, block/stride/context metadata, and doc IDs.
  - Stores `None` when `args.use_kl_loss` is false.
- `openrlhf/models/utils.py`
  - Defines `compute_approx_kl`.
  - Defines `masked_mean`.
- `openrlhf/models/actor.py`
  - Defines the full-sequence/strided next-token log-prob layout and continuation-only temperature scaling.

## Tensor Contract

The KL tensors are in the same next-token layout as the EBFT actor loss:

| Tensor | Shape | Semantics |
| --- | --- | --- |
| `action_log_probs` | `[B, L - 1]` or packed-equivalent rows | Current actor log probs from the EBFT actor forward, after `prepare_logprobs`/`prepare_labels` alignment. |
| `base_action_log_probs` | Same as `action_log_probs` | Initial/reference actor log probs from `initial_model_group.forward_strided_blocks(...)`, produced with the same EBFT sequence metadata. |
| `experience.action_mask` | Same trailing shape | True on generated tokens. The KL scalar is computed over this mask only. |

`base_action_log_probs` is not the rollout old-policy log prob and is not the prompt CE tensor. It is the reference/initial actor log prob in the same EBFT full-sequence next-token layout as the current actor's `action_log_probs`.

## Estimator Contract

OpenRLHF's EBFT KL uses `compute_approx_kl(log_probs, log_probs_base, kl_estimator=...)`:

```text
log_ratio = log_probs.float() - log_probs_base.float()

k1: kl = log_ratio
k2: kl = log_ratio ** 2 / 2
k3: kl = exp(-log_ratio) - 1 + log_ratio

kl = clamp(kl, min=-10, max=10)
```

Important parity details:

- Inputs are cast to `float()` before the estimator math.
- OpenRLHF supports `k1`, `k2`, and `k3` in this checkout.
- The clamp is applied after the estimator, for all three estimator choices.
- This differs from Slime's `compute_approx_kl`, which names the selector `kl_loss_type`, supports `low_var_kl`, has optional `importance_ratio`, and clamps only `low_var_kl`.

## Coef And Short-Circuit

OpenRLHF has two closely related coefficient concepts in this path:

- `init_kl_coef`: controls whether the reference model group exists and whether approximate KL is computed at all.
- `kl_ctl`: the controller value passed to actor training and multiplied into the scalar contribution.

The parity rule is:

```text
if use_kl_loss:
    if init_kl_coef > 0:
        kl = compute_approx_kl(action_log_probs, base_action_log_probs, kl_estimator)
    else:
        kl = zeros_like(action_log_probs)
    kl_loss = masked_mean(kl, experience.action_mask)
else:
    kl_loss = 0

loss += kl_loss * kl_ctl
```

For this task and CPU fixture, `kl_coef == 0` means the OpenRLHF short-circuit: return a zeros-like KL tensor and do not read `base_action_log_probs` through approximate KL. A later Slime implementation should preserve both pieces: no reference-logprob dependency when the coefficient is zero, and contribution scaling only after the masked scalar is formed.

## Mask And Reduction

OpenRLHF calls `masked_mean(kl, experience.action_mask)` with `dim=None`.

That is a global masked mean over generated tokens in the batch:

```text
kl_scalar = (kl * action_mask).sum() / action_mask.sum()
```

This is **not** `masked_mean(..., dim=-1).mean()` and is not a per-sequence average unless all rows have the same generated-token count. The RL/CE EBFT terms use per-row token means followed by batch mean; the KL scalar does not.

The mask is `experience.action_mask` only. QA masks are part of EBFT RL/CE semantics, but OpenRLHF EBFT KL reduction uses the generated-token action mask.

## Temperature Semantics

OpenRLHF computes both current `action_log_probs` and `base_action_log_probs` through the actor log-prob path. In `openrlhf/models/actor.py`, when `temperature != 1.0`, temperature scaling is applied only to continuation logits:

```text
processed_logits[:, prompt_len - 1:].div_(temperature)
log_probs_from_logits(..., temperature=1.0)
```

`prompt_len - 1` is used because `prepare_logprobs` has already dropped the final prompt-token row. Strict parity requires Slime to apply the same continuation-only scaling to both current and base/reference log-prob computation if the OpenRLHF reference run uses non-1 temperature.

## Why Slime PPO KL Is Not A Drop-In

Slime's default PPO/GRPO KL path is useful but not directly equivalent:

- It is wired to Slime's default policy-loss path, not the gated EBFT RL+CE actor loss layout.
- It consumes `ref_log_probs` from Slime batch plumbing, while OpenRLHF EBFT consumes `base_action_log_probs` from the initial/reference actor group in EBFT strided full-sequence layout.
- It reduces via Slime's `sum_of_sample_mean` machinery; OpenRLHF EBFT KL uses one global generated-token masked mean.
- Slime's estimator API and clamp behavior differ from this OpenRLHF checkout.
- Slime's default path also interacts with PPO clipping, entropy, TIS/OPSM, and mismatch metrics that are intentionally out of Phase 3 EBFT KL parity scope.

Therefore, do not enable `--g1-use-ebft-loss` with `--use-kl-loss` until a dedicated OpenRLHF EBFT KL path is implemented and validated against this contract.

## CPU Fixture Coverage

`tests/test_g1_ebft_loss.py` contains the CPU-only fixture/helper for this contract:

- It constructs `action_log_probs`, `base_action_log_probs`, and an unequal-row-count `action_mask`.
- It validates `k1`, `k2`, and `k3` estimator math with the all-estimator `[-10, 10]` clamp.
- It validates global masked reduction and coefficient scaling.
- It validates the zero-coefficient short-circuit by passing NaN base log probs and expecting a finite zero scalar/contribution.

This fixture documents semantics only; it does not call or relax the Slime training gate.

## Follow-Ups

- Implement dedicated OpenRLHF EBFT KL training support only after the full-sequence base/reference log-prob plumbing is present and parity dumps prove alignment.
- Keep the existing `--g1-use-ebft-loss` plus `--use-kl-loss` rejection until then.
- A small cleanup is available: `slime/backends/megatron_utils/loss.py` currently reports a `--use-kl-loss` message in the `args.use_opsm` guard, while `slime/utils/arguments.py` reports `--use-opsm`. This task records it as a follow-up rather than changing the training path.
