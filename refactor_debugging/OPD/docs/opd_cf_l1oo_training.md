# OPD-CF-L1OO Training Design

## Goal

Make OPD the main training signal, then use the existing CF-L1OO machinery for
group credit assignment over on-policy student rollouts.

This is intentionally not EffOPD. EffOPD should later accelerate the parameter
trajectory produced by this training objective.

## Method Definition

For each prompt, sample `N` student rollouts:

```text
y_1, ..., y_N ~ pi_student(. | prompt)
```

The teacher evaluates those same rollouts and returns token logprobs:

```text
log p_T(y_j,t | prompt, y_j,<t)
```

The student rollouts are embedded by the frozen G1/critic feature path:

```text
z_j,k = phi(y_j block k)
```

Teacher logprobs are reduced into one score per rollout:

```text
s_j = mean_t log p_T(y_j,t | prompt, y_j,<t)
```

`mean` is the default because raw sequence sums strongly favor shorter answers.
`sum` remains available for ablations.

The teacher preference distribution over the group is:

```text
q_j = softmax(s_j / tau)
```

## CF-L1OO Credit Assignment

In `--opd-credit-assignment cf_l1oo`, the student distribution is uniform over
the student rollout feature support:

```text
mu = (1 / N) sum_j delta(z_j,k)
```

The teacher target is on the same support, but teacher-weighted:

```text
nu = sum_j q_j delta(z_j,k)
```

For each rollout, compute leave-one-out credit:

```text
reward_j,k = D_CF(mu_without_j, nu) - D_CF(mu, nu)
```

If removing a rollout makes the teacher-weighted distribution match worse, that
rollout receives positive credit. If removing it improves the match, it receives
negative credit.

This keeps the OPD property that the teacher only scores on-policy samples. It
does not use teacher-generated completions as target samples.

## EBFT Credit Baseline

In `--opd-credit-assignment ebft`, the same OPD teacher scores define a
teacher-weighted on-policy feature centroid:

```text
c_k = sum_j q_j z_j,k
```

Each rollout receives pointwise EBFT-style credit:

```text
reward_j,k = cosine(z_j,k, c_k)
```

Then RLOO shaping is applied across the prompt group. This baseline isolates the
effect of CF distribution credit assignment from a simpler embedding-based
pointwise credit assignment.

Important comparison:

```text
cf_l1oo: distribution discrepancy + leave-one-out
ebft: pointwise feature similarity + RLOO shaping
```

Both use the same teacher logprobs, same student rollouts, same embeddings, and
same OPD KL application policy.

## Training Integration

Use:

```bash
--distribution-reward-type cf_l1oo
--cf-target-mode opd_onpolicy
--use-opd
--opd-credit-assignment cf_l1oo
```

Baseline:

```bash
--distribution-reward-type cf_l1oo
--cf-target-mode opd_onpolicy
--use-opd
--opd-credit-assignment ebft
```

Required surrounding settings:

```bash
--advantage-estimator g1
--g1-embedding-source megatron_ref
--g1-reward-location trainer
--use-whitening
--n-samples-per-prompt > 1
```

Teacher-completion settings are not required for `opd_onpolicy`:

```text
teacher_backend
teacher_api_base
teacher_model_name
cf_teacher_n_samples
g2_teacher_completions
```

## OPD KL Policy

OPD reverse-KL terms are always computed for logging:

```text
opd_reverse_kl = student_logprob - teacher_logprob
```

Default `--opd-kl-application auto` resolves as:

```text
cf_target_mode=opd_onpolicy -> cf_l1oo
otherwise                    -> token_penalty
```

So for OPD-CF-L1OO, the tokenwise KL penalty is not subtracted again by
default. This avoids double-counting the teacher signal, because teacher
logprobs already drive the group credit assignment.

Explicit overrides:

```bash
--opd-kl-application token_penalty
--opd-kl-application cf_l1oo
--opd-kl-application both
```

## Acceptance Checklist

Unit-level:

- `compute_opd_cf_l1oo_rewards` returns shape `[B, G, N, K]`.
- high teacher-score rollout receives higher CF-L1OO credit in asymmetric
  fixtures.
- uniform teacher scores produce symmetric CF-L1OO rewards.
- `opd_onpolicy` requires `teacher_log_probs`.
- `opd_onpolicy` does not require `g2_teacher_gen_embeddings`.
- `ebft` baseline returns finite token advantages and scalar rewards.
- `opd_kl_application=cf_l1oo` logs reverse-KL but does not mutate advantages.
- `opd_kl_application=both` logs reverse-KL and applies token penalty.

Integration-level:

- rollout train data contains `teacher_log_probs`.
- G2 remote teacher completion path is not triggered for `opd_onpolicy`.
- actor/ref side computes `g1_token_advantages`.
- actor/critic sync preserves `g1_token_advantages` and scalar `rewards`.
- logs include `opd_reverse_kl`, `rewards`, and `advantages`.

Experiment-level:

- compare ordinary tokenwise OPD vs `opd_onpolicy + cf_l1oo`.
- compare `opd_onpolicy + cf_l1oo` vs `opd_onpolicy + ebft`.
- keep sampling, teacher endpoint, critic/ref checkpoint, and OPD KL logging
  identical across the comparison.

## Known Risks

- Mean teacher logprob may underweight long but good reasoning traces; sum
  logprob may over-penalize length. Keep both as ablations.
- If teacher scores within a group are nearly uniform, CF-L1OO has little
  preference signal.
- If the frozen feature space is weak, both CF-L1OO and EBFT credit assignment
  can become noisy even when teacher logprobs are meaningful.
- This objective does not use GT for training in v1. GT should be used for eval
  and logging only unless a later plan explicitly adds GT anchoring.
