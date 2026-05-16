# EBFT loss parity contract (Slime ↔ OpenRLHF)

This document is the **implementation contract** for optional `EBFTPolicyLoss`-style actor training in Slime. It states OpenRLHF semantics exactly enough to code and test Megatron loss parity later.

**Scope:** loss definition, masks, reductions, coefficients, and tensor alignment. It does **not** implement the loss in Slime; that work stays gated until the strict runtime/dump/metrics gate passes (see `step5_loss_decision.md`).

**Authoritative OpenRLHF references:**

- `openrlhf/models/loss.py` — `EBFTPolicyLoss`
- `openrlhf/trainer/ray/ebft_actor.py` — actor train step: forward, loss composition, KL, coefficients
- `openrlhf/models/actor.py` — next-token log-prob layout and **continuation-only** temperature scaling
- `openrlhf/models/utils.py` — `compute_approx_kl`, `masked_mean`

---

## 1. Policy gradient term (ratio, no PPO clipping)

OpenRLHF `EBFTPolicyLoss.forward` computes:

```text
log_ratio = log_probs - log_probs.clone().detach()
```

For `policy_loss_type == "ppo"` (diff-dataset G1 default), `ratio = log_ratio.exp()`, which is **identically 1** in exact arithmetic (current policy vs detached copy of the same tensor).

- There is **no** `clip_ratio`, **no** twin surrogate, and **no** PPO-style clipped objective in `EBFTPolicyLoss`.
- Surrogate before masking: `surr_loss = -ratio * advantages`.

**Slime requirement:** Any EBFT loss mode must use the same `log_ratio` construction (or an explicitly equivalent formulation) so the RL term reduces to **`-advantages`** on masked positions, without importing PPO clipping from the default policy loss path.

**Note:** OpenRLHF also supports `policy_loss_type == "gspo"` (sequence-level ratio broadcast back to tokens). Slime parity must match ** whichever `policy_loss_type` the reference run uses** (diff-dataset G1 is expected to be `"ppo"` / ratio 1).

---

## 2. Masks

Let `action_mask` be bool (or 0/1), true for **generated / response** positions and false for **prompt** positions in the **same tensor layout as `log_probs`**.

Let `qa_masks` be bool, same trailing shape as `log_probs` after OpenRLHF’s `[:, 1:]` alignment (see §5).

OpenRLHF:

- If `qa_masking` is **False** (diff-dataset G1 default), it replaces `qa_masks` with `torch.ones_like(qa_masks)` inside `EBFTPolicyLoss` — i.e. QA masking is a no-op unless explicitly enabled.
- **RL mask:** `rl_mask = action_mask & qa_masks`
- **CE mask:** `ce_mask = (~action_mask) & qa_masks`

**Slime requirement:** Reproduce the same boolean logic. If QA masking is off, Slime must apply an all-ones QA mask of the correct shape (not omit the QA factor).

---

## 3. Aggregation

OpenRLHF:

```text
rl_loss  = masked_mean(surr_loss, rl_mask, dim=-1).mean()
ce_loss  = masked_mean(-log_probs, ce_mask, dim=-1).mean()
```

So: **per-sequence** masked mean along the token dimension, then **mean over batch** (unweighted by sequence length beyond what `masked_mean` does per row).

**`masked_mean` semantics** (`openrlhf/models/utils.py`): along `dim`, if the mask sum for a slice is zero, the implementation falls back to the **unmasked mean** for that slice. Strict parity tooling should use masks that avoid zero-sum rows in normal G1 training; if Slime ever hits the fallback, numbers may diverge from typical runs.

**Slime requirement:** Match `masked_mean(..., dim=-1).mean()` for both RL and CE terms (up to numerical tolerance), not a length-weighted global mean unless proven identical.

---

## 4. Coefficients (diff-dataset G1)

OpenRLHF combines:

```text
loss = actor_loss * rl_ctl + ce_loss * ce_ctl + kl_loss * kl_ctl
```

(`ebft_actor.py` training step.)

For **diff-dataset G1** reference configs:

- **`ce_loss_coef` (`ce_ctl`) default `0.03`** for the cross-entropy on prompt positions.
- `init_kl_coef` is commonly **`0.0`**; see §7.

**Slime requirement:** Expose a configurable CE coefficient defaulting to **`0.03`** for diff-dataset G1 parity runs; wire `rl_ctl`, `ce_ctl`, and `kl_ctl` the same way as the reference script.

---

## 5. Full-sequence next-token data contract

OpenRLHF’s critic/advantage path builds token-level advantages with shape **`[B, full_sequence_length - 1]`**: prompt positions **zero**, trailing generated region filled from expanded block rewards (`ebft_experience_maker.py`).

The actor loss consumes **next-token** `log_probs` aligned with that layout:

- In `ebft_actor.py`, `qa_masks` passed into `EBFTPolicyLoss` is **`qa_masks[:, 1:]`**, i.e. shifted to align with next-token prediction indices.
- **`action_mask`** passed to the loss matches the **response** mask in that same `[B, T]` window (same width as `experience.action_mask` in the training step).

**Contract shapes (conceptual):**

| Tensor | Shape | Notes |
|--------|--------|--------|
| `log_probs` | `[B, L-1]` | Next-token log prob of the label at each position in the shifted layout used by OpenRLHF actor |
| `advantages` | `[B, L-1]` | Prompt positions **0**; generated positions carry RLOO (or chosen estimator) signal |
| `action_mask` | `[B, L-1]` | True on generated tokens (for RL term) |
| `qa_masks` | `[B, L-1]` | Typically `qa_masks_full[:, 1:]` |

**Slime requirement:**

- Train on **full-sequence** next-token log probs for the prompt **and** response: prompt-only slices are required for the CE term. Response-only rollout logprob tensors are **insufficient** for OpenRLHF-parity EBFT loss.
- Align `qa_masks` with the same **`[:, 1:]`** convention as OpenRLHF when starting from full-sequence QA masks.

**OpenRLHF forward nuance:** In `ebft_actor.training_step`, the actor is called with `torch.ones_like(action_mask)` as the **forward** `action_mask` so log probs are **not** pre-zeroed by the actor’s internal `action_log_probs = log_probs * action_mask` path; the **real** `experience.action_mask` is passed only into `EBFTPolicyLoss`. Slime must preserve **non-zero prompt log probs** in the tensor that feeds the CE term.

---

## 6. Temperature (continuation-only)

In `openrlhf/models/actor.py`, after `prepare_logprobs`, if `self.temperature != 1.0`:

- OpenRLHF scales **continuation logits only**: `processed_logits[:, prompt_len - 1:].div_(self.temperature)`.
- Comment in code: `prepare_logprobs` has already dropped the last prompt token row, so the boundary index is **`prompt_len - 1`** in the processed tensor.
- `log_probs_from_logits` is then called with **`temperature=1.0`** (scaling applied in-place on logits, not inside log-softmax).

**Slime requirement for strict parity:** If the reference run uses `temperature != 1`, Megatron/actor log-prob computation must apply **the same continuation-only** scaling rule and the same `prompt_len` boundary semantics. If Slime keeps `temperature == 1` while the reference does not, loss parity is not expected.

**Documented alternative:** A Slime-only temperature policy is acceptable for *non-parity* experiments if explicitly flagged; it is out of scope for “strict OpenRLHF EBFT loss parity.”

---

## 7. KL semantics

When `use_kl_loss` is true (`ebft_actor.py`):

- If **`init_kl_coef > 0`**, OpenRLHF computes `kl = compute_approx_kl(action_log_probs, base_action_log_probs, kl_estimator=...)`.
- If **`init_kl_coef == 0`**, it sets `kl = torch.zeros_like(action_log_probs)` (no approximate KL computed).

`compute_approx_kl` (`openrlhf/models/utils.py`):

- For **`k1`**: `log_ratio = log_probs.float() - log_probs_base.float()`.
- For **`k2` / `k3`**: non-negative approximations per Schulman blog definitions in code.
- **All estimators:** `log_ratio = log_ratio.clamp(min=-10, max=10)` before return.

Reduction:

```text
kl_loss = masked_mean(kl, experience.action_mask)
```

with default **`dim=None`**: **global** mean over all batch/token positions where `experience.action_mask` is true (i.e. generated-token-only, length-weighted across the batch as a single scalar).

Total contribution: `kl_loss * kl_ctl` (with `kl_ctl` from the trainer).

**Slime requirement:** Match estimator name, **[-10, 10] clamp**, `float()` casting behavior in `compute_approx_kl`, and **global masked mean over generated tokens** using the same mask tensor semantics. When **`init_kl_coef == 0`**, match the **zeros tensor** short-circuit (no KL gradient through approximate KL).

---

## 8. Advantage normalization (strict parity)

`EBFTPolicyLoss` does **not** normalize advantages. OpenRLHF G1 experience building uses RLOO (or other estimators) and copies shaped rewards into `advantages` without an extra “normalize across batch” step inside the loss.

Slime may support `--normalize-advantages` (or equivalent) for PPO paths.

**Strict parity rule:** For OpenRLHF EBFT loss parity runs, **disable** Slime advantage normalization unless a proof or controlled experiment shows bitwise/numerical equivalence for the G1 tensors.

---

## 9. Gated implementation reminder

Do **not** implement this loss in production training until:

- strict dump/metrics gate passes (DP-safe dumps, ordering, PP key contract, thresholded parity reports — see parent plan and `step5_loss_decision.md`);
- full-sequence log-prob and mask plumbing is verified against OpenRLHF layouts.

---

## 10. Open items requiring user confirmation

- Final numeric thresholds for strict embedding/reward/advantage parity (engineering defaults exist in the parent plan; product owners may tighten/relax).
- **CLI shape** for the gated mode: e.g. `--g1-use-ebft-loss` vs `loss_type=g1_ebft_policy_loss`.
- **Phase-1 Slime implementation (documented default):** target **RL + CE only** with `init_kl_coef=0` / no entropy aux unless the user later requests **KL** and/or **entropy** parity; then implement and test §7 and any entropy path explicitly.
- Whether `openrlhf_exact` / diagnostic mask application may remain a fallback-only path or must eventually land on the native TE/thd fast attention.
