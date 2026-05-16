# Megatron G1 Exact Parity Status

## What Is Implemented

This branch now has a gated Megatron/ref diagnostic mode:

```text
--g1-megatron-ref-forward-mode openrlhf_exact
```

The default remains:

```text
--g1-megatron-ref-forward-mode standard
```

`openrlhf_exact` currently does two staged things:

- builds OpenRLHF G1 `position_ids` and applies them to Megatron RoPE during the ref hidden-state capture;
- builds the OpenRLHF EBFT dense strided attention mask and dumps it for parity checks.
- optionally applies the dense mask through a slow ref-only torch THD attention fallback with:

```text
--g1-megatron-ref-apply-dense-attention-mask
```

## Current Mask Status

The dense EBFT attention mask is constructed, dumped, and compared. It is not applied to Megatron Transformer Engine `thd` attention directly.

Without the fallback, the runtime dump records:

```text
g1_attention_mask_status = openrlhf_dense_mask_built_not_applied_to_megatron_te_thd
g1_attention_mask_applied = False
```

With the fallback, the runtime dump should record:

```text
g1_attention_mask_status = openrlhf_dense_mask_applied_via_torch_thd_fallback
g1_attention_mask_applied = True
```

This is intentional. Megatron/TE `thd` attention is optimized around packed causal/padding semantics, so the dense mask path bypasses TE core attention only for the G1 ref hidden-state diagnostic pass. Standard training remains on the normal fast path.

## Latest Runtime Parity Result

The mask-applied fallback run matched OpenRLHF Critic on the core embedding contracts:

```text
Token IDs match: True
QA masks match: True
Position ids match: True
Attention mask tensors match: True
Megatron attention mask applied: True
Full hidden cosine mean: 0.99935502
Gen block embedding cosine mean: 0.99980551
GT block embedding cosine mean: 0.99889201
```

Interpretation: position/RoPE parity and dense EBFT mask application are sufficient to bring Megatron/ref G1 embeddings close to OpenRLHF runtime parity. The remaining caveat is that the exact mask application currently uses the slow diagnostic torch fallback, not the standard TE/thd fast path.

## Latest Full-Group Reward / Advantage Result

The full-group mask-applied run now records all `n_samples_per_prompt=4` samples in the Megatron runtime dump and aligns closely with the OpenRLHF Critic reward/advantage reference:

```text
Token IDs match: True
QA masks match: True
Position ids match: True
Attention mask semantics match: True
Megatron attention mask applied: True
Reward max_abs: 0.00221942
Reward mean_abs: 0.00108439
Token advantage sample_cos_mean: 0.99966836
Token advantage sample_cos_min: 0.99948490
Token advantage max_abs: 0.02138489
Token advantage mean_abs: 0.00396629
```

The earlier full-group hidden-state cosine was polluted by THD padding in the dumped Megatron hidden stream. Use the compact full-group bundle/dump for any final full-hidden report refresh:

```text
refactor_debugging/g1_plan/openrlhf_parity_bundle_compact_full_group.tar.gz
artifacts/g1_runtime_megatron_ref_mask_applied_full_group_compact_dump.pt
```

## What To Look For In The Report

After rerunning dumps:

- `Position ids match` should be `True`.
- `Attention mask tensors match` should be `True` if both dumps include the mask.
- `Megatron attention mask applied` should be `True` only when `--g1-megatron-ref-apply-dense-attention-mask` was enabled.
- If generated-block embedding cosine still does not improve after the fallback is enabled, the remaining gap is no longer explained by position ids or dense mask construction/application alone.

## EBFT Loss Gate

Do not implement EBFTPolicyLoss in this branch. The current gate status is:

- embedding parity is proven through the gated slow fallback;
- full-group reward/advantage parity is close enough to proceed; and
- EBFTPolicyLoss should be implemented in a separate loss-focused branch or plan.

Until then, do not call this path `slime_g1_exact`.
