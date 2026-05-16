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

## Current Mask Status

The dense EBFT attention mask is constructed, dumped, and compared, but it is not yet applied to Megatron Transformer Engine `thd` attention.

The runtime dump records:

```text
g1_attention_mask_status = openrlhf_dense_mask_built_not_applied_to_megatron_te_thd
g1_attention_mask_applied = False
```

This is intentional. Megatron/TE `thd` attention is optimized around packed causal/padding semantics, and this phase should not silently pass a 4D additive mask unless the backend is proven to consume it correctly.

## What To Look For In The Report

After rerunning dumps:

- `Position ids match` should be `True`.
- `Attention mask tensors match` should be `True` if both dumps include the mask.
- `Megatron attention mask applied` will still be `False`.
- Any remaining generated-block embedding gap should be attributed primarily to attention-mask semantics, not position ids.

## EBFT Loss Gate

Do not implement EBFTPolicyLoss yet. The current gate remains:

- embedding runtime parity must improve after position/mask work, or
- the remaining embedding gap must be explicitly accepted before moving to loss parity.

Until then, do not call this path `slime_g1_exact`.
