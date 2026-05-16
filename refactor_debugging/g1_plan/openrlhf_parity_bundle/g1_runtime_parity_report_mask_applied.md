# G1 Runtime Parity Report

## Inputs

- Megatron dump: `artifacts/g1_runtime_megatron_ref_mask_applied_dump.pt`
- OpenRLHF dump: `artifacts/g1_runtime_openrlhf_critic_mask_applied_dump.pt`

## Contract Checks

- Token IDs match: `True`
- QA masks match: `True`
- Position ids match: `True`
- Attention mask tensors match: `True`
- Megatron ref forward mode: `openrlhf_exact`
- Megatron attention mask status: `openrlhf_dense_mask_applied_via_torch_thd_fallback`
- Megatron attention mask applied: `True`
- Megatron hidden shape: `(1, 760, 2560)`
- OpenRLHF hidden shape: `(1, 760, 2560)`
- Megatron gen embedding shape: `(1, 47, 2560)`
- OpenRLHF gen embedding shape: `(1, 47, 2560)`

## Mask Summary

- Megatron mask summary: `{'shape': (1, 1, 760, 760), 'allowed_count': 147804, 'blocked_count': 429796, 'finite_count': 577600}`
- OpenRLHF mask summary: `{'shape': (1, 1, 760, 760), 'allowed_count': 147804, 'blocked_count': 429796, 'finite_count': 577600}`

## Cosine Similarity

### Full Hidden States

```text
mean   = 0.99935502
median = 0.99963856
min    = 0.99294364
max    = 0.99995196
```

### Gen Block Embeddings

```text
mean   = 0.99980551
median = 0.99986261
min    = 0.99824637
max    = 0.99990648
```

### GT Block Embeddings

```text
mean   = 0.99889201
median = 0.99899292
min    = 0.99740916
max    = 0.99990654
```

### Rewards

```text
not available
not available
```

### Token Advantages

```text
not available
not available
not available
not available
```

## Interpretation

This report compares the current Slime Megatron/ref fast path against OpenRLHF Critic on identical token IDs and QA masks.

Tight tensor equality is expected only after both staged contracts pass: OpenRLHF position ids must drive Megatron RoPE, and EBFT dense strided mask semantics must reach the Megatron attention backend. If the report says the Megatron dense mask was built but not applied, remaining generated-block gaps should be treated as attention-mask work, not full exactness.
