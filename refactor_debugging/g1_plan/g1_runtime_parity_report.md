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
- `openrlhf_exact` active: `True`
- Dense mask state: `applied-via-torch-thd-fallback`
- Dense mask dumped: `True`
- Dense mask built: `True`
- `apply_dense_attention_mask` effective: `True`
- THD fallback active: `True`
- Runtime mask status consistency: `PASS`
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

Use the runtime mask status above to distinguish the staged contracts. `openrlhf_exact` means Megatron uses OpenRLHF-compatible position ids/RoPE inputs. `dense-mask-built-not-applied` means the OpenRLHF EBFT dense mask was dumped for comparison but did not affect Megatron attention. `applied-via-torch-thd-fallback` means `apply_dense_attention_mask` was effective through the slow diagnostic torch THD fallback, not the standard Megatron/TE `thd` fast path.
