# G1 Runtime Parity Report

## Inputs

- Megatron dump: `refactor_debugging/g1_plan/openrlhf_parity_bundle/artifacts/g1_runtime_megatron_ref_mask_applied_full_group_dump.pt`
- OpenRLHF dump: `refactor_debugging/g1_plan/openrlhf_parity_bundle/artifacts/g1_runtime_openrlhf_critic_mask_applied_full_group_dump.pt`

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
- Megatron hidden shape: `(4, 760, 2560)`
- OpenRLHF hidden shape: `(4, 760, 2560)`
- Megatron gen embedding shape: `(4, 47, 2560)`
- OpenRLHF gen embedding shape: `(4, 47, 2560)`

## Mask Summary

- Megatron mask summary: `{'shape': (4, 1, 760, 760), 'allowed_count': 591216, 'blocked_count': 1719184, 'finite_count': 2310400}`
- OpenRLHF mask summary: `{'shape': (4, 1, 760, 760), 'allowed_count': 591216, 'blocked_count': 1719184, 'finite_count': 2310400}`

## Cosine Similarity

### Full Hidden States

```text
mean   = 0.40281841
median = 0.21599233
min    = -0.12287912
max    = 0.99994892
```

### Gen Block Embeddings

```text
mean   = 0.99986154
median = 0.99987578
min    = 0.99901927
max    = 0.99991870
```

### GT Block Embeddings

```text
mean   = 0.99902207
median = 0.99910927
min    = 0.99689943
max    = 0.99991292
```

### Rewards

```text
max_abs  = 0.00221942
mean_abs = 0.00108439
```

### Token Advantages

```text
sample_cos_mean = 0.99966836
sample_cos_min  = 0.99948490
max_abs         = 0.02138489
mean_abs        = 0.00396629
```

## Interpretation

This report compares the current Slime Megatron/ref fast path against OpenRLHF Critic on identical token IDs and QA masks.

Use the runtime mask status above to distinguish the staged contracts. `openrlhf_exact` means Megatron uses OpenRLHF-compatible position ids/RoPE inputs. `dense-mask-built-not-applied` means the OpenRLHF EBFT dense mask was dumped for comparison but did not affect Megatron attention. `applied-via-torch-thd-fallback` means `apply_dense_attention_mask` was effective through the slow diagnostic torch THD fallback, not the standard Megatron/TE `thd` fast path.
