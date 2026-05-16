# G1 Runtime Parity Report

## Inputs

- Megatron dump: `refactor_debugging/g1_plan/artifacts/g1_runtime_megatron_ref_dump.pt`
- OpenRLHF dump: `refactor_debugging/g1_plan/artifacts/g1_runtime_openrlhf_critic_dump.pt`

## Contract Checks

- Token IDs match: `True`
- QA masks match: `True`
- Megatron hidden shape: `(1, 760, 2560)`
- OpenRLHF hidden shape: `(1, 760, 2560)`
- Megatron gen embedding shape: `(1, 47, 2560)`
- OpenRLHF gen embedding shape: `(1, 47, 2560)`

## Cosine Similarity

### Full Hidden States

```text
mean   = 0.95907682
median = 0.99750704
min    = 0.21169904
max    = 0.99994385
```

### Gen Block Embeddings

```text
mean   = 0.90571195
median = 0.91762459
min    = 0.70618188
max    = 0.97059512
```

### GT Block Embeddings

```text
mean   = 0.99899626
median = 0.99918962
min    = 0.99760079
max    = 0.99991310
```

## Interpretation

This report compares the current Slime Megatron/ref fast path against OpenRLHF Critic on identical token IDs and QA masks.

Tight tensor equality is **not** expected yet because OpenRLHF uses EBFT dense strided attention masks and custom position ids, while the current Megatron/ref path uses standard Megatron packed forward semantics.
