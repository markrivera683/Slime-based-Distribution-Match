# G1 Runtime Parity Plan

## Purpose

The current Megatron/ref smoke proves the Slime-native fast path can generate G1 token advantages and complete one actor train step. It does **not** prove that Megatron/ref hidden states are numerically equivalent to OpenRLHF `Critic` hidden states.

This document defines the next parity work so the project can distinguish:

1. helper-level parity, which should be exact on fixed hidden tensors;
2. HF/OpenRLHF critic parity, which should be close when the same model/mask path is used;
3. Megatron/ref runtime parity, which may differ because the current Megatron fast path does not yet use OpenRLHF's dense EBFT strided attention mask.

## Current Known Difference

OpenRLHF G1 critic uses:

```text
full_sequence
  -> build_strided_attention_mask_and_positions
  -> Critic.forward(output_hidden_states=True)
  -> L2 normalize hidden
  -> QA mask
  -> split/unfold/reshape
  -> last_token groom
```

Current Slime Megatron/ref path uses:

```text
g1_full_sequences
  -> standard Megatron packed forward
  -> capture decoder hidden
  -> sequence-parallel gather
  -> L2 normalize hidden
  -> split/unfold/reshape
  -> last_token groom
```

Because the attention path is not identical, tight full-hidden parity is **not** expected until the mask/position behavior is explicitly matched or the difference is accepted as an implementation choice.

## Parity Stages

### Stage 1: Fixed Hidden Helper Parity

Status: already covered by `tests/test_g1_core.py`.

Acceptance:

```text
hidden_states_to_g1_embeddings(fixed_hidden, qa_mask)
```

matches the OpenRLHF block/groom order with `rtol=1e-5`, `atol=1e-5`.

### Stage 2: OpenRLHF Critic Runtime Dump

Use the OpenRLHF environment to dump:

```text
sequences
qa_masks
doc_ids
attention_mask
position_ids
normalized_hidden_states
gen_embedding
gt_embedding
rewards
g1_token_advantages
```

Acceptance:

- Slow Slime HF/OpenRLHF path and OpenRLHF reference match within dtype-aware tolerance.
- This validates the shared helper and checkpoint path without involving Megatron.

### Stage 3: Megatron/ref Runtime Comparison

Use the same `g1_full_sequences` and `g1_qa_masks` as the passing Megatron/ref smoke.

Dump from Slime:

```text
captured_decoder_hidden_pre_norm
captured_decoder_hidden_post_sp_gather
normalized_hidden
gen_embedding
gt_embedding
g1_token_advantages
```

Compare against OpenRLHF:

- exact token ids and QA masks first;
- then embedding shapes and groom indices;
- then cosine/correlation of hidden and block embeddings;
- finally reward / advantage distributions.

Acceptance for current implementation:

- Shape and ordering must match exactly.
- Fixed-helper parity remains strict.
- Megatron-vs-OpenRLHF hidden parity is reported as a measured gap, not assumed exact.

## Required Artifact

The next parity run should write a single `.pt` or `.npz` bundle containing both sides' tensors and a markdown summary:

```text
refactor_debugging/g1_plan/artifacts/g1_runtime_parity_megatron_ref.pt
refactor_debugging/g1_plan/g1_runtime_parity_report.md
```

## Runtime Parity Result

Generated on 2026-05-16:

```text
Megatron dump:  refactor_debugging/g1_plan/artifacts/g1_runtime_megatron_ref_dump.pt
OpenRLHF dump:  refactor_debugging/g1_plan/artifacts/g1_runtime_openrlhf_critic_dump.pt
Report:         refactor_debugging/g1_plan/g1_runtime_parity_report.md
```

Contract checks:

```text
Token IDs match: True
QA masks match: True
Megatron hidden shape: (1, 760, 2560)
OpenRLHF hidden shape: (1, 760, 2560)
Megatron gen embedding shape: (1, 47, 2560)
OpenRLHF gen embedding shape: (1, 47, 2560)
```

Cosine summary:

```text
Full hidden mean / median / min: 0.9591 / 0.9975 / 0.2117
Gen block mean / median / min:  0.9057 / 0.9176 / 0.7062
GT block mean / median / min:   0.9990 / 0.9992 / 0.9976
```

Interpretation: helper-level shape/order is aligned and GT embeddings are nearly identical, but generated-block embeddings have a measurable gap. This is expected for the current implementation because OpenRLHF uses EBFT dense strided attention masks and custom position ids, while the Megatron/ref path currently uses standard packed forward semantics.

## Go / No-Go

Do not rename the path to `slime_g1_exact` until runtime parity and loss parity are both addressed. Until then, use names such as:

```text
slime_g1_megatron_ref_reward
slime_g1_advantage_smoke
```
