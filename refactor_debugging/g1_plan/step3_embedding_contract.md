# Step 3: G1 Embedding Contract

## Status

Step 3 contract 初版完成。它定义慢 HF/OpenRLHF embedding path 必须写入 slime `Sample.metadata` 的数据形状和失败条件。

## Goal

第一版目标是把 OpenRLHF diff-dataset G1 的 frozen critic embedding 语义接进 slime 小闭环：

```text
slime rollout sample
  -> full_sequence / qa_masks / doc_ids
  -> slow HF/OpenRLHF critic embedding helper
  -> Sample.metadata["g1_gen_embedding"]
  -> Sample.metadata["g1_gt_embedding"]
  -> group_rm custom_rm
  -> g1_token_advantages
```

这一步只解决 embedding 生产与 reward/RLOO/advantage 接线，不解决 EBFT actor loss。

## Fixed Geometry

第一版固定使用 diff-dataset G1 几何：

| Name | Value |
| --- | --- |
| `prompt_length` | `384` |
| `context_length` | `8` |
| `generate_length` | `8` |
| `stride` | `8` |
| `num_blocks` | `47` |
| `response_length` | `376` |
| `n_samples_per_prompt` | `4` |

Derived:

```text
num_blocks = (prompt_length - generate_length - context_length) / stride + 1
           = (384 - 8 - 8) / 8 + 1
           = 47

response_length = num_blocks * generate_length = 47 * 8 = 376
full_sequence_length = prompt_length + response_length = 760
```

First version rejects:

- `response_length != 376`
- `num_blocks != 47`
- truncation
- stop-early samples
- variable-length responses

## Required Sample Group Contract

G1 reward is group-relative. Each group passed to `group_rm` must satisfy:

- group size equals `n_samples_per_prompt == 4`
- all samples in the group correspond to the same prompt
- group order is stable and matches rollout order
- every sample has `response_length == 376`
- every sample has `metadata["g1_gen_embedding"]`
- every sample has `metadata["g1_gt_embedding"]`

The expected per-sample metadata shape is:

```text
g1_gen_embedding: [47, hidden_dim]
g1_gt_embedding:  [47, hidden_dim]
```

The group-level tensor shape consumed by `slime.rollout.rm_hub.g1_core.compute_group_g1_rewards` is:

```text
gen_embedding: [1, 1, 4, 47, hidden_dim]
gt_embedding:  [1, 1, 4, 47, hidden_dim]
```

## Slow Embedding Path Inputs

The slow HF/OpenRLHF helper must receive:

| Input | Shape | Notes |
| --- | --- | --- |
| `sequences` | `[B, 760]` | packed prompt + generated response |
| `qa_masks_full` | `[B, 760]` | first version may be all ones if `qa_masking=False`, but the field is preserved |
| `doc_ids_prompt` | `[B, 384]` | used only if document masking is enabled; first version uses a single-doc default |
| `prompt_length` | scalar | `384` |
| `context_length` | scalar | `8` |
| `generate_length` | scalar | `8` |
| `stride` | scalar | `8` |
| `num_blocks` | scalar | `47` |

The first version assumes `hidden_state_method=last_only`, so the hidden feature slot dimension is `NF=1`. That allows `g1_gen_embedding` / `g1_gt_embedding` to be flattened to `[47, hidden_dim]` before group-level whitening.

## Prompt Reconstruction For Slime Data

Slime diff-dataset JSONL stores:

- `Sample.prompt`: question / chat prompt
- `Sample.label`: reference answer
- generated response in `Sample.tokens[-response_length:]`

The slow helper reconstructs the OpenRLHF-style prompt sequence by tokenizing `prompt + label`, with answer tokens marked in `qa_masks_full`.

First version restrictions:

- if prompt+label exceeds `prompt_length`, raise an error rather than truncate silently
- if response token count is not exactly `response_length`, raise an error
- use pad token id when prompt+label is shorter than `prompt_length`
- set padding `doc_id` to `-1`

This is a temporary path for correctness and parity. It does not claim to reproduce OpenRLHF multi-document packing when multiple QA pairs share a single 384-token chunk.

## OpenRLHF Embedding Order To Replicate

The helper must follow OpenRLHF order:

```text
critic full-sequence forward
  -> hidden_state_method=last_only
  -> L2 normalize hidden states
  -> optional feature_adapter (off in G1)
  -> apply qa mask (qa_masking=false means all ones)
  -> slice gt region: [context_length:prompt_length]
  -> slice generated region: [prompt_length:]
  -> gt unfold(window=generate_length, stride=stride)
  -> gen reshape(generate_length, num_blocks).transpose(block/time)
  -> last_token + groom
  -> flatten NF * H
  -> feature_map identity
```

Whitening is not performed in the embedding producer for the first version. Whitening remains in `slime.utils.g1_core.compute_pointwise_rewards`, where it is applied across the group sample axis.

## CLI Contract

All geometry and model paths must be configurable. First-version defaults may match diff-dataset G1, but code must not rely on hidden constants.

Required knobs:

- `--g1-prompt-length`
- `--g1-context-length`
- `--g1-generate-length`
- `--g1-stride`
- `--g1-response-length`
- `--g1-critic-model-path`
- `--g1-tokenizer-path`
- `--g1-openrlhf-repo`
- `--g1-hidden-state-method`
- `--g1-embedding-device`
- `--g1-embedding-dtype`

Recommended fixed smoke values:

```text
--g1-prompt-length 384
--g1-context-length 8
--g1-generate-length 8
--g1-stride 8
--g1-response-length 376
--g1-hidden-state-method last_only
--rollout-max-response-len 376
```

## Failure Conditions

The implementation must fail loudly when:

- `response_length != g1_response_length`
- `sample.label is None`
- `prompt + label` cannot fit in `g1_prompt_length`
- tokenizer has no usable pad/eos token id
- `num_blocks` computed from geometry differs from the embedding shape
- any group sample is missing either embedding
- group samples have inconsistent embedding shapes

## Step 3 Gate

Before any real training run:

1. `tests/test_g1_core.py` remains green.
2. Embedding contract tests validate shape and failure behavior.
3. The slow helper can produce finite `[47, hidden_dim]` embeddings for a mocked or tiny model path.
4. Group RM writes `g1_token_advantages` of length `376`.
5. DP split preserves `g1_token_advantages`.
