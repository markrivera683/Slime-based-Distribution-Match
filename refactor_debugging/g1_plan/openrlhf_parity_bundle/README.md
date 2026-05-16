# OpenRLHF G1 Parity Bundle

This bundle contains Megatron/ref exact diagnostic dumps from the current Slime workspace and the scripts needed to generate an OpenRLHF reference dump on a machine whose Python environment can load `Qwen3_5ForCausalLM`.

## Contents

- `artifacts/g1_runtime_megatron_ref_mask_applied_full_group_compact_dump.pt`: full-group Megatron/ref dump produced with `--g1-megatron-ref-forward-mode openrlhf_exact --g1-megatron-ref-apply-dense-attention-mask`, with THD padding removed from the hidden-state dump.
- `artifacts/g1_runtime_megatron_ref_mask_applied_full_group_dump.pt`: earlier full-group dump that still contains padded hidden states; keep it only for debugging.
- `artifacts/g1_runtime_megatron_ref_exact_dump.pt`: earlier one-sample diagnostic dump without dense-mask application.
- `scripts/dump_openrlhf_g1_from_megatron_dump.py`: runs OpenRLHF Critic on the same token IDs and writes the reference dump.
- `scripts/compare_g1_runtime_parity.py`: compares Megatron and OpenRLHF dumps and writes a markdown report.
- `megatron_exact_parity_status.md`: current implementation status and mask blocker note.
- `baseline_g1_runtime_parity_report.md`: previous baseline report before the exact diagnostic rerun.

## Run On The OpenRLHF Machine

From the root of this bundle:

```bash
export BUNDLE_DIR="$(pwd)"
export OPENRLHF_REPO=/mnt/data/ebft-distribution-new/code
export SLIME_ROOT=/mnt/data/distribution-matching-slime/code/slime-0.2.4
export MODEL_PATH=/mnt/data/models/Qwen3.5-4B
export PYTHON_BIN=python

export PYTHONPATH="${SLIME_ROOT}:${OPENRLHF_REPO}:${PYTHONPATH:-}"

"${PYTHON_BIN}" scripts/dump_openrlhf_g1_from_megatron_dump.py \
  --megatron-dump artifacts/g1_runtime_megatron_ref_mask_applied_full_group_compact_dump.pt \
  --out artifacts/g1_runtime_openrlhf_critic_mask_applied_full_group_compact_dump.pt \
  --model-path "${MODEL_PATH}" \
  --openrlhf-repo "${OPENRLHF_REPO}" \
  --device cuda \
  --dtype bfloat16

"${PYTHON_BIN}" scripts/compare_g1_runtime_parity.py \
  --megatron-dump artifacts/g1_runtime_megatron_ref_mask_applied_full_group_compact_dump.pt \
  --openrlhf-dump artifacts/g1_runtime_openrlhf_critic_mask_applied_full_group_compact_dump.pt \
  --out g1_runtime_parity_report_mask_applied_full_group_compact.md
```

## Expected Checks

In `g1_runtime_parity_report_mask_applied_full_group_compact.md`, check:

```text
Position ids match: True
Attention mask tensors match: True
Megatron attention mask applied: True
```

For the full-group mask-applied run, `Megatron attention mask applied` should be `True`, and reward/advantage sections should no longer be `not available`.
