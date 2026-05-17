# G1 EBFT Smoke Evidence Report

## Run

- Command:
  `G1_USE_EBFT_LOSS=true G1_APPLY_DENSE_ATTENTION_MASK=true bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh`
- Job id: `raysubmit_iawYBacJcAgYXsME`
- Status: `succeeded`
- Exit status: `0`
- Artifact:
  `/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_phase1_evidence_0516_152210/smoke_artifacts`

## Artifact Bundle

The real smoke artifact contains:

```text
argv.sh
g1_smoke_metrics.json
g1_smoke_metrics.md
metric_extract.stderr
metric_extract.stdout
metric_extract_exit_status.txt
metrics_raw_lines.txt
ray_job_driver.log
ray_job_exit_status.txt
ray_log_archive.stderr
ray_session_latest_logs.tgz
ray_session_latest_path.txt
smoke_metadata.json
```

`metric_extract_exit_status.txt` and `ray_job_exit_status.txt` are both `0`.

## Raw Metric Evidence

`metrics_raw_lines.txt` records every parsed metric line with `source`, `line`, and the complete original metric line. This proves the values were extracted from full logs rather than only from a terminal tail.

Observed rows:

```text
/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_phase1_evidence_0516_152210/smoke_artifacts/ray_job_driver.log:2343
/tmp/ray_g1/session_latest/logs/job-driver-raysubmit_iawYBacJcAgYXsME.log:2331
```

Both rows contain the same full metric payload:

```text
step 0: {'train/loss': 0.07911529392004013, 'train/pg_loss': 9.313225746154785e-10, 'train/entropy_loss': 0.0, 'train/pg_clipfrac': 0.0, 'train/ppo_kl': 0.0, 'train/g1_ebft_ce_loss': 2.637176513671875, 'train/grad_norm': 6.475110451481636, 'train/lr-pg_0': 1e-06, 'train/lr-pg_1': 1e-06, 'train/step': 0}
```

## Phase 1 Checks

- Required metrics present: `true`
- Loss equation: `loss ~= pg_loss + 0.03 * g1_ebft_ce_loss`
- Expected loss: `0.07911529634147882`
- Observed loss: `0.07911529392004013`
- Absolute error: `2.421438688449129e-09`
- `ppo_kl == 0`: `true`
- `entropy_loss == 0`: `true`
- `pg_clipfrac == 0`: `true`

The zero KL, entropy, and clipfrac values are Phase 1 placeholders, not full OpenRLHF KL/entropy/clipping parity evidence.

## Validation

Completed before/alongside the real smoke:

```text
bash -n refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
python -m py_compile refactor_debugging/g1_plan/extract_g1_smoke_metrics.py
synthetic extractor test with --require-metrics and --output-raw-lines
PRINT_ONLY=1 smoke command
real Ray/GPU smoke command
```
