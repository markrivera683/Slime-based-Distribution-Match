# G1 EBFT Next Phase Status Report

## Closed Evidence Gaps

### 1. Full Smoke Metric Evidence

Real EBFT smoke with dense-mask diagnostic fallback completed:

```text
Command:
G1_USE_EBFT_LOSS=true G1_APPLY_DENSE_ATTENTION_MASK=true \
  bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh

Job:
raysubmit_iawYBacJcAgYXsME

Artifact:
/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_megatron_ref_smoke_phase1_evidence_0516_152210/smoke_artifacts
```

The artifact includes `ray_job_driver.log`, `ray_session_latest_logs.tgz`,
`smoke_metadata.json`, `argv.sh`, `g1_smoke_metrics.json`,
`g1_smoke_metrics.md`, and `metrics_raw_lines.txt`.

`metrics_raw_lines.txt` records the original log line and source location. The
observed Phase 1 check passes:

```text
loss = 0.07911529392004013
pg_loss = 9.313225746154785e-10
g1_ebft_ce_loss = 2.637176513671875
expected loss = pg_loss + 0.03 * g1_ebft_ce_loss
abs error = 2.421438688449129e-09
ppo_kl = entropy_loss = pg_clipfrac = 0
```

The zero KL/entropy/clipfrac values are Phase 1 placeholders only.

### 2. OpenRLHF Actor-Loss Runtime Parity

After fixing Ray runtime-env propagation and removing the rank-0-only dump gate,
a real Megatron/Ray actor-loss runtime dump was produced and replayed against
OpenRLHF `EBFTPolicyLoss`:

```text
Smoke job:
raysubmit_Ynj7wwLrZJbPypyS

Dump:
/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_ebft_actor_loss_runtime_0516_154631/g1_ebft_actor_loss_runtime.pt

Report:
/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_ebft_actor_loss_runtime_0516_154631/g1_ebft_actor_loss_runtime_parity_report.md
```

The report uses OpenRLHF commit
`27e854464a9355d251d7f93b44b09300921972e0` and passes:

```text
RL abs diff    = 3.72529029846e-09
CE abs diff    = 2.38418579102e-07
total abs diff = 1.11758708954e-08
```

The runtime batch is a real packed Megatron actor-loss microbatch:

```text
log_probs_next: 1 x (400,)
advantages_next: 1 x (400,)
action_mask_next: 1 x (400,)
qa_mask_next: 1 x (400,)
```

## Remaining Caveats

### KL / Entropy Parity

KL parity is not wired into training. The new contract
`openrlhf_kl_parity_contract.md` documents the target OpenRLHF semantics and
`tests/test_g1_ebft_loss.py` contains CPU fixture coverage. The existing
training guard that prevents `--g1-use-ebft-loss` with Slime KL/entropy remains
intentional.

### TE THD Fast Exact Mask

`te_thd_dense_mask_feasibility.md` records the feasibility spike. The installed
Megatron `0.16.0rc0` and Transformer Engine `2.10.0` stack does not expose a
standard packed THD fast backend for arbitrary dense masks or dense post-scale
bias. The exact dense mask path remains:

```text
applied-via-torch-thd-fallback
```

This is a diagnostic parity path, not a Transformer Engine fast path.

## Validation Commands

```bash
PYTHONPATH=. /root/venvs/slime/bin/python -m pytest -q \
  tests/test_g1_core.py \
  tests/test_g1_ebft_loss.py \
  tests/test_g1_ebft_data_contract.py \
  tests/test_g1_runtime_parity_metrics.py \
  tests/test_g1_ebft_actor_loss_runtime.py

bash -n refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh

python -m py_compile \
  refactor_debugging/g1_plan/extract_g1_smoke_metrics.py \
  refactor_debugging/g1_plan/compare_g1_runtime_parity.py \
  refactor_debugging/g1_plan/compare_g1_ebft_actor_loss_runtime.py

git diff --check
```

Observed result:

```text
73 passed
no lint errors
```
