# OPD Refactor Notes

This directory tracks the OPD-side refactor work that is separate from EffOPD.
EffOPD is a later acceleration layer; the current priority is to make the OPD
training signal itself explicit and testable.

## Current Direction

The active training target is **OPD-CF-L1OO**:

```text
student on-policy rollouts
+ teacher logprobs on those rollouts
+ frozen feature embeddings
+ group credit assignment
```

Teacher completions are not part of this main path. The teacher scores student
rollouts; it does not define a separate teacher-completion target distribution.

## Implemented Modes

Use the shared opt-in target mode:

```bash
--distribution-reward-type cf_l1oo
--cf-target-mode opd_onpolicy
--use-opd
```

Credit assignment is selected with:

```bash
--opd-credit-assignment cf_l1oo
--opd-credit-assignment ebft
```

- `cf_l1oo`: OPD teacher scores become weights on the student rollout feature
  support. Characteristic-function discrepancy plus leave-one-out produces the
  per-rollout/block reward.
- `ebft`: baseline. OPD teacher scores still weight the same on-policy rollout
  support, but credit is pointwise EBFT-style cosine similarity to the
  teacher-weighted on-policy feature centroid, followed by RLOO shaping.

## Smoke Scripts

Two smoke wrappers live under `smoke_scripts/`:

```bash
bash refactor_debugging/OPD/smoke_scripts/run_opd_cf_l1oo_smoke.sh
bash refactor_debugging/OPD/smoke_scripts/run_opd_ebft_credit_smoke.sh
```

Both wrappers call the shared main script:

```text
exper_scripts/main_test/run_g2_opd_qwen35_2b_main.sh
```

They set:

```text
CF_TARGET_MODE=opd_onpolicy
ENABLE_EFFOPD=false
G1_USE_EBFT_LOSS=false
NUM_ROLLOUT=2
ROLLOUT_BATCH_SIZE=2
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=4
```

By default the refactor smoke wrappers start a tiny local mock SGLang logprob
endpoint on `127.0.0.1:30123`. This is intended for plumbing smoke tests only:
it validates the OPD request/response contract without requiring a real teacher
model.

For an endpoint-only check that stops before Ray startup:

```bash
PREFLIGHT_ONLY=1 bash refactor_debugging/OPD/smoke_scripts/run_opd_cf_l1oo_smoke.sh
```

To use a real OPD logprob endpoint instead:

```bash
OPD_SMOKE_MOCK_RM=false \
TEACHER_API_BASE=http://host:port \
OPD_TEACHER_RM_URL=http://host:port/generate \
bash refactor_debugging/OPD/smoke_scripts/run_opd_cf_l1oo_smoke.sh
```

For `opd_onpolicy`, the main script skips the old G2 teacher-completion API
preflight and only checks the OPD reward/logprob endpoint.

## Code Pointers

- `slime/utils/g2_core.py`: `compute_opd_cf_l1oo_rewards`
- `slime/backends/megatron_utils/g1_fast.py`: OPD on-policy teacher-score
  aggregation, `cf_l1oo` branch, and `ebft` baseline branch
- `slime/backends/megatron_utils/loss.py`: OPD reverse-KL metric logging vs
  tokenwise KL application
- `slime/utils/arguments.py`: `opd_onpolicy` and OPD credit-assignment args
- `tests/test_g1_core.py`: unit coverage for core reward behavior and trainer
  wiring

## Validation Status

Local environment limitations:

- `pytest` is not installed.
- some repo imports are blocked by missing optional/runtime deps such as
  `pybase64`, `yaml`, and `megatron`.

Checks that currently pass in this environment:

```bash
python -m compileall -q \
  slime/utils/g2_core.py \
  slime/backends/megatron_utils/g1_fast.py \
  slime/backends/megatron_utils/loss.py \
  slime/utils/arguments.py \
  slime/backends/megatron_utils/actor.py \
  tests/test_g1_core.py
```

```bash
python - <<'PY'
import torch
from slime.utils.g2_core import compute_opd_cf_l1oo_rewards

r = compute_opd_cf_l1oo_rewards(
    torch.tensor([[[[[0.0]], [[1.0]], [[2.0]]]]]),
    torch.tensor([[[0.0, 0.0, 4.0]]]),
    cf_num_freqs=8,
    cf_seed=7,
)
print(tuple(r.shape), torch.isfinite(r).all().item())
PY
```

See `docs/opd_cf_l1oo_training.md` for the full design and acceptance checklist.
