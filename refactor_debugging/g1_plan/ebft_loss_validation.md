# EBFT loss mode — validation and smoke (Slime)

This note is for **reproducible validation** of the gated OpenRLHF-parity **EBFT policy loss** path in Slime. It does not implement the loss; see [`ebft_loss_parity_contract.md`](ebft_loss_parity_contract.md) for semantics and [`ebft_loss_subagent_execution.md`](ebft_loss_subagent_execution.md) for agent splits.

## Phase 1 scope (default for first Slime implementation)

- **In scope:** **RL + CE only** — policy-gradient term with ratio ≡ 1 (no PPO clip), plus prompt CE with configurable coefficient (OpenRLHF diff-dataset G1 default **`0.03`**).
- **Out of scope until explicitly requested:** **KL** (`init_kl_coef`, `use_kl_loss`, `compute_approx_kl` parity) and **entropy** aux terms. When those are needed, extend unit tests and smoke checks per §7–§8 of the contract doc.

Strict embedding/reward/advantage dumps and compare tooling remain prerequisites; see `step5_loss_decision.md`.

---

## Unit tests (loss parity)

Run from the repo root:

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4

# Existing G1 core tests (masks, advantages, dump helpers — keep green while wiring loss)
pytest -q tests/test_g1_core.py

# Runtime parity metric helpers (unchanged by EBFT loss, but part of the pre-loss gate)
pytest -q tests/test_g1_runtime_parity_metrics.py
```

**EBFT-specific tests:** the OpenRLHF-aligned loss primitive tests live in `tests/test_g1_ebft_loss.py`. Run the focused file with the repository root on `PYTHONPATH` when Slime is not installed into the environment:

```bash
PYTHONPATH=. pytest -q tests/test_g1_ebft_loss.py
```

You can still discover the broader EBFT slice with:

```bash
PYTHONPATH=. pytest -q tests/ -k ebft --maxfail=1
```

### OpenRLHF actor-loss numeric fixture

`tests/test_g1_ebft_loss.py` now includes a deterministic OpenRLHF EBFT actor-loss fixture that always runs against Slime and, when the local OpenRLHF checkout is available, compares the exact same tensors against `EBFTPolicyLoss`.

Fixture contract:

| Field | Shape | Semantics |
| --- | --- | --- |
| `log_probs` | `[B, S] = [2, 6]` | full-sequence next-token log probabilities in the OpenRLHF actor-loss layout |
| `advantages` | `[2, 6]` | token advantages; prompt positions may be zero but are excluded from RL by `action_mask` |
| `action_mask` | `[2, 6]`, bool | `True` = generated/response token for RL; `False` = prompt token for CE |
| `qa_masks` | `[2, 6]`, bool | already shifted like OpenRLHF `qa_masks[:, 1:]`; used only when `qa_masking=True` |
| Slime packed inputs | two tensors of shape `[6]` per field | rows from the same fixture passed to `ebft_mean_rl_ce_over_packed_samples` |

Mask/reduction semantics under test:

- `qa_masking=False`: OpenRLHF replaces QA with all-ones; RL mask is `action_mask`, CE mask is `~action_mask`.
- `qa_masking=True`: RL mask is `action_mask & qa_masks`; CE mask is `~action_mask & qa_masks`.
- RL component uses `log_ratio = log_probs - log_probs.detach()`, so PPO ratio is 1 and the masked term is `-advantages`.
- CE component is `masked_mean(-log_probs, ce_mask, dim=-1).mean()`.
- Both components reduce as per-row masked mean over tokens, then batch mean over rows. This matches Slime's packed-sample mean over one row per sequence.

Tolerance: `atol=1e-7`, `rtol=1e-6` for Slime vs hand/golden and Slime vs OpenRLHF.

### OpenRLHF KL scalar CPU fixture

Phase 3 adds a CPU-only KL fixture in `tests/test_g1_ebft_loss.py` for the contract in [`openrlhf_kl_parity_contract.md`](openrlhf_kl_parity_contract.md). It validates OpenRLHF-style `action_log_probs` vs `base_action_log_probs` estimator math for `k1`/`k2`/`k3`, all-estimator `[-10, 10]` clamp, generated-token-only global `masked_mean(..., dim=None)`, coefficient scaling, and the `kl_coef == 0` zeros-like short-circuit. This remains documentation/test coverage only; `--g1-use-ebft-loss` and `--use-kl-loss` stay incompatible in training.

Golden scalar table:

| `qa_masking` | RL loss | CE loss | Latest Slime vs OpenRLHF diff |
| --- | ---: | ---: | ---: |
| `False` | `-0.1666666716` | `0.7916666269` | RL `0`, CE `0` |
| `True` | `-0.4791666567` | `0.4499999881` | RL `0`, CE `0` |

Latest local validation:

```bash
PYTHONPATH=. pytest -q tests/test_g1_ebft_loss.py
```

Observed result on this machine after adding the Phase 3 KL fixture: `10 passed, 1 skipped`. The OpenRLHF parity test loaded `/mnt/data/ebft-distribution-new/code/openrlhf/models/loss.py` directly as a source-file import, avoiding optional package dependencies from `openrlhf.models.__init__`, and matched Slime with zero component diff on the fixture. The skipped test is the THD/Megatron helper test when `megatron` is not installed.

If OpenRLHF cannot be loaded locally, the optional parity test skips with the import reason, but the deterministic golden fixture still checks the same mask/reduction contract. To rerun OpenRLHF parity, mount or checkout OpenRLHF at `/mnt/data/ebft-distribution-new/code/openrlhf` (or update the path in the test helper), then rerun the focused pytest command above.

---

## Megatron / Ray smoke (trainer-side G1, `megatron_ref`)

Default behavior of [`run_g1_megatron_ref_smoke.sh`](run_g1_megatron_ref_smoke.sh) is **unchanged** and does **not** enable EBFT loss.

### Standard smoke (baseline)

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4
bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

Adjust `EBFT_ROOT`, `SLIME_ROOT`, `MEGATRON_PATH`, checkpoints, and `SLIME_TRAIN_DATA` per the script header and site layout.

### Opt-in EBFT loss on smoke (**after** CLI flags exist)

When `slime/utils/arguments.py` registers **`--g1-use-ebft-loss`** and **`--g1-ce-loss-coef`**, the smoke script can append parity-friendly defaults via env:

```bash
G1_USE_EBFT_LOSS=true \
  bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

This adds `--g1-use-ebft-loss --g1-ce-loss-coef 0.03` to the submitted command. If the flags are not in the codebase yet, the script **exits with an error** rather than passing unknown CLI options to Ray.

Dry-run the resolved argv without starting Ray:

```bash
PRINT_ONLY=1 G1_USE_EBFT_LOSS=true \
  bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

(With `PRINT_ONLY=1`, the script prints the command and exits before `ray start`.)

### Smoke checklist (EBFT mode)

- Job reaches a **finite** actor train step; no NaNs in logged loss scalars.
- With runtime dumps enabled elsewhere, metadata remains DP-safe and **group order** matches `n_samples_per_prompt` (see Strict Gate narrative in `ebft_loss_subagent_execution.md`).
- For phase 1, **do not** expect KL/entropy parity with OpenRLHF unless the reference run enables those and Slime implements the matching path.

### Latest observed EBFT smoke

The first Megatron/Ray EBFT RL+CE smoke completed successfully with:

```bash
G1_USE_EBFT_LOSS=true \
G1_APPLY_DENSE_ATTENTION_MASK=true \
bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

Observed terminal result:

```text
Job 'raysubmit_sBwPYa6Q2PycQDyD' succeeded
```

This validates the current integration path through argument parsing, ref checkpoint loading, trainer-side G1 embedding/advantage computation, rollout logging, EBFT loss shape alignment, actor train, and weight update.

Follow-up smoke runs now archive the full Ray/driver logs and parse the Phase 1 loss metrics automatically. The artifact bundle is written to `${SMOKE_ARTIFACT_DIR:-$(dirname "${LOAD_PATH}")/smoke_artifacts}` and includes `ray_job_driver.log`, `ray_session_latest_logs.tgz`, `smoke_metadata.json`, `argv.sh`, `g1_smoke_metrics.json`, `g1_smoke_metrics.md`, and `metrics_raw_lines.txt` with source path, line number, and the complete original metric log line. In `G1_USE_EBFT_LOSS=true` mode, metric extraction requires:

```text
loss ~= pg_loss + 0.03 * g1_ebft_ce_loss
ppo_kl == entropy_loss == pg_clipfrac == 0
```

The KL/entropy/clipfrac zeros are Phase 1 placeholders, not claims of full OpenRLHF KL or entropy parity.

### Actor-loss runtime dump and replay report

Slime can now dump the exact EBFT actor-loss tensors for one Megatron/Ray microbatch when the env var
`G1_EBFT_ACTOR_LOSS_DUMP_PATH` is set. The dump is opt-in and disabled by default.

Example smoke invocation:

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4

G1_USE_EBFT_LOSS=true \
G1_APPLY_DENSE_ATTENTION_MASK=true \
G1_EBFT_ACTOR_LOSS_DUMP_PATH=/tmp/g1_ebft_actor_loss_runtime.pt \
  bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

After the smoke writes the dump, replay it against local OpenRLHF `EBFTPolicyLoss` when available, or the
same-semantics torch fallback when OpenRLHF cannot be imported:

```bash
PYTHONPATH=. python refactor_debugging/g1_plan/compare_g1_ebft_actor_loss_runtime.py \
  /tmp/g1_ebft_actor_loss_runtime.pt \
  --output refactor_debugging/g1_plan/g1_ebft_actor_loss_runtime_report.md \
  --fail-on-diff
```

The report includes input shape/hash, OpenRLHF source/commit (or fallback reason), and RL/CE/total diffs with
tolerance. CPU tests use a small temporary dump fixture only to exercise replay and diff logic; that fixture is not
a real Megatron/Ray runtime report.

Latest real Megatron/Ray runtime report:

```text
Dump:
/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_ebft_actor_loss_runtime_0516_154631/g1_ebft_actor_loss_runtime.pt

Report:
/mnt/data/ebft-distribution-new/outputs/diff_dataset/g1_ebft_actor_loss_runtime_0516_154631/g1_ebft_actor_loss_runtime_parity_report.md
```

The report replays one real packed Megatron actor-loss microbatch against OpenRLHF `EBFTPolicyLoss`
from commit `27e854464a9355d251d7f93b44b09300921972e0` and passes with:

```text
RL abs diff    = 3.72529029846e-09
CE abs diff    = 2.38418579102e-07
total abs diff = 1.11758708954e-08
```

---

## Full parity compare (optional, separate from smoke)

Megatron vs OpenRLHF bundle comparison (paths vary by machine):

```bash
python refactor_debugging/g1_plan/compare_g1_runtime_parity.py --help
```

Thresholds and bundle layout are documented in `g1_runtime_parity_report.md` and bundle READMEs.

---

## Related docs

| Doc | Role |
| --- | --- |
| [`ebft_loss_parity_contract.md`](ebft_loss_parity_contract.md) | Loss semantics, masks, aggregation, KL/temperature notes |
| [`ebft_loss_subagent_execution.md`](ebft_loss_subagent_execution.md) | Subagent ownership and validation commands |
| [`step5_loss_decision.md`](step5_loss_decision.md) | Gate before turning on EBFT loss in production configs |
