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

**EBFT-specific tests:** when the Loss Formula work lands, they should live in a dedicated file (e.g. `tests/test_g1_ebft_loss.py`) or under `tests/test_g1_core.py` with `ebft` in the test name. Discover them with:

```bash
pytest -q tests/ -k ebft --maxfail=1
```

If that selects zero tests, the EBFT loss test module is not merged yet — use `pytest tests/ -k g1` for a broader slice.

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

This validates the current integration path through argument parsing, ref checkpoint loading, trainer-side G1 embedding/advantage computation, rollout logging, EBFT loss shape alignment, actor train, and weight update. The terminal transcript retained only the tail of the run, so a follow-up smoke should preserve the full Ray logs if exact `g1_ebft_ce_loss` and total-loss scalar values need to be archived.

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
