# EBFT Loss: Subagent Execution Breakdown

This artifact defines how to split the **future coding phase** (strict pre-EBFT gate + gated `EBFTPolicyLoss` parity) across specialized agents. It does **not** implement loss code; use it when the parent agent delegates work explicitly.

**Validation commands (unit tests + smoke + phase-1 RL+CE scope):** see [`ebft_loss_validation.md`](ebft_loss_validation.md).

Canonical planning context remains in Cursor plan **EBFT Strict Gate** (`ebft_strict_gate_*)` and Phase 2 scope in-repo (`step5_loss_decision.md`).

---

## Parent agent responsibilities

- **Review all diffs** before merge-ready state: correctness, naming, unintended behavior changes outside the subagent charter, and flag gating so EBFT stays opt-in.
- **Run targeted tests and lints** for touched areas (exact commands listed per subagent below). Do not sign off on “green enough” without the subagent’s acceptance commands.
- **Coordinate artifacts**: dump paths, OpenRLHF bundle revisions, parity reports (`g1_runtime_parity_report*.md`), smoke logs, and any new fixtures under `refactor_debugging/g1_plan/` or `tests/` data.
- **Serial dependencies**: Strict Gate → Parity Metrics refresh → Data Contract plumbing → Loss Formula + tests → Validation (full smoke). Parent blocks later agents until earlier acceptance criteria pass.
- **If the user requests subagents only**: parent does **not** apply edits; parent specifications, assigns, reviews subsidiary diffs, and runs validation commands. Coding is delegated exclusively to named subagents.

---

## Execution order

```text
1  Strict Gate Agent
2  Parity Metrics Agent  (depends on safe dumps / ordering guarantees)
4  Data Contract Agent    (parallel with 3 only after 1–2 clarify tensors; usually after 2)
3  Loss Formula Agent     (depends on frozen tensors / contract helpers from 4)
5  Validation Agent      (gates on 3–4 + refreshed reports)
```

Recommended: finish **Strict Gate** and **Parity Metrics** before substantive loss edits so regressions stay attributable.

---

## 1. Strict Gate Agent

### Scope

- **Dump safety**: no cross-rank races on shared `G1_RUNTIME_DUMP_PATH`; rank-suffixed or rank-0-only writes; atomic or append-safe patterns; metadata (DP / TP / PP rank, batch/microbatch indices); cross-field sample count validation (`tokens`, `g1_qa_masks`, embeddings, rewards, `g1_token_advantages`).
- **Dynamic batching / group order**: prove or enforce that samples stay grouped by `n_samples_per_prompt` before trainer-side G1/RLOO; if `use_dynamic_batch_size` can reorder, disable for the relevant config or restore original order prior to reward/advantage.
- **Pipeline-parallel key contract**: define visibility of G1 tensors across PP stages (last-stage-only vs broadcast); strip or gate large keys consistently; EBFT loss and training must never read keys missing on non-final stages without an explicit documented contract.

### Primary files

| Area | Paths |
| --- | --- |
| Dumps / loss glue | `slime/backends/megatron_utils/loss.py` |
| Actor / rollout of keys | `slime/backends/megatron_utils/actor.py` |
| Batch construction / grouping | `slime/backends/megatron_utils/data.py` |
| G1 THD/fast paths (ordering interactions) | `slime/backends/megatron_utils/g1_fast.py` |
| Model forward / PP | `slime/backends/megatron_utils/model.py` |

### Acceptance criteria

- No documented shared-path race between DP ranks for runtime dumps under concurrent training.
- Either tests or enforced config: `megatron_ref` + `trainer` reward path behaves correctly under `use_dynamic_batch_size=true`, or combination is rejected with clear error/docs.
- Written PP contract (short comment or `refactor_debugging/g1_plan/` note): which G1 keys exist per stage and how downstream loss consumes them.

### Validation commands

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4

# Core G1 unit tests (extend if new guards add helpers)
pytest -q tests/test_g1_core.py

# Directed runtime parity helper tests if dump/compact logic changes
pytest -q tests/test_g1_runtime_parity_metrics.py
```

Smoke (manual / GPU cluster; document machine and env):

```bash
# Adjust EBFT_ROOT, PYTHONPATH, Megatron paths per site; see script header.
bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

EBFT **policy loss** on smoke is **opt-in** only (`G1_USE_EBFT_LOSS=true`); see [`ebft_loss_validation.md`](ebft_loss_validation.md). Default smoke does not enable it.

Verify logs: finite step, dump metadata present, multi-rank configs do not collide on paths.

---

## 2. Parity Metrics Agent

### Scope

- **Compact full-group report**: exclude THD padding from Megatron hidden reporting; compare against OpenRLHF-derived tensors consistently.
- **Thresholded metrics**: `max_abs`, `mean_abs`, relative L2, cosine; explicit pass/fail for hidden states (post-compact), gen/GT embeddings, scalar rewards, token advantages—not cosine-only gates.
- **OpenRLHF bundle sync**: regenerate or version `openrlhf_parity_bundle*` dumps/scripts when Megatron/OpenRLHF contract changes; keep `compare_g1_runtime_parity.py` and bundle `dump_openrlhf_g1_from_megatron_dump.py` aligned on tensor keys and semantics.

### Primary files

| Area | Paths |
| --- | --- |
| Compare tooling | `refactor_debugging/g1_plan/compare_g1_runtime_parity.py` |
| Bundle scripts / reports | `refactor_debugging/g1_plan/openrlhf_parity_bundle/**` |
| Parity/status narrative | `refactor_debugging/g1_plan/g1_runtime_parity_report.md`, `megatron_exact_parity_status.md` |

### Acceptance criteria

- Scripted or documented thresholds match engineering targets (baseline from strict-gate plan, adjustable with user sign-off):

  ```text
  Embeddings:   cosine_min >= 0.998, max_abs <= 5e-2,  mean_abs <= 5e-3
  Rewards:      max_abs <= 5e-3,     mean_abs <= 2e-3
  Advantages:   cosine_min >= 0.995, max_abs <= 5e-2, mean_abs <= 1e-2
  ```

- Compact full-group report artifact checked in or produced by a reproducible command; link from status doc if path changes.

### Validation commands

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4

pytest -q tests/test_g1_runtime_parity_metrics.py

# Example: Megatron vs OpenRLHF bundles (exact paths vary by tarball layout)
python refactor_debugging/g1_plan/compare_g1_runtime_parity.py --help
# Run comparison with local bundle paths documented in parity report appendix
```

Regression: regenerate bundle after upstream script changes and re-run compare with exit code reflecting threshold failures.

---

## 3. Loss Formula Agent

### Scope

- Implement gated **Slime EBFT policy loss** matching OpenRLHF semantics (ratio effectively 1, no PPO clip, RL + CE masks, `masked_mean(..., dim=-1).mean()` aggregation, KL/entropy flags per OpenRLHF when in scope).
- **Unit tests** against **fixed tensors** (checked-in `.pt` or inline `torch` fixtures) mirroring OpenRLHF `EBFTPolicyLoss`; include edge cases—empty masks, mixed prompt/response lengths, `ce_coef` default `0.03`, `kl_coef=0` path.

### Primary files

| Area | Paths |
| --- | --- |
| Loss implementation | `slime/backends/megatron_utils/loss.py` |
| Loss dispatch / routing | Same + call sites in `actor.py` if needed |
| CLI / mode flags | `slime/utils/arguments.py` |

### Acceptance criteria

- New loss mode explicitly gated (CLI or `loss_type`); default training behavior unchanged unless flag set.
- Unit tests reproduce OpenRLHF numeric outputs on canned tensors within tight tolerance (`rtol/atol` documented in tests).
- No dependency on rollout-only tensors for correctness of the loss reducer itself (isolating formula tests).

### Validation commands

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4

pytest -q tests/test_g1_core.py
# EBFT loss tests once present: pytest -q tests/ -k ebft --maxfail=1
# (or tests/test_g1_ebft_loss.py — see ebft_loss_validation.md)
```

(Optional) cross-check against external OpenRLHF checkout if allowed:

```bash
PYTHONPATH=/path/to/openrlhf:/path/to/slime pytest -q tests/ -k ebft --maxfail=1
```

---

## 4. Data Contract Agent

### Scope

Wire **full-sequence** tensors into the training loss path:

| Tensor | Intended shape / alignment |
| --- | --- |
| Next-token `log_probs` | `[B, L-1]` or packed equivalent |
| `action_mask` | `[B, L-1]`, true on generated continuation positions |
| `qa_masks` | `[:, 1:]` aligned with next-token logprobs |
| Advantages | `[B, L-1]`, prompt zeros, response populated |
| Prompt CE | Must have prompt logprobs for CE—not response-only slices |

Ensure compatibility with `--g1-embedding-source megatron_ref`, rollout fields (`g1_full_sequences`, `g1_qa_masks`), and any `openrlhf_exact` forwarding path.

### Primary files

| Area | Paths |
| --- | --- |
| Loss inputs | `slime/backends/megatron_utils/loss.py`, `data.py` |
| Rollout payloads | `slime/ray/rollout.py`, `slime/rollout/g1_embedding.py` |
| Arguments / contracts | `slime/utils/arguments.py` |

### Acceptance criteria

- Single documented contract diagram or table matching `step5_loss_decision.md` EBFT prerequisites.
- Unit or integration-lite tests asserting shape/padding alignment between masks and logprobs (including variable-length batching paths used in G1 smokes).

### Validation commands

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4

pytest -q tests/test_g1_core.py
pytest -q tests/test_g1_runtime_parity_metrics.py

# Smoke with CE-capable masks (after wiring)
bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
```

---

## 5. Validation Agent

### Scope

- **Smoke scripts**: keep `run_g1_megatron_ref_smoke.sh` and launcher flags current; add `--g1-use-ebft-loss` (or chosen name) to smoke matrix when ready.
- **OpenRLHF-machine workflow**: document PYTHONPATH (`Megatron-LM`, Slime, EBFT), Ray, checkpoint paths on the reference machine.
- **Final report review**: update `g1_runtime_parity_report.md` / bundle README with command lines, thresholds, hardware, and artifact hashes or dates.

### Primary files

| Area | Paths |
| --- | --- |
| Smoke | `refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh` |
| Reporting | `refactor_debugging/g1_plan/g1_runtime_parity_report.md`, bundle `README.md` |
| Launcher touchpoints | EBFT launcher under `ebft_distribution` (external); document invocation only |

### Acceptance criteria

- One-command smoke reproduces “finite step + EBFT loss numeric sanity” checklist.
- Parity narrative matches actual compare script thresholds and archived bundle version.
- Go/no-go checklist from strict-gate plan is explicitly ticked or waived with rationale in report.

### Validation commands

```bash
cd /mnt/data/distribution-matching-slime/code/slime-0.2.4

# Full CI slice if available locally:
pytest -q tests/test_g1_core.py tests/test_g1_runtime_parity_metrics.py

bash refactor_debugging/g1_plan/run_g1_megatron_ref_smoke.sh
# EBFT loss on smoke (opt-in, after CLI flags exist): G1_USE_EBFT_LOSS=true bash ...

python refactor_debugging/g1_plan/compare_g1_runtime_parity.py …  # as documented in report
```

See [`ebft_loss_validation.md`](ebft_loss_validation.md) for `G1_USE_EBFT_LOSS` and RL+CE phase-1 scope.

---

## Risks remaining after delegation

| Risk | Mitigation owner |
| --- | --- |
| OpenRLHF reference tree path (`/mnt/data/ebft-distribution-new/code`) differs on machines | Validation Agent pins env vars / README |
| KL/entropy/temperature parity ambiguities | Loss Formula Agent + explicit user clarifications before merge |
| Dynamic batching + multi-DP interplay | Strict Gate Agent; block EBFT coding until deterministic |
| THD vs TE-fast path divergence for masks | Separate diagnostic vs production narrative; parity only on contracted path |
| Bundle drift vs Slime refactor | Parity Metrics Agent versions bundle with script hash or date |

---

## Reference (external planning)

See OpenRLHF sources on the parity machine (`openrlhf/models/loss.py`, `ebft_actor.py`, `actor.py`) listed in EBFT Strict Gate Phase 2. Slime implementations must trace to those definitions in PR notes.
