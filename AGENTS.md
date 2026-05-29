# Repository Guidelines

## Project Structure & Module Organization

This is a Python training framework centered on `slime/`. Core distributed training code lives in `slime/ray/`, rollout logic in `slime/rollout/`, shared utilities in `slime/utils/`, and Megatron integration in `slime/backends/megatron_utils/`. Entry points are `train.py` and `train_async.py`. Example launch scripts are under `examples/`, experiment scripts under `exper_scripts/`, and repository utilities under `scripts/`. Tests live in `tests/`, with focused utility tests in `tests/utils/` and plugin contract tests in `tests/plugin_contracts/`. Documentation is in `docs/`.

## Project Lineage

The G1/G2/G3 training line in this repo descends from EBFT / DMFT research code, not from generic RLHF alone. Read `dmft_agent_brief.md` before changing G2, `cf_l1oo`, OPD, teacher-target, or feature-embedding logic. The source reference repo is `/mnt/data/ebft-distribution-new/code`, especially `scripts/diff_dataset/` and `openrlhf/trainer/ppo_utils/ebft_experience_maker.py`.

## Orchestrator/Subagent Operating Contract

Every agent must identify whether it is acting as the root orchestrator or as a delegated subagent before doing work.

- The root orchestrator talks with the user, verifies scope and assumptions with the user, and summarizes delegated results back to the user.
- The root orchestrator does not inspect source code or write repository code directly. Its job is coordination, verification, and conversation with the user.
- The root orchestrator must delegate code inspection and code-writing work to multiple subagents whenever implementation work is needed.
- Use at least one read-only investigator subagent for code inspection and at least one implementation subagent for scoped edits. Use a separate verifier/test subagent when the change can be independently checked.
- Delegated subagents may inspect or edit code only inside the explicit scope given by the orchestrator. Each delegation should state the role, allowed files or areas, allowed operations, and expected report format.
- If no subagent mechanism is available in the current session, the orchestrator must pause and tell the user it cannot inspect or edit code under this contract.
- The orchestrator must not imply it personally inspected or edited code. It should report which subagent did the work, what was verified, and what still needs user confirmation.

## Build, Test, and Development Commands

- `pip install -e .`: install the package in editable mode.
- `pip install -r requirements.txt`: install runtime dependencies.
- `pytest`: run the full configured test suite from `tests/`.
- `pytest tests/test_g1_core.py`: run one focused test module.
- `pre-commit run --all-files`: run formatting, linting, and basic safety hooks before submitting changes.

Large GPU/Ray integration tests may require the expected model checkpoints, CUDA environment, and available ports; prefer focused unit tests for local iteration.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation. Format with Black using line length 119 where practical; imports are sorted by isort with the Black profile. Ruff checks core `E`, `F`, `B`, and `UP` rules, with long lines currently tolerated by configuration. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and descriptive flag names matching existing CLI patterns such as `--use-opd` or `--distribution-reward-type`.

## Testing Guidelines

Tests use pytest and should be named `test_*.py`. Add narrow unit tests for pure utilities, data contracts, reward math, and argument validation. Mark broader tests with existing markers such as `unit`, `integration`, or `system` when appropriate. Keep generated caches, logs, and heavyweight artifacts out of commits.

## Commit & Pull Request Guidelines

Recent history uses concise prefixes such as `docs:`, `chore:`, `refactor:`, `hotfix:`, and `feat:`. Follow that style and keep the subject imperative and specific, for example `feat: add g2 teacher embedding validation`. Pull requests should describe the behavior change, list tests run, call out GPU/Ray assumptions, and link related issues or experiment notes. Include logs or metrics for training-flow changes when they are relevant.

## Security & Configuration Tips

Do not commit private keys, credentials, local checkpoints, Ray temp data, or large logs. Prefer environment variables and local scripts for machine-specific paths. Before changing distributed training, verify actor, critic, rollout, and weight-sync paths separately.
