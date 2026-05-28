# Codex Migration Notes

This directory is the Codex-oriented migration of the repository-local `.claude`
configuration.

## Skills

The Claude skills were copied into `.codex/skills/` with their original
`SKILL.md` content preserved:

- `add-dynamic-filter`
- `add-eval-dataset-config`
- `add-reward-function`
- `add-rollout-function`
- `add-tests-and-ci`

These are advisory Codex skills for repository workflows. They document expected
entry points, contracts, and validation checks, but they do not grant permission
to edit outside the active task scope.

## Orchestrator Guard

The Claude hook in `.claude/settings.json` ran
`.claude/hooks/orchestrator_guard.py` before selected tools. Codex does not have
a repository-local equivalent to Claude's `PreToolUse` hook in this tree, so the
migration keeps the behavior as documentation plus a standalone script:

- `.codex/hooks/orchestrator_guard.policy.json` records the migrated hook intent.
- `.codex/hooks/orchestrator_guard.py` is a Codex-side port that can be invoked
  by an external wrapper or future hook integration.

By default the guard is advisory: it prints the repository contract reminder and
allows execution. If a wrapper invokes the script and sets
`ORCHESTRATOR_GUARD_STRICT=1`, the script blocks monitored tool use unless the
agent role is a delegated subagent role. The role can be set with
`SLIME_AGENT_ROLE` or `CODEX_AGENT_ROLE`.

The enforceable source of repository behavior remains `AGENTS.md` and the active
task delegation scope supplied by the user/root orchestrator.
