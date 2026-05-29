#!/usr/bin/env python3
"""Standalone Codex-side port of the repository orchestrator guard.

Codex does not automatically execute repository-local PreToolUse hooks from this
directory. This script preserves the migrated Claude hook behavior for external
wrappers or future hook integrations that pass a JSON payload on stdin with a
``tool_name`` field.

By default the guard is advisory so delegated subagents are not accidentally
blocked by a shared repository policy. Set ORCHESTRATOR_GUARD_STRICT=1 to block
direct code-inspection and code-editing tools unless SLIME_AGENT_ROLE or
CODEX_AGENT_ROLE is set to subagent, investigator, implementer, implementation,
verifier, or tester.
"""

from __future__ import annotations

import json
import os
import sys


MONITORED_TOOLS = {
    "Read",
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
    "Bash",
    "Grep",
    "Glob",
    "LS",
}

SUBAGENT_ROLES = {
    "subagent",
    "investigator",
    "implementer",
    "implementation",
    "verifier",
    "tester",
}

NOTICE = (
    "Repository contract reminder: the root orchestrator should talk with the "
    "user and verify assumptions, not inspect or write code directly. Delegate "
    "code inspection/editing to multiple scoped subagents."
)


def _agent_role() -> str:
    return (
        os.environ.get("SLIME_AGENT_ROLE")
        or os.environ.get("CODEX_AGENT_ROLE")
        or "orchestrator"
    ).strip().lower()


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}

    tool_name = str(payload.get("tool_name") or "")
    if tool_name not in MONITORED_TOOLS:
        return 0

    strict = os.environ.get("ORCHESTRATOR_GUARD_STRICT") == "1"
    role = _agent_role()

    if not strict or role in SUBAGENT_ROLES:
        print(NOTICE, file=sys.stderr)
        return 0

    print(f"Blocked {tool_name}: {NOTICE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
