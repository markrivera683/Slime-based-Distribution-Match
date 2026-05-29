---
name: dlc-submit
description: Use when the user asks to inspect, submit, monitor, fetch logs for, stop, delete, or troubleshoot Aliyun PAI DLC jobs. Before any submit, stop/delete, config, sudo/install, credential, or other cost-incurring operation, present the exact command and get explicit user confirmation.
---

# DLC Submit

## When to Use

Use this skill for Aliyun PAI DLC CLI work: job/quota/pod inspection, PyTorchJob submission, monitoring, log retrieval, stop/delete requests, and troubleshooting scheduling, image, mount, NCCL, or pod failures.

## Safety Rules

- Treat `dlc get`, `dlc logs`, and local file reads as read-only when inputs are clear.
- Get explicit user confirmation before `dlc submit`, `dlc stop`, delete operations, `dlc config`, `sudo`, CLI install/download, credential handling, config edits, or commands that spend GPU/quota.
- Never print, write, or commit AK/SK, API tokens, passwords, or runtime configs containing secrets.
- Do not assume workspace, resource, VPC, image, mount, or path values. Use user-provided values or ask/verify.
- Before submission, show the final command and highlight job name, workspace, resource quota, image, workers, GPUs, mounts, timeout, and command path.

## Required Inputs

- Read-only job lookup: `workspace_id` plus `job_id`, job name, or query filters.
- Logs: `workspace_id`, `job_id`, and `pod_id`; if `pod_id` is missing, ask or inspect job details first.
- Submit: job name, command, worker count, CPU/GPU/memory/shared memory, image, data mounts, `resource_id`, `workspace_id`, region/endpoint, networking fields, priority, and runtime limit.
- Stop/delete: exact `workspace_id` and `job_id`, then explicit confirmation.

## Read-only Discovery

- Locate the CLI with `command -v dlc` or use a user-provided binary path.
- Use read-only commands such as `dlc config --help`, `dlc get quota --workspace_id=<id>`, `dlc get job --workspace_id=<id>`, `dlc get job <job_id> --workspace_id=<id> --show_detail`, and `dlc get pod <job_id> <pod_id>`.
- If command output conflicts with the reference, trust the live CLI output and report the mismatch.

## Submit Workflow

1. Read `references/pai-dlc-cli.md` for CLI syntax and parameter notes.
2. For DSW command-line submission, also read `references/dsw-dlc-submit.md`.
3. Resolve placeholders and ensure `--command` uses absolute paths inside the mounted filesystem.
4. Validate image, mounts, workspace/resource IDs, node/GPU count, timeout, networking, log redirection, and secret handling.
5. Present the final command and risk/cost summary, then wait for explicit confirmation.
6. Submit only after confirmation. Capture the returned job ID and suggest the next monitoring command.

## Monitor/Logs Workflow

- Query job detail first when status, pods, or `pod_id` are unknown.
- For logs, run `dlc logs <job_id> <pod_id>` only after identifying the correct pod.
- Summarize status, pod IDs, events, key log errors, and the next read-only checks.

## Stop/Delete Workflow

- Show the exact stop/delete command and require explicit confirmation.
- Prefer `dlc stop job <job_id> --workspace_id=<id> --force` for stops.
- If delete syntax is unclear, inspect `dlc --help` or `dlc <subcommand> --help` before proposing a command.
- After a confirmed stop/delete, verify with `dlc get job <job_id> --workspace_id=<id> --show_detail`.

## Output Format

- State whether the action is read-only or requires confirmation.
- List exact commands run or proposed.
- Report `workspace_id`, `job_id`, `pod_id`, status, important errors, and next steps.
- Redact secrets and avoid persisting credentials.

## References

- Read `references/pai-dlc-cli.md` for detailed CLI examples, workspace notes, and troubleshooting patterns.
- Read `references/dsw-dlc-submit.md` for DSW setup, current Code/VL defaults, rotatelogs logging, and mount/path preflight checks.
