# DSW-Based DLC Submission Reference

Use this reference when the user wants to submit Aliyun PAI DLC jobs from a DSW instance instead of the console. Follow the safety rules in `../SKILL.md`: do not run `dlc config`, install tools, or submit jobs without showing the exact command and receiving explicit confirmation.

## DSW Setup Flow

1. Start a DSW instance. CPU is usually enough for submitting jobs.
2. Ensure a DLC CLI binary is available. Prefer `command -v dlc`; otherwise use the user-provided binary path or download the Linux client from the official PAI DLC docs.
3. Add execute permission if needed:

```bash
chmod +x /path/to/dlc
```

4. Configure authentication once per DSW home directory. Use placeholders in docs and commands; never record real AK/SK in repository files or final answers:

```bash
/path/to/dlc config --protocol https \
  --access_id <ACCESS_KEY_ID> \
  --access_key <ACCESS_KEY_SECRET> \
  --endpoint 'pai-dlc.cn-wulanchabu.aliyuncs.com' \
  --region cn-wulanchabu
```

The CLI writes config under `~/.dlc/config`. Verify only the presence/shape of the config; redact secrets.

## Submit Command Template

Use a multiline command for readability. Replace all paths with paths that exist inside the DLC worker container after `--data_source_uris` mounts.

```bash
/path/to/dlc submit pytorchjob \
  --name=<job_name> \
  --command='<startup_command>' \
  --data_source_uris='<mount_uris>' \
  --resource_id=<quota_id> \
  --workspace_id=<workspace_id> \
  --vpc_id=<vpc_id> \
  --switch_id=<switch_id> \
  --security_group_id=<security_group_id> \
  --priority=<priority> \
  --running_timeout=<seconds> \
  --extended_cidrs='<cidr_list>' \
  --workers=<node_count> \
  --worker_image='<image_path_not_image_id>' \
  --worker_cpu=<cpu_per_node> \
  --worker_memory=<memory_per_node> \
  --worker_shared_memory=<shared_memory_per_node> \
  --worker_gpu=<gpu_per_node>
```

`--worker_image` must be the image path copied from the PAI image page, not just an image ID.

## Log Redirection Pattern

For training commands, preserve DLC console output while writing rotating logs to CPFS/OSS-mounted storage. Ensure the log directory exists first. On Ubuntu/Debian, `rotatelogs` is commonly provided by `apache2-bin`; some existing jobs install `apache2-utils`, so check the image if the package name differs.

```bash
bash -c 'apt update && apt install -y apache2-bin && \
mkdir -p /mnt/data/cpfs/<user_or_project>/dlc_logs/<job_name> && \
bash /mnt/data/cpfs/<user_or_project>/<repo>/exper_scripts/main_test/run.sh 2>&1 | \
tee >(rotatelogs -l /mnt/data/cpfs/<user_or_project>/dlc_logs/<job_name>/$(date +%Y-%m-%d-%H-%M-%S).log 10M)'
```

Notes:

- `2>&1` combines stderr and stdout.
- `tee` keeps logs visible in DLC and writes them to files.
- `rotatelogs -l <prefix>.log 10M` rotates files at about 10 MB.
- Avoid putting the log directory under ephemeral container-only paths.

## Stale Path and Mount Preflight

If logs show `bash: <script>: No such file or directory`, the job usually failed before Python, Ray, Slime, Megatron, OPD, or training logic started. Treat it as a DLC path/mount/job-spec issue first.

Common causes:

- The local path, such as `/mnt/data/distribution-matching-slime/...`, is not mounted inside the DLC container.
- The job is retrying an old PyTorchJob spec after the script was created or renamed.
- The command uses a DSW/local path while DLC mounted CPFS/OSS under a different path, such as `/mnt/cpfs`, `/mnt/data/cpfs`, `/mnt/oss`, or `/mnt/data/oss`.
- The worker starts outside the repo root, so relative paths like `bash run.sh` or `./requirements.txt` fail.

Before launching expensive jobs, add a temporary preflight block to `--command`:

```bash
bash -c 'set -euxo pipefail
pwd
hostname
date
ls -la /mnt/cpfs || true
ls -la /mnt/data/cpfs || true
ls -la /mnt/oss || true
ls -la /mnt/data/oss || true
ls -la /mnt/data/cpfs/<user_or_project>/<repo>/exper_scripts/main_test/ || true
test -f /mnt/data/cpfs/<user_or_project>/<repo>/exper_scripts/main_test/run.sh
bash /mnt/data/cpfs/<user_or_project>/<repo>/exper_scripts/main_test/run.sh'
```

If the preflight fails, fix `--data_source_uris`, the script path, or resubmit a fresh job instead of relying on retries from the stale job.

## Current Workspace Defaults

These are team defaults from recent notes. They are examples, not assumptions: always confirm workspace, quota, image, mounts, priority, and run script with the user before submission.

### Code Workspace

```json
{
  "dlc": {
    "submit": true,
    "job_name": "sft_qwen3_8b_openthoughts400k",
    "binary": "/mnt/cpfs/yangyicun/dlc",
    "run_script": "/mnt/cpfs/yangyicun/slime/examples/on_policy_distillation/reproduce_blog/sft_worker_wrapper.sh",
    "workers": 2,
    "worker_gpu": 8,
    "worker_cpu": 116,
    "worker_memory": "1800Gi",
    "worker_shared_memory": "1800Gi",
    "priority": 1,
    "running_timeout": 86400,
    "worker_image": "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/python:3.13.9-gpu-cu129-ubuntu22.04-9e1c8c7e-1764331625",
    "data_source_uris": "cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/data/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/data/oss",
    "resource_id": "quotaev2tl4w6aw0",
    "workspace_id": "240810",
    "vpc_id": "vpc-0jl5rpw5qokp6p2ettip6",
    "switch_id": "vsw-0jlmr9rjzed093yr9c0kz",
    "security_group_id": "sg-0jl0pd5qaerdj75wmred",
    "extended_cidrs": "10.1.255.0/29,10.1.255.8/29,10.1.16.0/20"
  }
}
```

### VL Workspace

```json
{
  "dlc": {
    "submit": true,
    "job_name": "sft_qwen3_8b_openthoughts400k",
    "binary": "/mnt/cpfs/yangyicun/dlc",
    "run_script": "/mnt/cpfs/yangyicun/slime/examples/on_policy_distillation/reproduce_blog/sft_worker_wrapper.sh",
    "workers": 1,
    "worker_gpu": 8,
    "worker_cpu": 110,
    "worker_memory": "1500Gi",
    "worker_shared_memory": "1500Gi",
    "priority": 9,
    "running_timeout": 86400,
    "worker_image": "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/python:3.13.9-gpu-cu129-ubuntu22.04-9e1c8c7e-1764331625",
    "data_source_uris": "cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss",
    "resource_id": "quota1hdkwah70tk",
    "workspace_id": "245264",
    "vpc_id": "vpc-0jl5rpw5qokp6p2ettip6",
    "switch_id": "vsw-0jlmr9rjzed093yr9c0kz",
    "security_group_id": "sg-0jl0pd5qaerdj75wmred",
    "extended_cidrs": "10.1.255.0/29,10.1.255.8/29,10.1.16.0/20"
  }
}
```

## Lazy DSW Environment Script Guidance

If the user asks for a "lazy" DSW setup script, summarize or create a small maintained script rather than pasting a huge personal bootstrap into the skill. Include only generic, reusable behavior:

- Configure APT mirror and install basic tools such as `screen`.
- Install or reuse Miniconda or `uv` under a persistent CPFS path.
- Configure public caches for Hugging Face, pip, and uv; set mirror endpoints where appropriate.
- Persist optional HF token, DLC AK/SK, and API keys outside the repo with restrictive permissions.
- Add SSH public keys only from user-provided input; never embed a fixed personal key.
- Install Codex/Gemini/Node helpers only if the user explicitly wants them.
- Download/configure DLC CLI for the Wulanchabu endpoint after explicit confirmation.

Security constraints:

- Do not commit `.env_init_config`, `~/.dlc/config`, API keys, HF tokens, private keys, or personal SSH public keys.
- When documenting the script, use placeholders for credentials and personal paths.
- Prefer idempotent checks (`command -v`, directory existence, `grep -q`) so DSW restarts can rerun the script safely.
