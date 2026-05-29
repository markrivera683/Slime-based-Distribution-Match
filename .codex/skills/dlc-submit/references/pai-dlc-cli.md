# 阿里云 PAI DLC CLI Reference

> **前置条件**：确保 `dlc` CLI 已安装且有执行权限。优先使用 `command -v dlc` 查找，或由用户提供 `<DLC_BINARY_PATH>`。

## 快速参考

|操作|命令|
|---|---|
|配置 CLI|`dlc config`|
|提交 PyTorchJob|`dlc submit pytorchjob --name=<name> --workers=N ... --command="..."`|
|列出所有任务|`dlc get job --workspace_id=<id>`|
|查询单个任务|`dlc get job <job_id> --workspace_id=<id> --show_detail`|
|查看 Pod 日志|`dlc logs <job_id> <pod_id>`|
|停止任务|`dlc stop job <job_id> --workspace_id=<id> --force`|
|查看 Pod 事件|`dlc get pod <job_id> <pod_id>`|
|查询机器规格|`dlc get specs`|

---

## 1. CLI 配置

### 1.1 安装 DLC CLI

如果尚未安装，可从阿里云官方 OSS 下载：

```Bash
sudo wget -q -O /usr/local/bin/dlc \
    "https://dlc-release.oss-cn-zhangjiakou.aliyuncs.com/console/public/latest/dlc"
sudo chmod +x /usr/local/bin/dlc

```

> 项目环境中可能已有预装的 dlc；执行前用 `command -v dlc` 或用户提供的 `<DLC_BINARY_PATH>` 确认。

### 1.2 配置认证信息

> **安全确认**：安装 CLI、配置凭据或修改认证配置前，必须先向用户展示将执行的命令并取得明确确认。不要输出 AK/SK、token 或配置文件内容。命令行传入 AK/SK 会进入 shell history；优先使用交互式输入或用户确认的安全凭据通道。

DLC CLI 需要配置阿里云 AccessKey。配置会保存在 `~/.dlc/config`，后续命令无需重复传入 AK/SK。

```Bash
# 方式 1：交互式配置（推荐，密码隐藏输入）
dlc config

# 方式 2：仅在用户明确确认且已处理 shell history 风险时使用
dlc config \
    --protocol https \
    --access_id <你的AccessKeyId> \
    --access_key <你的AccessKeySecret> \
    --endpoint 'pai-dlc.cn-wulanchabu.aliyuncs.com' \
    --region cn-wulanchabu

# 验证配置
dlc config --help   # 查看配置命令帮助；不要 cat 配置文件
# 如必须检查 ~/.dlc/config，先取得用户确认，并在展示前脱敏

```

**常用 Endpoint 与 Region 对照表：**

|Region|Endpoint|
|---|---|
|上海|`pai-dlc.cn-shanghai.aliyuncs.com`|
|乌兰察布|`pai-dlc.cn-wulanchabu.aliyuncs.com`|
|杭州|`pai-dlc.cn-hangzhou.aliyuncs.com`|
|北京|`pai-dlc.cn-beijing.aliyuncs.com`|

---

## 2. 提交 PyTorchJob

### 2.1 基础命令

```Bash
dlc submit pytorchjob \
    --name=<任务名称> \
    --command='<启动命令>' \
    --workers=<节点数> \
    --worker_cpu=<每节点CPU核数> \
    --worker_gpu=<每节点GPU数> \
    --worker_memory=<每节点内存> \
    --worker_shared_memory=<每节点共享内存> \
    --worker_image=<镜像地址> \
    --data_source_uris=<数据挂载> \
    --resource_id=<资源配额ID> \
    --workspace_id=<工作空间ID> \
    --vpc_id=<VPC ID> \
    --switch_id=<交换机ID> \
    --security_group_id=<安全组ID> \
    --extended_cidrs=<扩展CIDR> \
    --priority=<优先级>

```

### 2.2 完整示例

```Bash
dlc submit pytorchjob \
    --name=sft_qwen3_8b \
    --command='source /path/to/conda.sh && conda activate myenv && bash run.sh' \
    --workers=2 \
    --worker_cpu=116 \
    --worker_gpu=8 \
    --worker_memory=1800Gi \
    --worker_shared_memory=1800Gi \
    --worker_image=dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/python:3.13.9-gpu-cu129-ubuntu22.04 \
    --data_source_uris='cpfs://cpfs-xxxx.cn-wulanchabu/xxx/::/mnt/data/cpfs,oss://bucket.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/data/oss' \
    --resource_id=quotaev2tl4w6aw0 \
    --workspace_id=240810 \
    --vpc_id=vpc-0jl5rpw5qokp6p2ettip6 \
    --switch_id=vsw-0jlmr9rjzed093yr9c0kz \
    --security_group_id=sg-0jl0pd5qaerdj75wmred \
    --extended_cidrs='10.1.255.0/29,10.1.255.8/29,10.1.16.0/20' \
    --priority=1

```

### 2.3 常用参数说明

|参数|说明|示例|
|---|---|---|
|`--name`|任务名称（唯一标识）|`sft_qwen3_8b_0427`|
|`--workers`|Worker 节点数量|`2`|
|`--worker_cpu`|每节点 CPU 核数|`110`, `116`|
|`--worker_gpu`|每节点 GPU 数量|`8`|
|`--worker_memory`|每节点内存|`1500Gi`, `1800Gi`|
|`--worker_shared_memory`|每节点共享内存|`1500Gi`|
|`--worker_image`|容器镜像|`dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/...`|
|`--data_source_uris`|数据挂载（CPFS/OSS/NAS）|`cpfs://...::/mnt/cpfs,oss://...::/mnt/oss`|
|`--resource_id`|资源配额 ID|`quota1hdkwah70tk`|
|`--workspace_id`|工作空间 ID|`240810`|
|`--vpc_id`|VPC ID|`vpc-...`|
|`--switch_id`|交换机 ID|`vsw-...`|
|`--security_group_id`|安全组 ID|`sg-...`|
|`--extended_cidrs`|扩展 CIDR（多机通信）|`10.1.255.0/29,10.1.255.8/29`|
|`--priority`|任务优先级（1-9，越小越优先）|`1`|
|`--command`|容器启动命令|`bash <CPFS_PROJECT_PATH>/run.sh`|
|`--envs`|环境变量|`-e 'KEY1=val1,KEY2=val2'`|
|`--job_max_running_time_minutes`|最大运行时间（分钟）|`1440`|

### 2.4 数据挂载格式

```Plain Text
<存储类型>://<存储地址>::<挂载路径>

```

**多数据源用逗号分隔：**

```Plain Text
cpfs://cpfs-xxxx.cn-wulanchabu/path/::/mnt/data/cpfs,oss://bucket.oss-cn-wulanchabu-internal.aliyuncs.com/path/::/mnt/data/oss

```

|存储类型|URI 格式|
|---|---|
|CPFS|`cpfs://<cpfs-id>.<region>/<path>::<mount_path>`|
|OSS|`oss://<bucket>.<oss-endpoint>/<path>::<mount_path>`|
|NAS|`nas://<nas-id>.<region>/<path>::<mount_path>`|

---

## 3. 任务生命周期管理

### 3.1 查询任务状态

```Bash
# 列出所有任务（默认最近 10 条）
dlc get job --workspace_id=240810

# 模糊搜索任务名
dlc get job --workspace_id=240810 --display_name="sft_qwen"

# 正则搜索
dlc get job --workspace_id=240810 --display_name_regex="sft.*qwen"

# 按状态过滤
dlc get job --workspace_id=240810 --status=Running
dlc get job --workspace_id=240810 --status=Succeeded
dlc get job --workspace_id=240810 --status=Failed
dlc get job --workspace_id=240810 --status=Stopped

# 分页查询
dlc get job --workspace_id=240810 --page_num=1 --page_size=50

# 查询单个任务详情
dlc get job <job_id> --workspace_id=240810 --show_detail

# 查看任务事件
dlc get job <job_id> --workspace_id=240810 --events

```

### 3.2 停止任务

```Bash
# 停止单个任务
dlc stop job <job_id> --workspace_id=240810 --force

# 停止并立即返回（不等待）
dlc stop job <job_id> --workspace_id=240810 --force --quiet

```

### 3.3 查看日志

```Bash
# 查看 Pod 日志（需要 job_id 和 pod_id）
dlc logs <job_id> <pod_id>

# 限制日志行数
dlc logs <job_id> <pod_id> --max_events_num=5000

# 时间范围过滤
dlc logs <job_id> <pod_id> --start_time="2026-04-27T00:00:00" --end_time="2026-04-27T12:00:00"

```

**获取 pod_id：** 先通过 `dlc get job <job_id> --workspace_id=<id> --show_detail` 查看任务详情，其中包含 pod 列表。

### 3.4 查看 Pod 事件

```Bash
# 查看 Pod 的事件（调度、启动、异常等）
dlc get pod <job_id> <pod_id>

# 限制事件数量
dlc get pod <job_id> <pod_id> --max_events_num=100

```

---

## 4. 项目组常用配置模板

以下是项目组两个工作空间的历史示例配置。执行前必须确认 workspace、resource、VPC、镜像、挂载和路径仍然适用于当前环境。

### 4.1 VL 工作空间默认配置

> `workspace_id=245264`，`resource_id=quota1hdkwah70tk`适用场景：评测任务、轻量级训练（单节点）

```JSON
{
  "dlc": {
    "submit": true,
    "job_name": "sft_qwen3_8b_openthoughts400k",
    "binary": "<DLC_BINARY_PATH>",
    "run_script": "<CPFS_PROJECT_PATH>/examples/on_policy_distillation/reproduce_blog/sft_worker_wrapper.sh",
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

### 4.2 Code 工作空间默认配置

> `workspace_id=240810`，`resource_id=quotaev2tl4w6aw0`适用场景：大规模训练、多机分布式任务（高优先级）

```JSON
{
  "dlc": {
    "submit": true,
    "job_name": "sft_qwen3_8b_openthoughts400k",
    "binary": "<DLC_BINARY_PATH>",
    "run_script": "<CPFS_PROJECT_PATH>/examples/on_policy_distillation/reproduce_blog/sft_worker_wrapper.sh",
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

### 4.3 两个工作空间对比

|配置项|VL 工作空间 (245264)|Code 工作空间 (240810)|
|---|---|---|
|`workspace_id`|`245264`|`240810`|
|`resource_id`|`quota1hdkwah70tk`|`quotaev2tl4w6aw0`|
|`workers`|`1`|`2`|
|`worker_cpu`|`110`|`116`|
|`worker_memory`|`1500Gi`|`1800Gi`|
|`worker_shared_memory`|`1500Gi`|`1800Gi`|
|`priority`|`9`|`1`|
|CPFS 挂载路径|`/mnt/cpfs`|`/mnt/data/cpfs`|
|OSS 挂载路径|`/mnt/oss`|`/mnt/data/oss`|
|典型用途|评测、轻量训练|大规模训练、分布式|

---

## 5. 项目中典型配置模式

### 5.1 配置拆分：dlc_config.json + 业务配置

项目中推荐将 **DLC 调度参数** 与 **业务逻辑配置** 分离：

**config_dlc.json — 仅包含 DLC 参数：**

```JSON
{
  "dlc": {
    "submit": true,
    "job_name": "sft_qwen3_8b_openthoughts400k",
    "binary": "<DLC_BINARY_PATH>",
    "run_script": "<CPFS_PROJECT_PATH>/scripts/worker.sh",
    "workers": 2,
    "worker_gpu": 8,
    "worker_cpu": 116,
    "worker_memory": "1800Gi",
    "worker_shared_memory": "1800Gi",
    "priority": 1,
    "running_timeout": 86400,
    "worker_image": "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/python:3.13.9-gpu-cu129-ubuntu22.04-9e1c8c7e-1764331625",
    "data_source_uris": "cpfs://cpfs-xxx.cn-wulanchabu/xxx/::/mnt/data/cpfs,oss://bucket.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/data/oss",
    "resource_id": "quotaev2tl4w6aw0",
    "workspace_id": "240810",
    "vpc_id": "vpc-0jl5rpw5qokp6p2ettip6",
    "switch_id": "vsw-0jlmr9rjzed093yr9c0kz",
    "security_group_id": "sg-0jl0pd5qaerdj75wmred",
    "extended_cidrs": "10.1.255.0/29,10.1.255.8/29,10.1.16.0/20"
  }
}

```

**业务配置（如 config_eval.json、config_service.json）— 包含模型、任务、环境参数。**

### 5.2 Submitter + Worker 模式

```Plain Text
┌─────────────────┐     submit      ┌─────────────────────────────┐
│  submitter.sh   │ ──────────────► │  PAI DLC PyTorchJob         │
│  (本地机器)      │                 │  ┌─────────────────────┐    │
│                 │                 │  │  Worker Node 0      │    │
│  1. 读取配置     │                 │  │  bash worker.sh     │    │
│  2. 生成 runtime │                 │  └─────────────────────┘    │
│  3. dlc submit   │                 │  ┌─────────────────────┐    │
└─────────────────┘                 │  │  Worker Node 1      │    │
                                    │  │  bash worker.sh     │    │
                                    │  └─────────────────────┘    │
                                    └─────────────────────────────┘

```

**Submitter 脚本核心逻辑：**

1. 读取 `config_dlc.json` 获取调度参数

2. 读取业务配置获取模型/任务参数

3. 生成 `runtime_config.json`（设置 `dlc.submit=false`，注入统一时间戳）

4. 构建 `dlc submit pytorchjob --command="bash worker.sh runtime_config.json"`

**Worker 脚本核心逻辑：**

5. 读取 `runtime_config.json`

6. 激活虚拟环境

7. 启动业务进程（训练 / 评测 / 推理服务）

8. 信号处理：SIGTERM 时优雅退出

### 5.3 项目中的典型使用场景

以下命令是外部项目示例，执行前确认脚本和配置路径存在，并替换为当前仓库/项目的实际路径。

#### 场景 A：lmms-eval 大规模评测提交

```Bash
# 1. 本地调试（单节点）
bash <WORKER_SCRIPT> <EVAL_CONFIG>

# 2. DLC 集群提交（submitter 读取双配置）
bash <SUBMITTER_SCRIPT> <DLC_CONFIG> <EVAL_CONFIG>

```

#### 场景 B：Rollout Service（推理服务）

```Bash
# 提交 rollout 服务到 DLC
bash <ROLLOUT_SUBMIT_SCRIPT> \
    <DLC_CONFIG> \
    <SERVICE_CONFIG>

# 查询服务状态
bash <ROLLOUT_MANAGE_SCRIPT> status <job_name>

# 测试服务连通性
bash <ROLLOUT_MANAGE_SCRIPT> test <endpoints_file>

# 停止服务
bash <ROLLOUT_MANAGE_SCRIPT> stop <job_name>

# 查看日志
bash <ROLLOUT_MANAGE_SCRIPT> logs <job_name>

```

#### 场景 C：OPD 训练任务

```Bash
# 提交 OPD 训练
dlc submit pytorchjob \
    --name=opd_train \
    --command='bash <CPFS_PROJECT_PATH>/run_opd_dlc.sh run_opd_qwen3.sh' \
    --workers=2 \
    --worker_gpu=8 \
    ...

```

---

## 6. 常用镜像

|镜像|用途|
|---|---|
|`dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/python:3.13.9-gpu-cu129-ubuntu22.04-...`|PAI 官方 Python GPU 镜像<br>|
|`pai-wl-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/llm_evaluation/lm-eval:lmms-eval`|lmms-eval 评测镜像<br>|

---

## 7. 故障排查

### 7.1 任务一直处于 Succeeded/Failed 但没看到输出

```Bash
# 检查 Pod 事件
dlc get pod <job_id> <pod_id>

# 查看日志
dlc logs <job_id> <pod_id> --max_events_num=5000

```

### 7.2 多机任务卡死（NCCL 超时）

- 检查 `--extended_cidrs` 是否包含所有节点的 PodCIDR

- 检查安全组是否放通了节点间通信端口

- 在代码中设置 `export NCCL_TIMEOUT=7200`

### 7.3 任务 Pending 长时间不启动

```Bash
# 查看 Pod 事件，检查资源配额是否充足
dlc get pod <job_id> <pod_id>

# 查看资源组配额
dlc get job --workspace_id=<id> --resource_id=<quota_id>

```

### 7.4 镜像拉取失败

- 确认镜像地址可访问（VPC 内网地址需在对应 VPC 内访问）

- 检查是否需要镜像仓库认证（`--image_repo_username` / `--image_repo_password`）

---

## 8. 安全注意事项

1. **AK/SK 不要硬编码在脚本中**：使用 `dlc config` 配置后，CLI 会自动读取 `~/.dlc/config`

2. **敏感信息通过环境变量注入**：如 API Key、Token 等，不要在配置文件中明文存储

3. **镜像安全**：优先使用团队内部构建的镜像，避免使用来源不明的公共镜像

---

## 9. 参考链接

- [PAI DLC 官方文档](https://help.aliyun.com/zh/pai/developer-reference/before-you-begin-1)

- [DLC PyTorchJob 参数说明](https://help.aliyun.com/zh/pai/user-guide/create-a-pytorchjob)

---

## ⚠️ 关键警告（实战踩坑记录）

### 警告 1：`--command` 必须使用绝对路径

DLC Worker 启动时的 **当前工作目录 (CWD) 不是脚本所在目录**。如果 `--command` 中使用相对路径，任务会直接失败：`bash: run.sh: No such file or directory`。

**❌ 错误写法：**

```Bash
--command="bash run_opd_dlc.sh"

```

**✅ 正确写法：**

```Bash
--command="cd <CPFS_PROJECT_PATH>/src/OPD && bash run_opd_dlc.sh"

```

> **惨痛教训**：OPD 任务 `dlc148943uoyreha` 因此失败，修复后重提才成功。

### 警告 2：`dlc stop job` 建议加 `--workspace_id`

如果未配置默认 workspace，`dlc stop job` 可能找不到任务：

```Bash
# 安全写法
dlc stop job <job_id> --workspace_id=<id> --force

```

### 警告 3：`dlc logs` 需要 pod_id

`dlc logs` 需要两个参数：`job_id` 和 `pod_id`。获取 pod_id 的方法：

```Bash
# 步骤 1：查询任务详情
dlc get job <job_id> --workspace_id=<id> --show_detail
# 在返回的 Pods 列表中找到 pod_id

# 步骤 2：查看日志
dlc logs <job_id> <pod_id>

```

### 警告 4：任务状态说明

|状态|含义|
|---|---|
|`Queuing`|排队中，等待资源调度|
|`Running`|运行中|
|`EnvPreparing`|环境准备中（拉取镜像、初始化）|
|`Succeeded`|成功完成|
|`Failed`|失败|
|`Stopped`|被手动停止|

> `Queuing` 超过 5-10 分钟通常意味着 quota 不足，需要检查 `dlc get quota` 或降低节点数。

### 警告 5：`dlc config` 的实际 flag

`dlc config` 子命令自身的 flag 只有 `--password` 和 `--username`。`--access_id` / `--access_key` / `--endpoint` / `--region` 是 **Global Flags**，写在 `dlc config` 后面也能工作，但本质是传给 `dlc` 主命令的。

```Bash
# 仅在用户确认且已处理 shell history 风险时使用
dlc config --protocol https -a <AK> -k <SK> -e <endpoint> -r <region>

```

### 补充命令

```Bash
# 查看资源配额（检查是否有足够 GPU）
dlc get quota --workspace_id=<id>

# 查看所有任务（分页）
dlc get job --workspace_id=<id> --page_num=1 --page_size=50

```

## 10. Agent/Codex 使用约定

本页是 PAI DLC CLI 参考资料，配合 `../SKILL.md` 使用。Codex/agent session 遇到 DLC、PAI DLC、提交任务、查询任务、查看日志、停止任务、quota、Pod 事件、PAI 工作空间配置等请求时，应先遵守 skill 的安全规则，再按需读取本页。

- 只读查询可以直接执行，例如按 job_id 查询详情、按任务名搜索、查看 quota、查看 pod 事件和日志。

- 提交新任务、停止任务、修改调度配置等写操作或有成本/风险的操作，先给出计划和关键命令，等待用户明确 go 后再执行。

- 查询具体任务时，优先要求或推断 workspace_id。Code 工作空间默认 workspace_id 为 240810；VL 工作空间默认 workspace_id 为 245264。

- 查看日志前先用 dlc get job JOB_ID --workspace_id=WORKSPACE_ID --show_detail 获取 pod_id，再执行 dlc logs JOB_ID POD_ID。

- 停止任务建议使用 dlc stop job JOB_ID --workspace_id=WORKSPACE_ID --force，避免默认 workspace 缺失导致找不到任务。

- 如果本页与实际 dlc CLI 输出或项目脚本行为不一致，以实际运行结果为准，并把不一致反馈给用户。

- 不要把 AK/SK、API token 或含密钥的 runtime 配置写入仓库或文档；认证信息只放在本地 dlc config、环境变量或用户确认的安全位置。
