# RHO：智能体 Harness 演化与隔离评测

本目录收录 RHO（Retrospective Harness Optimization）的源码，以及我们为 Codex CLI、DeepSeek 和 SWE-bench Pro 增加的容器隔离、请求预算、重试及评分适配。

RHO 从解题轨迹中诊断问题，生成多个候选 harness，再通过成对评审选出候选。Harness 可以包含 Markdown 指导、技能说明和辅助脚本；当前研究流程不修改 Codex 的 Rust 源码，也不训练模型权重。

**本目录只发布代码、说明和安装所需的项目元数据。数据集、题目划分文件、真实运行配置、API 密钥、登录状态、历史轨迹、候选产物和评测结果均不随仓库发布。运行前需在本机准备相应输入。**

上游：[wbopan/retro-harness](https://github.com/wbopan/retro-harness)。来源及本地适配说明见 [UPSTREAM.md](UPSTREAM.md)。

## 1. 目录结构

```text
rho/
├── README.md                         # 本说明
├── UPSTREAM.md                       # 来源、版本及适配范围
├── LICENSE                           # RHO 上游 MIT 许可证
├── CITATION.cff
├── pyproject.toml                    # Python 项目与依赖声明
├── uv.lock                           # 依赖锁文件
├── src/rho/
│   ├── cli.py                        # rho 命令入口
│   ├── loop.py                       # 演化主循环
│   ├── agent/                        # Codex、隔离容器、预算与重试
│   ├── datasets/                     # 数据集读取和评分适配代码
│   ├── orchestrators/                # 解题、诊断、成对评审
│   ├── strategies/                   # 候选生成策略
│   ├── selection/                    # 核心集选择
│   ├── stores/                       # Harness、轨迹存储代码
│   └── local_settings.py             # 从环境变量读取本机路径
├── scripts/
│   ├── run-swebench-matched.py        # 匹配请求预算的完整实验入口
│   ├── run-swebench-v1-evolution.py   # 单轮演化
│   ├── run-swebench-v1-evaluation.py  # Baseline / final 测试
│   ├── check-swebench-container-environments.py
│   ├── container-runtime/            # 容器初始化、依赖代理辅助程序
│   ├── rho-ds.py                     # DeepSeek 启动适配
│   └── ...                          # 审计、恢复和分析脚本
├── codex/retrospection.py             # 另一入口：项目原生技能演化
└── tests/                            # 测试代码，不包含数据文件及轨迹
```

运行后生成的 `runs/`、本地 `configs/`、`data/`、`.venv/` 等已加入忽略规则。`src/rho/datasets/` 是数据集适配源码，不是数据集本身。

## 2. 安装环境

Python 版本要求：**3.11、3.12 或 3.13**。容器实验使用 Linux 和 Docker，需要当前用户能够运行 Docker 命令。解题还需要可用的 Codex CLI 和模型访问权限。

在 MetaForge 仓库根目录执行：

```bash
cd rho
uv sync --extra dev --extra swebench-pro
uv run rho --help
```

也可以使用 Python 虚拟环境安装：

```bash
cd rho
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,swebench-pro]'
.venv/bin/rho --help
```

后续命令均从 `MetaForge/rho/` 执行。安装依赖会访问软件源；下文的实际解题、诊断和评审会调用模型 API。

## 3. 本机配置：不提交到 Git

推荐将配置和数据放在仓库外，通过环境变量指定：

```bash
export RHO_CODEX_BINARY="/absolute/path/to/codex"
export RHO_CODEX_CONFIG="/absolute/path/to/private/codex.dashscope.toml"
export RHO_DATASET_PATH="/absolute/path/to/private/swebench-pro/test.jsonl"
```

容器实验应指定能够在题目镜像中直接执行的 **Linux Codex 二进制文件**。我们的匹配预算脚本按 `0.163.0` 记录实验版本；请提供对应版本，避免元数据与实际版本不一致。不要将 npm 的 JavaScript 启动包装脚本当成容器中的独立二进制文件。

通过环境变量安全设置 `DASHSCOPE_API_KEY`；也可以设置 `RHO_DASHSCOPE_KEY_FILE` 指向仓库外的密钥文件。不要把真实密钥写入 README、命令示例或提交记录。启动器不会再自动搜索父目录的 `api_info.txt`。

下列内容是**配置格式说明**，请自行创建在 `RHO_CODEX_CONFIG` 指定的本地文件中；仓库不提供真实运行配置：

```toml
model = "deepseek-v4-flash-0731"
model_provider = "rho_dashscope"
model_reasoning_effort = "high"

[model_providers.rho_dashscope]
name = "DashScope"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
env_key = "DASHSCOPE_API_KEY"
wire_api = "responses"
requires_openai_auth = false
```

以上模型与接口对应本地实验适配，运行前应确认自己的服务权限与接口支持情况。如果要替换模型，通用 `rho` 命令可以通过参数配置；专用 SWE-bench 脚本仍保留本实验的模型和版本设置，需要同时检查演化、评测及其记录，不能只改一处。

可选环境变量：

| 变量 | 用途 |
|---|---|
| `RHO_CODEX_BINARY` | 显式指定 Codex 可执行文件 |
| `RHO_CODEX_CONFIG` | 显式指定仓库外的模型配置 |
| `RHO_DATASET_PATH` | 专用 SWE-bench 脚本读取的数据文件 |
| `RHO_HTTP_PROXY` | 可选，覆盖当前进程的 HTTP/HTTPS 代理 |
| `RHO_SPLITS_DIR` | 历史核心集选择脚本读取划分清单的目录 |
| `DASHSCOPE_API_KEY` | 模型认证，由调用者在环境中提供 |
| `RHO_DASHSCOPE_KEY_FILE` | 可选，仓库外密钥文件路径 |
| `RHO_DASHSCOPE_KEY_INDEX` | 可选，密钥文件中的条目下标，默认 0 |

未设置 `RHO_HTTP_PROXY` 时保留调用者已有的代理设置，不再强制使用原服务器的代理地址。

## 4. 通用 RHO 入口

先查看参数：

```bash
uv run rho evolve --help
uv run rho select --help
uv run rho solve --help
uv run rho grade --help
```

例如，使用自己准备的数据和核心题选择文件运行一轮研究流程：

```bash
uv run rho evolve \
  --dataset "swebench-pro:$RHO_DATASET_PATH" \
  --selection-json /absolute/path/to/private/selection.json \
  --rounds 1 \
  --max-evolve-tasks 10 \
  --optimize-strategy diagnosis \
  --optimize-samples 3 \
  --max-grading-tasks 0 \
  --cache off \
  --codex-config "$RHO_CODEX_CONFIG" \
  --codex-binary "$RHO_CODEX_BINARY" \
  --model deepseek-v4-flash-0731 \
  --codex-concurrency 8 \
  --run-dir runs/manual-evolution
```

这是通用研究入口，**不自动启用本地新增的“每次 agent 调用进入独立题目容器”和 170 次请求限制**。若要运行本文所述的隔离实验，使用下一节的专用入口。

## 5. 运行隔离的 170 次请求实验

### 5.1 准备本地输入

为每次实验创建新的目录，例如：

```text
runs/my-170-run/                     # 整个目录不提交到 Git
├── selection.json                  # 固定的 10 道核心题 ID
├── split_manifest.json             # 数据摘要、划分种子和题目 ID 清单
├── images.json                     # 核心 10 题 + 测试 100 题的本地镜像映射
└── grading_environment.json         # 仅评分阶段使用的环境修复设置
```

这些输入需要由你在本地准备。源数据至少要能提供 100 道演化池题目和 100 道不重叠测试题；专用脚本验证划分与 seed=0 的数据加载结果一致。

关键结构如下，尖括号是占位符，不是可直接执行的真实数据：

```text
selection.json
  selected_task_ids: 10 个不重复的核心题 ID
  provenance.judge_model: 可选，记录当初选择核心题时的模型

split_manifest.json
  split_seed: 0
  verified_local_source.sha256: RHO_DATASET_PATH 文件的 SHA-256
  groups.evolution_pool_100: 按数据加载器顺序列出的 100 道演化池题目
  groups.evaluation_100: 按数据加载器顺序列出的 100 道测试题目

images.json
  <task_id>:
    image_id: 本地 Docker 镜像的完整 sha256 ID
    base_commit: 当前题目的基准提交

grading_environment.json
  <task_id>: 该题的评分环境修复参数
```

`grading_environment.json` 没有额外设置时可为 `{}`。需要限制测试工作进程、补充系统头文件或提供静态依赖时，使用已核查的评分环境设置，并对两组保持一致。修复参数的实现见 `src/rho/datasets/swebench_pro/grading_environment.py`。若设置中引用静态资源路径，资源文件也必须在本机存在；这些文件只提供给评分流程。

如果已有经过核查的输入，可以复制这四个文件到新目录；不要复制旧的 `evolution/`、`heldout/`、轨迹、检查点或候选 harness。只有核心题 ID 与划分可以复用，运行产物从头生成。

检查本机权限和镜像：

```bash
docker info
"$RHO_CODEX_BINARY" --version
```

本流程使用本地固定镜像，不会替你自动下载缺失镜像。启动器会核对所需镜像是否存在，并重新执行环境预检。

### 5.2 启动完整流程

```bash
uv run python scripts/run-swebench-matched.py \
  --run-root "$PWD/runs/my-170-run" \
  --max-model-requests 170
```

也可以将日志保存在本地并后台运行：

```bash
nohup .venv/bin/python scripts/run-swebench-matched.py \
  --run-root "$PWD/runs/my-170-run" \
  --max-model-requests 170 \
  > runs/my-170-run/pipeline.log 2>&1 &
```

启动前先完成安装和输入准备，且使用一个没有开始过演化的新目录。启动器会拒绝直接覆盖已有演化实验；中断后的恢复需要核对本次目录内的检查点，不能通过重置预算来重新抽样。

### 5.3 自动执行顺序

1. 检查核心集 10 个环境；通过后启动演化，同时准备测试集 100 个环境。
2. 每道核心题生成 3 条 baseline 轨迹，共 30 条。
3. 对每道题进行诊断，共 10 次诊断。
4. 生成 3 个候选 harness。
5. 每个候选重新完成 10 道核心题，共 30 条候选轨迹。
6. 完成 30 组成对评审，按平均偏好分选择候选；未获得正分改进时保留原 harness。
7. 在 100 道测试题上运行 baseline，再运行最终选用的 harness，各生成新轨迹并正式评分。

默认参数：

| 项目 | 设置 |
|---|---|
| 解题、诊断、优化、成对评审模型 | 专用脚本使用 DeepSeek V4 Flash |
| 演化轮数 | 1 |
| 初始解题轨迹 | 10 题 × 3 条 |
| 候选数 | 3 |
| 演化解题上限 | 每条独立轨迹 170 次模型请求 |
| 最终测试解题上限 | 每题、每组 170 次模型请求 |
| 解题总时长上限 | 不设置 |
| 诊断、生成候选、成对评审 | 保留时间限制，不使用 170 次解题限制 |
| 演化初始并发 | 8；检测到限流时可自动降低 |
| 最终测试并发 | 8 路解题、8 路评分，两个线程池独立 |
| 缓存 | 关闭 |

这里的 steps 指**实际发出的模型请求**，包括 API 重试和上下文压缩，不是 Shell 命令数。初始的 3 条样本独立计数，同一条样本的重试沿用预算。达到上限后保留已有补丁，作为本次解题结果，不通过无限补跑绕过上限。

成对偏好分采用 `-10` 到 `+10`；它不是官方测试通过率。成对评审必须输出有效结果，无效输出会重试。最终评分在独立评分环境中进行。

### 5.4 查看进度

```bash
cat runs/my-170-run/pipeline_status.json
cat runs/my-170-run/environment_preflight/status.json
cat runs/my-170-run/evolution/concurrency.json
tail -f runs/my-170-run/evolution.log
```

阶段结束后再查看相应文件：

```bash
cat runs/my-170-run/evolution/reports/summary.json
cat runs/my-170-run/heldout/status.json
cat runs/my-170-run/heldout/summary.json
```

总体状态文件中的 8 路是配置值，实际演化准入并发以 `evolution/concurrency.json` 为准。每次执行的隔离记录、请求计数和重试状态分别保存在本次目录的 `runtime/calls/`、`runtime/budgets/`、`retry_queue/` 中。

## 6. 隔离范围

每次 agent 调用使用新的题目容器。容器不挂载宿主机工作目录、实验目录、数据集文件或 Docker socket；通过 Unix socket 代理访问模型和受控依赖下载服务。代码在容器中准备单一基准快照，并检查修复提交不可访问。

依赖下载仍被允许，因此“隔离宿主机和历史实验”不等于“无法获取任何依赖信息”。本实现不声称所有第三方依赖版本都被固定为白名单。题目解答站点、目标项目下载和普通直连网络受到现有代理与容器策略限制，具体规则见 `src/rho/agent/dependency_proxy.py`。

## 7. 其他入口和脚本

- `codex/retrospection.py`：基于项目已有 Codex 会话，演化 `AGENTS.md`、`.agents/skills/` 和辅助脚本；与本页的基准研究流程不同。使用前查看 `uv run python codex/retrospection.py --help`。
- `scripts/select-swebench-v1-coreset.py`：保留的核心集选择脚本，需要本地划分清单等输入；它的探测不等于本页的隔离容器解题流程。
- `scripts/resume-*`、`continue-clean-heldout.py`、`regrade-clean-environment.py` 等：保留的历史恢复与分析工具，可能含特定运行目录、服务名及任务条件。不要直接把它们当成任意新实验的一键入口，使用前检查参数与路径。

本次迁移不包含前端构建工程、论文站点、数据文件或测试数据快照。核心演化及容器运行代码不依赖这些发布内容；部分上游测试需要另外准备测试夹具，不宜直接把整个测试目录当成全部可运行的测试集。

## 8. 不调用模型的验证

下面的测试覆盖预算隔离、重试、容器输入、路径配置及评分环境修复；不运行实际基准任务：

```bash
uv run pytest -o addopts= -q \
  tests/hermetic/test_local_settings.py \
  tests/hermetic/test_container_codex.py \
  tests/hermetic/test_execution_retry_queue.py \
  tests/hermetic/test_request_budget.py \
  tests/hermetic/test_evaluation_resume.py \
  tests/hermetic/test_grading_environment.py
```

`-o addopts=` 用于覆盖上游默认的高并发 pytest 参数。不要在未检查配置时启动真实模型测试或完整基准测试。

## 9. 许可证

本目录保留 RHO 上游的 [MIT 许可证](LICENSE) 和 [引用信息](CITATION.cff)。MetaForge 根目录的许可证不能替代本目录所附的上游版权与许可说明。
