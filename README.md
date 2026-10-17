# Agent Recovery Eval

一个可复现的多工作流 Agent 执行、故障恢复与版本回归评测工程。它提供状态化业务工具、受控故障、独立判定器、真实模型工具循环、MCP stdio 接口和回归门禁。核心需要 Python 3.11+。

本版公开完整业务步骤，测量执行与恢复，不单独测量从自然语言发现计划的能力。业务、付款、通知与邀请都在本地模拟。项目适合 1–2 人继续维护；当前定位是可审查的工程原型。

## 1. 快速运行

```bash
python run_demo.py
```

Windows 可以双击 `run_demo.cmd`。每次创建新的 `runs/demo-时间戳/`，运行 256 回合规则基线，并将幂等重试切换为有界重试，展示一次预期的回归阻断。打开打印的 `report.html` 路径。

也可以安装命令行工具：

```bash
python -m pip install .
recovery-eval list
recovery-eval run --variants 4 --seeds 3 --out runs/my-baseline
python -m unittest discover -s tests -v
```

默认矩阵为 4 工作流 × 4 参数变体 × 3 种子 × 8 条件 = 384 个冻结案例；4 个策略共 1,536 回合。参数变体和种子不是新业务领域，也不是独立用户样本。已有结果不会被覆盖，重新运行请换输出路径。

## 2. 测什么

| 工作流 | 步骤 | 独立业务约束 |
| --- | --- | --- |
| 工单处理 `ticket_dispatch` | 创建、分派、解决、通知 | 内容、接收人、关联及顺序 |
| 库存订单 `inventory_order` | 占库、建单、扣款、发货、回执 | 库存容量、金额、收件地址与关联 |
| 会议预约 `meeting_booking` | 占用、建会、邀请、确认 | 房间容量、时间、参会人与依赖 |
| 文档发布 `document_publication` | 提交、检查、审批、发布、通知 | 版本、审核前置、渠道与受众 |

库存和会议流程包含分支依赖，允许多种合法拓扑执行顺序；不是只替换一条线性任务的名字。总共 18 个业务步骤，另有公共 `inspect_step` 查询。所有状态按回合隔离。

条件包括 `clean`、`pre_commit_timeout`、`lost_ack`、`late_commit`、`stale_read`、`duplicate_delivery`、`transient_rate_limit`、`permanent_failure`。评分前结算迟到写入，从状态和追加式账本检查完成、重复、缺失、内容、关联、先后与资源约束。策略声称成功不覆盖判分。

四个手写基线为 `no_retry`、`bounded_retry`、`verify_then_retry`、`idempotent_retry`。稳定幂等键的收益依赖模拟服务的去重契约；永久失败不能靠重试保证完成。

## 3. 实跑证据

发布快照、实际数量和发现见 [实验记录](docs/EXPERIMENTS.md)。完整 JSONL 与模型原始输出在单独的实验记录包中，仓库保留轻量结果快照，避免提交模型权重和大型日志。

规则策略、真实本地模型、测试用适配器和 Prime 静态诊断分别记账。测试通过或脚本通过率不能写成 LLM 成绩。真实小模型样本用于检查闭环，不支持总体模型排名。

## 4. 冻结案例、重放与回归

```bash
recovery-eval matrix --variants 2 --seeds 1 --out runs/cases.jsonl
recovery-eval run --variants 2 --seeds 1 --out runs/review
recovery-eval replay --episodes runs/review/episodes.jsonl --case-id ticket_dispatch-v001-s0-lost_ack --policy bounded_retry
recovery-eval compare --before runs/review/episodes.jsonl --after runs/review/episodes.jsonl --before-agent idempotent_retry --after-agent bounded_retry --out runs/regression.json --fail-on-block
```

`matrix` 导出用于审查的案例清单；`run` 用相同生成参数构建案例。重放读取已保存的案例和执行预算，重新执行确定性策略。真实模型再次生成不是确定性日志重放。

比较必须具有相同案例集合和完整条件哈希。以工作流及业务变体为簇做配对 bootstrap，少于 5 簇返回证据不足。输出 `pass`、`block` 或 `inconclusive`，并逐案例检查重复、缺失和错误成功声明；接口错误留在分母并阻止放行。`pass` 表示这组冻结案例满足声明的退化容忍门禁，不保证零个成功率退化案例，也不表示系统绝对安全。`--fail-on-block` 在阻断时返回 2；不确定结论写入 JSON，由调用方决定如何处理。

## 5. 真实模型执行

模型逐回合输出严格 JSON 动作；runner 真实调用同一个模拟器，再把公共返回交给模型。它不向模型发送故障标签、隐藏调度、判分答案或提交账本。原始生成、token IDs、EOS／长度上限、格式纠正、工具返回与预算结束原因均保留。

```bash
python -m pip install -r requirements-local-model.txt
recovery-eval model --backend local --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --allow-download --max-new-tokens 160 --max-turns 8 --max-tool-calls 12 --out runs/qwen
```

首次需要公开模型下载和 CPU 推理依赖。默认只使用本地缓存；`--allow-download` 才允许首次下载。离线缓存可通过 `--cache-dir` 指定。默认覆盖四个工作流、一个参数变体、一个种子、`clean/lost_ack` 和 `basic/guarded` 两种提示，共 16 回合。一次显式格式纠正计入 8 次模型请求预算，无静默提取或修复。

也支持文本 Chat Completions 的 OpenAI-compatible 端点：

```bash
# Set PRIME_API_KEY in your environment before running.
recovery-eval model --backend api --model YOUR_MODEL --base-url https://YOUR_ENDPOINT/v1 --api-key-env PRIME_API_KEY --out runs/api
```

密钥只从环境变量读取；模型和端点使用自己的配置。本次未调用付费模型 API。调用数和 token 上限限制请求规模，不构成美元硬预算。API 路径测试含 mock，不等于端点实测。

## 6. MCP、Modal 与 Prime Intellect

1. **MCP**：`python -m recovery_eval.mcp_server` 提供真实 stdio 工具 `get_task/execute/finish`，公共完成返回不含评分。已使用真实 SDK 子进程测试，调用方是程序化 client。安装 `requirements-mcp.txt` 后复现，见 [MCP 文档](docs/MCP.md)。
2. **Modal**：可选远程 CPU 规则批处理与本地模型入口，见 [Modal 文档](integrations/MODAL.md)。本轮仅本地 SDK／函数验证，没有启动云任务。
3. **Prime**：原生 Verifiers v1 静态诊断 Taskset，见 [任务包文档](integrations/prime_taskset/README.md)。它读取结算证据判别结果，不是动态恢复环境。本轮没有发布 Hub 或运行托管评测；动态模型可另经兼容端点接入主 runner。

## 7. 结果文件与项目结构

每次运行生成 `episodes.jsonl`、`summary.json`、`metrics.csv`、`example_traces.json`、自包含 `report.html` 和含校验和的 `manifest.json`。`progress.jsonl` 持续保存已完成回合，中断不会把它们变成成功运行。报告展示失败优先的代表轨迹，完整分母在 JSONL。

| 路径 | 作用 |
| --- | --- |
| `recovery_eval/workflows.py`, `engine.py` | 任务契约、真实状态、故障、策略及判分 |
| `recovery_eval/agents.py` | 本地／API 模型和真实工具执行循环 |
| `recovery_eval/regression.py` | 配对校验、簇 bootstrap、回归门禁 |
| `recovery_eval/cli.py`, `artifacts.py` | CLI、运行元数据及可筛选报告 |
| `recovery_eval/mcp_server.py` | MCP stdio 服务 |
| `tests/`, `integrations/` | 核心及可选接口验证 |
| `docs/`, `examples/` | 架构、实验、审查、提交及轻量快照 |

v0.1 的两步实验 API 仍在 `core.py/comparison.py/reporting.py/api_agent.py/local_model.py` 保留，用于兼容旧样本和测试；新命令使用 v0.2 模块，旧结果不合并为当前成绩。

## 8. 继续审查和提交 GitHub

依次阅读 [架构](docs/ARCHITECTURE.md)、[扩展指南](docs/EXTENDING.md)、[Benchmark Card](docs/BENCHMARK_CARD.md)、[审查清单](docs/REVIEW_CHECKLIST.md)、[GitHub 提交步骤](docs/GITHUB_SUBMISSION.md) 和 [PR 正文](PR_DESCRIPTION.md)。[简历写法](docs/RESUME.md) 只使用实测事实；[相关工作](docs/RELATED_WORK.md) 说明已有研究与本项目边界。

代码已推送至 [ltan0136/agent-recovery-eval](https://github.com/ltan0136/agent-recovery-eval)，变更与审查记录见 [PR #1](https://github.com/ltan0136/agent-recovery-eval/pull/1)。[CI 运行 39036962895](https://github.com/ltan0136/agent-recovery-eval/actions/runs/39036962895) 的 Python 3.11／3.12／3.13 核心任务及可选集成任务共四项均已通过。原始本地实验记录、补丁与 Git bundle 保留用于复核。MIT 是本地版本的许可选择，仓库所有者应复核署名与许可；模型和依赖各自保留原许可。
