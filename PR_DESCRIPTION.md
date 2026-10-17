# Add multi-workflow Agent recovery evaluation and paired regression gates

原有两步流程难以检查库存、容量、审批依赖和不同业务状态下的恢复问题；仅看完成率，也会漏掉重复副作用和错误的成功声明。本次加入四个状态化工作流及独立业务判定，并将同条件版本比较做成可阻断的命令行门禁。

## Changes

1. 四种工作流、18 个业务步骤、八类正常／故障条件和四个规则基线；库存及会议支持不同合法拓扑顺序。每回合隔离状态，判定前结算迟到提交，再检查内容、依赖、关联、资源、重复及缺失。
2. 本地 Transformers／OpenAI-compatible 文本模型执行循环：严格 JSON 动作、一次显式格式纠正、调用预算、原始输入输出及停止证据。MCP 通过真实 stdio 提供公开工具，评分写入 evaluator 结果文件。
3. 完整条件哈希配对、业务变体簇 bootstrap、逐案例安全退化检查及 pass／block／inconclusive。案例缺失、重复或条件不匹配时拒绝比较。
4. 不覆盖已有证据的 CLI、确定性策略重放、断点进度、保留失败的回合级续跑及并发锁、文件校验和、可筛选 HTML 报告。保留 v0.1 模块以兼容旧样本；当前 CLI 使用 v0.2。
5. 更新 Modal 的规则／可选模型入口与 Prime Verifiers v1 静态诊断包，补齐架构、Benchmark Card、贡献、安全、简历、实验与 GitHub 提交文档，以及 CI 配置。

## Validation

1. 本地执行核心及可选 MCP 测试 **128 项通过**，Prime 独立包 **9 项通过**，合计 **137 项**；MCP 包含真实客户端／服务端子进程传输，不是模型成绩。
2. 正式执行 **6,144 回合规则基线**，原始状态／账本／哈希复核一致。幂等策略成功 1,344/1,536（87.5%），其余 192 例为永久失败，无重复或错误成功声明。
3. 有意从幂等策略切换为有界重试，在 1,536 对回合、48 个业务簇中检测出 **744 个退化、0 改善**，门禁阻断；相同策略自比较通过。该对照不是一次真实模型版本升级。
4. 两个固定 revision 的本地小模型完成32个回合，业务成功0/32；可观察格式失败、查询循环、部分写入及重复效果。Qwen原进程中断后保留7个完整失败并续跑9例，未知在途工作量另行记录。完整成绩、原始输出、格式与截断计数见 [实验记录](docs/EXPERIMENTS.md)。规则与模型不合并解释为同类分数。
5. Modal 真实 SDK `.local()` 规则函数执行、Prime 原生 Taskset 加载／reward／dry-run、本地 wheel 安装和 CLI smoke 均已检查。日志与快照随审查材料提供。GitHub Actions 已配置，本轮未在线触发。

```bash
python run_demo.py
python -m unittest discover -s tests -v
python scripts/reproduce_release.py --out runs/reproduction
python scripts/verify_artifacts.py runs/reproduction/scripted
```

MCP 的六项测试需安装 `requirements-mcp.txt`；无此依赖会明确跳过。Prime 的九项测试需另装其任务包。真实模型需要可选推理依赖和权重缓存，未把下载计成模型推理。

## Scope and review

1. 本版提供公开计划，主要测执行与恢复；只有四种合成工作流，参数变体与调度种子不能扩写为独立企业任务。
2. 幂等收益依赖服务端契约；永久失败可能不可达。模型小样本及低生成预算不支持总体排名，失败及截断不排除出分母。
3. Prime 包是结算状态诊断，不是动态工具恢复环境。Modal／Prime 云任务、外部付费模型 API、Hub 发布、生产服务和多租户隔离均未验证。
4. MIT 为本地版本许可选择，发布者需确认署名与适用范围；第三方模型和依赖保留各自许可。
5. 建议按 [审查清单](docs/REVIEW_CHECKLIST.md) 复核原始轨迹。当前仅准备本地分支、提交、补丁、bundle 和 PR 正文；尚未指定 GitHub 目标仓库，没有创建在线 PR。若贡献上游，应按其抽象缩小并适配改动。
