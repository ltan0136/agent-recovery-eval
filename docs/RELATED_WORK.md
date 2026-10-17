# 相关工作与接口来源

查阅日期：2026-10-17。以下链接直接指向论文、作者仓库或平台官方文档。核对页面与仓库文档不等于复现其论文实验；本仓库的实际成绩只来自自己的运行记录。

## 1. 与故障恢复最直接相关的工作

| 工作与核实来源 | 时间或状态 | 已有内容 | 本项目的定位 |
| --- | --- | --- | --- |
| [UndoBench 论文](https://arxiv.org/abs/2610.05622)；[作者仓库](https://github.com/tradertanmay/undobench) | arXiv 首发 2026-10-04；本次按预印本引用，未确认正式录用信息 | 区分正常任务能力与故障恢复能力，使用同种子正常／故障对照，并检查提交效果历史及环境状态 | 本仓库提供较小、易复现的多工作流执行工程和版本门禁；没有复现其完整任务、模型矩阵或条件恢复指标，不能称为替代或超越 UndoBench |
| [Where Does Exactly-Once Live?](https://arxiv.org/abs/2609.29095)；[论文 HTML](https://arxiv.org/html/2609.29095v1) | arXiv 首发 2026-09-24；本次按预印本引用 | 分离模型、执行框架与工具契约对重复副作用的影响；研究丢响应、延迟提交、重复投递与幂等键 | 本仓库用相同问题意识检查恢复，保留账本并显式比较幂等契约；不把幂等收益当成新算法或模型智能证据 |
| [mit-hun-k/faultbench](https://github.com/mit-hun-k/faultbench) | 仓库页面可访问；README 自述 pre-alpha；未据此推定正式发布日期或维护承诺 | 通过 YAML 声明状态化模拟服务，经 MCP 提供工具、注入故障、记录调用，并用 pytest 断言终态 | 本仓库使用自己的四类显式状态与依赖约束、冻结条件配对和回归门禁；没有实现其完整 YAML 世界语言，也不声称首次把故障注入接到 MCP |

对两篇论文的核实包括 arXiv 标题、首发时间、摘要与可用正文。UndoBench 的代码地址由论文页面直接链接。Exactly-Once 论文虽讨论发布基准与适配器，本次未独立确认作者代码仓库地址，因此不补写猜测的 GitHub 链接。

`faultbench` 不是因账户名含某些字符就可归属给某个大学或机构；此处只使用其实际 GitHub 所有者名称。其 README 明确给出功能与 pre-alpha 状态，本项目不替它背书生产可用性。

## 2. 更早的状态化 Agent 评测依据

[τ-bench 论文](https://arxiv.org/abs/2406.12045) 首发于 **2024-06-17**，[官方仓库](https://github.com/sierra-research/tau-bench) 由 Sierra Research 发布。它评估用户、Agent、领域工具和政策约束间的交互，使用会话结束后的数据库状态检查目标，并讨论跨多次试验的一致性。

这里借鉴的是“以业务状态判断结果”的成熟思路。本仓库没有实现 τ-bench 的用户模拟和完整政策交互，也没有将自己的重复回合指标命名为它的 `pass^k`。官方旧仓库当前提示任务已较旧，并指向[后续仓库](https://github.com/sierra-research/tau2-bench)；如要使用其数据做新实验，应在后续仓库固定版本并重新检查任务修复与许可。

## 3. 平台与协议接口来源

| 来源 | 官方入口 | 本项目如何使用 | 不应声称的内容 |
| --- | --- | --- | --- |
| Modal 批处理 | [Batch Processing](https://modal.com/docs/guide/batch-processing)；[Modal 文档](https://modal.com/docs) | 为可选远程批处理提供接口依据；具体支持范围看 [Modal 集成说明](../integrations/MODAL.md) | SDK 加载、本地执行或 CLI 帮助检查不等于云端任务成功，也不构成费用硬上限 |
| Prime Intellect Verifiers | [官方仓库](https://github.com/PrimeIntellect-ai/verifiers)；[Verifiers v1 Tasksets](https://docs.primeintellect.ai/verifiers/v1/tasksets) | Taskset 定义任务、数据与评分；Toolset 通过受支持 harness 暴露 MCP 工具；具体接入看集成目录 | 旧静态诊断任务不能算成动态恢复结果；包加载与 dry-run 不等于 Hub 发布或托管推理成功 |
| MCP Python SDK | [官方仓库](https://github.com/modelcontextprotocol/python-sdk) | 通过真实 stdio 客户端／服务端验证公共工具传输；具体版本与命令见 [MCP 说明](MCP.md) | 传输测试通过不证明模型能力，也不表示生产级隔离、多租户或远程部署已经完成 |

平台文档会变化；安装与复现使用本仓库声明的固定依赖和实际结果元数据。若上游接口改变，先更新适配及测试，再更改版本约束，不用旧运行成绩证明新接口仍有效。

## 4. 贡献定位与新颖性边界

本仓库的贡献是一个可审查的工程组合：不同状态化工作流、受控故障、公开观察边界、独立业务判定、原始轨迹、真实模型动作循环、MCP 传输和严格配对的版本门禁。相关组成思想已有先行工作，当前没有证据支持“首创通用恢复 Benchmark”或“提出新恢复算法”的描述。

本地小模型运行用于检验从推理到工具执行再到判定的闭环；不是前沿模型排名，不是与上述论文直接可比的复现。更大的样本量、更多标题或简单加入置信区间，也不能单独构成研究创新。

如向上述或其他上游贡献，需要先确定目标仓库、确认维护者需求、遵守其架构与许可，再抽取可独立验证的小范围改动。当前独立仓库不能未经适配就作为整个上游 PR 提交。参见 [GitHub 提交说明](GITHUB_SUBMISSION.md)。

## 5. 许可与归属

1. 引用论文或仓库不代表本项目获得其所有代码、数据和模型的统一授权。
2. UndoBench 仓库分别列出软件及 SDK 的 Apache-2.0 与基准材料的 CC BY 4.0；复用时应核查具体文件所在范围。
3. `faultbench` 页面列为 Apache-2.0，τ-bench 页面列为 MIT；迁移代码或数据前仍应检查目标版本 LICENSE 与文件说明。
4. 本项目的本地 MIT 许可选择不能覆盖第三方材料；直接复用内容必须保留所需版权、归属和许可文件。
