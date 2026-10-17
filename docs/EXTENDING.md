# 扩展与维护指南

先阅读 [架构与执行边界](ARCHITECTURE.md) 和 [Benchmark Card](BENCHMARK_CARD.md)。当前扩展方式是修改 Python 实现并提交测试；没有自动插件发现、YAML 工作流加载或自动指标注册功能。对 1–2 人维护者，建议一次只扩展一个业务域、一种故障或一个模型后端，并保留可单独审查的提交。

## 1. 添加一个真正不同的工作流

| 修改位置 | 必须完成的内容 |
| --- | --- |
| `workflows.py` 的 `workflow_specs()` | 添加 `WorkflowSpec` 和 `Step`，声明实际工具、依赖与公开参数；`Step.resolve()` 将 `$参数名` 替换为业务值 |
| `workflows.py` 的 `_parameters()` | 为新标识增加明确分支，生成有意义的参数变体；不能让新工作流意外落入文档工作流的默认参数分支 |
| `engine.py` 的 `Environment._initial_state()` | 定义独立业务状态，例如资源余额、引用关系、审核记录，而非只改标题 |
| `Environment._apply()` | 实现工具的状态变化和真实业务约束；拒绝错误引用、资源不足或缺少审批等操作 |
| `Environment._oracle()` | 更新步骤与状态表的映射，以及内容、关联、资源和最终业务不变量检查 |

通用执行层已根据 `requires` 在提交前检查依赖，但具体业务约束仍需在 `_apply()` 实现。公开步骤计划必须可执行，且允许满足依赖的其他合法拓扑顺序。`Environment.tool_schemas` 由解析后的参数生成；当前参数校验支持本项目使用的字符串、整数及字符串数组。引入其他字段类型时，需要同时扩展工具规范与输入校验，不能只声明新参数。

至少补充以下独立性质测试：正常流程达到预先写明的业务结果；违反前置条件不产生副作用；正确参数与错误内容能被区分；跨资源引用正确；重复请求不会超用资源；有分支时至少验证两种合法顺序。测试应给出独立预期，不应直接调用生产判分函数生成“标准答案”。

## 2. 添加故障条件

1. 在 `workflows.py` 的 `FAULTS` 声明新标识，明确触发位置、持续时间及是否可恢复。
2. 在 `Environment._mutate()` 实现请求与提交之间的行为。需要异步送达时使用已有调度队列与 `_advance()`；需要改变查询可见性时检查 `_inspect()`。每回合状态应保存在环境实例中。
3. 工具响应只能提供调用者实际能够观察到的信息。隐藏故障标签、标准答案及最终账本应留在结果包中，不能放进模型消息。
4. 用最小案例分别验证首次触发、重试、幂等键、预算耗尽与最终结算。无法恢复的条件也必须计入完成率，不能因“没有重复”就判为成功。

如故障改变迟到提交行为，必须验证 `finish()` 结算后才评分；如使用超时模拟不确定结果，检查调用者不能直接从错误文本辨认“已提交”和“未提交”。

## 3. 添加模型适配器

`agents.py` 的 `ModelAdapter` 是 Python 协议。实现 `model_id`、`kind`、`complete(messages) -> ModelResponse` 以及 `describe() -> dict`，再将实例传入 `run_agent(case, adapter, ...)`。`describe()` 应记录模型版本和采样配置；`ModelResponse.text` 保留原始回答，并如实记录可获得的 usage、finish reason、延迟与 token ID。

适配器只接收消息，不能读取环境或案例的隐藏字段。继续使用现有文本动作协议：`{"tool":"tool_name","arguments":{...}}` 或 `{"finish":{"success":true}}`。不要在适配器中偷偷提取 JSON、补全缺失动作或把脚本操作冒充模型输出。后端错误应通过 `ModelAdapterError` 或明确异常保留为基础设施错误；不能丢弃这些回合。若希望从命令行选择新适配器，还需更新 `cli.py` 的参数与构造分支。

先用可控响应测试公共消息、原始记录、错误分类和预算，再进行少量真实调用。凭据不得进入 `describe()`、消息或结果文件。

## 4. 扩展指标及条件哈希

业务指标应由 `Environment._oracle()` 读取结算后的状态或账本计算，再由 `finish()` 写入结果。新增汇总需修改 `artifacts.py` 的 `summarize()`；新增展示需修改 `render_report()`。`regression.py` 的 `compare_runs()` 只对明确实现的指标提供比较与门禁，新指标不会自动参与放行判断。需要门禁时，应同时声明方向、容忍阈值、缺失值处理和对应测试。

为指标准备至少一个已知正确、一个已知错误及一个边界案例。区分“有重复的回合占比”和“重复副作用总数”；未测得的费用、耗时保持缺失，不能填零。不要把不同 Agent、故障条件或同一参数变体的重复种子当作独立样本随意混合。

`config_hash()` 对配置生成 SHA-256；案例哈希来自 `build_matrix()`，执行条件哈希来自 `Environment` 的 `execution_config`，覆盖完整案例、预算、幂等契约、解析后的工作流计划、引擎版本与结算设置。修改这些条件后必须得到不同的执行哈希，并重新生成结果。业务或故障实现的语义变化，还应更新引擎版本并保留 `artifacts.source_hashes()` 输出的源码哈希；不能仅凭配置哈希相同声称代码语义相同。模型升级通过不同的 `agent_id` 与模型元数据记录；对比双方仍需相同案例和环境条件，并另外审查 `prompt_hash` 与生成配置。

## 5. 一个无需模型的最小调用

从仓库根目录运行下面的 Python 代码。它只执行一个正常会议案例，先创建会议及邀请，再占用房间，最后确认，验证一条不同于公开列表顺序的合法执行路径。

```python
from recovery_eval.engine import Environment
from recovery_eval.workflows import build_matrix

case = build_matrix("meeting_booking", variants=1, seeds=1, faults=["clean"])[0]
environment = Environment(case)
steps = {step["name"]: step for step in environment.public_task["steps"]}

for name in ("meeting", "invite", "hold", "confirm"):
    step = steps[name]
    response = environment.invoke(step["tool"], step["arguments"])
    assert response["ok"], response

result = environment.finish(True, agent_id="extension-smoke", agent_kind="scripted")
assert result["success"] and result["duplicates"] == 0
print(result["success"], result["tool_calls"], len(result["ledger"]))
```

预期输出为 `True 4 4`。这是接口与业务约束的冒烟检查，不是语言模型实验，也不能替代新增功能的性质测试。
