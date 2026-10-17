# Prime Verifiers：v0.2 多工作流静态诊断 Taskset

这个独立包将 Agent Recovery Eval v0.2 的**结算后业务状态和已提交操作账本**转换为 Verifiers **0.4.0 原生 v1 Taskset**。模型读取请求及证据，输出严格 JSON：

```json
{"duplicates": 0, "missing": 0, "success": true}
```

这是静态证据诊断：模型不执行工具、不改变环境，也不参与故障恢复。它与主项目动态 `model` 命令及 MCP stdio 入口分别测量不同能力。

## 1. 字段与判定

每个 case 包含 `id`、`request`、`trace` 和可信 `expected`。当前 `export-diagnosis` 格式为：

```json
{
  "id": "evaluator-owned-case-and-agent-id",
  "request": {"workflow_id": "ticket_dispatch", "goal": "...", "steps": []},
  "trace": [
    {"tool": "inspect_settled_state", "result": {}},
    {"tool": "inspect_settled_ledger", "result": []}
  ],
  "expected": {"duplicates": 0, "missing": 0, "success": true}
}
```

1. `duplicates`：每个请求步骤超出一次的已提交副作用总数。重试次数不等于副作用次数。
2. `missing`：没有已提交副作用的请求步骤数量。错误内容另由完整成功判定处理。
3. `success`：全部步骤恰好正确执行一次，业务参数、引用、先后依赖与资源约束均满足。
4. **零重复、零漏做并不足以说明成功**：参数错误、引用错误或依赖违规仍须判失败。fixture 包含四个这种已知答案控制样本。
5. 评分器严格逐字段匹配，不调用 LLM Judge。多余字段、重复 JSON 键、Markdown、布尔冒充整数等均无效。

`expected` 只存于 TaskData，不拼接进提示词。模型提示词只使用 `request` 和 `trace`，拒绝 `fault`、`target_step`、`oracle`、`expected`、`case_config` 等私有标签字段。MCP/运行配置仍需独立隔离文件权限；提示词检查不是 OS 沙箱。

## 2. 数据与重新导出

随包提供 **40 个静态样本**：四工作流 × clean/lost_ack/permanent_failure × 三脚本策略 = 36 个运行，加四个错误参数控制。它是可核验 smoke fixture，不是充分覆盖或独立模型能力评测。

在主项目根目录导出新数据：

```bash
python -m recovery_eval run --variants 1 --seeds 1 --faults clean,lost_ack,permanent_failure --policies no_retry,bounded_retry,idempotent_retry --out runs/prime-source
python -m recovery_eval export-diagnosis --episodes runs/prime-source/episodes.jsonl --out runs/prime-diagnosis.jsonl
```

两个命令均要求新输出路径以保留原证据。可复现随包 fixture（包括四个控制样本）的生成器：

```bash
python integrations/prime_taskset/scripts/build_fixture.py --out /new/path/diagnosis_cases.jsonl
```

## 3. 安装与真正的本地验证

在本目录：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -c "from recovery_diagnosis import load_environment; tasks=load_environment().load(); assert len(tasks)==40; print(len(tasks), tasks[0].data.id)"
```

`load_environment()` 仅为本包便捷函数，返回原生 `DiagnosisTaskset`。真实 v1 Taskset 接口是 `.load()`，没有 `.get_task_ids()`；每项 ID 位于 `task.data.id`。此过程不启动 runtime、不请求模型、不上传平台。

可用 CLI 验证运行配置：

```bash
vf-eval recovery_diagnosis --env.agent.harness.id null --env.agent.runtime.type subprocess --no-push -n 5 --dry-run
```

`--dry-run` 只解析验证配置，不能替代 Taskset 加载、评分器测试或真实推理。

本次本地检查实际使用 Verifiers 0.4.0：加载全部 40 项任务、为每项构造真实 `Trace` 调用原生 reward（正确引用为 1.0、无效回答为 0.0），并通过配置 dry-run。当前独立测试包含旧观察字段拒绝、私有字段拒绝、唯一 ID 和四个错误内容控制；fixture 重新生成后与随包文件字节一致。未请求模型或上传平台。

## 4. 可选真实模型端点

设定用户自己的端点、模型名和相应密钥环境变量后：

```bash
vf-eval recovery_diagnosis --env.agent.harness.id null --env.agent.runtime.type subprocess --client.base-url https://YOUR_PROVIDER/v1 --client.api-key-var PRIME_API_KEY --model YOUR_AVAILABLE_MODEL --sampling.temperature 0 --no-push -n 5 -r 1 -c 2
```

这只是配置模板；项目没有自动配置模型账户、充值、创建实例、发布 Hub 或请求云端推理。请自行确认端点支持的模型名。自定义数据可传 `--env.taskset.data-path /absolute/path/to/diagnosis_cases.jsonl`。`subprocess` 用于这个不执行用户代码的静态任务，不提供容器级隔离。

官方接口依据：[Verifiers v0.4.0](https://github.com/PrimeIntellect-ai/verifiers/releases/tag/v0.4.0)、[Tasksets](https://docs.primeintellect.ai/verifiers/v1/tasksets)、[Evaluation](https://docs.primeintellect.ai/verifiers/v1/evaluation)。版本核对日期：2026-10-17。
