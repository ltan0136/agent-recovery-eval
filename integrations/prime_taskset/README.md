# Prime Verifiers：工具轨迹故障诊断

这个独立包将 Recovery Eval Lite 的公开操作轨迹转成 **Verifiers 0.4.0 原生 v1 Taskset**。模型读取请求与工具轨迹，输出严格 JSON：

```json
{"duplicates": 0, "missing": 0, "success": true}
```

评分器逐字段比较可信账本生成的答案，不调用大模型裁判。额外字段、重复 JSON 键、Markdown、布尔值冒充整数等均判为无效。`expected` 只存储在 TaskData 中，模型提示词仅由 `request` 和 `trace` 构造。

**本任务测量已有轨迹的故障识别能力。它没有让模型执行或恢复操作，因此不能据此宣称模型具有 Agent 故障恢复能力。** 超时后的真实提交情况必须能从公开状态查询结果判断；无法判断的轨迹不应进入这个三字段任务。

## 数据

先由主项目导出 `data/diagnosis_cases.jsonl`。每行结构如下；`expected` 必须来自主项目可信账本评分器，不能由被测模型提供。

```json
{"id":"case-0001","request":{"operations":[{"operation_id":"op-1"}]},"trace":[{"tool":"query_state","result":{"committed_records":[{"operation_id":"op-1"}]}}],"expected":{"duplicates":0,"missing":0,"success":true}}
```

`duplicates` 是每个请求操作超出一条的已提交记录总数；`missing` 是没有匹配已提交记录的请求操作数；两者均为零时 `success` 为真。调用次数不等于提交次数。公开输入禁止包含 `ground_truth`、`expected`、`oracle`、`reference` 等答案字段；这项检查不能替代人工泄漏审计。

## 安装和本地运行

在本目录执行：

```bash
python -m pip install -e .
vf-eval recovery_diagnosis --env.agent.harness.id null --env.agent.runtime.type subprocess --no-push -n 5 --dry-run
```

`--dry-run` 只验证运行配置，不请求模型。真实评测还需要模型端点和对应密钥。例如提前在本机设置 `OPENAI_API_KEY` 后，运行：

```bash
vf-eval recovery_diagnosis \
  --env.agent.harness.id null \
  --env.agent.runtime.type subprocess \
  --client.base-url https://api.openai.com/v1 \
  --client.api-key-var OPENAI_API_KEY \
  --model gpt-4.1-mini \
  --sampling.temperature 0 \
  --no-push -n 5 -r 1 -c 2
```

上面的模型名称仅为端点配置示例；运行前请确认该提供商仍提供所选模型。为复现结果，请明确固定模型与采样参数。不要将 API 密钥写进数据、配置仓库或报告。通过 Prime Inference 运行时，改用该平台当前可用模型，并配置 `PRIME_API_KEY`。本项目不会自动充值或创建付费资源。

可通过 `--env.taskset.data-path /absolute/path/to/diagnosis_cases.jsonl` 改变数据位置。`subprocess` 适用于这个不执行用户代码的静态诊断任务；它不提供容器安全隔离。

## 本地评分器验证

```bash
python -m unittest discover -s tests -v
```

测试和 dry-run 均不代表已经完成云端运行。发布到 Environments Hub、托管评测和 Modal 运行需要相应认证及计费配置。

API 依据：

1. https://github.com/PrimeIntellect-ai/verifiers/releases/tag/v0.4.0
2. https://docs.primeintellect.ai/verifiers/v1/tasksets
3. https://docs.primeintellect.ai/verifiers/v1/evaluation
