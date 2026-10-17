# Modal：v0.2 多工作流 CPU 执行入口

`modal_eval.py` 使用当前 `engine.run_scripted` 和 `agents.run_agent`，支持四个业务工作流、八类条件（含 clean 对照）及四种脚本策略。只导入文件或调用 Modal Function 的 `.local()` 不会启动云任务。云端执行必须由用户明确运行 `modal run`。

## 1. 无云凭据的本地 SDK 验证

在项目根目录安装可选依赖：

```bash
python -m pip install -e '.[modal]'
python -c "import modal_eval; from recovery_eval.workflows import build_matrix; rows=modal_eval.evaluate_batch.local([{'case': c, 'policy': 'idempotent_retry'} for c in build_matrix(variants=1, seeds=1, faults='clean')]); assert len(rows)==4; assert all(r['success'] for r in rows); print('4 local SDK baseline episodes passed; cloud_execution=False')"
```

此命令真实导入 `modal==1.6.2`、注册函数，并在本地执行其底层函数，返回 v0.2 结果。它不构建远程镜像、不请求 Modal 控制面，不代表云端验证。

## 2. 可选云端脚本基线

先自行认证：

```bash
modal setup
modal token info
modal run modal_eval.py --mode scripted --variants 1 --seeds 1 --limit 16 --out runs/modal-smoke
```

1. `limit` 是 **case × comparison 回合总数**。scripted 模式每个 case 固定比较四策略，非四倍数向下取整，保持完整配对。默认 16 回合属于矩阵前缀连通性检查，不能代表总体效果。
2. 使用 `--workflows ticket_dispatch,inventory_order`、`--faults clean,lost_ack` 选择子集。工作流 ID 见 `python -m recovery_eval list`。
3. 每批最多 16 回合、同时最多两个 CPU 容器；CPU 请求范围 0.25–0.5，内存范围 256–512 MiB，单批 120 秒、整批不自动重试。
4. 当前默认矩阵四工作流 × 四参数变体 × 三种子 × 八条件 × 四策略 = **1536 回合**。完整运行需显式指定：

```bash
modal run modal_eval.py --mode scripted --variants 4 --seeds 3 --limit 0 --out runs/modal-full
```

## 3. 可选真实模型执行工具

`local-model` 指模型权重在 Modal 容器的 CPU 上推理，不指在用户电脑上运行。该模式复用当前 `LocalTransformersAdapter` 和 `run_agent`：模型每回合输出 JSON 动作，工具真正作用于模拟业务环境；格式错误、截断、原始输出和模型 revision 均保留。

```bash
modal run modal_eval.py --mode local-model --workflows ticket_dispatch --faults clean --variants 1 --seeds 1 --limit 2 --prompts basic,guarded --max-turns 2 --max-new-tokens 96 --max-calls 8 --allow-model-download --out runs/modal-model-smoke
```

1. 每次最多 16 个模型回合，禁止 `limit=0`。默认模型为 `HuggingFaceTB/SmolLM2-135M-Instruct`，它只用于端到端检查，不代表可靠业务能力。
2. 必须显式 `--allow-model-download`：容器无持久模型缓存，会下载公开权重。镜像固定 CPU PyTorch、Transformers 等依赖，不使用 GPU、Volume、服务部署或定时任务。
3. 远程模型函数仅一个 CPU 容器，2 CPU、2048 MiB 内存、900 秒超时、不自动重试。镜像构建和权重下载会增加耗时与用量。
4. `--revision` 可设置模型 commit SHA；默认 `main` 会记录实际解析的 SHA，正式实验应固定版本。
5. 本地也能检查同一个函数。先安装 `requirements-local-model.txt`，再在 Python 中执行下列代码；默认只读已有缓存：

```python
import modal_eval
from recovery_eval.workflows import build_matrix

rows = modal_eval.evaluate_model_batch.local(
    build_matrix('ticket_dispatch', variants=1, seeds=1, faults='clean'),
    prompts=('basic',),
    max_turns=1,
    max_new_tokens=32,
    cache_dir='/absolute/path/to/huggingface/hub',
    allow_download=False,
)
assert len(rows) == 1
assert rows[0]['evaluation_mode'] == 'live_model_recovery'
```

需同时检查 `run_status` 和 `model_calls`：缺依赖或缺权重时会保留 `infrastructure_error`，不能把“返回了一行结果”等同于成功推理。

## 4. 输出与运行边界

1. 所有远程函数实际返回且回合数量匹配后，才由本地 `artifacts.finalize_run` 生成 `episodes.jsonl`、`summary.json`、`metrics.csv`、`example_traces.json`、`report.html` 和 `manifest.json`。
2. manifest 写入 Modal App ID、控制台地址、SDK 版本、运行模式、调用次数与 `cloud_execution: true`。本地 `.local()` 不会自动设置这个标记。
3. 输出目录必须为空；不会覆盖已存在的证据。远程失败会抛错，不会悄悄使用本地结果代替。
4. `modal run` 为临时 App；没有 `modal deploy` 或持续服务。容器并发和超时限制不是金额硬预算，不能承诺零费用。认证和计费资源由用户决定是否使用。
5. 这个交付版本完成了本地 SDK 导入与 `.local()` 验证；没有宣称已运行 Modal 云任务。

接口核对：Modal SDK **1.6.2**，2026-10-17。官方资料：[临时 App](https://modal.com/docs/guide/apps)、[批量调用](https://modal.com/docs/guide/function-invocation-methods)、[镜像](https://modal.com/docs/guide/images)、[资源](https://modal.com/docs/guide/resources)。

## 5. 本次实际验证记录

1. 真实 SDK `evaluate_batch.local()` 完成 **32 个脚本回合**：四工作流 × clean/lost_ack × 四策略，全部结果为 schema 2.0，均标记 `agent_kind=scripted`。
2. 核对两个函数的 Modal 注册、`Function.local` 和 `starmap` 签名；验证限额仍保留完整策略配对。
3. 模型入口通过 SDK `.local()` 的**参数转发单元检查**；该检查将模型 adapter 和 runner 替换为测试对象，只验证 revision、预算、缓存、下载开关和提示版本确实传到当前 runner。未另跑模型生成，不将该检查计入模型成绩。
4. 没有启动 Modal 云任务、构建远程镜像或产生本项目的云端执行结果。
