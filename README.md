# Recovery Eval Lite

一个可独立复现的小型 Agent 故障恢复与轨迹诊断评测项目。核心实验需要 Python 3.11+。

## 1.内容

1. 执行 1,080 回合规则策略实验：12 组业务参数 × 5 个调度种子 × 6 种条件 × 3 种恢复策略。
2. 运行 40 项核心、报告、API 适配及回归比较单元测试；另运行 4 项 Prime 任务包评分测试，全部通过。
3. 比较同条件的 360 对回合，将幂等策略切换为查询后重试策略，检测出 180 个退化案例。这是故意构造的策略变更检验，不是某次真实模型版本升级事件。
4. 导出 94 个去重、带可信答案的最终状态诊断案例，包含实际执行产生的缺失、重复及错误内容控制样例。
5. 在 CPU 上真实运行公开模型 SmolLM2-135M-Instruct，对前 12 个分层交错排列样例进行诊断。严格 JSON 格式有效 8/12，完整答案正确 0/12；4 个无效回答触及 96 token 生成上限。保留全部原始输出，没有修复答案。
6. Modal SDK 1.6.2 入口注册、CLI 帮助和本地函数接口检查通过；Prime Verifiers 0.4.0 原生任务包加载、评分和 CLI dry-run 通过。两个平台的云端任务均未运行。

## 2. 最快复现

在项目目录运行：

```bash
python run_demo.py
```

Windows 可以双击 `run_demo.cmd`，它会运行测试、规则基线和回归比较，再打开 HTML 报告。

核心部分零第三方依赖，也可以分别运行：

```bash
python -m unittest discover -s tests -v
python -m recovery_eval demo --tasks 12 --seeds 5 --out results/local
python -m recovery_eval compare --before results/local/episodes.jsonl --after results/local/episodes.jsonl --before-policy idempotent_retry --after-policy verify_then_retry --out results/regression
```

打开 `results/overview.html` 查看本次交付总览；重新运行产生的新结果位于 `results/local` 和 `results/regression`。总览是本次交付时的快照，不会随重新运行自动更新。

## 3. 本次测量范围

工作流：创建服务记录，再发送关联通知。

故障条件：无故障对照、提交前失败、提交后丢失响应、延迟提交、丢失响应加陈旧查询、重复投递。种子实际改变故障作用于哪一步及延迟提交时间。

三个策略共享同一工具契约和观测接口：

| 策略 | 本次全条件完成率 | 含重复操作的回合率 |
| --- | ---: | ---: |
| 有界重试 | 33.3% | 66.7% |
| 查询后重试 | 50.0% | 50.0% |
| 稳定幂等键重试 | 100.0% | 0.0% |

模拟环境中的手写规则基线，不能推断为现实系统或所有故障下的保证。已知故障都只触发一次；没有覆盖永久故障、多服务事务、权限变化、真实网络以及多智能体并发。

评分依据最终业务状态和已提交操作账本。先送达所有迟到写入，再检查正确内容、正确关联、重复和缺失。

## 4. 失败复现

```bash
python -m recovery_eval replay --case-id task-001-s0-stale_read --policy verify_then_retry
```

输出含工具调用、故障、提交事件、最终状态和评分。

## 5. 真实本地模型诊断

这是独立的**最终状态轨迹诊断**任务：模型读取原始请求与 settled state，输出 duplicates、missing、success。它不直接执行恢复动作，因此结果不能解释成模型恢复成功率。

本次模型 revision：`12fd25f77366fa6b3b4b768ec3050bf629380bac`。CPU、float32、4 线程、greedy、每例最多 96 个新 token。模型权重不打包，首次运行会下载约 269 MB 的公开权重。

```bash
python -m pip install -r requirements-local-model.txt
python -m recovery_eval.local_model --data integrations/prime_taskset/data/diagnosis_cases.jsonl --out results/local_model_rerun.json --limit 12 --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac
```

精确 CPU PyTorch 安装说明见依赖文件注释；CUDA 版本没有必要。首次下载需要网络，下载后可以加 `--local-files-only`。原始首轮结果位于 `results/local_model_diagnosis.json`；输出格式失败如实计零，未用修复器。首轮终止原因由生成长度及原文回溯标注，后续 runner 已增加精确 token/EOS 记录。

## 6. Modal 与 Prime 入口

Modal 用于可选 CPU 批处理，不调用语言模型、不启动常驻服务：

```bash
python -m pip install -r requirements-modal.txt
modal setup
modal run modal_eval.py --limit 60 --out results/modal
```

Prime 的 `integrations/prime_taskset` 是原生 Verifiers v1 **轨迹诊断任务包**，与动态恢复实验的测量对象不同。可本地检查任务，再配置模型端点；需要托管运行时再安装 Prime CLI、认证并发布任务包。详见该目录的 README。不要使用过时的 v0 Rubric / StatefulToolEnv API。

## 7. 可选真实模型恢复入口

`recovery_eval/api_agent.py` 可将 OpenAI 兼容模型接到状态模拟工具。模型只收到业务请求和公共工具返回，不接触故障元数据与 oracle。此入口已经通过 mock tool-loop 测试，但本次未调用外部模型 API。

```bash
python -m recovery_eval.api_agent --help
```

实际运行时提供 `--model`、`--base-url`、`--api-key-env`，密钥只从环境变量读取，不写入代码和报告。不要把参数省略的帮助检查称为真实模型恢复实验。单回合最多 12 次模型请求、24 次工具调用、每次最多 512 个输出 token；费用由实际端点决定。

## 8. 文件与复现证据

1. `recovery_eval/core.py`：状态环境、事件账本、恢复策略、可信评分。
2. `recovery_eval/comparison.py`：相同条件哈希下的成对回归检查。
3. `recovery_eval/reporting.py`：JSONL/CSV/HTML 和诊断数据导出。
4. `recovery_eval/local_model.py`：真实 CPU 模型诊断。
5. `tests/`：40 项核心测试；Prime 独立包另有 4 项评分测试。
6. `results/`：本次真实执行结果与原始模型输出。
7. `.github/workflows/eval.yml`：无密钥的自动测试与规则评测流程。

## 9. 来源与边界

本项目是小型工程实现。相关先行工作包括 [UndoBench](https://arxiv.org/abs/2610.05622)、[Where Does Exactly-Once Live?](https://arxiv.org/abs/2609.29095)、[faultbench](https://github.com/mit-hun-k/faultbench)。接口参考 [Modal 批处理](https://modal.com/docs/guide/batch-processing)、[Verifiers v1](https://docs.primeintellect.ai/verifiers/v1/tasksets)；模型来自 [SmolLM2 官方仓库](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)。

下一步可扩展真实业务工具版本变化、更多工作流、两个以上模型、更多生成预算、测试选择策略及独立外部接入。
