# 在 Modal 运行一次 CPU 评测

此入口运行项目中的三个**规则恢复策略基线**。它不调用大模型，不训练模型，也不能据此声称已测得任何 LLM 的能力。

## 1. 安装与认证

在项目根目录运行：

```bash
python -m pip install -r requirements-modal.txt
modal setup
modal token info
```

`modal setup` 需要你在浏览器中登录并授权。已有的 Modal API Token 也可通过环境变量 `MODAL_TOKEN_ID`、`MODAL_TOKEN_SECRET` 提供；不要把凭证提交到 Git 或写进报告。

## 2. 小规模首跑

```bash
modal run modal_eval.py --tasks 12 --seeds 5 --limit 60 --out results/modal
```

1. `tasks` 与 `seeds` 决定候选测试矩阵。
2. `limit` 限制实际执行的 **case × policy 回合数**。三个策略总是成组比较，非 3 的倍数会向下取整；最小非零值为 3。
3. 默认选取测试矩阵前 20 个 case，每个运行三个策略，共 60 回合。这是连通性与端到端流程检查，不代表整个测试矩阵的总体效果。
4. 每个远程输入最多包含 12 个回合，同时最多运行两个 CPU 容器。单个远程输入超时为 120 秒，不重试整个批次。
5. 所有批次确实从云端返回后，才调用本地报告生成器保存结果。任何远程异常都会中止运行，不会把本地回退结果标记为 Modal 结果。

## 3. 运行完整矩阵

```bash
modal run modal_eval.py --tasks 12 --seeds 5 --limit 0 --out results/modal_full
```

`limit=0` 表示全部运行。候选矩阵大小取决于 `recovery_eval.core.build_cases` 中的故障种类；当前六类故障下，12 个任务 × 5 个种子 × 6 类故障 × 3 个策略，共 1080 回合。

## 4. 生命周期、输出与费用

1. `modal run` 创建临时 App，客户端正常退出后自动停止。本项目没有使用 `modal deploy`、GPU、Volume、定时任务或 Web 服务。
2. JSON、CSV、HTML 等产物由 `recovery_eval.reporting.write_artifacts` 在本地输出。远程容器的临时文件不作为持久存储使用。
3. 报告元数据记录真实运行的 App ID、控制台地址、SDK 版本、回合数量以及 `backend: modal`。
4. CPU 和内存运行会产生计费使用量，账户额度是否覆盖取决于你的账户状态。`max_containers` 与 `timeout` 限制资源和单次时长，不是美元硬预算，也不保证零费用。
5. 运行中的日志会输出到终端。运行结束后，可使用控制台链接查看记录，或按真实 App ID 查询：

```bash
modal app logs ap-YOUR_ACTUAL_APP_ID --tail 1000
```

## 5. 没有云凭证时的结果边界

项目仍可在本地执行和验证。单纯导入 `modal_eval.py` 只注册 SDK 对象，不启动云任务。只有实际完成远程调用后才能称为“Modal 云端已运行”；本地执行、SDK 注册检查和云端执行需要分别说明。

## 官方依据

1. [安装与认证](https://modal.com/docs/guide/getting-started)
2. [临时 App 与本地入口](https://modal.com/docs/guide/apps)
3. [map 与 starmap 调用](https://modal.com/docs/guide/function-invocation-methods)
4. [本地 Python 源码打包](https://modal.com/docs/guide/images)
5. [CPU、内存与计费](https://modal.com/docs/guide/resources)
6. [Function 配置参数](https://modal.com/docs/sdk/py/latest/App)
7. [日志 CLI](https://modal.com/docs/cli/latest/app)

接口查阅日期：2026-10-17。此项目固定使用 Modal SDK 1.6.2。
