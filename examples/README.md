# 轻量结果快照

`release_snapshot.json` 汇总三个正式运行的原始 summary、manifest 和两组配对比较摘要；`validation_summary.json` 记录本轮实际验证范围。它们由本地真实结果生成，不是理论组合数或模型输出替身。

完整逐回合 JSONL、progress、模型原始输入输出、全部比较对及校验和在配套实验记录 ZIP 中。此目录刻意只保留轻量快照，不是完整运行目录；不要直接对本目录运行 `verify_artifacts.py`。原始运行目录可用该验证器检查。

1. `scripted`：6,144 个规则策略回合，24 次工具预算。
2. `smol` 与 `qwen`：各16个真实 CPU 模型回合，8 次模型请求、12 次工具、每次160个新token。
3. 两类实验使用不同执行预算，不能用聚合表做同预算模型／策略排名。
4. 模型前期工程中断记录另存，不混入32个正式回合；正式矩阵所有失败均保留。
5. 测试用 MCP client、Modal 本地规则调用和 Prime 静态任务检查不计作真实 LLM 成绩。

快照生成与完整性检查：

```bash
python scripts/build_release_review.py --scripted runs/scripted-release --smol runs/model-smol-release --qwen runs/model-qwen-release --comparisons runs/comparisons --snapshot examples/release_snapshot.json --out runs/review.html
```

请先将实验记录包按上述目录放回仓库，或改用实际绝对路径。程序会拒绝未完成、运行中源文件变化、数量错误或校验和不匹配的正式运行。
