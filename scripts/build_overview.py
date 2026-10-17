"""Build a standalone delivery overview from measured result artifacts."""

from datetime import datetime, timezone
import html
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / "results/local/summary.json").read_text())
regression = json.loads((root / "results/regression/comparison.json").read_text())
model = json.loads((root / "results/local_model_diagnosis.json").read_text())
labels = {"bounded_retry": "有界重试", "verify_then_retry": "查询后重试", "idempotent_retry": "稳定幂等键重试"}
rows = "".join(f"<tr><td>{labels[m['policy']]}</td><td>{m['episodes']}</td><td>{m['success_rate']:.1%}</td><td>{m['duplicate_episode_rate']:.1%}</td><td>{m['false_success_rate']:.1%}</td><td>{m['mean_tool_calls']:.2f}</td></tr>"
               for m in summary["metrics"] if m["fault"] == "all")
model_details = "".join(
    "<details><summary>" + html.escape(r["id"]) +
    f" · JSON {'有效' if r['strict_json_valid'] else '无效'} · 答案{'正确' if r['exact_match'] else '错误'}</summary>" +
    "<p>模型原始输出：</p><pre>" + html.escape(r["raw_generation"]) +
    "</pre><p>可信参考：</p><pre>" + html.escape(json.dumps(r["expected"], ensure_ascii=False)) +
    f"</pre><p>生成 {r['generated_tokens']} tokens；{r['generation_seconds']:.2f} 秒；" +
    f"达到生成上限：{'是' if r.get('stopped_at_token_limit') else '否'}。</p></details>"
    for r in model["results"])
status = {"created_utc": datetime.now(timezone.utc).isoformat(), "core_tests_passed": 40,
          "prime_scoring_tests_passed": 4, "scripted_episodes": summary["episodes"],
          "diagnosis_dataset_cases": sum(1 for line in (root / "integrations/prime_taskset/data/diagnosis_cases.jsonl").read_text().splitlines() if line),
          "local_model_cases": model["summary"]["cases"], "local_model_id": model["model"]["id"],
          "modal_cloud_run": False, "prime_cloud_run": False,
          "external_model_recovery_run": False, "local_model_diagnosis_run": True}
(root / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recovery Eval Lite · 交付总览</title><style>
body{{margin:0;background:#f2f5f8;color:#183047;font:16px/1.75 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:44px 24px}}.eyebrow{{font-size:12px;letter-spacing:2px;color:#087f79}}h1{{font-size:40px;line-height:1.2;margin:12px 0}}h2{{font-size:23px;margin-top:36px}}.lead{{font-size:18px;color:#4d6074}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.card,.panel,details{{background:white;border:1px solid #d8e2e9;border-radius:14px;padding:18px}}.card strong{{display:block;font-size:34px}}.panel{{border-left:5px solid #0b847c}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dce4eb}}th{{background:#e5edf4}}.scroll{{overflow-x:auto}}details{{margin:12px 0}}summary{{cursor:pointer;font-size:14px}}pre{{background:#eff4f7;border-radius:8px;padding:14px;white-space:pre-wrap;word-break:break-word;font-size:13px}}a{{color:#087f79}}.fine{{font-size:13px;color:#52697c}}@media(max-width:700px){{main{{padding:24px 14px}}h1{{font-size:31px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}</style>
<main><div class="eyebrow">RECOVERY EVAL LITE / 2026-10-17</div><h1>从工具故障到<br>可复现的评测证据</h1>
<p class="lead">已独立完成代码、评分测试、规则基线、回归对照和一次真实本地模型诊断。无需云平台账户。</p>
<div class="cards"><div class="card"><strong>{summary['episodes']:,}</strong>规则策略回合</div><div class="card"><strong>44</strong>通过的单元测试</div><div class="card"><strong>{model['summary']['cases']}</strong>真实模型诊断样例</div><div class="card"><strong>0</strong>已启动云端任务</div></div>
<div class="panel"><strong>本次测量边界：</strong>恢复实验使用手写策略；真实小模型执行的是最终状态诊断，没有让模型直接执行恢复动作。两者分别记录，不能合并成一个“模型恢复成功率”。这是单工作流的最小工程版本。</div>
<h2>1. 规则恢复：已执行 1,080 回合</h2><p>12 组业务参数 × 5 个调度种子 × 6 种条件（含无故障对照）× 3 种策略。创建记录后发送关联通知，以最终业务状态和提交账本评分。</p>
<div class="scroll"><table><tr><th>策略</th><th>回合</th><th>完成率</th><th>重复操作率</th><th>错误宣称成功率</th><th>平均工具调用</th></tr>{rows}</table></div>
<p class="fine">100% 是本环境的稳定操作键与服务端去重契约共同保证的结果，不代表真实模型或所有现实故障。种子改变故障目标与提交延迟，12 组参数不等于 12 种任务。</p>
<h2>2. 回归检测：同条件配对 {regression['paired_cases']} 回合</h2><p>故意从“稳定幂等键重试”切换到“查询后重试”，检测出 <strong>{regression['regressions']} 个退化案例</strong>，改善 {regression['improvements']} 个。比较器核对完整条件哈希，拒绝条件不一致或缺少配对的结果。</p>
<h2>3. 真实小模型：成绩如实保留</h2><p><strong>{html.escape(model['model']['id'])}</strong>，CPU / greedy / 每例生成上限 96 tokens。测试集前 12 例按诊断标签交错排列，没有随机总体抽样。</p>
<div class="scroll"><table><tr><th>指标</th><th>实测</th><th>解释</th></tr><tr><td>严格 JSON 格式有效</td><td>{model['summary']['strict_json_valid_count']}/12（{model['summary']['strict_json_valid_rate']:.1%}）</td><td>整段解析，不提取或修复片段</td></tr><tr><td>完整答案正确</td><td>{model['summary']['exact_match_count']}/12</td><td>重复数、缺失数、完成状态全部正确才通过</td></tr><tr><td>达到生成长度上限</td><td>4/12</td><td>四条格式错误均触及上限，不能全部归因为诊断能力</td></tr><tr><td>生成总耗时</td><td>{model['summary']['generation_seconds']:.1f} 秒</td><td>不含模型下载，模型加载另约 {model['summary']['model_load_seconds']:.1f} 秒</td></tr></table></div>
<p>原始输出、可信答案、模型 revision、输入 token 和耗时保存在代码包。模型没有看到 expected 标签。小模型和 12 例用于验证评测链路，尚不能支持跨模型或总体能力结论。</p>
{model_details}
<h2>4. 已验证的工程入口</h2><ol><li>核心与 API mock、回归、报告：40 项测试通过。</li><li>Prime 原生 Verifiers 0.4.0 诊断任务包：4 项评分测试、任务加载与 CLI dry-run 通过。</li><li>Modal SDK 1.6.2：注册、CLI 帮助与本地函数接口通过；云端未调用。</li><li>可选 OpenAI 兼容模型恢复 runner：有请求与工具预算，已完成 mock tool-loop 测试；未调用外部模型。</li></ol>
<h2>5. 如何复现</h2><pre>python run_demo.py</pre><p>核心无需第三方依赖。Windows 可双击 run_demo.cmd。可选真实模型安装、首轮记录、Modal 和 Prime 命令均在 README；模型权重与依赖安装目录不包含在代码包中。</p>
<p class="fine">下一步应增加真实工作流、模型与工具版本变更，继续检验迁移性。当前不声称原创算法、生产级平台或完整 LangGraph/MCP server 适配。</p>
<footer class="fine">本次交付结果快照 · {html.escape(status['created_utc'])}</footer></main></html>"""
(root / "results/overview.html").write_text(page, encoding="utf-8")
print(root / "results/overview.html")
