# GitHub 提交与 PR 审查说明

代码已推送至已核实的目标仓库 [ltan0136/agent-recovery-eval](https://github.com/ltan0136/agent-recovery-eval)。[PR #1](https://github.com/ltan0136/agent-recovery-eval/pull/1)（`main` ← `feat/multi-workflow-evaluation`）已于 2026-10-17 合并；原始本地实验记录与审查包继续保留。[CI 运行 39036962895](https://github.com/ltan0136/agent-recovery-eval/actions/runs/39036962895) 的 Python 3.11／3.12／3.13 核心任务及可选集成任务共四项均已通过。

## 1. 先区分发布项目和贡献上游

| 目的 | 正确操作 | 本地材料的作用 |
| --- | --- | --- |
| 发布自己的新项目 | 创建或选择自己的 GitHub 仓库，推送代码 | 完整源码和文档可以作为仓库初始内容 |
| 在自己的已有项目中审查变更 | 推送开发分支，向默认分支发 PR | 本地审查分支、提交与 PR 正文 |
| 贡献给第三方开源项目 | 阅读贡献规范，fork 上游，适配并提交较小变更 | 当前实现是参考；不能直接把整套项目塞进上游 |

上传 ZIP 是交付文件，不等于创建 GitHub PR。PR 必须有明确的源分支、目标仓库及目标分支。

## 2. 本地先审查

1. 按 README 运行核心测试和无密钥演示。
2. 阅读 [架构](ARCHITECTURE.md)、[Benchmark Card](BENCHMARK_CARD.md) 和 [审查清单](REVIEW_CHECKLIST.md)。
3. 查看 Git 状态和改动，确认没有密钥、大型模型权重、临时缓存或真实私人数据。
4. 核对本地提交、补丁或 bundle 所含的分支。具体文件名以交付说明为准，不假定每种材料都已生成。

```bash
git status --short
git log --oneline --decorate -5
git diff --stat
```

ZIP 可解压检查；Git bundle 可保留提交历史并在不依赖原目录的情况下复核。补丁适合审查增量，不能替代其依赖的基线。

## 3. 推送到自己的仓库

本次推送与草稿 PR 已完成，不需重复创建。以下仅为今后向其他已确认仓库提交时的操作模板；`YOUR_ACCOUNT`、`YOUR_REPOSITORY`、`BASE_BRANCH` 和 `REVIEW_BRANCH` 不是本次实际地址或分支名。

```bash
git remote -v
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin REVIEW_BRANCH
gh pr create --base BASE_BRANCH --head REVIEW_BRANCH --title "Add multi-workflow recovery evaluation" --body-file PR_DESCRIPTION.md
```

如果已有 `origin`，先确认它的归属和地址，不重复添加。如果新仓库还没有可作为基线的默认分支，先按自己选择的初始化方式建立基线，再创建审查 PR。不要使用强制推送覆盖已有内容。

发布前由仓库所有者确认 MIT 许可选择、署名以及第三方代码来源。模型权重与依赖仍保留各自许可；本项目的 LICENSE 不为它们重新授权。

## 4. 贡献给上游仓库

1. 先阅读上游的贡献说明、许可证、架构、版本支持和现有 issue，确认维护者希望解决该问题。
2. 选一个可以独立解释与测试的功能，例如一个状态化任务环境、一个判定器测试或一个报告适配器。
3. 使用上游现有抽象与测试方式实现兼容变更，避免把独立仓库的目录与依赖整体搬进去。
4. 在自己的 fork 分支中提交，保留必要出处和许可；运行上游要求的检查。
5. PR 正文写清触发问题、变更后行为、复现与验证，以及本次未覆盖的内容。

现有独立项目可以展示能力，但不代表任何上游会接收整个项目。是否适合贡献，要在明确目标仓库后判断。

## 5. 本轮 PR 正文需要讲清的事实

1. 四个本地合成工作流、八类条件和四个手写恢复基线。
2. 真实模型工具循环与旧静态诊断的区别；公开步骤计划的限制。
3. 独立状态判定、迟到写入结算、原始轨迹和条件哈希。
4. 严格配对、簇 bootstrap、门禁及不确定结论的规则。
5. 实际通过的测试、真实运行结果与未运行项。
6. MCP 的真实传输验证范围，以及 Modal / Prime 可选入口的任务范围、本地检查与云端状态。
7. 人工故障、有限合成业务和非生产级保证等限制。

数据必须来自对应执行结果。没有证据的吞吐、成本节约、企业部署或模型提升不得写入 PR。

## 6. 本次 Git 审查包

交付的源码 ZIP 根目录另有 `review/repository.bundle`、`review/changes.patch` 与 `review/START_HERE.md`。bundle 含 v0.1 基线 `main` 和开发分支 `feat/multi-workflow-evaluation`，可以直接恢复本地审查环境：

```bash
git clone --branch feat/multi-workflow-evaluation review/repository.bundle agent-recovery-eval-review
cd agent-recovery-eval-review
git branch main origin/main
git diff --stat main...HEAD
git diff main...HEAD
git log --oneline --decorate -5
python run_demo.py
```

上例在解压包根目录执行；若已经进入源码目录，先回到包含 `review/` 的父目录。完整源码在 `agent-recovery-eval/`，可不使用 Git 直接运行。`changes.patch` 是相对于包内 `main` 的增量，不能假设可直接应用到任意第三方仓库。

审查包中的原始实现提交为 `b402658`；远程对应实现提交 `2f3262e8c18f2d4f4d8d7118388ca38b87a5fbe9` 与它具有相同 Git tree。原始本地历史也保存在本地 `archive/local-v0.1-baseline` 和 `archive/local-v0.2-review` 分支中；提交标识变化没有改变已保存实验结果。
