# LLM 推理技术雷达

每天从官方 API 采集推理论文与工程进展，每周输出一份 Markdown。默认报告是带出处、日期、机制标签和上游状态的「待综合」清单。可选模型将相关材料合并成最多 5 个技术方向，并给出下一步验证。

## 本地运行

在项目根目录使用 Python 3.11 或以上版本。

```bash
python -m venv .venv
# 激活虚拟环境后
python -m pip install -e '.[test]'
radar collect --lookback-days 14
radar report --week 2026-W47 --backend none
radar add https://example.org/paper --note "关注这个机制"
pytest -q
```

Windows 如果提示应用控制阻止 pip 生成的 `radar.exe`，可以直接调用虚拟环境里的 Python。此用法也适用于 `add`。

```powershell
.\.venv\Scripts\python.exe -m radar.cli collect --lookback-days 14
.\.venv\Scripts\python.exe -m radar.cli report --week 2026-W47 --backend none
.\.venv\Scripts\python.exe -m pytest -q
```

默认首次回看 14 天。后续从各渠道的上次成功游标回退 3 天，`--lookback-days 30` 可扩大查询窗口。单渠道发生分页或连接错误时，本轮已取得的记录仍会保存，失败渠道的游标保持原值。GitHub 匿名访问受较低的速率限制，如遇 403/429，可通过环境变量 `GITHUB_TOKEN` 提供具备公开仓库只读权限的令牌。

`report` 默认生成上海时区本周截至运行时的报告。用 `--week YYYY-Www` 生成指定历史周，或用 `--previous-week` 生成上一个完整自然周。报告以已入库版本的首次发现时间分期，同一期写到 `reports/weekly/YYYY-Www.md`。无模型时默认展开前 12 个候选，其余可在「其他候选」中展开。报告记录查询失败，部分覆盖时请结合原始链接判断。

## 信源与取舍

- `config/sources.yaml` 保留原有 S01–S50 和后续筛选条目，部分条目的 `related_urls` 供人工查阅。默认启用 arXiv、HF Daily Papers，以及 vLLM、SGLang、vLLM-Ascend 的 GitHub Releases、Issues、PRs。需要追踪某个可选 GitHub 仓库时，将对应渠道的 `enabled` 设为 `true`。
- `config/keywords.yaml` 维护通用机制词，`config/leads.yaml` 维护人工线索。`radar add` 支持 `--excerpt`。定时任务从 `main` 读取线索，请提交并推送到 `main` 或经 PR 合入。
- 本地数据位于 `.radar-data/`。自动运行把 `state.json`、`records.jsonl` 和 `works.json` 提交到 `data` 分支。报告按版本去重，旧论文首次发现单独标识，arXiv 与 HF 的同论文材料归到同一 Work。

## 可选模型综合

在运行环境注入 `RADAR_DEEPSEEK_API_KEY`，然后运行 `radar report --backend deepseek`。该后端通过 `https://api.deepseek.com/chat/completions` 使用 `deepseek-flash`。有候选时，程序会把待综合的标题、短摘要和状态发送给模型端点，其中可能包含人工线索的备注和摘录。超时、缺少凭据或响应校验失败时，自动生成「待综合」候选清单。

如使用其他兼容接口，可设置以下变量并运行 `radar report --backend openai`。

| 名称 | 用途 |
| --- | --- |
| `RADAR_OPENAI_API_KEY` | 兼容接口的 Key |
| `RADAR_OPENAI_MODEL` | 服务端的模型名 |
| `RADAR_OPENAI_BASE_URL` | 兼容接口的基础地址，默认 `https://api.openai.com/v1`，程序追加 `/chat/completions` |

## GitHub Actions

将项目推送到 GitHub 的 `main` 后，`ci.yml` 跑离线测试。`radar.yml` 每天北京时间约 09:00 采集，周一处理上个完整周并创建或更新一份 Draft PR。使用 Actions 的 **Run workflow** 可指定周次、补采天数，或显式重开已关闭且未合并的 PR。每期只有一个报告分支，已合并的周次由人工维护。

工作流需要 `contents: write` 推送 `data` 和报告分支，需要 `pull-requests: write` 及仓库设置 **Allow GitHub Actions to create and approve pull requests** 创建 Draft PR。生成的 Markdown 同时保存在运行产物中，便于在 PR 权限不足时取回。可选配置 Actions secret `RADAR_GITHUB_READ_TOKEN` 供采集查询公开 GitHub 仓库。

在 GitHub 仓库 **Settings → Secrets and variables → Actions** 新建 repository secret `RADAR_DEEPSEEK_API_KEY`。工作流会在生成周报时自动选用 DeepSeek。把密钥保存在 **Secrets** 中，启用 DeepSeek 时其余参数由工作流和程序提供。

仓库变量 `RADAR_BACKEND` 可以显式覆盖自动选择。设为 `none` 可生成待综合清单，设为 `openai` 时使用上表中的模型与基础地址变量及同名 Actions secret `RADAR_OPENAI_API_KEY`。
