# LLM 推理技术雷达

每天从官方 API 采集推理论文与工程进展。每周生成精简周报与初筛候选清单。精简周报侧重可迁移的推理机制、昇腾工程落地和实测系统优化，最多给出 5 个带依据和下一步动作的方向。

## 本地运行

在项目根目录使用 Python 3.11 或以上版本。

```bash
python -m venv .venv
# 激活虚拟环境后
python -m pip install -e '.[test]'
radar collect
radar collect --channel arxiv --channel hf_daily --lookback-days 30
radar report --week 2026-W47 --backend none
radar add https://example.org/paper --note "关注这个机制"
python -m pytest -q
```

Windows 如果提示应用控制阻止 pip 生成的 `radar.exe`，可以直接调用虚拟环境里的 Python。此用法也适用于 `add`。

```powershell
.\.venv\Scripts\python.exe -m radar.cli collect --lookback-days 14
.\.venv\Scripts\python.exe -m radar.cli report --week 2026-W47 --backend none
.\.venv\Scripts\python.exe -m pytest -q
```

首次运行回看 30 天论文和 14 天 GitHub 记录。后续从各渠道的上次成功游标回退 3 天，显式 `--lookback-days` 优先。重复使用 `--channel` 只查询指定的已启用渠道、更新这些渠道的状态，并跳过人工线索导入。单渠道分页或连接失败时会保存已取得的记录并保留原游标。全渠道运行时，无效线索逐条跳过，线索文件无法解析时其余渠道照常保存。GitHub 匿名访问受较低的速率限制，如遇 403/429，可通过 `GITHUB_TOKEN` 提供具备公开仓库只读权限的令牌。

`report` 默认生成上海时区本周截至运行时的内容。用 `--week YYYY-Www` 指定周次，或用 `--previous-week` 生成上一个完整自然周。同一输出目录（默认 `reports/weekly`）下生成两个可以互相跳转的文件。CLI 标准输出仅返回周报路径。

- `YYYY-Www.md` 提供最多 5 个重点、证据出处和覆盖统计。
- `candidates/YYYY-Www.md` 按人工线索、论文、工程列出合格 Work 及其送模状态。超过 500 个时保留实际送模项，再按相关性补足到 500 个。省略数写入两个文件。

当前内容版本在首次观察后 14 天内仍可跨周竞争送模。通过响应校验后才记为已评估，同周已入选主 Work 可保留。历史重放使用数据副本，最近评估与入选字段无法复原完整历史。

## 信源与取舍

- `config/sources.yaml` 保留原有 S01–S50 和后续筛选条目，部分条目的 `related_urls` 供人工查阅。默认启用 arXiv、HF Daily Papers，以及 vLLM、SGLang、vLLM-Ascend 的 GitHub Releases、Issues、PRs。需要追踪某个可选 GitHub 仓库时，将对应渠道的 `enabled` 设为 `true`。
- `config/keywords.yaml` 维护通用机制词，`config/leads.yaml` 维护人工线索。`radar add` 支持 `--excerpt`。定时任务从 `main` 读取线索，请提交并推送到 `main` 或经 PR 合入。
- 本地数据位于 `.radar-data/`。自动运行把 `state.json`、`records.jsonl` 和 `works.json` 提交到 `data` 分支。GitHub Releases 在窗口内全收。Issue/PR 清理模板后按标题、原始标签或正文中不同的机制词初筛。已有记录用离线派生摘要复评，原记录和出处保留。arXiv 与 HF 的同论文材料归到同一 Work，同内容补齐沿用评估资格。

## 可选模型综合

在运行环境注入 `RADAR_DEEPSEEK_API_KEY`，然后运行 `radar report --backend deepseek`。该后端通过 `https://api.deepseek.com/chat/completions` 使用 `deepseek-flash`。每批最多送 30 个 Work，默认给论文 14 个、工程 12 个、人工线索和近期未评估材料 4 个名额。候选可互借空位，单仓库的 5 项软上限仅在仍有空位时放宽。发送内容包括标题、短摘要、状态及可能的人工线索摘录。模型未启用、调用失败或响应无效时，两个文件仍会生成，精简周报注明原因并引导查看初筛。

如使用其他兼容接口，可设置以下变量并运行 `radar report --backend openai`。

| 名称 | 用途 |
| --- | --- |
| `RADAR_OPENAI_API_KEY` | 兼容接口的 Key |
| `RADAR_OPENAI_MODEL` | 服务端的模型名 |
| `RADAR_OPENAI_BASE_URL` | 兼容接口的基础地址，默认 `https://api.openai.com/v1`，程序追加 `/chat/completions` |

## GitHub Actions

将项目推送到 GitHub 的 `main` 后，`ci.yml` 跑离线测试。`radar.yml` 每天北京时间约 09:00 采集，周一处理上个完整周，将两个 Markdown 放进同一个 Draft PR 和运行产物。使用 Actions 的 **Run workflow** 可指定周次、补采天数、空格分隔的渠道 ID，或显式重开已关闭且未合并的 PR。每期只有一个报告分支，已合并的周次由人工维护。

工作流需要 `contents: write` 推送 `data` 和报告分支，需要 `pull-requests: write` 及仓库设置 **Allow GitHub Actions to create and approve pull requests** 创建 Draft PR。从开发分支手动运行时可在 artifact 核验两份文件，写入 `data` 与发布 Draft PR 由 `main` 的运行负责。可选配置 Actions secret `RADAR_GITHUB_READ_TOKEN` 供采集查询公开 GitHub 仓库。

在 GitHub 仓库 **Settings → Secrets and variables → Actions** 新建 repository secret `RADAR_DEEPSEEK_API_KEY`。工作流会在生成周报时自动选用 DeepSeek。把密钥保存在 **Secrets** 中，启用 DeepSeek 时其余参数由工作流和程序提供。

仓库变量 `RADAR_BACKEND` 可以显式覆盖自动选择。设为 `none` 可生成待综合清单，设为 `openai` 时使用上表中的模型与基础地址变量及同名 Actions secret `RADAR_OPENAI_API_KEY`。
