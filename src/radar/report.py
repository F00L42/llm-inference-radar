"""Build a weekly Markdown report from observed versions."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .collectors import ROADMAP, SHANGHAI
from .models import Keywords, Store, atomic_write, compact, digest, timestamp, web_url, yaml_read
from .pipeline import PAPER_KINDS, STAT_KEYS, axes, content_version, latest_records, qualifying_record, rebuild, substantive

LOG = logging.getLogger(__name__)
LEVELS = ("值得验证", "值得关注", "暂缓")
FIELDS = {"change": "新变化", "value": "价值", "conditions": "适用条件与证据", "next_step": "下一步"}
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
MODEL_LIMIT = 30
REPO_LIMIT = 5
DISPLAY_LIMIT = 500
WEEKLY_LIMIT = 12 * 1024
REVIEW_WINDOW = timedelta(days=14)
GROUPS = {"manual": "人工线索", "paper": "论文", "engineering": "工程"}
KIND_ORDER = {"paper": 0, "paper_listing": 1, "release": 2, "pr": 3, "issue": 4, "lead": 5}
CORE = {"投机、并行与扩散生成", "稀疏注意力与索引器", "KV、调度与 P/D 分离", "量化、Kernel 与编译"}
MEASURED = re.compile(r"\b(?:p\d{2}|ttft|tpot|latency|throughput|tokens/s|speedup|ablation|benchmarks?)\b", re.I)
MECHANISM = re.compile(r"\b(?:algorithm|mechanism|fused?|pipeline|parallel|overlap|prefetch|reuse|evict|graph capture|draft)\b", re.I)
SYSTEM_CONDITIONS = re.compile(r"\b(?:batch(?:ing|es)?|concurrency|workloads?|context length)\b", re.I)
HARDWARE = re.compile(r"\b(?:GPUs?|NPUs?|accelerators?|A\d{2,3}|H\d{2,3})\b", re.I)
MAINTENANCE = re.compile(r"\b(?:bump|typo|docs?|lint|refactor|deprecat\w*|migration|upgrade|rename|install)\b", re.I)
TEST_ONLY = re.compile(r"\[(?:ci|nightly|test|doc)\]|\b(?:nightly|test case|test coverage|failure tracker)\b", re.I)


class DigestValidationError(ValueError):
    """The model response did not meet the report's evidence rules."""


class ModelConfigError(ValueError):
    """Required model settings are missing or invalid."""


def week_id(moment: datetime) -> str:
    year, week, _ = moment.astimezone(SHANGHAI).isocalendar()
    return f"{year}-W{week:02}"


def week_bounds(week: str):
    if not re.fullmatch(r"\d{4}-W\d{2}", week):
        raise ValueError("周次格式为 YYYY-Www")
    start = datetime.fromisocalendar(int(week[:4]), int(week[-2:]), 1).replace(tzinfo=SHANGHAI)
    return start, start + timedelta(days=7)


def is_visible(record, end, now):
    seen = timestamp(record.observed_at)
    return seen is not None and seen < end and seen <= now


def relevance(record, tags, keywords):
    # Specific mechanism and measurement signals outweigh routine repository activity.
    repo = record.upstream.get("repo")
    text = record.title + " " + record.summary
    title_tags = keywords.tags(record.title) if record.kind in PAPER_KINDS else []
    score = 2 * len(tags & CORE) + len(tags - CORE) + 2 * len(set(title_tags) & CORE)
    score += 3 * bool(MECHANISM.search(text))
    score += 2 * bool(MEASURED.search(text)) + 2 * bool(MEASURED.search(text) and re.search(r"\d", text))
    score += 2 * bool(record.upstream.get("verification") and SYSTEM_CONDITIONS.search(text) and HARDWARE.search(text))
    score += 2 * bool(repo == "vllm-project/vllm-ascend" and tags and MECHANISM.search(text))
    if record.kind not in PAPER_KINDS:
        score += int(record.kind in {"pr", "release"} or bool(ROADMAP.search(record.title)))
        score -= 4 * bool(MAINTENANCE.search(record.title))
        score -= 9 * bool(TEST_ONLY.search(record.title) and not MEASURED.search(record.title))
    return score


def candidate_works(store, week, start, end, now, keywords):
    candidates = {}
    for work_id, work in store.works.items():
        records = [store.records[key] for key in work["records"] if is_visible(store.records[key], end, now)]
        latest = [qualified for record in latest_records(records) if (qualified := qualifying_record(record, keywords))]
        if not latest:
            continue
        latest.sort(key=lambda record: (KIND_ORDER[record.kind], record.id))
        primary = latest[0]
        meaningful = substantive(latest)
        versions = {content_version(record): timestamp(record.observed_at) for record in meaningful}
        for record in records:
            version = content_version(record)
            if version in versions:
                versions[version] = min(versions[version], timestamp(record.observed_at))
        version_hash = digest(sorted(versions))
        first_seen = max(versions.values())
        retained = work.get("last_selected_week") == week
        reviewed = version_hash == work.get("last_reviewed_hash")
        carryover = first_seen < start and now - first_seen <= REVIEW_WINDOW
        if not retained and first_seen < start and (reviewed or not carryover):
            continue
        previous = [record for record in latest_records(
            record for record in records if timestamp(record.observed_at) < first_seen
        ) if content_version(record) not in versions]
        tags = {tag for record in latest for tag in record.tags}
        group = "manual" if any(record.kind == "lead" for record in latest) else (
            "paper" if primary.kind in PAPER_KINDS else "engineering"
        )
        candidates[work_id] = {
            "primary": primary,
            "records": latest,
            "hash": version_hash,
            "first_seen": first_seen,
            "previous": [item for record in previous if (item := qualifying_record(record, keywords))],
            "status": axes(latest),
            "retained": retained,
            "reviewed": reviewed,
            "carryover": carryover,
            "group": group,
            "tags": tags,
            "repo": primary.upstream.get("repo"),
            "score": relevance(primary, tags, keywords),
            "historical": primary.kind in PAPER_KINDS and bool(primary.published_at and timestamp(primary.published_at) < start),
        }
    return dict(sorted(candidates.items(), key=lambda pair: (
        -pair[1]["score"],
        -pair[1]["primary"].updated.timestamp(),
        pair[0],
    )))


def select_candidates(candidates):
    selected, reasons = {}, {}
    repos, buckets = Counter(), Counter()
    eligible = {key: item for key, item in candidates.items() if item["retained"] or not item["reviewed"]}

    def add(key, bucket, reason):
        item = selected[key] = eligible.pop(key)
        if item["repo"]:
            repos[item["repo"]] += 1
        buckets[bucket] += 1
        reasons[key] = reason

    def fill(bucket, limit, reason, predicate, *, relax=False, carryover=False):
        while len(selected) < MODEL_LIMIT and buckets[bucket] < limit:
            options = [key for key, item in eligible.items() if predicate(item) and
                       (relax or repos[item["repo"]] < REPO_LIMIT)]
            if not options:
                break
            # Relevance order breaks ties across repositories with equal representation.
            key = min(options, key=lambda key: (carryover and not eligible[key]["carryover"],
                                               repos[eligible[key]["repo"]]))
            add(key, bucket, reason)

    for key, item in list(eligible.items()):
        if item["retained"] and len(selected) < MODEL_LIMIT:
            add(key, item["group"], "同周保留")
    fill("manual", 4, "人工线索", lambda item: item["group"] == "manual")
    fill("paper", 14, "论文名额", lambda item: item["group"] == "paper")
    fill("engineering", 12, "工程名额", lambda item: item["group"] == "engineering")
    fill("manual", 4, "待评估名额", lambda item: True, carryover=True)
    fill("borrow", MODEL_LIMIT, "空缺互借", lambda item: True)
    if eligible and len(selected) < MODEL_LIMIT:
        LOG.info("送模仍有 %d 个空位，所有剩余门槛内候选均受仓库软上限限制，放宽单仓上限。", MODEL_LIMIT - len(selected))
        fill("borrow", MODEL_LIMIT, "无其他候选可填，放宽仓库上限", lambda item: True, relax=True)
    return selected, reasons


def md(value: str) -> str:
    return re.sub(r"([\\\x60*_{}\[\]()<>#+!|~&])", r"\\\1", compact(value, 5000))


def link(title: str, url: str) -> str:
    return f"[{md(title)}](<{web_url(url).replace('<', '%3C').replace('>', '%3E')}>)"


def evidence_links(record):
    links = [link({"paper": "arXiv", "paper_listing": "HF Daily Papers"}.get(record.kind, record.kind), record.url)]
    if url := record.upstream.get("paper_url"):
        links.append(link("arXiv 原论文", url))
    for url in record.upstream.get("author_code", {}).get("urls", []):
        links.append(link("作者声明的代码或项目页", url))
    for url in record.upstream.get("research_forks", []):
        links.append(link("研究 fork", url))
    return links


def day(value):
    return timestamp(value).astimezone(SHANGHAI).strftime("%Y-%m-%d") if value else "未知"


def state_line(status) -> str:
    code = "作者声明已公开" if status["author_code"] else "未知"
    integration = "，".join(
        f"{md(repo)} {md(item['status'])}" for repo, values in status["integration"].items() for item in values
    ) or "未知"
    verification = "、".join(sorted({item["status"] for item in status["verification"]})) or "未知"
    return f"作者代码 {code} · 集成 {integration} · 验证 {verification}"


def candidate_row(candidate, status):
    primary = candidate["primary"]
    sources = " / ".join(dict.fromkeys(
        record.upstream.get("repo") or {"arxiv": "arXiv", "hf_daily": "HF Daily Papers", "manual": "人工线索"}.get(
            record.channel, record.channel
        ) for record in candidate["records"]
    ))
    tags = " / ".join(sorted(candidate["tags"])) or "—"
    dates = f"发布/创建 {day(primary.published_at)} · 发现 {day(candidate['first_seen'].isoformat())}"
    if listed := primary.upstream.get("listed_on"):
        dates += f" · HF 收录 {listed}"
    summary = compact(primary.summary, 120) or "—"
    return f"| {status} | {md(sources)} | {link(primary.title, primary.url)} | {md(tags)} | {dates} | {md(summary)} |"


def record_for_model(record):
    return {"title": record.title, "summary": record.summary, "revision": record.revision,
            "kind": record.kind, "published_at": record.published_at, "updated_at": record.updated_at,
            "repo": record.upstream.get("repo"), "status": record.upstream.get("integration"),
            "author_code": record.upstream.get("author_code", {}).get("status"),
            "research_fork": bool(record.upstream.get("research_forks")),
            "verification": record.upstream.get("verification", {}).get("status")}


def request_digest(candidates, prompt: Path, backend: str, attempted: set):
    key_name = "RADAR_DEEPSEEK_API_KEY" if backend == "deepseek" else "RADAR_OPENAI_API_KEY"
    key = os.getenv(key_name, "").strip()
    if not key:
        raise ModelConfigError(f"缺少 {key_name}")
    if backend == "deepseek":
        base, model = "https://api.deepseek.com", "deepseek-flash"
    else:
        try:
            base = web_url(os.getenv("RADAR_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        except ValueError as exc:
            raise ModelConfigError("RADAR_OPENAI_BASE_URL 无效") from exc
        model = os.getenv("RADAR_OPENAI_MODEL")
        if not model:
            raise ModelConfigError("缺少 RADAR_OPENAI_MODEL")
    offered, work_ids, record_ids = {}, {}, {}
    for work_id, item in candidates.items():
        work_ref = f"W{len(work_ids) + 1}"
        work_ids[work_ref] = work_id
        records = []
        for record in item["records"]:
            record_ref = f"R{len(record_ids) + 1}"
            record_ids[record_ref] = record.key
            records.append({"id": record_ref, **record_for_model(record)})
        offered[work_ref] = {"records": records,
                             "previous": [record_for_model(record) for record in item["previous"]],
                             "retain": item["retained"], "historical": item["historical"]}
    payload = {"model": model, "messages": [
        {"role": "system", "content": prompt.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(offered, ensure_ascii=False)},
    ]}
    if backend == "deepseek":
        payload.update(response_format={"type": "json_object"}, thinking={"type": "disabled"}, max_tokens=4096)
    attempts, timeout = (2, 90) if backend == "deepseek" else (1, 25)
    for attempt in range(attempts):
        request = Request(base + "/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        request.add_unredirected_header("Authorization", f"Bearer {key}")
        try:
            attempted.update(candidates)
            with urlopen(request, timeout=timeout) as response:
                raw = json.load(response)
            items = validate_digest(json.loads(raw["choices"][0]["message"]["content"]), offered)
            return [
                {**item, "works": [work_ids[ref] for ref in item["works"]],
                 "evidence": [record_ids[ref] for ref in item["evidence"]]}
                for item in items
            ]
        except (DigestValidationError, json.JSONDecodeError, KeyError, TypeError, IndexError, UnicodeError) as exc:
            reason = str(exc) if isinstance(exc, DigestValidationError) else "响应不是有效的综合 JSON"
            if attempt + 1 == attempts:
                raise DigestValidationError(reason) from exc
            LOG.info("DeepSeek 响应未通过校验（%s），重新请求一次。", reason)
            payload["messages"][0]["content"] += (
                f"\n前一次响应未通过校验，原因为「{reason}」。重新输出完整 JSON。"
                "正文五个字段一律使用无阿拉伯数字的表述，并逐字复制 works 和 evidence 中的 ID。"
            )


def validate_digest(result, offered):
    if not isinstance(result, dict) or not isinstance(items := result.get("items"), list) or len(items) > 5:
        raise DigestValidationError("报告应包含 0–5 个技术点")
    used = set()
    for item in items:
        if not isinstance(item, dict) or item.get("priority") not in LEVELS:
            raise DigestValidationError("优先级无效")
        works = item.get("works")
        if not isinstance(works, list) or not works or not all(isinstance(key, str) for key in works) or len(works) != len(set(works)) or any(
            work in used or work not in offered for work in works
        ):
            raise DigestValidationError("Work 引用无效或重复")
        used.update(works)
        records_by_work = [{record["id"]: record for record in offered[work]["records"]} for work in works]
        records = {key: record for group in records_by_work for key, record in group.items()}
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence or any(not isinstance(key, str) or key not in records for key in evidence) or any(
            not group.keys() & set(evidence) for group in records_by_work
        ):
            raise DigestValidationError("证据引用无效")
        allowed_numbers = set(NUMBER.findall(" ".join(
            f"{records[key]['title']} {records[key]['summary']} {records[key]['revision']}" for key in evidence
        )))
        for field in ("title", *FIELDS):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > (80 if field == "title" else 120) or re.search(r"https?://|www\.|\]\(", value, re.I):
                raise DigestValidationError("正文缺失或包含未授权链接")
            if set(NUMBER.findall(value)) - allowed_numbers:
                raise DigestValidationError("正文包含证据未给出的数字")
        if len("".join(item[field] for field in FIELDS)) > 400:
            raise DigestValidationError("重点正文超过 400 字")
    main_works = {item["works"][0] for item in items}
    if any(value["retain"] and key not in main_works for key, value in offered.items()):
        raise DigestValidationError("同周重跑遗漏了已有主 Work")
    return items


def coverage(store, start, end, now, candidates, shown, attempted, assessed, selected):
    eligible = {
        channel: state for channel, state in store.state.items()
        if (window := timestamp(state.get("window_start"))) is not None and window < end
        and start <= timestamp(state["last_attempt"]) <= now
    }
    latest = max((state["last_attempt"] for state in eligible.values()), default=None)
    batch = {channel: state for channel, state in eligible.items() if state["last_attempt"] == latest}
    failures = [f"{md(channel)}（{md(state['error'])}）" for channel, state in batch.items() if state.get("error")]
    counts = Counter()
    for state in batch.values():
        counts.update(state.get("counts", {}))
    details = (
        "最近一次相关采集"
        f"（{day(latest)}，{'、'.join(md(channel) for channel in batch)}）"
        f"　读取 {counts['read']} · 窗口内 {counts['in_window']} · 通过 {counts['passed']}"
        f" · 重复 {counts['duplicates']} · 新增 {counts['new']} · 失败 {counts['failed']}"
        if counts else "最近一次相关采集的分项统计不可用。"
    )
    related = f"失败渠道　{'、'.join(failures)}" if failures else "最近一次相关采集未记录失败渠道。" if batch else ""
    lines = [
        f"本期（上海时间）　{start:%Y-%m-%d} 至 {min(end - timedelta(microseconds=1), now).astimezone(SHANGHAI):%Y-%m-%d}",
        f"本期覆盖　合格 Work {len(candidates)} · 清单 {len(shown)} · 省略 {len(candidates) - len(shown)}"
        f" · 请求输入 {len(attempted)} · 成功评估 {assessed} · 重点方向 {len(selected)} · 主 Work {len(selected)}",
        details,
    ]
    if related:
        lines.append(related)
    return "\n\n".join(lines)


def display_candidates(candidates, attempted):
    if len(candidates) <= DISPLAY_LIMIT:
        return candidates
    pinned = set(attempted)
    remaining = [key for key in candidates if key not in pinned]
    show = pinned | set(remaining[:DISPLAY_LIMIT - len(pinned)])
    return {key: item for key, item in candidates.items() if key in show}


def render_candidates(week, shown, attempted, selected, items, stats, total):
    lines = [f"# 初筛候选清单 · {week}", "", f"[返回本期精简周报](../{week}.md)", "", stats, ""]
    if total > DISPLAY_LIMIT:
        lines.extend([f"**本期触发防爆截断，请检查过滤规则。** 合格 {total}，展示 {len(shown)}，省略 {total - len(shown)}。", ""])
    if attempted and items is None:
        lines.extend(["**本次综合未完成，已送模条目尚未形成有效取舍。**", ""])
    for group, label in GROUPS.items():
        lines.extend([f"## {label}", "",
                      "| 漏斗状态 | 来源 | 标题与链接 | 机制标签 | 发布/发现日 | 清洗摘要 |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for key, candidate in shown.items():
            if candidate["group"] != group:
                continue
            if key in selected or (candidate["retained"] and candidate["reviewed"]):
                status = "已精选"
            elif key in attempted or candidate["reviewed"]:
                status = "送模未选"
            else:
                status = "未送模"
            lines.append(candidate_row(candidate, status))
        lines.append("")
    return "\n".join(lines)


def render_weekly(week, candidates, items, stats, reason):
    header = [f"# LLM 推理技术雷达 · {week}", "", stats, "",
              f"[查看本期初筛候选清单](candidates/{week}.md)", ""]
    if items is None:
        return "\n".join(header + [reason, "", "**本期未完成模型综合，请直接审阅初筛清单。**", ""])
    if not items:
        return "\n".join(header + ["本期已完成综合，未选出重点。", ""])

    def render(link_limit, text_limit, state_limit):
        lines = header.copy()
        for priority in LEVELS:
            section = [item for item in items if item["priority"] == priority]
            if not section:
                continue
            lines.extend([f"## {priority}", ""])
            for item in section:
                primary = candidates[item["works"][0]]
                sources = list(dict.fromkeys(
                    source for work in item["works"] for record in candidates[work]["records"]
                    if record.key in item["evidence"] for source in evidence_links(record)
                ))
                sources = [source for source in sources if len(source.encode("utf-8")) <= 420][:link_limit]
                if not sources:
                    sources = [f"[原始链接见本期初筛清单](candidates/{week}.md)"]
                history = " · 历史材料" if primary["historical"] else ""
                lines.extend([f"### {md(item['title'])}{history}", ""])
                lines.extend(f"- **{label}**　{md(compact(item[key], text_limit))}" for key, label in FIELDS.items())
                lines.extend([f"- **状态**　{compact(state_line(primary['status']), state_limit)}",
                              f"- **出处**　{'、'.join(sources)}", ""])
        return "\n".join(lines)

    body = render(3, 120, 160)
    if len(body.encode("utf-8")) > WEEKLY_LIMIT:
        body = render(1, 120, 120)
    if len(body.encode("utf-8")) > WEEKLY_LIMIT:
        body = render(1, 68, 96)
    return body


def report(store: Store, target: Path, week: str, now: datetime, backend: str, prompt: Path, config=Path("config")):
    start, end = week_bounds(week)
    if now < start:
        raise ValueError("不能生成未来周次的报告")
    keywords = Keywords(yaml_read(config / "keywords.yaml"))
    rebuild(store)
    candidates = candidate_works(store, week, start, end, now, keywords)
    shortlisted, _ = select_candidates(candidates)
    attempted, items = set(), None
    reason = "模型未启用。" if backend == "none" else "本次无待综合材料。"
    if backend != "none" and shortlisted:
        try:
            items = request_digest(shortlisted, prompt, backend, attempted)
        except ModelConfigError as exc:
            reason = f"模型未启用（{exc}）。"
        except DigestValidationError as exc:
            reason = f"模型响应无效（{exc}）。"
        except (OSError, ValueError, HTTPException) as exc:
            failure = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
            reason = f"模型调用失败（{failure}）。"
        if items is None:
            LOG.warning("%s 本次实际尝试输入 %d 个 Work。", reason, len(attempted))
        else:
            LOG.info("%s 综合成功，评估 %d 个 Work，精选 %d 项。", backend, len(attempted), len(items))
    selected = {item["works"][0] for item in items or []}
    assessed = len(attempted) if items is not None else 0
    shown = display_candidates(candidates, attempted)
    stats = coverage(store, start, end, now, candidates, shown, attempted, assessed, selected)
    path = target / f"{week}.md"
    atomic_write(path, render_weekly(week, candidates, items, stats, reason))
    atomic_write(target / "candidates" / path.name, render_candidates(
        week, shown, attempted, selected, items, stats, len(candidates)
    ))
    if items is not None:
        for key in attempted:
            store.works[key]["last_reviewed_hash"] = candidates[key]["hash"]
        for key in selected:
            prior = store.works[key].get("last_selected_week", "")
            store.works[key]["last_selected_week"] = max(week, prior)
        store.save_works()
    return path
