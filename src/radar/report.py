"""Build a weekly Markdown report from observed versions."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from difflib import get_close_matches
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

from .collectors import SHANGHAI
from .models import Store, atomic_write, compact, digest, timestamp, web_url
from .pipeline import axes, latest_records, rebuild, substantive

LOG = logging.getLogger(__name__)
LEVELS = ("值得验证", "值得关注", "暂缓")
FIELDS = {"change": "新变化", "value": "价值", "conditions": "适用条件与证据", "next_step": "下一步"}
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
CANDIDATE_PREVIEW = 12


class DigestValidationError(ValueError):
    """The model response did not meet the report's evidence rules."""


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


def candidate_works(store, week, start, end, now):
    candidates = {}
    for work_id, work in store.works.items():
        records = [store.records[key] for key in work["records"] if is_visible(store.records[key], end, now)]
        if not records:
            continue
        latest = latest_records(records)
        meaningful = substantive(latest)
        versions = sorted(record.key for record in meaningful)
        weeks = work["weeks"]
        earlier = [(date, entry) for date, entry in weeks.items() if date < week]
        previous = max(earlier, default=(None, {}))[1]
        if week not in weeks:
            if versions == previous.get("versions") or not any(timestamp(record.observed_at) >= start for record in meaningful):
                continue
        priority = {"paper": 0, "paper_listing": 1, "pr": 2, "release": 3, "issue": 4, "lead": 5}
        latest.sort(key=lambda record: (priority[record.kind], record.id))
        primary = latest[0]
        old_paper = primary.kind == "paper" and primary.published_at and timestamp(primary.published_at) < start
        candidates[work_id] = {
            "primary": primary,
            "records": latest,
            "versions": versions,
            "previous": [store.records[key] for key in previous.get("versions", []) if key in store.records],
            "status": axes(latest),
            "retained": weeks.get(week, {}).get("selected", False),
            "historical": bool(old_paper and not previous),
        }
    return dict(sorted(candidates.items(), key=lambda pair: (
        not pair[1]["retained"],
        not bool(pair[1]["previous"]),
        pair[1]["primary"].kind == "lead",
        -max(record.updated.timestamp() for record in pair[1]["records"]),
        pair[0],
    )))


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


def candidate_line(candidate) -> str:
    primary = candidate["primary"]
    observed = primary.observed_at
    tags = " / ".join(sorted({tag for record in candidate["records"] for tag in record.tags})) or "未标注"
    history = " · 本周发现的历史工作" if candidate["historical"] else ""
    if primary.kind == "paper":
        revised = day(primary.updated_at) if primary.revision != "v1" else "—"
        dates = f"首次提交 {day(primary.published_at)} · 修订 {revised}"
    elif primary.kind == "paper_listing":
        dates = f"HF 标注的论文日期 {day(primary.published_at)} · 首次提交、修订待 arXiv 核对"
    else:
        dates = f"发布或创建 {day(primary.published_at)} · 上游更新 {day(primary.updated_at)}"
    sources = dict.fromkeys(entry for record in candidate["records"] for entry in evidence_links(record)[
        1 if record.key == primary.key else 0:
    ])
    source_text = f" · {'、'.join(sources)}" if sources else ""
    return (
        f"- {link(primary.title, primary.url)}{(' · ' + md(primary.revision)) if primary.revision else ''}{history}\n"
        f"  - {dates} · 首次发现当前所列版本 {day(observed)} · {tags}\n"
        f"  - {state_line(candidate['status'])}{source_text}"
    )


def record_for_model(record):
    return {"title": record.title, "summary": record.summary, "revision": record.revision,
            "kind": record.kind, "published_at": record.published_at, "updated_at": record.updated_at,
            "repo": record.upstream.get("repo"), "status": record.upstream.get("integration"),
            "author_code": record.upstream.get("author_code", {}).get("status"),
            "research_fork": bool(record.upstream.get("research_forks")),
            "verification": record.upstream.get("verification", {}).get("status")}


def request_digest(candidates, prompt: Path, backend: str):
    key_name = "RADAR_DEEPSEEK_API_KEY" if backend == "deepseek" else "RADAR_OPENAI_API_KEY"
    key = os.getenv(key_name, "").strip()
    if not key:
        raise ValueError(f"缺少 {key_name}")
    if backend == "deepseek":
        base, model = "https://api.deepseek.com", "deepseek-flash"
    else:
        base = web_url(os.getenv("RADAR_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        model = os.getenv("RADAR_OPENAI_MODEL")
        if not model:
            raise ValueError("缺少 RADAR_OPENAI_MODEL")
    offered, work_ids, record_ids = {}, {}, {}
    for work_id, item in list(candidates.items())[:30]:
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
            with urlopen(request, timeout=timeout) as response:
                raw = json.load(response)
            items = validate_digest(json.loads(raw["choices"][0]["message"]["content"]), offered)
            return [
                {**item, "works": [work_ids[ref] for ref in item["works"]],
                 "evidence": [record_ids[ref] for ref in item["evidence"]]}
                for item in items
            ]
        except (DigestValidationError, json.JSONDecodeError) as exc:
            if attempt + 1 == attempts:
                raise
            reason = str(exc) if isinstance(exc, DigestValidationError) else "响应不是有效 JSON"
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
        if not isinstance(works, list) or not works or len(works) != len(set(works)) or any(
            work in used or work not in offered for work in works
        ):
            raise DigestValidationError("Work 引用无效或重复")
        used.update(works)
        records_by_work = [{record["id"]: record for record in offered[work]["records"]} for work in works]
        records = {key: record for group in records_by_work for key, record in group.items()}
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence or any(key not in records for key in evidence) or any(
            not group.keys() & set(evidence) for group in records_by_work
        ):
            raise DigestValidationError("证据引用无效")
        allowed_numbers = set(NUMBER.findall(" ".join(
            f"{records[key]['title']} {records[key]['summary']} {records[key]['revision']}" for key in evidence
        )))
        for field in ("title", *FIELDS):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > 450 or re.search(r"https?://|www\.|\]\(", value, re.I):
                raise DigestValidationError("正文缺失或包含未授权链接")
            if set(NUMBER.findall(value)) - allowed_numbers:
                raise DigestValidationError("正文包含证据未给出的数字")
    if any(value["retain"] and key not in used for key, value in offered.items()):
        raise DigestValidationError("同周重跑遗漏了已有入选项")
    return items


def report(store: Store, target: Path, week: str, now: datetime, backend: str, prompt: Path):
    start, end = week_bounds(week)
    if now < start:
        raise ValueError("不能生成未来周次的报告")
    rebuild(store)
    candidates = candidate_works(store, week, start, end, now)
    lines = [f"# LLM 推理技术雷达 · {week}", ""]
    items, fallback = None, False
    if backend != "none" and candidates:
        try:
            items = request_digest(candidates, prompt, backend)
        except (OSError, ValueError, KeyError, IndexError, TypeError, HTTPException) as exc:
            reason = str(exc) if isinstance(exc, DigestValidationError) else type(exc).__name__
            LOG.warning("模型综合失败，已回退候选清单（%s）。", reason)
            fallback = True
        else:
            LOG.info("模型综合完成（%s），入选 %s 项。", backend, len(items))
    selected = set()
    if items:
        for level in LEVELS:
            group = [item for item in items if item["priority"] == level]
            if not group:
                continue
            lines.extend([f"## {level}", ""])
            for item in group:
                selected.update(item["works"])
                title = md(item["title"])
                histories = [candidates[work]["historical"] for work in item["works"]]
                if any(histories):
                    title += " · 本周发现的历史工作" if all(histories) else " · 含本周发现的历史工作"
                lines.extend([f"### {title}", ""])
                for key, label in FIELDS.items():
                    lines.append(f"- **{label}**　{md(item[key])}")
                sources = [source for work in item["works"] for record in candidates[work]["records"]
                           if record.key in item["evidence"]
                           for source in [link(record.title, record.url), *evidence_links(record)[1:]]]
                lines.extend([f"- **依据**　{'、'.join(dict.fromkeys(sources))}", ""])
    remaining = {key: value for key, value in candidates.items() if key not in selected}
    if remaining:
        lines.extend(["## 其他候选" if items is not None else "## 待综合 · 候选清单", ""])
        if fallback:
            lines.extend(["模型综合不可用，以下为已采集候选。", ""])
        if items is not None and not items:
            lines.extend(["本期精选 0 项。", ""])
        title_groups = {}
        for key, value in remaining.items():
            normalized = re.sub(r"[^\w]+", "", value["primary"].title.casefold())
            title_groups.setdefault(normalized, []).append(key)
        if items is None:
            lines.extend([f"本期共 {len(remaining)} 项候选，同一工作关联的材料列在同一项下。", ""])
        for index, (key, value) in enumerate(remaining.items()):
            if items is None:
                if index == CANDIDATE_PREVIEW:
                    lines.extend(["", "## 其他候选", "", "<details>",
                                  f"<summary>展开其余 {len(remaining) - index} 项候选</summary>", ""])
                lines.append(candidate_line(value))
            else:
                primary = value["primary"]
                lines.append(f"- {link(primary.title, primary.url)}" +
                             (" · 本周发现的历史工作" if value["historical"] else ""))
            normalized = re.sub(r"[^\w]+", "", value["primary"].title.casefold())
            if items is None and value["primary"].kind != "lead":
                closest = get_close_matches(normalized, title_groups, n=3, cutoff=0.91)
                related = [link(remaining[other]["primary"].title, remaining[other]["primary"].url)
                           for title in closest for other in title_groups[title] if other != key][:2]
                if not related:
                    continue
                lines.append(f"  - 标题相近，疑似关联　{'、'.join(related)}")
        if items is None and len(remaining) > CANDIDATE_PREVIEW:
            lines.extend(["", "</details>"])
        lines.append("")
    if not candidates:
        lines.extend(["本期暂未从已有采集结果发现新的候选进展。", ""])
    target.mkdir(parents=True, exist_ok=True)
    active = {key: state for key, state in store.state.items() if timestamp(state["window_start"]) < end
              and timestamp(state["last_attempt"]) >= start}
    local_until = min(end - timedelta(microseconds=1), now).astimezone(SHANGHAI)
    lines.extend(["## 采集时间范围", "",
                  f"报告范围为 {start:%Y-%m-%d %H:%M} 至 {local_until:%Y-%m-%d %H:%M}（Asia/Shanghai）。", ""])
    if active:
        first = min(timestamp(state["window_start"]) for state in active.values()).astimezone(SHANGHAI)
        last = max(timestamp(state["last_attempt"]) for state in active.values()).astimezone(SHANGHAI)
        lines.extend([f"相关渠道最近尝试的查询窗口覆盖 {first:%Y-%m-%d %H:%M} 至 {last:%Y-%m-%d %H:%M}。", ""])
    else:
        lines.extend(["该周无可对应的最近采集状态。", ""])
    lines.extend(["## 失败渠道", ""])
    failures = [(key, state) for key, state in active.items() if state.get("error")]
    lines.extend([f"- {md(key)} · {md(state['error'])}" for key, state in failures] or ["相关最近采集未记录失败渠道。" if active else "采集状态未知。"])
    path = target / f"{week}.md"
    atomic_write(path, "\n".join(lines) + "\n")
    for key, item in candidates.items():
        versions = item["versions"]
        store.works[key]["weeks"][week] = {"versions": versions, "hash": digest(versions),
                                          "selected": key in selected or item["retained"]}
    store.save_works()
    return path
