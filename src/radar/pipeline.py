"""Incremental collection and evidence-based grouping."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from http.client import HTTPException
from pathlib import Path
from xml.etree.ElementTree import ParseError

import yaml

from .collectors import ADAPTERS, Client, clean_github, github_tags
from .models import ARXIV_REF, GITHUB_ITEM, SUMMARY_LIMIT, Keywords, Record, Store, compact, digest, iso, timestamp, web_url, yaml_read

LOG = logging.getLogger(__name__)
STAT_KEYS = ("read", "in_window", "passed", "duplicates", "new", "failed")
PAPER_KINDS = {"paper", "paper_listing"}
GITHUB_KINDS = {"release", "pr", "issue"}


def make_lead(entry: dict) -> Record:
    if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
        raise ValueError("缺少有效 URL 字段")
    if any(not isinstance(entry.get(key, ""), str) for key in ("note", "excerpt")):
        raise ValueError("备注或摘录须为文本")
    url = web_url(entry["url"])
    match = ARXIV_REF.fullmatch(url)
    upstream = {"verification": {"status": "待核验", "url": url}}
    revision = ""
    stable_url = url
    if match:
        paper, version = match.groups()
        upstream["paper_id"] = paper.lower()
        revision = f"v{version}" if version else ""
        stable_url = f"https://arxiv.org/abs/{paper.lower()}"
    elif match := GITHUB_ITEM.fullmatch(url.split("?")[0].rstrip("/")):
        repo, kind, number = match.groups()
        upstream["project_links"] = [f"github:{repo.lower()}:{'pr' if kind == 'pull' else 'issue'}:{number}"]
        stable_url = f"https://github.com/{repo.lower()}/{kind}/{number}"
    summary = compact(" ".join(entry.get(key, "") for key in ("note", "excerpt")), SUMMARY_LIMIT)
    return Record(f"lead:{digest(stable_url)[:20]}", "lead", stable_url, stable_url, summary, "manual",
                  revision, upstream=upstream)


def latest_records(records):
    latest = {}
    for record in records:
        version = int(record.revision[1:]) if record.id.startswith("arxiv:") and record.revision[1:].isdigit() else 0
        rank = version, record.updated, record.observed_at or ""
        if record.id not in latest or rank >= latest[record.id][0]:
            latest[record.id] = rank, record
    return [record for _, record in latest.values()]


def substantive(records):
    kinds = {record.kind for record in records}
    if "paper" in kinds:
        records = [record for record in records if record.kind != "paper_listing"]
    if kinds - {"lead"}:
        records = [record for record in records if record.kind != "lead" or record.summary]
    return records


def qualifying_record(record, keywords):
    if record.kind not in GITHUB_KINDS:
        return replace(record, summary=compact(record.summary, SUMMARY_LIMIT))
    cleaned = clean_github(record.summary)
    labels = record.upstream.get("labels", [])
    if record.kind != "release":
        accepted, tags = github_tags(keywords, record.title, cleaned, labels)
        if record.upstream.get("passed_filter"):
            # New records passed on the full cleaned text, before excerpting.
            tags = record.tags
        elif not accepted:
            return None
    else:
        tags = keywords.tags(record.title, cleaned)
    return replace(record, summary=compact(cleaned, SUMMARY_LIMIT), tags=tags)


def content_version(record):
    identity, revision = record.id, record.revision
    text = record.summary
    if record.kind in PAPER_KINDS:
        identity = "arxiv:" + record.id.split(":", 1)[1].lower()
        revision = revision or "v1"
    elif record.kind in GITHUB_KINDS:
        text = clean_github(text)
    return digest([identity, revision, compact(record.title), compact(text, SUMMARY_LIMIT),
                   record.upstream.get("integration"), record.upstream.get("state")])


def assessment(records):
    current = substantive(latest_records(records))
    return digest(sorted({content_version(record) for record in current}))


def axes(records):
    result = {"author_code": [], "integration": {}, "verification": []}
    for record in records:
        evidence = record.upstream
        if code := evidence.get("author_code"):
            result["author_code"].append(code)
        if verification := evidence.get("verification"):
            result["verification"].append(verification)
        if repo := evidence.get("repo"):
            result["integration"].setdefault(repo, []).append({"status": evidence["integration"], "url": record.url})
        for fork in evidence.get("research_forks", []):
            repo = "/".join(fork.split("/")[3:5]).lower()
            result["integration"].setdefault(repo, []).append(
                {"status": "研究 fork（来源声明）", "url": fork, "source": record.url}
            )
    return result


def rebuild(store: Store) -> None:
    parents = {}

    def root(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def order(node):
        return (0 if node.startswith("arxiv:") else 1 if node.startswith("github:") else 2, node)

    by_id = {record.id: record for record in latest_records(store.records.values())}
    for record in by_id.values():
        refs = []
        if record.kind == "paper_listing":
            refs.append("arxiv:" + record.id.removeprefix("hf:"))
        if paper := record.upstream.get("paper_id"):
            refs.append(f"arxiv:{paper}")
        refs.extend(record.upstream.get("project_links", []))
        for ref in refs:
            first, second = sorted((root(record.id), root(ref)), key=order)
            # A bridge that refers to distinct arXiv papers is ambiguous.
            if first.startswith("arxiv:") and second.startswith("arxiv:") and first != second:
                continue
            parents[second] = first

    groups, old_works = {}, store.works
    for record in store.records.values():
        groups.setdefault(root(record.id), []).append(record)
    works = {}
    for work_id, records in groups.items():
        latest = latest_records(records)
        primary = next((record for record in latest if record.id == work_id), latest[0])
        prior_ids = {record.work_id for record in records} | {work_id}
        reviews, selected_weeks = [], []
        for prior_id in sorted(prior_ids, key=lambda item: (item != work_id, item)):
            if prior_id != work_id and prior_id in groups:
                continue
            prior = old_works.get(prior_id, {})
            if reviewed := prior.get("last_reviewed_hash"):
                reviews.append(reviewed)
            if selected := prior.get("last_selected_week"):
                selected_weeks.append(selected)
            for week, entry in sorted(prior.get("weeks", {}).items(), reverse=True):
                if not entry.get("selected"):
                    continue
                selected_weeks.append(week)
                old = [store.records[key] for key in entry.get("versions", []) if key in store.records]
                if old and len(old) == len(entry["versions"]):
                    reviews.append(assessment(old))
                break
        for record in records:
            record.work_id = work_id
        work = {"id": work_id, "name": primary.title, "records": [record.key for record in records],
                "status": axes(latest)}
        if reviews:
            current = assessment(latest)
            work["last_reviewed_hash"] = current if current in reviews else reviews[0]
        if selected_weeks:
            work["last_selected_week"] = max(selected_weeks)
        works[work_id] = work
    store.works = works


def collect(config: Path, store: Store, now: datetime, lookback_days: int | None = None, client=None, channels=()):
    keywords = Keywords(yaml_read(config / "keywords.yaml"))
    sources = yaml_read(config / "sources.yaml")
    client = client if client is not None else Client()
    enabled = {channel["id"]: channel for source in sources for channel in source["channels"] if channel["enabled"]}
    selected = set(channels)
    if selected - enabled.keys():
        raise ValueError("渠道未知或未启用：" + ", ".join(sorted(selected - enabled.keys())))
    stats = {}
    def paper_key(record):
        evidence = {name: value for name, value in record.upstream.items() if name != "listed_on"}
        return record.id, content_version(record), digest(evidence)

    papers = {paper_key(record): record for record in reversed(list(store.records.values()))
              if record.kind in PAPER_KINDS}

    def remember(record):
        key = record.key
        previous = store.records.get(key)
        if record.kind in PAPER_KINDS:
            previous = papers.setdefault(paper_key(record), previous or record)
            if previous is record:
                previous = None
        if previous is None:
            record.observed_at = iso(now)
            store.records[key] = record
            return True
        if not previous.published_at:
            previous.published_at = record.published_at
        if listed := record.upstream.get("listed_on"):
            previous.upstream.setdefault("listed_on", listed)
        return False

    def save_stats(channel_id, state, counts):
        state["counts"] = {key: counts[key] for key in STAT_KEYS}
        store.state[channel_id] = state
        stats[channel_id] = counts["new"]
        LOG.info("%s 读取 %d，窗口内 %d，通过 %d，重复 %d，新增 %d，失败 %d", channel_id,
                 *(counts[key] for key in STAT_KEYS))

    for channel_id, channel in enabled.items():
        if selected and channel_id not in selected:
            continue
        previous = store.state.get(channel_id, {})
        cursor = timestamp(previous.get("cursor"))
        adapter = channel["adapter"]
        if lookback_days is not None:
            since = now - timedelta(days=lookback_days)
        elif cursor is not None:
            since = cursor - timedelta(days=3)
        else:
            since = now - timedelta(days=14 if adapter == "github" else 30)
        state = {**previous, "last_attempt": iso(now), "window_start": iso(since)}
        counts = Counter()
        try:
            for record in ADAPTERS[adapter](client, channel, since, now, keywords, counts):
                counts["passed"] += 1
                counts["new" if remember(record) else "duplicates"] += 1
        except (OSError, ValueError, KeyError, TypeError, ParseError, HTTPException) as exc:
            error = f"HTTP {exc.code}" if isinstance(getattr(exc, "code", None), int) else type(exc).__name__
            counts["failed"] += 1
            state.update(failures=previous.get("failures", 0) + 1, error=error)
            LOG.warning("%s 采集未完成，保留已取得记录（%s）。", channel_id, error)
        else:
            state.update(last_success=iso(now), cursor=iso(max(now, cursor) if cursor else now), failures=0, error=None)
        save_stats(channel_id, state, counts)

    leads = config / "leads.yaml"
    if not selected and leads.exists():
        counts = Counter()
        try:
            entries = yaml_read(leads)
            if not isinstance(entries, dict) or not isinstance(rows := entries.get("leads", []), list):
                raise ValueError("线索配置需要 leads 列表")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            counts["failed"] += 1
            LOG.warning("leads.yaml 无法导入（%s）。", type(exc).__name__)
        else:
            for index, entry in enumerate(rows, 1):
                counts["read"] += 1
                counts["in_window"] += 1
                try:
                    lead = make_lead(entry)
                except (ValueError, KeyError, TypeError):
                    counts["failed"] += 1
                    LOG.warning("leads.yaml 第 %d 条跳过（字段格式或 URL 无效）。", index)
                    continue
                counts["passed"] += 1
                counts["new" if remember(lead) else "duplicates"] += 1
        save_stats("manual", {"last_attempt": iso(now), "window_start": iso(now),
                              "error": "线索导入有失败" if counts["failed"] else None}, counts)
    rebuild(store)
    store.save()
    return stats
