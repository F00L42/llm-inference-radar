"""Incremental collection and evidence-based grouping."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from http.client import HTTPException
from pathlib import Path
from xml.etree.ElementTree import ParseError

from .collectors import ADAPTERS, Client
from .models import ARXIV_REF, GITHUB_ITEM, Keywords, Record, Store, compact, digest, iso, timestamp, web_url, yaml_read

LOG = logging.getLogger(__name__)


def make_lead(entry: dict) -> Record:
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
    summary = compact(" ".join(entry.get(key, "") for key in ("note", "excerpt")))
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
        records = [record for record in records if record.kind != "lead"]
    return records


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
        weeks = {}
        prior_ids = {record.work_id for record in records} | {work_id}
        for prior_id in sorted(prior_ids):
            for week, entry in old_works.get(prior_id, {}).get("weeks", {}).items():
                previous = weeks.get(week, {})
                versions = sorted(set(previous.get("versions", []) + entry["versions"]))
                weeks[week] = {"versions": versions, "hash": digest(versions),
                               "selected": entry.get("selected", False) or previous.get("selected", False)}
        for record in records:
            record.work_id = work_id
        works[work_id] = {"id": work_id, "name": primary.title, "records": [record.key for record in records],
                          "status": axes(latest), "weeks": weeks}
    store.works = works


def collect(config: Path, store: Store, now: datetime, lookback_days: int | None = None, client=None):
    keywords = Keywords(yaml_read(config / "keywords.yaml"))
    sources = yaml_read(config / "sources.yaml")
    client = client if client is not None else Client()
    stats = {}

    def remember(record):
        if record.key not in store.records:
            record.observed_at = iso(now)
            store.records[record.key] = record
            return True
        previous = store.records[record.key]
        if not previous.published_at:
            previous.published_at = record.published_at
        return False

    for source in sources:
        for channel in source["channels"]:
            if not channel["enabled"]:
                continue
            channel_id = channel["id"]
            previous = store.state.get(channel_id, {})
            cursor = timestamp(previous.get("cursor"))
            since = now - timedelta(days=lookback_days if lookback_days is not None else 14)
            if cursor is not None:
                overlap = cursor - timedelta(days=3)
                since = overlap if lookback_days is None else min(since, overlap)
            state = {**previous, "last_attempt": iso(now), "window_start": iso(since)}
            new_count = 0
            try:
                for record in ADAPTERS[channel["adapter"]](client, channel, since, now, keywords):
                    new_count += remember(record)
            except (OSError, ValueError, KeyError, TypeError, ParseError, HTTPException) as exc:
                # Do not log request headers or response contents.
                error = f"{type(exc).__name__}: {compact(str(exc), 260)}"
                state.update(failures=previous.get("failures", 0) + 1, error=error)
                LOG.warning("%s，已保留本轮 %s 条新记录。%s", channel_id, new_count, error)
            else:
                state.update(last_success=iso(now), cursor=iso(max(now, cursor) if cursor else now), failures=0, error=None)
                LOG.info("%s，新增 %s 条记录", channel_id, new_count)
            store.state[channel_id] = state
            stats[channel_id] = new_count

    leads = config / "leads.yaml"
    if leads.exists():
        stats["manual"] = sum(remember(make_lead(entry)) for entry in yaml_read(leads).get("leads", []))
    rebuild(store)
    store.save()
    return stats
