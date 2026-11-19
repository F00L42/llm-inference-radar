"""The three official-API adapters. Each yields records as pages arrive."""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .models import ARXIV_ID, ARXIV_REF, GITHUB_ITEM, URL, Keywords, Record, compact, digest, iso, timestamp, web_url

ARXIV = "https://export.arxiv.org/api/query"
HF = "https://huggingface.co/api/daily_papers"
GITHUB = "https://api.github.com"
SHANGHAI = ZoneInfo("Asia/Shanghai")
ATOM = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom"}
ROADMAP = re.compile(r"\b(?:rfc|roadmap)\b", re.I)
CODE_CLAIM = re.compile(
    r"\b(?:code(?:\s+and\s+checkpoints)?\s+(?:is\s+available|at)|"
    r"(?:release|provide)\s+(?:our\s+)?code(?:\s+and\s+checkpoints)?)[^.!?\n]{0,35}?(?=https?://)",
    re.I,
)
RELATED_PAPER = re.compile(r"\b(?:paper|implements?|implementation of|based on|port of)\s*(?:[:=]?\s*[\[<(]?|\]\()$", re.I)
RESEARCH_FORK = re.compile(r"\bresearch fork\s*[:=]?\s*[\[<(]?(?=https://github\.com/)", re.I)


class Client:
    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self.arxiv_finished = None

    def get(self, url: str, params: dict, headers: dict | None = None):
        arxiv = url == ARXIV
        request_headers = {
            "User-Agent": "llm-inference-radar/0.1",
            **(headers or {}),
        }
        token = request_headers.pop("Authorization", None)
        request = Request(f"{url}?{urlencode(params)}", headers=request_headers)
        if token:
            request.add_unredirected_header("Authorization", token)
        for attempt in range(3):
            if arxiv and self.arxiv_finished is not None:
                delay = 3 - (self.clock() - self.arxiv_finished)
                if delay > 0:
                    self.sleep(delay)
            try:
                # A single synchronous connection, closed before a retry or next page.
                with urlopen(request, timeout=15) as response:
                    return response.read(), response.headers
            except HTTPError as exc:
                exc.close()
                if exc.code not in {502, 503, 504} or attempt == 2:
                    raise
            except (URLError, TimeoutError, ConnectionError, HTTPException):
                if attempt == 2:
                    raise
            finally:
                if arxiv:
                    self.arxiv_finished = self.clock()
            self.sleep(attempt + 1)

    def json(self, url: str, params: dict, headers: dict | None = None):
        raw, response_headers = self.get(url, params, headers)
        return json.loads(raw), response_headers


def clean_url(value: str) -> str:
    return web_url(value.rstrip(".,;)]"))


def paper_evidence(text: str, source: str) -> dict:
    code = [clean_url(match.group()) for claim in CODE_CLAIM.finditer(text)
            if (match := URL.match(text, claim.end()))]
    links = sorted({f"github:{repo.lower()}:{'pr' if kind == 'pull' else 'issue'}:{number}"
                    for repo, kind, number in GITHUB_ITEM.findall(text)})
    evidence = {"project_links": links}
    if code:
        evidence["author_code"] = {"status": "作者声明已公开", "urls": sorted(set(code)), "source": source}
    if re.search(r"\b(?:benchmarks?|experiments?|evaluation|experiments? show)\b", text, re.I):
        evidence["verification"] = {"status": "作者报告", "url": source}
    return evidence


def arxiv(client: Client, channel: dict, since: datetime, until: datetime, keywords: Keywords):
    for query in keywords.arxiv_queries():
        if channel.get("query"):
            query = f"({channel['query']}) AND {query}"
        yield from arxiv_pages(client, channel["id"], query, since, until, keywords)


def arxiv_pages(client, channel_id, query, since, until, keywords):
    start = 0
    while True:
        body, _ = client.get(ARXIV, {"search_query": query, "sortBy": "lastUpdatedDate",
                                     "sortOrder": "descending", "start": start, "max_results": 100})
        feed = ET.fromstring(body)
        total = int(feed.findtext("o:totalResults", namespaces=ATOM))
        entries = feed.findall("a:entry", ATOM)
        for entry in entries:
            def value(name):
                return entry.findtext(name, default="", namespaces=ATOM)

            paper = ARXIV_REF.fullmatch(value("a:id"))
            updated = timestamp(value("a:updated"))
            if not paper or updated is None:
                raise ValueError("arXiv 返回了无有效 ID 或更新时间的条目")
            if updated < since:
                return
            if updated > until:
                continue
            title, summary = compact(value("a:title")), compact(value("a:summary"), 4000)
            tags = keywords.tags(title, summary)
            if tags:
                base, version = paper.groups()
                url = f"https://arxiv.org/abs/{base}" + (f"v{version}" if version else "")
                evidence = paper_evidence(summary + " " + value("arxiv:comment"), url)
                yield Record(f"arxiv:{base}", "paper", title, url, compact(summary), channel_id,
                             f"v{version}" if version else "", value("a:published") or None,
                             value("a:updated"), tags=tags, upstream=evidence)
        start += len(entries)
        if start >= total:
            return
        if not entries:
            raise ValueError("arXiv 提前返回空页")


def hf_daily(client: Client, channel: dict, since: datetime, until: datetime, keywords: Keywords):
    day = until.astimezone(SHANGHAI).date()
    first = since.astimezone(SHANGHAI).date()
    while day >= first:
        rows, _ = client.json(HF, {"date": day.isoformat()})
        if not isinstance(rows, list):
            raise ValueError("HF Daily Papers 未返回每日列表")
        for row in rows:
            paper = row["paper"]
            match = re.fullmatch(rf"({ARXIV_ID})(?:v(\d+))?", paper["id"], re.I)
            if not match:
                continue
            base, version = match.groups()
            title, summary = compact(paper["title"]), compact(paper.get("summary"), 4000)
            tags = keywords.tags(title, summary)
            if tags:
                yield Record(f"hf:{base}", "paper_listing", title, f"https://huggingface.co/papers/{base}",
                             compact(summary), channel["id"], f"v{version}" if version else "",
                             published_at=paper.get("publishedAt"), tags=tags,
                             upstream={"paper_url": f"https://arxiv.org/abs/{base}"})
        day -= timedelta(days=1)


def github_pages(client, url, params, headers):
    page = 1
    while True:
        rows, response = client.json(url, {**params, "per_page": 100, "page": page}, headers)
        if not isinstance(rows, list):
            raise ValueError("GitHub 未返回列表")
        has_next = 'rel="next"' in response.get("Link", "")
        if has_next and not rows:
            raise ValueError("GitHub 分页提前返回空页")
        yield from rows
        if not has_next:
            return
        page += 1


def github(client: Client, channel: dict, since: datetime, until: datetime, keywords: Keywords):
    repo = channel["repo"].lower()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("GitHub repo 需要 owner/name")
    api = f"{GITHUB}/repos/{repo}"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    # Releases 按创建时间而非编辑时间排序，扫描所有页才能发现旧 Release 的修订。
    for row in github_pages(client, f"{api}/releases", {}, headers):
        if row.get("draft"):
            continue
        published = row.get("published_at")
        updated = max((timestamp(row.get(key)) for key in ("created_at", "updated_at", "published_at")
                       if row.get(key)), default=None)
        if updated is None or not since <= updated <= until:
            continue
        title, body = row.get("name") or row["tag_name"], row.get("body") or ""
        status = "预发布" if row.get("prerelease") else "正式发布"
        yield Record(f"github:{repo}:release:{row['id']}", "release", compact(title),
                     web_url(row["html_url"]), compact(body), channel["id"], row["tag_name"],
                     published, iso(updated), tags=keywords.tags(title, body),
                     upstream={"repo": repo, "integration": status, "details_hash": body_hash(title, body)})

    params = {"since": iso(since), "state": "all", "sort": "updated", "direction": "desc"}
    for row in github_pages(client, f"{api}/issues", params, headers):
        updated = timestamp(row.get("updated_at"))
        if updated is None or not since <= updated <= until:
            continue
        title, body = row["title"], row.get("body") or ""
        labels = " ".join(label["name"] for label in row.get("labels", []))
        tags = keywords.tags(title, body, labels)
        is_rfc = ROADMAP.search(f"{title} {labels}")
        if not tags and not is_rfc:
            continue
        kind = "pr" if "pull_request" in row else "issue"
        status = "RFC/Roadmap" if is_rfc and row["state"] == "open" else "Issue 已关闭" if row["state"] == "closed" else "开放 Issue"
        url = web_url(row["html_url"])
        evidence = {"repo": repo, "state": row["state"], "details_hash": body_hash(title, body)}
        if kind == "pr":
            if row["state"] == "open":
                status = "开放 PR"
            else:
                pr = row["pull_request"]
                if "merged_at" not in pr:
                    pr, _ = client.json(f"{api}/pulls/{row['number']}", {}, headers)
                if pr["merged_at"]:
                    if not pr.get("base"):
                        pr, _ = client.json(f"{api}/pulls/{row['number']}", {}, headers)
                    base = pr.get("base") or {}
                    default = (base.get("repo") or {}).get("default_branch")
                    status = "已合并（目标分支未知）"
                    if base.get("ref") and default:
                        status = "主干合并" if base["ref"] == default else "合并到非默认分支"
                    evidence["merged_at"] = pr["merged_at"]
                else:
                    status = "关闭未合并"
        evidence["integration"] = status
        refs = list(ARXIV_REF.finditer(body))
        declared = {match[1].lower() for match in refs if RELATED_PAPER.search(body[max(0, match.start() - 45):match.start()])}
        if len(declared) == 1:
            evidence["paper_id"] = declared.pop()
        forks = [clean_url(match.group()) for claim in RESEARCH_FORK.finditer(body)
                 if (match := URL.match(body, claim.end()))]
        if forks:
            evidence["research_forks"] = sorted(set(forks))
        if re.search(r"(?im)^\s*(?:[-*#]+\s*)?independent reproduction\s*:", body):
            evidence["verification"] = {"status": "独立复现（来源声明）", "url": url}
        yield Record(f"github:{repo}:{kind}:{row['number']}", kind, compact(title), url,
                     compact(body), channel["id"], published_at=row.get("created_at"),
                     updated_at=row["updated_at"], tags=tags, upstream=evidence)


def body_hash(title, body):
    return digest([title, body.replace("\r\n", "\n").strip()])


ADAPTERS = {"arxiv": arxiv, "hf_daily": hf_daily, "github": github}
