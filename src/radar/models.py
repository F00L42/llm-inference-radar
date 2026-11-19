"""Configuration, identifiers and local storage."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z][\w.-]*/\d{7})"
ARXIV_REF = re.compile(
    rf"(?:arxiv:\s*|https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/"
    rf"|https?://huggingface\.co/papers/)({ARXIV_ID})(?:v(\d+))?(?![\w/])", re.I
)
GITHUB_ITEM = re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/(issues|pull)/(\d+)(?!\d)", re.I)
URL = re.compile(r"https?://[^\s<>\]{}\x60\"']+", re.I)


def timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def digest(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def compact(value: str | None, limit: int = 1600) -> str:
    return " ".join((value or "").split())[:limit]


def web_url(value: str) -> str:
    if not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("需要有效的 HTTP(S) URL")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None:
        raise ValueError("需要有效的 HTTP(S) URL")
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def yaml_read(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def json_write(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


@dataclass
class Record:
    id: str
    kind: str
    title: str
    url: str
    summary: str
    channel: str
    revision: str = ""
    published_at: str | None = None
    updated_at: str | None = None
    observed_at: str | None = None
    tags: list[str] = field(default_factory=list)
    upstream: dict = field(default_factory=dict)
    work_id: str = ""

    @property
    def content_hash(self) -> str:
        return digest([self.title, self.summary, self.revision, self.upstream])

    @property
    def key(self) -> str:
        return f"{self.id}@{self.content_hash}"

    @property
    def updated(self) -> datetime:
        return timestamp(self.updated_at or self.published_at or self.observed_at) or datetime.min.replace(tzinfo=UTC)


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.state = self.read_json("state.json")
        self.works = self.read_json("works.json")
        raw = path / "records.jsonl"
        records = (Record(**json.loads(line)) for line in raw.read_text(encoding="utf-8").splitlines() if line) if raw.exists() else ()
        self.records = {record.key: record for record in records}

    def read_json(self, name: str) -> dict:
        path = self.path / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save(self) -> None:
        atomic_write(self.path / "records.jsonl", "".join(
            json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in self.records.values()
        ))
        self.save_works()
        json_write(self.path / "state.json", self.state)

    def save_works(self) -> None:
        json_write(self.path / "works.json", self.works)


class Keywords:
    def __init__(self, config: dict):
        self.context = config["context"]
        self.branches = config["branches"]
        self.rules = {
            name: (self.pattern(group["terms"]), self.pattern(group.get("contextual", [])))
            for name, group in self.branches.items()
        }
        self.context_rule = self.pattern(self.context)

    @staticmethod
    def pattern(terms: list[str]) -> re.Pattern:
        alternatives = "|".join(re.escape(term) for term in terms) or r"(?!)"
        return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.I)

    def tags(self, *parts: str) -> list[str]:
        value = compact(" ".join(parts), 500_000)
        in_context = self.context_rule.search(value)
        return [
            name for name, (direct, contextual) in self.rules.items()
            if direct.search(value) or (in_context and contextual.search(value))
        ]

    def arxiv_queries(self):
        # 各分支结果取并集。用短查询控制 arXiv 后端执行时间，本地再应用完整规则。
        for group in self.branches.values():
            yield "(" + " OR ".join(f'all:"{term}"' for term in group["arxiv"]) + ")"
