import json
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
import yaml

from radar import collectors
from radar.cli import add_lead
from radar.models import Keywords, Record, Store, iso, timestamp, yaml_read
from radar.pipeline import collect
from radar.report import report, validate_digest

ROOT = Path(__file__).resolve().parents[1]
NOW = timestamp("2026-11-19T07:00:00Z")
WEEK = "2026-W47"
PROMPT = ROOT / "prompts/weekly_digest.md"


@pytest.fixture
def keywords():
    return Keywords(yaml_read(ROOT / "config/keywords.yaml"))


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config"
    path.mkdir()
    (path / "keywords.yaml").write_bytes((ROOT / "config/keywords.yaml").read_bytes())
    return path


def channels(config, *adapters):
    sources = [{"id": index, "channels": [{"id": adapter, "adapter": adapter, "enabled": True,
                                         "repo": "some/framework"}]}
               for index, adapter in enumerate(adapters)]
    (config / "sources.yaml").write_text(yaml.safe_dump(sources), encoding="utf-8")


def test_official_examples_use_generic_mechanisms_and_ignore_noise(keywords):
    cases = json.loads((ROOT / "tests/fixtures/arxiv_cases.json").read_text(encoding="utf-8"))
    query = " OR ".join(keywords.arxiv_queries()).lower()
    for case, indicative_phrase in zip(cases, ("speculative decoding", "discrete diffusion", "sparse attention"), strict=True):
        assert case["branch"] in keywords.tags(case["title"], case["excerpt"])
        assert f'all:"{indicative_phrase}"' in query
        assert case["url"].split("/")[-1].removesuffix("v1") not in query
    assert all(name not in query for name in ("pard-2", "uno", "dosa"))
    for title, body in [
        ("Fix readme typo", "Updated installation spelling."),
        ("Need help installing Python", "Environment config is missing."),
        ("Discrete diffusion for protein folding", "Predictions from protein sequences."),
        ("Batch job scheduler crash", "General infrastructure maintenance."),
    ]:
        assert not keywords.tags(title, body)
    sources = yaml_read(ROOT / "config/sources.yaml")
    ids = [source["id"] for source in sources]
    assert ids[:50] == [f"S{n:02}" for n in range(1, 51)]
    assert len(ids) == len(set(ids))
    configured_channels = [channel for source in sources for channel in source["channels"]]
    assert len(configured_channels) == len({channel["id"] for channel in configured_channels})
    assert {channel["id"] for channel in configured_channels if channel["enabled"]} == {
        "arxiv", "hf_daily", "github/vllm-project/vllm", "github/sgl-project/sglang", "github/vllm-project/vllm-ascend",
    }


def atom(base, revision, updated, published="2026-10-25T01:02:00Z", total=1):
    return f"""<feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
        <opensearch:totalResults>{total}</opensearch:totalResults>
        <entry><id>http://arxiv.org/abs/{base}{revision}</id><updated>{updated}</updated>
        <published>{published}</published><title>Speculative decoding for test workloads</title>
        <summary>{escape('Evaluation of a parallel draft.')}</summary></entry></feed>""".encode()


class Papers:
    revision = "v1"
    updated = "2026-11-18T01:02:00Z"
    include_hf = True

    def get(self, url, params):
        assert url == collectors.ARXIV
        return atom("2601.12345", self.revision, self.updated), {}

    def json(self, url, params):
        assert url == collectors.HF
        papers = [{"paper": {"id": "2601.12345", "title": "Speculative decoding for test workloads",
                             "summary": "Evaluation of a parallel draft."}}]
        return (papers if self.include_hf and params["date"] == "2026-11-19" else []), {}


def test_deduplication_and_revisions(config, tmp_path):
    channels(config, "arxiv", "hf_daily")
    store = Store(tmp_path / "data")
    client = Papers()
    collect(config, store, NOW, client=client)
    collect(config, store, NOW + timedelta(hours=1), client=client)
    assert len(store.records) == 2
    assert list(store.works) == ["arxiv:2601.12345"]
    client.revision, client.updated = "v2", iso(NOW + timedelta(hours=2))
    collect(config, store, NOW + timedelta(hours=2), client=client)
    assert len(store.records) == 3
    assert {r.revision for r in store.records.values() if r.kind == "paper"} == {"v1", "v2"}
    assert len(Store(store.path).works["arxiv:2601.12345"]["records"]) == 3


def issue(number, title, body="", state="open", kind="issues", **extra):
    return {"number": number, "title": title, "body": body, "state": state,
            "html_url": f"https://github.com/some/framework/{kind}/{number}", "updated_at": iso(NOW),
            "created_at": "2026-10-01T00:00:00Z", **extra}


def test_distinct_integration_and_verification_states(config, tmp_path):
    channels(config, "github")

    class GitHub:
        def json(self, url, params, headers):
            if url.endswith("/releases"):
                return [{"id": 1, "name": "Tiny release", "tag_name": "v0.1rc1", "prerelease": True,
                         "published_at": iso(NOW), "html_url": "https://github.com/some/framework/releases/tag/v0.1rc1"}], {}
            assert url.endswith("/issues")
            return [
                issue(1, "RFC: draft generation", "Research fork: https://github.com/authors/research"),
                issue(2, "Support KV cache", state="closed", kind="pull", pull_request={"merged_at": None},
                      labels=[{"name": "bug"}]),
                issue(3, "Sparse attention", kind="pull", pull_request={}),
                issue(4, "Fix readme typo", labels=[{"name": "chore"}]),
                issue(5, "GPU installation help", "I cannot install the dependencies."),
                issue(6, "Expert parallel", state="closed", kind="pull",
                      pull_request={"merged_at": iso(NOW), "base": {"ref": "main", "repo": {"default_branch": "main"}}}),
            ], {}

    store = Store(tmp_path / "data")
    collect(config, store, NOW, client=GitHub())
    assert {record.id: record.upstream["integration"] for record in store.records.values()} == {
        "github:some/framework:release:1": "预发布", "github:some/framework:issue:1": "RFC/Roadmap",
        "github:some/framework:pr:2": "关闭未合并", "github:some/framework:pr:3": "开放 PR",
        "github:some/framework:pr:6": "主干合并",
    }
    states = store.works["github:some/framework:issue:1"]["status"]
    assert states["integration"]["authors/research"][0]["status"] == "研究 fork（来源声明）"
    assert states["integration"]["some/framework"][0]["url"].endswith("/issues/1")
    assert not states["author_code"] and not states["verification"]


def test_partial_pagination_keeps_records_and_cursor(config, tmp_path):
    channels(config, "github", "arxiv")
    source_bytes = (config / "sources.yaml").read_bytes()
    add_lead(config / "leads.yaml", "https://example.org/project", "notes", "")
    lead_bytes = (config / "leads.yaml").read_bytes()
    previous_cursor = iso(NOW - timedelta(days=1))

    class Partial(Papers):
        def json(self, url, params, headers):
            if url.endswith("/releases"):
                return [], {}
            assert params["since"] == iso(NOW - timedelta(days=4))
            if params["page"] == 1:
                return [issue(1, "Enable sparse attention")], {"Link": '<url>; rel="next"'}
            raise OSError("page two failed")

    store = Store(tmp_path / "data")
    store.state["github"] = {"cursor": previous_cursor, "last_success": previous_cursor}
    collect(config, store, NOW, client=Partial())
    collect(config, store, NOW, client=Partial())
    assert store.state["github"]["cursor"] == previous_cursor
    assert store.state["github"]["failures"] == 2
    assert store.state["arxiv"]["cursor"] == iso(NOW)
    assert {record.kind for record in store.records.values()} == {"issue", "paper", "lead"}
    assert (config / "sources.yaml").read_bytes() == source_bytes
    assert (config / "leads.yaml").read_bytes() == lead_bytes


def test_dates_and_historical_discovery(config, tmp_path):
    channels(config, "arxiv")
    store = Store(tmp_path / "data")
    client = Papers()
    client.revision = "v2"
    collect(config, store, NOW, client=client)
    (record,) = store.records.values()
    assert (record.published_at, record.updated_at, record.observed_at) == ("2026-10-25T01:02:00Z", client.updated, iso(NOW))
    path = report(store, tmp_path, WEEK, NOW, "none", PROMPT)
    body = path.read_text(encoding="utf-8")
    assert "首次提交 2026-10-25 · 修订 2026-11-18 · 首次发现当前所列版本 2026-11-19" in body
    assert "本周发现的历史工作" in body
    assert "Speculative decoding" not in report(store, tmp_path, "2026-W46", NOW, "none", PROMPT).read_text(encoding="utf-8")


def test_arxiv_connection_and_interval(monkeypatch, keywords):
    ticks, started, active = [0.0], [], set()

    class Response:
        headers = {}

        def __enter__(self):
            assert not active
            active.add(True)
            return self

        def __exit__(self, *args):
            active.clear()
            ticks[0] += 0.5

        def read(self):
            return atom(f"2601.1234{len(started)}", "v1", iso(NOW), total=2)

    def open_request(request, timeout):
        started.append(ticks[0])
        return Response()

    monkeypatch.setattr(collectors, "urlopen", open_request)
    client = collectors.Client(clock=lambda: ticks[0], sleep=lambda delay: ticks.__setitem__(0, ticks[0] + delay))
    result = list(collectors.arxiv_pages(client, "arxiv", next(keywords.arxiv_queries()),
                                        NOW - timedelta(days=1), NOW, keywords))
    assert len(result) == 2
    assert started == [0, 3.5]


@pytest.mark.parametrize("mode", ["no-key", "timeout", "invalid"])
def test_model_failure_falls_back(config, tmp_path, monkeypatch, mode):
    channels(config, "arxiv")
    store = Store(tmp_path / "data")
    collect(config, store, NOW, client=Papers())
    monkeypatch.delenv("RADAR_OPENAI_API_KEY", raising=False)
    if mode != "no-key":
        monkeypatch.setenv("RADAR_OPENAI_API_KEY", "fake-test-key")
        monkeypatch.setenv("RADAR_OPENAI_MODEL", "test")

        def unavailable(*args, **kwargs):
            if mode == "timeout":
                raise TimeoutError
            return Path(config / "invalid.json").open("rb")

        (config / "invalid.json").write_text('{"choices":[{"message":{"content":"malformed"}}]}', encoding="utf-8")
        monkeypatch.setattr("radar.report.urlopen", unavailable)
    path = report(store, tmp_path, WEEK, NOW, "openai", PROMPT)
    body = path.read_text(encoding="utf-8")
    assert "待综合" in body and "模型综合不可用" in body
    assert "https://arxiv.org/abs/2601.12345v1" in body
    assert len(Store(store.path).records) == 1


def test_deepseek_report_uses_dedicated_key_and_endpoint(config, tmp_path, monkeypatch):
    channels(config, "arxiv")
    store = Store(tmp_path / "data")
    collect(config, store, NOW, client=Papers())
    monkeypatch.setenv("RADAR_DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("RADAR_OPENAI_API_KEY", "test-other-key")
    monkeypatch.setenv("RADAR_OPENAI_BASE_URL", "https://example.org/elsewhere")

    def complete(request, timeout):
        assert request.full_url == "https://api.deepseek.com/chat/completions"
        assert request.unredirected_hdrs["Authorization"] == "Bearer test-deepseek-key"
        assert b"test-deepseek-key" not in request.data
        payload = json.loads(request.data)
        assert payload["model"] == "deepseek-flash"
        assert payload["response_format"] == {"type": "json_object"}
        return BytesIO(b'{"choices":[{"message":{"content":"{\\"items\\":[]}"}}]}')

    monkeypatch.setattr("radar.report.urlopen", complete)
    path = report(store, tmp_path, WEEK, NOW, "deepseek", PROMPT)
    assert "本期精选 0 项" in path.read_text(encoding="utf-8")
    monkeypatch.delenv("RADAR_DEEPSEEK_API_KEY")
    path = report(store, tmp_path, WEEK, NOW, "deepseek", PROMPT)
    assert "模型综合不可用" in path.read_text(encoding="utf-8")


def test_report_reruns_and_changed_evidence(config, tmp_path, monkeypatch):
    channels(config, "arxiv", "hf_daily")
    store, client = Store(tmp_path / "data"), Papers()
    client.include_hf = False
    collect(config, store, NOW, client=client)

    def digest_for(candidates, prompt, backend):
        work, item = next(iter(candidates.items()))
        evidence = item["primary"].key
        assert bool(item["previous"]) == (item["primary"].revision == "v2")
        return [{"priority": "值得验证", "works": [work], "evidence": [evidence], "title": "并行草稿",
                 "change": f"当前版本 {item['primary'].revision}", "value": "推断可减少串行解码开销",
                 "conditions": "硬件、负载和质量条件未知", "next_step": "固定模型和输入，对比接受长度及解码延迟。"}]

    monkeypatch.setattr("radar.report.request_digest", digest_for)
    first = report(store, tmp_path, WEEK, NOW, "openai", PROMPT)
    assert first == report(store, tmp_path, WEEK, NOW + timedelta(hours=1), "openai", PROMPT)
    assert "下一步" in first.read_text(encoding="utf-8")
    assert first.read_text(encoding="utf-8").count("### ") == 1
    client.include_hf = True
    later = NOW + timedelta(days=7)
    collect(config, store, later, client=client)
    next_path = report(store, tmp_path, "2026-W48", later, "none", PROMPT)
    assert "暂未从已有采集结果发现新的候选" in next_path.read_text(encoding="utf-8")
    client.revision, client.updated = "v2", iso(later)
    collect(config, store, later, client=client)
    report(store, tmp_path, "2026-W48", later, "openai", PROMPT)
    assert "v2" in next_path.read_text(encoding="utf-8")
    assert "固定模型和输入" in next_path.read_text(encoding="utf-8")


def test_digest_validation_uses_supplied_evidence_only():
    item = {"priority": "值得关注", "works": ["work"], "evidence": ["record"], "title": "并行草稿",
            "change": "当前版本 v2", "value": "推断可能降低延迟", "conditions": "条件未知",
            "next_step": "固定输入长度并比较两条路径。"}
    offered = {"work": {"records": [{"id": "record", "title": "Parallel draft", "summary": "", "revision": "v2"}],
                        "retain": True}}
    assert validate_digest({"items": [item]}, offered) == [item]
    with pytest.raises(ValueError):
        validate_digest({"items": [{**item, "value": "提升 999 倍"}]}, offered)
    with pytest.raises(ValueError):
        validate_digest({"items": [{**item, "evidence": ["made-up"]}]}, offered)
    with pytest.raises(ValueError):
        validate_digest({"items": []}, offered)
