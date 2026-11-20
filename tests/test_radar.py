import json
from collections import Counter
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
import yaml

from radar import collectors
from radar.cli import add_lead, main
from radar.models import Keywords, Record, Store, iso, timestamp, yaml_read
from radar.pipeline import collect, rebuild
from radar.report import candidate_works, display_candidates, report, select_candidates, validate_digest, week_bounds, week_id

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


def atom(base, revision, updated, published="2026-10-25T01:02:00Z", total=1,
         title="Speculative decoding for test workloads", summary="Evaluation of a parallel draft."):
    return f"""<feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
        <opensearch:totalResults>{total}</opensearch:totalResults>
        <entry><id>http://arxiv.org/abs/{base}{revision}</id><updated>{updated}</updated>
        <published>{published}</published><title>{escape(title)}</title>
        <summary>{escape(summary)}</summary></entry></feed>""".encode()


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
    assert "本期未完成模型综合" in body
    assert "[查看本期初筛候选清单](candidates/2026-W47.md)" in body
    assert "发布/创建 2026-10-25 · 发现 2026-11-19" in (tmp_path / "candidates" / path.name).read_text(encoding="utf-8")
    report(store, tmp_path, "2026-W46", NOW, "none", PROMPT)
    assert "Speculative decoding" not in (tmp_path / "candidates/2026-W46.md").read_text(encoding="utf-8")


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
    assert "本期未完成模型综合" in body
    assert {"no-key": "模型未启用", "timeout": "模型调用失败", "invalid": "模型响应无效"}[mode] in body
    listing = (tmp_path / "candidates" / path.name).read_text(encoding="utf-8")
    assert "https://arxiv.org/abs/2601.12345v1" in listing
    assert ("综合未完成，已送模" in listing) == (mode != "no-key")
    assert f"请求输入 {int(mode != 'no-key')} · 成功评估 0" in listing
    assert "fake-test-key" not in body + listing
    assert "last_reviewed_hash" not in next(iter(store.works.values()))
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
    assert "本期已完成综合，未选出重点" in path.read_text(encoding="utf-8")
    assert "请求输入 1 · 成功评估 1" in path.read_text(encoding="utf-8")
    work = Store(store.path).works["arxiv:2601.12345"]
    assert work["last_reviewed_hash"] and "last_selected_week" not in work


def test_deepseek_retries_unsupported_number_without_weakening_evidence(config, tmp_path, monkeypatch):
    channels(config, "arxiv")
    store = Store(tmp_path / "data")
    collect(config, store, NOW, client=Papers())
    monkeypatch.setenv("RADAR_DEEPSEEK_API_KEY", "test-deepseek-key")
    calls = []

    def complete(request, timeout):
        assert timeout == 90
        messages = json.loads(request.data)["messages"]
        offered = json.loads(messages[1]["content"])
        work, candidate = next(iter(offered.items()))
        assert work == "W1" and "id" not in candidate["records"][0]
        calls.append(messages[0]["content"])
        item = {"priority": "值得验证", "works": [work], "title": "并行草稿",
                "change": "本周首次收录", "value": "提升 999 倍" if len(calls) == 1 else "推断可减少解码耗时",
                "conditions": "详细负载未知", "next_step": "固定输入长度并比较解码耗时"}
        return BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({"items": [item]})}}]}).encode())

    monkeypatch.setattr("radar.report.urlopen", complete)
    path = report(store, tmp_path, WEEK, NOW, "deepseek", PROMPT)
    body = path.read_text(encoding="utf-8")
    assert len(calls) == 2
    assert "正文包含证据未给出的数字" in calls[1]
    assert "999" not in body
    assert "https://arxiv.org/abs/2601.12345v1" in body and "## 值得验证" in body
    assert path.stat().st_size <= 12 * 1024


def test_report_reruns_and_changed_evidence(config, tmp_path, monkeypatch):
    channels(config, "arxiv", "hf_daily")
    store, client = Store(tmp_path / "data"), Papers()
    client.include_hf = False
    collect(config, store, NOW, client=client)

    def digest_for(candidates, prompt, backend, attempted):
        attempted.update(candidates)
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
    assert "合格 Work 0" in next_path.read_text(encoding="utf-8")
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


def pool(store, keywords, now=NOW):
    week = week_id(now)
    rebuild(store)
    return candidate_works(store, week, *week_bounds(week), now, keywords)


def test_frozen_papers_reach_model_shortlist_in_their_release_windows(config, tmp_path, keywords):
    channels(config, "arxiv")
    cases = json.loads((ROOT / "tests/fixtures/arxiv_cases.json").read_text(encoding="utf-8"))
    for case in cases:
        base = case["url"].split("/")[-1].removesuffix("v1")
        published = case["submitted_at"]

        class Frozen:
            def get(self, url, params):
                assert base not in params["search_query"]
                return atom(base, "v1", published, published, title=case["title"], summary=case["excerpt"]), {}

        now = timestamp(published) + timedelta(days=1)
        store = Store(tmp_path / base)
        collect(config, store, now, client=Frozen())
        assert f"arxiv:{base}" in select_candidates(pool(store, keywords, now))[0]


def test_clean_filter_and_legacy_replay(config, tmp_path, keywords):
    channels(config, "github")
    library = "some/framework"
    old_noise = Record(f"github:{library}:pr:90", "pr", "Upgrade SDK", "https://example.org/90",
                       "<!-- Fix KV cache -->", "github", tags=["KV、调度与 P/D 分离"],
                       observed_at=iso(NOW), upstream={"repo": library, "integration": "开放 PR"})

    class GitHub:
        def json(self, url, params, headers):
            if url.endswith("/releases"):
                return [{"id": 8, "tag_name": "v1", "published_at": iso(NOW),
                         "html_url": "https://example.org/release"}], {}
            return [
                issue(1, "Upgrade SDK", "<!-- KV cache and sparse attention -->"),
                issue(2, "Fix ordering", "KV cache, KV-cache, KV cache."),
                issue(3, "Fix ordering", "We fix KV cache and prefix caching.\n## Environment\ninternal logs"),
                issue(4, "Fix ordering", "## Checklist\n- [ ] KV cache and sparse attention\n## Notes\nSetup."),
                issue(5, "Fix ordering", "```txt\n" + "cache log " * 70 + "KV cache, spare token, sparse attention\n```"),
                issue(6, "Fix ordering", "```\nshort: KV cache / sparse attention\n```"),
                issue(7, "Discuss plans", labels=[{"name": "Roadmap"}]),
                issue(8, "Fix sparse attention", labels=[{"name": "chore"}], body="substantive " * 100),
                issue(9, "Fix ordering", labels=[{"name": "KV cache"}]),
                issue(10, "Preserve branch and tail checkpoints",
                      "Restore recurrent state on the hybrid linear prefill path."),
            ], {}

    store = Store(tmp_path / "data")
    store.records[old_noise.key] = old_noise
    collect(config, store, NOW, client=GitHub())
    assert set(pool(store, keywords)) == {
        f"github:{library}:issue:{number}" for number in (3, 6, 7, 8, 9, 10)
    } | {f"github:{library}:release:8"}
    assert store.records[old_noise.key].summary == old_noise.summary
    assert store.state["github"]["counts"] == {
        "read": 11, "in_window": 11, "passed": 7, "duplicates": 0, "new": 7, "failed": 0,
    }
    by_id = {record.id: record for record in store.records.values()}
    assert by_id[f"github:{library}:issue:3"].summary == "We fix KV cache and prefix caching."
    assert len(by_id[f"github:{library}:issue:8"].summary) == 800
    assert by_id[f"github:{library}:issue:9"].upstream["labels"] == ["KV cache"]
    path = report(store, tmp_path, WEEK, NOW, "none", PROMPT, config)
    text = (tmp_path / "candidates" / path.name).read_text(encoding="utf-8")
    assert all(len(line.rsplit("|", 2)[-2].strip()) <= 120 for line in text.splitlines() if line.startswith("|"))


def test_targeted_backfill_recalls_frozen_sources(config, tmp_path, keywords):
    channels(config, "arxiv", "hf_daily", "github")
    base = "2611.02501"
    submitted = "2026-11-02T04:21:47Z"
    then = timestamp("2026-11-19T07:43:00Z")

    class Frozen(Papers):
        def get(self, url, params):
            assert 'all:"speculative decoding"' in " ".join(keywords.arxiv_queries())
            assert base not in params["search_query"] and "specpilot" not in params["search_query"].lower()
            return atom(base, "v1", submitted, submitted), {}

        def json(self, url, params, headers=None):
            if url == collectors.HF:
                return ([{"paper": {"id": base, "title": "Speculative decoding for test workloads",
                                    "summary": "Evaluation of a parallel draft.", "publishedAt": submitted}}]
                        if params["date"] == "2026-11-04" else []), {}
            if url.endswith("/releases"):
                return [], {}
            return [{**issue(65813, "RFC: Speculative decoding"), "updated_at": "2026-11-05T07:01:23Z"},
                    {**issue(66753, "RFC: Tree speculative decoding",
                             body="See https://github.com/vllm-project/vllm/issues/65813"),
                     "updated_at": "2026-11-11T02:05:28Z"}], {}

    (config / "leads.yaml").write_text("leads: [{url: 'https://example.org/lead'}]", encoding="utf-8")
    store = Store(tmp_path / "data")
    collect(config, store, then, 14, client=Frozen())
    assert {record.id for record in store.records.values() if record.kind != "lead"} == {
        "github:some/framework:issue:66753",
    }
    prior = {key: store.state[key] for key in ("github", "manual")}
    store = Store(tmp_path / "fresh")
    collect(config, store, then, client=Frozen(), channels=["arxiv"])
    assert store.state["arxiv"]["window_start"] == iso(then - timedelta(days=30))
    store = Store(tmp_path / "data")
    collect(config, store, then + timedelta(hours=1), 30, client=Frozen(), channels=["arxiv", "hf_daily"])
    assert {key: store.state[key] for key in prior} == prior
    assert len(store.works[f"arxiv:{base}"]["records"]) == 2
    assert f"arxiv:{base}" in select_candidates(pool(store, keywords, then + timedelta(hours=1)))[0]


def test_bad_leads_are_isolated_and_channel_state_is_saved(config, tmp_path, caplog):
    channels(config, "arxiv")
    leads = config / "leads.yaml"
    leads.write_text("leads: [{url: 'file://private-marker'}, {}, {url: 'https://example.org/good'}]", encoding="utf-8")
    store = Store(tmp_path / "data")
    collect(config, store, NOW, client=Papers())
    assert {record.kind for record in Store(store.path).records.values()} == {"paper", "lead"}
    assert store.state["arxiv"]["cursor"] == iso(NOW)
    assert store.state["manual"]["counts"]["failed"] == 2
    assert "第 1 条跳过" in caplog.text and "private-marker" not in caplog.text
    leads.write_text("leads: [private-marker", encoding="utf-8")
    collect(config, store, NOW + timedelta(hours=1), client=Papers())
    saved = Store(store.path)
    assert saved.state["arxiv"]["cursor"] == iso(NOW + timedelta(hours=1))
    assert saved.state["manual"]["counts"]["failed"] == 1
    assert "private-marker" not in caplog.text


def add_records(store, *records):
    for record in records:
        store.records[record.key] = record


def test_hf_to_arxiv_and_carryover_keep_first_observation(tmp_path, keywords):
    store = Store(tmp_path)
    url = "https://arxiv.org/abs/2601.12345"
    title, summary = "Speculative decoding", "Parallel draft work"
    first = NOW - timedelta(days=9)
    hf = Record("hf:2601.12345", "paper_listing", title, url, summary, "hf_daily", observed_at=iso(first))
    arxiv = Record("arxiv:2601.12345", "paper", title, url, summary, "arxiv", revision="v1", observed_at=iso(NOW))
    add_records(store, hf)
    key = arxiv.id
    hf_only = pool(store, keywords, first)
    store.works[key]["weeks"] = {week_id(first): {"versions": [hf.key], "selected": False}}
    add_records(store, arxiv)
    current = pool(store, keywords)
    assert current[key]["first_seen"] == first and current[key]["carryover"]
    assert current[key]["hash"] == hf_only[key]["hash"]
    assert "weeks" not in store.works[key] and "last_reviewed_hash" not in store.works[key]
    assert key in pool(store, keywords, first + timedelta(days=14))
    assert key not in pool(store, keywords, first + timedelta(days=15))
    store.works[key]["last_reviewed_hash"] = current[key]["hash"]
    store.works[key]["last_selected_week"] = week_id(first)
    assert key not in pool(store, keywords)
    revised = Record("arxiv:2601.12345", "paper", title, url, summary, "arxiv", revision="v2", observed_at=iso(NOW))
    add_records(store, revised)
    assert not pool(store, keywords)[key]["reviewed"]
    assert pool(store, keywords)[key]["first_seen"] == NOW


def test_merged_work_requires_evaluating_the_combined_content(tmp_path, keywords):
    store = Store(tmp_path)
    pr = Record("github:some/framework:pr:1", "pr", "Speculative decoding",
                "https://github.com/some/framework/pull/1", "Evaluate draft tokens", "github", observed_at=iso(NOW))
    paper = Record("arxiv:2601.12345", "paper", "Speculative decoding",
                   "https://arxiv.org/abs/2601.12345", "New parallel draft", "arxiv", revision="v1", observed_at=iso(NOW),
                   upstream={"project_links": [pr.id]})
    add_records(store, pr)
    candidate = pool(store, keywords)[pr.id]
    store.works[pr.id]["weeks"] = {WEEK: {"versions": [pr.key], "selected": True}}
    add_records(store, paper)
    merged = pool(store, keywords)[paper.id]
    assert merged["retained"] and not merged["reviewed"]
    assert store.works[paper.id]["last_reviewed_hash"] == candidate["hash"]
    assert store.works[paper.id]["last_selected_week"] == WEEK


def test_model_quota_soft_limit_and_independent_display():
    candidates = {}
    for group, size in (("engineering", 20), ("paper", 16), ("manual", 4)):
        for number in range(size):
            candidates[f"{group}-{number}"] = {
                "group": group, "repo": f"org/repo-{number % 3}" if group == "engineering" else None,
                "retained": False, "reviewed": False, "carryover": False,
            }
    selected, reasons = select_candidates(candidates)
    assert Counter(reasons.values()) == {"论文名额": 14, "工程名额": 12, "人工线索": 4}
    assert max(Counter(item["repo"] for item in selected.values() if item["repo"]).values()) <= 5
    engineering = {key: {**item, "repo": "org/same"} for key, item in candidates.items()
                   if item["group"] == "engineering"}
    _, reasons = select_candidates(engineering)
    assert "放宽仓库上限" in reasons["engineering-5"]
    full = {f"display-{i}": {**engineering["engineering-0"], "reviewed": True} for i in range(500)}
    assert display_candidates(full, set()) is full
    full.update(candidates)
    sends, _ = select_candidates(full)
    shown = display_candidates(full, sends)
    assert len(shown) == 500 and sends.keys() <= shown.keys()
    assert select_candidates(full)[0] == sends


def test_only_main_work_is_selected_and_cli_emits_one_path(config, tmp_path, monkeypatch, capsys):
    channels(config, "arxiv")
    store = Store(tmp_path / "data")
    url = "https://arxiv.org/abs/"
    ids = ("arxiv:2601.12345", "arxiv:2601.12346", "arxiv:2601.12347")
    for key in ids:
        add_records(store, Record(key, "paper", "Sparse attention",
                                  url + key[6:], "example | with\nline breaks", "arxiv", observed_at=iso(NOW),
                                  upstream={"verification": {"status": "作者报告"}} if key == ids[1] else {}))
    rebuild(store)
    store.save()

    def complete(request, timeout):
        offered = json.loads(json.loads(request.data)["messages"][1]["content"])
        selected = list(offered)[:2]
        item = {"priority": "值得关注", "works": selected, "title": "稀疏注意力", "change": "新方法",
                "value": "缓存读取", "conditions": "评测条件未知", "next_step": "检查实现"}
        return BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({"items": [item]})}}]}).encode())

    monkeypatch.setenv("RADAR_DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr("radar.report.urlopen", complete)
    monkeypatch.setattr("radar.cli.datetime", type("FrozenTime", (), {"now": staticmethod(lambda tz: NOW)}))
    out = tmp_path / "reports"
    assert main(["--data-dir", str(store.path), "--config-dir", str(config),
                 "report", "--week", WEEK, "--backend", "deepseek", "--output-dir", str(out)]) == 0
    path = out / f"{WEEK}.md"
    assert capsys.readouterr().out.strip() == str(path)
    listing = (out / "candidates" / path.name).read_text(encoding="utf-8")
    assert "验证 作者报告" in path.read_text(encoding="utf-8")
    assert listing.count("| 已精选 |") == 1 and listing.count("| 送模未选 |") == 2
    assert r"example \| with line breaks" in listing and f"](../{path.name})" in listing
    works = Store(store.path).works
    assert all(work.get("last_reviewed_hash") for work in works.values())
    assert [key for key in works if works[key].get("last_selected_week") == WEEK] == [ids[0]]


def test_deepseek_retry_keeps_all_five_prior_main_works(config, tmp_path, keywords, monkeypatch):
    store = Store(tmp_path / "data")
    for number in range(6):
        base = f"2601.1234{number}"
        add_records(store, Record(f"arxiv:{base}", "paper", "Sparse attention",
                                  f"https://arxiv.org/abs/{base}", "A new indexer.",
                                  "arxiv", revision="v1", observed_at=iso(NOW)))
    retained = list(pool(store, keywords))[:5]
    for key in retained:
        store.works[key]["last_selected_week"] = WEEK
    monkeypatch.setenv("RADAR_DEEPSEEK_API_KEY", "test-deepseek-key")
    calls = []

    def complete(request, timeout):
        messages = json.loads(request.data)["messages"]
        system = messages[0]["content"]
        calls.append(system)
        offered = json.loads(messages[1]["content"])
        assert [key for key, item in offered.items() if item["retain"]] == [f"W{n}" for n in range(1, 6)]
        assert 'works 字段依次为 ["W1"]、["W2"]、["W3"]、["W4"]、["W5"]' in system
        groups = [["W1", "W2"], ["W3"], ["W4"], ["W5"]] if len(calls) == 1 else [
            [f"W{n}"] for n in range(1, 6)
        ]
        items = [{"priority": "值得关注", "works": group, "title": "稀疏注意力", "change": "新方法",
                  "value": "缓存读取", "conditions": "评测条件未知", "next_step": "检查实现"} for group in groups]
        return BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({"items": items})}}]}).encode())

    monkeypatch.setattr("radar.report.urlopen", complete)
    path = report(store, tmp_path, WEEK, NOW, "deepseek", PROMPT, config)
    assert len(calls) == 2 and "同周重跑遗漏了已有主 Work（W2）" in calls[1]
    assert "成功评估 6 · 重点方向 5" in path.read_text(encoding="utf-8")
    assert [key for key in store.works if store.works[key].get("last_selected_week") == WEEK] == retained
