from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app as app_module
import pipeline_runner as pr
from app import build_system_prompt


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def raw(title="Useful agent release", url="https://example.com/post", published=None):
    return {"title": title, "url": url, "summary": "Evidence", "published_at": published}


def norm(source_name, url="https://example.com/post", published=None, title="Useful agent release"):
    source = next(
        feed
        for feeds in pr.SOURCE_FEEDS.values()
        for feed in feeds
        if feed["name"] == source_name
    )
    return pr.normalize_item(raw(title, url, published), source, fetched_at=NOW)


def signal(item, **overrides):
    value = {
        "item_id": item["item_id"],
        "relevance": 3,
        "actionability": 3,
        "novelty": 3,
        "hype_penalty": 0,
        "confidence": 3,
        "reason": "Directly useful evidence.",
        "action": "Read the release notes.",
    }
    value.update(overrides)
    return value


def test_source_policy_starter_set_is_deterministic():
    expected = {
        "Anthropic Blog": "official", "OpenAI Blog": "official",
        "Google AI Blog": "official", "Meta AI Blog": "official",
        "Hugging Face Blog": "official", "Simon Willison": "expert",
        "Lilian Weng": "expert", "Ars Technica AI": "reporting",
        "TechCrunch AI": "reporting", "The Verge AI": "reporting",
        "The Batch": "reporting", "GitHub Trending": "discovery",
        "Hacker News AI": "discovery", "Product Hunt": "discovery",
        "r/MachineLearning": "discovery", "r/LocalLLaMA": "discovery",
        "r/artificial": "discovery", "ArXiv AI": "primary",
    }
    policies = {f["name"]: f for feeds in pr.SOURCE_FEEDS.values() for f in feeds}
    for name, source_class in expected.items():
        assert policies[name]["source_class"] == source_class
        assert 1 <= policies[name]["trust_weight"] <= 5
        assert policies[name]["rationale"]
    assert set(policies) == set(expected)


def test_normalization_uses_utc_timestamps_and_source_evidence_policy():
    source = next(feed for feed in pr.SOURCE_FEEDS["RSS Blogs"] if feed["name"] == "OpenAI Blog")
    item = pr.normalize_item(
        raw(published="2026-07-16T13:30:00+02:00"),
        source,
        fetched_at=datetime(2026, 7, 16, 8, 0),
    )
    assert item["published_at"] == "2026-07-16T11:30:00+00:00"
    assert item["fetched_at"] == "2026-07-16T08:00:00+00:00"
    assert item["evidence_level"] == "primary"
    assert item["source_policy"] == {
        "source_class": "official",
        "trust_weight": 5,
        "rationale": "First-party OpenAI announcements.",
    }


def test_reddit_atom_normalizes_destination_and_preserves_discussion_url(monkeypatch):
    atom = b'''<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>r/LocalLLaMA</title>
      <entry>
        <title>Linked evidence</title>
        <link href="https://www.reddit.com/r/LocalLLaMA/comments/abc/linked_evidence/" />
        <updated>2026-07-16T12:00:00+00:00</updated>
        <content type="html">&lt;span&gt;&lt;a href="HTTPS://WWW.Example.com/report/?utm_source=reddit&amp;amp;a=1#comments"&gt;[link]&lt;/a&gt;&lt;/span&gt; &lt;a href="https://www.reddit.com/r/LocalLLaMA/comments/abc/linked_evidence/"&gt;[comments]&lt;/a&gt;</content>
      </entry>
    </feed>'''
    bounded = Mock(return_value=atom)
    monkeypatch.setattr(pr, "_get_bounded", bounded)
    source = next(feed for feed in pr.SOURCE_FEEDS["Reddit"] if feed["name"] == "r/LocalLLaMA")
    item = pr.fetch_rss(source)[0]
    bounded.assert_called_once_with("https://www.reddit.com/r/LocalLLaMA/.rss")
    assert item["url"] == "HTTPS://WWW.Example.com/report/?utm_source=reddit&a=1#comments"
    assert item["discussion_url"] == "https://www.reddit.com/r/LocalLLaMA/comments/abc/linked_evidence/"
    assert item["published_at"] == "2026-07-16T12:00:00+00:00"
    assert item["evidence_level"] == "community discovery"


def test_reddit_sources_are_atom_rss_and_use_bounded_rss_collector(monkeypatch):
    sources = pr.SOURCE_FEEDS["Reddit"]
    assert all(source["type"] == "rss" for source in sources)
    assert all(source["url"] == f"https://www.reddit.com/{source['name']}/.rss" for source in sources)
    rss = Mock(return_value=[])
    monkeypatch.setattr(pr, "fetch_rss", rss)
    assert not hasattr(pr, "fetch_reddit")
    pr.fetch_all_for_pipeline({"sources": ["Reddit"]})
    assert [call.args[0] for call in rss.call_args_list] == sources


def test_canonicalization_and_immutable_stable_id():
    a = "HTTPS://WWW.Example.com:443/path/?utm_source=x&b=2&a=1#frag"
    tracked_variant = "https://www.example.com:443/path/?b=2&a=1"
    semantic_variant = "https://example.com/path?a=1&b=2"
    assert pr.canonicalize_url(a) == tracked_variant
    assert pr.stable_item_id(a) == pr.stable_item_id(tracked_variant)
    assert pr.stable_item_id(a) != pr.stable_item_id(semantic_variant)
    assert pr.stable_item_id(a) == pr.hashlib.sha256(tracked_variant.encode()).hexdigest()[:20]
    assert len(pr.stable_item_id(a)) == 20
    assert all(character in "0123456789abcdef" for character in pr.stable_item_id(a))
    assert pr.canonicalize_url("javascript:alert(1)") == ""


def test_short_item_ids_remain_distinct_for_distinct_canonical_urls():
    urls = [f"https://example.com/releases/{index}" for index in range(1000)]
    item_ids = [pr.stable_item_id(url) for url in urls]
    assert len(set(item_ids)) == len(urls)
    assert max(map(len, item_ids)) == 20


def test_web_only_high_value_sources_remain_in_policy_but_are_not_fetched(monkeypatch):
    expected = {
        "Anthropic Blog": "https://www.anthropic.com/news",
        "Meta AI Blog": "https://ai.meta.com/blog/",
        "The Batch": "https://www.deeplearning.ai/the-batch/",
    }
    registry = {source["name"]: source for source in pr.SOURCE_FEEDS["RSS Blogs"]}
    for name, url in expected.items():
        source = registry[name]
        assert source["url"] == url
        assert source["type"] == "web_only"
        assert source["rationale"] and source["trust_weight"] >= 3

    rss = Mock(return_value=[])
    monkeypatch.setattr(pr, "fetch_rss", rss)
    pr.fetch_all_for_pipeline({"sources": ["RSS Blogs"]})
    fetched_names = [call.args[0]["name"] for call in rss.call_args_list]
    assert not set(expected) & set(fetched_names)
    assert set(fetched_names) == {
        source["name"] for source in pr.SOURCE_FEEDS["RSS Blogs"] if source["type"] == "rss"
    }


def test_freshness_invalid_filter_and_undated_item():
    fresh = norm("OpenAI Blog", published=(NOW - timedelta(days=2)).isoformat())
    stale = norm("OpenAI Blog", url="https://example.com/old", published=(NOW - timedelta(days=8)).isoformat())
    undated = norm("OpenAI Blog", url="https://example.com/undated")
    invalid = dict(fresh, title="", item_id="different")
    assert pr.filter_and_dedupe([fresh, stale, undated, invalid], now=NOW) == [fresh, undated]


def test_primary_source_dedupe_preserves_alternate_and_discussion_provenance():
    official = norm("OpenAI Blog")
    reddit = norm("r/MachineLearning")
    reddit["discussion_url"] = "https://reddit.com/r/MachineLearning/comments/abc/story"
    result = pr.filter_and_dedupe([reddit, official], now=NOW)
    assert len(result) == 1
    assert result[0]["source"] == "OpenAI Blog"
    assert result[0]["alternate_sources"][0]["source"] == "r/MachineLearning"
    assert result[0]["discussion_urls"] == [reddit["discussion_url"]]


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.models = self
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def alias_signal(reference, item, **overrides):
    return signal(item, item_id=reference, **overrides)


def test_classifier_maps_exact_batch_aliases_to_complex_stable_ids_and_hides_hashes():
    one = norm("OpenAI Blog")
    two = norm("Anthropic Blog", "https://example.com/two")
    assert len(one["item_id"]) == len(two["item_id"]) == 20
    client = FakeClient([
        pr.json.dumps([alias_signal("i001", one)]),
        pr.json.dumps([alias_signal("i001", two)]),
    ])
    ranked = pr.classify([one, two], "prompt", client=client, batch_size=1, sleep_seconds=0)
    assert [x["item_id"] for x in ranked["tier1"]] == [one["item_id"], two["item_id"]]
    assert ranked["tier1"][0]["title"] == one["title"]  # identity always stays local
    payloads = [pr.json.loads(call["contents"]) for call in client.calls]
    assert [[value["item_id"] for value in payload] for payload in payloads] == [["i001"], ["i001"]]
    for call in client.calls:
        assert one["item_id"] not in call["contents"]
        assert two["item_id"] not in call["contents"]


@pytest.mark.parametrize("bad_reference", ["i003", "i00l", "I001", "i001 "])
def test_classifier_rejects_unknown_or_mutated_batch_alias(bad_reference):
    item = norm("OpenAI Blog")
    response = pr.json.dumps([alias_signal(bad_reference, item)])
    with pytest.raises(pr.ClassificationError, match="unknown"):
        pr.classify([item], "prompt", client=FakeClient([response]), sleep_seconds=0)


def test_classifier_rejects_duplicate_and_missing_batch_aliases():
    one = norm("OpenAI Blog")
    two = norm("Anthropic Blog", "https://example.com/two")
    duplicate = pr.json.dumps([
        alias_signal("i001", one), alias_signal("i001", one, relevance=0),
        alias_signal("i002", two),
    ])
    with pytest.raises(pr.ClassificationError, match="duplicate"):
        pr.classify([one, two], "prompt", client=FakeClient([duplicate]), sleep_seconds=0)

    missing = pr.json.dumps([alias_signal("i001", one)])
    with pytest.raises(pr.ClassificationError, match="missing"):
        pr.classify([one, two], "prompt", client=FakeClient([missing]), sleep_seconds=0)


@pytest.mark.parametrize("field,bad_value", [
    ("relevance", -1), ("relevance", 4),
    ("actionability", -1), ("actionability", 4),
    ("novelty", -1), ("novelty", 4),
    ("hype_penalty", -1), ("hype_penalty", 4),
    ("confidence", -1), ("confidence", 4),
    ("confidence", True), ("confidence", 2.5),
])
def test_signal_range_and_integer_validation_is_independent_of_duplicate_ids(field, bad_value):
    item = norm("OpenAI Blog", url=f"https://example.com/range/{field}/{bad_value}")
    assert pr.rank_items([item], [signal(item, **{field: bad_value})])["scored"] == []


def test_global_scoring_strict_gates_and_caps():
    items = [norm("OpenAI Blog", f"https://example.com/{i}") for i in range(20)]
    signals = [signal(item) for item in items]
    result = pr.rank_items(items, signals)
    assert len(result["tier1"]) == 5
    assert len(result["tier2"]) == 10
    assert result["tier1"][0]["score"] >= result["tier1"][-1]["score"]


def test_community_title_only_hype_cannot_be_tier1():
    community = norm("GitHub Trending", title="SHOCKING viral agent changes everything!!!")
    result = pr.rank_items([community], [signal(community)])
    assert not result["tier1"]
    assert result["tier2"]


@pytest.mark.parametrize("source_name,overrides", [
    ("GitHub Trending", {}),
    ("OpenAI Blog", {"relevance": 1}),
    ("OpenAI Blog", {"actionability": 1}),
    ("OpenAI Blog", {"confidence": 1}),
    ("OpenAI Blog", {"hype_penalty": 2}),
])
def test_each_tier1_gate_independently_blocks_selection(source_name, overrides):
    item = norm(source_name)
    result = pr.rank_items([item], [signal(item, **overrides)])
    assert result["tier1"] == []


def test_tier1_score_gate_and_inclusive_signal_boundaries():
    item = norm("ArXiv AI")
    tampered = dict(item, trust_weight=1)
    rejected = pr.rank_items([tampered], [signal(tampered)])
    assert rejected["scored"] == []

    accepted = pr.rank_items([item], [signal(
        item, relevance=2, actionability=2, novelty=0, confidence=2, hype_penalty=1,
    )])
    assert accepted["scored"][0]["score"] == 29
    assert accepted["tier1"]


@pytest.mark.parametrize("overrides", [
    {"relevance": 0},
    {"confidence": 0},
    {"relevance": 1, "actionability": 0, "novelty": 0, "confidence": 1, "hype_penalty": 3},
])
def test_each_tier2_gate_blocks_selection(overrides):
    item = norm("GitHub Trending")
    result = pr.rank_items([item], [signal(item, **overrides)])
    assert result["tier1"] == []
    assert result["tier2"] == []


def test_safe_html_and_telegram_rendering():
    item = norm("OpenAI Blog", title='<img src=x onerror=alert(1)> & "news"')
    item.update(signal(item, reason="<script>bad()</script>", action="Use > now"))
    item["url"] = 'javascript:alert(1)'
    classified = {"tier1": [item], "tier2": []}
    html = pr.build_email_html(classified, 1, 1)
    telegram = pr.build_telegram_message(classified, now=NOW)
    assert "<script>" not in html and "javascript:" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert "<img" not in telegram and "javascript:" not in telegram
    assert "official · trust 5/5" in html


def test_all_batch_failure_raises_instead_of_false_quiet_day():
    item = norm("OpenAI Blog")
    with pytest.raises(pr.ClassificationError):
        pr.classify([item], "prompt", client=FakeClient([RuntimeError("down")]), sleep_seconds=0)


def test_all_batches_with_empty_or_invalid_arrays_raise_instead_of_false_quiet_day():
    one = norm("OpenAI Blog")
    two = norm("Anthropic Blog", "https://example.com/two")
    invalid = pr.json.dumps([alias_signal("unknown", two), {"item_id": "i001"}])
    with pytest.raises(pr.ClassificationError, match="missing validated signals"):
        pr.classify(
            [one, two], "prompt", client=FakeClient(["[]", invalid]),
            batch_size=1, sleep_seconds=0,
        )


def test_dry_run_never_delivers_and_writes_collision_safe_audit(monkeypatch, tmp_path):
    item = norm("OpenAI Blog")
    monkeypatch.setattr(pr, "fetch_all_for_pipeline", lambda pipeline: [item])
    monkeypatch.setattr(pr, "classify", lambda items, prompt: pr.rank_items(items, [signal(item)]))
    email = Mock()
    telegram = Mock()
    monkeypatch.setattr(pr, "send_email", email)
    monkeypatch.setattr(pr, "send_telegram", telegram)
    pipeline = {"name": "Trial", "sources": ["RSS Blogs"], "channels": {
        "email": {"on": True, "value": "x@example.com"},
        "telegram": {"on": True, "value": "token:123"},
    }}
    first = pr.run_single_pipeline(pipeline, deliver=False, audit_dir=tmp_path)
    second = pr.run_single_pipeline(pipeline, deliver=False, audit_dir=tmp_path)
    email.assert_not_called(); telegram.assert_not_called()
    assert first["audit_path"] != second["audit_path"]
    assert first["html"].startswith("<!DOCTYPE html>")
    with open(first["audit_path"], encoding="utf-8") as handle:
        audit = pr.json.load(handle)
    assert audit["pipeline"] == "Trial"
    assert audit["delivery_enabled"] is False
    assert audit["total_fetched"] == audit["total_after_filtering"] == 1
    assert audit["sources_checked"] == 1
    assert audit["tier1_count"] == 1 and audit["tier2_count"] == 0
    assert audit["tier1"][0]["item_id"] == item["item_id"]
    assert audit["tier1"][0]["source_policy"] == item["source_policy"]
    assert audit["scored"][0]["reason"] == "Directly useful evidence."


def test_prompt_has_bounded_id_only_contract_and_personal_interests():
    prompt = build_system_prompt({})
    for text in ("item_id", "0 to 3", "AI agents", "context engineering", "memory",
                 "tool use", "orchestration", "evaluation", "AI coding", "anti-hype"):
        assert text.lower() in prompt.lower()
    assert '"title"' not in prompt.split("Respond ONLY", 1)[-1]


def test_arxiv_feed_uses_https():
    assert pr.SOURCE_FEEDS["ArXiv"][0]["url"] == "https://arxiv.org/rss/cs.AI"


def test_canonical_identity_preserves_url_semantics_and_delivery_url():
    url = "HTTPS://WWW.Example.com/path/?ref=first&b=2&a=1&a=0&source=x&utm_medium=no#section"
    assert pr.canonicalize_url(url) == "https://www.example.com/path/?ref=first&b=2&a=1&a=0&source=x"
    item = norm("OpenAI Blog", url=url)
    assert item["url"] == url
    assert item["item_id"] == pr.stable_item_id(url)
    assert pr.stable_item_id("https://www.example.com/path/?ref=first&b=2&a=1&a=0&source=x") == item["item_id"]
    assert pr.stable_item_id("https://example.com/path?b=2&a=0&a=1&source=x&ref=first") != item["item_id"]


def test_partial_batch_failure_and_omitted_ids_abort_classification():
    one = norm("OpenAI Blog")
    two = norm("Anthropic Blog", "https://example.com/two")
    with pytest.raises(pr.ClassificationError, match="batch 2"):
        pr.classify([one, two], "prompt", client=FakeClient([pr.json.dumps([alias_signal("i001", one)]), RuntimeError("down")]), batch_size=1, sleep_seconds=0)
    with pytest.raises(pr.ClassificationError, match="missing"):
        pr.classify([one, two], "prompt", client=FakeClient([pr.json.dumps([alias_signal("i001", one)])]), sleep_seconds=0)


def test_ranking_configuration_and_local_trust_policy_are_validated():
    item = norm("OpenAI Blog")
    escalated = dict(item, trust_weight=1, source_class="discovery")
    assert pr.rank_items([escalated], [signal(escalated)])["scored"] == []
    for kwargs in ({"tier1_cap": -1}, {"tier2_cap": 101}):
        with pytest.raises(ValueError):
            pr.rank_items([item], [signal(item)], **kwargs)
    with pytest.raises(ValueError):
        pr.classify([item], "prompt", client=FakeClient([]), batch_size=0, sleep_seconds=0)


def test_unreasonable_future_timestamp_is_filtered():
    future = norm("OpenAI Blog", published=(NOW + timedelta(days=2)).isoformat())
    assert pr.filter_and_dedupe([future], now=NOW) == []


class TagBalanceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag in {"b", "i", "a"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack.pop() == tag


def test_long_telegram_is_structurally_bounded_with_complete_entities():
    items = []
    for index in range(15):
        item = norm("OpenAI Blog", f"https://example.com/{index}?a=1&b=2", title=("A & <title> " * 100))
        item.update(signal(item, reason="R & <reason> " * 100, action="Do & <action> " * 100))
        items.append(item)
    message = pr.build_telegram_message({"tier1": items[:5], "tier2": items[5:]}, now=NOW)
    assert len(message) <= 4000
    assert not message.endswith(("&", "&a", "&am", "&amp", "&#"))
    parser = TagBalanceParser()
    parser.feed(message)
    parser.close()
    assert parser.stack == []


def test_rss_uses_timeout_bounded_content_and_oversize_is_rejected(monkeypatch):
    response = Mock(headers={"Content-Length": str(pr.MAX_RESPONSE_BYTES + 1)})
    response.raise_for_status.return_value = None
    get = Mock(return_value=response)
    monkeypatch.setattr(pr.requests, "get", get)
    source = pr.SOURCE_FEEDS["ArXiv"][0]
    assert pr.fetch_rss(source) == []
    assert get.call_args.kwargs["timeout"] == pr.REQUEST_TIMEOUT
    assert get.call_args.kwargs["stream"] is True
    response.close.assert_called_once()
    response.json.assert_not_called()

    streamed = Mock(headers={})
    streamed.raise_for_status.return_value = None
    streamed.iter_content.return_value = [b"x" * pr.MAX_RESPONSE_BYTES, b"x"]
    monkeypatch.setattr(pr.requests, "get", Mock(return_value=streamed))
    with pytest.raises(ValueError, match="exceeds"):
        pr._get_bounded(source["url"])
    streamed.close.assert_called_once()


def test_delivery_audit_records_each_channel_and_retry_skips_success(monkeypatch, tmp_path):
    item = norm("OpenAI Blog")
    pipeline = {"name": "Trial", "sources": ["RSS Blogs"], "channels": {
        "email": {"on": True, "value": "x@example.com"},
        "telegram": {"on": True, "value": "token:123"},
    }}
    fetched_runs = iter(([item], []))
    monkeypatch.setattr(pr, "fetch_all_for_pipeline", lambda pipeline: next(fetched_runs))
    monkeypatch.setattr(
        pr, "classify", lambda items, prompt: pr.rank_items(items, [signal(items[0])])
    )
    observed = []
    telegram_item_ids = []

    def email_sender(*args):
        audit_files = list(tmp_path.glob("*.json"))
        assert len(audit_files) == 1
        with audit_files[0].open(encoding="utf-8") as handle:
            observed.append(pr.json.load(handle)["delivery"]["email"]["status"])

    def telegram_sender(classified, config):
        # A retry must deliver the same audited briefing, not newly fetched content.
        telegram_item_ids.append(classified["tier1"][0]["item_id"])
        raise RuntimeError("telegram down")

    email, telegram = Mock(side_effect=email_sender), Mock(side_effect=telegram_sender)
    monkeypatch.setattr(pr, "send_email", email)
    monkeypatch.setattr(pr, "send_telegram", telegram)
    for _ in range(2):
        with pytest.raises(pr.DeliveryError) as error:
            pr.run_single_pipeline(pipeline, audit_dir=tmp_path, delivery_key="trial-2026-07-16")
        with open(error.value.audit_path, encoding="utf-8") as handle:
            audit = pr.json.load(handle)
        assert audit["delivery"]["email"]["status"] == "success"
        assert audit["delivery"]["telegram"]["status"] == "failed"
    email.assert_called_once()
    assert telegram.call_count == 2
    assert telegram_item_ids == [item["item_id"], item["item_id"]]
    assert observed == ["attempting"]


def test_delivery_audit_sanitizes_token_bearing_http_errors(monkeypatch, tmp_path):
    item = norm("OpenAI Blog")
    token = "123456:SUPER-SECRET-TOKEN"
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    pipeline = {"name": "Secret-safe", "sources": ["RSS Blogs"], "channels": {
        "telegram": {"on": True, "value": f"{token}:-100123"},
    }}
    monkeypatch.setattr(pr, "fetch_all_for_pipeline", lambda pipeline: [item])
    monkeypatch.setattr(
        pr, "classify", lambda items, prompt: pr.rank_items(items, [signal(items[0])])
    )
    response = pr.requests.Response()
    response.status_code = 502
    response.request = pr.requests.Request("POST", api_url).prepare()
    exception = pr.requests.HTTPError(
        f"502 Server Error for url: {api_url}", response=response
    )
    monkeypatch.setattr(pr, "send_telegram", Mock(side_effect=exception))

    with pytest.raises(pr.DeliveryError) as error:
        pr.run_single_pipeline(pipeline, audit_dir=tmp_path, delivery_key="secret-safe")

    with open(error.value.audit_path, encoding="utf-8") as handle:
        audit_text = handle.read()
    audit = pr.json.loads(audit_text)
    assert token not in audit_text
    assert api_url not in audit_text
    assert audit["delivery"]["telegram"]["error"] == {
        "channel": "telegram", "category": "http_error", "http_status": 502,
    }


def test_attempting_delivery_is_ambiguous_on_retry_and_is_not_resent(monkeypatch, tmp_path):
    item = norm("OpenAI Blog")
    pipeline = {"name": "Crash-safe", "sources": ["RSS Blogs"], "channels": {
        "email": {"on": True, "value": "recipient@example.com"},
    }}
    monkeypatch.setattr(pr, "fetch_all_for_pipeline", lambda pipeline: [item])
    monkeypatch.setattr(
        pr, "classify", lambda items, prompt: pr.rank_items(items, [signal(items[0])])
    )
    crashing_sender = Mock(side_effect=KeyboardInterrupt("simulated process death"))
    monkeypatch.setattr(pr, "send_email", crashing_sender)

    with pytest.raises(KeyboardInterrupt):
        pr.run_single_pipeline(pipeline, audit_dir=tmp_path, delivery_key="crash-key")
    audit_path = next(tmp_path.glob("*.json"))
    assert pr.json.loads(audit_path.read_text())["delivery"]["email"]["status"] == "attempting"

    replacement_sender = Mock()
    monkeypatch.setattr(pr, "send_email", replacement_sender)
    result = pr.run_single_pipeline(pipeline, audit_dir=tmp_path, delivery_key="crash-key")

    replacement_sender.assert_not_called()
    assert result["status"] == "ambiguous"
    assert result["ambiguous_channels"] == ["email"]
    assert pr.json.loads(audit_path.read_text())["delivery"]["email"]["status"] == "ambiguous"


def test_delivery_key_lock_is_hashed_and_exclusive_across_processes(tmp_path):
    secret_key = "recipient@example.com:123456:BOT-TOKEN"
    lock_path = pr._delivery_lock_path(tmp_path, secret_key)
    assert secret_key not in os.path.basename(lock_path)
    assert "recipient" not in os.path.basename(lock_path)
    probe = (
        "import fcntl, sys; "
        "f = open(sys.argv[1], 'a+'); "
        "\ntry: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)"
        "\nexcept BlockingIOError: raise SystemExit(2)"
        "\nraise SystemExit(0)"
    )

    with pr._delivery_lock(tmp_path, secret_key):
        held = subprocess.run([sys.executable, "-c", probe, lock_path], check=False)
        assert held.returncode == 2
    released = subprocess.run([sys.executable, "-c", probe, lock_path], check=False)
    assert released.returncode == 0

    with pytest.raises(RuntimeError, match="inside claim"):
        with pr._delivery_lock(tmp_path, secret_key):
            raise RuntimeError("inside claim")
    released_after_error = subprocess.run(
        [sys.executable, "-c", probe, lock_path], check=False
    )
    assert released_after_error.returncode == 0


def _successful_delivery_setup(monkeypatch):
    item = norm("OpenAI Blog")
    fetch = Mock(return_value=[item])
    monkeypatch.setattr(pr, "fetch_all_for_pipeline", fetch)
    monkeypatch.setattr(
        pr, "classify", lambda items, prompt: pr.rank_items(items, [signal(items[0])])
    )
    email = Mock()
    monkeypatch.setattr(pr, "send_email", email)
    return fetch, email


def test_delivery_identity_separates_same_name_pipeline_ids(monkeypatch, tmp_path):
    fetch, email = _successful_delivery_setup(monkeypatch)
    base = {"name": "Duplicate", "sources": ["RSS Blogs"], "channels": {
        "email": {"on": True, "value": "same@example.com"},
    }}

    pr.run_single_pipeline({**base, "id": "pipeline-a"}, audit_dir=tmp_path, delivery_key="period-1")
    pr.run_single_pipeline({**base, "id": "pipeline-b"}, audit_dir=tmp_path, delivery_key="period-1")

    assert fetch.call_count == 2
    assert email.call_count == 2
    audits = [pr.json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len({record["delivery_identity"] for record in audits}) == 2


def test_delivery_success_is_bound_to_normalized_destination(monkeypatch, tmp_path):
    fetch, email = _successful_delivery_setup(monkeypatch)
    pipeline = {"id": "stable-id", "name": "Trial", "sources": ["RSS Blogs"], "channels": {
        "email": {"on": True, "value": " FIRST@example.com "},
    }}

    first = pr.run_single_pipeline(pipeline, audit_dir=tmp_path, delivery_key="period-1")
    unchanged = {**pipeline, "channels": {"email": {"on": True, "value": "first@example.com"}}}
    second = pr.run_single_pipeline(unchanged, audit_dir=tmp_path, delivery_key="period-1")
    changed = {**pipeline, "channels": {"email": {"on": True, "value": "other@example.com"}}}
    third = pr.run_single_pipeline(changed, audit_dir=tmp_path, delivery_key="period-1")

    assert first["audit_path"] == second["audit_path"]
    assert third["audit_path"] != first["audit_path"]
    assert fetch.call_count == 2
    assert email.call_count == 2
    first_audit = pr.json.loads(open(first["audit_path"], encoding="utf-8").read())
    third_audit = pr.json.loads(open(third["audit_path"], encoding="utf-8").read())
    assert first_audit["delivery"]["email"]["destination_fingerprint"]
    assert (first_audit["delivery"]["email"]["destination_fingerprint"] !=
            third_audit["delivery"]["email"]["destination_fingerprint"])


def test_secret_delivery_key_is_only_persisted_as_sha256(monkeypatch, tmp_path):
    _successful_delivery_setup(monkeypatch)
    secret_key = "2026-07-16:recipient@example.com:SUPER-SECRET-TOKEN"
    pipeline = {"id": "safe-id", "name": "Safe", "sources": ["RSS Blogs"], "channels": {
        "email": {"on": True, "value": "recipient@example.com"},
    }}

    result = pr.run_single_pipeline(
        pipeline, audit_dir=tmp_path, delivery_key=secret_key
    )
    audit_bytes = open(result["audit_path"], "rb").read()
    audit = pr.json.loads(audit_bytes)

    assert secret_key.encode() not in audit_bytes
    assert b"recipient@example.com" not in audit_bytes
    assert "delivery_key" not in audit
    assert len(audit["delivery_key_fingerprint"]) == 64
    assert len(audit["delivery_identity"]) == 64


def test_logs_redact_sensitive_fields_from_legacy_audits(monkeypatch, tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    secret_key = "period:recipient@example.com:BOT-TOKEN"
    (daily / "legacy.json").write_text(pr.json.dumps({
        "pipeline": "Legacy", "delivery_key": secret_key,
        "delivery": {"email": {"status": "success", "destination": "recipient@example.com"},
                     "telegram": {"status": "failed", "token": "BOT-TOKEN"}},
    }))
    monkeypatch.setattr(app_module, "DATA_DIR", str(tmp_path))
    client = app_module.app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True

    response = client.get("/api/logs")

    assert response.status_code == 200
    assert secret_key.encode() not in response.data
    assert b"recipient@example.com" not in response.data
    assert b"BOT-TOKEN" not in response.data
