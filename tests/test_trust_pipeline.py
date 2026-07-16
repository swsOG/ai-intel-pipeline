from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
        "r/MachineLearning": "discovery",
    }
    policies = {f["name"]: f for feeds in pr.SOURCE_FEEDS.values() for f in feeds}
    for name, source_class in expected.items():
        assert policies[name]["source_class"] == source_class
        assert 1 <= policies[name]["trust_weight"] <= 5
        assert policies[name]["rationale"]


def test_canonicalization_and_immutable_stable_id():
    a = "HTTPS://WWW.Example.com:443/path/?utm_source=x&b=2&a=1#frag"
    b = "https://example.com/path?a=1&b=2"
    assert pr.canonicalize_url(a) == pr.canonicalize_url(b)
    assert pr.stable_item_id(a) == pr.stable_item_id(b)
    assert pr.canonicalize_url("javascript:alert(1)") == ""


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

    def generate_content(self, **kwargs):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def test_classifier_rejects_unknown_duplicate_and_model_altered_ids():
    one = norm("OpenAI Blog")
    two = norm("Anthropic Blog", "https://example.com/two")
    reply = pr.json.dumps([
        signal(one), signal(one, relevance=0), signal(two, item_id="made-up"),
        signal(two), {**signal(two), "relevance": 4},
    ])
    ranked = pr.classify([one, two], "prompt", client=FakeClient([reply]), sleep_seconds=0)
    selected = ranked["tier1"]
    assert [x["item_id"] for x in selected] == [one["item_id"], two["item_id"]]
    assert selected[0]["title"] == one["title"]  # local identity, never model identity


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


def test_prompt_has_bounded_id_only_contract_and_personal_interests():
    prompt = build_system_prompt({})
    for text in ("item_id", "0 to 3", "AI agents", "context engineering", "memory",
                 "tool use", "orchestration", "evaluation", "AI coding", "anti-hype"):
        assert text.lower() in prompt.lower()
    assert '"title"' not in prompt.split("Respond ONLY", 1)[-1]
