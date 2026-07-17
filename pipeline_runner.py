"""Fetch, filter, rank, render, and optionally deliver AI intelligence."""

import fcntl
import hashlib
import html
import json
import os
import smtplib
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import unquote_plus, urlsplit, urlunsplit

import requests
from dotenv import load_dotenv

load_dotenv()


def _source(name, url, kind, source_class, trust_weight, rationale, **metadata):
    return {
        "name": name, "url": url, "type": kind, "source_class": source_class,
        "trust_weight": trust_weight, "rationale": rationale, **metadata,
    }


# These are the project's existing feeds with deterministic policy added. No feed
# URL is inferred or generated at runtime.
SOURCE_FEEDS = {
    "RSS Blogs": [
        _source("Anthropic Blog", "https://www.anthropic.com/news", "web_only", "official", 5, "First-party Anthropic announcements; canonical web page only because no working RSS feed is currently available."),
        _source("OpenAI Blog", "https://openai.com/blog/rss.xml", "rss", "official", 5, "First-party OpenAI announcements."),
        _source("Google AI Blog", "https://blog.google/technology/ai/rss/", "rss", "official", 5, "First-party Google AI announcements."),
        _source("Meta AI Blog", "https://ai.meta.com/blog/", "web_only", "official", 5, "First-party Meta AI announcements; canonical web page only because no working RSS feed is currently available."),
        _source("Simon Willison", "https://simonwillison.net/atom/everything/", "rss", "expert", 4, "Named practitioner's technical analysis."),
        _source("The Batch", "https://www.deeplearning.ai/the-batch/", "web_only", "reporting", 3, "Edited secondary AI reporting; canonical web page only because no working RSS feed is currently available."),
        _source("Lilian Weng", "https://lilianweng.github.io/index.xml", "rss", "expert", 4, "Named researcher's technical analysis."),
        _source("Ars Technica AI", "https://arstechnica.com/ai/feed/", "rss", "reporting", 3, "Edited technology reporting."),
    ],
    "Reddit": [
        _source("r/MachineLearning", "https://www.reddit.com/r/MachineLearning/.rss", "rss", "discovery", 1, "Community discussion discovered through Reddit Atom; verify linked claims.", discussion_feed=True),
        _source("r/LocalLLaMA", "https://www.reddit.com/r/LocalLLaMA/.rss", "rss", "discovery", 1, "Community discussion discovered through Reddit Atom; verify linked claims.", discussion_feed=True),
        _source("r/artificial", "https://www.reddit.com/r/artificial/.rss", "rss", "discovery", 1, "Community discussion discovered through Reddit Atom; verify linked claims.", discussion_feed=True),
    ],
    "GitHub Trending": [_source("GitHub Trending", "https://github.com/trending?since=daily", "github", "discovery", 1, "Popularity-based repository discovery, not verification.")],
    "Hacker News": [_source("Hacker News AI", "https://hnrss.org/newest?q=AI+OR+LLM+OR+GPT&points=50", "rss", "discovery", 1, "Community popularity-based discovery.")],
    "Product Hunt": [_source("Product Hunt", "https://www.producthunt.com/feed", "rss", "discovery", 1, "Launch and popularity-based discovery.")],
    "Hugging Face": [_source("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", "rss", "official", 5, "First-party Hugging Face announcements.")],
    "ArXiv": [_source("ArXiv AI", "https://arxiv.org/rss/cs.AI", "rss", "primary", 5, "Primary research manuscripts; not necessarily peer reviewed.")],
    "TechCrunch": [_source("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "rss", "reporting", 3, "Edited technology reporting.")],
    "The Verge": [_source("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "rss", "reporting", 3, "Edited technology reporting.")],
}

HEADERS = {"User-Agent": "AI-Intel-Pipeline/2.0"}
REQUEST_TIMEOUT = 15
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TRACKING_PARAMS = {"fbclid", "gclid"}
EVIDENCE_LEVELS = {
    "official": "primary", "primary": "primary research", "expert": "expert analysis",
    "reporting": "reporting", "discovery": "community discovery",
}


class ClassificationError(RuntimeError):
    """Raised when Gemini supplies an incomplete or unusable batch response."""


class DeliveryError(RuntimeError):
    """Raised after all configured channels are attempted and one failed."""

    def __init__(self, message, audit_path):
        super().__init__(message)
        self.audit_path = audit_path


def _safe_delivery_url(url):
    """Return the original HTTP(S) URL when safe, without semantic rewriting."""
    if not isinstance(url, str):
        return ""
    value = url.strip()
    if not value or len(value) > 2048 or any(ord(char) < 32 for char in value):
        return ""
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return ""
        if parts.username is not None or parts.password is not None:
            return ""
        parts.port  # force port validation
        return value
    except (ValueError, TypeError):
        return ""


def canonicalize_url(url):
    """Return a conservative identity URL, or an empty string."""
    try:
        safe = _safe_delivery_url(url)
        if not safe:
            return ""
        parts = urlsplit(safe)
        scheme = parts.scheme.lower()
        host = str(parts.hostname).lower()
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        params = []
        for component in parts.query.split("&") if parts.query else []:
            encoded_key = component.split("=", 1)[0]
            key = unquote_plus(encoded_key).lower()
            if key.startswith("utm_") or key in TRACKING_PARAMS:
                continue
            params.append(component)
        return urlunsplit((scheme, host, parts.path, "&".join(params), ""))
    except (ValueError, TypeError):
        return ""


def stable_item_id(url):
    canonical = canonicalize_url(url)
    # 80 bits keeps model round trips manageable while retaining ample collision
    # resistance for this local feed corpus. Identity is still derived locally.
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20] if canonical else ""


def _iso_timestamp(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, timezone.utc)
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def normalize_item(raw, source, fetched_at=None):
    """Attach local, immutable identity and source/evidence policy."""
    fetched = fetched_at or datetime.now(timezone.utc)
    original_url = _safe_delivery_url(raw.get("url", ""))
    policy = {key: source[key] for key in ("source_class", "trust_weight", "rationale")}
    return {
        "item_id": stable_item_id(original_url),
        "title": str(raw.get("title") or "").strip()[:300],
        "url": original_url,
        "summary": str(raw.get("summary") or "").strip()[:1000],
        "source": source["name"],
        "source_policy": policy,
        **policy,
        "evidence_level": EVIDENCE_LEVELS[source["source_class"]],
        "published_at": _iso_timestamp(raw.get("published_at")),
        "fetched_at": _iso_timestamp(fetched),
        "discussion_url": _safe_delivery_url(raw.get("discussion_url", "")),
        "alternate_sources": [],
        "discussion_urls": [],
    }


def _date_from_feed(entry):
    value = entry.get("published_parsed") or entry.get("updated_parsed")
    if value:
        return datetime(*value[:6], tzinfo=timezone.utc).isoformat()
    return entry.get("published") or entry.get("updated")


def _get_bounded(url, max_bytes=MAX_RESPONSE_BYTES):
    """Fetch a response with timeout and enforce a byte bound while streaming."""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
    try:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        if declared:
            try:
                if int(declared) > max_bytes:
                    raise ValueError(f"response exceeds {max_bytes} bytes")
            except ValueError as exc:
                if "exceeds" in str(exc):
                    raise
        chunks = []
        size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"response exceeds {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        response.close()


def fetch_rss(source):
    try:
        import feedparser
        from bs4 import BeautifulSoup
        content = _get_bounded(source["url"])
        feed = feedparser.parse(content)
        result = []
        for entry in feed.entries[:15]:
            raw_summary = entry.get("summary", "")
            if entry.get("content"):
                raw_summary = entry["content"][0].get("value", raw_summary)
            summary_soup = BeautifulSoup(raw_summary, "html.parser")
            summary = summary_soup.get_text()[:300]
            link = entry.get("link", "")
            discussion_url = ""
            if source.get("discussion_feed"):
                discussion_url = link
                # Reddit Atom labels the external destination '[link]'. Self-posts
                # intentionally fall back to the Reddit discussion URL.
                destination = next((
                    anchor.get("href", "") for anchor in summary_soup.find_all("a")
                    if anchor.get_text(" ", strip=True).lower() == "[link]"
                    and _safe_delivery_url(anchor.get("href", ""))
                ), "")
                link = destination or discussion_url
            result.append(normalize_item({
                "title": entry.get("title", "Untitled"), "url": link,
                "summary": summary.strip(), "published_at": _date_from_feed(entry),
                "discussion_url": discussion_url,
            }, source))
        return result
    except Exception as exc:
        print(f"  [ERROR] RSS: {source['name']}: {exc}")
        return []


def fetch_github(source):
    try:
        from bs4 import BeautifulSoup
        content = _get_bounded(source["url"])
        soup = BeautifulSoup(content, "html.parser")
        result = []
        for article in soup.select("article.Box-row")[:15]:
            link = article.select_one("h2 a")
            if not link:
                continue
            repo = link.get("href", "").strip("/")
            description = article.select_one("p")
            result.append(normalize_item({
                "title": repo.replace("/", " / "), "url": f"https://github.com/{repo}",
                "summary": description.get_text(strip=True)[:300] if description else "",
            }, source))
        return result
    except Exception as exc:
        print(f"  [ERROR] GitHub: {exc}")
        return []


def fetch_all_for_pipeline(pipeline):
    items = []
    for key in pipeline.get("sources", []):
        for source in SOURCE_FEEDS.get(key, []):
            if source["type"] == "web_only":
                print(f"  Skipping web-only source: {source['name']}")
                continue
            print(f"  Fetching: {source['name']}...")
            fetcher = {"rss": fetch_rss, "github": fetch_github}.get(source["type"])
            fetched = fetcher(source) if fetcher else []
            print(f"    -> {len(fetched)} items")
            items.extend(fetched)
    print(f"  Total fetched: {len(items)}")
    return items


def _valid_item(item):
    if not isinstance(item, dict):
        return False
    policies = {source["name"]: source for feeds in SOURCE_FEEDS.values() for source in feeds}
    local = policies.get(item.get("source"))
    if not local:
        return False
    expected_policy = {key: local[key] for key in ("source_class", "trust_weight", "rationale")}
    return bool(
        item.get("title") and _safe_delivery_url(item.get("url"))
        and item.get("item_id") == stable_item_id(item.get("url"))
        and item.get("source_policy") == expected_policy
        and all(item.get(key) == value for key, value in expected_policy.items())
        and item.get("evidence_level") == EVIDENCE_LEVELS[local["source_class"]]
    )


def filter_and_dedupe(items, now=None, max_age_days=7):
    """Filter stale/invalid records and retain the strongest duplicate evidence."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    groups = {}
    for item in items:
        if not _valid_item(item):
            continue
        published = _iso_timestamp(item.get("published_at"))
        if published:
            published_dt = datetime.fromisoformat(published)
            if published_dt < cutoff or published_dt > now + timedelta(days=1):
                continue
        groups.setdefault(item["item_id"], []).append(item)

    result = []
    class_rank = {"official": 5, "primary": 4, "expert": 3, "reporting": 2, "discovery": 1}
    for group in groups.values():
        ordered = sorted(group, key=lambda value: (
            value["trust_weight"], class_rank[value["source_class"]], value["source"]
        ), reverse=True)
        chosen = dict(ordered[0])
        alternates = []
        discussions = []
        for candidate in ordered:
            discussion = _safe_delivery_url(candidate.get("discussion_url", ""))
            if discussion and discussion not in discussions:
                discussions.append(discussion)
        for candidate in ordered[1:]:
            alternates.append({
                "source": candidate["source"], "source_class": candidate["source_class"],
                "trust_weight": candidate["trust_weight"], "url": candidate["url"],
                "discussion_url": candidate.get("discussion_url", ""),
            })
        chosen["alternate_sources"] = alternates
        chosen["discussion_urls"] = discussions
        result.append(chosen)
    return sorted(result, key=lambda value: value["item_id"])


def _valid_signal(value, known_ids, seen):
    if not isinstance(value, dict) or value.get("item_id") not in known_ids or value.get("item_id") in seen:
        return False
    for key in ("relevance", "actionability", "novelty", "hype_penalty", "confidence"):
        signal = value.get(key)
        if isinstance(signal, bool) or not isinstance(signal, int) or not 0 <= signal <= 3:
            return False
    return isinstance(value.get("reason"), str) and isinstance(value.get("action"), str)


def _concrete_action(action):
    """Return whether an action goes beyond passive reading/monitoring."""
    value = str(action or "").strip().lower()
    if not value:
        return False
    passive_starts = (
        "read ", "skim ", "review ", "study ", "monitor ", "watch ", "track ",
        "look into ", "check out ", "keep an eye ", "be aware ",
        "prioritize reading ", "prioritise reading ",
        "prioritize reviewing ", "prioritise reviewing ",
    )
    return not value.startswith(passive_starts)


def _tier1_eligible(item):
    if not (
        item["source_class"] != "discovery" and item["relevance"] >= 2
        and item["actionability"] >= 2 and item["confidence"] >= 2
        and item["hype_penalty"] <= 1 and item["score"] >= 25
        and _concrete_action(item.get("action"))
    ):
        return False
    if item["source_class"] == "primary":
        # Research papers are easy to over-rank as "Act now". Require a
        # genuinely strong signal before promoting them above "Worth knowing".
        return (
            item["relevance"] == 3 and item["actionability"] == 3
            and item["confidence"] == 3 and item["hype_penalty"] == 0
        )
    return True


def _select_diverse(items, cap, max_per_source=2):
    selected = []
    source_counts = {}
    for item in items:
        source = item.get("source", "")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        selected.append(item)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= cap:
            break
    return selected


def rank_items(items, signals, tier1_cap=5, tier2_cap=10):
    """Attach validated model signals, then rank globally using local policy."""
    for name, cap in (("tier1_cap", tier1_cap), ("tier2_cap", tier2_cap)):
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 <= cap <= 100:
            raise ValueError(f"{name} must be an integer from 0 to 100")
    by_id = {item["item_id"]: item for item in items if _valid_item(item)}
    seen = set()
    scored = []
    for model_value in signals:
        if not _valid_signal(model_value, by_id, seen):
            continue
        item_id = model_value["item_id"]
        seen.add(item_id)
        item = dict(by_id[item_id])
        for key in ("relevance", "actionability", "novelty", "hype_penalty", "confidence"):
            item[key] = model_value[key]
        item["reason"] = model_value["reason"].strip()[:500]
        item["action"] = model_value["action"].strip()[:500]
        item["score"] = (
            item["trust_weight"] * 3 + item["relevance"] * 4 + item["actionability"] * 3
            + item["novelty"] * 2 + item["confidence"] * 2 - item["hype_penalty"] * 4
        )
        scored.append(item)
    scored.sort(key=lambda value: (-value["score"], -value["trust_weight"], value["item_id"]))

    tier1_candidates = [item for item in scored if _tier1_eligible(item)]
    tier1 = _select_diverse(tier1_candidates, tier1_cap)
    used = {item["item_id"] for item in tier1}
    tier2_candidates = [item for item in scored if (
        item["item_id"] not in used and item["relevance"] >= 1
        and item["confidence"] >= 1 and item["score"] >= 10
    )]
    tier2 = _select_diverse(tier2_candidates, tier2_cap)
    return {"tier1": tier1, "tier2": tier2, "scored": scored}


def classify(items, system_prompt, client=None, batch_size=25, sleep_seconds=5):
    """Request bounded signals only; require one validated signal per item."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be an integer from 1 to 100")
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        from google import genai
        client = genai.Client(api_key=api_key)
    batches = [items[index:index + batch_size] for index in range(0, len(items), batch_size)]
    all_signals = []
    seen = set()
    for index, batch in enumerate(batches):
        references = {f"i{position:03d}": item["item_id"] for position, item in enumerate(batch, 1)}
        payload = [{
            "item_id": reference, "title": item["title"], "summary": item["summary"][:500],
            "source": item["source"], "source_class": item["source_class"],
            "evidence_level": item["evidence_level"],
        } for reference, item in zip(references, batch)]
        known = set(references.values())
        returned_references = set()
        batch_signals = []
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=json.dumps(payload),
                config={"system_instruction": system_prompt, "response_mime_type": "application/json", "temperature": 0.1},
            )
            values = json.loads(response.text)
            if not isinstance(values, list):
                raise ValueError("Gemini response must be an array")
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError("malformed signal")
                reference = value.get("item_id")
                if reference not in references:
                    raise ValueError("unknown model item reference")
                if reference in returned_references:
                    raise ValueError("duplicate model item reference")
                returned_references.add(reference)
                mapped_value = dict(value, item_id=references[reference])
                if not _valid_signal(mapped_value, known, seen):
                    raise ValueError("malformed or duplicate signal")
                seen.add(mapped_value["item_id"])
                batch_signals.append(mapped_value)
            missing = set(references) - returned_references
            if missing:
                raise ValueError(f"missing validated signals for {len(missing)} item(s)")
        except Exception as exc:
            raise ClassificationError(
                f"Gemini batch {index + 1} failed or was incomplete: {exc}; briefing not delivered"
            ) from exc
        all_signals.extend(batch_signals)
        if index < len(batches) - 1 and sleep_seconds:
            time.sleep(sleep_seconds)
    return rank_items(items, all_signals)


def _safe_url(value):
    return _safe_delivery_url(value)


def _label(item):
    return f"{item.get('source_class', 'unknown')} · trust {item.get('trust_weight', '?')}/5"


def build_email_html(classified, total_fetched, sources_count):
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    tier1 = classified.get("tier1", [])
    tier2 = classified.get("tier2", [])

    def stat(value, label, accent="#0B53CC"):
        return (
            '<td style="padding:0 8px 8px 0;">'
            '<table cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E6EAF0;border-radius:14px;background:#FFFFFF;">'
            f'<tr><td style="padding:10px 12px;"><div style="font-size:18px;line-height:22px;font-weight:700;color:{accent};">{value}</div>'
            f'<div style="font-size:11px;line-height:14px;color:#667085;text-transform:uppercase;letter-spacing:.04em;">{html.escape(label)}</div></td></tr>'
            '</table></td>'
        )

    def card(item, index, actionable):
        title = html.escape(str(item.get("title", "")))
        url = _safe_url(item.get("url", ""))
        title_markup = (
            f'<a href="{html.escape(url, quote=True)}" style="color:#101828;text-decoration:none;font-weight:700;">{title}</a>'
            if url else f'<span style="color:#101828;font-weight:700;">{title}</span>'
        )
        source = html.escape(str(item.get("source", "")))
        label = html.escape(_label(item))
        evidence = html.escape(str(item.get("evidence_level", "")))
        reason = html.escape(str(item.get("reason", "")))
        action = html.escape(str(item.get("action", "")))
        score = html.escape(str(item.get("score", "")))
        action_block = ""
        if actionable and action:
            action_block = (
                '<tr><td style="padding-top:12px;">'
                '<div style="font-size:11px;line-height:14px;color:#0B53CC;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Action</div>'
                f'<div style="font-size:14px;line-height:21px;color:#101828;font-weight:600;">{action}</div>'
                '</td></tr>'
            )
        return (
            '<table width="100%" cellpadding="0" cellspacing="0" role="presentation" '
            'style="margin:0 0 12px 0;border:1px solid #E6EAF0;border-radius:16px;background:#FFFFFF;">'
            '<tr><td style="padding:16px 16px 14px 16px;">'
            '<table width="100%" cellpadding="0" cellspacing="0" role="presentation"><tr>'
            f'<td width="34" valign="top"><div style="width:26px;height:26px;border-radius:999px;background:#EEF4FF;color:#0B53CC;text-align:center;font-size:13px;line-height:26px;font-weight:700;">{index}</div></td>'
            '<td valign="top">'
            f'<div style="font-size:16px;line-height:22px;margin-bottom:6px;">{title_markup}</div>'
            f'<div style="font-size:12px;line-height:16px;color:#667085;margin-bottom:12px;">{source} · {label} · Evidence: {evidence} · Score: {score}</div>'
            '<table width="100%" cellpadding="0" cellspacing="0" role="presentation">'
            '<tr><td>'
            '<div style="font-size:11px;line-height:14px;color:#667085;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Why it matters</div>'
            f'<div style="font-size:14px;line-height:21px;color:#344054;">{reason}</div>'
            '</td></tr>'
            f'{action_block}'
            '</table></td></tr></table></td></tr></table>'
        )

    def cards(items, actionable):
        if not items:
            message = "Quiet day. Nothing passed the action gate." if actionable else "Nothing notable."
            return (
                '<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px dashed #D0D5DD;border-radius:16px;background:#FFFFFF;">'
                f'<tr><td style="padding:16px;color:#667085;font-size:14px;line-height:21px;">{message}</td></tr></table>'
            )
        return "".join(card(item, index, actionable) for index, item in enumerate(items, 1))

    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#F6F8FB;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;color:#101828;">
<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F6F8FB;"><tr><td align="center" style="padding:24px 12px;">
<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="max-width:680px;margin:0 auto;">
<tr><td style="padding:22px 22px 18px 22px;background:#101828;border-radius:18px;color:#FFFFFF;">
<div style="font-size:12px;line-height:16px;color:#98A2B3;text-transform:uppercase;letter-spacing:.10em;font-weight:700;">Filip AI Intelligence</div>
<div style="font-size:28px;line-height:34px;font-weight:750;margin-top:6px;">Weekly briefing</div>
<div style="font-size:14px;line-height:21px;color:#D0D5DD;margin-top:6px;">{today}</div>
<div style="font-size:13px;line-height:19px;color:#98A2B3;margin-top:10px;">{total_fetched} scanned · {len(tier1)} act now · {len(tier2)} worth knowing · {sources_count} source groups</div>
</td></tr>
<tr><td style="padding:16px 0 4px 0;">
<table cellpadding="0" cellspacing="0" role="presentation"><tr>
{stat(total_fetched, "scanned")}{stat(len(tier1), "act now", "#D92D20")}{stat(len(tier2), "worth knowing", "#0B53CC")}{stat(sources_count, "source groups", "#475467")}
</tr></table>
</td></tr>
<tr><td style="padding:12px 0 14px 0;">
<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E6EAF0;border-radius:16px;background:#FFFFFF;">
<tr><td style="padding:16px;">
<div style="font-size:13px;line-height:18px;color:#667085;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Executive brief</div>
<div style="font-size:15px;line-height:23px;color:#344054;margin-top:6px;">Start with <strong style="color:#101828;">ACT NOW</strong>. Those are the only items with enough evidence and practical value to deserve attention this week. Treat <strong style="color:#101828;">WORTH KNOWING</strong> as context, not tasks.</div>
</td></tr></table>
</td></tr>
<tr><td style="padding:10px 0 8px 0;"><div style="font-size:13px;line-height:18px;color:#D92D20;text-transform:uppercase;letter-spacing:.10em;font-weight:800;">ACT NOW</div></td></tr>
<tr><td>{cards(tier1, True)}</td></tr>
<tr><td style="padding:18px 0 8px 0;"><div style="font-size:13px;line-height:18px;color:#475467;text-transform:uppercase;letter-spacing:.10em;font-weight:800;">WORTH KNOWING</div></td></tr>
<tr><td>{cards(tier2, False)}</td></tr>
<tr><td style="padding:14px 2px 0 2px;color:#98A2B3;font-size:12px;line-height:18px;">Evidence labels distinguish official/primary sources from expert analysis, reporting, and community discovery. Rankings are deterministic after model scoring.</td></tr>
</table>
</td></tr></table>
</body></html>'''


def build_telegram_message(classified, now=None):
    """Build valid Telegram HTML, adding only complete structural blocks."""
    today = (now or datetime.now(timezone.utc)).strftime("%d %b %Y")
    tier1, tier2 = classified.get("tier1", []), classified.get("tier2", [])
    parts = [f"📡 <b>AI Intel — {today}</b>\n<b>{len(tier1)}</b> act now · <b>{len(tier2)}</b> worth knowing\n\n"]

    def add(block):
        if sum(map(len, parts)) + len(block) <= 4000:
            parts.append(block)
            return True
        return False

    if tier1:
        add("<b>🎯 ACT NOW</b>\n\n")
        for item in tier1[:5]:
            title = html.escape(str(item.get("title", ""))[:160])
            reason = html.escape(str(item.get("reason", ""))[:300])
            action = html.escape(str(item.get("action", ""))[:160])
            source = html.escape(str(item.get("source", ""))[:100])
            block = f"<b>{title}</b>\n{reason}"
            if action:
                block += f"\n→ {action}"
            block += f"\n<i>{source} · {html.escape(_label(item))}</i>"
            url = _safe_url(item.get("url", ""))
            if url:
                block += f'\n<a href="{html.escape(url, quote=True)}">Read →</a>'
            if not add(block + "\n\n"):
                break
    else:
        add("Nothing passed the action gate.\n\n")
    if tier2:
        heading_index = len(parts)
        if add("<b>📋 WORTH KNOWING</b>\n\n"):
            added = False
            for item in tier2[:10]:
                title = html.escape(str(item.get("title", ""))[:180])
                reason = html.escape(str(item.get("reason", ""))[:250])
                block = f"• {title} — {reason} <i>({html.escape(_label(item))})</i>\n"
                if not add(block):
                    break
                added = True
            if not added:
                parts.pop(heading_index)
    return "".join(parts)


def send_email(content, recipient, subject=None):
    gmail, password = os.getenv("GMAIL_ADDRESS"), os.getenv("GMAIL_APP_PASSWORD")
    if not gmail or not password:
        raise ValueError("Gmail credentials not set in .env")
    message = MIMEMultipart("alternative")
    message["Subject"] = subject or f"AI Intel Briefing — {datetime.now(timezone.utc):%d %b %Y}"
    message["From"], message["To"] = f"AI Intel Pipeline <{gmail}>", recipient
    message.attach(MIMEText(content, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls(); server.login(gmail, password); server.sendmail(gmail, recipient, message.as_string())


def send_telegram(classified, config_value):
    split = config_value.rfind(":")
    if split <= 0:
        raise ValueError("Telegram format must be bot_token:chat_id")
    token, chat_id = config_value[:split].strip(), config_value[split + 1:].strip()
    if not chat_id.lstrip("-").isdigit():
        raise ValueError("Telegram chat ID must be numeric")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": build_telegram_message(classified), "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=10,
    )
    response.raise_for_status()


def _save_audit(path, record):
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(temp_path, "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _delivery_lock_path(audit_dir, delivery_key):
    """Return a secret-safe, deterministic path for a delivery claim lock."""
    identity = str(delivery_key).encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(identity).hexdigest()
    return os.path.join(str(audit_dir), f".delivery-{digest}.lock")


def _sha256(value):
    """Return a stable fingerprint without retaining the source value."""
    return hashlib.sha256(
        str(value).encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _destination_fingerprint(channel, value):
    """Fingerprint the effective, normalized channel destination."""
    normalized = str(value or "").strip()
    if channel == "email":
        normalized = normalized.lower()
    elif channel == "telegram" and ":" in normalized:
        token, chat_id = normalized.rsplit(":", 1)
        normalized = f"{token.strip()}:{chat_id.strip()}"
    return _sha256(f"{channel}:{normalized}")


def _delivery_fingerprints(pipeline, delivery_key):
    """Build opaque identities for one pipeline, period, and destination set."""
    pipeline_id = pipeline.get("id")
    if pipeline_id is not None and str(pipeline_id).strip():
        pipeline_source = {"id": str(pipeline_id).strip()}
    else:
        # Compatibility for callers with pre-id pipeline configurations.
        pipeline_source = {
            key: value for key, value in pipeline.items() if key != "channels"
        }
    pipeline_fingerprint = _sha256(json.dumps(
        pipeline_source, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ))
    key_fingerprint = _sha256(delivery_key)
    destinations = {}
    for channel in ("email", "telegram"):
        config = pipeline.get("channels", {}).get(channel, {})
        if config.get("on") and config.get("value"):
            destinations[channel] = _destination_fingerprint(channel, config["value"])
    identity_source = json.dumps({
        "pipeline": pipeline_fingerprint,
        "delivery_key": key_fingerprint,
        "destinations": destinations,
    }, sort_keys=True, separators=(",", ":"))
    return {
        "pipeline_fingerprint": pipeline_fingerprint,
        "delivery_key_fingerprint": key_fingerprint,
        "destination_fingerprints": destinations,
        "delivery_identity": _sha256(identity_source),
    }


@contextmanager
def _delivery_lock(audit_dir, delivery_key):
    """Exclusively claim one delivery key across Linux processes."""
    os.makedirs(audit_dir, exist_ok=True)
    path = _delivery_lock_path(audit_dir, delivery_key)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sanitized_delivery_error(channel, exc):
    """Describe a delivery failure without persisting exception text or URLs."""
    error = {"channel": channel}
    if isinstance(exc, requests.HTTPError):
        error["category"] = "http_error"
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            error["http_status"] = status
    elif isinstance(exc, requests.Timeout):
        error["category"] = "timeout"
    elif isinstance(exc, requests.ConnectionError):
        error["category"] = "connection_error"
    elif isinstance(exc, (ValueError, TypeError)):
        error["category"] = "configuration_error"
    else:
        error["category"] = "delivery_error"
    return error


def _prior_audit(audit_dir, delivery_identity):
    if not delivery_identity or not os.path.isdir(audit_dir):
        return None, None
    for filename in os.listdir(audit_dir):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(str(audit_dir), filename)
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            if record.get("delivery_identity") == delivery_identity:
                return path, record
        except (OSError, ValueError, TypeError):
            continue
    return None, None


def _write_audit(audit_dir, pipeline, fetched_count, filtered, classified, deliver,
                 fingerprints=None):
    os.makedirs(audit_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    fingerprints = fingerprints or {}
    prior_path, prior = _prior_audit(
        audit_dir, fingerprints.get("delivery_identity")
    ) if deliver else (None, None)
    path = prior_path or os.path.join(
        str(audit_dir), f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:10]}.json"
    )
    prior_delivery = (prior or {}).get("delivery", {})
    delivery = {}
    channels = pipeline.get("channels", {})
    for channel in ("email", "telegram"):
        configured = bool(channels.get(channel, {}).get("on") and channels.get(channel, {}).get("value"))
        destination_fingerprint = fingerprints.get("destination_fingerprints", {}).get(channel)
        old = prior_delivery.get(channel, {})
        same_destination = old.get("destination_fingerprint") == destination_fingerprint
        if deliver and same_destination and old.get("status") == "success":
            delivery[channel] = old
        elif (deliver and configured and same_destination and
              old.get("status") in {"attempting", "ambiguous"}):
            # The process may have died after the provider accepted the message.
            # Avoid a blind resend when the external outcome cannot be known.
            delivery[channel] = {
                "status": "ambiguous", "updated_at": now.isoformat(),
                "destination_fingerprint": destination_fingerprint,
            }
        elif deliver and configured:
            delivery[channel] = {
                "status": "pending", "destination_fingerprint": destination_fingerprint,
            }
        else:
            delivery[channel] = {"status": "dry_run" if not deliver and configured else "not_configured"}
    if prior:
        # A delivery identity identifies one immutable briefing. Retrying a failed
        # channel must not combine newly fetched content with channels that
        # already received the original briefing.
        record = dict(prior)
        record["last_attempt_at"] = now.isoformat()
        record["delivery"] = delivery
    else:
        record = {
            "run_at": now.isoformat(), "pipeline": pipeline.get("name", ""),
            "delivery_enabled": deliver, "delivery": delivery,
            "pipeline_fingerprint": fingerprints.get("pipeline_fingerprint"),
            "delivery_key_fingerprint": fingerprints.get("delivery_key_fingerprint"),
            "delivery_identity": fingerprints.get("delivery_identity"),
            "total_fetched": fetched_count, "total_after_filtering": len(filtered),
            "sources_checked": len(pipeline.get("sources", [])),
            "tier1_count": len(classified.get("tier1", [])),
            "tier2_count": len(classified.get("tier2", [])),
            "tier1": classified.get("tier1", []), "tier2": classified.get("tier2", []),
            "scored": classified.get("scored", []),
        }
    _save_audit(path, record)
    return path, record


def _run_single_pipeline_locked(pipeline, deliver=True, audit_dir=None, delivery_key=None):
    """Run while the caller holds the delivery-key lock when delivering."""
    from app import build_system_prompt
    started = time.time()
    directory = audit_dir or os.path.join(os.path.dirname(__file__), "data", "daily")
    if deliver and delivery_key is None:
        period = datetime.now(timezone.utc).date().isoformat()
        delivery_key = period
    fingerprints = _delivery_fingerprints(pipeline, delivery_key) if deliver else {}

    _, prior = _prior_audit(
        directory, fingerprints.get("delivery_identity")
    ) if deliver else (None, None)
    if prior:
        # Retry directly from the immutable audited payload. Feed or model
        # availability must not prevent retrying a channel that previously failed.
        classified = {
            "tier1": prior.get("tier1", []), "tier2": prior.get("tier2", []),
            "scored": prior.get("scored", []),
        }
        audit_path, audit = _write_audit(
            directory, pipeline, prior.get("total_fetched", 0), [], classified, True,
            fingerprints,
        )
    else:
        fetched = fetch_all_for_pipeline(pipeline)
        if not fetched:
            return {"status": "no_items", "message": "No items fetched"}
        filtered = filter_and_dedupe(fetched)
        if not filtered:
            return {"status": "no_items", "message": "No fresh valid items"}
        classified = classify(filtered, build_system_prompt(pipeline))
        audit_path, audit = _write_audit(
            directory, pipeline, len(fetched), filtered, classified, deliver, fingerprints
        )

    delivery_classified = {
        "tier1": audit.get("tier1", []), "tier2": audit.get("tier2", []),
        "scored": audit.get("scored", []),
    }
    content = build_email_html(
        delivery_classified, audit.get("total_fetched", 0),
        audit.get("sources_checked", len(pipeline.get("sources", []))),
    )
    failures = []
    if deliver:
        channels = pipeline.get("channels", {})
        senders = {
            "email": lambda value: send_email(content, value),
            "telegram": lambda value: send_telegram(delivery_classified, value),
        }
        for channel, sender in senders.items():
            config = channels.get(channel, {})
            if not (config.get("on") and config.get("value")):
                continue
            if audit["delivery"][channel]["status"] in {"success", "ambiguous"}:
                continue
            # This durable marker closes the crash window before an external call.
            audit["delivery"][channel] = {
                "status": "attempting", "updated_at": datetime.now(timezone.utc).isoformat(),
                "destination_fingerprint": fingerprints["destination_fingerprints"][channel],
            }
            _save_audit(audit_path, audit)
            try:
                sender(config["value"])
                audit["delivery"][channel] = {
                    "status": "success", "updated_at": datetime.now(timezone.utc).isoformat(),
                    "destination_fingerprint": fingerprints["destination_fingerprints"][channel],
                }
            except Exception as exc:
                audit["delivery"][channel] = {
                    "status": "failed", "updated_at": datetime.now(timezone.utc).isoformat(),
                    "destination_fingerprint": fingerprints["destination_fingerprints"][channel],
                    "error": _sanitized_delivery_error(channel, exc),
                }
                failures.append(channel)
            _save_audit(audit_path, audit)
    if failures:
        raise DeliveryError(f"Delivery failed for: {', '.join(failures)}", audit_path)
    result = {
        "total_fetched": audit.get("total_fetched", 0),
        "total_filtered": audit.get("total_after_filtering", 0),
        "tier1": len(delivery_classified["tier1"]), "tier2": len(delivery_classified["tier2"]),
        "elapsed": round(time.time() - started, 1), "delivered": deliver,
        "audit_path": audit_path, "html": content if not deliver else None,
    }
    ambiguous = [
        channel for channel, state in audit.get("delivery", {}).items()
        if state.get("status") == "ambiguous"
    ]
    if ambiguous:
        result["status"] = "ambiguous"
        result["ambiguous_channels"] = ambiguous
    return result


def run_single_pipeline(pipeline, deliver=True, audit_dir=None, delivery_key=None):
    """Run, audit, and optionally deliver with a cross-process idempotency claim."""
    directory = audit_dir or os.path.join(os.path.dirname(__file__), "data", "daily")
    if deliver and delivery_key is None:
        period = datetime.now(timezone.utc).date().isoformat()
        delivery_key = period
    if not deliver:
        return _run_single_pipeline_locked(pipeline, False, directory, delivery_key)
    delivery_identity = _delivery_fingerprints(pipeline, delivery_key)["delivery_identity"]
    with _delivery_lock(directory, delivery_identity):
        return _run_single_pipeline_locked(pipeline, True, directory, delivery_key)
