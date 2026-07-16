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

    tier1_candidates = [item for item in scored if (
        item["source_class"] != "discovery" and item["relevance"] >= 2
        and item["actionability"] >= 2 and item["confidence"] >= 2
        and item["hype_penalty"] <= 1 and item["score"] >= 25
    )]
    tier1 = tier1_candidates[:tier1_cap]
    used = {item["item_id"] for item in tier1}
    tier2 = [item for item in scored if (
        item["item_id"] not in used and item["relevance"] >= 1
        and item["confidence"] >= 1 and item["score"] >= 10
    )][:tier2_cap]
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
        payload = [{
            "item_id": item["item_id"], "title": item["title"], "summary": item["summary"][:500],
            "source": item["source"], "source_class": item["source_class"],
            "evidence_level": item["evidence_level"],
        } for item in batch]
        known = {item["item_id"] for item in batch}
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
                if not _valid_signal(value, known, seen):
                    raise ValueError("malformed, unknown, or duplicate signal")
                seen.add(value["item_id"])
                batch_signals.append(value)
            missing = known - {value["item_id"] for value in batch_signals}
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

    def rows(items, actionable):
        output = ""
        for item in items:
            title = html.escape(str(item.get("title", "")))
            url = _safe_url(item.get("url", ""))
            title_markup = f'<a href="{html.escape(url, quote=True)}" style="font-weight:600;color:#1A1A2E;text-decoration:none;">{title}</a>' if url else f"<strong>{title}</strong>"
            action = html.escape(str(item.get("action", "")))
            action_markup = f'<br/><span style="color:#0B53CC;font-weight:600;">→ {action}</span>' if actionable and action else ""
            output += (
                '<tr><td style="padding:10px 16px;border-bottom:1px solid #F0F0F0;">'
                f'{title_markup}<br/><span style="font-size:11px;color:#777;">{html.escape(str(item.get("source", "")))} · {html.escape(_label(item))}</span><br/>'
                f'<span style="font-size:13px;color:#444;">{html.escape(str(item.get("reason", "")))}</span>{action_markup}</td></tr>'
            )
        return output

    tier1 = classified.get("tier1", [])
    tier2 = classified.get("tier2", [])
    t1 = rows(tier1, True) or '<tr><td style="padding:12px 16px;color:#999;">Quiet day. Nothing passed the action gate.</td></tr>'
    t2 = rows(tier2, False) or '<tr><td style="padding:12px 16px;color:#999;">Nothing notable.</td></tr>'
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0;background:#F5F5F5;font-family:Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:auto;background:#FFF;"><tr><td style="padding:20px 16px;border-bottom:3px solid #0B53CC;"><strong style="font-size:20px;color:#0B53CC;">AI Intel</strong> <span style="color:#777;">{today}</span><br/><small>{total_fetched} scanned · {len(tier1)} act now · {len(tier2)} worth knowing · {sources_count} source groups</small></td></tr>
<tr><td style="padding:14px 16px 4px;color:#0B53CC;"><strong>ACT NOW</strong></td></tr>{t1}
<tr><td style="padding:16px 16px 4px;color:#777;"><strong>WORTH KNOWING</strong></td></tr>{t2}
</table></body></html>'''


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
