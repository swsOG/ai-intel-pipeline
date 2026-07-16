"""Fetch, filter, rank, render, and optionally deliver AI intelligence."""

import hashlib
import html
import json
import os
import smtplib
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from dotenv import load_dotenv

load_dotenv()


def _source(name, url, kind, source_class, trust_weight, rationale):
    return {
        "name": name, "url": url, "type": kind, "source_class": source_class,
        "trust_weight": trust_weight, "rationale": rationale,
    }


# These are the project's existing feeds with deterministic policy added. No feed
# URL is inferred or generated at runtime.
SOURCE_FEEDS = {
    "RSS Blogs": [
        _source("Anthropic Blog", "https://www.anthropic.com/rss.xml", "rss", "official", 5, "First-party Anthropic announcements."),
        _source("OpenAI Blog", "https://openai.com/blog/rss.xml", "rss", "official", 5, "First-party OpenAI announcements."),
        _source("Google AI Blog", "https://blog.google/technology/ai/rss/", "rss", "official", 5, "First-party Google AI announcements."),
        _source("Meta AI Blog", "https://ai.meta.com/blog/rss/", "rss", "official", 5, "First-party Meta AI announcements."),
        _source("Simon Willison", "https://simonwillison.net/atom/everything/", "rss", "expert", 4, "Named practitioner's technical analysis."),
        _source("The Batch", "https://www.deeplearning.ai/the-batch/feed/", "rss", "reporting", 3, "Edited secondary AI reporting."),
        _source("Lilian Weng", "https://lilianweng.github.io/index.xml", "rss", "expert", 4, "Named researcher's technical analysis."),
        _source("Ars Technica AI", "https://arstechnica.com/ai/feed/", "rss", "reporting", 3, "Edited technology reporting."),
    ],
    "Reddit": [
        _source("r/MachineLearning", "https://www.reddit.com/r/MachineLearning/hot.json?limit=10", "reddit", "discovery", 1, "Community discussion; verify linked claims."),
        _source("r/LocalLLaMA", "https://www.reddit.com/r/LocalLLaMA/hot.json?limit=10", "reddit", "discovery", 1, "Community discussion; verify linked claims."),
        _source("r/artificial", "https://www.reddit.com/r/artificial/hot.json?limit=10", "reddit", "discovery", 1, "Community discussion; verify linked claims."),
    ],
    "GitHub Trending": [_source("GitHub Trending", "https://github.com/trending?since=daily", "github", "discovery", 1, "Popularity-based repository discovery, not verification.")],
    "Hacker News": [_source("Hacker News AI", "https://hnrss.org/newest?q=AI+OR+LLM+OR+GPT&points=50", "rss", "discovery", 1, "Community popularity-based discovery.")],
    "Product Hunt": [_source("Product Hunt", "https://www.producthunt.com/feed", "rss", "discovery", 1, "Launch and popularity-based discovery.")],
    "Hugging Face": [_source("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", "rss", "official", 5, "First-party Hugging Face announcements.")],
    "ArXiv": [_source("ArXiv AI", "http://arxiv.org/rss/cs.AI", "rss", "primary", 5, "Primary research manuscripts; not necessarily peer reviewed.")],
    "TechCrunch": [_source("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "rss", "reporting", 3, "Edited technology reporting.")],
    "The Verge": [_source("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "rss", "reporting", 3, "Edited technology reporting.")],
}

HEADERS = {"User-Agent": "AI-Intel-Pipeline/2.0"}
REQUEST_TIMEOUT = 15
TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"}
EVIDENCE_LEVELS = {
    "official": "primary", "primary": "primary research", "expert": "expert analysis",
    "reporting": "reporting", "discovery": "community discovery",
}


class ClassificationError(RuntimeError):
    """Raised when Gemini supplies no usable batch response."""


def canonicalize_url(url):
    """Return a conservative HTTP(S) canonical URL, or an empty string."""
    if not isinstance(url, str):
        return ""
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        if scheme not in {"http", "https"} or not parts.hostname:
            return ""
        host = parts.hostname.lower()
        if host.startswith("www."):
            host = host[4:]
        port = parts.port
        if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
            host = f"{host}:{port}"
        params = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            lower = key.lower()
            if lower.startswith("utm_") or lower in TRACKING_PARAMS:
                continue
            params.append((key, value))
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/")
        return urlunsplit((scheme, host, path, urlencode(sorted(params)), ""))
    except (ValueError, TypeError):
        return ""


def stable_item_id(url):
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() if canonical else ""


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
    canonical = canonicalize_url(raw.get("url", ""))
    policy = {key: source[key] for key in ("source_class", "trust_weight", "rationale")}
    return {
        "item_id": stable_item_id(canonical),
        "title": str(raw.get("title") or "").strip(),
        "url": canonical,
        "summary": str(raw.get("summary") or "").strip()[:1000],
        "source": source["name"],
        "source_policy": policy,
        **policy,
        "evidence_level": EVIDENCE_LEVELS[source["source_class"]],
        "published_at": _iso_timestamp(raw.get("published_at")),
        "fetched_at": _iso_timestamp(fetched),
        "discussion_url": canonicalize_url(raw.get("discussion_url", "")),
        "alternate_sources": [],
        "discussion_urls": [],
    }


def _date_from_feed(entry):
    value = entry.get("published_parsed") or entry.get("updated_parsed")
    if value:
        return datetime(*value[:6], tzinfo=timezone.utc).isoformat()
    return entry.get("published") or entry.get("updated")


def fetch_rss(source):
    try:
        import feedparser
        from bs4 import BeautifulSoup
        feed = feedparser.parse(source["url"])
        result = []
        for entry in feed.entries[:15]:
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:300]
            result.append(normalize_item({
                "title": entry.get("title", "Untitled"), "url": entry.get("link", ""),
                "summary": summary.strip(), "published_at": _date_from_feed(entry),
            }, source))
        return result
    except Exception as exc:
        print(f"  [ERROR] RSS: {source['name']}: {exc}")
        return []


def fetch_reddit(source):
    try:
        response = requests.get(source["url"], headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = []
        for post in response.json().get("data", {}).get("children", []):
            value = post.get("data", {})
            if value.get("stickied"):
                continue
            discussion = f"https://reddit.com{value.get('permalink', '')}"
            target = value.get("url_overridden_by_dest") or value.get("url") or discussion
            result.append(normalize_item({
                "title": value.get("title", "Untitled"), "url": target,
                "summary": (value.get("selftext") or "")[:300].strip(),
                "published_at": value.get("created_utc"), "discussion_url": discussion,
            }, source))
        return result
    except Exception as exc:
        print(f"  [ERROR] Reddit: {source['name']}: {exc}")
        return []


def fetch_github(source):
    try:
        from bs4 import BeautifulSoup
        response = requests.get(source["url"], headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
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
            print(f"  Fetching: {source['name']}...")
            fetcher = {"rss": fetch_rss, "reddit": fetch_reddit, "github": fetch_github}.get(source["type"])
            fetched = fetcher(source) if fetcher else []
            print(f"    -> {len(fetched)} items")
            items.extend(fetched)
    print(f"  Total fetched: {len(items)}")
    return items


def _valid_item(item):
    return bool(
        isinstance(item, dict) and item.get("title") and item.get("url")
        and item.get("item_id") == stable_item_id(item.get("url"))
        and item.get("source_class") in EVIDENCE_LEVELS
        and isinstance(item.get("trust_weight"), int)
        and 1 <= item["trust_weight"] <= 5
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
        if published and datetime.fromisoformat(published) < cutoff:
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
            discussion = canonicalize_url(candidate.get("discussion_url", ""))
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
    """Request bounded signals only; all identity and final ranking stay local."""
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        from google import genai
        client = genai.Client(api_key=api_key)
    batches = [items[index:index + batch_size] for index in range(0, len(items), batch_size)]
    all_signals = []
    successful_batches = 0
    seen = set()
    for index, batch in enumerate(batches):
        payload = [{
            "item_id": item["item_id"], "title": item["title"], "summary": item["summary"][:500],
            "source": item["source"], "source_class": item["source_class"],
            "evidence_level": item["evidence_level"],
        } for item in batch]
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=json.dumps(payload),
                config={"system_instruction": system_prompt, "response_mime_type": "application/json", "temperature": 0.1},
            )
            values = json.loads(response.text)
            if not isinstance(values, list):
                raise ValueError("Gemini response must be an array")
            successful_batches += 1
            known = {item["item_id"] for item in batch}
            for value in values:
                if _valid_signal(value, known, seen):
                    seen.add(value["item_id"])
                    all_signals.append(value)
        except Exception as exc:
            print(f"  [ERROR] Gemini batch {index + 1}: {exc}")
        if index < len(batches) - 1 and sleep_seconds:
            time.sleep(sleep_seconds)
    if batches and successful_batches == 0:
        raise ClassificationError("All Gemini classification batches failed; briefing not delivered")
    return rank_items(items, all_signals)


def _safe_url(value):
    return canonicalize_url(value)


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
    today = (now or datetime.now(timezone.utc)).strftime("%d %b %Y")
    tier1, tier2 = classified.get("tier1", []), classified.get("tier2", [])
    message = f"📡 <b>AI Intel — {today}</b>\n<b>{len(tier1)}</b> act now · <b>{len(tier2)}</b> worth knowing\n\n"
    if tier1:
        message += "<b>🎯 ACT NOW</b>\n\n"
        for item in tier1[:5]:
            message += f"<b>{html.escape(str(item.get('title', '')))}</b>\n{html.escape(str(item.get('reason', '')))}"
            if item.get("action"):
                message += f"\n→ {html.escape(str(item['action']))}"
            message += f"\n<i>{html.escape(str(item.get('source', '')))} · {html.escape(_label(item))}</i>"
            url = _safe_url(item.get("url", ""))
            if url:
                message += f'\n<a href="{html.escape(url, quote=True)}">Read →</a>'
            message += "\n\n"
    else:
        message += "Nothing passed the action gate.\n\n"
    if tier2:
        message += "<b>📋 WORTH KNOWING</b>\n\n"
        for item in tier2[:10]:
            message += f"• {html.escape(str(item.get('title', '')))} — {html.escape(str(item.get('reason', '')))} <i>({_label(item)})</i>\n"
    return message[:4000]


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


def _write_audit(audit_dir, pipeline, fetched_count, filtered, classified, deliver):
    os.makedirs(audit_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:10]}.json"
    path = os.path.join(str(audit_dir), filename)
    record = {
        "run_at": now.isoformat(), "pipeline": pipeline.get("name", ""), "delivery_enabled": deliver,
        "total_fetched": fetched_count, "total_after_filtering": len(filtered),
        "sources_checked": len(pipeline.get("sources", [])),
        "tier1_count": len(classified.get("tier1", [])), "tier2_count": len(classified.get("tier2", [])),
        "tier1": classified.get("tier1", []), "tier2": classified.get("tier2", []),
        "scored": classified.get("scored", []),
    }
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    return path


def run_single_pipeline(pipeline, deliver=True, audit_dir=None):
    """Run the pipeline. ``deliver=False`` renders and audits without sending."""
    from app import build_system_prompt
    started = time.time()
    fetched = fetch_all_for_pipeline(pipeline)
    if not fetched:
        return {"status": "no_items", "message": "No items fetched"}
    filtered = filter_and_dedupe(fetched)
    if not filtered:
        return {"status": "no_items", "message": "No fresh valid items"}
    classified = classify(filtered, build_system_prompt(pipeline))
    content = build_email_html(classified, len(fetched), len(pipeline.get("sources", [])))
    channels = pipeline.get("channels", {})
    if deliver:
        email_config = channels.get("email", {})
        telegram_config = channels.get("telegram", {})
        if email_config.get("on") and email_config.get("value"):
            send_email(content, email_config["value"])
        if telegram_config.get("on") and telegram_config.get("value"):
            send_telegram(classified, telegram_config["value"])
    directory = audit_dir or os.path.join(os.path.dirname(__file__), "data", "daily")
    audit_path = _write_audit(directory, pipeline, len(fetched), filtered, classified, deliver)
    return {
        "total_fetched": len(fetched), "total_filtered": len(filtered),
        "tier1": len(classified.get("tier1", [])), "tier2": len(classified.get("tier2", [])),
        "elapsed": round(time.time() - started, 1), "delivered": deliver,
        "audit_path": audit_path, "html": content if not deliver else None,
    }
