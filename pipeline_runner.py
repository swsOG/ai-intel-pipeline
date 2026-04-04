"""
AI Intelligence Pipeline - Runner
Executes a single pipeline based on its config from the control panel.
"""

import os
import json
import time
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from google import genai
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

load_dotenv()
from email.mime.multipart import MIMEMultipart


# ============================================================
# SOURCE MAPPING
# ============================================================
SOURCE_FEEDS = {
    "RSS Blogs": [
        {"name": "Anthropic Blog", "url": "https://www.anthropic.com/rss.xml", "type": "rss"},
        {"name": "OpenAI Blog", "url": "https://openai.com/blog/rss.xml", "type": "rss"},
        {"name": "Google AI Blog", "url": "https://blog.google/technology/ai/rss/", "type": "rss"},
        {"name": "Meta AI Blog", "url": "https://ai.meta.com/blog/rss/", "type": "rss"},
        {"name": "Simon Willison", "url": "https://simonwillison.net/atom/everything/", "type": "rss"},
        {"name": "The Batch", "url": "https://www.deeplearning.ai/the-batch/feed/", "type": "rss"},
        {"name": "Lilian Weng", "url": "https://lilianweng.github.io/index.xml", "type": "rss"},
        {"name": "Ars Technica AI", "url": "https://arstechnica.com/ai/feed/", "type": "rss"},
    ],
    "Reddit": [
        {"name": "r/MachineLearning", "url": "https://www.reddit.com/r/MachineLearning/hot.json?limit=10", "type": "reddit"},
        {"name": "r/LocalLLaMA", "url": "https://www.reddit.com/r/LocalLLaMA/hot.json?limit=10", "type": "reddit"},
        {"name": "r/artificial", "url": "https://www.reddit.com/r/artificial/hot.json?limit=10", "type": "reddit"},
    ],
    "GitHub Trending": [
        {"name": "GitHub Trending", "url": "https://github.com/trending?since=daily", "type": "github"},
    ],
    "Hacker News": [
        {"name": "Hacker News AI", "url": "https://hnrss.org/newest?q=AI+OR+LLM+OR+GPT&points=50", "type": "rss"},
    ],
    "Product Hunt": [
        {"name": "Product Hunt", "url": "https://www.producthunt.com/feed", "type": "rss"},
    ],
    "Hugging Face": [
        {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog/feed.xml", "type": "rss"},
    ],
    "ArXiv": [
        {"name": "ArXiv AI", "url": "http://arxiv.org/rss/cs.AI", "type": "rss"},
    ],
    "TechCrunch": [
        {"name": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "type": "rss"},
    ],
    "The Verge": [
        {"name": "The Verge AI", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "type": "rss"},
    ],
}

HEADERS = {"User-Agent": "AI-Intel-Pipeline/1.0"}
REQUEST_TIMEOUT = 15


# ============================================================
# FETCHERS
# ============================================================
def fetch_rss(source):
    try:
        feed = feedparser.parse(source["url"])
        items = []
        for entry in feed.entries[:15]:
            summary = ""
            if hasattr(entry, "summary"):
                summary = BeautifulSoup(entry.summary, "html.parser").get_text()[:300]
            items.append({
                "title": entry.get("title", "Untitled"),
                "url": entry.get("link", ""),
                "summary": summary.strip(),
                "source": source["name"],
            })
        return items
    except Exception as e:
        print(f"  [ERROR] RSS: {source['name']}: {e}")
        return []


def fetch_reddit(source):
    try:
        resp = requests.get(source["url"], headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        items = []
        for post in data.get("data", {}).get("children", []):
            p = post.get("data", {})
            if p.get("stickied"):
                continue
            items.append({
                "title": p.get("title", "Untitled"),
                "url": f"https://reddit.com{p.get('permalink', '')}",
                "summary": (p.get("selftext", "") or "")[:300].strip(),
                "source": source["name"],
            })
        return items
    except Exception as e:
        print(f"  [ERROR] Reddit: {source['name']}: {e}")
        return []


def fetch_github(source):
    try:
        resp = requests.get(source["url"], headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = []
        for article in soup.select("article.Box-row")[:15]:
            h2 = article.select_one("h2 a")
            if not h2:
                continue
            repo_path = h2.get("href", "").strip("/")
            p = article.select_one("p")
            desc = p.get_text(strip=True) if p else ""
            items.append({
                "title": repo_path.replace("/", " / "),
                "url": f"https://github.com/{repo_path}",
                "summary": desc[:300],
                "source": "GitHub Trending",
            })
        return items
    except Exception as e:
        print(f"  [ERROR] GitHub: {e}")
        return []


def fetch_all_for_pipeline(pipeline):
    """Fetch from sources enabled in the pipeline config."""
    all_items = []
    enabled_sources = pipeline.get("sources", [])

    for source_key in enabled_sources:
        feeds = SOURCE_FEEDS.get(source_key, [])
        for feed in feeds:
            print(f"  Fetching: {feed['name']}...")
            if feed["type"] == "rss":
                items = fetch_rss(feed)
            elif feed["type"] == "reddit":
                items = fetch_reddit(feed)
            elif feed["type"] == "github":
                items = fetch_github(feed)
            else:
                items = []
            print(f"    -> {len(items)} items")
            all_items.extend(items)

    print(f"  Total fetched: {len(all_items)}")
    return all_items


# ============================================================
# CLASSIFIER
# ============================================================
def classify(items, system_prompt):
    """Send items to Gemini for classification."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    all_classified = []
    batch_size = 25

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"  Classifying {len(items)} items in {len(batches)} batches...")

    for i, batch in enumerate(batches):
        print(f"  Batch {i + 1}/{len(batches)}...")
        stripped = [{"title": it.get("title", ""), "source": it.get("source", ""),
                     "url": it.get("url", ""), "summary": it.get("summary", "")[:200]} for it in batch]

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=json.dumps(stripped),
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "temperature": 0.3,
                }
            )
            results = json.loads(response.text)
            if isinstance(results, list):
                all_classified.extend(results)
        except Exception as e:
            print(f"  [ERROR] Gemini batch {i + 1}: {e}")

        if i < len(batches) - 1:
            time.sleep(5)

    tier1 = [it for it in all_classified if it.get("tier") == 1]
    tier2 = [it for it in all_classified if it.get("tier") == 2]
    print(f"  Tier 1: {len(tier1)}, Tier 2: {len(tier2)}")
    return {"tier1": tier1, "tier2": tier2}


# ============================================================
# EMAIL BUILDER & SENDER
# ============================================================
def build_email_html(classified, total_fetched, sources_count):
    """Clean, scannable email — compact, plain English."""
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    tier1 = classified.get("tier1", [])
    tier2 = classified.get("tier2", [])

    t1_html = ""
    for it in tier1:
        action = it.get("action", "")
        action_html = f'<br/><span style="color:#0B53CC;font-weight:600;font-size:13px;">&#8594; {action}</span>' if action else ""
        t1_html += f'''<tr><td style="padding:10px 16px;border-bottom:1px solid #F0F0F0;">
<a href="{it.get("url","#")}" style="font-size:14px;font-weight:600;color:#1A1A2E;text-decoration:none;">{it.get("title","")}</a>
<span style="font-size:11px;color:#999;margin-left:6px;">{it.get("source","")}</span><br/>
<span style="font-size:13px;color:#444;">{it.get("reason","")}</span>{action_html}
</td></tr>'''

    t2_html = ""
    for it in tier2:
        t2_html += f'''<tr><td style="padding:6px 16px;border-bottom:1px solid #F8F8F8;">
<a href="{it.get("url","#")}" style="font-size:13px;font-weight:500;color:#1A1A2E;text-decoration:none;">{it.get("title","")}</a>
<span style="font-size:11px;color:#999;margin-left:6px;">{it.get("source","")}</span><br/>
<span style="font-size:12px;color:#777;">{it.get("reason","")}</span>
</td></tr>'''

    if not t1_html:
        t1_html = '<tr><td style="padding:12px 16px;color:#999;font-style:italic;font-size:13px;">Quiet day. Nothing urgent.</td></tr>'
    if not t2_html:
        t2_html = '<tr><td style="padding:12px 16px;color:#999;font-style:italic;font-size:13px;">Nothing notable.</td></tr>'

    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#F5F5F5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:0 auto;background:#FFF;">
<tr><td style="padding:20px 16px 12px;border-bottom:3px solid #0B53CC;">
<span style="font-size:20px;font-weight:700;color:#0B53CC;">AI Intel</span>
<span style="font-size:13px;color:#999;margin-left:8px;">{today}</span><br/>
<span style="font-size:12px;color:#888;">{total_fetched} scanned &middot; <strong style="color:#0B53CC;">{len(tier1)}</strong> act now &middot; <strong>{len(tier2)}</strong> worth knowing</span>
</td></tr>
<tr><td style="padding:12px 16px 4px;">
<span style="font-size:12px;font-weight:600;color:#0B53CC;text-transform:uppercase;letter-spacing:1px;">Act now</span>
</td></tr>
{t1_html}
<tr><td style="padding:16px 16px 4px;">
<span style="font-size:12px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:1px;">Worth knowing</span>
</td></tr>
{t2_html}
<tr><td style="padding:12px 16px;background:#F8F9FA;text-align:center;">
<span style="font-size:10px;color:#BBB;">AI Intel Pipeline</span>
</td></tr>
</table></body></html>'''


def send_email(html, recipient, subject=None):
    gmail = os.getenv("GMAIL_ADDRESS")
    pwd = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail or not pwd:
        raise ValueError("Gmail credentials not set in .env")
    if not subject:
        subject = f"AI Intel Briefing — {datetime.now(timezone.utc).strftime('%d %b %Y')}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"AI Intel Pipeline <{gmail}>"
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gmail, pwd)
        server.sendmail(gmail, recipient, msg.as_string())
    print(f"  Email sent to {recipient}")


def send_telegram(classified, config_value):
    """Send a clean Telegram summary with proper token parsing."""
    try:
        # Split on LAST colon to separate bot_token from chat_id
        last_colon = config_value.rfind(":")
        if last_colon <= 0:
            print("  [ERROR] Telegram format must be bot_token:chat_id")
            return

        bot_token = config_value[:last_colon].strip()
        chat_id = config_value[last_colon + 1:].strip()

        if not chat_id.isdigit():
            print(f"  [ERROR] Chat ID must be a number, got: {chat_id}")
            return

        today = datetime.now(timezone.utc).strftime("%d %b %Y")
        tier1 = classified.get("tier1", [])
        tier2 = classified.get("tier2", [])

        msg = f"\U0001F4E1 <b>AI Intel \u2014 {today}</b>\n"
        msg += f"<b>{len(tier1)}</b> act now \u00B7 <b>{len(tier2)}</b> worth knowing\n\n"

        if tier1:
            msg += "<b>\U0001F3AF ACT NOW</b>\n\n"
            for it in tier1[:8]:
                title = it.get("title", "")
                reason = it.get("reason", "")
                action = it.get("action", "")
                url = it.get("url", "")
                msg += f"<b>{title}</b>\n"
                msg += f"{reason}"
                if action:
                    msg += f"\n\u2192 {action}"
                if url:
                    msg += f'\n<a href="{url}">Read \u2192</a>'
                msg += "\n\n"
        else:
            msg += "Quiet day. Nothing urgent.\n\n"

        if tier2:
            msg += "<b>\U0001F4CB WORTH KNOWING</b>\n\n"
            for it in tier2[:10]:
                msg += f"\u2022 {it.get('title','')} \u2014 {it.get('reason','')}\n"

        if len(msg) > 4000:
            msg = msg[:3997] + "..."

        api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(api_url, json={
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=10)

        if resp.status_code == 200:
            print(f"  Telegram sent to {chat_id}")
        else:
            print(f"  [ERROR] Telegram API returned {resp.status_code}: {resp.text}")

    except Exception as e:
        print(f"  [ERROR] Telegram: {e}")


# ============================================================
# MAIN RUNNER
# ============================================================
def run_single_pipeline(pipeline):
    """Execute a single pipeline end-to-end."""
    from app import build_system_prompt

    start = time.time()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n{'='*50}")
    print(f"  Running: {pipeline.get('name', 'Unnamed')}")
    print(f"{'='*50}")

    # Fetch
    print("\n[1/4] Fetching...")
    items = fetch_all_for_pipeline(pipeline)
    if not items:
        return {"status": "no_items", "message": "No items fetched"}

    # Classify
    print("\n[2/4] Classifying...")
    system_prompt = build_system_prompt(pipeline)
    classified = classify(items, system_prompt)

    # Build email
    print("\n[3/4] Building email...")
    html = build_email_html(classified, len(items), len(pipeline.get("sources", [])))

    # Deliver
    print("\n[4/4] Delivering...")
    channels = pipeline.get("channels", {})

    if channels.get("email", {}).get("on") and channels["email"].get("value"):
        send_email(html, channels["email"]["value"])

    if channels.get("telegram", {}).get("on") and channels["telegram"].get("value"):
        send_telegram(classified, channels["telegram"]["value"])

    # Save log
    data_dir = os.path.join(os.path.dirname(__file__), "data", "daily")
    os.makedirs(data_dir, exist_ok=True)
    log = {
        "date": today,
        "pipeline": pipeline.get("name", ""),
        "total_fetched": len(items),
        "sources_checked": len(pipeline.get("sources", [])),
        "tier1_count": len(classified.get("tier1", [])),
        "tier2_count": len(classified.get("tier2", [])),
        "tier1": classified.get("tier1", []),
        "tier2": classified.get("tier2", []),
    }
    log_path = os.path.join(data_dir, f"{today}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")
    return {"total_fetched": len(items), "tier1": len(classified.get("tier1", [])),
            "tier2": len(classified.get("tier2", [])), "elapsed": round(elapsed, 1)}
