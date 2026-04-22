from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import feedparser

BASE_DIR = Path(__file__).resolve().parents[1]
DEMO_FILE = BASE_DIR / "data" / "demo_articles.json"

LIVE_RSS_SOURCES = [
    {
        "name": "MarketWatch Top Stories",
        "url": "http://feeds.marketwatch.com/marketwatch/topstories/",
    },
    {
        "name": "Investing.com Stock Market News",
        "url": "https://www.investing.com/rss/news_25.rss",
    },
    {
        "name": "Investing.com All News",
        "url": "https://www.investing.com/rss/news.rss",
    },
]


def load_demo_articles() -> List[Dict[str, Any]]:
    with open(DEMO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_live_articles(limit_per_source: int = 12) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    articles: List[Dict[str, Any]] = []
    health: List[Dict[str, Any]] = []

    for source in LIVE_RSS_SOURCES:
        source_name = source["name"]
        url = source["url"]

        try:
            feed = feedparser.parse(url)

            if getattr(feed, "bozo", 0):
                status = f"warning: {type(feed.bozo_exception).__name__}"
            else:
                status = "ok"

            kept = 0
            for entry in getattr(feed, "entries", [])[:limit_per_source]:
                articles.append(
                    {
                        "source": source_name,
                        "title": entry.get("title", "") or "",
                        "summary": entry.get("summary", "") or entry.get("description", "") or "",
                        "link": entry.get("link", "") or "",
                        "published": entry.get("published", "") or entry.get("updated", "") or "",
                    }
                )
                kept += 1

            health.append(
                {
                    "source": source_name,
                    "status": status,
                    "articles_kept": kept,
                    "url": url,
                }
            )

        except Exception as e:
            health.append(
                {
                    "source": source_name,
                    "status": f"error: {type(e).__name__}",
                    "articles_kept": 0,
                    "url": url,
                }
            )

    return articles, health