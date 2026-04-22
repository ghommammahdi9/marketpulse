"""
snapshot_manager.py
-------------------
Make Demo mode authentic.

Before: Demo mode loaded a handcrafted JSON that felt staged.
After: Demo mode replays the most recent real pipeline run captured as a
snapshot. The curated JSON stays as a *last-resort* fallback, and the app
labels it honestly ("placeholder dataset") so recruiters know the difference.

Usage
-----
- In Live mode, after a successful run, call `save_snapshot(raw_articles)` to
  persist what the pipeline actually saw. These files live in
  `data/snapshots/snapshot_<UTC timestamp>.json`.
- In Demo mode, call `load_demo_articles()`. It returns the latest snapshot
  (authentic) if one exists, otherwise the curated JSON (placeholder).
- `meta["source"]` is one of:
    - "replayed_snapshot"   → real past extraction being replayed
    - "curated_placeholder" → handcrafted JSON, used only if no snapshots yet
    - "none"                → nothing available (empty state)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
SNAP_DIR = BASE_DIR / "data" / "snapshots"
LEGACY_RICH = BASE_DIR / "data" / "demo_articles_rich.json"


def _ensure_dir() -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)


def save_snapshot(articles: List[Dict], label: Optional[str] = None) -> Path:
    """Persist a real pipeline run so Demo mode can replay it later."""
    _ensure_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"snapshot_{ts}" + (f"_{label}" if label else "") + ".json"
    path = SNAP_DIR / name
    payload = {
        "captured_at": ts,
        "article_count": len(articles),
        "articles": articles,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_snapshots() -> List[Path]:
    _ensure_dir()
    return sorted(SNAP_DIR.glob("snapshot_*.json"), reverse=True)


def load_snapshot(path: Path) -> Tuple[List[Dict], Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = {
        "source": "replayed_snapshot",
        "path": str(path),
        "captured_at": data.get("captured_at"),
        "article_count": data.get("article_count", len(data.get("articles", []))),
    }
    return data.get("articles", []), meta


def load_latest_snapshot() -> Tuple[List[Dict], Dict]:
    snaps = list_snapshots()
    if not snaps:
        return [], {"source": "none", "path": None}
    return load_snapshot(snaps[0])


def load_legacy_rich() -> Tuple[List[Dict], Dict]:
    if not LEGACY_RICH.exists():
        return [], {"source": "none", "path": None}
    articles = json.loads(LEGACY_RICH.read_text(encoding="utf-8"))
    return articles, {
        "source": "curated_placeholder",
        "path": str(LEGACY_RICH),
        "article_count": len(articles),
    }


def load_demo_articles(preferred_snapshot: Optional[Path] = None) -> Tuple[List[Dict], Dict]:
    """
    Return (articles, meta) for Demo mode, preferring a real replayed snapshot
    over the curated JSON placeholder. If `preferred_snapshot` is provided and
    valid, it is used regardless of recency.
    """
    if preferred_snapshot is not None and preferred_snapshot.exists():
        return load_snapshot(preferred_snapshot)
    articles, meta = load_latest_snapshot()
    if articles:
        return articles, meta
    return load_legacy_rich()
