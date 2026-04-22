from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "marketpulse.db"


def init_db(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            fingerprint TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            summary TEXT,
            link TEXT,
            published TEXT,
            published_dt TEXT,
            title_clean TEXT,
            summary_clean TEXT,
            combined_text TEXT,
            primary_ticker TEXT,
            primary_company TEXT,
            sector TEXT,
            label TEXT,
            confidence REAL,
            sentiment_score REAL,
            evidence_terms TEXT,
            uncertainty_flag INTEGER,
            ambiguity_reason TEXT,
            primary_narrative TEXT,
            matched_narratives TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def _fingerprint(row: pd.Series) -> str:
    raw = "||".join(
        [
            str(row.get("source", "")),
            str(row.get("title_clean", "")),
            str(row.get("published", "")),
            str(row.get("link", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def store_articles(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    if df.empty:
        return 0

    init_db(db_path)
    conn = sqlite3.connect(db_path)
    inserted = 0

    for _, row in df.iterrows():
        fingerprint = _fingerprint(row)
        values = (
            fingerprint,
            row.get("source", ""),
            row.get("title", ""),
            row.get("summary", ""),
            row.get("link", ""),
            row.get("published", ""),
            str(row.get("published_dt", "")),
            row.get("title_clean", ""),
            row.get("summary_clean", ""),
            row.get("combined_text", ""),
            row.get("primary_ticker", ""),
            row.get("primary_company", ""),
            row.get("sector", ""),
            row.get("label", ""),
            float(row.get("confidence", 0.0)) if pd.notna(row.get("confidence", 0.0)) else 0.0,
            float(row.get("sentiment_score", 0.0)) if pd.notna(row.get("sentiment_score", 0.0)) else 0.0,
            row.get("evidence_terms", ""),
            int(row.get("uncertainty_flag", 0)) if pd.notna(row.get("uncertainty_flag", 0)) else 0,
            row.get("ambiguity_reason", ""),
            row.get("primary_narrative", ""),
            row.get("matched_narratives", ""),
        )

        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO articles (
                fingerprint, source, title, summary, link, published, published_dt,
                title_clean, summary_clean, combined_text,
                primary_ticker, primary_company, sector,
                label, confidence, sentiment_score,
                evidence_terms, uncertainty_flag, ambiguity_reason,
                primary_narrative, matched_narratives
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        inserted += cur.rowcount

    conn.commit()
    conn.close()
    return inserted


def load_history(limit: int = 500, db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT *
        FROM articles
        ORDER BY published_dt DESC
        LIMIT {int(limit)}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df