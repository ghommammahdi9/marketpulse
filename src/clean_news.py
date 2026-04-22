from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
from bs4 import BeautifulSoup


def strip_html(text: str) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def normalize_text(text: str) -> str:
    text = strip_html(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_matching(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_parse_datetime(value: str):
    if not value:
        return pd.NaT
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def articles_to_dataframe(articles: List[Dict[str, Any]]) -> pd.DataFrame:
    if not articles:
        return pd.DataFrame(
            columns=[
                "source",
                "title",
                "summary",
                "link",
                "published",
                "title_clean",
                "summary_clean",
                "title_match",
                "summary_match",
                "combined_text",
                "published_dt",
            ]
        )

    df = pd.DataFrame(articles).copy()

    for col in ["source", "title", "summary", "link", "published"]:
        if col not in df.columns:
            df[col] = ""

    df["source"] = df["source"].fillna("")
    df["title"] = df["title"].fillna("")
    df["summary"] = df["summary"].fillna("")
    df["link"] = df["link"].fillna("")
    df["published"] = df["published"].fillna("")

    df["title_clean"] = df["title"].apply(normalize_text)
    df["summary_clean"] = df["summary"].apply(normalize_text)
    df["title_match"] = df["title"].apply(normalize_for_matching)
    df["summary_match"] = df["summary"].apply(normalize_for_matching)
    df["combined_text"] = (df["title_clean"] + ". " + df["summary_clean"]).str.strip(". ").str.strip()
    df["published_dt"] = df["published"].apply(safe_parse_datetime)

    return df