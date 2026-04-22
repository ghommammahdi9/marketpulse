from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_market_mood_index(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round(_clip(df["sentiment_score"].mean() * 100, -100, 100), 1)


def compute_divergence_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    pos = int((df["label"] == "positive").sum())
    neg = int((df["label"] == "negative").sum())

    if pos + neg == 0:
        base_divergence = 0.0
    else:
        base_divergence = 200 * min(pos, neg) / (pos + neg)

    if "source" in df.columns and df["source"].nunique() > 1:
        source_means = df.groupby("source")["sentiment_score"].mean()
        source_component = min(100.0, abs(source_means.max() - source_means.min()) * 50)
    else:
        source_component = 0.0

    return round(0.6 * base_divergence + 0.4 * source_component, 1)


def compute_signal_quality(df: pd.DataFrame, health: List[Dict], dedup_stats: Dict) -> float:
    if df.empty:
        return 0.0

    mapped_ratio = float((df["primary_ticker"] != "UNMAPPED").mean()) if "primary_ticker" in df.columns else 0.0

    if health:
        ok_sources = sum(1 for h in health if str(h.get("status", "")).startswith("ok") or str(h.get("status", "")).startswith("warning"))
        source_ratio = ok_sources / len(health)
    else:
        source_ratio = 0.0

    raw_count = dedup_stats.get("raw_count", 0)
    kept_count = dedup_stats.get("kept_count", 0)
    uniqueness_ratio = (kept_count / raw_count) if raw_count else 1.0

    uncertainty_ratio = float(df["uncertainty_flag"].mean()) if "uncertainty_flag" in df.columns else 0.0
    uncertainty_factor = max(0.0, 1 - uncertainty_ratio * 0.35)

    quality = 100 * ((mapped_ratio + source_ratio + uniqueness_ratio) / 3) * uncertainty_factor
    return round(_clip(quality, 0, 100), 1)


def build_summary(df: pd.DataFrame, health: List[Dict], dedup_stats: Dict) -> Dict:
    if df.empty:
        return {
            "article_count": 0,
            "market_mood_index": 0.0,
            "divergence_score": 0.0,
            "signal_quality": 0.0,
            "dominant_narrative": "N/A",
            "top_company": "N/A",
            "uncertainty_count": 0,
        }

    narrative_counts = df["primary_narrative"].value_counts()
    dominant_narrative = narrative_counts.index[0] if not narrative_counts.empty else "N/A"

    mapped = df[df["primary_company"] != "Unmapped"] if "primary_company" in df.columns else df
    top_company = mapped["primary_company"].value_counts().index[0] if not mapped.empty else "Unmapped"

    return {
        "article_count": int(len(df)),
        "market_mood_index": compute_market_mood_index(df),
        "divergence_score": compute_divergence_score(df),
        "signal_quality": compute_signal_quality(df, health, dedup_stats),
        "dominant_narrative": dominant_narrative,
        "top_company": top_company,
        "uncertainty_count": int(df["uncertainty_flag"].sum()) if "uncertainty_flag" in df.columns else 0,
    }