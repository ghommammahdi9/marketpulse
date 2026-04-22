from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, Tuple

import pandas as pd


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def deduplicate_articles(df: pd.DataFrame, fuzzy_threshold: float = 0.90) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df.empty:
        return df.copy(), {
            "raw_count": 0,
            "kept_count": 0,
            "exact_duplicates_removed": 0,
            "fuzzy_duplicates_removed": 0,
        }

    working = df.copy().reset_index(drop=True)

    if "published_dt" in working.columns:
        working = working.sort_values(by="published_dt", ascending=False, na_position="last").reset_index(drop=True)

    kept_rows = []
    kept_titles = []
    exact_removed = 0
    fuzzy_removed = 0
    seen_fingerprints = set()

    for _, row in working.iterrows():
        title_key = row.get("title_match", "") or ""
        summary_key = row.get("summary_match", "") or ""
        fingerprint = f"{title_key}||{summary_key[:120]}"

        if fingerprint in seen_fingerprints:
            exact_removed += 1
            continue

        is_fuzzy_duplicate = False
        for prev_title in kept_titles:
            if title_similarity(title_key, prev_title) >= fuzzy_threshold:
                is_fuzzy_duplicate = True
                break

        if is_fuzzy_duplicate:
            fuzzy_removed += 1
            continue

        seen_fingerprints.add(fingerprint)
        kept_titles.append(title_key)
        kept_rows.append(row)

    dedup_df = pd.DataFrame(kept_rows).reset_index(drop=True)

    stats = {
        "raw_count": len(df),
        "kept_count": len(dedup_df),
        "exact_duplicates_removed": exact_removed,
        "fuzzy_duplicates_removed": fuzzy_removed,
    }

    return dedup_df, stats