from __future__ import annotations

from typing import Dict, List

import pandas as pd

NARRATIVE_RULES: Dict[str, List[str]] = {
    "Demand recovery": ["demand", "recovery", "orders", "stronger", "cycle"],
    "Regulation pressure": ["recall", "regulatory", "scrutiny", "review", "pressure"],
    "AI infrastructure": ["ai", "gpu", "cloud", "infrastructure", "workloads"],
    "Earnings beat": ["beats", "estimates", "earnings", "results", "expectations"],
    "Margin improvement": ["margin", "margins", "cost control", "efficiency", "discipline"],
    "Advertising slowdown": ["advertising", "ad growth", "monetization", "engagement"],
    "Cloud growth": ["cloud", "azure", "enterprise", "services"],
    "Energy pressure": ["oil", "supply", "demand concerns", "energy"],
    "Market uncertainty": ["mixed", "uncertain", "lingers", "concerns"]
}


def assign_narratives(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    primary_narratives = []
    matched_narratives = []

    for _, row in out.iterrows():
        text = f"{row.get('title_clean', '')} {row.get('summary_clean', '')}".lower()
        matched = []

        for narrative, keywords in NARRATIVE_RULES.items():
            if any(keyword in text for keyword in keywords):
                matched.append(narrative)

        if not matched:
            matched = ["General market noise"]

        primary_narratives.append(matched[0])
        matched_narratives.append(", ".join(matched))

    out["primary_narrative"] = primary_narratives
    out["matched_narratives"] = matched_narratives

    return out