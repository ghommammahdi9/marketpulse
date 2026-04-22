from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
COMPANY_MAP_FILE = BASE_DIR / "data" / "company_map.json"

MACRO_BUCKETS = {
    "MACRO": {
        "company": "Macro Market",
        "aliases": ["fed", "federal reserve", "ecb", "inflation", "rates", "interest rates", "treasury", "bond yields", "recession", "economy"],
        "sector": "Macro"
    },
    "COMMOD": {
        "company": "Commodities",
        "aliases": ["oil", "crude", "brent", "gold", "copper", "natural gas", "commodity"],
        "sector": "Commodities"
    },
    "CRYPTO": {
        "company": "Crypto Market",
        "aliases": ["bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency"],
        "sector": "Crypto"
    },
    "REG": {
        "company": "Regulation / Policy",
        "aliases": ["regulation", "regulatory", "antitrust", "lawsuit", "policy", "tariff", "sanction"],
        "sector": "Policy"
    }
}


def load_company_map() -> Dict:
    with open(COMPANY_MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def contains_alias(text: str, alias: str) -> bool:
    pattern = r"\b" + re.escape(alias.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def map_entities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    company_map = load_company_map()
    out = df.copy()

    primary_tickers: List[str] = []
    primary_companies: List[str] = []
    sectors: List[str] = []
    all_matches: List[str] = []

    for _, row in out.iterrows():
        text = f"{row.get('title_clean', '')} {row.get('summary_clean', '')}".lower()

        matched = []
        for ticker, payload in company_map.items():
            aliases = payload.get("aliases", [])
            if any(contains_alias(text, alias) for alias in aliases):
                matched.append(ticker)

        if matched:
            first = matched[0]
            primary_tickers.append(first)
            primary_companies.append(company_map[first]["company"])
            sectors.append(company_map[first]["sector"])
            all_matches.append(", ".join(matched))
        else:
            macro_match = None
            for bucket_ticker, payload in MACRO_BUCKETS.items():
                if any(contains_alias(text, alias) for alias in payload["aliases"]):
                    macro_match = bucket_ticker
                    break

            if macro_match:
                primary_tickers.append(macro_match)
                primary_companies.append(MACRO_BUCKETS[macro_match]["company"])
                sectors.append(MACRO_BUCKETS[macro_match]["sector"])
                all_matches.append(macro_match)
            else:
                primary_tickers.append("UNMAPPED")
                primary_companies.append("Unmapped")
                sectors.append("Unknown")
                all_matches.append("")

    out["primary_ticker"] = primary_tickers
    out["primary_company"] = primary_companies
    out["sector"] = sectors
    out["matched_tickers"] = all_matches

    return out