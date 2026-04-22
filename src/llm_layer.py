from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
from google import genai

MODEL_NAME = "gemini-2.5-flash"


def get_client() -> genai.Client:
    return genai.Client()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _safe_json_loads(text: str) -> Dict:
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def build_relevance_prompt(story: Dict, articles: List[Dict]) -> str:
    payload = {
        "task": (
            "You are scoring whether each article is good evidence for a market story. "
            "Return strict JSON only. No markdown. No explanation outside JSON."
        ),
        "story": {
            "narrative": story.get("narrative", ""),
            "headline": story.get("headline", ""),
            "plain_english": story.get("plain_english", ""),
            "why_it_matters": story.get("why_it_matters", ""),
            "companies_mentioned": story.get("companies_mentioned", []),
        },
        "articles": articles,
        "rules": [
            "A strong article directly supports the same market story.",
            "A weak article is related but only partially supports it.",
            "An irrelevant article should be marked irrelevant.",
            "Be strict. Do not force a match.",
            "Prefer semantic relevance over company-name overlap alone."
        ],
        "output_schema": {
            "results": [
                {
                    "article_id": "string",
                    "label": "strong|weak|irrelevant",
                    "relevance_score": "integer from 0 to 100",
                    "reason": "short string"
                }
            ]
        }
    }
    return json.dumps(payload, ensure_ascii=False)


def score_story_evidence_with_gemini(story: Dict, articles: List[Dict]) -> Dict:
    client = get_client()
    prompt = build_relevance_prompt(story, articles)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    text = response.text or ""
    return _safe_json_loads(text)


def rerank_evidence_df(story: Dict, evidence_df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if evidence_df is None or evidence_df.empty:
        return evidence_df

    working = evidence_df.copy().reset_index(drop=True)

    articles = []
    for idx, row in working.iterrows():
        articles.append(
            {
                "article_id": str(idx),
                "title": str(row.get("title_clean", "")),
                "source": str(row.get("source", "")),
                "primary_company": str(row.get("primary_company", "")),
                "published_dt": str(row.get("published_dt", "")),
            }
        )

    try:
        result = score_story_evidence_with_gemini(story, articles)
        scored_items = result.get("results", [])

        score_map = {}
        for item in scored_items:
            article_id = str(item.get("article_id", ""))
            score_map[article_id] = {
                "label": item.get("label", "irrelevant"),
                "score": int(item.get("relevance_score", 0)),
                "reason": item.get("reason", ""),
            }

        working["_article_id"] = [str(i) for i in range(len(working))]
        working["_llm_label"] = working["_article_id"].map(
            lambda x: score_map.get(x, {}).get("label", "irrelevant")
        )
        working["_llm_score"] = working["_article_id"].map(
            lambda x: score_map.get(x, {}).get("score", 0)
        )

        label_rank = {"strong": 0, "weak": 1, "irrelevant": 2}
        working["_label_rank"] = working["_llm_label"].map(lambda x: label_rank.get(x, 2))

        if (working["_llm_label"] != "irrelevant").any():
            working = working[working["_llm_label"] != "irrelevant"].copy()

        working = working.sort_values(
            by=["_label_rank", "_llm_score"],
            ascending=[True, False],
            na_position="last",
        )

        return working.drop(
            columns=["_article_id", "_llm_label", "_llm_score", "_label_rank"],
            errors="ignore",
        ).head(limit).copy()

    except Exception:
        return evidence_df.head(limit).copy()