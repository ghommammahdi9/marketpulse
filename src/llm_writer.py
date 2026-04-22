from __future__ import annotations

import json
from typing import Dict

import pandas as pd
import streamlit as st
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
    return json.loads(_strip_code_fences(text))


def _evidence_payload(evidence_df: pd.DataFrame, limit: int = 3) -> list[dict]:
    evidence = []
    if evidence_df is None or evidence_df.empty:
        return evidence

    for _, row in evidence_df.head(limit).iterrows():
        evidence.append(
            {
                "title": str(row.get("title_clean", "")),
                "source": str(row.get("source", "")),
                "published_dt": str(row.get("published_dt", "")),
                "company": str(row.get("primary_company", "")),
            }
        )
    return evidence


def _fallback_headline_from_context(data: Dict) -> str:
    companies = data.get("companies", []) or data.get("companies_mentioned", []) or []
    narrative = str(data.get("narrative", "")).strip()

    lead = ", ".join(companies[:2]) if companies else ""

    mapping = {
        "AI infrastructure": "L’infrastructure IA reste le signal dominant",
        "Earnings beat": "Les résultats restent au centre de la lecture du marché",
        "Demand recovery": "Des signes de reprise de la demande ressortent",
        "Regulation pressure": "La pression réglementaire refait surface",
        "Margin improvement": "Les marges montrent une amélioration",
        "Cloud growth": "Le cloud continue de soutenir la croissance",
        "Advertising slowdown": "Le signal publicitaire ralentit",
        "Energy pressure": "La pression sur l’énergie reste présente",
        "Market uncertainty": "Le marché reste hésitant",
    }

    base = mapping.get(narrative, "Un signal de marché ressort du flux")
    if lead:
        return f"{base} — {lead}"
    return base


def _fallback_one_line_from_context(data: Dict) -> str:
    companies = data.get("companies", []) or data.get("companies_mentioned", []) or []
    narrative = str(data.get("narrative", "")).strip()

    lead = ", ".join(companies[:2]) if companies else "Le flux du jour"

    mapping = {
        "AI infrastructure": f"{lead} restent associés à la demande en calcul, cloud et composants liés à l’IA.",
        "Earnings beat": f"{lead} tirent le flux grâce à des publications perçues comme plus solides que prévu.",
        "Demand recovery": f"{lead} bénéficient d’indices de reprise sur la demande ou les volumes.",
        "Regulation pressure": f"{lead} sont rattrapés par un risque réglementaire plus visible.",
        "Margin improvement": f"{lead} profitent d’un discours de marché centré sur l’amélioration des marges.",
        "Cloud growth": f"{lead} restent soutenus par la croissance du cloud et des usages entreprise.",
        "Advertising slowdown": f"{lead} subissent une lecture plus prudente autour du marché publicitaire.",
        "Energy pressure": f"{lead} restent exposés à une pression liée à l’énergie ou aux matières premières.",
        "Market uncertainty": f"{lead} s’inscrivent dans un flux encore trop mitigé pour produire une conviction forte.",
    }

    return mapping.get(narrative, f"{lead} concentrent aujourd’hui l’essentiel du signal de marché.")


def _is_vague(text: str) -> bool:
    if not text:
        return True

    t = text.strip().lower()

    vague_starts = [
        "des résultats",
        "une amélioration",
        "des signes",
        "le marché",
        "une hausse",
        "une baisse",
        "le flux",
        "des publications",
    ]

    for start in vague_starts:
        if t.startswith(start):
            return True

    short_bad_patterns = [
        "améliorent",
        "soutiennent",
        "pèsent sur",
        "profitent de",
    ]

    if len(t) < 55 and any(p in t for p in short_bad_patterns):
        return True

    return False


def _sanitize_rewritten_output(base: Dict, out: Dict) -> Dict:
    cleaned = dict(out)

    if _is_vague(str(cleaned.get("headline", ""))):
        cleaned["headline"] = _fallback_headline_from_context(base)

    if _is_vague(str(cleaned.get("one_line", ""))):
        cleaned["one_line"] = _fallback_one_line_from_context(base)

    return cleaned


def _rewrite_pulse_impl(pulse: Dict, evidence: list[dict]) -> Dict:
    payload = {
        "task": (
            "Rewrite a market pulse for a financial news product in sharp, grounded French. "
            "Stay strictly faithful to the supplied pulse and evidence. "
            "Return strict JSON only."
        ),
        "rules": [
            "Use concise, product-quality French.",
            "Do not invent facts.",
            "Do not exaggerate certainty.",
            "The headline must mention a company or a concrete market theme when possible.",
            "Avoid vague openings such as 'Des résultats', 'Le marché', 'Des signes', 'Une amélioration'.",
            "The one-line summary must explicitly answer: who, what happened, why it matters.",
            "Do not write generic finance clichés.",
            "Keep the market-regime sentence sober.",
        ],
        "style_constraints": {
            "headline": "8 to 16 words, concrete, not vague, not generic",
            "one_line": "1 sentence, concrete subject first",
            "why_it_matters": "1 to 2 sentences, practical market consequence",
            "tomorrow_morning": "1 sentence, what to check next",
            "regime_line": "short sober sentence"
        },
        "input": {
            "pulse": {
                "headline": pulse.get("headline", ""),
                "one_line": pulse.get("one_line", ""),
                "why_it_matters": pulse.get("why_it_matters", ""),
                "tomorrow_morning": pulse.get("tomorrow_morning", ""),
                "regime_line": pulse.get("regime_line", ""),
                "narrative": pulse.get("narrative", ""),
                "companies": pulse.get("companies", []),
            },
            "evidence": evidence,
        },
        "output_schema": {
            "headline": "string",
            "one_line": "string",
            "why_it_matters": "string",
            "tomorrow_morning": "string",
            "regime_line": "string",
        },
    }

    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=json.dumps(payload, ensure_ascii=False),
    )
    parsed = _safe_json_loads(response.text or "")

    out = pulse.copy()
    out["headline"] = parsed.get("headline", pulse.get("headline", ""))
    out["one_line"] = parsed.get("one_line", pulse.get("one_line", ""))
    out["why_it_matters"] = parsed.get("why_it_matters", pulse.get("why_it_matters", ""))
    out["tomorrow_morning"] = parsed.get("tomorrow_morning", pulse.get("tomorrow_morning", ""))
    out["regime_line"] = parsed.get("regime_line", pulse.get("regime_line", ""))
    return _sanitize_rewritten_output(pulse, out)


def _rewrite_story_impl(story: Dict, evidence: list[dict]) -> Dict:
    payload = {
        "task": (
            "Rewrite a secondary financial market story for a product UI in sharp, grounded French. "
            "Stay strictly faithful to the supplied story and evidence. "
            "Return strict JSON only."
        ),
        "rules": [
            "Use concise, natural French.",
            "Do not invent facts.",
            "Do not overstate certainty.",
            "The headline must mention a company or a concrete market theme when possible.",
            "Avoid vague openings such as 'Des résultats', 'Le marché', 'Des signes', 'Une amélioration'.",
            "Make the plain-English sentence easy to understand and concrete.",
            "The audience field should be concrete and useful.",
        ],
        "style_constraints": {
            "headline": "8 to 16 words, concrete, not vague",
            "plain_english": "1 sentence, explicit subject first",
            "why_it_matters": "1 short practical explanation",
            "who_it_affects": "specific audience, not generic",
            "watch_next": "1 sentence"
        },
        "input": {
            "story": {
                "headline": story.get("headline", ""),
                "plain_english": story.get("plain_english", ""),
                "why_it_matters": story.get("why_it_matters", ""),
                "who_it_affects": story.get("who_it_affects", ""),
                "watch_next": story.get("watch_next", ""),
                "narrative": story.get("narrative", ""),
                "companies_mentioned": story.get("companies_mentioned", []),
            },
            "evidence": evidence,
        },
        "output_schema": {
            "headline": "string",
            "plain_english": "string",
            "why_it_matters": "string",
            "who_it_affects": "string",
            "watch_next": "string",
        },
    }

    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=json.dumps(payload, ensure_ascii=False),
    )
    parsed = _safe_json_loads(response.text or "")

    out = story.copy()
    out["headline"] = parsed.get("headline", story.get("headline", ""))
    out["plain_english"] = parsed.get("plain_english", story.get("plain_english", ""))
    out["why_it_matters"] = parsed.get("why_it_matters", story.get("why_it_matters", ""))
    out["who_it_affects"] = parsed.get("who_it_affects", story.get("who_it_affects", ""))
    out["watch_next"] = parsed.get("watch_next", story.get("watch_next", ""))
    return _sanitize_rewritten_output(story, out)


@st.cache_data(show_spinner=False, ttl=900)
def _cached_rewrite_pulse(pulse_json: str, evidence_json: str) -> Dict:
    pulse = json.loads(pulse_json)
    evidence = json.loads(evidence_json)
    return _rewrite_pulse_impl(pulse, evidence)


@st.cache_data(show_spinner=False, ttl=900)
def _cached_rewrite_story(story_json: str, evidence_json: str) -> Dict:
    story = json.loads(story_json)
    evidence = json.loads(evidence_json)
    return _rewrite_story_impl(story, evidence)


def rewrite_pulse_with_gemini(pulse: Dict, evidence_df: pd.DataFrame) -> Dict:
    if not pulse:
        return pulse

    try:
        evidence = _evidence_payload(evidence_df, limit=3)
        pulse_json = json.dumps(pulse, sort_keys=True, ensure_ascii=False)
        evidence_json = json.dumps(evidence, sort_keys=True, ensure_ascii=False)
        return _cached_rewrite_pulse(pulse_json, evidence_json)
    except Exception:
        return pulse


def rewrite_story_with_gemini(story: Dict, evidence_df: pd.DataFrame) -> Dict:
    if not story:
        return story

    try:
        evidence = _evidence_payload(evidence_df, limit=3)
        story_json = json.dumps(story, sort_keys=True, ensure_ascii=False)
        evidence_json = json.dumps(evidence, sort_keys=True, ensure_ascii=False)
        return _cached_rewrite_story(story_json, evidence_json)
    except Exception:
        return story