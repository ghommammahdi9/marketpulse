from __future__ import annotations

import re
from typing import List

import pandas as pd
from transformers import pipeline

_SENTIMENT_PIPE = None

POSITIVE_TERMS = [
    "beat", "beats", "growth", "strong", "stronger", "recovery", "recover",
    "gain", "gains", "upside", "robust", "resilient", "improves", "improve",
    "tailwind", "lift", "lifts", "expands", "expand", "healthy", "support"
]

NEGATIVE_TERMS = [
    "recall", "pressure", "weak", "weaker", "slow", "slows", "slowdown",
    "risk", "risks", "scrutiny", "cuts", "cut", "decline", "slip", "drops",
    "drop", "concern", "concerns", "constraint", "constraints", "soft", "cools"
]

AMBIGUITY_TERMS = [
    "but", "however", "while", "despite", "mixed", "although", "yet", "offset"
]


def get_sentiment_pipeline():
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is None:
        _SENTIMENT_PIPE = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
    return _SENTIMENT_PIPE


def _find_terms(text: str, terms: List[str]) -> List[str]:
    lowered = text.lower()
    hits = []
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            hits.append(term)
    return hits


def _build_explanation(text: str, label: str, confidence: float):
    pos_hits = _find_terms(text, POSITIVE_TERMS)
    neg_hits = _find_terms(text, NEGATIVE_TERMS)
    amb_hits = _find_terms(text, AMBIGUITY_TERMS)

    evidence = pos_hits[:3] + neg_hits[:3]
    if not evidence:
        evidence = amb_hits[:2]

    uncertainty_flag = False
    reasons = []

    if confidence < 0.65:
        uncertainty_flag = True
        reasons.append("confiance modèle modérée")

    if label == "neutral":
        uncertainty_flag = True
        reasons.append("signal neutre")

    if pos_hits and neg_hits:
        uncertainty_flag = True
        reasons.append("indices positifs et négatifs simultanés")

    if amb_hits:
        uncertainty_flag = True
        reasons.append("formulation contrastée")

    ambiguity_reason = " ; ".join(reasons) if reasons else "signal cohérent"

    return evidence, uncertainty_flag, ambiguity_reason


def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    texts = out["combined_text"].fillna("").tolist()

    pipe = get_sentiment_pipeline()
    results = pipe(texts, truncation=True, max_length=512, batch_size=8)

    labels = []
    confidences = []
    sentiment_scores = []
    evidence_terms = []
    uncertainty_flags = []
    ambiguity_reasons = []

    for text, result in zip(texts, results):
        label = result["label"].lower()
        confidence = float(result["score"])

        if label == "positive":
            sentiment_score = confidence
        elif label == "negative":
            sentiment_score = -confidence
        else:
            sentiment_score = 0.0

        evidence, uncertainty_flag, ambiguity_reason = _build_explanation(
            text=text,
            label=label,
            confidence=confidence,
        )

        labels.append(label)
        confidences.append(confidence)
        sentiment_scores.append(sentiment_score)
        evidence_terms.append(", ".join(evidence))
        uncertainty_flags.append(int(uncertainty_flag))
        ambiguity_reasons.append(ambiguity_reason)

    out["label"] = labels
    out["confidence"] = confidences
    out["sentiment_score"] = sentiment_scores
    out["evidence_terms"] = evidence_terms
    out["uncertainty_flag"] = uncertainty_flags
    out["ambiguity_reason"] = ambiguity_reasons

    return out