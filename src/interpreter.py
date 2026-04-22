from __future__ import annotations

from typing import Dict, List

import pandas as pd


NARRATIVE_LIBRARY = {
    "AI infrastructure": {
        "plain": "Les entreprises continuent d'investir dans les serveurs, puces et services cloud liés à l'IA.",
        "why": "Cela compte car ce cycle d'investissement soutient les groupes technologiques qui vendent du calcul, du cloud et des composants.",
        "watch": "Surveiller les prochains commentaires de résultats, les budgets cloud et les capacités de production des fournisseurs."
    },
    "Demand recovery": {
        "plain": "Le marché perçoit un début de reprise de la demande sur certaines activités ou produits.",
        "why": "Une demande qui se redresse peut améliorer les revenus attendus et rassurer les investisseurs sur la trajectoire commerciale.",
        "watch": "Vérifier si d'autres sources, fournisseurs ou résultats confirment cette reprise."
    },
    "Earnings beat": {
        "plain": "Des résultats meilleurs que prévu améliorent le ton du marché sur une entreprise ou un secteur.",
        "why": "Des résultats solides peuvent changer les anticipations de bénéfices et influencer la valorisation des actions.",
        "watch": "Regarder si la bonne surprise vient d'un élément durable ou ponctuel."
    },
    "Margin improvement": {
        "plain": "Le marché récompense les entreprises qui contrôlent mieux leurs coûts et améliorent leurs marges.",
        "why": "Une meilleure marge signifie souvent plus de résistance si la croissance ralentit.",
        "watch": "Suivre si cette discipline peut être maintenue sur les prochains trimestres."
    },
    "Regulation pressure": {
        "plain": "Le risque réglementaire augmente et peut freiner certaines entreprises.",
        "why": "Une pression réglementaire peut retarder des produits, alourdir les coûts ou affaiblir la confiance du marché.",
        "watch": "Surveiller les décisions des autorités, les amendes éventuelles et la réaction de la direction."
    },
    "Advertising slowdown": {
        "plain": "Le marché voit des signes de ralentissement sur les revenus publicitaires.",
        "why": "Cela touche directement les plateformes numériques dépendantes de la publicité pour leur croissance.",
        "watch": "Regarder si le ralentissement est temporaire ou devient une tendance plus large."
    },
    "Cloud growth": {
        "plain": "La demande pour les services cloud reste un moteur de croissance important.",
        "why": "Le cloud est un indicateur clé de dépenses technologiques des entreprises et de l'adoption de l'IA.",
        "watch": "Surveiller les niveaux de dépenses des entreprises et les commentaires sur l'utilisation réelle."
    },
    "Energy pressure": {
        "plain": "Le mouvement des prix de l'énergie change les perspectives pour les producteurs et les consommateurs.",
        "why": "Des prix du pétrole plus faibles peuvent alléger certains budgets mais réduire les attentes sur les groupes énergétiques.",
        "watch": "Regarder si le mouvement vient de l'offre, de la demande ou d'un choc géopolitique."
    },
    "Market uncertainty": {
        "plain": "Le marché reçoit des signaux contradictoires et peine à trancher.",
        "why": "Quand le contexte est flou, les décisions d'investissement deviennent plus prudentes et la volatilité peut monter.",
        "watch": "Attendre des confirmations supplémentaires avant d'interpréter le signal comme durable."
    },
    "General market noise": {
        "plain": "Les informations publiées ne permettent pas encore de dégager une direction claire.",
        "why": "Tout ne mérite pas une action immédiate ; une partie du flux quotidien reste du bruit.",
        "watch": "Laisser le sujet évoluer avant d'en tirer une conclusion forte."
    }
}


def sentiment_tone(score: float) -> str:
    if score >= 0.20:
        return "positive"
    if score <= -0.20:
        return "negative"
    return "warning"


def classify_importance(article_count: int, source_count: int, narrative: str) -> str:
    if article_count >= 3 and source_count >= 2:
        return "Élevée"
    if narrative in ["Regulation pressure", "Macro Market", "Earnings beat", "Energy pressure"]:
        return "Élevée"
    if article_count >= 2:
        return "Moyenne"
    return "Faible"


def classify_confidence(article_count: int, source_count: int, uncertainty_ratio: float) -> str:
    if source_count >= 2 and uncertainty_ratio < 0.35 and article_count >= 2:
        return "Haute"
    if source_count >= 1 and uncertainty_ratio < 0.60:
        return "Moyenne"
    return "Faible"


def signal_label(confidence: str, uncertainty_ratio: float, tone: str) -> str:
    if confidence == "Haute" and uncertainty_ratio < 0.25:
        return "Signal fort"
    if confidence == "Faible" or uncertainty_ratio >= 0.60:
        return "Signal faible"
    return "Signal mitigé"


def who_it_affects(narrative: str, sector: str) -> str:
    if narrative in ["Macro Market", "Market uncertainty"]:
        return "Épargnants, emprunteurs et investisseurs."
    if narrative in ["Energy pressure"]:
        return "Consommateurs, industriels et investisseurs énergie."
    if sector in ["Technology", "Semiconductors", "Enterprise Software"]:
        return "Investisseurs tech, salariés du secteur et clients entreprises."
    if sector in ["Banking", "Payments"]:
        return "Investisseurs financiers, ménages et entreprises exposées au crédit."
    if sector in ["Consumer / Cloud", "Media / Streaming", "Luxury"]:
        return "Consommateurs et investisseurs exposés à la demande."
    return "Investisseurs et observateurs du marché."


def build_story_cards(df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
    if df.empty:
        return []

    working = df.copy()

    working["primary_narrative"] = working["primary_narrative"].fillna("General market noise")
    working["sector"] = working["sector"].fillna("Unknown")
    working["primary_company"] = working["primary_company"].fillna("Unmapped")

    def compute_bucket(row):
        narrative = row["primary_narrative"]
        sector = row["sector"]
        company = row["primary_company"]

        if narrative in {
            "Demand recovery",
            "Earnings beat",
            "Margin improvement",
            "AI infrastructure",
            "Cloud growth",
            "Advertising slowdown",
            "Regulation pressure",
            "Energy pressure",
            "Macro Market",
            "Market uncertainty",
        }:
            if sector not in ["Unknown", ""]:
                return sector
            if company not in ["Unmapped", "Marché / contexte"]:
                return company
        return "Market"

    working["story_bucket"] = working.apply(compute_bucket, axis=1)
    working["story_key"] = working["primary_narrative"] + " | " + working["story_bucket"]

    def story_quality_score(article_count, source_count, mentioned_companies, uncertainty_ratio, narrative):
        score = 0
        score += min(article_count, 5) * 2
        score += min(source_count, 4) * 3
        score += min(len(mentioned_companies), 3) * 2

        if uncertainty_ratio >= 0.50:
            score -= 3
        elif uncertainty_ratio >= 0.30:
            score -= 1

        if article_count == 1:
            score -= 4
        if source_count == 1:
            score -= 2
        if not mentioned_companies:
            score -= 2
        if narrative == "General market noise":
            score -= 5

        return score

    stories: List[Dict] = []

    for _, group in working.groupby("story_key"):
        group = group.sort_values("published_dt", ascending=False, na_position="last")
        first = group.iloc[0]

        narrative = first.get("primary_narrative", "General market noise")
        bucket = first.get("story_bucket", "Market")

        article_count = len(group)
        source_count = group["source"].nunique()
        uncertainty_ratio = float(group["uncertainty_flag"].mean()) if "uncertainty_flag" in group.columns else 0.0
        avg_sentiment = float(group["sentiment_score"].mean()) if "sentiment_score" in group.columns else 0.0

        mentioned_companies = sorted([
            c for c in group["primary_company"].dropna().unique().tolist()
            if c not in ["Unmapped", "Marché / contexte", "Macro Market", "Commodities", "Crypto Market", "Regulation / Policy"]
        ])

        sector_values = [
            s for s in group["sector"].dropna().unique().tolist()
            if s not in ["Unknown", ""]
        ]
        sector = sector_values[0] if sector_values else "Unknown"

        # Hard filters to remove junk stories
        if narrative == "General market noise" and article_count < 3:
            continue
        if article_count == 1 and source_count == 1 and not mentioned_companies:
            continue
        if article_count < 2 and not mentioned_companies:
            continue

        lib = NARRATIVE_LIBRARY.get(narrative, NARRATIVE_LIBRARY["General market noise"])
        tone = sentiment_tone(avg_sentiment)
        importance = classify_importance(article_count, source_count, narrative)
        confidence = classify_confidence(article_count, source_count, uncertainty_ratio)
        signal = signal_label(confidence, uncertainty_ratio, tone)

        quality_score = story_quality_score(
            article_count=article_count,
            source_count=source_count,
            mentioned_companies=mentioned_companies,
            uncertainty_ratio=uncertainty_ratio,
            narrative=narrative,
        )

        if quality_score < 2:
            continue

        if bucket == "Market":
            headline = narrative
        else:
            company_display = ", ".join(mentioned_companies[:2]) if mentioned_companies else bucket
            headline = f"{narrative} — {company_display}"

        stories.append(
            {
                "headline": headline,
                "plain_english": lib["plain"],
                "why_it_matters": lib["why"],
                "watch_next": lib["watch"],
                "who_it_affects": who_it_affects(narrative, sector),
                "importance": importance,
                "confidence": confidence,
                "signal": signal,
                "tone": tone,
                "narrative": narrative,
                "company": mentioned_companies[0] if mentioned_companies else "Marché / contexte",
                "companies_mentioned": mentioned_companies,
                "sector": sector,
                "sector_bucket": bucket,
                "article_count": article_count,
                "source_count": source_count,
                "uncertainty_ratio": uncertainty_ratio,
                "avg_sentiment": avg_sentiment,
                "sample_title": first.get("title_clean", ""),
                "quality_score": quality_score,
            }
        )

    importance_rank = {"Élevée": 3, "Moyenne": 2, "Faible": 1}
    signal_rank = {"Signal fort": 3, "Signal mitigé": 2, "Signal faible": 1}

    stories = sorted(
        stories,
        key=lambda x: (
            x.get("quality_score", 0),
            importance_rank.get(x["importance"], 0),
            signal_rank.get(x["signal"], 0),
            x["article_count"],
            x["source_count"],
        ),
        reverse=True,
    )

    return stories[:top_n]              

    importance_rank = {"Élevée": 3, "Moyenne": 2, "Faible": 1}
    signal_rank = {"Signal fort": 3, "Signal mitigé": 2, "Signal faible": 1}

    stories = sorted(
        stories,
        key=lambda x: (
            importance_rank.get(x["importance"], 0),
            signal_rank.get(x["signal"], 0),
            x["article_count"],
            x["source_count"]
        ),
        reverse=True,
    )

    return stories[:top_n]