from src.llm_layer import score_story_evidence_with_gemini

story = {
    "narrative": "AI infrastructure",
    "headline": "L’infrastructure IA reste au centre du flux",
    "plain_english": "Les entreprises continuent d’investir dans les serveurs, puces et services cloud liés à l’IA.",
    "why_it_matters": "Ce cycle d’investissement soutient les groupes technologiques qui vendent du calcul, du cloud et des composants.",
    "companies_mentioned": ["Amazon", "Apple", "Nvidia"],
}

articles = [
    {
        "article_id": "a1",
        "title": "Nvidia demand for AI chips remains strong as cloud spending rises",
        "source": "Reuters",
        "primary_company": "Nvidia",
    },
    {
        "article_id": "a2",
        "title": "Tesla investors wait for quarterly earnings results",
        "source": "MarketWatch",
        "primary_company": "Tesla",
    },
    {
        "article_id": "a3",
        "title": "Amazon expands cloud infrastructure for enterprise AI workloads",
        "source": "CNBC",
        "primary_company": "Amazon",
    },
]

result = score_story_evidence_with_gemini(story, articles)
print(result)