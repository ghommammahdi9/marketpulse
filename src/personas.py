from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


PERSONAS = ["Saver", "Borrower", "Investor", "Consumer", "Tech worker"]


def _tone_rank(tone: str) -> int:
    return {"negative": 3, "warning": 2, "positive": 1, "neutral": 0}.get(tone, 0)


def _merge_tone(current: str, new: str) -> str:
    return new if _tone_rank(new) > _tone_rank(current) else current


def story_to_persona_impacts(story: Dict) -> List[Dict]:
    narrative = story.get("narrative", "")
    company = story.get("company", "")
    tone = story.get("tone", "neutral")
    why_it_matters = story.get("why_it_matters", "")
    watch_next = story.get("watch_next", "")

    impacts: List[Dict] = []

    def add(persona: str, label: str, detail: str, impact_tone: str):
        impacts.append(
            {
                "persona": persona,
                "label": label,
                "detail": detail,
                "tone": impact_tone,
                "watch_next": watch_next,
            }
        )

    if narrative in ["Demand recovery", "Earnings beat", "Margin improvement", "Cloud growth", "AI infrastructure"]:
        add("Investor", "Le marché voit un soutien sur certaines actions", why_it_matters, "positive" if tone == "positive" else "warning")
    if narrative in ["Regulation pressure", "Market uncertainty", "Advertising slowdown"]:
        add("Investor", "Le risque d'exécution ou de valorisation remonte", why_it_matters, "warning" if tone != "negative" else "negative")

    if narrative in ["Macro Market", "Market uncertainty"]:
        add("Saver", "Le contexte de taux et d'inflation peut influencer l'épargne", why_it_matters, "warning")
        add("Borrower", "Le coût du crédit peut rester sous pression", why_it_matters, "negative" if tone == "negative" else "warning")

    if narrative in ["Energy pressure"]:
        add("Consumer", "Les prix de l'énergie peuvent influencer le budget quotidien", why_it_matters, "positive" if tone == "positive" else "warning")

    if narrative in ["Demand recovery", "Margin improvement"]:
        add("Consumer", "Le signal touche la consommation et les prix à moyen terme", why_it_matters, "neutral")

    if narrative in ["AI infrastructure", "Cloud growth", "Regulation pressure", "Advertising slowdown"]:
        add("Tech worker", f"{company} et son secteur envoient un signal métier", why_it_matters, "warning" if narrative == "Regulation pressure" else "positive")

    if not impacts:
        add("Investor", "Signal de marché à surveiller", why_it_matters or "Le sujet mérite une veille complémentaire.", "neutral")

    return impacts


def build_persona_panels(stories: List[Dict]) -> Dict[str, Dict]:
    grouped = defaultdict(list)

    for story in stories:
        impacts = story_to_persona_impacts(story)
        for impact in impacts:
            grouped[impact["persona"]].append(impact)

    panels: Dict[str, Dict] = {}

    for persona in PERSONAS:
        items = grouped.get(persona, [])

        if not items:
            panels[persona] = {
                "tone": "neutral",
                "headline": "Pas de signal majeur détecté",
                "details": ["Aucun thème dominant ne ressort pour ce profil dans la sélection actuelle."],
                "watch_next": "Surveiller les prochaines mises à jour.",
            }
            continue

        final_tone = "neutral"
        for item in items:
            final_tone = _merge_tone(final_tone, item["tone"])

        headline = items[0]["label"]
        details = [item["detail"] for item in items[:3]]
        watch_next = items[0]["watch_next"]

        panels[persona] = {
            "tone": final_tone,
            "headline": headline,
            "details": details,
            "watch_next": watch_next,
        }

    return panels