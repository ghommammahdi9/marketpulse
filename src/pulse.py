"""
pulse.py
--------
Compute *one* hero read from the list of story cards produced by interpreter.py.

This module is the centerpiece decision-maker. The rest of the app can keep
showing depth, but the user should open MarketPulse and get a single clear
answer within 5 seconds:

    - What is the thing that matters today?
    - What should I do tomorrow morning?
    - How confident is this read?

The scoring combines importance, signal strength, confidence, source coverage
and article volume. "General market noise" is floored to zero so it never wins.
"""

from __future__ import annotations

from typing import Dict, List, Optional


IMPORTANCE_WEIGHT = {"Élevée": 3.0, "Moyenne": 1.5, "Faible": 0.5}
SIGNAL_WEIGHT = {"Signal fort": 3.0, "Signal mitigé": 1.5, "Signal faible": 0.5}
CONFIDENCE_WEIGHT = {"Haute": 1.5, "Moyenne": 1.0, "Faible": 0.5}


def _score_story(story: Dict) -> float:
    if story.get("narrative") == "General market noise":
        return 0.0
    imp = IMPORTANCE_WEIGHT.get(story.get("importance"), 1.0)
    sig = SIGNAL_WEIGHT.get(story.get("signal"), 1.0)
    conf = CONFIDENCE_WEIGHT.get(story.get("confidence"), 1.0)
    coverage = 1.0 + 0.25 * max(story.get("source_count", 1) - 1, 0)
    volume = 1.0 + 0.10 * max(story.get("article_count", 1) - 1, 0)
    return imp * sig * conf * coverage * volume


def _tomorrow_action(story: Dict) -> str:
    """
    Concrete, one-sentence action the reader can take at market open.
    Tied to narrative type so it doesn't feel generic.
    """
    narrative = story.get("narrative", "")
    mapping = {
        "AI infrastructure": (
            "Regarder les premières réactions des fournisseurs cloud et "
            "des fabricants de puces à l'ouverture."
        ),
        "Earnings beat": (
            "Observer si la hausse tient 48 heures ou s'efface après le bruit initial."
        ),
        "Macro Market": (
            "Vérifier si les taux longs confirment ou contredisent le récit d'aujourd'hui."
        ),
        "Regulation pressure": (
            "Guetter tout commentaire officiel ou nouveau dépôt réglementaire."
        ),
        "Energy pressure": (
            "Contrôler le prix du baril et les réactions des majors énergétiques."
        ),
        "Demand recovery": (
            "Lire les premiers volumes de trading pour voir si le marché y croit."
        ),
        "Margin improvement": (
            "Chercher confirmation sur d'autres titres du même secteur."
        ),
        "Cloud growth": (
            "Suivre les commentaires de résultats cloud des prochains trimestres."
        ),
        "Advertising slowdown": (
            "Surveiller les guidances publicitaires des autres plateformes."
        ),
        "Market uncertainty": (
            "Ne rien faire. Attendre un signal plus clair avant d'agir."
        ),
    }
    return mapping.get(
        narrative,
        story.get("watch_next", "Revenir avec un regard neuf demain matin."),
    )


def _regime_line(stories: List[Dict]) -> str:
    """
    Single-sentence description of the overall market posture.
    Derived from the tone distribution across the top stories.
    """
    if not stories:
        return "Flux trop pauvre pour dégager un régime aujourd'hui."
    tones = [s.get("tone") for s in stories]
    pos = tones.count("positive")
    neg = tones.count("negative")
    warn = tones.count("warning")
    if pos >= 2 and neg == 0:
        return "Le marché est en posture offensive : plusieurs thèmes se renforcent."
    if neg >= 2:
        return "Le marché est en posture défensive : la prudence domine les signaux forts."
    if warn >= 2:
        return "Le marché est en suspens : des signaux exploitables, mais pas de conviction claire."
    if pos == 1 and neg == 1:
        return "Le marché est écartelé : bonne nouvelle et mauvaise nouvelle se contrebalancent."
    return "Le marché est en transition : le décor change, la direction reste à confirmer."


def _confidence_pct(story: Dict) -> int:
    """
    Map qualitative confidence + coverage + uncertainty into a 10–95 % band.
    Never 0 (would look broken) and never 100 (would look dishonest).
    """
    base = {"Haute": 78, "Moyenne": 55, "Faible": 32}.get(
        story.get("confidence", "Moyenne"), 50
    )
    bonus = min(story.get("source_count", 1) - 1, 3) * 4
    penalty = int(round(story.get("uncertainty_ratio", 0.0) * 25))
    return max(10, min(95, base + bonus - penalty))


def _one_line_read(story: Dict) -> str:
    """
    Strip the plain-English story down to a single dense sentence usable as a
    hero subtitle. If the plain text is already short enough, use it as-is.
    """
    plain = story.get("plain_english", "").strip()
    if not plain:
        return story.get("why_it_matters", "").strip()
    if "." in plain:
        first = plain.split(".")[0].strip()
        if 40 <= len(first) <= 180:
            return first + "."
    return plain


def compute_pulse(stories: List[Dict]) -> Optional[Dict]:
    """
    Return the single hero read, or None if no story is strong enough to lead.
    """
    if not stories:
        return None
    ranked = sorted(stories, key=_score_story, reverse=True)
    top = ranked[0]
    if _score_story(top) == 0.0:
        return None
    return {
        "headline": top["headline"],
        "one_line": _one_line_read(top),
        "why_it_matters": top.get("why_it_matters", ""),
        "tomorrow_morning": _tomorrow_action(top),
        "regime_line": _regime_line(stories),
        "confidence_pct": _confidence_pct(top),
        "tone": top.get("tone", "warning"),
        "signal": top.get("signal", "Signal mitigé"),
        "importance": top.get("importance", "Moyenne"),
        "companies": top.get("companies_mentioned", []),
        "source_count": top.get("source_count", 1),
        "article_count": top.get("article_count", 1),
        "narrative": top.get("narrative", ""),
    }


def rank_supporting_stories(stories: List[Dict], pulse: Optional[Dict], limit: int = 2) -> List[Dict]:
    """
    Return the next best stories after the pulse, for the 'what else matters'
    row. Filters out the hero story and general noise, ranks by score.
    """
    if not stories:
        return []
    hero_headline = pulse["headline"] if pulse else None
    candidates = [
        s for s in stories
        if s.get("headline") != hero_headline
        and s.get("narrative") != "General market noise"
    ]
    candidates.sort(key=_score_story, reverse=True)
    return candidates[:limit]
