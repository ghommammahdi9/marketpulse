from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fetch_news import fetch_live_articles
from clean_news import articles_to_dataframe
from deduplicate import deduplicate_articles
from entity_mapper import map_entities
from sentiment import score_articles
from narratives import assign_narratives
from interpreter import build_story_cards
from personas import build_persona_panels
from storage import init_db, store_articles, load_history
from pulse import compute_pulse, rank_supporting_stories

# Safe LLM imports so the app does not break if the LLM layer is missing or misconfigured.
try:
    from llm_layer import rerank_evidence_df
except Exception:
    def rerank_evidence_df(story: Dict, evidence_df: pd.DataFrame, limit: int = 2) -> pd.DataFrame:
        if evidence_df is None or evidence_df.empty:
            return pd.DataFrame()
        return evidence_df.head(limit).copy()

try:
    from llm_writer import rewrite_pulse_with_gemini, rewrite_story_with_gemini
except Exception:
    def rewrite_pulse_with_gemini(pulse: Dict, evidence_df: pd.DataFrame) -> Dict:
        return pulse

    def rewrite_story_with_gemini(story: Dict, evidence_df: pd.DataFrame) -> Dict:
        return story


st.set_page_config(
    page_title="MarketPulse — Live Market Brief",
    page_icon="📈",
    layout="wide",
)

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:wght@600;700;800&display=swap');

:root {
    --bg: #07111f;
    --bg-soft: #0d1829;
    --panel: rgba(12, 22, 38, 0.92);
    --panel-2: rgba(16, 28, 46, 0.96);
    --border: rgba(120, 146, 190, 0.16);
    --text: #edf2fb;
    --muted: #9caecb;
    --accent: #8cb4ff;
    --positive: #76d8ac;
    --warning: #f0c46a;
    --danger: #f08a8a;
}

html, body, [class*="css"] {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 18%, rgba(140,180,255,0.06), transparent 28%),
        radial-gradient(circle at 86% 8%, rgba(140,180,255,0.04), transparent 22%),
        linear-gradient(180deg, #060d18 0%, #091322 58%, #07111f 100%);
    color: var(--text);
}

.block-container {
    max-width: 1260px;
    padding-top: 1.0rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    font-family: "Source Serif 4", Georgia, serif !important;
    letter-spacing: -0.02em;
    color: var(--text);
}

h1 {
    font-weight: 800 !important;
    line-height: 1.02 !important;
}

h2 {
    font-weight: 700 !important;
    line-height: 1.08 !important;
}

p, li, div, span, label {
    color: var(--text);
}

a {
    color: var(--accent) !important;
    text-decoration: none !important;
}

a:hover {
    text-decoration: underline !important;
}

.live-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    border: 1px solid rgba(240,138,138,0.22);
    color: #ffc1c1;
    background: rgba(240,138,138,0.08);
    margin: 0.35rem 0 1rem 0;
}

.identity-lede {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.75;
    max-width: 780px;
    margin-top: 0.3rem;
}

.identity-wrap {
    margin-bottom: 1.35rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 10px 12px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 700 !important;
}

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
    gap: 0.35rem;
}

div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] {
    border-radius: 18px;
}

div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] > div {
    border-radius: 18px;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
}

button[kind="primary"] {
    border-radius: 14px !important;
    font-weight: 600 !important;
}

div[data-baseweb="tab-list"] {
    gap: 8px;
}

button[role="tab"] {
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.03) !important;
    color: var(--muted) !important;
    padding: 0.45rem 0.9rem !important;
}

button[role="tab"][aria-selected="true"] {
    background: rgba(140,180,255,0.10) !important;
    color: var(--text) !important;
    border-color: rgba(140,180,255,0.22) !important;
}

section[data-testid="stSidebar"] {
    background: rgba(7, 15, 26, 0.82);
    border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

.disclaimer {
    color: #7b8aa6;
    font-size: 0.82rem;
    margin-top: 2rem;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


def fmt_dt(value) -> str:
    if pd.isna(value):
        return "date inconnue"
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return str(value)
        return ts.strftime("%d/%m %H:%M UTC")
    except Exception:
        return str(value)


def safe_merge(base: Dict | None, rewritten: Dict | None) -> Dict | None:
    if not base:
        return rewritten
    if not rewritten:
        return base
    merged = dict(base)
    merged.update({k: v for k, v in rewritten.items() if v not in [None, ""]})
    return merged


def render_identity_block(raw_count: int, article_count: int, source_count: int, refreshed_at: str):
    st.markdown('<div class="identity-wrap">', unsafe_allow_html=True)
    st.caption("LIVE MARKET INTERPRETATION")
    st.markdown("# MarketPulse")
    st.markdown(
        '<div class="identity-lede">Lecture claire d’un flux réel d’actualités financières : '
        'ce qui bouge, pourquoi cela compte, et ce qu’il faut vérifier ensuite.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<span class="live-badge">🔴 Live · {raw_count} article(s) bruts extraits puis analysés</span>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Articles retenus", article_count)
    with m2:
        st.metric("Sources actives", source_count)
    with m3:
        st.metric("Dernière mise à jour", refreshed_at)

    st.markdown("</div>", unsafe_allow_html=True)


def render_loading_shell(title: str = "Construction du briefing"):
    box = st.empty()
    with box.container():
        st.markdown(f"## {title}")
        status_text = st.empty()
        progress = st.progress(0)
    return box, status_text, progress


def build_live_dataset():
    raw_articles, health = fetch_live_articles(limit_per_source=25)

    raw_df = articles_to_dataframe(raw_articles)
    dedup_df, dedup_stats = deduplicate_articles(raw_df)
    mapped_df = map_entities(dedup_df)
    scored_df = score_articles(mapped_df)
    final_df = assign_narratives(scored_df)

    inserted = store_articles(final_df)
    history_df = load_history(limit=800)

    return final_df, health, dedup_stats, inserted, history_df, raw_articles


def build_action_board(pulse: Dict | None, support_stories: List[Dict]) -> List[Dict]:
    if not pulse:
        return [
            {
                "kicker": "Maintenant",
                "title": "Le fait du jour",
                "body": "Le flux actuel ne permet pas encore de dégager un signal dominant suffisamment propre.",
            },
            {
                "kicker": "Lecture",
                "title": "Pourquoi ça compte",
                "body": "Une partie du marché reste encore trop bruitée ou trop peu confirmée.",
            },
            {
                "kicker": "À valider",
                "title": "Le prochain test",
                "body": "Attendre davantage de publications ou de confirmations croisées avant de conclure.",
            },
        ]

    watch_body = pulse.get("tomorrow_morning", "")
    if support_stories:
        watch_body = support_stories[0].get("watch_next", watch_body)

    return [
        {
            "kicker": "Maintenant",
            "title": "Ce qui bouge",
            "body": pulse.get("one_line", ""),
        },
        {
            "kicker": "Lecture",
            "title": "Pourquoi c’est important",
            "body": pulse.get("why_it_matters", ""),
        },
        {
            "kicker": "À valider",
            "title": "Ce qu’il faut suivre",
            "body": watch_body,
        },
    ]


def render_board(cards: List[Dict]):
    st.markdown("## Brief du jour")
    st.caption("Les trois réponses les plus utiles à comprendre immédiatement.")

    cols = st.columns(3)
    for col, card in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.caption(card["kicker"].upper())
                st.markdown(f"### {card['title']}")
                st.write(card["body"])


NARRATIVE_HINTS = {
    "AI infrastructure": [
        "ai", "artificial intelligence", "gpu", "chip", "chips", "semiconductor",
        "server", "servers", "cloud", "aws", "azure", "data center", "datacenter",
        "nvidia", "openai", "inference", "training"
    ],
    "Earnings beat": [
        "earnings", "results", "quarterly", "q1", "q2", "q3", "q4",
        "profit", "profits", "revenue", "sales", "eps", "guidance", "outlook"
    ],
    "Demand recovery": [
        "demand", "orders", "sales", "recovery", "pickup", "consumer", "spending"
    ],
    "Regulation pressure": [
        "regulator", "regulation", "antitrust", "probe", "investigation",
        "fine", "ban", "compliance", "lawsuit"
    ],
    "Margin improvement": [
        "margin", "cost cuts", "efficiency", "profitability", "operating leverage"
    ],
    "Cloud growth": [
        "cloud", "aws", "azure", "google cloud", "datacenter", "enterprise software"
    ],
    "Advertising slowdown": [
        "advertising", "ad market", "marketing spend", "ad revenue"
    ],
    "Energy pressure": [
        "oil", "energy", "fuel", "gas", "crude", "power prices"
    ],
    "Market uncertainty": [
        "uncertainty", "volatility", "mixed", "unclear", "caution", "wait-and-see"
    ],
}

NEGATIVE_EARNINGS_HINTS = [
    "miss", "misses", "cut", "cuts", "warning", "slump", "falls short",
    "weak", "drop", "decline", "lawsuit", "probe"
]


def evidence_relevance(row: pd.Series, story: Dict) -> int:
    title = str(row.get("title_clean", "")).lower()
    primary_company = str(row.get("primary_company", "")).lower()
    narrative = str(story.get("narrative", ""))
    companies = [str(c).lower() for c in story.get("companies_mentioned", []) if c]

    score = 0

    for company in companies:
        if company and company in title:
            score += 5

    if primary_company and primary_company != "unmapped" and primary_company in companies:
        score += 3

    for hint in NARRATIVE_HINTS.get(narrative, []):
        if hint in title:
            score += 2

    if narrative == "Earnings beat":
        for bad in NEGATIVE_EARNINGS_HINTS:
            if bad in title:
                score -= 3

    if narrative == "AI infrastructure":
        if not any(h in title for h in NARRATIVE_HINTS["AI infrastructure"]):
            score -= 2

    if narrative == "Regulation pressure":
        if not any(h in title for h in NARRATIVE_HINTS["Regulation pressure"]):
            score -= 2

    return score


def story_evidence(df: pd.DataFrame, story: Dict, limit: int = 3) -> pd.DataFrame:
    working = df.copy()

    narrative = story.get("narrative")
    if narrative and "primary_narrative" in working.columns:
        narrowed = working[working["primary_narrative"] == narrative]
        if len(narrowed) >= 1:
            working = narrowed

    companies = [c for c in story.get("companies_mentioned", []) if c]
    if companies and "primary_company" in working.columns:
        narrowed = working[working["primary_company"].isin(companies)]
        if len(narrowed) >= 1:
            working = narrowed

    if working.empty:
        return working

    working = working.copy()
    working["_relevance"] = working.apply(lambda row: evidence_relevance(row, story), axis=1)

    if (working["_relevance"] > 0).any():
        working = working[working["_relevance"] > 0]

    if "published_dt" in working.columns:
        working = working.sort_values(
            by=["_relevance", "published_dt"],
            ascending=[False, False],
            na_position="last",
        )
    else:
        working = working.sort_values(by="_relevance", ascending=False)

    cols = [c for c in ["title_clean", "source", "link", "published_dt", "primary_company"] if c in working.columns]
    return working[cols].head(limit).copy()


def render_evidence_list(evidence_df: pd.DataFrame):
    if evidence_df.empty:
        st.caption("Pas de titres d’appui affichables pour ce sujet.")
        return

    with st.expander(f"Voir les preuves ({len(evidence_df)})", expanded=False):
        for idx, (_, row) in enumerate(evidence_df.iterrows()):
            title = str(row.get("title_clean", "Article"))
            source = str(row.get("source", "Source inconnue"))
            dt = fmt_dt(row.get("published_dt"))
            link = str(row.get("link", "")).strip()

            if link and link != "nan":
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")

            st.caption(f"{source} · {dt}")

            if idx < len(evidence_df) - 1:
                st.divider()


def display_story_title(story: Dict) -> str:
    narrative = story.get("narrative", "")
    companies = story.get("companies_mentioned", []) or []
    lead = ", ".join(companies[:2]) if companies else ""

    if narrative == "AI infrastructure":
        return f"L’infrastructure IA reste au centre du flux{f' — {lead}' if lead else ''}"
    if narrative == "Earnings beat":
        return f"Le thème des résultats domine la séance{f' — {lead}' if lead else ''}"
    if narrative == "Demand recovery":
        return f"Des signes de reprise de la demande ressortent{f' — {lead}' if lead else ''}"
    if narrative == "Regulation pressure":
        return f"La pression réglementaire refait surface{f' — {lead}' if lead else ''}"
    if narrative == "Margin improvement":
        return f"Les marges montrent un mieux{f' — {lead}' if lead else ''}"
    if narrative == "Cloud growth":
        return f"Le cloud continue de soutenir la lecture du marché{f' — {lead}' if lead else ''}"
    if narrative == "Advertising slowdown":
        return f"Le signal publicitaire ralentit{f' — {lead}' if lead else ''}"
    if narrative == "Energy pressure":
        return f"La pression énergie reste présente{f' — {lead}' if lead else ''}"
    if narrative == "Market uncertainty":
        return "Le marché reste hésitant"

    return story.get("headline", "Sujet de marché")


def secondary_signal_label(story: Dict, idx: int) -> str:
    signal = str(story.get("signal", "")).strip()
    importance = str(story.get("importance", "")).strip()

    if idx == 1:
        base = "Signal secondaire"
    else:
        base = "Signal complémentaire"

    if importance:
        return f"{base} · {importance}"
    if signal:
        return f"{base} · {signal}"
    return base


def story_scope_line(story: Dict) -> str:
    companies = story.get("companies_mentioned", []) or []
    article_count = int(story.get("article_count", 0))
    source_count = int(story.get("source_count", 0))

    scope = ", ".join(companies[:3]) if companies else "Contexte de marché"
    return f"{scope} · {article_count} article(s) · {source_count} source(s)"


def render_story_card(story: Dict, evidence_df: pd.DataFrame, idx: int = 1):
    with st.container(border=True):
        st.caption(secondary_signal_label(story, idx))
        st.markdown(f"### {story.get('headline', display_story_title(story))}")
        st.write(story.get("plain_english", ""))

        st.caption(story_scope_line(story))

        left, right = st.columns(2)

        with left:
            st.markdown("**Pourquoi c’est important**")
            st.write(story.get("why_it_matters", ""))

            st.markdown("**Qui devrait regarder**")
            st.write(story.get("who_it_affects", ""))

        with right:
            st.markdown("**À surveiller ensuite**")
            st.write(story.get("watch_next", ""))

            tone = story.get("tone", "")
            signal = story.get("signal", "")
            if tone or signal:
                st.markdown("**Lecture**")
                bits = [b for b in [signal, tone] if b]
                st.write(" · ".join(bits))

        render_evidence_list(evidence_df)


PERSONA_LABELS = {
    "Saver": "Épargnant",
    "Borrower": "Emprunteur",
    "Investor": "Investisseur",
    "Consumer": "Consommateur",
    "Tech worker": "Salarié tech",
}

TONE_LABELS = {
    "neutral": "neutre",
    "warning": "prudence",
    "positive": "positif",
    "negative": "risque",
}


def clean_persona_details(details: List[str]) -> List[str]:
    out = []
    seen = set()

    for detail in details:
        text = str(detail).strip()
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)

    return out


def persona_priority_label(panel: Dict) -> str:
    tone = str(panel.get("tone", "neutral")).lower()
    if tone == "positive":
        return "À suivre de près"
    if tone == "warning":
        return "À regarder avec prudence"
    if tone == "negative":
        return "Point de risque"
    return "À surveiller"


def persona_takeaway(panel: Dict) -> str:
    details = panel.get("details", []) or []
    if details:
        return str(details[0])
    headline = str(panel.get("headline", "")).strip()
    if headline:
        return headline
    return "Pas de lecture spécifique pour ce profil."


def render_quick_persona_strip(persona_panels: Dict):
    if not persona_panels:
        return

    filtered = {}
    for name, panel in persona_panels.items():
        headline = str(panel.get("headline", "")).strip().lower()
        details = clean_persona_details(panel.get("details", []))

        if "pas de signal majeur" in headline:
            continue
        if not details and not headline:
            continue

        filtered[name] = {**panel, "details": details}

    if not filtered:
        return

    st.markdown("## Ce que ça change selon le profil")
    st.caption("Même marché, conséquences différentes selon la personne qui lit le signal.")

    raw_names = list(filtered.keys())[:4]
    visible_names = [PERSONA_LABELS.get(name, name) for name in raw_names]
    tabs = st.tabs(visible_names)

    for tab, raw_name, visible_name in zip(tabs, raw_names, visible_names):
        panel = filtered[raw_name]
        details = panel.get("details", [])[:3]

        with tab:
            left, right = st.columns([1.8, 1])

            with left:
                st.markdown(f"### {visible_name}")
                st.caption(persona_priority_label(panel))
                st.write(panel.get("headline", ""))

                st.markdown("**En pratique aujourd’hui**")
                st.write(persona_takeaway(panel))

                if len(details) > 1:
                    st.markdown("**Points utiles**")
                    for detail in details[1:3]:
                        st.write(f"• {detail}")

            with right:
                with st.container(border=True):
                    st.caption("Ton")
                    st.write(TONE_LABELS.get(panel.get("tone", "neutral"), panel.get("tone", "neutral")))

                    st.caption("À surveiller")
                    st.write(panel.get("watch_next", "Rien de spécifique pour le moment."))

                    st.caption("Lecture produit")
                    st.write("Impact traduit pour un profil concret, pas seulement pour un analyste.")


def pulse_kicker_text(pulse: Dict) -> str:
    narrative = str(pulse.get("narrative", "")).strip()
    conf = int(pulse.get("confidence_pct", 0))

    narrative_map = {
        "AI infrastructure": "INFRASTRUCTURE IA",
        "Earnings beat": "RÉSULTATS",
        "Demand recovery": "DEMANDE",
        "Regulation pressure": "RÉGULATION",
        "Margin improvement": "MARGES",
        "Cloud growth": "CLOUD",
        "Advertising slowdown": "PUBLICITÉ",
        "Energy pressure": "ÉNERGIE",
        "Market uncertainty": "MARCHÉ",
    }

    conviction = "HIGH CONVICTION" if conf >= 80 else "MEDIUM CONVICTION" if conf >= 60 else "LOW CONVICTION"
    return f"{narrative_map.get(narrative, 'LECTURE MARCHÉ')} · {conviction}"


def confidence_commentary(conf: int) -> str:
    if conf >= 80:
        return "Le signal est bien soutenu par plusieurs articles cohérents."
    if conf >= 60:
        return "Le signal est crédible, mais demande encore un peu de confirmation."
    return "Le signal existe, mais il reste encore fragile."


def render_primary_evidence(evidence_df: pd.DataFrame, limit: int = 2):
    if evidence_df.empty:
        return

    st.markdown("**Titres qui soutiennent ce signal**")
    for idx, (_, row) in enumerate(evidence_df.head(limit).iterrows(), start=1):
        title = str(row.get("title_clean", "Article"))
        source = str(row.get("source", "Source inconnue"))
        dt = fmt_dt(row.get("published_dt"))
        link = str(row.get("link", "")).strip()

        if link and link != "nan":
            st.markdown(f"**{idx}. [{title}]({link})**")
        else:
            st.markdown(f"**{idx}. {title}**")

        st.caption(f"{source} · {dt}")


def render_pulse_hero(
    pulse: Dict,
    article_count: int,
    source_count: int,
    refreshed_at: str,
    pulse_evidence_df: pd.DataFrame,
):
    companies_str = ", ".join(pulse.get("companies", [])[:4]) if pulse.get("companies") else "Contexte de marché large"
    conf = int(pulse.get("confidence_pct", 0))
    kicker = pulse_kicker_text(pulse)

    st.markdown("## Note de marché")
    st.caption("Le signal principal extrait du flux réel.")

    left, right = st.columns([1.9, 1])

    with left:
        with st.container(border=True):
            st.caption(kicker)
            st.markdown(f"# {pulse.get('headline', display_story_title(pulse))}")
            st.markdown(f"### {pulse.get('one_line', '')}")
            st.caption(
                f"Basé sur {source_count} source(s) · {article_count} article(s) retenu(s) · mise à jour {refreshed_at}"
            )

            render_primary_evidence(pulse_evidence_df, limit=2)

            st.divider()

            info1, info2, info3 = st.columns(3)
            with info1:
                st.caption("Pourquoi ça compte")
                st.write(pulse.get("why_it_matters", ""))
            with info2:
                st.caption("Demain matin")
                st.write(pulse.get("tomorrow_morning", ""))
            with info3:
                st.caption("Régime de marché")
                st.write(pulse.get("regime_line", ""))

    with right:
        with st.container(border=True):
            st.caption("Niveau de confiance")
            st.markdown(f"## {conf}%")
            st.progress(conf / 100)
            st.write(confidence_commentary(conf))

            st.divider()

            st.caption("Périmètre")
            st.write(companies_str)

            st.caption("Dernière mise à jour")
            st.write(refreshed_at)

            st.caption("Lecture")
            st.write("Signal principal sélectionné à partir des articles les plus cohérents du flux.")


def build_watch_summary(stories: List[Dict], limit: int = 4) -> List[str]:
    unique = []
    for story in stories:
        item = story.get("watch_next", "")
        if item and item not in unique:
            unique.append(item)
    return unique[:limit]


def select_distinct_support_stories(stories: List[Dict], pulse: Dict | None, limit: int = 2) -> List[Dict]:
    if not stories:
        return []

    pulse_narrative = pulse.get("narrative") if pulse else None
    pulse_companies = set(pulse.get("companies", [])) if pulse else set()

    selected = []

    for story in stories:
        narrative = story.get("narrative")
        companies = set(story.get("companies_mentioned", []))

        if pulse_narrative and narrative == pulse_narrative:
            continue

        if pulse_companies and companies:
            overlap = len(companies & pulse_companies)
            if overlap >= 1 and len(companies) <= 2:
                continue

        duplicate = False
        for existing in selected:
            ex_narr = existing.get("narrative")
            ex_companies = set(existing.get("companies_mentioned", []))

            if narrative == ex_narr:
                duplicate = True
                break

            if companies and ex_companies and len(companies & ex_companies) >= 1:
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(story)
        if len(selected) >= limit:
            break

    return selected


def render_methodology_section():
    st.markdown("## Comment MarketPulse construit sa lecture")
    st.caption("Le produit ne résume pas des titres au hasard. Il transforme un flux brut en briefing exploitable.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        with st.container(border=True):
            st.caption("1")
            st.markdown("**Sources**")
            st.write("Flux RSS financiers récupérés en direct.")

    with c2:
        with st.container(border=True):
            st.caption("2")
            st.markdown("**Nettoyage**")
            st.write("Normalisation des titres, structuration et filtrage du bruit.")

    with c3:
        with st.container(border=True):
            st.caption("3")
            st.markdown("**Déduplication**")
            st.write("Suppression des doublons exacts et rapprochement des doublons flous.")

    with c4:
        with st.container(border=True):
            st.caption("4")
            st.markdown("**Mapping**")
            st.write("Rattachement à des entreprises et à des narratifs de marché.")

    with c5:
        with st.container(border=True):
            st.caption("5")
            st.markdown("**Vérification**")
            st.write("Reranking des preuves avec Gemini pour garder les titres les plus pertinents.")

    with c6:
        with st.container(border=True):
            st.caption("6")
            st.markdown("**Brief final**")
            st.write("Production d’un signal principal, de signaux secondaires et d’une lecture par profil.")


init_db()

for key, default in [
    ("dataset_df", None),
    ("health", []),
    ("dedup_stats", {}),
    ("history_df", pd.DataFrame()),
    ("inserted_count", 0),
    ("raw_articles", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.markdown("## Marché")
    run_clicked = st.button("Actualiser le marché", use_container_width=True)

    st.markdown("---")
    st.caption("Live only")
    st.write("Flux réel. Pas de dataset démo.")


if run_clicked or st.session_state.dataset_df is None:
    loading_box, loading_status, loading_progress = render_loading_shell("Chargement des données live")

    loading_status.markdown("**1/2** · Extraction du flux live…")
    loading_progress.progress(20)
    df, health, dedup_stats, inserted_count, history_df, raw_articles = build_live_dataset()

    loading_status.markdown("**2/2** · Structuration et préparation des données…")
    loading_progress.progress(100)

    st.session_state.dataset_df = df
    st.session_state.health = health
    st.session_state.dedup_stats = dedup_stats
    st.session_state.history_df = history_df
    st.session_state.inserted_count = inserted_count
    st.session_state.raw_articles = raw_articles

    loading_box.empty()

df = st.session_state.dataset_df
health = st.session_state.health
dedup_stats = st.session_state.dedup_stats
history_df = st.session_state.history_df
inserted_count = st.session_state.inserted_count
raw_articles = st.session_state.raw_articles

if df is None or df.empty:
    st.info("Clique sur « Actualiser le marché » pour lancer l’extraction live.")
    st.stop()

# Global metrics for the full market, before any filter
global_article_count = len(df)
global_source_count = df["source"].nunique() if "source" in df.columns and not df.empty else 0
global_refreshed_at = "date inconnue"
if "published_dt" in df.columns and not df.empty:
    global_refreshed_at = fmt_dt(df["published_dt"].max())

render_identity_block(
    raw_count=dedup_stats.get("raw_count", len(raw_articles)),
    article_count=global_article_count,
    source_count=global_source_count,
    refreshed_at=global_refreshed_at,
)

BAD_SCOPE_VALUES = {
    "Commodities",
    "Technology",
    "Energy",
    "Banking",
    "Consumer / Cloud",
    "Semiconductors",
    "Automotive",
    "Contexte de marché",
    "Contexte de marché large",
}

valid_scope_df = df.copy()

if "primary_company" in valid_scope_df.columns:
    valid_scope_df = valid_scope_df[
        valid_scope_df["primary_company"].notna()
        & (valid_scope_df["primary_company"] != "Unmapped")
        & (valid_scope_df["primary_company"] != "")
    ].copy()

scope_values = [
    c for c in valid_scope_df["primary_company"].unique().tolist()
    if c not in BAD_SCOPE_VALUES
]

scope_options = ["Tout le marché"] + sorted(scope_values)
selected_scope = st.selectbox("Périmètre d’analyse", scope_options)

mapped_companies_count = len(scope_options) - 1
unmapped_count = int((df["primary_company"] == "Unmapped").sum()) if "primary_company" in df.columns else 0

st.caption(
    f"{mapped_companies_count} entreprise(s) détectée(s) dans ce refresh live · "
    f"{unmapped_count} article(s) non rattaché(s) à une entreprise"
)

view_df = df.copy()
if selected_scope != "Tout le marché":
    view_df = view_df[view_df["primary_company"] == selected_scope].copy()

if selected_scope == "Tout le marché":
    st.caption("Lecture du flux global.")
else:
    st.caption(f"Lecture filtrée sur : {selected_scope}")

if view_df.empty:
    st.warning("Aucun article exploitable sur ce périmètre pour ce refresh.")
    st.stop()

analysis_loader_box, analysis_status, analysis_progress = render_loading_shell("Construction du briefing")

analysis_status.markdown("**1/5** · Construction des signaux marché…")
analysis_progress.progress(15)
stories = build_story_cards(view_df, top_n=6)
pulse = compute_pulse(stories)

analysis_status.markdown("**2/5** · Sélection des signaux secondaires…")
analysis_progress.progress(35)
raw_support_stories = rank_supporting_stories(stories, pulse, limit=6) if pulse else stories[:6]
support_stories = select_distinct_support_stories(raw_support_stories, pulse, limit=2)

analysis_status.markdown("**3/5** · Traduction par profil…")
analysis_progress.progress(50)
persona_panels = build_persona_panels(stories[:3]) if stories else {}
watch_items = build_watch_summary(support_stories if support_stories else stories, limit=4)

article_count = len(view_df)
source_count = view_df["source"].nunique() if "source" in view_df.columns and not view_df.empty else 0
refreshed_at = "date inconnue"
if "published_dt" in view_df.columns and not view_df.empty:
    refreshed_at = fmt_dt(view_df["published_dt"].max())

analysis_status.markdown("**4/5** · Vérification et réécriture du signal principal…")
analysis_progress.progress(70)

pulse_evidence_df = pd.DataFrame()
pulse_display = pulse

if pulse:
    pulse_evidence_df = story_evidence(view_df, pulse, limit=6)
    pulse_evidence_df = rerank_evidence_df(pulse, pulse_evidence_df, limit=2)
    pulse_display = safe_merge(pulse, rewrite_pulse_with_gemini(pulse, pulse_evidence_df))

board_cards = build_action_board(pulse_display, support_stories)
render_board(board_cards)

if pulse_display:
    render_pulse_hero(
        pulse_display,
        article_count=article_count,
        source_count=source_count,
        refreshed_at=refreshed_at,
        pulse_evidence_df=pulse_evidence_df,
    )
else:
    st.warning("Le flux actuel ne permet pas encore de dégager un signal principal assez propre.")

render_quick_persona_strip(persona_panels)

analysis_status.markdown("**5/5** · Finalisation des signaux secondaires…")
analysis_progress.progress(88)

if support_stories:
    st.markdown("## Deux signaux à garder à l’œil")
    st.caption("Pas un mur de cartes. Juste les deux lectures secondaires qui méritent encore ton attention.")

    secondary_cols = st.columns(2)
    for idx, (col, story) in enumerate(zip(secondary_cols, support_stories), start=1):
        with col:
            evidence_df = story_evidence(view_df, story, limit=6)
            evidence_df = rerank_evidence_df(story, evidence_df, limit=3)

            story_display = story
            if not evidence_df.empty:
                story_display = safe_merge(story, rewrite_story_with_gemini(story, evidence_df))

            render_story_card(story_display, evidence_df, idx=idx)
else:
    st.info("Pas assez de sujets secondaires solides sur ce périmètre pour le moment.")

if watch_items:
    st.markdown("## À surveiller ensuite")
    st.caption("Les prochaines informations qui peuvent confirmer ou casser le signal du jour.")
    with st.container(border=True):
        for item in watch_items:
            st.write(f"• {item}")

analysis_progress.progress(100)
analysis_loader_box.empty()

render_methodology_section()

st.markdown("## Vérifier derrière la lecture")
st.caption("La lecture visible reste simple. Cette section montre la qualité de la pipeline et les données qui soutiennent le briefing.")

proof_tabs = st.tabs([
    "Qualité pipeline",
    "Articles utilisés",
    "Historique & sources",
])

with proof_tabs[0]:
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Articles bruts", dedup_stats.get("raw_count", 0))
    p2.metric("Doublons exacts retirés", dedup_stats.get("exact_duplicates_removed", 0))
    p3.metric("Doublons flous retirés", dedup_stats.get("fuzzy_duplicates_removed", 0))
    p4.metric(
        "Historique local",
        len(history_df) if history_df is not None else 0,
        delta=f"+{inserted_count}" if inserted_count else None,
    )

    st.markdown("### Santé des sources")
    st.caption("Vue de contrôle sur les flux utilisés pendant ce refresh.")
    st.dataframe(pd.DataFrame(health), use_container_width=True, height=260)

with proof_tabs[1]:
    st.markdown("### Articles utilisés dans ce périmètre")
    st.caption("Ce tableau montre les articles réellement retenus pour la vue courante.")

    display_cols = [
        "source",
        "published_dt",
        "primary_company",
        "primary_narrative",
        "label",
        "confidence",
        "title_clean",
        "link",
    ]
    existing_cols = [c for c in display_cols if c in view_df.columns]
    display_df = view_df[existing_cols].copy()

    if "published_dt" in display_df.columns:
        display_df["published_dt"] = display_df["published_dt"].astype(str)

    st.dataframe(display_df, use_container_width=True, height=420)

with proof_tabs[2]:
    st.markdown("### Historique local")
    st.caption("Accumulation des articles déjà stockés localement par la pipeline.")

    if history_df is not None and not history_df.empty:
        hist_cols = [
            c for c in ["source", "primary_company", "primary_narrative", "published_dt", "title"]
            if c in history_df.columns
        ]
        hist_df = history_df[hist_cols].copy()

        if "published_dt" in hist_df.columns:
            hist_df["published_dt"] = hist_df["published_dt"].astype(str)

        st.dataframe(hist_df, use_container_width=True, height=360)
    else:
        st.info("Aucun historique local disponible pour le moment.")

    st.markdown("### Note de lecture")
    st.caption("Les tableaux ci-dessus servent de preuve technique. Le cœur du produit reste la lecture synthétique affichée plus haut.")

st.markdown(
    '<div class="disclaimer">MarketPulse aide à comprendre l’actualité financière. Ce n’est pas un conseil d’investissement.</div>',
    unsafe_allow_html=True,
)