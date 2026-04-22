from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict

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


st.set_page_config(
    page_title="MarketPulse — Explain the Market",
    page_icon="📈",
    layout="wide"
)

APP_CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(113,167,255,0.08), transparent 30%),
        radial-gradient(circle at 85% 10%, rgba(126,108,255,0.08), transparent 28%),
        linear-gradient(180deg, #040b17 0%, #071222 55%, #06101f 100%);
}

.block-container {
    max-width: 1380px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

.hero-box {
    padding: 2rem 2.2rem;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(20,40,110,0.88), rgba(40,25,110,0.88));
    border: 1px solid rgba(113,167,255,0.14);
    box-shadow: 0 20px 70px rgba(0,0,0,0.28);
    margin-bottom: 1.1rem;
}

.hero-kicker {
    color: #9ec1ff;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.55rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 850;
    color: white;
    margin: 0;
}

.hero-subtitle {
    color: #d4def4;
    font-size: 1.06rem;
    line-height: 1.85;
    margin-top: 0.75rem;
}

.small-note {
    color: #9fb0cb;
    line-height: 1.85;
    font-size: 0.95rem;
}

.section-title {
    color: white;
    font-size: 2.05rem;
    font-weight: 800;
    margin: 1.6rem 0 0.35rem 0;
}

.section-sub {
    color: #9fb0cb;
    margin-bottom: 1rem;
    line-height: 1.75;
}

.info-card {
    background: linear-gradient(180deg, rgba(12,24,48,0.92), rgba(10,19,36,0.96));
    border: 1px solid rgba(113,167,255,0.12);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 16px 50px rgba(0,0,0,0.22);
    height: 100%;
}

.info-label {
    color: #9fb0cb;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-size: 0.78rem;
    margin-bottom: 10px;
}

.info-value {
    color: white;
    font-size: 1.85rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.info-body {
    color: #d7e0f2;
    line-height: 1.75;
    font-size: 0.96rem;
}

.pulse-card {
    background: linear-gradient(180deg, rgba(10,22,42,0.96), rgba(7,16,30,0.98));
    border: 1px solid rgba(113,167,255,0.10);
    border-radius: 24px;
    padding: 20px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.22);
    margin-bottom: 16px;
}

.pulse-title {
    color: white;
    font-size: 1.22rem;
    font-weight: 760;
    margin-bottom: 8px;
}

.pulse-body {
    color: #d7e0f2;
    line-height: 1.75;
    font-size: 0.97rem;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    border: 1px solid transparent;
}

.badge-high {
    background: rgba(255, 193, 92, 0.14);
    color: #ffd082;
    border-color: rgba(255, 193, 92, 0.22);
}

.badge-medium {
    background: rgba(113,167,255,0.12);
    color: #b6d0ff;
    border-color: rgba(113,167,255,0.22);
}

.badge-low {
    background: rgba(160,170,190,0.12);
    color: #c8d0dd;
    border-color: rgba(160,170,190,0.18);
}

.badge-positive {
    background: rgba(70, 215, 164, 0.13);
    color: #6ff0bc;
    border-color: rgba(70, 215, 164, 0.24);
}

.badge-warning {
    background: rgba(255, 196, 102, 0.14);
    color: #ffd083;
    border-color: rgba(255, 196, 102, 0.24);
}

.badge-negative {
    background: rgba(255, 122, 122, 0.14);
    color: #ff9a9a;
    border-color: rgba(255, 122, 122, 0.24);
}

.signal-banner {
    background: linear-gradient(135deg, rgba(9,20,40,0.96), rgba(16,27,55,0.96));
    border: 1px solid rgba(113,167,255,0.12);
    border-radius: 22px;
    padding: 20px 22px;
    margin-bottom: 16px;
}

.signal-title {
    color: white;
    font-size: 1.2rem;
    font-weight: 760;
    margin-bottom: 8px;
}

.signal-text {
    color: #d7e0f2;
    line-height: 1.8;
    font-size: 0.98rem;
}

.story-box {
    background: linear-gradient(180deg, rgba(10,22,42,0.96), rgba(7,16,30,0.98));
    border: 1px solid rgba(113,167,255,0.10);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.22);
    margin-bottom: 16px;
}

.story-headline {
    color: white;
    font-size: 1.3rem;
    font-weight: 800;
    line-height: 1.45;
    margin-bottom: 12px;
}

.story-key {
    color: #edf2fb;
    line-height: 1.7;
    font-size: 1rem;
    margin-bottom: 10px;
}

.story-label {
    color: #9fb0cb;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    margin-bottom: 5px;
}

.story-mini {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.045);
    border-radius: 16px;
    padding: 14px;
    height: 100%;
}

.story-mini-text {
    color: #d7e0f2;
    line-height: 1.7;
    font-size: 0.95rem;
}

.watch-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.045);
    border-radius: 18px;
    padding: 15px 16px;
    margin-bottom: 10px;
    color: #dbe5f6;
    line-height: 1.7;
}

.divider-space {
    height: 12px;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_demo_rich_articles() -> List[Dict]:
    data_path = Path(__file__).parent / "data" / "demo_articles_rich.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tone_class(tone: str) -> str:
    return {
        "positive": "badge-positive",
        "warning": "badge-warning",
        "negative": "badge-negative",
    }.get(tone, "badge-medium")


def importance_class(level: str) -> str:
    return {
        "Élevée": "badge-high",
        "Moyenne": "badge-medium",
        "Faible": "badge-low",
    }.get(level, "badge-medium")


def signal_class(signal: str) -> str:
    if signal == "Signal fort":
        return "badge-positive"
    if signal == "Signal faible":
        return "badge-negative"
    return "badge-warning"


def global_signal_message(stories: List[Dict]) -> Dict[str, str]:
    if not stories:
        return {
            "title": "Pas assez de signal",
            "text": "Le flux actuel ne permet pas encore de dégager une lecture claire."
        }

    strong_count = sum(1 for s in stories if s["signal"] == "Signal fort")
    weak_count = sum(1 for s in stories if s["signal"] == "Signal faible")
    high_importance = sum(1 for s in stories if s["importance"] == "Élevée")

    if strong_count >= 2 or high_importance >= 3:
        return {
            "title": "Le marché envoie plusieurs signaux structurants",
            "text": "Plusieurs thèmes importants ressortent avec assez de répétition ou de cohérence pour mériter une attention sérieuse. Ce n’est pas juste du bruit quotidien."
        }

    if weak_count >= 2:
        return {
            "title": "Le flux est encore confus",
            "text": "Une partie importante des signaux reste contradictoire ou peu confirmée. Il faut éviter de surinterpréter la journée."
        }

    return {
        "title": "Le signal est présent mais encore mitigé",
        "text": "Certains thèmes ressortent clairement, mais il manque encore un peu de confirmation pour transformer la lecture en conviction forte."
    }


def top_takeaways(stories: List[Dict], limit: int = 3) -> List[str]:
    cleaned = [s for s in stories if s["narrative"] != "General market noise"]
    out = []
    for story in cleaned[:limit]:
        out.append(f"**{story['headline']}** — {story['plain_english']}")
    return out


def build_market_regime(stories: List[Dict]) -> Dict[str, str]:
    if not stories:
        return {
            "title": "Lecture insuffisante",
            "text": "Pas assez d’éléments solides pour décrire le régime de marché."
        }

    narratives = [s.get("narrative", "") for s in stories]
    tones = [s.get("tone", "warning") for s in stories]

    has_ai = "AI infrastructure" in narratives or "Cloud growth" in narratives
    has_macro = "Macro Market" in narratives or "Market uncertainty" in narratives
    has_reg = "Regulation pressure" in narratives
    pos = tones.count("positive")
    neg = tones.count("negative")
    warn = tones.count("warning")

    if has_ai and has_macro:
        return {
            "title": "Optimisme tech, prudence macro",
            "text": "La technologie garde un soutien structurel, mais l’environnement de taux ou d’incertitude limite encore une lecture franchement euphorique."
        }
    if has_reg and neg >= 1:
        return {
            "title": "Régime sous pression réglementaire",
            "text": "Le marché surveille des risques externes capables de peser sur la valorisation, même lorsque les fondamentaux restent corrects."
        }
    if pos >= 2 and neg == 0:
        return {
            "title": "Régime constructif",
            "text": "Les thèmes dominants soutiennent le ton du marché et les signaux négatifs restent contenus."
        }
    if neg >= 2:
        return {
            "title": "Régime prudent",
            "text": "Les nouvelles importantes penchent vers la retenue. Le marché n’a pas encore assez d’éléments pour basculer vers un vrai optimisme."
        }
    if warn >= 2:
        return {
            "title": "Régime mitigé",
            "text": "Le marché reçoit des signaux exploitables, mais pas encore assez cohérents pour justifier une conviction forte."
        }

    return {
        "title": "Régime en transition",
        "text": "Le flux donne une direction partielle, mais il faut encore des confirmations avant d’en faire une lecture durable."
    }


def build_do_i_care(stories: List[Dict]) -> Dict[str, str]:
    narratives = {s.get("narrative", "") for s in stories}

    investor = "Oui"
    investor_text = "Les thèmes du jour peuvent influencer la perception des secteurs et la hiérarchie entre actions fortes et plus fragiles."

    borrower = "À surveiller"
    borrower_text = "Le sujet ne change pas immédiatement ton crédit, mais des thèmes macro ou inflation peuvent garder le coût de l’emprunt sous pression."

    consumer = "Impact limité"
    consumer_text = "Peu d’effet immédiat sur la vie quotidienne, sauf si l’actualité touche l’énergie, les taux ou la consommation."

    if "Macro Market" in narratives or "Market uncertainty" in narratives:
        borrower = "Oui"
        borrower_text = "Le contexte de taux et d’inflation peut compter si tu envisages bientôt un crédit ou un refinancement."

    if "Energy pressure" in narratives:
        consumer = "Oui"
        consumer_text = "Les mouvements sur l’énergie peuvent finir par influencer le budget transport, chauffage ou certains prix."

    if "Regulation pressure" in narratives and investor == "Oui":
        investor_text = "Le risque réglementaire peut changer plus vite que prévu la thèse d’investissement sur certaines grandes valeurs."

    return {
        "investor": f"**Investisseur** — {investor}. {investor_text}",
        "borrower": f"**Emprunteur** — {borrower}. {borrower_text}",
        "consumer": f"**Consommateur** — {consumer}. {consumer_text}",
    }


def build_watch_summary(stories: List[Dict], limit: int = 4) -> List[str]:
    unique_items = []
    for story in stories:
        item = story.get("watch_next", "")
        if item and item not in unique_items:
            unique_items.append(item)
    return unique_items[:limit]


def build_dataset(mode: str):
    if mode == "Demo":
        raw_articles = load_demo_rich_articles()
        health = [
            {
                "source": "Curated rich demo",
                "status": "ok",
                "articles_kept": len(raw_articles),
                "url": "local curated dataset"
            }
        ]
    else:
        raw_articles, health = fetch_live_articles(limit_per_source=25)

    raw_df = articles_to_dataframe(raw_articles)
    dedup_df, dedup_stats = deduplicate_articles(raw_df)
    mapped_df = map_entities(dedup_df)
    scored_df = score_articles(mapped_df)
    final_df = assign_narratives(scored_df)

    inserted = store_articles(final_df)
    history_df = load_history(limit=800)

    return final_df, health, dedup_stats, inserted, history_df


def render_takeaway_card(text: str):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">À retenir</div>
            <div class="info-body">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_story_card(story: Dict):
    st.markdown(
        f"""
        <div class="badge-row">
            <span class="badge {importance_class(story["importance"])}">Importance : {story["importance"]}</span>
            <span class="badge {signal_class(story["signal"])}">{story["signal"]}</span>
            <span class="badge badge-medium">Confiance : {story["confidence"]}</span>
            <span class="badge {tone_class(story["tone"])}">Ton : {story["tone"]}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="story-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="story-headline">{story["headline"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="story-key">{story["plain_english"]}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="story-mini">', unsafe_allow_html=True)
        st.markdown('<div class="story-label">Pourquoi c’est important</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="story-mini-text">{story["why_it_matters"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="story-mini">', unsafe_allow_html=True)
        st.markdown('<div class="story-label">Qui devrait vraiment regarder</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="story-mini-text">{story["who_it_affects"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="story-mini">', unsafe_allow_html=True)
        st.markdown('<div class="story-label">À surveiller ensuite</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="story-mini-text">{story["watch_next"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        companies = story.get("companies_mentioned", [])
        companies_text = ", ".join(companies[:5]) if companies else "Contexte de marché plus large"
        meta = []
        if story.get("article_count") is not None:
            meta.append(f"{story['article_count']} article(s)")
        if story.get("source_count") is not None:
            meta.append(f"{story['source_count']} source(s)")
        meta_text = " · ".join(meta) if meta else ""

        st.markdown('<div class="story-mini">', unsafe_allow_html=True)
        st.markdown('<div class="story-label">Périmètre</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="story-mini-text">{companies_text}</div>', unsafe_allow_html=True)
        if meta_text:
            st.caption(meta_text)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_persona_card(name: str, panel: Dict):
    st.markdown(
        f"""
        <div class="badge-row">
            <span class="badge {tone_class(panel["tone"])}">{panel["tone"]}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.markdown(f"### {name}")
        st.markdown(f"**{panel['headline']}**")
        for detail in panel["details"][:2]:
            st.markdown(f"- {detail}")
        st.caption("À surveiller : " + panel["watch_next"])


init_db()

if "dataset_df" not in st.session_state:
    st.session_state.dataset_df = None
    st.session_state.health = []
    st.session_state.dedup_stats = {}
    st.session_state.history_df = pd.DataFrame()
    st.session_state.inserted_count = 0
    st.session_state.current_mode = "Demo"

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    mode = st.radio("Mode", ["Demo", "Live"], horizontal=True)
    run_clicked = st.button("Construire le briefing", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div class="small-note">
        <strong>Demo</strong> = version stable, riche et rapide pour comprendre le produit.<br><br>
        <strong>Live</strong> = lecture réelle de flux RSS, plus variable et parfois plus lente.<br><br>
        L’objectif de l’outil n’est pas d’empiler des headlines, mais d’expliquer ce qui compte vraiment.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-kicker">AI project · finance · useful interpretation</div>
        <div class="hero-title">📈 MarketPulse</div>
        <div class="hero-subtitle">
            Comprendre les actualités financières sans se noyer dans le jargon.
            L’application transforme un flux de news en réponses simples : ce qui s’est passé, pourquoi c’est important, qui est concerné et ce qu’il faut surveiller ensuite.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if run_clicked or st.session_state.dataset_df is None or mode != st.session_state.current_mode:
    with st.spinner("Construction du briefing de marché..."):
        df, health, dedup_stats, inserted_count, history_df = build_dataset(mode)
        st.session_state.dataset_df = df
        st.session_state.health = health
        st.session_state.dedup_stats = dedup_stats
        st.session_state.history_df = history_df
        st.session_state.inserted_count = inserted_count
        st.session_state.current_mode = mode

df = st.session_state.dataset_df
health = st.session_state.health
dedup_stats = st.session_state.dedup_stats
history_df = st.session_state.history_df
inserted_count = st.session_state.inserted_count

if df is None or df.empty:
    st.info("Choisis un mode puis clique sur « Construire le briefing ».")
    st.stop()

scope_options = ["Tout le marché"] + sorted(
    [c for c in df["primary_company"].dropna().unique().tolist() if c != "Unmapped"]
)
selected_scope = st.selectbox("Périmètre d’analyse", scope_options)

view_df = df.copy()
if selected_scope != "Tout le marché":
    view_df = view_df[view_df["primary_company"] == selected_scope].copy()

stories = build_story_cards(view_df, top_n=3)
persona_panels = build_persona_panels(stories)
signal_msg = global_signal_message(stories)
takeaways = top_takeaways(stories, limit=3)
regime = build_market_regime(stories)
do_i_care = build_do_i_care(stories)
watch_items = build_watch_summary(stories, limit=4)

mapped_rate = (view_df["primary_company"] != "Unmapped").mean() * 100 if not view_df.empty else 0.0
history_count = len(history_df) if history_df is not None else 0
source_count = view_df["source"].nunique() if not view_df.empty else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Briefing du jour</div>
            <div class="info-value">{len(stories)}</div>
            <div class="info-body">Nombre de sujets réellement utiles retenus après nettoyage, regroupement et interprétation.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Sources actives</div>
            <div class="info-value">{source_count}</div>
            <div class="info-body">Nombre de sources qui alimentent la lecture actuelle du marché.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Couverture utile</div>
            <div class="info-value">{mapped_rate:.1f}%</div>
            <div class="info-body">Part des articles rattachés à une entreprise, un thème ou un contexte exploitable.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Base locale</div>
            <div class="info-value">{history_count}</div>
            <div class="info-body">Articles déjà conservés dans l’historique local. Dernier ajout : {inserted_count}.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

top1, top2, top3 = st.columns(3)
with top1:
    st.markdown(
        f"""
        <div class="pulse-card">
            <div class="info-label">Régime de marché</div>
            <div class="pulse-title">{regime["title"]}</div>
            <div class="pulse-body">{regime["text"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with top2:
    st.markdown(
        f"""
        <div class="pulse-card">
            <div class="info-label">Do I care?</div>
            <div class="pulse-body">{do_i_care["investor"]}</div>
            <div class="pulse-body" style="margin-top:10px;">{do_i_care["borrower"]}</div>
            <div class="pulse-body" style="margin-top:10px;">{do_i_care["consumer"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with top3:
    watch_html = "".join([f"<div class='pulse-body' style='margin-top:8px;'>• {w}</div>" for w in watch_items]) or "<div class='pulse-body'>Aucun watchpoint fort identifié.</div>"
    st.markdown(
        f"""
        <div class="pulse-card">
            <div class="info-label">Watch next</div>
            <div class="pulse-title">Ce qu’il faut confirmer</div>
            {watch_html}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Ce qu’il faut retenir en 20 secondes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Avant d’entrer dans les détails, voici les 3 idées qui résument vraiment la journée.</div>',
    unsafe_allow_html=True
)

if takeaways:
    take_cols = st.columns(len(takeaways))
    for col, takeaway in zip(take_cols, takeaways):
        with col:
            render_takeaway_card(takeaway)
else:
    st.info("Pas encore assez d’éléments clairs pour générer les points clés.")

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="signal-banner">
        <div class="signal-title">{signal_msg["title"]}</div>
        <div class="signal-text">{signal_msg["text"]}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section-title">Ce qui compte aujourd’hui</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Trois sujets maximum. Pas de mur de cartes. Seulement ce qui mérite vraiment d’être compris.</div>',
    unsafe_allow_html=True
)

for story in stories:
    render_story_card(story)

st.markdown('<div class="section-title">Impact pour vous</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Même actualité, conséquences différentes selon le profil. Cette section sert à rendre le marché concret.</div>',
    unsafe_allow_html=True
)

persona_names = list(persona_panels.keys())
if persona_names:
    first_row = st.columns(3)
    for idx, name in enumerate(persona_names[:3]):
        with first_row[idx]:
            render_persona_card(name, persona_panels[name])

    if len(persona_names) > 3:
        second_row = st.columns(2)
        for idx, name in enumerate(persona_names[3:5]):
            with second_row[idx]:
                render_persona_card(name, persona_panels[name])

st.markdown('<div class="section-title">Ce qu’il faut surveiller ensuite</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">On ne prédit pas le futur. On montre simplement quelles prochaines données peuvent confirmer ou casser le signal du jour.</div>',
    unsafe_allow_html=True
)

for item in watch_items:
    st.markdown(f'<div class="watch-card">• {item}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Vérifier derrière l’interprétation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">La lecture grand public reste simple, mais on garde en bas les preuves que l’analyse repose sur une vraie pipeline.</div>',
    unsafe_allow_html=True
)

with st.expander("Santé des sources"):
    st.dataframe(pd.DataFrame(health), width="stretch")

with st.expander("Pipeline — qualité de traitement"):
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.metric("Articles bruts", dedup_stats.get("raw_count", 0))
    with qc2:
        st.metric("Doublons exacts retirés", dedup_stats.get("exact_duplicates_removed", 0))
    with qc3:
        st.metric("Doublons flous retirés", dedup_stats.get("fuzzy_duplicates_removed", 0))

with st.expander("Articles détaillés"):
    display_cols = [
        "source",
        "primary_company",
        "primary_narrative",
        "label",
        "confidence",
        "title_clean",
        "link",
    ]
    existing_cols = [c for c in display_cols if c in view_df.columns]
    st.dataframe(view_df[existing_cols], width="stretch", height=360)

with st.expander("Historique local"):
    if history_df is not None and not history_df.empty:
        hist_cols = [c for c in ["source", "primary_company", "primary_narrative", "published_dt", "title"] if c in history_df.columns]
        st.dataframe(history_df[hist_cols], width="stretch", height=360)
    else:
        st.info("Aucun historique local disponible pour le moment.")

st.caption("MarketPulse aide à comprendre l’actualité financière. Ce n’est pas un conseil d’investissement.")