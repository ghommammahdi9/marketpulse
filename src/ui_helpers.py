APP_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #040b17 0%, #071222 55%, #06101f 100%);
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

.hero-box {
    padding: 1.8rem 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15,40,90,0.85), rgba(35,20,90,0.85));
    border: 1px solid rgba(113,167,255,0.14);
    box-shadow: 0 20px 70px rgba(0,0,0,0.25);
    margin-bottom: 1.3rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
}

.hero-subtitle {
    color: #c7d6f0;
    margin-top: 0.6rem;
    line-height: 1.7;
}

.metric-card {
    padding: 18px;
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    min-height: 120px;
}

.metric-label {
    color: #9fb0cb;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.metric-sub {
    color: #cad5ea;
    line-height: 1.6;
    font-size: 0.92rem;
}

.metric-positive .metric-value { color: #46d7a4; }
.metric-negative .metric-value { color: #ff7a7a; }
.metric-warning .metric-value { color: #ffc266; }

.section-kicker {
    color: #71a7ff;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}

.small-note {
    color: #9fb0cb;
    font-size: 0.92rem;
}
</style>
"""

def metric_card(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    tone_class = {
        "positive": "metric-positive",
        "negative": "metric-negative",
        "warning": "metric-warning",
        "neutral": ""
    }.get(tone, "")

    return f'''
    <div class="metric-card {tone_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    '''

def mood_tone(score: float) -> str:
    if score >= 20:
        return "positive"
    if score <= -20:
        return "negative"
    return "warning"

def divergence_tone(score: float) -> str:
    if score >= 45:
        return "warning"
    return "neutral"