from __future__ import annotations

import os
import base64
import time
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from wordcloud import WordCloud

from src.data_utils import load_or_create_dataset
from src.nlp_utils import preprocess_text


BASE_DIR = Path(__file__).parent
MPL_CONFIG_DIR = BASE_DIR / "outputs/.matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt

MODEL_PATH = BASE_DIR / "models/model_bundle.joblib"
FONT_DIR = BASE_DIR / "assets/fonts"
BERT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_hero_font_css() -> str:
    for font_name, font_format in (
        ("Ralmone.ttf", "truetype"),
        ("Ralmone.otf", "opentype"),
        ("ralmone.ttf", "truetype"),
        ("ralmone.otf", "opentype"),
        ("Gullying.ttf", "truetype"),
        ("Gullying.otf", "opentype"),
        ("gullying.ttf", "truetype"),
        ("gullying.otf", "opentype"),
    ):
        font_path = FONT_DIR / font_name
        if font_path.exists():
            encoded_font = base64.b64encode(font_path.read_bytes()).decode("ascii")
            return (
                "@font-face {"
                "font-family: 'HeroDisplayLocal';"
                f"src: url(data:font/{font_format};base64,{encoded_font}) format('{font_format}');"
                "font-weight: 100 900;"
                "font-style: normal;"
                "font-display: swap;"
                "}"
            )
    return ""


def image_to_data_uri(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "image/png")
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


st.set_page_config(
    page_title="Sentiment · Film Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── XNRGY-inspired dark editorial theme ──────────────────────────────────────
custom_font_css = load_hero_font_css()
if custom_font_css:
    st.markdown(f"<style>{custom_font_css}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,300;0,400;0,600;0,700;0,900;1,300&family=Barlow+Condensed:wght@300;400;700;900&family=JetBrains+Mono:wght@300;400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Lilita+One&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poiret+One&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0c0c0c !important;
    color: #e8e3dc !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header {
    visibility: visible !important;
    background: #0c0c0c !important;
}
.stDeployButton { display: none; }

/* ── App container ── */
.main .block-container {
    padding: 3rem 3.5rem 5rem !important;
    max-width: 1280px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #1f1f1f !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label {
    color: #a09890 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p {
    color: #e8e3dc !important;
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0 !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    box-sizing: border-box !important;
    min-height: 2.7rem !important;
    border-bottom: 1px solid #1f1f1f !important;
    padding: 0.15rem 0.85rem !important;
    cursor: pointer !important;
    transition: color 0.18s ease, background-color 0.18s ease, border-color 0.18s ease, padding-left 0.18s ease !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: rgba(200, 184, 154, 0.08) !important;
    padding-left: 1.1rem !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #a09890 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover p {
    color: #c8b89a !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
    background-color: rgba(200, 184, 154, 0.12) !important;
    border-left: 2px solid #c8b89a !important;
    padding-left: 1.1rem !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p {
    color: #e8e3dc !important;
}
/* Sidebar title */
[data-testid="stSidebar"] h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #e8e3dc !important;
    border-bottom: 1px solid #2a2a2a !important;
    padding-bottom: 1.5rem !important;
    margin-bottom: 1.5rem !important;
}

/* ── Page headings ── */
h1 {
    font-family: 'HeroDisplayLocal', 'Ralmone', 'Poiret One', 'Century Gothic', 'Montserrat', sans-serif !important;
    font-weight: 400 !important;
    font-size: clamp(3.2rem, 6vw, 6.4rem) !important;
    letter-spacing: -0.07em !important;
    line-height: 0.92 !important;
    text-transform: uppercase !important;
    color: #e8e3dc !important;
    margin-bottom: 0.25rem !important;
}

.hero-title-gullying {
    font-family: 'HeroDisplayLocal', 'Ralmone', 'Poiret One', 'Century Gothic', 'Montserrat', sans-serif !important;
    font-weight: 400 !important;
    font-size: clamp(4.8rem, 9.4vw, 9.4rem) !important;
    line-height: 0.9 !important;
    text-transform: uppercase !important;
    color: #e8e3dc !important;
    margin: 0 0 1.5rem 0 !important;
    letter-spacing: -0.075em !important;
}

.model-output-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin: 2rem 0 0;
}

.model-output-card {
    background: #0f0f0f;
    border: 1px solid #1f1f1f;
    border-top: 2px solid #c8b89a;
    padding: 1.6rem 1.8rem;
    min-height: 156px;
}

.model-output-model {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.66rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #c8b89a !important;
    max-width: none !important;
    margin: 0 0 1.2rem 0 !important;
}

.model-output-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 0.8fr);
    gap: 1.4rem;
    align-items: end;
}

.model-output-label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #584f46 !important;
    max-width: none !important;
    margin: 0 0 0.35rem 0 !important;
}

.model-output-verdict {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 900 !important;
    font-size: 2.6rem !important;
    line-height: 1 !important;
    text-transform: uppercase !important;
    max-width: none !important;
    margin: 0 !important;
}

.model-output-confidence {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    line-height: 1 !important;
    color: #e8e3dc !important;
    max-width: none !important;
    margin: 0 !important;
}

.model-output-note {
    color: #a09890 !important;
    font-size: 0.82rem !important;
    line-height: 1.45 !important;
    max-width: none !important;
    margin: 1rem 0 0 0 !important;
}

@media (max-width: 900px) {
    .model-output-grid { grid-template-columns: 1fr; }
    .model-output-row { grid-template-columns: 1fr; }
}

h2 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #e8e3dc !important;
    border-bottom: 1px solid #222 !important;
    padding-bottom: 0.6rem !important;
    margin-top: 2.5rem !important;
}

h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #a09890 !important;
}

/* ── Body text ── */
p, .stMarkdown p {
    font-family: 'Barlow', sans-serif !important;
    font-weight: 300 !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    color: #a09890 !important;
    max-width: 68ch !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #131313 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 0 !important;
    padding: 1.5rem 1.75rem !important;
    position: relative !important;
}
[data-testid="stMetric"]::before {
    content: '' !important;
    position: absolute !important;
    top: 0; left: 0; right: 0 !important;
    height: 1px !important;
    background: #c8b89a !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #584f46 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 900 !important;
    font-size: 2.2rem !important;
    color: #e8e3dc !important;
    letter-spacing: -0.01em !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 0.85rem 2.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background-color: #e8e3dc !important;
    color: #0c0c0c !important;
    border: none !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
    color: #050505 !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #c8b89a !important;
    color: #0c0c0c !important;
}
.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span,
.stButton > button[kind="primary"]:hover div {
    color: #050505 !important;
}
.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: #e8e3dc !important;
    border: 1px solid #2a2a2a !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #584f46 !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background-color: #111111 !important;
    border: 1px solid #222 !important;
    border-radius: 0 !important;
    color: #e8e3dc !important;
    font-family: 'Barlow', sans-serif !important;
    font-weight: 300 !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.25rem !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: #c8b89a !important;
    box-shadow: none !important;
}
.stTextArea label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #584f46 !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background-color: #111111 !important;
    border: 1px solid #222 !important;
    border-radius: 0 !important;
    color: #e8e3dc !important;
}
.stSelectbox label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #584f46 !important;
}

/* ── Dataframes / tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1f1f1f !important;
}
[data-testid="stDataFrame"] table {
    background-color: #0c0c0c !important;
}
[data-testid="stDataFrame"] th {
    background-color: #111111 !important;
    color: #584f46 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #222 !important;
}
[data-testid="stDataFrame"] td {
    color: #a09890 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border-bottom: 1px solid #161616 !important;
}

/* ── Code blocks ── */
.stCode, .stCode code, pre, code {
    background-color: #0f0f0f !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #c8b89a !important;
}

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    background-color: #111111 !important;
    border: 1px solid #222 !important;
    border-radius: 0 !important;
    border-left: 2px solid #c8b89a !important;
    color: #a09890 !important;
}
[data-testid="stAlert"] p { color: #a09890 !important; }

/* ── Warning ── */
.stWarning {
    background-color: #0f0f0f !important;
    border-left: 2px solid #7a6a50 !important;
}

/* ── Error ── */
.stError {
    border-left: 2px solid #8b3a3a !important;
}

/* ── Plotly chart wrapper ── */
[data-testid="stPlotlyChart"] {
    border: 1px solid #1a1a1a !important;
    background: #0f0f0f !important;
}

/* ── Radio buttons (sidebar nav) ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.6rem 0 !important;
    border-bottom: 1px solid #1a1a1a !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #1a1a1a !important;
    margin: 2rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0c0c0c; }
::-webkit-scrollbar-thumb { background: #2a2a2a; }
::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle(model_mtime: float):
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    try:
        df = load_or_create_dataset()
    except (FileNotFoundError, ValueError):
        return pd.DataFrame(columns=["text", "label", "label_name"])
    df["word_count"] = df["text"].str.split().str.len()
    return df


def label_to_display(label: str, bundle: dict) -> str:
    return bundle.get("label_names", {}).get(label, label)


def predict_review(bundle: dict, review: str) -> tuple[str, pd.DataFrame]:
    model = bundle["model"]
    prediction = model.predict([review])[0]
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review])[0]
    else:
        scores = model.decision_function([review])
        if np.ndim(scores) == 1:
            positive_probability = 1 / (1 + np.exp(-scores[0]))
            probabilities = np.array([1 - positive_probability, positive_probability])
        else:
            scores = np.asarray(scores[0])
            scores = scores - np.max(scores)
            probabilities = np.exp(scores) / np.exp(scores).sum()

    rows = [
        {
            "sentiment": label_to_display(label, bundle),
            "confidence": float(probabilities[index]),
        }
        for index, label in enumerate(classes)
    ]
    return prediction, pd.DataFrame(rows)


def load_bert_sentiment_pipeline():
    if "bert_sentiment_pipeline" in st.session_state:
        return st.session_state["bert_sentiment_pipeline"], ""

    try:
        from transformers import pipeline
    except ImportError:
        return None, "Install transformers and torch, then restart Streamlit."

    try:
        bert_pipeline = pipeline(
            "text-classification",
            model=BERT_MODEL_NAME,
            tokenizer=BERT_MODEL_NAME,
        )
    except Exception as exc:
        return None, (
            "BERT package is installed, but the pretrained model files are not downloaded yet. "
            "Run download_bert_model.bat once, then restart Streamlit."
        )

    st.session_state["bert_sentiment_pipeline"] = bert_pipeline
    return bert_pipeline, ""


def predict_bert_review(review: str) -> tuple[str | None, float | None, str]:
    bert_pipeline, error = load_bert_sentiment_pipeline()
    if error:
        return None, None, error

    try:
        result = bert_pipeline(review, truncation=True, max_length=512)[0]
    except Exception as exc:
        return None, None, f"BERT prediction failed: {exc}"

    raw_label = str(result.get("label", "")).upper()
    label_map = {
        "LABEL_0": "Negative",
        "NEGATIVE": "Negative",
        "LABEL_1": "Positive",
        "POSITIVE": "Positive",
    }
    display_label = label_map.get(raw_label, raw_label.title() if raw_label else "Unknown")
    confidence = float(result.get("score", 0.0))
    return display_label, confidence, ""


def model_card_html(model_name: str, verdict: str, confidence: float | None, note: str = "") -> str:
    if verdict == "Unavailable":
        color = GOLD_COLOR
    else:
        is_positive = verdict == "Positive"
        color = POSITIVE_COLOR if is_positive else NEGATIVE_COLOR
    confidence_text = f"{confidence:.1%}" if confidence is not None else "Not available"
    note_html = f'<p class="model-output-note">{escape(note)}</p>' if note else ""
    return f"""
    <div class="model-output-card" style="border-top-color: {color};">
        <p class="model-output-model">{escape(model_name)}</p>
        <div class="model-output-row">
            <div>
                <p class="model-output-label">Verdict</p>
                <p class="model-output-verdict" style="color: {color};">{escape(verdict)}</p>
            </div>
            <div>
                <p class="model-output-label">Confidence</p>
                <p class="model-output-confidence">{confidence_text}</p>
            </div>
        </div>
        {note_html}
    </div>
    """


def explain_review(bundle: dict, review: str, prediction: str, top_n: int = 10) -> pd.DataFrame:
    model = bundle["model"]
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    matrix = vectorizer.transform([review])
    feature_names = vectorizer.get_feature_names_out()
    nonzero = matrix.nonzero()[1]

    if len(nonzero) == 0:
        return pd.DataFrame(columns=["word", "importance"])

    values = np.asarray(matrix[:, nonzero].todense()).ravel()
    scores = values.copy()

    if hasattr(classifier, "coef_"):
        classes = list(classifier.classes_)
        class_index = classes.index(prediction)
        coefs = classifier.coef_
        if coefs.shape[0] == 1 and len(classes) == 2:
            signed = coefs[0, nonzero]
            scores = values * (signed if prediction == classes[1] else -signed)
        else:
            scores = values * coefs[class_index, nonzero]
    elif hasattr(classifier, "feature_log_prob_"):
        classes = list(classifier.classes_)
        class_index = classes.index(prediction)
        scores = values * classifier.feature_log_prob_[class_index, nonzero]

    order = np.argsort(scores)[::-1][:top_n]
    rows = [
        {
            "word": feature_names[nonzero[index]],
            "importance": round(float(scores[index]), 4),
        }
        for index in order
        if scores[index] > 0
    ]
    if not rows:
        order = np.argsort(values)[::-1][:top_n]
        rows = [
            {
                "word": feature_names[nonzero[index]],
                "importance": round(float(values[index]), 4),
            }
            for index in order
        ]
    return pd.DataFrame(rows)


def require_artifacts(bundle: dict | None, df: pd.DataFrame) -> bool:
    if bundle is not None and not df.empty:
        return True
    st.markdown("""
    <div style="
        margin-top: 4rem;
        padding: 3rem;
        border: 1px solid #1f1f1f;
        border-top: 2px solid #8b3a3a;
        background: #0f0f0f;
    ">
        <p style="
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #8b3a3a;
            margin: 0 0 0.5rem 0;
            max-width: none;
        ">System Notice</p>
        <p style="
            font-family: 'Barlow Condensed', sans-serif;
            font-weight: 700;
            font-size: 1.8rem;
            text-transform: uppercase;
            color: #e8e3dc;
            margin: 0 0 1.5rem 0;
            max-width: none;
        ">Model files not found</p>
        <p style="color: #584f46; font-size: 0.9rem; margin: 0 0 1rem 0; max-width: none;">
            Run these commands once from the project folder to generate required assets:
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.code(
        "python -m pip install -r requirements.txt\n"
        "python scripts/prepare_data.py\n"
        "python train_model.py",
        language="bash",
    )
    return False


TEXT_COLOR = "#f4efe7"
MUTED_TEXT = "#d2c7ba"
GRID_COLOR = "#303030"
PLOT_BG = "#151515"
PAPER_BG = "#101010"
POSITIVE_COLOR = "#39d98a"
NEGATIVE_COLOR = "#ff6b6b"
GOLD_COLOR = "#f0c987"
BLUE_COLOR = "#7aa7ff"


def readable_dark_layout(height: int | None = None) -> dict:
    layout = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Barlow, sans-serif", color=TEXT_COLOR, size=14),
        title_font=dict(family="Ralmone, Poiret One, Century Gothic, Montserrat, sans-serif", size=22, color=TEXT_COLOR),
        legend=dict(
            bgcolor=PAPER_BG,
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR, size=13),
            title_font=dict(color=TEXT_COLOR),
        ),
        margin=dict(l=56, r=30, t=58, b=54),
        xaxis=dict(
            showgrid=False,
            color=TEXT_COLOR,
            title_font=dict(color=TEXT_COLOR, size=14),
            tickfont=dict(color=MUTED_TEXT, size=13),
            zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            color=TEXT_COLOR,
            title_font=dict(color=TEXT_COLOR, size=14),
            tickfont=dict(color=MUTED_TEXT, size=13),
            zerolinecolor=GRID_COLOR,
        ),
    )
    if height:
        layout["height"] = height
    return layout


def render_animated_metric_grid(cards: list[dict], columns: int = 3) -> None:
    def format_card_value(card: dict, eased: float) -> str:
        value = card["value"]
        if card.get("animated", True) and isinstance(value, (int, float, np.integer, np.floating)):
            decimals = int(card.get("decimals", 0))
            current = float(value) * eased
            if decimals:
                return f"{current:,.{decimals}f}"
            return f"{round(current):,}"
        return escape(str(value))

    def metric_html(eased: float) -> str:
        card_markup = ""
        for card in cards:
            value = format_card_value(card, eased)
            suffix = card.get("suffix", "")
            suffix_markup = f'<span class="animated-metric-suffix">{escape(str(suffix))}</span>' if suffix else ""
            card_markup += (
                '<div class="animated-metric-card">'
                f'<div class="animated-metric-label">{escape(str(card["label"]))}</div>'
                f'<div><span class="animated-metric-value">{value}</span>{suffix_markup}</div>'
                "</div>"
            )
        return (
            "<style>"
            f".animated-metric-row{{display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));gap:14px;width:100%;}}"
            ".animated-metric-card{min-height:96px;padding:24px 26px;background:#111111;border:1px solid #222222;border-top:1px solid #c8b89a;box-sizing:border-box;}"
            ".animated-metric-label{font-family:'Barlow Condensed',sans-serif;font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;color:#c8b89a;margin-bottom:10px;}"
            ".animated-metric-value{font-family:'Barlow Condensed',sans-serif;font-size:1.45rem;line-height:1.15;color:#f4efe7;font-weight:400;}"
            ".animated-metric-suffix{font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;color:#d2c7ba;margin-left:4px;}"
            "@media(max-width:1000px){.animated-metric-row{grid-template-columns:repeat(2,minmax(0,1fr));}}"
            "@media(max-width:640px){.animated-metric-row{grid-template-columns:1fr;}}"
            "</style>"
            f'<div class="animated-metric-row">{card_markup}</div>'
        )

    placeholder = st.empty()
    frame_count = 24
    for frame in range(frame_count + 1):
        progress = frame / frame_count
        eased = 1 - (1 - progress) ** 3
        placeholder.markdown(metric_html(eased), unsafe_allow_html=True)
        if frame < frame_count:
            time.sleep(0.025)


def render_animated_home_metrics(total_reviews: int, best_model: str, best_accuracy: float) -> None:
    render_animated_metric_grid(
        [
            {"label": "Dataset", "value": total_reviews, "suffix": "reviews"},
            {"label": "Best model", "value": best_model, "animated": False},
            {"label": "Accuracy", "value": best_accuracy * 100, "suffix": "%", "decimals": 2},
        ],
        columns=3,
    )


bundle = load_model_bundle(MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0)
df = load_dataset()

st.markdown("""
<div style="display: none;">
    <p style="
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.6rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #584f46;
        margin: 0 0 0.3rem 0;
    ">NLP · Film Intelligence</p>
    <p style="
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 900;
        font-size: 1.3rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #e8e3dc;
        margin: 0 0 1.5rem 0;
        border-bottom: 1px solid #1f1f1f;
        padding-bottom: 1.5rem;
    ">Sentiment</p>
</div>
""", unsafe_allow_html=True)
page = "Home/About"

st.markdown("""
<style>
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) {
    position: sticky;
    top: 3rem;
    z-index: 100;
    align-items: center;
    gap: 1.5rem;
    width: 100%;
    margin: -1.5rem 0 2.75rem 0;
    padding: 0.8rem 0;
    background: rgba(12, 12, 12, 0.94);
    border-top: 1px solid #1f1f1f;
    border-bottom: 1px solid #2a2a2a;
    backdrop-filter: blur(12px);
}

.top-nav-brand {
    min-height: 2.65rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.top-nav-kicker {
    margin: 0 0 0.12rem 0;
    color: #584f46;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    white-space: nowrap;
}

.top-nav-title {
    margin: 0;
    color: #e8e3dc;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    line-height: 1;
    text-transform: uppercase;
    white-space: nowrap;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] {
    display: flex;
    justify-content: flex-end;
    gap: 1.15rem;
    width: 100%;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label {
    min-height: 2.65rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    margin: 0;
    padding: 0.55rem 0.12rem;
    border: 0;
    border-bottom: 1px solid transparent;
    background: transparent;
    cursor: pointer;
    transition: color 0.18s ease, background-color 0.18s ease, border-color 0.18s ease;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
    display: none !important;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label p {
    margin: 0;
    color: #8b827a;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    line-height: 1.1;
    text-align: center;
    text-transform: uppercase;
    white-space: nowrap;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label:hover {
    background: transparent;
    border-bottom-color: rgba(200, 184, 154, 0.5);
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label:hover p {
    color: #c8b89a;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"],
div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: transparent;
    border-bottom-color: #c8b89a;
}

div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] p,
div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
    color: #e8e3dc;
}

@media (max-width: 850px) {
    div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) {
        position: relative;
        top: 0;
        flex-direction: column;
        align-items: stretch;
        gap: 0.65rem;
        margin-top: -1.5rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
    div[data-testid="stRadio"] [role="radiogroup"] {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    div[data-testid="stHorizontalBlock"]:has(.top-nav-brand)
    div[data-testid="stRadio"] [role="radiogroup"] label p {
        white-space: normal;
    }
}
</style>
""", unsafe_allow_html=True)

brand_column, navigation_column = st.columns([1.35, 3.65])
with brand_column:
    st.markdown(
        """
        <div class="top-nav-brand">
            <p class="top-nav-kicker">NLP · Film Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with navigation_column:
    page = st.radio(
        "Navigate",
        ["Home/About", "Text Analyzer", "Data Explorer", "Visualizations", "Model Info"],
        horizontal=True,
        label_visibility="collapsed",
    )

if not require_artifacts(bundle, df):
    st.stop()

assert bundle is not None

# Three.js wireframe background for every page.
st.markdown("""
<style>
.main .block-container,
[data-testid="stSidebar"] {
    position: relative;
    z-index: 2;
}

#bg-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
}

#bg-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 1;
    pointer-events: none;
    background: rgba(12, 12, 12, 0.28);
}
</style>

<canvas id="bg-canvas" aria-hidden="true"></canvas>
<div id="bg-overlay" aria-hidden="true"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
    const canvas = document.getElementById("bg-canvas");
    if (!canvas || typeof THREE === "undefined") return;

    const renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        0.1,
        200
    );
    camera.position.set(0, 0, 28);

    const gold = new THREE.Color(0xc8b89a);
    const dim = new THREE.Color(0x584f46);
    const bright = new THREE.Color(0xe8e3dc);

    const rings = [];
    const ringMaterial = new THREE.MeshBasicMaterial({
        color: gold,
        wireframe: true,
        transparent: true,
        opacity: 0.18
    });
    for (let index = 0; index < 5; index += 1) {
        const geometry = new THREE.TorusGeometry(4 + index * 1.8, 0.06, 6, 48);
        const mesh = new THREE.Mesh(geometry, ringMaterial.clone());
        mesh.position.set(
            (Math.random() - 0.5) * 30,
            (Math.random() - 0.5) * 16,
            (Math.random() - 0.5) * 12 - 5
        );
        mesh.rotation.set(
            Math.random() * Math.PI,
            Math.random() * Math.PI,
            Math.random() * Math.PI
        );
        mesh.userData = {
            speedX: (Math.random() - 0.5) * 0.0012,
            speedY: (Math.random() - 0.5) * 0.0008,
            driftX: (Math.random() - 0.5) * 0.003,
            driftY: (Math.random() - 0.5) * 0.002
        };
        scene.add(mesh);
        rings.push(mesh);
    }

    const shapes = [];
    const shapeMaterial = new THREE.MeshBasicMaterial({
        color: dim,
        wireframe: true,
        transparent: true,
        opacity: 0.22
    });
    for (let index = 0; index < 4; index += 1) {
        const geometry = new THREE.IcosahedronGeometry(
            1.4 + Math.random() * 1.2,
            1
        );
        const mesh = new THREE.Mesh(geometry, shapeMaterial.clone());
        mesh.position.set(
            (Math.random() - 0.5) * 38,
            (Math.random() - 0.5) * 20,
            (Math.random() - 0.5) * 8 - 3
        );
        mesh.userData = {
            speedX: (Math.random() - 0.5) * 0.004,
            speedY: (Math.random() - 0.5) * 0.003,
            driftX: (Math.random() - 0.5) * 0.004,
            driftY: (Math.random() - 0.5) * 0.003
        };
        scene.add(mesh);
        shapes.push(mesh);
    }

    const pointCount = 380;
    const pointGeometry = new THREE.BufferGeometry();
    const pointPositions = new Float32Array(pointCount * 3);
    for (let index = 0; index < pointCount; index += 1) {
        pointPositions[index * 3] = (Math.random() - 0.5) * 120;
        pointPositions[index * 3 + 1] = (Math.random() - 0.5) * 60;
        pointPositions[index * 3 + 2] = (Math.random() - 0.5) * 40 - 10;
    }
    pointGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(pointPositions, 3)
    );
    const pointMaterial = new THREE.PointsMaterial({
        color: bright,
        size: 0.06,
        transparent: true,
        opacity: 0.35
    });
    scene.add(new THREE.Points(pointGeometry, pointMaterial));

    const lineGroup = new THREE.Group();
    scene.add(lineGroup);
    for (let index = 0; index < 18; index += 1) {
        const originX = (Math.random() - 0.5) * 50;
        const originY = (Math.random() - 0.5) * 28;
        const originZ = (Math.random() - 0.5) * 14 - 6;
        const length = 1.2 + Math.random() * 3.5;
        const angle = Math.random() * Math.PI;
        const points = [
            new THREE.Vector3(originX, originY, originZ),
            new THREE.Vector3(
                originX + Math.cos(angle) * length,
                originY + Math.sin(angle) * length,
                originZ
            )
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: Math.random() > 0.5 ? gold : dim,
            transparent: true,
            opacity: 0.12 + Math.random() * 0.1
        });
        const line = new THREE.Line(geometry, material);
        line.userData = {
            driftX: (Math.random() - 0.5) * 0.002,
            driftY: (Math.random() - 0.5) * 0.001
        };
        lineGroup.add(line);
    }

    let mouseX = 0;
    let mouseY = 0;
    document.addEventListener("mousemove", function(event) {
        mouseX = (event.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (event.clientY / window.innerHeight - 0.5) * 2;
    });

    window.addEventListener("resize", function() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    let time = 0;
    function animate() {
        requestAnimationFrame(animate);
        time += 0.008;

        rings.forEach(function(mesh, index) {
            mesh.rotation.x += mesh.userData.speedX;
            mesh.rotation.y += mesh.userData.speedY;
            mesh.position.x += Math.sin(time * 0.4 + index) * mesh.userData.driftX;
            mesh.position.y += Math.cos(time * 0.3 + index) * mesh.userData.driftY;
        });

        shapes.forEach(function(mesh, index) {
            mesh.rotation.x += mesh.userData.speedX;
            mesh.rotation.y += mesh.userData.speedY;
            mesh.position.x += Math.cos(time * 0.25 + index * 1.3) * mesh.userData.driftX;
            mesh.position.y += Math.sin(time * 0.2 + index * 0.9) * mesh.userData.driftY;
        });

        lineGroup.children.forEach(function(line, index) {
            line.position.x += Math.sin(time * 0.2 + index) * line.userData.driftX;
            line.position.y += Math.cos(time * 0.15 + index) * line.userData.driftY;
        });

        camera.position.x += (mouseX * 1.8 - camera.position.x) * 0.025;
        camera.position.y += (-mouseY - camera.position.y) * 0.025;
        camera.lookAt(0, 0, 0);
        renderer.render(scene, camera);
    }
    animate();
})();
</script>
""", unsafe_allow_html=True)

components.html(
    (BASE_DIR / "assets/wireframe_background.html").read_text(encoding="utf-8"),
    height=0,
)

if page == "Home/About":
    # Hero header
    st.markdown("""
    <div style="
        border-bottom: 1px solid #1a1a1a;
        padding-bottom: 2.5rem;
        margin-bottom: 3rem;
    ">
        <p style="
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 0.65rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #584f46;
            margin: 0 0 0.75rem 0;
            max-width: none;
        ">Supervised NLP · Binary Classification</p>
        <h1 class="hero-title-gullying">READ THE<br><span style="color: #c8b89a;">ROOM.</span></h1>
        <p style="
            font-family: 'Barlow', sans-serif;
            font-weight: 300;
            font-size: 1.05rem;
            color: #584f46;
            line-height: 1.7;
            max-width: 52ch;
            margin: 0;
        ">Predicts whether a movie review carries positive or negative sentiment
        using a supervised NLP classification pipeline trained on real audience data.</p>
    </div>
    """, unsafe_allow_html=True)

    best_model_metrics = next(
        (row for row in bundle["metrics"] if row["model"] == bundle["best_model_name"]),
        max(bundle["metrics"], key=lambda row: row["accuracy"]),
    )
    render_animated_home_metrics(
        bundle["dataset_stats"]["total_reviews"],
        bundle["best_model_name"],
        best_model_metrics["accuracy"],
    )

    st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown("""
        <p style="
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #584f46;
            margin: 0 0 0.5rem 0;
            max-width: none;
        ">The Problem</p>
        <p style="
            font-family: 'Barlow Condensed', sans-serif;
            font-weight: 700;
            font-size: 1.4rem;
            text-transform: uppercase;
            color: #e8e3dc;
            margin: 0 0 1rem 0;
            max-width: none;
        ">Thousands of opinions. Zero time.</p>
        <p style="max-width: 48ch;">
            Movie platforms accumulate thousands of reviews continuously. Manual reading
            is impractical. This system classifies audience sentiment automatically —
            so viewers, studios, and analysts can understand reception at scale.
        </p>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <p style="
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #584f46;
            margin: 0 0 0.5rem 0;
            max-width: none;
        ">How to use</p>
        <p style="
            font-family: 'Barlow Condensed', sans-serif;
            font-weight: 700;
            font-size: 1.4rem;
            text-transform: uppercase;
            color: #e8e3dc;
            margin: 0 0 1rem 0;
            max-width: none;
        ">One paste. One click.</p>
        <p style="max-width: 48ch;">
            Navigate to <strong style="color: #e8e3dc; font-weight: 600;">Text Analyzer</strong>
            in the top navigation, paste any movie review, and press
            <strong style="color: #e8e3dc; font-weight: 600;">Analyze Review</strong>.
            The model returns a prediction, confidence score, and word-level explanation.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    team_1_photo = image_to_data_uri(BASE_DIR / "assets/team/hazieq.png")
    team_1_photo_markup = (
        f'<img src="{team_1_photo}" alt="HAZIEQ AIMAN BIN HAMIZI">'
        if team_1_photo
        else "Image"
    )
    team_2_photo = image_to_data_uri(BASE_DIR / "assets/team/faris.png")
    team_2_photo_markup = (
        f'<img src="{team_2_photo}" alt="NIK MUHAMMAD FARIS BIN NIK ZAKI">'
        if team_2_photo
        else "Image"
    )
    team_3_photo = image_to_data_uri(BASE_DIR / "assets/team/hassan.png")
    team_3_photo_markup = (
        f'<img src="{team_3_photo}" alt="MUHAMMAD HASSAN NAEIM BIN NORIZAN">'
        if team_3_photo
        else "Image"
    )
    team_4_photo = image_to_data_uri(BASE_DIR / "assets/team/muqri.png")
    team_4_photo_markup = (
        f'<img src="{team_4_photo}" alt="NIK MUHAMMAD MUQRI HAZIM BIN NIK NOR AHMARIZAM">'
        if team_4_photo
        else "Image"
    )
    st.markdown("""
    <style>
    .team-section {
        border-top: 1px solid #1a1a1a;
        padding-top: 2rem;
    }

    .team-heading {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #584f46;
        margin: 0 0 1.4rem 0;
        max-width: none;
    }

    .team-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
    }

    .team-card {
        min-width: 0;
    }

    .team-photo-slot {
        width: 86%;
        aspect-ratio: 1 / 1.08;
        margin: 0 auto;
        border: 1px solid #c8b89a;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(200, 184, 154, 0.06), transparent 45%),
            #111111;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #584f46;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
    }

    .team-photo-slot img {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: cover;
        object-position: center top;
    }

    .team-name {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.08em !important;
        line-height: 1.25 !important;
        text-transform: uppercase !important;
        color: #e8e3dc !important;
        margin: 0.9rem 0 0.2rem 0 !important;
        max-width: none !important;
    }

    .team-matric {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        color: #c8b89a !important;
        margin: 0 !important;
        max-width: none !important;
    }

    @media (max-width: 980px) {
        .team-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 560px) {
        .team-grid { grid-template-columns: 1fr; }
        .team-photo-slot { width: 100%; }
    }
    </style>

    <div class="team-section">
        <p class="team-heading">Team</p>
        <div class="team-grid">
            <div class="team-card">
                <div class="team-photo-slot">__TEAM_1_PHOTO__</div>
                <p class="team-name">HAZIEQ AIMAN BIN HAMIZI</p>
                <p class="team-matric">A24AI0033</p>
            </div>
            <div class="team-card">
                <div class="team-photo-slot">__TEAM_2_PHOTO__</div>
                <p class="team-name">NIK MUHAMMAD FARIS BIN NIK ZAKI</p>
                <p class="team-matric">A24AI0070</p>
            </div>
            <div class="team-card">
                <div class="team-photo-slot">__TEAM_3_PHOTO__</div>
                <p class="team-name">MUHAMMAD HASSAN NAEIM BIN NORIZAN</p>
                <p class="team-matric">A24AI0056</p>
            </div>
            <div class="team-card">
                <div class="team-photo-slot">__TEAM_4_PHOTO__</div>
                <p class="team-name">NIK MUHAMMAD MUQRI HAZIM BIN NIK NOR AHMARIZAM</p>
                <p class="team-matric">A24AI0071</p>
            </div>
        </div>
    </div>
    """.replace("__TEAM_1_PHOTO__", team_1_photo_markup).replace("__TEAM_2_PHOTO__", team_2_photo_markup).replace("__TEAM_3_PHOTO__", team_3_photo_markup).replace("__TEAM_4_PHOTO__", team_4_photo_markup), unsafe_allow_html=True)

elif page == "Text Analyzer":
    st.markdown("""
    <p style="
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #584f46;
        margin: 0 0 0.4rem 0;
    ">Live Inference</p>
    <h1 style="margin: 0 0 0.1rem 0;">Text<br>Analyzer</h1>
    <div style="height: 2rem"></div>
    """, unsafe_allow_html=True)

    sample_reviews = {
        "Positive example": "The movie was beautifully acted, emotionally powerful, and completely satisfying.",
        "Negative example": "The plot was boring, the pacing was terrible, and the ending made no sense.",
    }
    choice = st.selectbox("Load an example", ["Write my own"] + list(sample_reviews.keys()))
    default_text = "" if choice == "Write my own" else sample_reviews[choice]
    review = st.text_area("Paste a movie review", value=default_text, height=160, placeholder="Type or paste your review here…")

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    if st.button("Analyze Review", type="primary"):
        if not review.strip():
            st.warning("Enter a review to analyze.")
        else:
            prediction, proba_df = predict_review(bundle, review)
            display_label = label_to_display(prediction, bundle)
            confidence = proba_df["confidence"].max()

            with st.spinner("Running pretrained BERT comparison..."):
                bert_label, bert_confidence, bert_error = predict_bert_review(review)

            bert_card = (
                model_card_html("Pretrained DistilBERT", bert_label, bert_confidence)
                if not bert_error and bert_label
                else model_card_html("Pretrained DistilBERT", "Unavailable", None, bert_error)
            )
            st.markdown(
                """
                <p style="
                    font-family: 'Barlow Condensed', sans-serif;
                    font-size: 0.65rem;
                    letter-spacing: 0.2em;
                    text-transform: uppercase;
                    color: #584f46;
                    margin: 2rem 0 0.75rem 0;
                    max-width: none;
                ">Model outputs</p>
                <div class="model-output-grid">
                """
                + model_card_html(bundle["best_model_name"], display_label, confidence)
                + bert_card
                + "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div style="height: 3.4rem"></div>
                <p style="
                    font-family: 'Ralmone', 'Poiret One', 'Century Gothic', 'Montserrat', sans-serif;
                    font-size: clamp(1.7rem, 2.4vw, 2.5rem);
                    line-height: 1;
                    letter-spacing: -0.035em;
                    text-transform: uppercase;
                    color: #e8e3dc;
                    margin: 0 0 1.25rem 0;
                    max-width: none;
                ">SVM confidence breakdown</p>
                """,
                unsafe_allow_html=True,
            )

            proba_fig = px.bar(
                proba_df,
                x="sentiment",
                y="confidence",
                text=proba_df["confidence"].map(lambda v: f"{v:.1%}"),
                range_y=[0, 1.08],
                color="sentiment",
                color_discrete_map={"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR},
            )
            proba_layout = readable_dark_layout(height=340)
            proba_layout.update(
                title=dict(text=""),
                showlegend=False,
                margin=dict(l=54, r=26, t=28, b=46),
                xaxis=dict(showgrid=False, title="", tickfont=dict(size=13, color=MUTED_TEXT)),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=GRID_COLOR,
                    title="",
                    tickformat=".0%",
                    tickfont=dict(size=13, color=MUTED_TEXT),
                ),
            )
            proba_fig.update_layout(**proba_layout)
            proba_fig.update_traces(
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>",
                marker_line_width=0,
                textfont=dict(family="Barlow Condensed", size=15, color=TEXT_COLOR),
                textposition="outside",
            )
            st.plotly_chart(
                proba_fig,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )

            st.markdown("""
            <p style="
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 0.65rem;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                color: #584f46;
                margin: 2rem 0 0.5rem 0;
                max-width: none;
            ">Signal Words</p>
            <p style="font-weight: 700; font-size: 1.1rem; text-transform: uppercase;
                font-family: 'Barlow Condensed'; color: #e8e3dc; max-width: none; margin-bottom: 1rem;">
                Words that drove the prediction
            </p>
            """, unsafe_allow_html=True)

            explanation = explain_review(bundle, review, prediction)
            if explanation.empty:
                st.markdown("<p style='color: #2a2a2a; font-style: italic;'>No strong signal words detected.</p>", unsafe_allow_html=True)
            else:
                st.dataframe(explanation, width="stretch", hide_index=True)

            st.markdown("""
            <p style="
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 0.65rem;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                color: #584f46;
                margin: 2rem 0 0.5rem 0;
                max-width: none;
            ">Pipeline Output</p>
            <p style="font-weight: 700; font-size: 1.1rem; text-transform: uppercase;
                font-family: 'Barlow Condensed'; color: #e8e3dc; max-width: none; margin-bottom: 0.75rem;">
                Preprocessed text
            </p>
            """, unsafe_allow_html=True)
            st.code(preprocess_text(review), language="text")

elif page == "Data Explorer":
    st.markdown("""
    <p style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.65rem;
        letter-spacing: 0.22em; text-transform: uppercase; color: #584f46; margin: 0 0 0.4rem 0;">
        Dataset Overview</p>
    <h1 style="margin: 0 0 0.1rem 0;">Data<br>Explorer</h1>
    <div style="height: 2rem"></div>
    """, unsafe_allow_html=True)

    stats = bundle["dataset_stats"]
    render_animated_metric_grid(
        [
            {"label": "Reviews", "value": stats["total_reviews"]},
            {"label": "Avg. words", "value": stats["average_words"], "decimals": 2},
            {"label": "Median words", "value": stats["median_words"], "decimals": 1},
            {"label": "Labels", "value": len(stats["labels"])},
        ],
        columns=4,
    )

    st.markdown("""
    <p style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.65rem;
        letter-spacing: 0.2em; text-transform: uppercase; color: #584f46;
        margin: 2.5rem 0 0.5rem 0; max-width: none;">Sample records</p>
    """, unsafe_allow_html=True)
    st.dataframe(df[["text", "label_name", "word_count"]].head(20), width="stretch")

    dark_layout = readable_dark_layout()

    label_counts = df["label_name"].value_counts().reset_index()
    label_counts.columns = ["sentiment", "count"]
    col1, col2 = st.columns(2)

    dist_fig = px.bar(label_counts, x="sentiment", y="count", color="sentiment",
                      color_discrete_map={"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR},
                      title="Class distribution")
    dist_fig.update_layout(**dark_layout)
    dist_fig.update_traces(showlegend=False)
    col1.plotly_chart(dist_fig, width="stretch")

    hist_fig = px.histogram(df, x="word_count", nbins=40, color="label_name",
                            color_discrete_map={"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR},
                            title="Review length distribution")
    hist_fig.update_layout(**dark_layout)
    col2.plotly_chart(hist_fig, width="stretch")

elif page == "Visualizations":
    st.markdown("""
    <p style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.65rem;
        letter-spacing: 0.22em; text-transform: uppercase; color: #584f46; margin: 0 0 0.4rem 0;">
        Charts &amp; Analysis</p>
    <h1 style="margin: 0 0 0.1rem 0;">Visualizations</h1>
    <div style="height: 2rem"></div>
    """, unsafe_allow_html=True)

    dark_layout = readable_dark_layout()

    # Word clouds with dark background
    positive_text = " ".join(df.loc[df["label"] == "pos", "text"].sample(400, random_state=42))
    negative_text = " ".join(df.loc[df["label"] == "neg", "text"].sample(400, random_state=42))

    st.markdown("""<p style="font-family: 'Barlow Condensed'; font-size: 0.65rem;
        letter-spacing: 0.2em; text-transform: uppercase; color: #584f46;
        margin-bottom: 0.75rem; max-width: none;">Word clouds</p>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    def plot_wordcloud_dark(text: str, title: str, colormap: str):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#0f0f0f")
        cloud = WordCloud(
            width=1000, height=500,
            background_color="#0f0f0f",
            colormap=colormap,
            max_words=120,
        ).generate(text)
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, color=TEXT_COLOR, fontsize=16, loc="left",
                     fontfamily="DejaVu Sans", fontweight="normal", pad=10)
        return fig

    col1.pyplot(plot_wordcloud_dark(positive_text, "Positive reviews", "YlGn"))
    col2.pyplot(plot_wordcloud_dark(negative_text, "Negative reviews", "OrRd"))

    # Sentiment distribution pie
    label_counts = df["label_name"].value_counts().reset_index()
    label_counts.columns = ["sentiment", "count"]
    pie_fig = px.pie(label_counts, names="sentiment", values="count",
                     title="Sentiment distribution",
                     color="sentiment",
                     color_discrete_map={"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR})
    pie_fig.update_layout(**{**dark_layout, "xaxis": None, "yaxis": None})
    pie_fig.update_traces(textfont=dict(family="Ralmone, Poiret One, Century Gothic, sans-serif", color="#111111", size=16))
    st.plotly_chart(pie_fig, width="stretch")

    # Confusion matrix
    cm = np.array(bundle["confusion_matrix"])
    display_labels = [label_to_display(label, bundle) for label in bundle["labels"]]
    cm_fig = px.imshow(
        cm, x=display_labels, y=display_labels,
        text_auto=True, color_continuous_scale=[[0, "#202020"], [1, POSITIVE_COLOR]],
        title=f"Confusion matrix — {bundle['best_model_name']}",
        labels={"x": "Predicted", "y": "Actual", "color": "Reviews"},
    )
    cm_fig.update_layout(**{**dark_layout, "xaxis": dict(title="Predicted", color=TEXT_COLOR),
                             "yaxis": dict(title="Actual", color=TEXT_COLOR)})
    st.plotly_chart(cm_fig, width="stretch")

    # Model comparison
    metrics_df = pd.DataFrame(bundle["metrics"])
    model_fig = px.bar(
        metrics_df, x="model",
        y=["accuracy", "precision", "recall", "f1_score"],
        barmode="group", title="Model comparison",
        color_discrete_sequence=[POSITIVE_COLOR, GOLD_COLOR, NEGATIVE_COLOR, BLUE_COLOR],
    )
    model_fig.update_layout(**dark_layout)
    st.plotly_chart(model_fig, width="stretch")

    # Top words
    top_words = pd.DataFrame(bundle["top_words"]["Positive"] + bundle["top_words"]["Negative"])
    top_words["sentiment"] = ["Positive"] * 30 + ["Negative"] * 30
    words_fig = px.bar(
        top_words.groupby(["sentiment", "word"], as_index=False)["count"].sum().head(40),
        x="count", y="word", color="sentiment", orientation="h",
        title="Top words by sentiment",
        color_discrete_map={"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR},
    )
    words_fig.update_layout(**readable_dark_layout(height=650))
    st.plotly_chart(words_fig, width="stretch")

    # Bigrams
    ngrams_df = pd.DataFrame(bundle["top_ngrams"])
    ngrams_fig = px.bar(ngrams_df, x="count", y="ngram", orientation="h",
                        title="Top bigrams",
                        color_discrete_sequence=[GOLD_COLOR])
    ngrams_fig.update_layout(**readable_dark_layout(height=560))
    st.plotly_chart(ngrams_fig, width="stretch")

elif page == "Model Info":
    st.markdown("""
    <p style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.65rem;
        letter-spacing: 0.22em; text-transform: uppercase; color: #584f46; margin: 0 0 0.4rem 0;">
        Technical Details</p>
    <h1 style="margin: 0 0 0.1rem 0;">Model<br>Info</h1>
    <div style="height: 2rem"></div>
    """, unsafe_allow_html=True)

    render_animated_metric_grid(
        [
            {"label": "Training samples", "value": bundle["train_size"]},
            {"label": "Testing samples", "value": bundle["test_size"]},
        ],
        columns=2,
    )

    st.markdown("""<p style="font-family: 'Barlow Condensed'; font-size: 0.65rem;
        letter-spacing: 0.2em; text-transform: uppercase; color: #584f46;
        margin: 2.5rem 0 0.5rem 0; max-width: none;">Preprocessing pipeline</p>""", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"step": bundle["preprocessing_steps"]}), width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<p style="font-family: 'Barlow Condensed'; font-size: 0.65rem;
            letter-spacing: 0.2em; text-transform: uppercase; color: #584f46;
            margin: 2rem 0 0.5rem 0; max-width: none;">Feature extraction</p>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"method": bundle["feature_methods"]}), width="stretch", hide_index=True)
    with col_b:
        st.markdown("""<p style="font-family: 'Barlow Condensed'; font-size: 0.65rem;
            letter-spacing: 0.2em; text-transform: uppercase; color: #584f46;
            margin: 2rem 0 0.5rem 0; max-width: none;">Models trained</p>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"model": bundle["models_trained"]}), width="stretch", hide_index=True)

    st.markdown("""<p style="font-family: 'Barlow Condensed'; font-size: 0.65rem;
        letter-spacing: 0.2em; text-transform: uppercase; color: #584f46;
        margin: 2.5rem 0 0.5rem 0; max-width: none;">Evaluation metrics</p>""", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(bundle["metrics"]), width="stretch", hide_index=True)
