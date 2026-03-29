"""
AI-Driven Sentiment Analysis Dashboard
Entry point — wires together data loading, NLTK setup, and tab components.
"""

import streamlit as st

from data.loader import load_sample_data
from utils.nlp import download_nltk_resources
from components import overview, exploration, model, insights, prediction

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Driven Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.sub-header  { font-size:1.5em; font-weight:bold; color:#26A69A; margin-bottom:0.3em; }
.info-text   { font-size:1em; color:#555; }
.card        { padding:1.5rem; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,.1);
               margin-bottom:1rem; background-color:#f8f9fa; }
.metric-card { text-align:center; padding:1rem; border-radius:8px;
               box-shadow:0 2px 4px rgba(0,0,0,.05); background-color:white; }
</style>
""", unsafe_allow_html=True)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
download_nltk_resources()
data = load_sample_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 AI-Driven Sentiment Analysis of E-commerce Product Reviews")
st.markdown("""
<div class="info-text">
<p>This dashboard analyzes customer sentiment from e-commerce product reviews using
NLP and machine learning — turning raw feedback into actionable business insights.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_labels = [
    "📋 Project Overview",
    "📊 Data Exploration",
    "🤖 Model Development",
    "📈 Results & Insights",
    "🔮 Live Prediction",
]
tabs = st.tabs(tab_labels)

with tabs[0]: overview.render(data)
with tabs[1]: exploration.render(data)
with tabs[2]: model.render(data)
with tabs[3]: insights.render(data)
with tabs[4]: prediction.render(data)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info(
    "An interactive NLP dashboard that classifies e-commerce product reviews as "
    "positive, neutral, or negative and extracts key product aspects."
)

st.sidebar.title("Technologies")
st.sidebar.markdown("""
- Python 3.9+
- Streamlit
- Scikit-learn
- NLTK
- Pandas & NumPy
- Plotly & Matplotlib
""")
