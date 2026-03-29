# AI-Driven Sentiment Analysis Dashboard

An interactive Streamlit dashboard that analyzes customer sentiment from e-commerce product reviews using NLP and machine learning. Reviews are classified as positive, neutral, or negative, key product aspects are extracted, and the results are surfaced as actionable business insights.

---

## Features

- **Data Exploration** — filter reviews by category, sentiment, or keyword; view rating distributions, word clouds, and sentiment trends over time
- **Model Development** — train Logistic Regression, Random Forest, or an Ensemble model live in the browser with configurable vectorization and SMOTE balancing
- **Results & Insights** — computed sentiment metrics, category breakdowns, strategic recommendations, and an ROI calculator
- **Live Prediction** — paste any review and get an instant sentiment classification with aspect detection and improvement suggestions

---

## Project Structure

```
sentiment_analysis_app/
├── app.py                  # Entry point — wires tabs together
├── run.sh                  # One-command setup and launch script
├── requirements.txt        # Pinned Python dependencies
│
├── data/
│   └── loader.py           # Synthetic dataset generation + preprocessing
│
├── components/
│   ├── overview.py         # Tab 1 — Project Overview
│   ├── exploration.py      # Tab 2 — Data Exploration
│   ├── model.py            # Tab 3 — Model Development
│   ├── insights.py         # Tab 4 — Results & Insights
│   └── prediction.py       # Tab 5 — Live Prediction
│
└── utils/
    └── nlp.py              # NLTK setup, text preprocessing, aspect extraction, word cloud
```

---

## Quick Start

### Option 1 — run.sh (recommended)

```bash
cd sentiment_analysis_app
bash run.sh
```

The script will:
1. Check for Python 3
2. Create a `.venv` virtual environment
3. Install all dependencies
4. Launch the app at `http://localhost:8501`

### Option 2 — manual setup

```bash
cd sentiment_analysis_app

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Requirements

- Python 3.9+
- Internet connection on first run (NLTK downloads ~5 MB of language data)

Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.45.1 | Dashboard framework |
| scikit-learn | 1.3.2 | ML models and metrics |
| nltk | 3.9.1 | Text preprocessing |
| pandas / numpy | 2.1.1 / 1.26.4 | Data wrangling |
| plotly | 6.1.2 | Interactive charts |
| imbalanced-learn | 0.13.0 | SMOTE class balancing |
| wordcloud | 1.9.4 | Word cloud visualization |
| torch / transformers | 2.2.2 / 4.40.0 | BERT (advanced NLP, future use) |

---

## How It Works

1. **Data** — 2 000 synthetic reviews are generated deterministically (seed 42) across 5 product categories, mimicking real Amazon review distributions.
2. **Preprocessing** — reviews are lowercased, cleaned, tokenized, stopword-filtered, and lemmatized using NLTK. This runs once and is cached.
3. **Modeling** — TF-IDF or Bag-of-Words vectorization feeds into Logistic Regression, Random Forest, or an Ensemble (LR + RF with RF probability tiebreaker).
4. **Aspect extraction** — keyword matching identifies which product aspects (quality, price, shipping, etc.) are mentioned in each review.
5. **Live prediction** — a lexicon-based classifier gives instant results without needing a pre-trained model loaded in memory.

---

## Screenshots

| Tab | Description |
|---|---|
| Project Overview | Goal, dataset summary, tech stack, methodology diagram |
| Data Exploration | Filters, charts, word clouds, time-series trends |
| Model Development | Live training with configurable options and evaluation metrics |
| Results & Insights | Computed KPIs, recommendations, ROI calculator |
| Live Prediction | Real-time sentiment classification with aspect detection |
