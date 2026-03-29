"""
NLP utilities — NLTK setup, text preprocessing, aspect extraction, word cloud generation.
"""

import re
import nltk
import matplotlib.pyplot as plt
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


@st.cache_resource
def download_nltk_resources():
    """Download required NLTK data if not already present."""
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package)


def preprocess_text(text: str) -> str:
    """
    Clean and normalize a review string.
    Steps: lowercase → strip non-alpha → tokenize → remove stopwords → lemmatize.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)


def extract_aspects(texts: list) -> dict:
    """
    Scan a list of review strings and return the percentage of reviews
    that mention each product aspect (quality, price, shipping, etc.).
    """
    aspect_keywords = {
        'quality':          ['quality', 'durability', 'durable', 'sturdy', 'flimsy', 'solid'],
        'price':            ['price', 'expensive', 'cheap', 'cost', 'value', 'worth', 'bargain', 'overpriced'],
        'shipping':         ['shipping', 'delivery', 'arrived', 'package', 'packaging', 'shipped'],
        'usability':        ['easy', 'difficult', 'simple', 'complicated', 'intuitive', 'user-friendly', 'confusing'],
        'customer_service': ['service', 'support', 'customer service', 'help', 'refund', 'replacement', 'warranty'],
        'design':           ['design', 'look', 'color', 'style', 'appearance', 'aesthetics', 'beautiful', 'ugly'],
    }

    counts = {aspect: 0 for aspect in aspect_keywords}
    lowered = [t.lower() for t in texts]

    for text in lowered:
        for aspect, keywords in aspect_keywords.items():
            if any(kw in text for kw in keywords):
                counts[aspect] += 1

    total = len(texts) if texts else 1
    return {aspect: (count / total) * 100 for aspect, count in counts.items()}


def generate_wordcloud(text: str, title: str, colormap: str):
    """Return a matplotlib figure containing a word cloud for the given text."""
    wc = WordCloud(
        width=400, height=200,
        background_color='white',
        colormap=colormap,
        max_words=50,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    return fig
