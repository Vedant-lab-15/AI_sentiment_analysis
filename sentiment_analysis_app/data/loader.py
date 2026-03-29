"""
Synthetic dataset generation for e-commerce product reviews.
All data is generated deterministically (seed=42) so the app is reproducible.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.nlp import preprocess_text


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """
    Build a 2 000-row synthetic review dataset that mimics real Amazon review data.
    Preprocessing is done here so it's cached alongside the data.
    """
    np.random.seed(42)
    n_samples = 2000

    # --- Review text pools ---
    positive_texts = [
        "This product is amazing! I love it and would highly recommend.",
        "Great quality for the price, exceeded my expectations.",
        "Works perfectly, exactly what I was looking for!",
        "Excellent product, fast shipping. Very satisfied.",
        "The best purchase I've made this year. Worth every penny.",
        "Incredible value for money, definitely recommend.",
        "Perfect fit and exceptional quality.",
        "This is exactly as described and arrived promptly.",
        "Absolutely love this product, can't imagine life without it now.",
        "Five stars! Excellent customer service and product quality.",
    ]
    neutral_texts = [
        "It's okay, nothing special but gets the job done.",
        "Average product, meets basic expectations.",
        "Not bad, but there's room for improvement.",
        "Decent quality for the price point.",
        "Functional but lacks some features I was hoping for.",
        "It works as expected, nothing more nothing less.",
        "The product is fine, shipping took longer than expected.",
        "Acceptable quality, though there are better alternatives.",
        "Neither impressed nor disappointed with this purchase.",
        "It serves its purpose but doesn't stand out from similar products.",
    ]
    negative_texts = [
        "Disappointed with this purchase, wouldn't recommend it.",
        "Poor quality, broke after just a few uses.",
        "Doesn't work as advertised, complete waste of money.",
        "Very frustrating experience with this product.",
        "Terrible customer service when I had issues with this item.",
        "The product arrived damaged and support was unhelpful.",
        "Completely overpriced for the quality you get.",
        "Save your money and look elsewhere, this is junk.",
        "Worst online purchase I've ever made.",
        "The item doesn't match the description at all.",
    ]

    # --- Category / subcategory map ---
    categories = {
        "Electronics":            ["Smartphones", "Laptops", "Headphones", "Cameras", "Smart Home Devices", "Gaming Accessories"],
        "Home & Kitchen":         ["Cookware", "Small Appliances", "Furniture", "Bedding", "Home Decor", "Storage & Organization"],
        "Beauty & Personal Care": ["Skincare", "Hair Care", "Makeup", "Fragrance", "Bath & Body", "Men's Grooming"],
        "Books":                  ["Fiction", "Non-fiction", "Children's Books", "Textbooks", "Self-help", "Cookbooks"],
        "Clothing":               ["Women's Fashion", "Men's Fashion", "Kids' Clothing", "Shoes", "Accessories", "Active Wear"],
    }
    flat_cats, flat_subs = [], []
    for cat, subs in categories.items():
        for sub in subs:
            flat_cats.append(cat)
            flat_subs.append(sub)

    # --- Generate rows ---
    seasons       = ["Regular", "Black Friday", "Holiday Season", "Prime Day", "Back to School"]
    season_w      = [0.70, 0.08, 0.12, 0.05, 0.05]
    sentiments    = np.random.choice(["positive", "neutral", "negative"], n_samples, p=[0.6, 0.2, 0.2])
    cust_segments = np.random.choice(["New", "Occasional", "Regular", "Loyal"], n_samples, p=[0.2, 0.3, 0.3, 0.2])

    ratings, texts, verified, helpful_votes, product_ids, shopping_seasons = [], [], [], [], [], []

    for s in sentiments:
        if s == "positive":
            ratings.append(np.random.choice([4, 5], p=[0.3, 0.7]))
            texts.append(np.random.choice(positive_texts) + " " + np.random.choice(["Love it!", "Great product!", "Highly recommended!", "Very satisfied!", "Excellent quality!"]))
            verified.append(np.random.choice([True, False], p=[0.9, 0.1]))
            helpful_votes.append(np.random.randint(5, 100))
        elif s == "neutral":
            ratings.append(3)
            texts.append(np.random.choice(neutral_texts) + " " + np.random.choice(["It's okay.", "Nothing special.", "Average product.", "Just okay.", "Could be better."]))
            verified.append(np.random.choice([True, False], p=[0.7, 0.3]))
            helpful_votes.append(np.random.randint(1, 20))
        else:
            ratings.append(np.random.choice([1, 2], p=[0.4, 0.6]))
            texts.append(np.random.choice(negative_texts) + " " + np.random.choice(["Disappointed.", "Wouldn't recommend.", "Save your money.", "Not worth it.", "Avoid this product."]))
            verified.append(np.random.choice([True, False], p=[0.6, 0.4]))
            helpful_votes.append(np.random.randint(0, 30))

        product_ids.append(f"PROD-{np.random.randint(1000, 9999)}")
        shopping_seasons.append(np.random.choice(seasons, p=season_w))

    # --- Dates spanning 2022-2023 ---
    start = pd.Timestamp('2022-01-01')
    days  = (pd.Timestamp('2023-12-31') - start).days
    dates = [start + pd.Timedelta(days=int(np.random.randint(0, days))) for _ in range(n_samples)]

    # --- Category assignment ---
    idx = np.random.randint(0, len(flat_cats), n_samples)

    df = pd.DataFrame({
        'review_text':      texts,
        'rating':           ratings,
        'date':             dates,
        'category':         [flat_cats[i] for i in idx],
        'subcategory':      [flat_subs[i] for i in idx],
        'verified_purchase': verified,
        'helpful_votes':    helpful_votes,
        'product_id':       product_ids,
        'customer_segment': cust_segments,
        'shopping_season':  shopping_seasons,
    })

    df['sentiment']     = df['rating'].apply(lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive'))
    df['review_length'] = df['review_text'].apply(len)
    df['word_count']    = df['review_text'].apply(lambda x: len(x.split()))

    # Preprocessing is cached here — no re-running on every page reload
    df['processed_text'] = df['review_text'].apply(preprocess_text)

    return df
