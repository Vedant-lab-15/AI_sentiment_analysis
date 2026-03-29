"""
Tab 2 — Data Exploration
Search/filter controls, rating distribution, sentiment by category,
word clouds, time-series trends, and aspect analysis.
"""

import streamlit as st
import plotly.express as px

from utils.nlp import generate_wordcloud, extract_aspects

COLOR_MAP = {"positive": "#69F0AE", "neutral": "#FFD740", "negative": "#FF5252"}


def render(data):
    st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # --- Filters ---
    st.markdown("### 🔍 Search and Filter Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search in reviews", "")
    with col2:
        selected_category = st.selectbox("Filter by Category", ["All"] + sorted(data["category"].unique().tolist()))
    with col3:
        sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "positive", "neutral", "negative"])

    filtered = data.copy()
    if search_term:
        filtered = filtered[filtered["review_text"].str.contains(search_term, case=False, na=False)]
    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    # --- Data table + stats ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Filtered Reviews")
        st.dataframe(filtered[['review_text', 'rating', 'sentiment', 'category', 'subcategory']].head(10))
    with col2:
        st.subheader("Quick Stats")
        stats = {
            'Total Reviews':    len(filtered),
            'Average Rating':   round(filtered['rating'].mean(), 2) if len(filtered) else 0,
            'Positive Reviews': len(filtered[filtered['sentiment'] == 'positive']),
            'Neutral Reviews':  len(filtered[filtered['sentiment'] == 'neutral']),
            'Negative Reviews': len(filtered[filtered['sentiment'] == 'negative']),
            'Categories':       filtered['category'].nunique(),
            'Subcategories':    filtered['subcategory'].nunique(),
        }
        import pandas as pd
        st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))

    # --- Rating distribution ---
    st.subheader("Rating Distribution")
    rating_counts = filtered['rating'].value_counts().sort_index()
    total = rating_counts.sum()
    fig = px.bar(
        x=rating_counts.index, y=rating_counts.values,
        color=rating_counts.index, color_continuous_scale="RdYlGn",
        labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
        title="Rating Distribution",
    )
    for i, count in enumerate(rating_counts.values):
        fig.add_annotation(x=rating_counts.index[i], y=count, text=f"{count/total:.1%}", showarrow=False, yshift=10)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Sentiment by category ---
    st.subheader("Sentiment Distribution by Category")
    sent_by_cat = filtered.groupby(['category', 'sentiment']).size().reset_index(name='count')
    fig = px.bar(
        sent_by_cat, x="category", y="count", color="sentiment",
        color_discrete_map=COLOR_MAP, barmode='group',
        labels={"category": "Product Category", "count": "Reviews", "sentiment": "Sentiment"},
        title="Sentiment Across Product Categories",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Word clouds ---
    st.subheader("Most Common Words by Sentiment")
    wc_col1, wc_col2, wc_col3 = st.columns(3)
    for col, label, colormap in [
        (wc_col1, "negative", "Reds"),
        (wc_col2, "neutral",  "Oranges"),
        (wc_col3, "positive", "Greens"),
    ]:
        with col:
            text = ' '.join(filtered[filtered['sentiment'] == label]['processed_text'])
            if text.strip():
                st.pyplot(generate_wordcloud(text, f"{label.capitalize()} Reviews", colormap))
            else:
                st.info(f"No {label} reviews to display.")

    # --- Sentiment over time ---
    st.subheader("Sentiment Trends Over Time")
    filtered = filtered.copy()
    filtered['month'] = filtered['date'].dt.to_period('M').astype(str)
    over_time = filtered.groupby(['month', 'sentiment']).size().reset_index(name='count')
    fig = px.line(
        over_time, x="month", y="count", color="sentiment",
        color_discrete_map=COLOR_MAP, markers=True,
        labels={"month": "Month", "count": "Reviews", "sentiment": "Sentiment"},
        title="Sentiment Trends Over Time",
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # --- Aspect analysis ---
    st.subheader("Product Aspect Analysis by Sentiment")
    a_col1, a_col2, a_col3 = st.columns(3)
    for col, label in [(a_col1, "negative"), (a_col2, "neutral"), (a_col3, "positive")]:
        with col:
            reviews = filtered[filtered['sentiment'] == label]['review_text'].tolist()
            aspects = extract_aspects(reviews)
            st.markdown(f"#### {label.capitalize()} Aspects")
            fig = px.bar(
                x=list(aspects.keys()), y=list(aspects.values()),
                color=list(aspects.keys()),
                labels={"x": "Aspect", "y": "% of Reviews"},
                title=f"Aspects in {label.capitalize()} Reviews",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
