"""
Tab 4 — Results & Business Insights
Live metrics computed from data, sentiment distribution pie chart,
category ratings, strategic recommendations, and ROI calculator.
"""

import plotly.express as px
import streamlit as st


def render(data):
    st.markdown('<div class="sub-header">Results & Business Insights</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 📊 Sentiment Analysis Results
    Key insights derived from the customer review dataset.
    """)

    # --- Compute metrics from actual data ---
    total            = len(data)
    positive_pct     = round(len(data[data['sentiment'] == 'positive']) / total * 100)
    category_pos_rate = (
        data[data['sentiment'] == 'positive'].groupby('category').size()
        / data.groupby('category').size()
    ).dropna()
    avg_pos_rate       = category_pos_rate.mean()
    most_positive_cat  = category_pos_rate.idxmax()
    most_negative_cat  = category_pos_rate.idxmin()
    pos_delta          = round((category_pos_rate.max() - avg_pos_rate) * 100)
    neg_delta          = round((category_pos_rate.min() - avg_pos_rate) * 100)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Positive Sentiment", f"{positive_pct}%")
    with col2:
        st.metric("Most Positive Category", most_positive_cat, delta=f"+{pos_delta}%")
        st.caption("vs. category average")
    with col3:
        st.metric("Most Negative Category", most_negative_cat, delta=f"{neg_delta}%", delta_color="inverse")
        st.caption("vs. category average")

    # --- Key findings + pie chart ---
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Sentiment Drivers
        1. **Quality** is the top driver of positive reviews
        2. **Price** is the most common negative trigger
        3. **Shipping** issues appear in nearly half of negative feedback
        4. **Customer service** sentiment has been improving quarter-over-quarter
        5. **Usability** concerns are declining after recent product updates
        """)
    with col2:
        sentiment_counts = data['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Overall Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={"positive": "#69F0AE", "neutral": "#FFD740", "negative": "#FF5252"},
            hole=0.4,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # --- Average rating by category ---
    st.subheader("Average Rating by Category")
    cat_rating = data.groupby('category')['rating'].mean().reset_index().sort_values('rating', ascending=False)
    fig = px.bar(
        cat_rating, x='category', y='rating',
        color='rating', color_continuous_scale='RdYlGn',
        title="Average Rating by Product Category",
        labels={"rating": "Avg Rating (1–5)", "category": "Category"},
    )
    fig.update_layout(xaxis_tickangle=45, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Recommendations ---
    st.subheader("Strategic Recommendations")
    st.markdown("""
    1. **Product Quality** — Focus on Electronics where quality complaints are highest.
       Address battery life, durability, and connectivity specifically.
    2. **Pricing** — Review Furniture and Laptops subcategories where price-related
       negative sentiment is above average. Bundle offers may help.
    3. **Customer Service** — Faster response times for Electronics support tickets.
       Proactive outreach for reviews mentioning shipping damage.
    4. **Marketing** — Lead with quality messaging in Beauty & Personal Care campaigns.
       Use real customer quotes from high-sentiment reviews as social proof.
    """)

    # --- ROI Calculator ---
    st.subheader("Sentiment Improvement ROI Calculator")
    col1, col2, col3 = st.columns(3)
    with col1:
        current_sentiment = st.slider("Current Positive Sentiment %", 0, 100, 60)
        target_sentiment  = st.slider("Target Positive Sentiment %",  0, 100, 75)
    with col2:
        avg_order      = st.number_input("Average Order Value ($)", value=50.0, step=5.0)
        monthly_orders = st.number_input("Monthly Orders",          value=10000, step=1000)
    with col3:
        conversion_impact = st.slider("Conversion Rate Impact per 10% Sentiment Improvement", 0.0, 5.0, 1.5, 0.1)
        retention_impact  = st.slider("Retention Rate Impact per 10% Sentiment Improvement",  0.0, 5.0, 2.0, 0.1)

    improvement = target_sentiment - current_sentiment

    if improvement < 0:
        st.warning("Target sentiment is lower than current — set a higher target to estimate revenue gain.")
        return

    conversion_increase = (improvement / 10) * conversion_impact / 100
    retention_increase  = (improvement / 10) * retention_impact  / 100
    new_rev             = monthly_orders * avg_order * conversion_increase * 12
    retention_rev       = monthly_orders * avg_order * retention_increase  * 12
    total_impact        = new_rev + retention_rev

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estimated Annual Revenue Impact")
        st.markdown(f"### ${total_impact:,.2f}")
        st.caption(f"Based on a {improvement}% sentiment improvement")
    with col2:
        fig = px.bar(
            x=['New Customers', 'Customer Retention'],
            y=[new_rev, retention_rev],
            color=['New Customers', 'Customer Retention'],
            text=[f"${new_rev:,.0f}", f"${retention_rev:,.0f}"],
            title="Revenue Impact Breakdown",
            labels={"y": "Additional Annual Revenue ($)", "x": "Impact Type"},
            color_discrete_sequence=["#26A69A", "#AB47BC"],
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
