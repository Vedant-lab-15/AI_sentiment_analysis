"""
Tab 1 — Project Overview
Describes the project goal, dataset, tech stack, and methodology diagram.
"""

import streamlit as st


def render(data):
    st.markdown('<div class="sub-header">Project Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 Project Goal
        This project analyzes customer sentiment from e-commerce product reviews using
        Natural Language Processing (NLP). Reviews are classified as positive, negative,
        or neutral while key product aspects are extracted to surface actionable insights.

        #### Business Value
        - 📈 **Product Improvement**: Pinpoint aspects driving negative reviews
        - 🔍 **Customer Pain Points**: Understand what's frustrating buyers
        - 📊 **Trend Monitoring**: Track sentiment shifts over time and by category
        - 💹 **Revenue Impact**: Quantify the financial upside of sentiment improvements
        - 🚀 **Competitive Edge**: React to customer feedback faster than competitors
        """)

    with col2:
        st.markdown('### 📱 Sample Review')
        st.markdown('''
        <div class="card">
            <p><i>"This smartphone exceeded my expectations! The camera quality is amazing
            and battery life is exceptional. Definitely worth the price."</i></p>
            <p><strong>Category:</strong> Electronics › Smartphones</p>
            <p><strong>Sentiment:</strong> <span style="color:#69F0AE;font-weight:bold;">Positive</span></p>
            <p><strong>Key Aspects:</strong> Quality, Battery, Camera, Price</p>
        </div>''', unsafe_allow_html=True)

    # --- Dataset summary ---
    st.markdown("""
    ### 📚 Dataset
    - **2 000+ product reviews** across 5 categories and 30 subcategories
    - Ratings, verified purchase status, and helpfulness votes included
    - Time-series data spanning 2022–2023 for trend analysis
    - Customer segmentation (New / Occasional / Regular / Loyal)
    - Shopping season tags (Regular, Black Friday, Holiday Season, Prime Day, Back to School)
    """)

    col1, col2, col3, col4, col5 = st.columns(5)
    avg_rating = round(data['rating'].mean(), 1)
    for col, label, value in zip(
        [col1, col2, col3, col4, col5],
        ["Reviews Analyzed", "Product Categories", "Subcategories", "Unique Products", "Avg Rating"],
        [
            "2,000+",
            len(data['category'].unique()),
            len(data['subcategory'].unique()),
            data['product_id'].nunique(),
            f"{avg_rating}/5.0",
        ],
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'><h3>{value}</h3><p>{label}</p></div>",
                unsafe_allow_html=True,
            )

    # --- Tech stack ---
    st.markdown("### 🛠️ Technologies Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Data & ML
        - **Python** — core language
        - **Pandas & NumPy** — data wrangling
        - **Scikit-learn** — ML models and metrics
        - **NLTK** — text preprocessing
        - **Imbalanced-learn** — SMOTE class balancing
        """)
    with col2:
        st.markdown("""
        #### Visualization & Deployment
        - **Streamlit** — interactive dashboard
        - **Plotly & Matplotlib** — charts
        - **WordCloud** — text visualization
        - **Transformers / PyTorch** — BERT (advanced NLP)
        """)

    # --- Methodology diagram ---
    st.markdown("### 🔄 Methodology")
    st.graphviz_chart("""
    digraph {
        node [shape=box, style="rounded,filled", fontname=Arial, fontsize=12];
        edge [color=gray];

        A [label="Data Collection\\n& Preprocessing", fillcolor="#EBF5FB"];
        B [label="Exploratory\\nData Analysis",       fillcolor="#E8F8F5"];
        C [label="Feature\\nEngineering",             fillcolor="#FDEBD0"];
        D [label="Model Training\\n& Selection",      fillcolor="#F5EEF8"];
        E [label="Model\\nEvaluation",                fillcolor="#EBF5FB"];
        F [label="Business Insight\\nGeneration",     fillcolor="#E8F8F5"];
        G [label="Live Prediction\\nSystem",          fillcolor="#FDEBD0"];

        A -> B -> C -> D -> E -> F -> G;

        C1 [label="Text Cleaning\\nStopword Removal", fillcolor="#FADBD8"];
        C2 [label="Tokenization\\nLemmatization",     fillcolor="#FADBD8"];
        C3 [label="TF-IDF\\nVectorization",           fillcolor="#FADBD8"];
        C4 [label="Aspect\\nExtraction",              fillcolor="#FADBD8"];

        C -> C1 -> C2 -> C3 -> C4 [style=dotted];
        C4 -> D [style=dotted];

        D1 [label="Logistic Regression", fillcolor="#D5F5E3"];
        D2 [label="Random Forest",       fillcolor="#D5F5E3"];
        D3 [label="Ensemble Model",      fillcolor="#D5F5E3"];

        D -> D1 -> D2 -> D3 [style=dotted];
        D3 -> E [style=dotted];
    }
    """)
