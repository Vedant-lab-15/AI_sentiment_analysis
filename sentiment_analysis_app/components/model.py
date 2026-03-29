"""
Tab 3 — Model Development
Lets the user pick a model type, vectorizer, and training options,
then trains and evaluates the model live with metrics + confusion matrix.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def render(data):
    st.markdown('<div class="sub-header">Model Development</div>', unsafe_allow_html=True)

    st.markdown("""
    ### NLP Model Architecture

    This tab lets you train and evaluate a sentiment classifier on the review dataset.

    **Pipeline:**
    1. Text vectorization (TF-IDF or Bag-of-Words)
    2. Optional SMOTE oversampling to handle class imbalance
    3. Model training (Logistic Regression, Random Forest, or an Ensemble of both)
    4. Evaluation with accuracy, precision, recall, F1, confusion matrix, and classification report
    """)

    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)
    with col1:
        model_type      = st.selectbox("Model Type",            ["Logistic Regression", "Random Forest", "Ensemble"])
        vectorizer_type = st.selectbox("Vectorization Method",  ["TF-IDF", "Count Vectorization (Bag of Words)"])
    with col2:
        balance_classes = st.checkbox("Apply SMOTE for Class Balancing", value=True)
        test_size       = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

    if not st.button("Train Model"):
        return

    with st.spinner("Training… this may take a moment."):
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
            time.sleep(0.01)

        # --- Prepare data ---
        X = data["processed_text"]
        y = data["sentiment"]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
        )

        # --- Vectorize ---
        if vectorizer_type == "TF-IDF":
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        else:
            vec = CountVectorizer(max_features=5000, ngram_range=(1, 2))

        X_train_v = vec.fit_transform(X_train)
        X_test_v  = vec.transform(X_test)

        # --- SMOTE ---
        if balance_classes:
            X_train_v, y_train = SMOTE(random_state=42).fit_resample(X_train_v, y_train)

        # --- Train ---
        if model_type == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
            clf.fit(X_train_v, y_train)
            y_pred = clf.predict(X_test_v)

        elif model_type == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            clf.fit(X_train_v, y_train)
            y_pred = clf.predict(X_test_v)

        else:  # Ensemble — LR + RF, RF probabilities break ties
            lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            lr.fit(X_train_v, y_train)
            rf.fit(X_train_v, y_train)
            p_lr = lr.predict(X_test_v)
            p_rf = rf.predict(X_test_v)
            y_pred = np.where(p_lr == p_rf, p_lr, rf.predict_proba(X_test_v).argmax(axis=1))

    st.success("Training complete!")

    # --- Metrics ---
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    st.subheader("Evaluation Metrics")
    import pandas as pd
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value":  [accuracy, precision, recall, f1],
    })
    fig = px.bar(
        metrics_df, x="Metric", y="Value", color="Metric", text="Value",
        title=f"Performance — {model_type}",
        color_discrete_sequence=["#1E88E5", "#26A69A", "#AB47BC", "#FFA726"],
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Confusion matrix ---
    st.subheader("Confusion Matrix")
    cm          = confusion_matrix(y_test, y_pred)
    class_names = le.classes_
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # --- Classification report ---
    st.subheader("Classification Report")
    report    = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.highlight_max(axis=0))
