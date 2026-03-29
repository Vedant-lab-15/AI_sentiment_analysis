"""
Tab 5 — Live Prediction
Rule-based lexicon classifier for demo purposes.
Classifies user-entered review text and extracts product aspects.
"""

import streamlit as st

from utils.nlp import preprocess_text

# Sentiment word lexicons
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best', 'perfect',
    'wonderful', 'fantastic', 'superior', 'outstanding', 'exceptional', 'impressive',
    'remarkable', 'phenomenal', 'terrific', 'superb', 'delightful', 'brilliant',
}
NEGATIVE_WORDS = {
    'bad', 'poor', 'terrible', 'horrible', 'awful', 'disappointed', 'disappointing',
    'worst', 'useless', 'defective', 'inferior', 'mediocre', 'broken', 'waste',
    'regret', 'frustrating', 'cheap', 'expensive', 'overpriced', 'problem',
}
NEUTRAL_WORDS = {
    'okay', 'ok', 'average', 'decent', 'fine', 'acceptable', 'standard',
    'ordinary', 'moderate', 'typical', 'regular', 'fair', 'reasonable',
    'adequate', 'sufficient', 'satisfactory', 'normal',
}

# Aspect keyword map
ASPECT_KEYWORDS = {
    'price':       ['price', 'expensive', 'cheap', 'cost', 'value', 'worth', 'bargain', 'overpriced'],
    'quality':     ['quality', 'durability', 'durable', 'sturdy', 'flimsy', 'solid'],
    'performance': ['fast', 'slow', 'speed', 'performance', 'responsive', 'lag', 'efficient'],
    'design':      ['design', 'look', 'color', 'style', 'appearance', 'aesthetics', 'beautiful', 'ugly'],
    'features':    ['feature', 'functionality', 'function', 'capabilities', 'option'],
    'battery':     ['battery', 'charge', 'long-lasting', 'power', 'drain'],
    'camera':      ['camera', 'photo', 'picture', 'video', 'image', 'resolution', 'megapixel'],
    'service':     ['service', 'support', 'customer service', 'help', 'assistance'],
}


def _classify(processed: str):
    """Return (sentiment_label, emoji, confidence, color) from processed text."""
    words = processed.lower().split()
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    neu = sum(w in NEUTRAL_WORDS  for w in words)

    if pos > neg + neu:
        return "positive",        "😃", min(0.5 + 0.1 * pos, 0.98), "#69F0AE"
    if neg > pos + neu:
        return "negative",        "☹️", min(0.5 + 0.1 * neg, 0.98), "#FF5252"
    if pos > neg:
        return "slightly positive","🙂", 0.5 + 0.05 * (pos - neg),  "#AED581"
    if neg > pos:
        return "slightly negative","😐", 0.5 + 0.05 * (neg - pos),  "#FF8A65"
    return "neutral",             "😐", 0.5 + 0.05 * neu,           "#FFD740"


def _find_aspects(raw: str) -> list:
    lower = raw.lower()
    return [aspect for aspect, kws in ASPECT_KEYWORDS.items() if any(kw in lower for kw in kws)]


def render(data):
    st.markdown('<div class="sub-header">Live Sentiment Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🔮 Try the Sentiment Analyzer
    Paste any product review below. The model will classify its sentiment,
    extract the product aspects mentioned, and show similar reviews from the dataset.
    """)

    user_input = st.text_area(
        "Enter a product review:",
        "This smartphone has an amazing camera and the battery lasts all day! The price is a bit high though.",
        height=150,
    )

    if not st.button("Analyze Sentiment"):
        return

    if not user_input.strip():
        st.error("Please enter a review to analyze.")
        return

    with st.spinner("Analyzing…"):
        processed = preprocess_text(user_input)
        sentiment, emoji, confidence, color = _classify(processed)
        found_aspects = _find_aspects(user_input)

    # --- Result header ---
    st.markdown(
        f"<h2 style='color:{color}'>Predicted Sentiment: {sentiment.capitalize()} {emoji}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.1%}")

    # --- Aspect + explanation ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Aspect Analysis")
        if found_aspects:
            st.markdown(f"**Aspects detected:** {', '.join(found_aspects)}")
        else:
            st.info("No specific product aspects detected.")

    with col2:
        st.subheader("Explanation")
        words = processed.lower().split()
        if sentiment == "positive":
            hits = [w for w in words if w in POSITIVE_WORDS][:3]
            st.write(f"Strong positive language detected: *{', '.join(hits)}*. Indicates customer satisfaction.")
        elif sentiment == "negative":
            hits = [w for w in words if w in NEGATIVE_WORDS][:3]
            st.write(f"Negative language detected: *{', '.join(hits)}*. Suggests customer dissatisfaction.")
        elif sentiment == "slightly positive":
            st.write("Leans positive but with some reservations or mixed feelings.")
        elif sentiment == "slightly negative":
            st.write("Leans negative but not strongly critical — some positive aspects may be present.")
        else:
            st.write("Balanced review with neither strong positive nor negative language.")

    # --- Similar reviews ---
    st.subheader("Similar Reviews from Dataset")
    base = sentiment.split()[-1]  # handles "slightly positive/negative"
    pool = data[data['sentiment'] == base]
    st.dataframe(pool.sample(min(3, len(pool)))[['review_text', 'rating']])

    # --- Improvement suggestions for negative reviews ---
    if sentiment in ("negative", "slightly negative") and found_aspects:
        st.subheader("Improvement Suggestions")
        suggestions = {
            'price':       "Review the pricing strategy — customers feel the value doesn't match the cost.",
            'quality':     "Investigate quality control — durability or build issues are being flagged.",
            'performance': "Look into performance optimizations — speed or responsiveness is a concern.",
            'battery':     "Battery life is a pain point — consider hardware or software improvements.",
            'camera':      "Camera quality needs attention — consider sensor or software enhancements.",
            'service':     "Customer service response times and resolution quality need improvement.",
            'design':      "Design or aesthetics are being criticized — gather more specific feedback.",
            'features':    "Missing features are frustrating customers — review the product roadmap.",
        }
        for aspect in found_aspects:
            if aspect in suggestions:
                st.markdown(f"- **{aspect.capitalize()}**: {suggestions[aspect]}")
