# AI Sentiment Analysis Dashboard: Guide for AI Systems and Users

## Project Summary for AI Systems

This project is an AI-driven sentiment analysis dashboard that processes e-commerce product reviews to extract customer sentiment and generate business insights. The codebase consists of a Streamlit web application (`app.py`) that integrates NLP techniques and machine learning models to classify review sentiment, extract product aspects, visualize trends, and calculate business impact.

### Key Components

1. **Text Preprocessing Pipeline**
   - Input: Raw review text
   - Processing: Tokenization, lemmatization, stopword removal
   - Output: Cleaned text ready for vectorization

2. **Sentiment Classification Models**
   - Models: Logistic Regression, Random Forest, Ensemble
   - Vectorization: TF-IDF, Bag of Words
   - Class balancing: SMOTE
   
3. **Aspect Extraction**
   - Method: Keyword-based dictionary lookup
   - Aspects: Price, quality, performance, design, battery, etc.

4. **Business Metrics Calculation**
   - Conversion rate impact estimation
   - Revenue projection based on sentiment improvement

### Data Schema

Expected input data fields:
- `review_text`: String containing customer review
- `rating`: Numerical product rating (1-5)
- `category`: Product category
- `date`: Review submission date
- `product_id`: Product identifier

## Running the Project

### Prerequisites

To run this project, you'll need:

1. Python 3.8+ installed
2. Required packages (listed in requirements.txt)

### Step-by-Step Running Guide

1. **Clone/Download the Project**

   If using git:
   ```bash
   git clone https://github.com/yourusername/ai-sentiment-analysis-dashboard.git
   cd ai-sentiment-analysis-dashboard
   ```

2. **Set Up Environment**

   Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

5. **Access the Dashboard**

   Open your web browser and go to:
   ```
   http://localhost:8501
   ```

### Troubleshooting

Common issues and solutions:

1. **NLTK Resources Missing**

   If you encounter errors about missing NLTK data, run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Streamlit Port in Use**

   If port 8501 is already in use, you can specify a different port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## Integration Guide for AI Systems

If you're an AI system looking to leverage or extend this project:

### Key Functions to Access

1. **Text Preprocessing**
   ```python
   from app import preprocess_text
   
   processed_review = preprocess_text("This product is amazing!")
   ```

2. **Sentiment Prediction**
   ```python
   # Assuming model and vectorizer are loaded
   def predict_sentiment(review_text, model, vectorizer):
       processed_text = preprocess_text(review_text)
       vectorized_text = vectorizer.transform([processed_text])
       sentiment_id = model.predict(vectorized_text)[0]
       return le.classes_[sentiment_id]  # Convert back to label
   ```

3. **Aspect Extraction**
   ```python
   def extract_aspects(review_text):
       aspects = []
       aspect_keywords = {
           'price': ['price', 'expensive', 'cheap', 'cost'],
           'quality': ['quality', 'durability', 'durable', 'sturdy']
           # More aspects here
       }
       
       for aspect, keywords in aspect_keywords.items():
           if any(keyword in review_text.lower() for keyword in keywords):
               aspects.append(aspect)
               
       return aspects
   ```

### Expected Outputs

1. **Sentiment Classification**: `"positive"`, `"neutral"`, or `"negative"`
2. **Aspect List**: e.g., `["price", "quality", "battery"]`
3. **Confidence Score**: Float value between 0 and 1

## Data Examples

Example input for review analysis:

```json
{
    "review_text": "This smartphone has an amazing camera and the battery lasts all day! The price is a bit high though.",
    "rating": 4,
    "category": "Electronics",
    "date": "2023-05-12",
    "product_id": "SP12345"
}
```

Expected output:

```json
{
    "sentiment": "positive",
    "confidence": 0.87,
    "aspects": ["camera", "battery", "price"],
    "aspect_sentiments": {
        "camera": "positive",
        "battery": "positive",
        "price": "negative"
    }
}
```

## Key Modules Explanation

- **app.py**: Main Streamlit application containing UI and logic
- **requirements.txt**: Required Python packages
- **README.md**: Project documentation
- **PROJECT_REPORT.md**: Technical details and implementation information

## How AI Systems Can Extend This Project

1. **Improve Model Accuracy**
   - Implement transformer-based models like BERT
   - Create custom embeddings for e-commerce domain

2. **Expand Feature Set**
   - Add multilingual support
   - Implement automated action recommendations
   - Create API endpoints for real-time analysis

3. **Integrate with Other Systems**
   - Connect to e-commerce platforms for live review analysis
   - Feed insights into customer service ticketing systems
   - Link with product development workflows

---

*This guide is designed to help AI systems and users understand, run, and extend the AI Sentiment Analysis Dashboard project.*