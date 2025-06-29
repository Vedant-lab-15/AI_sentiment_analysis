# AI-Driven Sentiment Analysis Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## Overview

This project provides an interactive dashboard for analyzing sentiment in e-commerce product reviews using AI and Natural Language Processing techniques. The dashboard helps businesses extract actionable insights from customer feedback to improve products, services, and overall customer experience.

![Dashboard Preview](https://example.com/preview.png)

## Features

### 1. Project Overview
- Executive summary of business value and project goals
- Dataset statistics and visualization
- Methodology workflow explanation

### 2. Data Exploration
- Interactive review search and filtering
- Sentiment distribution across product categories
- Word clouds for positive/neutral/negative sentiment
- Time series analysis of sentiment trends
- Product aspect analysis by sentiment

### 3. Model Development
- Multiple ML models:
  - Logistic Regression
  - Random Forest
  - Ensemble approach
- Text vectorization options (TF-IDF, Bag of Words)
- Class imbalance handling with SMOTE
- Comprehensive model evaluation metrics

### 4. Results & Business Insights
- Key sentiment drivers and findings
- Business impact analysis with revenue projections
- Strategic recommendations based on sentiment analysis
- ROI calculator for sentiment improvement initiatives

### 5. Live Prediction
- Real-time sentiment analysis of user-entered reviews
- Aspect extraction and explanation
- Similar reviews identification
- Improvement suggestions for negative feedback

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-sentiment-analysis-dashboard.git
cd ai-sentiment-analysis-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Usage

Once the app is running, you can:

1. Explore the dataset through various visualizations in the Data Exploration tab
2. Train and evaluate different sentiment analysis models in the Model Development tab
3. View business insights and ROI calculations in the Results tab
4. Test the model on your own review text in the Live Prediction tab

## Dependencies

- streamlit==1.45.1
- pandas==2.1.1
- numpy==1.26.4
- matplotlib==3.10.3
- seaborn==0.13.2
- nltk==3.9.1
- plotly==6.1.2
- scikit-learn==1.3.2
- wordcloud==1.9.4
- imbalanced-learn==0.13.0

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── project_report.pdf     # Detailed project report
```

## Data

The application is designed to work with e-commerce product review datasets that contain the following fields:
- Review text
- Product category
- Rating (numerical)
- Date of review
- Product ID
- User ID (optional)

The dashboard automatically processes and cleans the text data, extracts sentiment and aspects, and provides visualizations of the results.

## Future Improvements

- Integration with transformer-based models (BERT/RoBERTa) for higher accuracy
- Automated report generation for downloading insights as PDF/CSV
- Real-time monitoring of new reviews from multiple platforms
- Competitor sentiment analysis comparison
- Multi-language support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback about this project, please contact:
- Email: contact@example.com
- GitHub: [github.com/example](https://github.com/example)