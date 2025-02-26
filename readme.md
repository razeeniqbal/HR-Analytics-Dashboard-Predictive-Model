# HR Analytics: Talent Retention & Engagement Predictor

A data science project to predict employee attrition, analyze HR trends, and provide actionable insights for talent retention.

## Project Overview

This HR Analytics project uses machine learning to help organizations:
- Predict which employees are at risk of leaving
- Understand key factors driving attrition
- Analyze department-level trends and metrics
- Generate actionable recommendations for improving retention

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset, which contains:
- Employee demographic information
- Job-related features
- Satisfaction metrics
- Compensation data
- Work history

## Features

- **Exploratory Data Analysis**: Visualize key HR metrics and identify patterns
- **Predictive Modeling**: Machine learning models to predict attrition risk
- **Interactive Dashboard**: User-friendly interface to explore data and get predictions
- **Actionable Insights**: Practical recommendations for employee retention

## Technical Components

- **Data Preprocessing**: Feature engineering, scaling, and encoding
- **Model Training**: Multiple ML algorithms (Logistic Regression, Random Forest, etc.)
- **Model Evaluation**: Comprehensive metrics and comparisons
- **Dashboard**: Interactive Streamlit application for visualization and predictions

## Project Structure
hr_analytics_project/
├── data/                  # Dataset storage
├── src/                   # Source code
│   ├── eda.py             # Exploratory data analysis
│   ├── data_preprocessing.py  # Data preprocessing and feature engineering
│   ├── model_training.py  # ML model training and evaluation
│   └── dashboard.py       # Streamlit dashboard
├── models/                # Saved models and preprocessing objects
├── output/                # Generated visualizations and results
│   ├── eda/
│   └── models/
└── requirements.txt       # Project dependencies
Copy
## Setup and Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the IBM HR Attrition dataset from Kaggle and save as `data/hr_data.csv`
4. Run the pipeline:
python src/eda.py
python src/data_preprocessing.py
python src/model_training.py
5. Launch the dashboard: `streamlit run src/dashboard.py`

## Skills Demonstrated

- Data preprocessing and feature engineering
- Machine learning model development and evaluation
- Data visualization and dashboard creation
- HR analytics domain knowledge
- Python programming (pandas, scikit-learn, streamlit, etc.)

## Future Improvements

- Implement more advanced ML models (XGBoost, Deep Learning)
- Add time-series analysis for temporal patterns
- Develop an API for model deployment
- Integrate with HR systems for real-time predictions
