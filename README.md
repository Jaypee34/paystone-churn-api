
# PayStone Customer Retention Intelligence Platform

### AI-Based Bank Customer Churn Prediction & Explainable Analytics

An end-to-end machine learning platform designed to **predict customer churn, explain churn drivers, and support proactive customer retention**.

## 📌 Project Overview

PayStone analyses **10,127 credit-card customers**, including **1,627 churned customers (16.1%)**, using behavioural, financial, transactional, and engagement data.

**Objective:** Identify customers at risk of churn, understand why they are at risk, and support targeted retention actions.

## 🏗️ Solution Architecture

```text
Customer Data
     ↓
Data Preparation & Feature Engineering
     ↓
Model Development & Evaluation
     ↓
CatBoost Production Model
     ↓
SHAP Explainability
     ↓
Churn Risk Scoring
     ↓
FastAPI REST API
     ↓
Docker → GitHub → Render
     ↓
Power BI
     ↓
Retention Action

🤖 Machine Learning

Models evaluated:

Logistic Regression
Random Forest
XGBoost
LightGBM
CatBoost

Production Model: CatBoost

Metric	Result
ROC-AUC	92.5%
Accuracy	91.2%
Macro F1	82.9%
Churn Recall	62.8%

The model correctly identified 204 of 325 actual churners in the evaluated test set.

🧠 Explainable AI

SHAP was used to understand the factors driving individual churn predictions.

Key churn drivers included:

Transaction activity changes
Inactivity
Relationship count
Contact frequency
Average transaction value
Transaction amount changes
Credit utilisation

This enables the business to move from "Who is likely to churn?" to "Why is this customer at risk?"

🚀 API & Deployment

The trained CatBoost model is deployed as a REST API using:

FastAPI → Docker → GitHub → Render

Endpoints
Endpoint	Purpose
/health	API health check
/model-info	Model information
/predict	Customer churn prediction
/docs	Swagger documentation


The production model uses 25 features, with the expected feature schema stored in:

model/feature_columns.json
Example Response
{
  "prediction": 1,
  "churn_prediction": "Churn",
  "churn_probability": 0.8995,
  "no_churn_probability": 0.1005
}
📊 Power BI

Power BI supports monitoring of:

Churn rate
High-risk customers
Churn probability
Customer segments
High-value customers at risk
Behavioural indicators
SHAP churn drivers
📁 Project Structure
paystone-churn-api/
├── app.py
├── requirements.txt
├── Dockerfile
├── feature_columns.json
├── model/
│   └── CatBoost_churn_model.pkl
└── README.md
🧩 Technology Stack

Python | Pandas | Scikit-learn | CatBoost | XGBoost | LightGBM | SHAP | FastAPI | Docker | GitHub | Render | Power BI

💼 Business Value

PayStone enables organisations to:

Identify high-risk customers
Predict individual churn probability
Explain churn risk
Prioritise retention activity
Support targeted customer engagement
Predict → Explain → Prioritise → Retain

👤 Author

Eugene Osae
Data Scientist - AMDARI INC

The public Swagger/API link you provided is: https://paystone-churn-api.onrender.com/docs
