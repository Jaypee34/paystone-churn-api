# PayStone Customer Retention Intelligence Platform
## AI-Based Bank Customer Churn Prediction & Explainable Analytics

An end-to-end machine learning platform designed to predict customer churn, explain churn drivers, and support proactive customer retention.

📌 Project Overview

PayStone analyses 10,127 credit-card customers, including 1,627 churned customers (16.1%), using behavioural, financial, transactional, and engagement data.

Objective:

Identify customers likely to churn before they leave and support targeted retention action.

🏗️ Solution Architecture
Customer Data
     ↓
Data Preparation
     ↓
Feature Engineering
     ↓
EDA
     ↓
Model Development
     ↓
CatBoost
     ↓
SHAP Explainability
     ↓
Churn Risk Scoring
     ↓
FastAPI
     ↓
Docker
     ↓
GitHub → Render
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
Production Model- CatBoost
Metric	Result
ROC-AUC	92.5%
Accuracy	91.2%
Macro F1	82.9%
Churn Recall	62.8%

CatBoost correctly identified 204 of 325 actual churners.

🧠 Explainable AI

SHAP was used to explain model predictions.

Key churn drivers included:

Transaction activity changes
Inactivity
Relationship count
Contact frequency
Average transaction value
Transaction amount changes
Credit utilisation

The analysis indicates that customer behaviour and engagement are important indicators of churn risk.

🚀 FastAPI & Cloud Deployment

The CatBoost model was deployed as a REST API using:

FastAPI → Docker → GitHub → Render

API Endpoints
Endpoint	Purpose
/health	Health check
/model-info	Model information
/predict	Churn prediction
/docs	Swagger documentation

The API expects 25 model features. These are automatically extracted from the trained CatBoost model and stored in:

model/feature_columns.json

This allows the API to validate incoming data against the exact production model schema.

Live API

PayStone Live API

Swagger Documentation

PayStone API Documentation

📊 Power BI

Power BI can be used to monitor:

Churn rate
High-risk customers
Churn probability
Customer segments
Behavioural indicators
High-value customers at risk
SHAP churn drivers
📁 Project Structure
PayStone-Customer-Retention-Intelligence/
│
├── data/
├── notebooks/
├── models/
├── explainability/
├── api/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── model/
│       ├── CatBoost_churn_model.pkl
│       └── feature_columns.json
│
├── dashboard/
└── README.md
🧩 Technology Stack

Python | Pandas | Scikit-learn | CatBoost | XGBoost | LightGBM | SHAP | FastAPI | Docker | GitHub | Render | Power BI

💼 Business Value

PayStone enables businesses to:

Identify high-risk customers
Predict individual churn probability
Understand why customers are at risk
Prioritise retention activity
Monitor behavioural deterioration
Support targeted customer engagement
🏆 Key Achievement

92.5% ROC-AUC CatBoost model + Explainable AI + FastAPI + Docker + Render + Power BI

Predict → Explain → Prioritise → Retain

The public Swagger/API link you provided is: https://paystone-churn-api.onrender.com/docs
