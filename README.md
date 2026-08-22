

# PayStone Customer Retention Intelligence Platform

### AI-Based Bank Customer Churn Prediction & Explainable Analytics

 An end-to-end machine learning and Explainable AI platform designed to identify customers at risk of churn, explain the drivers behind churn predictions, and support proactive customer-retention strategies.



## 📌 Project Overview

**PayStone Customer Retention Intelligence Platform** is an end-to-end fintech customer analytics project that combines **machine learning, Explainable AI (XAI), customer risk scoring, cloud API deployment, and business intelligence** to transform historical customer data into a proactive retention capability.

The project analyses **10,127 credit-card customer records**, including **1,627 churned customers (16.1%)**, to identify the behavioural, transactional, financial, and engagement factors associated with customer attrition.

The solution moves PayStone from:

**Reactive Churn Reporting → Predictive Risk Identification → Explainable Retention Action**



## 🎯 Business Objective

The key business question addressed by this project is:

> **How can PayStone identify customers who are likely to churn before they leave?**

Customer churn can result in:

* Lost revenue
* Reduced customer lifetime value
* Higher customer acquisition costs
* Reduced product utilisation
* Lower customer engagement

The project therefore develops a predictive early-warning system capable of identifying high-risk customers and providing insights into **why they are likely to churn**.



## 💡 Solution Overview

The platform integrates the following capabilities:

1. Customer data preparation
2. Exploratory data analysis
3. Behavioural and financial feature engineering
4. Machine learning model development
5. Model comparison and optimisation
6. Customer churn probability scoring
7. SHAP-based Explainable AI
8. FastAPI model deployment
9. Cloud hosting using Render
10. Power BI business Monitoring



## 🏗️ End-to-End Architecture


                    CUSTOMER DATA
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Cleaning &     │
              │ Quality Assessment  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Feature Engineering │
              │ & Transformation    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Exploratory Data    │
              │ Analysis (EDA)      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Model Development   │
              │ & Comparison        │
              └──────────┬──────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │ CatBoost Production Model    │
          │ 92.5% ROC-AUC                │
          └──────────────┬───────────────┘
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
   ┌───────────────────┐    ┌──────────────────┐
   │ SHAP Explainable  │    │ Churn Risk       │
   │ AI                │    │ Scoring          │
   └─────────┬─────────┘    └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
                ┌─────────────────┐
                │ FastAPI REST API│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Render Cloud    │
                │ Deployment      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Power BI        │
                │ Dashboard       │
                └────────┬────────┘
                         │
                         ▼
                RETENTION ACTION




# 📊 Dataset

The project uses **10,127 customer records** containing demographic, financial, transactional, and customer-engagement information.

### Customer Base

| Metric             |  Value |
| ------------------ | -----: |
| Total Customers    | 10,127 |
| Churned Customers  |  1,627 |
| Existing Customers |  8,500 |
| Overall Churn Rate |  16.1% |

The dataset contains information relating to:

* Customer demographics
* Income category
* Education
* Marital status
* Credit-card category
* Customer tenure
* Product relationships
* Transaction activity
* Credit utilisation
* Revolving balance
* Customer contact frequency
* Inactivity
* Behavioural changes between quarters



# 🔍 Data Preparation & Feature Engineering

The modelling pipeline included:

* Data-quality assessment
* Missing-value analysis
* Duplicate detection and removal
* Removal of customer identifiers and non-predictive fields
* Categorical-variable encoding
* Numerical feature preparation
* Correlation and redundancy analysis
* Data-leakage prevention
* Behavioural feature engineering

### Key Engineered Features

| Feature                           | Description                                    |
| --------------------------------- | ---------------------------------------------- |
| `Customer_Tenure_Years`           | Customer relationship duration                 |
| `Average_Transaction_Value`       | Average transaction value                      |
| `Available_Credit_Ratio`          | Available credit relative to credit limit      |
| `Revolving_Balance_Ratio`         | Revolving balance relative to credit exposure  |
| `Contact_Frequency`               | Frequency of customer contacts                 |
| `Inactivity_Ratio`                | Relative level of customer inactivity          |
| `Transaction_Count_Change_Q4_Q1`  | Change in transaction count between Q1 and Q4  |
| `Transaction_Amount_Change_Q4_Q1` | Change in transaction amount between Q1 and Q4 |

These engineered variables were designed to capture **customer engagement, behavioural deterioration, credit utilisation, and relationship depth**.



# 📈 Exploratory Data Analysis

EDA was conducted across four major dimensions:

### Demographic Analysis

Customer churn was assessed across:

* Gender
* Education level
* Marital status
* Income category

### Financial Analysis

The analysis examined:

* Credit-card category
* Credit utilisation
* Revolving balances
* Available credit

### Behavioural Analysis

Key behavioural indicators included:

* Transaction activity
* Transaction-count changes
* Transaction-amount changes
* Inactivity
* Customer contact frequency

### Relationship Analysis

The analysis also considered:

* Total relationship count
* Months on book
* Customer tenure

The overall analysis indicates that **behavioural and engagement characteristics provide stronger churn signals than most static demographic characteristics**.



# 🤖 Machine Learning Model Development

Multiple classification algorithms were evaluated using a consistent modelling framework.

### Models Evaluated

* Logistic Regression — Baseline Model
* Random Forest
* XGBoost
* LightGBM
* CatBoost

### Model Evaluation

Models were evaluated using:

* ROC-AUC
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Cross-validation

Hyperparameter optimisation was performed using:

* `GridSearchCV`
* `RandomizedSearchCV`

---

# 🏆 Model Performance

**CatBoost was selected as the recommended production model.**

| Metric                        |  CatBoost |
| ----------------------------- | --------: |
| ROC-AUC                       | **92.5%** |
| Accuracy                      | **91.2%** |
| Macro Recall                  | **80.0%** |
| Macro F1-score                | **82.9%** |
| Actual Churners in Test Set   |       325 |
| Churners Correctly Identified |   **204** |
| Churn Recall                  | **62.8%** |

CatBoost achieved a **92.5% ROC-AUC**, demonstrating strong discrimination between customers who churn and those who remain.

The model correctly identified **204 of 325 actual churners**, resulting in a **62.8% recall for the churn class**.

### Validated Fallback Model

XGBoost achieved a **92.2% ROC-AUC** and was retained as a validated alternative model.



# 🧠 Explainable AI — SHAP

To improve model transparency and business interpretability, **SHAP (SHapley Additive exPlanations)** was applied to the CatBoost model.

SHAP provides both:

* **Global explanations** — what generally drives churn predictions
* **Customer-level explanations** — why an individual customer received a particular prediction

### Top Churn Drivers

The strongest model features included:

1. `Transaction_Count_Change_Q4_Q1`
2. `Inactivity_Ratio`
3. `Total_Relationship_Count`
4. `Contact_Frequency`
5. `Average_Transaction_Value`
6. `Transaction_Amount_Change_Q4_Q1`
7. `Gender`
8. `Revolving_Balance_Ratio`
9. `Avg_Utilization_Ratio`
10. `Education_Level_Graduate`

### Key Interpretation

The SHAP analysis reinforces the central business finding:

> **Customer behaviour and engagement are substantially stronger indicators of churn risk than most demographic characteristics.**

In particular, changes in transaction activity, inactivity, relationship depth, and contact frequency provide important early-warning signals.



# 📊 Key Business Insights

### 1. Customer Engagement Is the Strongest Churn Signal

Behavioural variables such as inactivity, transaction changes, contact frequency, and relationship count are among the strongest predictors of churn.

### 2. Churners Show Signs of Disengagement

Customers who churn tend to demonstrate:

* Reduced transaction activity
* Lower credit utilisation
* Fewer product relationships
* Higher levels of inactivity

### 3. Transaction Activity Is a Major Warning Signal

Changes in transaction count between Q1 and Q4 provide one of the strongest signals of customer attrition.

A decline in transaction activity may therefore provide an opportunity for **early intervention before the customer leaves**.

### 4. Customer Support Interactions Matter

Higher contact frequency may indicate:

* Customer dissatisfaction
* Service issues
* Increased support requirements
* Emerging relationship problems

This makes customer contact behaviour a potentially valuable retention signal.

### 5. Demographics Provide Supporting Context

Demographic characteristics contribute to churn prediction, but the model indicates that **behavioural and engagement variables generally provide stronger predictive information**.

Certain smaller segments, including Platinum cardholders and Doctorate-level customers, show elevated churn rates and should be investigated further using additional customer feedback and retention analysis


# 💼 Business Recommendations

Based on the modelling and explainability analysis, PayStone could implement the following retention strategy:

### 1. Monthly Customer Risk Scoring

Score the active customer base regularly using the production CatBoost model.

### 2. Prioritised Retention

Create risk tiers such as:


HIGH RISK
   ↓
Immediate Retention Intervention

MEDIUM RISK
   ↓
Targeted Engagement

LOW RISK
   ↓
Standard Customer Management


### 3. Monitor Behavioural Deterioration

Create alerts for:

* Declining transaction activity
* Increasing inactivity
* Reduced product relationships
* Changes in credit utilisation
* Increasing contact frequency

### 4. Targeted Customer Engagement

Use churn-risk scores and SHAP explanations to support personalised retention strategies rather than applying the same intervention to every customer.

### 5. Investigate High-Risk Segments

Further investigate segments with elevated churn rates, particularly where the sample size is sufficient to support reliable conclusions.

### 6. Continuous Model Monitoring

Monitor:

* ROC-AUC
* Recall
* Precision
* F1-score
* Churn-rate changes
* Population Stability Index (PSI)
* Prediction distributions

Retrain the model when customer behaviour or model performance changes materially.



# 🚀 FastAPI Deployment

The trained CatBoost model was operationalised through a **FastAPI REST API**.

### Deployment Components

CatBoost Model
      │
      ▼
CatBoost_churn_model.pkl
      │
      ▼
FastAPI Application
      │
      ├── /health
      ├── /model-info
      └── /predict
      │
      ▼
Docker Container
      │
      ▼
Render Cloud Deployment




# 🔌 Prediction API

The `/predict` endpoint accepts customer information and returns a churn prediction with associated probabilities.

### API Response


{
  "prediction": 1,
  "churn_prediction": "Churn",
  "churn_probability": 0.8995,
  "no_churn_probability": 0.1005
}


The deployed API successfully returned a **89.95% predicted churn probability** for the test customer.



# 🛡️ API Input Validation

The API includes validation to ensure that incoming customer data conforms to the model's expected schema.

The application validates:

* Missing features
* Unexpected features
* Invalid feature values
* Required categorical inputs
* Model-compatible feature structure

The deployed model expects **25 features**.

### Model Feature Schema

Income_Category
Card_Category
Education_Level_Doctorate
Education_Level_Graduate
Education_Level_High School
Education_Level_Post-Graduate
Education_Level_Uneducated
Education_Level_Unknown
Marital_Status_Married
Marital_Status_Single
Marital_Status_Unknown
Customer_Age
Gender
Dependent_count
Months_on_book
Total_Relationship_Count
Transaction_Amount_Change_Q4_Q1
Transaction_Count_Change_Q4_Q1
Avg_Utilization_Ratio
Customer_Tenure_Years
Average_Transaction_Value
Available_Credit_Ratio
Revolving_Balance_Ratio
Contact_Frequency
Inactivity_Ratio




# 📚 API Documentation

FastAPI automatically provides interactive Swagger/OpenAPI documentation through:



The deployed application can be accessed through the public Render service.



# 🔄 GitHub Deployment Workflow

The API was deployed using a GitHub-based CI/CD workflow.

### Deployment Process


Model Training
      │
      ▼
Save CatBoost Model
      │
      ▼
Develop FastAPI Application
      │
      ▼
Create Feature Schema
      │
      ▼
Create requirements.txt
      │
      ▼
Create Dockerfile
      │
      ▼
Push to GitHub
      │
      ▼
Connect Repository to Render
      │
      ▼
Build Docker Image
      │
      ▼
Deploy FastAPI Application
      │
      ▼
Test API Endpoints
```

### Repository Components

The GitHub repository contains:

```text
paystone-churn-api/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── feature_columns.json
├── model/
│   └── CatBoost_churn_model.pkl
│
└── README.md




# ☁️ Cloud Deployment

The FastAPI application was deployed to **Render** using Docker and GitHub integration.

### Deployment Features

* GitHub repository integration
* Docker-based deployment
* Automated build and deployment
* FastAPI REST API
* Cloud-hosted prediction service
* Swagger/OpenAPI documentation
* Health monitoring endpoint

### API Endpoints

| Endpoint      | Method | Purpose                               |
| ------------- | ------ | ------------------------------------- |
| `/health`     | GET    | API health check                      |
| `/model-info` | GET    | Model and feature information         |
| `/predict`    | POST   | Generate customer churn prediction    |
| `/docs`       | GET    | Interactive Swagger API documentation |


# 📊 Power BI Business Intelligence

The model outputs can be integrated into **Power BI** to support customer-retention monitoring.

Potential dashboard views include:

### Executive Overview

* Total customers
* Churn rate
* High-risk customers
* Average churn probability
* Retention rate

### Customer Risk

* High-risk customer count
* Risk distribution
* Churn probability distribution
* High-value customers at risk

### Behavioural Analysis

* Transaction activity
* Inactivity
* Contact frequency
* Credit utilisation
* Relationship count

### Segment Analysis

* Churn by income
* Churn by education
* Churn by marital status
* Churn by gender
* Churn by card category

### Explainability

* Top global churn drivers
* Segment-level drivers
* Customer-level SHAP explanations



# 🧩 Technology Stack

### Programming & Data

* Python
* Pandas
* NumPy
* Jupyter

### Machine Learning

* Scikit-learn
* CatBoost
* XGBoost
* LightGBM

### Explainable AI

* SHAP

### Visualisation


* Plotly

### Deployment & API

* FastAPI
* Docker
* Render
* Joblib

### Business Intelligence

* Power BI

### Development & Version Control

* VS Code
* Git
* GitHub



# 📁 Project Structure

PayStone-Customer-Retention-Intelligence/
│
├── data/
│   └── customer_churn_data.csv
│
├── notebooks/
│   └── churn_modelling.ipynb
│
├── models/
│   └── CatBoost_churn_model.pkl
│
├── explainability/
│   └── SHAP_analysis.ipynb
│
├── api/
│   ├── app.py
│   ├── feature_columns.json
│   ├── requirements.txt
│   └── Dockerfile
│
├── dashboard/
│   └── PayStone_Churn_Dashboard.pbix
│
├── reports/
│   └── model_performance.csv
│
└── README.md



# 🎯 Business Value

The PayStone platform demonstrates how machine learning can transform customer analytics from **descriptive reporting into a proactive retention capability**.

The solution enables PayStone to:

* Identify customers at risk of churn
* Quantify individual churn probability
* Understand why customers are considered high risk
* Prioritise retention activity
* Monitor behavioural deterioration
* Support targeted customer engagement
* Operationalise machine learning through an API
* Integrate predictive insights into business intelligence

Ultimately, the platform provides a framework for moving from:

> **"Which customers have already churned?"**

to:

> **"Which customers are likely to churn next, why are they at risk, and what can we do about it?"**



# 🔮 Future Enhancements

Potential future improvements include:

* Real-time churn scoring
* Automated retention campaigns
* Customer-level SHAP explanations through the API
* Risk-tier-based intervention workflows
* Model drift monitoring
* Automated model retraining
* Customer Lifetime Value integration
* Retention campaign uplift modelling
* A/B testing of retention strategies
* Automated Power BI data refresh
* Integration with CRM platforms


# ⚠️ Model Risk & Business Considerations

The model should be treated as a **decision-support tool rather than an autonomous decision-maker**.

Before production use at scale, PayStone should:

* Validate model performance on new customer cohorts
* Monitor model drift
* Review false positives and false negatives
* Assess segment-level model performance
* Validate the stability of high-risk segments
* Monitor potential bias across customer groups
* Establish appropriate model governance
* Ensure retention interventions are proportionate and commercially appropriate


# 👩‍💻 Project Summary

**PayStone Customer Retention Intelligence Platform** demonstrates an end-to-end machine learning lifecycle:


Data
 ↓
Data Quality
 ↓
Feature Engineering
 ↓
EDA
 ↓
Model Development
 ↓
Model Comparison
 ↓
Hyperparameter Optimisation
 ↓
CatBoost Production Model
 ↓
SHAP Explainability
 ↓
Customer Risk Scoring
 ↓
FastAPI
 ↓
Docker
 ↓
Render
 ↓
Power BI
 ↓
Proactive Customer Retention


### Key Achievement

**92.5% ROC-AUC CatBoost churn model + Explainable AI + Cloud API + Business Intelligence**

The project demonstrates the practical application of **machine learning, explainability, API engineering, cloud deployment, and business analytics** to a real-world customer-retention problem.



## 📌 Project Links

**GitHub Repository:** PayStone Customer Churn API

**Live API:** PayStone Customer Churn Prediction API

**Interactive API Documentation:** `/docs`

**Technology:** Python | CatBoost | SHAP | FastAPI | Docker | Render | Power BI



## ⭐ Key Takeaway

> **PayStone transforms customer data into an explainable early-warning system that helps identify churn risk before customer attrition occurs, enabling more proactive and targeted retention strategies.**

The public Swagger/API link you provided is: https://paystone-churn-api.onrender.com/docs
