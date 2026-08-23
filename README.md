
## PayStone Customer Retention Intelligence Platform — Project Explanation

**PayStone** is an end-to-end **customer churn prediction and retention platform** that uses machine learning to identify bank customers who are likely to leave, explain the reasons behind their risk, and make those predictions available through a deployed API.

### 1. Business Problem

Customer churn is a major challenge for financial institutions. If a bank can identify customers who are becoming less engaged **before they leave**, it can take proactive action such as targeted communication, offers, or relationship management.

The project therefore answers three key questions:

1. **Who is likely to churn?**
2. **How likely are they to churn?**
3. **Why are they at risk?**

---

### 2. Data

The project analyses **10,127 credit-card customers**, of which **1,627 (16.1%) had churned**.

The data contains information relating to:

* Customer demographics
* Account relationships
* Transaction behaviour
* Credit utilisation
* Customer engagement
* Inactivity
* Contact frequency
* Transaction changes

Feature engineering was also used to create behavioural indicators such as transaction changes, utilisation ratios, customer tenure, and inactivity measures.

---

### 3. Machine Learning

Several classification models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost

**CatBoost was selected as the production model**, achieving:

| Metric       |    Result |
| ------------ | --------: |
| ROC-AUC      | **92.5%** |
| Accuracy     | **91.2%** |
| Macro F1     | **82.9%** |
| Churn Recall | **62.8%** |

The model correctly identified **204 of 325 actual churners** in the evaluated test set.

The model produces a **churn probability**, allowing customers to be prioritised according to their level of risk rather than simply classifying them as churn/no-churn.

---

### 4. Explainable AI

A major part of the project is **Explainable AI using SHAP**.

A traditional model might say:

> "This customer has an 89.95% probability of churning."

SHAP helps answer:

> **"What factors caused the model to assign this customer a high risk?"**

Important churn drivers identified included:

* Changes in transaction activity
* Customer inactivity
* Relationship count
* Contact frequency
* Average transaction value
* Transaction amount changes
* Credit utilisation

This makes the model more useful for business users because they can understand the **reason behind the prediction**.

---

### 5. API Deployment

Rather than keeping the model inside a notebook, the trained CatBoost model was turned into a **production-style REST API using FastAPI**.

The workflow is:

```text
Customer Information
        ↓
FastAPI
        ↓
Input Validation
        ↓
Feature Validation
        ↓
CatBoost Model
        ↓
Churn Probability
        ↓
Churn Prediction
```

The API provides endpoints such as:

* `/health` — checks whether the API is operational
* `/model-info` — provides model information
* `/predict` — generates a churn prediction
* `/docs` — interactive Swagger documentation

The API expects **25 production features**, with the exact feature schema stored in `feature_columns.json`. This helps ensure that incoming data matches the structure expected by the trained model.

---

### 6. Docker and Cloud Deployment

The API was containerised using **Docker** and deployed through:

```text
FastAPI
   ↓
Docker
   ↓
GitHub
   ↓
Render
```

---

### 7. Power BI

The predictive outputs can then be used in **Power BI** to provide business-level monitoring of:

* Overall churn rate
* High-risk customers
* Churn probability
* Customer segments
* High-value customers at risk
* Behavioural indicators
* SHAP churn drivers

This connects the machine-learning model with **business decision-making**.

---

## Overall Project Flow

The complete project can be summarised as:

```text
Customer Data
      ↓
Data Preparation
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
CatBoost Churn Model
      ↓
Churn Probability
      ↓
SHAP Explainability
      ↓
FastAPI
      ↓
Docker
      ↓
Render
      ↓
Power BI
      ↓
Targeted Retention
```

### In simple terms

**PayStone takes customer data, predicts who is likely to leave, calculates how likely they are to leave, explains why they are at risk, and makes those predictions available through a deployed API so that businesses can prioritise retention activity.**

The overall concept is:

> **Predict → Explain → Prioritise → Retain**


👤 Author

Eugene Osae
Data Scientist - AMDARI INC

The public Swagger/API link you provided is: https://paystone-churn-api.onrender.com/docs
