
# =============================================================================
# PAYSTONE CUSTOMER CHURN PREDICTION
# FASTAPI PREDICTION APPLICATION
# =============================================================================
#
# This application exposes the trained PayStone CatBoost churn model through
# a RESTful API.
#
# ENDPOINTS
# ---------
#
# GET  /health
# GET  /model-info
# POST /predict
# GET  /docs
# GET  /redoc
#
# =============================================================================


# =============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================
# Standard library imports
import json                  # Read and write JSON files
import os                    # Work with file and directory paths
from pathlib import Path     # Create reliable file paths

# Data and model libraries
import joblib                # Load the saved CatBoost model
import numpy as np           # Numerical operations
import pandas as pd          # Data manipulation

# FastAPI imports
from fastapi import FastAPI, HTTPException  # API framework and error handling
from pydantic import BaseModel              # Validate API input data
from typing import Dict, Any # Type hints for API input data


# =============================================================================
# STEP 2: DEFINE APPLICATION PATH
# =============================================================================
#
# IMPORTANT:
#
# Unlike the Jupyter Notebook, app.py DOES have access to __file__.
#
# Therefore, we use __file__ here.
#
# This makes the application portable across:
#
#     Windows
#     GitHub
#     Render
#
# The application automatically finds the model folder relative to app.py.
#
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# STEP 3: DEFINE MODEL DIRECTORY
# =============================================================================

MODEL_DIR = BASE_DIR / "model"


# =============================================================================
# STEP 4: DEFINE CATBOOST MODEL PATH
# =============================================================================

MODEL_PATH = MODEL_DIR / "CatBoost_churn_model.pkl"


# =============================================================================
# STEP 5: DEFINE FEATURE SCHEMA PATH
# =============================================================================

FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_columns.json"


# =============================================================================
# STEP 6: VERIFY FEATURE SCHEMA
# =============================================================================

if not FEATURE_SCHEMA_PATH.exists():

    raise FileNotFoundError(
        f"Feature schema not found: {FEATURE_SCHEMA_PATH}"
    )


# =============================================================================
# STEP 7: LOAD FEATURE SCHEMA
# =============================================================================

with open(
    FEATURE_SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as file:

    feature_schema = json.load(file)


FEATURE_COLUMNS = feature_schema["feature_columns"]

NUMBER_OF_FEATURES = feature_schema["number_of_features"]


# =============================================================================
# STEP 8: VALIDATE FEATURE SCHEMA
# =============================================================================

if NUMBER_OF_FEATURES != len(FEATURE_COLUMNS):

    raise ValueError(
        "Feature schema validation failed."
    )


# =============================================================================
# STEP 9: LOAD CATBOOST MODEL
# =============================================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"CatBoost model not found: {MODEL_PATH}"
    )


tuned_catboost = joblib.load(
    MODEL_PATH
)


# =============================================================================
# STEP 10: CREATE FASTAPI APPLICATION
# =============================================================================

app = FastAPI(

    title="PayStone Customer Churn Prediction API",

    description=(
        "REST API for predicting bank customer churn "
        "using a trained CatBoost model."
    ),

    version="1.0.0"
)


# =============================================================================
# STEP 11: DEFINE CUSTOMER REQUEST FORMAT
# =============================================================================

class CustomerData(BaseModel):

    data: Dict[str, Any]


# =============================================================================
# STEP 12: HEALTH CHECK ENDPOINT
# =============================================================================
#
# GET /health
#
# Used to confirm that the API and model are operational.
#
# =============================================================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "model_loaded": tuned_catboost is not None,

        "model": "CatBoost",

        "number_of_features": NUMBER_OF_FEATURES

    }


# =============================================================================
# STEP 13: MODEL INFORMATION ENDPOINT
# =============================================================================
#
# GET /model-info
#
# Returns information about the deployed model.
#
# =============================================================================

@app.get("/model-info")
def model_info():

    return {

        "model_type": "CatBoost",

        "model_file": MODEL_PATH.name,

        "number_of_features": NUMBER_OF_FEATURES,

        "feature_columns": FEATURE_COLUMNS,

        "api_version": "1.0.0"

    }


# =============================================================================
# STEP 14: CUSTOMER CHURN PREDICTION ENDPOINT
# =============================================================================
#
# POST /predict
#
# Receives customer information and returns:
#
#     - prediction
#     - churn prediction
#     - churn probability
#     - no-churn probability
#
# =============================================================================

@app.post("/predict")
def predict_churn(customer: CustomerData):


    # =========================================================================
    # STEP 14.1: EXTRACT CUSTOMER DATA
    # =========================================================================

    customer_data = customer.data


    # =========================================================================
    # STEP 14.2: CHECK FOR MISSING FEATURES
    # =========================================================================

    missing_features = [

        feature

        for feature in FEATURE_COLUMNS

        if feature not in customer_data

    ]


    if missing_features:

        raise HTTPException(

            status_code=400,

            detail={

                "error": "Missing required features",

                "missing_features": missing_features

            }

        )


    # =========================================================================
    # STEP 14.3: CHECK FOR UNEXPECTED FEATURES
    # =========================================================================

    unexpected_features = [

        feature

        for feature in customer_data

        if feature not in FEATURE_COLUMNS

    ]


    if unexpected_features:

        raise HTTPException(

            status_code=400,

            detail={

                "error": "Unexpected features supplied",

                "unexpected_features": unexpected_features

            }

        )


    # =========================================================================
    # STEP 14.4: ARRANGE FEATURES IN MODEL ORDER
    # =========================================================================

    ordered_data = {

        feature: customer_data[feature]

        for feature in FEATURE_COLUMNS

    }


    # =========================================================================
    # STEP 14.5: CREATE PREDICTION DATAFRAME
    # =========================================================================

    input_df = pd.DataFrame(

        [ordered_data],

        columns=FEATURE_COLUMNS

    )


    # =========================================================================
    # STEP 14.6: RUN CATBOOST PREDICTION
    # =========================================================================

    try:

        prediction = tuned_catboost.predict(
            input_df
        )

        probabilities = tuned_catboost.predict_proba(
            input_df
        )

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail={

                "error": "Model prediction failed",

                "message": str(error)

            }

        )


    # =========================================================================
    # STEP 14.7: EXTRACT PREDICTION RESULTS
    # =========================================================================

    predicted_class = int(
        prediction[0]
    )


    no_churn_probability = float(
        probabilities[0][0]
    )


    churn_probability = float(
        probabilities[0][1]
    )


    # =========================================================================
    # STEP 14.8: CREATE BUSINESS-FRIENDLY LABEL
    # =========================================================================

    if predicted_class == 1:

        churn_prediction = "Churn"

    else:

        churn_prediction = "No Churn"


    # =========================================================================
    # STEP 14.9: RETURN PREDICTION
    # =========================================================================

    return {

        "prediction": predicted_class,

        "churn_prediction": churn_prediction,

        "churn_probability": round(
            churn_probability,
            4
        ),

        "no_churn_probability": round(
            no_churn_probability,
            4
        )

    }


# =============================================================================
# END OF FASTAPI APPLICATION
# =============================================================================
