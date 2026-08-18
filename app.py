# =============================================================================
# PAYSTONE CUSTOMER CHURN PREDICTION
# FASTAPI PREDICTION APPLICATION
# =============================================================================
#
# PROJECT:
# PayStone - Explainable AI Bank Customer Churn Prediction
#
# PURPOSE:
# --------
# This application exposes the trained CatBoost customer churn model through
# a REST API using FastAPI.
#
# The API allows an external application, dashboard, website, or testing tool
# to submit customer information and receive a churn prediction.
#
#
# API ENDPOINTS:
# --------------
#
# GET  /health
#      Confirms that the API is running and that the model is loaded.
#
# GET  /model-info
#      Returns information about the deployed model and its feature schema.
#
# POST /predict
#      Accepts customer information and returns a churn prediction together
#      with the probability of churn and no churn.
#
# GET  /docs
#      Automatically generated Swagger API documentation.
#
# GET  /redoc
#      Automatically generated ReDoc API documentation.
#
#
# DEPLOYMENT ARCHITECTURE:
# ------------------------
#
# Customer / Dashboard / Swagger
#             |
#             v
#        FastAPI API
#             |
#             v
#     Input Validation
#             |
#             v
#     Feature Validation
#             |
#             v
#      Preprocessing
#             |
#             v
#      Feature Ordering
#             |
#             v
#       CatBoost Model
#             |
#             v
#       Churn Prediction
#             |
#             v
#       JSON Response
#
# =============================================================================


# =============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================
#
# This section imports all Python libraries required by the API.
#
# The imports are divided into:
#
# 1. Standard Python libraries
# 2. Machine-learning/data-processing libraries
# 3. FastAPI and Pydantic libraries
#
# =============================================================================


# -----------------------------------------------------------------------------
# STEP 1.1: STANDARD PYTHON LIBRARIES
# -----------------------------------------------------------------------------

import json
# Used to read feature_columns.json.
#
# feature_columns.json contains the exact feature names expected by the
# deployed CatBoost model.


from pathlib import Path
# Path provides a reliable and platform-independent way of constructing
# file paths.
#
# This is preferable to manually writing Windows paths such as:
#
# C:\Users\EUGENE\...
#
# Using Path allows the same application to work on:
#
# - Windows
# - Linux
# - Render
# - Docker
# - GitHub deployment environments


from typing import Dict, Any
# Dict and Any are used for type hints in the customer request model.
#
# The API receives a dictionary containing the customer's input variables.


# -----------------------------------------------------------------------------
# STEP 1.2: MACHINE LEARNING AND DATA PROCESSING LIBRARIES
# -----------------------------------------------------------------------------

import joblib
# joblib is used to load the previously trained and saved CatBoost model.
#
# The model was trained before deployment and saved as:
#
# CatBoost_churn_model.pkl
#
# The API does NOT retrain the model.
#
# It simply loads the existing trained model and uses it to generate
# predictions.


import pandas as pd
# pandas is used to construct a DataFrame containing the customer's
# information.
#
# CatBoost expects the input data to have the same feature structure used
# during model training.


# -----------------------------------------------------------------------------
# STEP 1.3: FASTAPI LIBRARIES
# -----------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
# FastAPI:
#     Creates the REST API.
#
# HTTPException:
#     Allows the API to return meaningful HTTP error messages when invalid
#     input is supplied.


from pydantic import BaseModel
# BaseModel is used to define the structure of both:
#
# 1. The incoming prediction request
# 2. The outgoing prediction response
#
# Pydantic automatically validates the API data structure.


# =============================================================================
# STEP 2: DEFINE APPLICATION PATH
# =============================================================================
#
# The application needs to locate:
#
#     app.py
#     feature_columns.json
#     model/CatBoost_churn_model.pkl
#
# __file__ represents the location of the currently running Python file.
#
# Path(__file__).resolve().parent gives us the directory containing app.py.
#
# This is important for deployment because the application may run on:
#
#     Local Windows machine
#     GitHub
#     Render
#     Docker
#
# The location of the project can change between environments.
#
# Using BASE_DIR prevents hard-coded computer-specific paths.
#
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# STEP 3: DEFINE MODEL DIRECTORY
# =============================================================================
#
# The trained CatBoost model is stored inside the "model" folder.
#
# Expected project structure:
#
# paystone-churn-api/
#
#     app.py
#     feature_columns.json
#     requirements.txt
#     model/
#         CatBoost_churn_model.pkl
#
# =============================================================================

MODEL_DIR = BASE_DIR / "model"


# =============================================================================
# STEP 4: DEFINE CATBOOST MODEL PATH
# =============================================================================
#
# This creates the complete path to the saved CatBoost model.
#
# The API will load this file during application startup.
#
# =============================================================================

MODEL_PATH = MODEL_DIR / "CatBoost_churn_model.pkl"


# =============================================================================
# STEP 5: DEFINE FEATURE SCHEMA PATH
# =============================================================================
#
# feature_columns.json contains the exact 25 features required by the
# deployed model.
#
# This file is extremely important because it ensures that the API uses
# the same feature names and ordering expected by the trained model.
#
# =============================================================================

FEATURE_SCHEMA_PATH = BASE_DIR / "feature_columns.json"


# =============================================================================
# STEP 6: VERIFY FEATURE SCHEMA EXISTS
# =============================================================================
#
# Before starting the API, verify that feature_columns.json exists.
#
# If the file does not exist, the API should stop immediately rather than
# starting with an incomplete model configuration.
#
# This is preferable to discovering the problem later when a customer sends
# a prediction request.
#
# =============================================================================

if not FEATURE_SCHEMA_PATH.exists():

    raise FileNotFoundError(
        f"Feature schema not found: {FEATURE_SCHEMA_PATH}"
    )


# =============================================================================
# STEP 7: LOAD FEATURE SCHEMA
# =============================================================================
#
# Open feature_columns.json and load its contents into Python.
#
# Example file:
#
# [
#     "Income_Category",
#     "Card_Category",
#     "Education_Level_Doctorate",
#     ...
# ]
#
# The feature schema is loaded only once when the API starts.
#
# =============================================================================

with open(
    FEATURE_SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as file:

    feature_schema = json.load(file)


# =============================================================================
# STEP 8: HANDLE FEATURE SCHEMA FORMAT
# =============================================================================
#
# The feature_columns.json file may be stored in one of two formats.
#
# FORMAT 1:
#
# [
#     "Income_Category",
#     "Card_Category",
#     ...
# ]
#
#
# FORMAT 2:
#
# {
#     "feature_columns": [
#         "Income_Category",
#         "Card_Category",
#         ...
#     ]
# }
#
# This code supports both formats.
#
# =============================================================================


if isinstance(feature_schema, list):

    # If the JSON file directly contains a list, use that list.
    FEATURE_COLUMNS = feature_schema


elif isinstance(feature_schema, dict):

    # If the JSON file contains a dictionary, check whether the expected
    # "feature_columns" key exists.

    if "feature_columns" not in feature_schema:

        raise ValueError(
            "feature_columns.json is a dictionary but "
            "'feature_columns' was not found."
        )

    FEATURE_COLUMNS = feature_schema["feature_columns"]


else:

    # Any other structure is invalid.
    raise ValueError(
        "Invalid feature_columns.json format. "
        "Expected a list or dictionary."
    )


# =============================================================================
# STEP 9: DEFINE NUMBER OF FEATURES
# =============================================================================
#
# Count the number of features contained in the feature schema.
#
# The deployed CatBoost model was trained using 25 features.
#
# =============================================================================

NUMBER_OF_FEATURES = len(FEATURE_COLUMNS)


# =============================================================================
# STEP 10: VALIDATE FEATURE SCHEMA
# =============================================================================
#
# This step protects the API from deploying with an incorrect feature
# configuration.
#
# Two checks are performed:
#
# 1. Confirm exactly 25 features exist.
# 2. Confirm there are no duplicate feature names.
#
# =============================================================================


# -----------------------------------------------------------------------------
# STEP 10.1: VERIFY NUMBER OF FEATURES
# -----------------------------------------------------------------------------

if NUMBER_OF_FEATURES != 25:

    raise ValueError(
        f"Expected 25 features, "
        f"but found {NUMBER_OF_FEATURES}."
    )


# -----------------------------------------------------------------------------
# STEP 10.2: VERIFY NO DUPLICATE FEATURES
# -----------------------------------------------------------------------------

if len(set(FEATURE_COLUMNS)) != NUMBER_OF_FEATURES:

    raise ValueError(
        "Feature schema contains duplicate feature names."
    )


# =============================================================================
# STEP 11: LOAD CATBOOST MODEL
# =============================================================================
#
# Before loading the model, verify that the model file exists.
#
# =============================================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"CatBoost model not found: {MODEL_PATH}"
    )


# -----------------------------------------------------------------------------
# LOAD THE TRAINED MODEL
# -----------------------------------------------------------------------------
#
# joblib.load() loads the trained CatBoost model from disk.
#
# IMPORTANT:
# ----------
# The model is NOT trained again here.
#
# Training occurred during the machine-learning modelling pipeline.
#
# This FastAPI application is the deployment/inference stage.
#
# =============================================================================

tuned_catboost = joblib.load(
    MODEL_PATH
)


# =============================================================================
# STEP 12: CREATE FASTAPI APPLICATION
# =============================================================================
#
# FastAPI creates the web application.
#
# The title, description and version are automatically displayed in Swagger
# documentation at:
#
#     /docs
#
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
# STEP 13: DEFINE CUSTOMER REQUEST FORMAT
# =============================================================================
#
# This Pydantic model defines the structure of incoming POST /predict data.
#
# The expected request structure is:
#
# {
#     "data": {
#         "Income_Category": "$60K - $80K",
#         "Card_Category": "Blue",
#         ...
#     }
# }
#
# The Dict[str, Any] structure allows the API to receive different data
# types such as:
#
#     strings
#     integers
#     decimal numbers
#
# Individual features are subsequently validated by the prediction logic.
#
# =============================================================================

class CustomerData(BaseModel):

    data: Dict[str, Any]


# =============================================================================
# STEP 14: DEFINE CUSTOMER RESPONSE FORMAT
# =============================================================================
#
# This is the recommended addition to the original API.
#
# The response model explicitly tells FastAPI what the /predict endpoint
# returns.
#
# Without response_model:
#
#     Swagger may display:
#
#         "string"
#
# With this response model:
#
#     Swagger understands that the API returns four fields.
#
# Expected response:
#
# {
#     "prediction": 1,
#     "churn_prediction": "Churn",
#     "churn_probability": 0.8995,
#     "no_churn_probability": 0.1005
# }
#
# =============================================================================

class PredictionResponse(BaseModel):

    prediction: int
    # Binary model prediction.
    #
    # 0 = No Churn
    # 1 = Churn


    churn_prediction: str
    # Business-friendly interpretation of the prediction.
    #
    # Possible values:
    #
    # "Churn"
    # "No Churn"


    churn_probability: float
    # Probability that the customer will churn.
    #
    # Example:
    #
    # 0.8995 = 89.95%


    no_churn_probability: float
    # Probability that the customer will not churn.
    #
    # Example:
    #
    # 0.1005 = 10.05%


# =============================================================================
# STEP 15: HEALTH CHECK ENDPOINT
# =============================================================================
#
# Endpoint:
#
#     GET /health
#
# Purpose:
# --------
# Used by Render, monitoring systems, developers, or external applications
# to verify that the API is operational.
#
# Example response:
#
# {
#     "status": "healthy",
#     "model_loaded": true,
#     "model": "CatBoost",
#     "number_of_features": 25
# }
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
# STEP 16: MODEL INFORMATION ENDPOINT
# =============================================================================
#
# Endpoint:
#
#     GET /model-info
#
# Purpose:
# --------
# Provides information about the model currently deployed by the API.
#
# This is useful for:
#
#     - debugging
#     - model governance
#     - deployment verification
#     - feature validation
#     - audit documentation
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
# STEP 17: CUSTOMER CHURN PREDICTION ENDPOINT
# =============================================================================
#
# Endpoint:
#
#     POST /predict
#
# This is the main prediction endpoint.
#
# The response_model parameter is important because it tells FastAPI that
# the endpoint returns a PredictionResponse object.
#
# As a result, Swagger will automatically document the four response fields.
#
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(customer: CustomerData):


    # =========================================================================
    # STEP 17.1: EXTRACT CUSTOMER DATA
    # =========================================================================
    #
    # customer.data contains the dictionary submitted in the request.
    #
    # .copy() prevents accidental modification of the original Pydantic data.
    #
    # =========================================================================

    customer_data = customer.data.copy()


    # =========================================================================
    # STEP 17.2: CHECK FOR MISSING FEATURES
    # =========================================================================
    #
    # Every feature required by the CatBoost model must be present.
    #
    # If even one required feature is missing, the API cannot safely make
    # a prediction.
    #
    # Example:
    #
    # If Customer_Age is missing:
    #
    # {
    #     "error": "Missing required features",
    #     "missing_features": ["Customer_Age"]
    # }
    #
    # HTTP 400 means the client supplied invalid input.
    #
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
    # STEP 17.3: CHECK FOR UNEXPECTED FEATURES
    # =========================================================================
    #
    # This checks whether the user has supplied variables that are NOT part
    # of the trained model.
    #
    # This is useful because it prevents accidental changes to the model
    # input structure.
    #
    # Example:
    #
    # If the user supplies:
    #
    #     "Customer_Name": "John"
    #
    # but Customer_Name was not used during model training, the API rejects
    # the request.
    #
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
    # STEP 17.4: ARRANGE FEATURES IN MODEL ORDER
    # =========================================================================
    #
    # Machine-learning models require the input variables to correspond to
    # the structure used during training.
    #
    # Even if the user sends the variables in a different order, this code
    # reconstructs the data using FEATURE_COLUMNS.
    #
    # This is an important protection against feature-order errors.
    #
    # =========================================================================

    ordered_data = {

        feature: customer_data[feature]

        for feature in FEATURE_COLUMNS

    }


    # =========================================================================
    # STEP 17.5: PREPROCESS CATEGORICAL FEATURES
    # =========================================================================
    #
    # The API accepts human-readable category values such as:
    #
    #     "$60K - $80K"
    #     "Blue"
    #
    # However, the saved model expects these categories in their numerical
    # representation.
    #
    # Therefore, the API performs the same mapping used during model
    # preparation.
    #
    # IMPORTANT:
    # ----------
    # These mappings MUST exactly match the mappings used during model
    # development/training.
    #
    # Changing these mappings after model training could produce incorrect
    # predictions.
    #
    # =========================================================================


    # -----------------------------------------------------------------------------
    # STEP 17.5.1: INCOME CATEGORY MAPPING
    # -----------------------------------------------------------------------------

    income_mapping = {

        "Less than $40K": 0,

        "$40K - $60K": 1,

        "$60K - $80K": 2,

        "$80K - $120K": 3,

        "$120K +": 4,

        "Unknown": 5

    }


    # -----------------------------------------------------------------------------
    # STEP 17.5.2: CARD CATEGORY MAPPING
    # -----------------------------------------------------------------------------

    card_mapping = {

        "Blue": 0,

        "Silver": 1,

        "Gold": 2,

        "Platinum": 3

    }


    # =========================================================================
    # STEP 17.5.3: CONVERT INCOME CATEGORY
    # =========================================================================
    #
    # If Income_Category arrives as text, convert it to the numerical value
    # expected by the model.
    #
    # If an invalid category is supplied, return HTTP 400 rather than allowing
    # the model to make a prediction from an unknown value.
    #
    # =========================================================================

    if isinstance(
        ordered_data["Income_Category"],
        str
    ):

        income_value = ordered_data["Income_Category"]


        if income_value not in income_mapping:

            raise HTTPException(

                status_code=400,

                detail={

                    "error": "Invalid Income_Category",

                    "value": income_value,

                    "allowed_values": list(
                        income_mapping.keys()
                    )

                }

            )


        ordered_data["Income_Category"] = (

            income_mapping[income_value]

        )


    # =========================================================================
    # STEP 17.5.4: CONVERT CARD CATEGORY
    # =========================================================================
    #
    # Convert the human-readable Card_Category into the numerical representation
    # expected by the model.
    #
    # =========================================================================

    if isinstance(
        ordered_data["Card_Category"],
        str
    ):

        card_value = ordered_data["Card_Category"]


        if card_value not in card_mapping:

            raise HTTPException(

                status_code=400,

                detail={

                    "error": "Invalid Card_Category",

                    "value": card_value,

                    "allowed_values": list(
                        card_mapping.keys()
                    )

                }

            )


        ordered_data["Card_Category"] = (

            card_mapping[card_value]

        )


    # =========================================================================
    # STEP 17.5.5: CREATE PREDICTION DATAFRAME
    # =========================================================================
    #
    # Convert the ordered customer dictionary into a pandas DataFrame.
    #
    # The columns parameter explicitly specifies the feature order.
    #
    # This ensures that the DataFrame matches the model's expected input
    # structure.
    #
    # =========================================================================

    input_df = pd.DataFrame(

        [ordered_data],

        columns=FEATURE_COLUMNS

    )


    # =========================================================================
    # STEP 17.5.6: CONVERT NUMERIC FEATURES
    # =========================================================================
    #
    # Convert all feature values to numeric values.
    #
    # errors="raise" is intentional.
    #
    # If a value such as:
    #
    #     "abc"
    #
    # is supplied for a numeric variable, the API will immediately identify
    # the problem rather than silently converting it to an incorrect value.
    #
    # =========================================================================

    for feature in FEATURE_COLUMNS:

        input_df[feature] = pd.to_numeric(

            input_df[feature],

            errors="raise"

        )


    # =========================================================================
    # STEP 17.6: RUN CATBOOST PREDICTION
    # =========================================================================
    #
    # The prepared customer DataFrame is passed to the trained CatBoost model.
    #
    # predict():
    #     Returns the predicted class.
    #
    # predict_proba():
    #     Returns the probability for each class.
    #
    # For this binary churn model:
    #
    #     probabilities[0][0] = probability of No Churn
    #
    #     probabilities[0][1] = probability of Churn
    #
    # =========================================================================

    try:

        prediction = tuned_catboost.predict(
            input_df
        )


        probabilities = tuned_catboost.predict_proba(
            input_df
        )


    except Exception as error:

        # If the model fails during prediction, return HTTP 500.
        #
        # HTTP 500 indicates that the server could not complete the prediction
        # because of an internal model/application error.

        raise HTTPException(

            status_code=500,

            detail={

                "error": "Model prediction failed",

                "message": str(error)

            }

        )


    # =========================================================================
    # STEP 17.7: EXTRACT PREDICTION RESULTS
    # =========================================================================
    #
    # CatBoost returns arrays.
    #
    # Because we are predicting one customer, we extract the first result.
    #
    # Example:
    #
    # prediction = [1]
    #
    # becomes:
    #
    # predicted_class = 1
    #
    # =========================================================================

    predicted_class = int(

        prediction[0]

    )


    # -------------------------------------------------------------------------
    # Probability of No Churn
    # -------------------------------------------------------------------------

    no_churn_probability = float(

        probabilities[0][0]

    )


    # -------------------------------------------------------------------------
    # Probability of Churn
    # -------------------------------------------------------------------------

    churn_probability = float(

        probabilities[0][1]

    )


    # =========================================================================
    # STEP 17.8: CREATE BUSINESS-FRIENDLY LABEL
    # =========================================================================
    #
    # The model produces a binary numerical prediction:
    #
    #     0 = No Churn
    #     1 = Churn
    #
    # For business users, a text label is easier to understand.
    #
    # Therefore:
    #
    #     0 -> "No Churn"
    #     1 -> "Churn"
    #
    # =========================================================================

    if predicted_class == 1:

        churn_prediction = "Churn"

    else:

        churn_prediction = "No Churn"


    # =========================================================================
    # STEP 17.9: RETURN STRUCTURED PREDICTION RESPONSE
    # =========================================================================
    #
    # PredictionResponse ensures that the API response follows the schema
    # defined in STEP 14.
    #
    # This also improves the automatically generated Swagger documentation.
    #
    # The probabilities are rounded to four decimal places.
    #
    # Example:
    #
    # 0.8995 = 89.95%
    #
    # =========================================================================

    return PredictionResponse(

        prediction=predicted_class,

        churn_prediction=churn_prediction,

        churn_probability=round(

            churn_probability,

            4

        ),

        no_churn_probability=round(

            no_churn_probability,

            4

        )

    )


# =============================================================================
# END OF FASTAPI APPLICATION
# =============================================================================
#
# The application is now ready to be started with Uvicorn.
#
# Local command:
#
# python -m uvicorn app:app --reload
#
# Render deployment command:
#
# uvicorn app:app --host 0.0.0.0 --port $PORT
#
# =============================================================================