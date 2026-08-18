**PayStone Customer Churn API - GitHub Deployment Description**
The PayStone Customer Churn Prediction API was deployed using a GitHub-based workflow. The trained CatBoost churn model was saved as a .pkl file together with the FastAPI application, feature schema, and dependency files in the GitHub repository.

- Model Training: Trained a CatBoost customer churn classification model and saved it as CatBoost_churn_model.pkl.
- FastAPI Development: Created app.py to expose the trained model through REST API endpoints.
- Feature Management: Created feature_columns.json containing the 25 features required by the deployed model.
- Input Validation: Implemented validation to check for missing, unexpected, and invalid customer features.

**Prediction API: Developed the /predict endpoint to return:**
- Churn prediction
- Churn probability
- No-churn probability
- **API Documentation:** Integrated FastAPI Swagger documentation through /docs.
- **GitHub Repository:** Uploaded the FastAPI application, model, feature schema, and dependency files to the GitHub repository.
- **Version Control:** Committed and pushed changes to the main branch using Git.
- **Cloud Deployment:** Connected the GitHub repository to Render for cloud-based deployment.
- **Deployment Verification:** Tested /health, /model-info, and /predict endpoints to confirm the application and CatBoost model were successfully deployed.
- **Live Prediction:** Verified that the deployed API successfully returned a live prediction of 89.95% churn probability for the test customer.

  **Public API:** https://paystone-churn-api.onrender.com/docs
